#!/usr/bin/env python3
"""Qwen3-TTS — CUDA EP pipeline (talker+cp on GPU, vocoder on TRT/CUDA)."""
import argparse, os, time
import numpy as np, onnxruntime as ort, wave

FP32 = "/tmp/qwen3-tts-bench/model"
CEMB = "/tmp/correct-emb"
CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]

CODEC_EOS = 2150; CODEC_THINK = 2154; CODEC_THINK_BOS = 2156
CODEC_THINK_EOS = 2157; CODEC_PAD = 2148; CODEC_BOS = 2149
TTS_BOS = 151672; TTS_EOS = 151673; TTS_PAD = 151671; IM_START = 151644
LANG_IDS = {"chinese": 2055, "english": 2050, "japanese": 2058, "korean": 2064}

print("Loading models on CUDA...")
o = ort.SessionOptions()
o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

prefill_s = ort.InferenceSession(f"{FP32}/talker_prefill.onnx", sess_options=o, providers=CUDA)
decode_s = ort.InferenceSession(f"{FP32}/talker_decode.onnx", sess_options=o, providers=CUDA)
cp_s = ort.InferenceSession(f"{FP32}/code_predictor.onnx", sess_options=o, providers=CUDA)
voc_s = ort.InferenceSession(f"{FP32}/vocoder.onnx", providers=CUDA)

pf_on = [x.name for x in prefill_s.get_outputs()]
dc_on = [x.name for x in decode_s.get_outputs()]
cp_on = [x.name for x in cp_s.get_outputs()]
vi = voc_s.get_inputs()[0].name

print("Loading embeddings...")
talker_ce = np.load(f"{CEMB}/talker_codec_embedding.npy")
cp_ces = [np.load(f"{CEMB}/cp_codec_embedding_{i}.npy") for i in range(15)]
text_emb_w = np.load(f"{FP32}/embeddings/text_embedding.npy")
fc1_w = np.load(f"{CEMB}/tp_fc1_weight.npy"); fc1_b = np.load(f"{CEMB}/tp_fc1_bias.npy")
fc2_w = np.load(f"{CEMB}/tp_fc2_weight.npy"); fc2_b = np.load(f"{CEMB}/tp_fc2_bias.npy")

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
tok = Tokenizer(BPE(f"{FP32}/tokenizer/vocab.json", f"{FP32}/tokenizer/merges.txt"))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
def tokenize(t): return tok.encode(t).ids

def tp(ids):
    e = []
    for tid in ids:
        x = text_emb_w[tid].astype(np.float32)
        h = x @ fc1_w.T + fc1_b; h = h * (1/(1+np.exp(-h)))
        e.append(h @ fc2_w.T + fc2_b)
    return np.array(e).reshape(1, len(ids), 1024)
def ce(t): return talker_ce[t].reshape(1,1,1024).astype(np.float32)
def cpe(t,g): return cp_ces[g][t].reshape(1,1,1024).astype(np.float32)
def sample(l, k=50, t=0.9):
    l = l.flatten().astype(np.float64)/t
    ti = np.argpartition(l,-k)[-k:]
    m = np.full_like(l,-np.inf); m[ti]=l[ti]
    e = np.exp(m-np.max(m)); p = e/e.sum()
    return int(np.random.choice(len(p),p=p))

def build_emb(text, lang="english"):
    lid = LANG_IDS.get(lang, 2050); tids = tokenize(text)
    tts_bos_e=tp([TTS_BOS]); tts_eos_e=tp([TTS_EOS]); tts_pad_e=tp([TTS_PAD])
    role_e = tp([IM_START, 77091, 198])
    cin = np.concatenate([ce(CODEC_THINK),ce(CODEC_THINK_BOS),ce(lid),ce(CODEC_THINK_EOS),ce(CODEC_PAD),ce(CODEC_BOS)],axis=1)
    pf = np.concatenate([np.tile(tts_pad_e,(1,4,1)),tts_bos_e],axis=1) + cin[:, :-1, :]
    em = np.concatenate([role_e, pf], axis=1)
    te = tp(tids); teos = np.concatenate([te,tts_eos_e],axis=1)
    cpn = np.tile(ce(CODEC_PAD),(1,len(tids)+1,1))
    em = np.concatenate([em, teos+cpn, tts_pad_e+ce(CODEC_BOS)], axis=1)
    return em.astype(np.float32), tts_pad_e

def synthesize(text, lang="english", output="/tmp/tts_out.wav", mf=60, seed=None):
    if seed: np.random.seed(seed)
    print(f"\nSynth: \"{text}\" ({lang})")
    tt = time.perf_counter(); timings = {}

    ie, pad_e = build_emb(text, lang)
    seq = ie.shape[1]
    t0 = time.perf_counter()
    pos = np.arange(seq,dtype=np.int64).reshape(1,1,seq).repeat(3,axis=0)
    po = prefill_s.run(None, {"inputs_embeds":ie,"attention_mask":np.ones((1,seq),dtype=np.int64),"position_ids":pos})
    pm = dict(zip(pf_on,po))
    logits=pm["logits"]; hidden=pm["hidden_states"][:,-1:,:]
    kv={"pk":np.stack([pm[f"present_key_{i}"] for i in range(28)]),"pv":np.stack([pm[f"present_value_{i}"] for i in range(28)])}
    timings["prefill"]=(time.perf_counter()-t0)*1000
    print(f"  Prefill: {timings['prefill']:.0f}ms")

    codes=[]; ts=seq; dt=[]; ct=[]
    for fr in range(mf):
        fc = sample(logits[0,-1,:])
        if fc==CODEC_EOS: print(f"  EOS@{fr}"); break
        t_cp=time.perf_counter()
        fce=ce(fc); ci=np.concatenate([hidden,fce],axis=1)
        cpk=np.zeros((5,1,8,0,128),dtype=np.float32); cpv=cpk.copy()
        subs=[]
        for step in range(15):
            co=cp_s.run(None,{"inputs_embeds":ci,"generation_steps":np.array([step],dtype=np.int64),"past_keys":cpk,"past_values":cpv})
            cm=dict(zip(cp_on,co)); sc=sample(cm["logits"][0,-1,:]); subs.append(sc)
            cpk=cm["present_keys"]; cpv=cm["present_values"]; ci=cpe(sc,step)
        ct.append(time.perf_counter()-t_cp)
        codes.append([fc]+subs)
        se=fce.copy()
        for i,sc in enumerate(subs): se=se+cpe(sc,i)
        ne=se+pad_e
        t_d=time.perf_counter(); ts+=1
        do=decode_s.run(None,{"inputs_embeds":ne,"attention_mask":np.ones((1,ts),dtype=np.int64),"position_ids":np.array([[[ts-1]]]*3,dtype=np.int64),"past_keys":kv["pk"],"past_values":kv["pv"]})
        dm=dict(zip(dc_on,do)); logits=dm["logits"]; hidden=dm["hidden_states"][:,-1:,:]
        kv={"pk":dm["present_keys"],"pv":dm["present_values"]}
        dt.append(time.perf_counter()-t_d)
        if (fr+1)%10==0: print(f"  Frame {fr+1}")

    n=len(codes); dur=n/12.5
    if n==0: print("  No frames!"); return
    t0=time.perf_counter()
    wav=voc_s.run(None,{vi:np.array(codes,dtype=np.int64).T[np.newaxis,:,:]})[0].flatten()
    tv=(time.perf_counter()-t0)*1000

    with wave.open(output,"w") as wf:
        wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(24000)
        wf.writeframes((wav*32767).clip(-32768,32767).astype(np.int16).tobytes())

    total=(time.perf_counter()-tt)*1000
    da=np.mean(dt)*1000; ca=np.mean(ct)*1000
    print(f"\n  === TIMING ({n} frames, {dur:.1f}s audio) ===")
    print(f"  Prefill:     {timings['prefill']:>7.0f}ms")
    print(f"  Talker/step: {da:>7.1f}ms (CUDA)")
    print(f"  CP/step:     {ca:>7.1f}ms (CUDA)")
    print(f"  Per-step:    {da+ca:>7.1f}ms  RTF={((da+ca)/80):.2f}")
    print(f"  Vocoder:     {tv:>7.0f}ms (CUDA)")
    print(f"  Total:       {total:>7.0f}ms  RTF={total/1000/dur:.2f}")
    print(f"  Saved: {output}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--text",default="Hello, welcome to the voice synthesis system.")
    p.add_argument("--lang",default="english",choices=list(LANG_IDS.keys()))
    p.add_argument("--output",default="/tmp/tts_cuda.wav")
    p.add_argument("--max-frames",type=int,default=60)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    synthesize(a.text,a.lang,a.output,a.max_frames,a.seed)
