#!/usr/bin/env python3
"""Qwen3-TTS — Hybrid pipeline: best achievable on Orin Nano 15GB.
- Talker prefill: FP32 CUDA EP (runs once)
- Talker decode: FP32 CUDA EP (41ms/step, KV cache)
- Code predictor: TRT FP16 (cached engine, accumulated input, ~2.3ms/step)
- Vocoder: CUDA EP (565ms, TRT cache builds on second run)
"""
import argparse, os, time
import numpy as np, onnxruntime as ort, wave

FP32 = "/tmp/qwen3-tts-bench/model"
CEMB = "/tmp/correct-emb"
CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]

def trt_ep(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    return [("TensorrtExecutionProvider", {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
        "trt_max_workspace_size": str(512*1024*1024),
    }), "CUDAExecutionProvider", "CPUExecutionProvider"]

CODEC_EOS=2150;CODEC_THINK=2154;CODEC_THINK_BOS=2156;CODEC_THINK_EOS=2157
CODEC_PAD=2148;CODEC_BOS=2149;TTS_BOS=151672;TTS_EOS=151673;TTS_PAD=151671
IM_START=151644
LANG_IDS={"chinese":2055,"english":2050,"japanese":2058,"korean":2064}

print("Loading models...")
o = ort.SessionOptions()
o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Talker: CUDA EP (FP32, has KV cache, handles If nodes fine)
prefill_s = ort.InferenceSession(f"{FP32}/talker_prefill.onnx", sess_options=o, providers=CUDA)
decode_s = ort.InferenceSession(f"{FP32}/talker_decode.onnx", sess_options=o, providers=CUDA)
pf_on = [x.name for x in prefill_s.get_outputs()]
dc_on = [x.name for x in decode_s.get_outputs()]

# CP: TRT FP16 (cached engine from pre-build step)
print("  CP (TRT cached)...")
cp_s = ort.InferenceSession("/tmp/cp_kv_no_if.onnx", sess_options=o, providers=trt_ep("/tmp/trt_cp"))
cp_on = [x.name for x in cp_s.get_outputs()]

# Vocoder: CUDA EP
voc_s = ort.InferenceSession(f"{FP32}/vocoder.onnx", providers=CUDA)
vi = voc_s.get_inputs()[0].name

print("Loading embeddings...")
tce = np.load(f"{CEMB}/talker_codec_embedding.npy")
cces = [np.load(f"{CEMB}/cp_codec_embedding_{i}.npy") for i in range(15)]
tew = np.load(f"{FP32}/embeddings/text_embedding.npy")
f1w=np.load(f"{CEMB}/tp_fc1_weight.npy");f1b=np.load(f"{CEMB}/tp_fc1_bias.npy")
f2w=np.load(f"{CEMB}/tp_fc2_weight.npy");f2b=np.load(f"{CEMB}/tp_fc2_bias.npy")

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
tok=Tokenizer(BPE(f"{FP32}/tokenizer/vocab.json",f"{FP32}/tokenizer/merges.txt"))
tok.pre_tokenizer=ByteLevel(add_prefix_space=False)
def tokenize(t): return tok.encode(t).ids

def tp(ids):
    e=[]
    for tid in ids:
        x=tew[tid].astype(np.float32);h=x@f1w.T+f1b;h=h*(1/(1+np.exp(-h)));e.append(h@f2w.T+f2b)
    return np.array(e).reshape(1,len(ids),1024)
def ce(t): return tce[t].reshape(1,1,1024).astype(np.float32)
def cpe(t,g): return cces[g][t].reshape(1,1,1024).astype(np.float32)
def sample(l,k=50,t=0.9):
    l=l.flatten().astype(np.float64)/t;ti=np.argpartition(l,-k)[-k:]
    m=np.full_like(l,-np.inf);m[ti]=l[ti];e=np.exp(m-np.max(m));p=e/e.sum()
    return int(np.random.choice(len(p),p=p))

def build_emb(text,lang="english"):
    lid=LANG_IDS.get(lang,2050);tids=tokenize(text)
    tbs=tp([TTS_BOS]);tes=tp([TTS_EOS]);tps=tp([TTS_PAD])
    re=tp([IM_START,77091,198])
    cin=np.concatenate([ce(CODEC_THINK),ce(CODEC_THINK_BOS),ce(lid),ce(CODEC_THINK_EOS),ce(CODEC_PAD),ce(CODEC_BOS)],axis=1)
    pf=np.concatenate([np.tile(tps,(1,4,1)),tbs],axis=1)+cin[:,:-1,:]
    em=np.concatenate([re,pf],axis=1)
    te=tp(tids);teos=np.concatenate([te,tes],axis=1)
    cpn=np.tile(ce(CODEC_PAD),(1,len(tids)+1,1))
    em=np.concatenate([em,teos+cpn,tps+ce(CODEC_BOS)],axis=1)
    return em.astype(np.float32),tps

def synthesize(text,lang="english",output="/tmp/tts_hybrid.wav",mf=60,seed=None):
    if seed: np.random.seed(seed)
    print(f"\nSynth: \"{text}\" ({lang})")
    tt=time.perf_counter()

    ie,pad_e=build_emb(text,lang)
    seq=ie.shape[1]

    # Prefill
    t0=time.perf_counter()
    pos=np.arange(seq,dtype=np.int64).reshape(1,1,seq).repeat(3,axis=0)
    po=prefill_s.run(None,{"inputs_embeds":ie,"attention_mask":np.ones((1,seq),dtype=np.int64),"position_ids":pos})
    pm=dict(zip(pf_on,po))
    logits=pm["logits"];hidden=pm["hidden_states"][:,-1:,:]
    pk=np.stack([pm[f"present_key_{i}"] for i in range(28)])
    pv=np.stack([pm[f"present_value_{i}"] for i in range(28)])
    pf_ms=(time.perf_counter()-t0)*1000
    print(f"  Prefill: {pf_ms:.0f}ms (CUDA)")

    codes=[];ts=seq;dt=[];ct=[]
    for fr in range(mf):
        fc=sample(logits[0,-1,:])
        if fc==CODEC_EOS: print(f"  EOS@{fr}"); break

        # CP: TRT accumulated input
        t_cp=time.perf_counter()
        fce=ce(fc)
        accum=np.concatenate([hidden,fce],axis=1)
        subs=[]
        for step in range(15):
            co=cp_s.run(None,{"inputs_embeds":accum,"generation_steps":np.array([step],dtype=np.int64)})
            sl=co[0][0,-1,:] if co[0].ndim==3 else co[0].flatten()
            sc=sample(sl);subs.append(sc)
            accum=np.concatenate([accum,cpe(sc,step)],axis=1)
        ct.append(time.perf_counter()-t_cp)
        codes.append([fc]+subs)

        # Next input
        se=fce.copy()
        for i,sc in enumerate(subs): se=se+cpe(sc,i)
        ne=se+pad_e

        # Talker decode: CUDA EP with KV cache
        t_d=time.perf_counter()
        ts+=1
        do=decode_s.run(None,{"inputs_embeds":ne,"attention_mask":np.ones((1,ts),dtype=np.int64),
                              "position_ids":np.array([[[ts-1]]]*3,dtype=np.int64),
                              "past_keys":pk,"past_values":pv})
        dm=dict(zip(dc_on,do))
        logits=dm["logits"];hidden=dm["hidden_states"][:,-1:,:]
        pk=dm["present_keys"];pv=dm["present_values"]
        dt.append(time.perf_counter()-t_d)

        if (fr+1)%10==0: print(f"  Frame {fr+1}")

    n=len(codes);dur=n/12.5
    if n==0: print("  No frames!"); return

    t0=time.perf_counter()
    wav=voc_s.run(None,{vi:np.array(codes,dtype=np.int64).T[np.newaxis,:,:]})[0].flatten()
    tv=(time.perf_counter()-t0)*1000

    with wave.open(output,"w") as wf:
        wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(24000)
        wf.writeframes((wav*32767).clip(-32768,32767).astype(np.int16).tobytes())

    total=(time.perf_counter()-tt)*1000
    da=np.mean(dt)*1000;ca=np.mean(ct)*1000
    print(f"\n  === TIMING ({n} frames, {dur:.1f}s audio) ===")
    print(f"  Prefill:     {pf_ms:>7.0f}ms (CUDA FP32)")
    print(f"  Talker/step: {da:>7.1f}ms (CUDA FP32)")
    print(f"  CP/step:     {ca:>7.1f}ms (TRT FP16)")
    print(f"  Per-step:    {da+ca:>7.1f}ms  RTF={((da+ca)/80):.2f}")
    print(f"  Vocoder:     {tv:>7.0f}ms (CUDA)")
    print(f"  Total:       {total:>7.0f}ms  RTF={total/1000/dur:.2f}")
    print(f"  Saved: {output}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--text",default="Hello, welcome to the voice synthesis system.")
    p.add_argument("--lang",default="english",choices=list(LANG_IDS.keys()))
    p.add_argument("--output",default="/tmp/tts_hybrid.wav")
    p.add_argument("--max-frames",type=int,default=60)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    synthesize(a.text,a.lang,a.output,a.max_frames,a.seed)
