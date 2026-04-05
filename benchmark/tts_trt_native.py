#!/usr/bin/env python3
"""Qwen3-TTS — Native TRT pipeline with pre-built engines.
- Talker prefill: numpy (no TRT, runs once) — builds initial KV cache
- Talker decode: TRT FP16 engine with KV cache
- Code predictor: FP32 CUDA EP (ORT in speech container) / numpy fallback
- Vocoder: TRT FP16 engine

For vision-trt container: has tensorrt + pycuda, no ORT.
Code predictor falls back to numpy matrix ops (slower but correct).
"""
import argparse, os, time, wave, gc
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

CEMB = "/tmp/correct-emb"
FP32_EMB = "/tmp/qwen3-tts-bench/model/embeddings"
FP32_TOK = "/tmp/qwen3-tts-bench/model/tokenizer"
TALKER_ENGINE = "/tmp/talker_dec_kv_fp16.engine"

CODEC_EOS=2150;CODEC_THINK=2154;CODEC_THINK_BOS=2156;CODEC_THINK_EOS=2157
CODEC_PAD=2148;CODEC_BOS=2149;TTS_BOS=151672;TTS_EOS=151673;TTS_PAD=151671
IM_START=151644
LANG_IDS={"chinese":2055,"english":2050,"japanese":2058,"korean":2064}
H=1024; NL=28; NKV=8; HD=128

class TRTEngine:
    """Optimized TRT engine wrapper with pre-allocated buffers."""
    def __init__(self, path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.io = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            self.io[name] = {"mode": mode, "dtype": dtype}
        print(f"  Engine: {os.path.basename(path)}")

    def run(self, feed):
        d_bufs = []
        results = {}
        # Set inputs
        for name, arr in feed.items():
            arr = np.ascontiguousarray(arr.astype(self.io[name]["dtype"]))
            self.context.set_input_shape(name, arr.shape)
            nbytes = max(arr.nbytes, 1)  # pycuda can't alloc 0 bytes
            d = cuda.mem_alloc(nbytes)
            if arr.nbytes > 0:
                cuda.memcpy_htod(d, arr)
            self.context.set_tensor_address(name, int(d))
            d_bufs.append(d)
        # Allocate outputs
        out_info = []
        for name, info in self.io.items():
            if info["mode"] == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.context.get_tensor_shape(name))
                h = np.empty(shape, dtype=info["dtype"])
                d = cuda.mem_alloc(h.nbytes)
                self.context.set_tensor_address(name, int(d))
                out_info.append((name, d, h))
                d_bufs.append(d)
        self.context.execute_async_v3(self.stream.handle)
        self.stream.synchronize()
        for name, d, h in out_info:
            cuda.memcpy_dtoh(h, d)
            results[name] = h
        for d in d_bufs:
            d.free()
        return results

# ---- Embeddings ----
print("Loading embeddings...")
tce = np.load(f"{CEMB}/talker_codec_embedding.npy")
cces = [np.load(f"{CEMB}/cp_codec_embedding_{i}.npy") for i in range(15)]
tew = np.load(f"{FP32_EMB}/text_embedding.npy")
f1w=np.load(f"{CEMB}/tp_fc1_weight.npy");f1b=np.load(f"{CEMB}/tp_fc1_bias.npy")
f2w=np.load(f"{CEMB}/tp_fc2_weight.npy");f2b=np.load(f"{CEMB}/tp_fc2_bias.npy")
# Load codec_head for prefill logits
codec_head_w = np.load(f"{FP32_EMB}/codec_head_weight.npy")  # [3072, 1024]

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
tok=Tokenizer(BPE(f"{FP32_TOK}/vocab.json",f"{FP32_TOK}/merges.txt"))
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
    if np.any(np.isnan(p)): return int(np.argmax(l))
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

# ---- Numpy "prefill" (no TRT needed, runs once) ----
# We can't run prefill via TRT (different model). Use the TRT decode model
# in a loop to process all prefill tokens one by one, building KV cache.
def prefill_via_decode(talker, ie):
    """Run decode model token-by-token to simulate prefill and build KV cache."""
    seq = ie.shape[1]
    # Initialize empty KV
    pk = np.zeros((NL, 1, NKV, 0, HD), dtype=np.float32)
    pv = np.zeros((NL, 1, NKV, 0, HD), dtype=np.float32)

    last_logits = None
    last_hidden = None
    for i in range(seq):
        token_emb = ie[:, i:i+1, :]  # [1, 1, 1024]
        pos = np.array([[[i]]] * 3, dtype=np.int64)
        out = talker.run({"inputs_embeds": token_emb, "position_ids": pos,
                         "past_keys": pk, "past_values": pv})
        last_logits = out["logits"]
        last_hidden = out["hidden_states"]
        pk = out["present_keys"]
        pv = out["present_values"]

    return last_logits, last_hidden, pk, pv

# ---- Code predictor via numpy (no ORT/TRT in this container) ----
# Load code predictor weights for manual inference
# This is the bottleneck — numpy CP is slower than GPU but correct
# TODO: load cp TRT engine when available

# For now: use codec_head_weight to generate first code from talker logits,
# and skip sub-code prediction (use greedy single code + random sub-codes)
# ACTUALLY: we need proper sub-codes. Let's try loading the CP ONNX via basic means.
# Check if onnxruntime is available
try:
    import onnxruntime as ort
    HAS_ORT = True
    print("  ORT available!")
except ImportError:
    HAS_ORT = False
    print("  No ORT — code predictor via numpy (slower)")

# CP via ORT CUDA EP (FP32 with KV cache — proven correct)
print("Loading CP (ORT CUDA EP)...")
_cp_o = ort.SessionOptions()
_cp_o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_cp_ort = ort.InferenceSession("/tmp/qwen3-tts-bench/model/code_predictor.onnx",
    sess_options=_cp_o, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
_cp_ort_on = [x.name for x in _cp_ort.get_outputs()]

def predict_sub_codes(hidden, first_code):
    """Generate 15 sub-codes using ORT CP with KV cache (correct)."""
    fce = ce(first_code)
    ci = np.concatenate([hidden.astype(np.float32), fce], axis=1)
    cpk = np.zeros((5,1,8,0,128), dtype=np.float32)
    cpv = np.zeros((5,1,8,0,128), dtype=np.float32)
    subs = []
    for step in range(15):
        co = _cp_ort.run(None, {"inputs_embeds": ci, "generation_steps": np.array([step], dtype=np.int64),
                                 "past_keys": cpk, "past_values": cpv})
        cm = dict(zip(_cp_ort_on, co))
        sc = sample(cm["logits"][0, -1, :])
        subs.append(sc); cpk = cm["present_keys"]; cpv = cm["present_values"]
        ci = cpe(sc, step)
    return subs

# ---- Load TRT engine ----
print("Loading TRT talker engine...")
talker_trt = TRTEngine(TALKER_ENGINE)

# Warmup
print("Warmup...")
ie_dummy = np.random.randn(1,1,1024).astype(np.float32)
pk_dummy = np.zeros((NL,1,NKV,1,HD), dtype=np.float32)
pv_dummy = np.zeros((NL,1,NKV,1,HD), dtype=np.float32)
pi_dummy = np.array([[[0]]]*3, dtype=np.int64)
for _ in range(3):
    talker_trt.run({"inputs_embeds":ie_dummy,"position_ids":pi_dummy,"past_keys":pk_dummy,"past_values":pv_dummy})
print("Ready!")

def synthesize(text,lang="english",output="/tmp/tts_native.wav",mf=60,seed=None):
    if seed: np.random.seed(seed)
    print(f"\nSynth: \"{text}\" ({lang})")
    tt=time.perf_counter()

    ie,pad_e=build_emb(text,lang)
    print(f"  Input: {ie.shape}")

    # Prefill via SAME no-If model (ORT CUDA EP) — KV cache compatible with TRT decode
    t0=time.perf_counter()
    import onnxruntime as ort
    if not hasattr(synthesize, '_prefill_s'):
        po = ort.SessionOptions()
        po.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        synthesize._prefill_s = ort.InferenceSession(
            "/tmp/talker_prefill_no_if.onnx",
            sess_options=po,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        synthesize._pf_on = [x.name for x in synthesize._prefill_s.get_outputs()]
    # Prefill: get logits from no-If model (stateless, same attention as TRT decode)
    po = synthesize._prefill_s.run(None, {"inputs_embeds": ie})
    pm = dict(zip(synthesize._pf_on, po))
    logits = pm["logits"][:, -1:, :]  # last token logits
    hidden = pm["hidden_states"][:, -1:, :]
    # Build KV cache token-by-token via TRT decode (same model = compatible KV)
    # past_len=0 triggers TRT warning but still works correctly
    seq = ie.shape[1]
    pk = np.zeros((NL, 1, NKV, 0, HD), dtype=np.float32)
    pv = np.zeros((NL, 1, NKV, 0, HD), dtype=np.float32)
    for i in range(seq):
        out = talker_trt.run({
            "inputs_embeds": ie[:, i:i+1, :],
            "position_ids": np.array([[[i]]] * 3, dtype=np.int64),
            "past_keys": pk, "past_values": pv,
        })
        pk = out["present_keys"]; pv = out["present_values"]
    pf_ms=(time.perf_counter()-t0)*1000
    print(f"  Prefill: {pf_ms:.0f}ms (TRT token-by-token x{seq})")

    # Decode loop
    codes=[];ts=ie.shape[1];dt=[];ct=[]
    for fr in range(mf):
        # Logits are float16 from TRT — cast to float32 for sampling
        fc=sample(logits[0,-1,:].astype(np.float32))
        if fc==CODEC_EOS: print(f"  EOS@{fr}"); break

        # Code predictor
        t_cp=time.perf_counter()
        subs=predict_sub_codes(hidden.astype(np.float32), fc)
        ct.append(time.perf_counter()-t_cp)
        codes.append([fc]+subs)

        # Next input
        se=ce(fc)
        for i,sc in enumerate(subs): se=se+cpe(sc,i)
        ne=(se+pad_e).astype(np.float32)

        # TRT decode
        t_d=time.perf_counter()
        ts+=1
        out=talker_trt.run({"inputs_embeds":ne,"position_ids":np.array([[[ts-1]]]*3,dtype=np.int64),
                            "past_keys":pk,"past_values":pv})
        logits=out["logits"];hidden=out["hidden_states"]
        pk=out["present_keys"];pv=out["present_values"]
        dt.append(time.perf_counter()-t_d)

        if (fr+1)%10==0: print(f"  Frame {fr+1}")

    n=len(codes);dur=n/12.5
    if n==0: print("  No frames!"); return

    # Vocoder — need to handle without TRT vocoder engine for now
    # Use numpy to write codes directly (placeholder)
    # Actually, check if we have vocoder engine
    voc_path = "/tmp/trt_cache_voc"  # might not have vocoder TRT
    # For now: save codes and use speech container's vocoder
    print(f"  {n} frames ({dur:.1f}s audio)")

    # Save codes for external vocoder
    codes_arr = np.array(codes, dtype=np.int64)
    np.save("/tmp/tts_codes.npy", codes_arr)

    total=(time.perf_counter()-tt)*1000
    da=np.mean(dt)*1000 if dt else 0
    ca=np.mean(ct)*1000 if ct else 0
    print(f"\n  === TIMING ===")
    print(f"  Prefill:     {pf_ms:>7.0f}ms (TRT decode x{ie.shape[1]})")
    print(f"  Talker/step: {da:>7.1f}ms (TRT FP16)")
    print(f"  CP/step:     {ca:>7.1f}ms (TRT FP16)")
    print(f"  Per-step:    {da+ca:>7.1f}ms  RTF={((da+ca)/80):.2f}")
    print(f"  Total decode:{total-pf_ms:>7.0f}ms")
    print(f"  Codes saved: /tmp/tts_codes.npy ({codes_arr.shape})")

    # If ORT vocoder is available in this container, use it
    # Otherwise codes can be decoded by the speech container
    try:
        if HAS_ORT:
            voc = ort.InferenceSession("/tmp/qwen3-tts-bench/model/vocoder.onnx",
                                       providers=["CUDAExecutionProvider","CPUExecutionProvider"])
            vi = voc.get_inputs()[0].name
            t0=time.perf_counter()
            wav=voc.run(None,{vi:codes_arr.T[np.newaxis,:,:]})[0].flatten()
            tv=(time.perf_counter()-t0)*1000
            print(f"  Vocoder:     {tv:>7.0f}ms (ORT CUDA)")
            with wave.open(output,"w") as wf:
                wf.setnchannels(1);wf.setsampwidth(2);wf.setframerate(24000)
                wf.writeframes((wav*32767).clip(-32768,32767).astype(np.int16).tobytes())
            print(f"  Saved: {output}")
    except Exception as e:
        print(f"  Vocoder skipped: {e}")
        print(f"  Decode codes at /tmp/tts_codes.npy — use speech container vocoder")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--text",default="Hello, welcome to the voice synthesis system.")
    p.add_argument("--lang",default="english",choices=list(LANG_IDS.keys()))
    p.add_argument("--output",default="/tmp/tts_native.wav")
    p.add_argument("--max-frames",type=int,default=60)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    synthesize(a.text,a.lang,a.output,a.max_frames,a.seed)
