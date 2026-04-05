#!/usr/bin/env python3
"""Minimal diagnosis: is INT8 hurting us? CUDA vs CPU?"""
import time, gc, numpy as np, onnxruntime as ort, os

INT8 = "/tmp/qwen3-tts-bench/model-int8"
FP32 = "/tmp/qwen3-tts-bench/model"
W, R = 2, 5

def sess(path, prov):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = 4
    return ort.InferenceSession(path, sess_options=o, providers=prov)

def bench(label, s, feed):
    for _ in range(W): s.run(None, feed)
    ts = []
    for _ in range(R):
        t0 = time.perf_counter()
        s.run(None, feed)
        ts.append((time.perf_counter()-t0)*1000)
    m = np.mean(ts); p = np.median(ts)
    print(f"  {label:30s}: {m:8.1f}ms (p50={p:.1f})")
    return m

print("="*60)
print("DIAGNOSIS: INT8 vs FP32, CUDA vs CPU")
print("="*60)
print(f"ORT {ort.__version__}\n")

codes_int8 = np.random.randint(0, 2048, (1, 25, 16), dtype=np.int64)
codes_fp32 = np.random.randint(0, 2048, (1, 16, 25), dtype=np.int64)

# === VOCODER ===
print("[VOCODER] 25 frames / 2s audio")
print("-"*60)

s = sess(f"{INT8}/tokenizer12hz_decode_q.onnx", ["CUDAExecutionProvider","CPUExecutionProvider"])
v_int8_cuda = bench("INT8 + CUDA EP", s, {"audio_codes": codes_int8})
del s; gc.collect()

s = sess(f"{INT8}/tokenizer12hz_decode_q.onnx", ["CPUExecutionProvider"])
v_int8_cpu = bench("INT8 + CPU only", s, {"audio_codes": codes_int8})
del s; gc.collect()

if os.path.exists(f"{FP32}/vocoder.onnx"):
    s = sess(f"{FP32}/vocoder.onnx", ["CUDAExecutionProvider","CPUExecutionProvider"])
    inp = [i.name for i in s.get_inputs()]
    v_fp32_cuda = bench("FP32 + CUDA EP", s, {inp[0]: codes_fp32})
    del s; gc.collect()

    s = sess(f"{FP32}/vocoder.onnx", ["CPUExecutionProvider"])
    inp = [i.name for i in s.get_inputs()]
    v_fp32_cpu = bench("FP32 + CPU only", s, {inp[0]: codes_fp32})
    del s; gc.collect()
else:
    v_fp32_cuda = v_fp32_cpu = 0
    print("  (FP32 vocoder not found)")

print()
if v_int8_cuda > v_int8_cpu * 0.9:
    print("  ⚠️  INT8 CUDA ≈ CPU → ops falling back to CPU!")
if v_fp32_cuda > 0 and v_fp32_cuda < v_int8_cuda:
    print(f"  ⚠️  FP32 ({v_fp32_cuda:.0f}ms) FASTER than INT8 ({v_int8_cuda:.0f}ms)!")
    print(f"     → INT8 quantization is HURTING performance ({v_int8_cuda/v_fp32_cuda:.1f}x slower)")

# === TALKER DECODE ===
print()
print("[TALKER DECODE] past_len=100, single step")
print("-"*60)

emb = np.random.randn(1, 1, 1024).astype(np.float32)
mask = np.ones((1, 101), dtype=np.int64)
kv = {}
for i in range(28):
    kv[f"past_key_{i}"] = np.zeros((1, 8, 100, 128), dtype=np.float32)
    kv[f"past_value_{i}"] = np.zeros((1, 8, 100, 128), dtype=np.float32)
feed_d = {"inputs_embeds": emb, "attention_mask": mask, **kv}

s = sess(f"{INT8}/talker_decode_q.onnx", ["CUDAExecutionProvider","CPUExecutionProvider"])
d_cuda = bench("INT8 + CUDA EP", s, feed_d)
del s; gc.collect()

s = sess(f"{INT8}/talker_decode_q.onnx", ["CPUExecutionProvider"])
d_cpu = bench("INT8 + CPU only", s, feed_d)
del s; gc.collect()

print()
if d_cuda < d_cpu * 0.5:
    print(f"  ✅ CUDA is {d_cpu/d_cuda:.1f}x faster — GPU is working for talker")
else:
    print(f"  ⚠️  CUDA ({d_cuda:.0f}ms) vs CPU ({d_cpu:.0f}ms) — minimal speedup")

# === CODE PREDICTOR ===
print()
print("[CODE PREDICTOR] 15 steps serial")
print("-"*60)

cp_emb = np.random.randn(1, 1, 1024).astype(np.float32)
gs = np.array([0], dtype=np.int64)

s = sess(f"{INT8}/code_predictor_q.onnx", ["CUDAExecutionProvider","CPUExecutionProvider"])
for _ in range(W):
    for j in range(15):
        s.run(None, {"inputs_embeds": cp_emb, "generation_step": np.array([j], dtype=np.int64)})
ts = []
for _ in range(R):
    t0 = time.perf_counter()
    for j in range(15):
        s.run(None, {"inputs_embeds": cp_emb, "generation_step": np.array([j], dtype=np.int64)})
    ts.append((time.perf_counter()-t0)*1000)
cp_cuda = np.mean(ts)
print(f"  {'INT8 + CUDA EP':30s}: {cp_cuda:8.1f}ms (per step: {cp_cuda/15:.1f}ms)")
del s; gc.collect()

s = sess(f"{INT8}/code_predictor_q.onnx", ["CPUExecutionProvider"])
for _ in range(W):
    for j in range(15):
        s.run(None, {"inputs_embeds": cp_emb, "generation_step": np.array([j], dtype=np.int64)})
ts = []
for _ in range(R):
    t0 = time.perf_counter()
    for j in range(15):
        s.run(None, {"inputs_embeds": cp_emb, "generation_step": np.array([j], dtype=np.int64)})
    ts.append((time.perf_counter()-t0)*1000)
cp_cpu = np.mean(ts)
print(f"  {'INT8 + CPU only':30s}: {cp_cpu:8.1f}ms (per step: {cp_cpu/15:.1f}ms)")
del s; gc.collect()

# === SUMMARY ===
print()
print("="*60)
print("SUMMARY")
print("="*60)
print(f"""
Component          INT8+CUDA   INT8+CPU    FP32+CUDA   Bottleneck
Vocoder (2s)       {v_int8_cuda:7.0f}ms   {v_int8_cpu:7.0f}ms   {v_fp32_cuda:7.0f}ms   {'CPU fallback!' if v_int8_cuda > v_int8_cpu*0.9 else 'OK'}
Talker decode      {d_cuda:7.0f}ms   {d_cpu:7.0f}ms   {'N/A':>7s}     {'CPU fallback!' if d_cuda > d_cpu*0.5 else 'OK'}
Code pred (x15)    {cp_cuda:7.0f}ms   {cp_cpu:7.0f}ms   {'N/A':>7s}     {'CPU fallback!' if cp_cuda > cp_cpu*0.9 else 'OK'}

Target per-step:   <80ms decode + <80ms code_pred = <160ms total
Audio per step:    80ms (12.5 Hz codec)

Recommended next:
""")
if v_int8_cuda > v_int8_cpu * 0.9:
    print("  1. USE FP32/FP16 VOCODER instead of INT8 (avoid CPU fallback)")
if d_cuda > d_cpu * 0.5:
    print("  2. Convert talker to TensorRT engine (need separate trtexec)")
if v_fp32_cuda > 0 and v_fp32_cuda < 500:
    print(f"  3. FP32 vocoder is {v_fp32_cuda:.0f}ms — probably fine for streaming!")
print("  4. Try FP16 ONNX export for all models (best CUDA utilization on Ampere)")
