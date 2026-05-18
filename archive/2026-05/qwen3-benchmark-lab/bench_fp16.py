#!/usr/bin/env python3
"""Benchmark FP16 CUDA models vs baseline (INT8 CPU)."""
import time, gc, json, numpy as np, onnxruntime as ort

INT8 = "/tmp/qwen3-tts-bench/model-int8"
FP16 = "/tmp/qwen3-tts-bench/model-fp16"
H = 1024; W = 3; R = 10
CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CPU = ["CPUExecutionProvider"]

def sess(path, provs):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = 6
    return ort.InferenceSession(path, sess_options=o, providers=provs)

def bench(s, feed, w=W, r=R):
    for _ in range(w): s.run(None, feed)
    ts = []
    for _ in range(r):
        t0 = time.perf_counter()
        s.run(None, feed)
        ts.append((time.perf_counter()-t0)*1000)
    return round(np.mean(ts), 1), round(np.median(ts), 1)

print("=" * 65)
print("FP16 CUDA vs INT8 CPU — Key Bottleneck Components")
print("=" * 65)
print(f"ORT {ort.__version__}\n")
res = {}

# === CODE PREDICTOR (the bottleneck: 102ms baseline) ===
print("[CODE PREDICTOR x15]")
print("-" * 50)
cp_emb = np.random.randn(1, 1, H).astype(np.float32)

for label, path, prov in [
    ("INT8 CPU (baseline)", f"{INT8}/code_predictor_q.onnx", CPU),
    ("FP16 CUDA", f"{FP16}/code_predictor.onnx", CUDA),
    ("FP16 CPU", f"{FP16}/code_predictor.onnx", CPU),
]:
    s = sess(path, prov)
    for _ in range(W):
        for j in range(15):
            s.run(None, {"inputs_embeds": cp_emb, "generation_step": np.array([j], dtype=np.int64)})
    ts = []
    for _ in range(R):
        t0 = time.perf_counter()
        for j in range(15):
            s.run(None, {"inputs_embeds": cp_emb, "generation_step": np.array([j], dtype=np.int64)})
        ts.append((time.perf_counter()-t0)*1000)
    m = round(np.mean(ts), 1)
    print(f"  {label:25s}: {m:6.1f}ms (per step: {m/15:.1f}ms)")
    res[f"cp_{label}"] = m
    del s; gc.collect()

# === VOCODER ===
print(f"\n[VOCODER — 25 frames / 2s audio]")
print("-" * 50)

for label, path, prov, shape in [
    ("FP32 CUDA (baseline)", f"/tmp/qwen3-tts-bench/model/vocoder.onnx", CUDA, (1, 16, 25)),
    ("FP16 CUDA", f"{FP16}/vocoder.onnx", CUDA, None),  # detect shape
    ("FP16 CPU", f"{FP16}/vocoder.onnx", CPU, None),
]:
    s = sess(path, prov)
    inp_name = s.get_inputs()[0].name
    inp_shape = [d for d in s.get_inputs()[0].shape]
    # Determine shape
    if shape:
        codes = np.random.randint(0, 2048, shape, dtype=np.int64)
    elif inp_shape and len(inp_shape) == 3:
        # Could be [1, 16, N] or [1, N, 16]
        if inp_shape[1] == 16:
            codes = np.random.randint(0, 2048, (1, 16, 25), dtype=np.int64)
        else:
            codes = np.random.randint(0, 2048, (1, 25, 16), dtype=np.int64)
    else:
        codes = np.random.randint(0, 2048, (1, 16, 25), dtype=np.int64)

    m, p = bench(s, {inp_name: codes})
    rtf = round(m / 1000 / 2.0, 4)
    print(f"  {label:25s}: {m:6.1f}ms (RTF={rtf})")
    res[f"voc_{label}"] = m
    del s; gc.collect()

# === SUMMARY ===
print(f"\n{'='*65}")
print("COMPARISON: Baseline vs FP16 CUDA")
print(f"{'='*65}")

cp_base = res.get("cp_INT8 CPU (baseline)", 102)
cp_fp16 = res.get("cp_FP16 CUDA", 0)
voc_base = res.get("voc_FP32 CUDA (baseline)", 194)
voc_fp16 = res.get("voc_FP16 CUDA", 0)

# Talker decode stays at INT8 CPU = 35ms (no FP16 talker yet)
talker = 35

print(f"""
Component          Baseline      FP16 CUDA     Speedup
Code pred x15      {cp_base:>6.0f}ms      {cp_fp16:>6.0f}ms      {cp_base/cp_fp16:.1f}x
Vocoder (2s)       {voc_base:>6.0f}ms      {voc_fp16:>6.0f}ms      {voc_base/voc_fp16:.1f}x
Talker decode        {talker}ms        {talker}ms      (unchanged, INT8 CPU)

Per-step total:    {talker + cp_base:.0f}ms       {talker + cp_fp16:.0f}ms
Audio per step:      80ms         80ms
Streaming RTF:     {(talker+cp_base)/80:.2f}x        {(talker+cp_fp16)/80:.2f}x       {"✅ REAL-TIME" if (talker+cp_fp16) < 80 else "❌" if (talker+cp_fp16) > 80 else "⚠️ BORDERLINE"}
""")

print(json.dumps(res, indent=2))
