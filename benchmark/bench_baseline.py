#!/usr/bin/env python3
"""
Baseline benchmark: best known combo on Orin Nano.
- Talker/Code predictor: INT8 on CPU (faster than broken CUDA)
- Vocoder: FP32 on CUDA (34x faster than INT8)
- Text projection: INT8 CUDA
"""

import time, gc, json, numpy as np, onnxruntime as ort

INT8 = "/tmp/qwen3-tts-bench/model-int8"
FP32 = "/tmp/qwen3-tts-bench/model"
H = 1024; L = 28; KV = 8; HD = 128
W = 3; R = 10

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
        out = s.run(None, feed)
        ts.append((time.perf_counter()-t0)*1000)
    return np.mean(ts), np.median(ts), np.std(ts), out

CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CPU = ["CPUExecutionProvider"]

print("=" * 65)
print("BASELINE: Mixed INT8-CPU + FP32-CUDA on Orin Nano")
print("=" * 65)
print(f"ORT {ort.__version__}\n")
res = {}

# 1. Text projection — INT8 CUDA (small, works fine)
print("[1/5] Text projection (INT8 CUDA)")
s = sess(f"{INT8}/text_project_q.onnx", CUDA)
ids50 = np.random.randint(0, 150000, (1, 50), dtype=np.int64)
m, p, sd, _ = bench(s, {"input_ids": ids50})
print(f"  seq=50: {m:.1f}ms (p50={p:.1f})")
res["text_proj"] = round(m, 1)
del s; gc.collect()

# 2. Prefill — INT8 CPU
print("\n[2/5] Talker prefill (INT8 CPU)")
s = sess(f"{INT8}/talker_prefill_q.onnx", CPU)
emb50 = np.random.randn(1, 50, H).astype(np.float32)
mask50 = np.ones((1, 50), dtype=np.int64)
m, p, sd, out = bench(s, {"inputs_embeds": emb50, "attention_mask": mask50})
print(f"  seq=50: {m:.1f}ms (p50={p:.1f})")
res["prefill_50"] = round(m, 1)

# Extract KV cache
kv = {}
names = [o.name for o in s.get_outputs()]
out1 = s.run(None, {"inputs_embeds": emb50, "attention_mask": mask50})
for i, n in enumerate(names):
    if "present_key" in n or "present_value" in n:
        kv[n] = out1[i]
del s; gc.collect()

# 3. Decode step — INT8 CPU
print("\n[3/5] Talker decode step (INT8 CPU)")
s = sess(f"{INT8}/talker_decode_q.onnx", CPU)
inp_names = [i.name for i in s.get_inputs()]

for past_len in [50, 100, 200]:
    emb1 = np.random.randn(1, 1, H).astype(np.float32)
    mask = np.ones((1, past_len + 1), dtype=np.int64)
    feed = {"inputs_embeds": emb1, "attention_mask": mask}
    for n in inp_names:
        if n.startswith("past_key_"):
            idx = n.replace("past_key_", "")
            pk = f"present_key_{idx}"
            feed[n] = kv[pk] if pk in kv and past_len == 50 else np.zeros((1, KV, past_len, HD), dtype=np.float32)
        elif n.startswith("past_value_"):
            idx = n.replace("past_value_", "")
            pv = f"present_value_{idx}"
            feed[n] = kv[pv] if pv in kv and past_len == 50 else np.zeros((1, KV, past_len, HD), dtype=np.float32)
    m, p, sd, _ = bench(s, feed)
    rt = "✅" if m < 80 else "❌"
    print(f"  past={past_len}: {m:.1f}ms (p50={p:.1f}) {rt}")
    res[f"decode_{past_len}"] = round(m, 1)
del s; gc.collect()

# 4. Code predictor x15 — INT8 CPU
print("\n[4/5] Code predictor x15 (INT8 CPU)")
s = sess(f"{INT8}/code_predictor_q.onnx", CPU)
cp_emb = np.random.randn(1, 1, H).astype(np.float32)
for _ in range(W):
    for j in range(15):
        s.run(None, {"inputs_embeds": cp_emb, "generation_step": np.array([j], dtype=np.int64)})
ts = []
for _ in range(R):
    t0 = time.perf_counter()
    for j in range(15):
        s.run(None, {"inputs_embeds": cp_emb, "generation_step": np.array([j], dtype=np.int64)})
    ts.append((time.perf_counter()-t0)*1000)
m = np.mean(ts)
print(f"  15 steps: {m:.1f}ms (per step: {m/15:.1f}ms)")
res["code_pred_15"] = round(m, 1)
del s; gc.collect()

# 5. Vocoder — FP32 CUDA
print("\n[5/5] Vocoder (FP32 CUDA)")
s = sess(f"{FP32}/vocoder.onnx", CUDA)
inp_name = s.get_inputs()[0].name
for nf in [10, 25, 50]:
    codes = np.random.randint(0, 2048, (1, 16, nf), dtype=np.int64)
    m, p, sd, _ = bench(s, {inp_name: codes})
    dur = nf / 12.5
    rtf = m / 1000 / dur
    print(f"  frames={nf} ({dur:.1f}s): {m:.1f}ms (RTF={rtf:.4f})")
    res[f"vocoder_{nf}"] = round(m, 1)
del s; gc.collect()

# === SUMMARY ===
prefill = res.get("prefill_50", 0)
decode = res.get("decode_100", 0)
code_pred = res.get("code_pred_15", 0)
text_proj = res.get("text_proj", 0)
vocoder = res.get("vocoder_25", 0)
per_step = decode + code_pred

print(f"\n{'='*65}")
print("BASELINE vs MATCHA COMPARISON")
print(f"{'='*65}")
print(f"""
                        Baseline (mixed)    Matcha (current)
                        ----------------    ----------------
Text projection:        {text_proj:>6.0f}ms            N/A
Prefill (TTFA base):    {prefill:>6.0f}ms            ~60ms (zh)
Decode/step:            {decode:>6.0f}ms            N/A (non-AR)
Code pred x15/step:     {code_pred:>6.0f}ms            N/A
Per-step total:         {per_step:>6.0f}ms            N/A
Audio per step:             80ms               N/A
Vocoder (2s):           {vocoder:>6.0f}ms            included
""")

ttfa = text_proj + prefill + decode + code_pred
batch_2s = text_proj + prefill + per_step * 25 + vocoder
rtf_step = per_step / 80

if rtf_step < 1:
    print(f"Streaming:              ✅ RTF={rtf_step:.2f} (real-time OK)")
else:
    print(f"Streaming:              ❌ RTF={rtf_step:.2f} (need <1.0)")

print(f"TTFA:                   ~{ttfa:.0f}ms          ~60ms")
print(f"Batch 2s audio:         ~{batch_2s:.0f}ms         ~150ms")
print(f"Ratio vs Matcha TTFA:   {ttfa/60:.1f}x")
print(f"Ratio vs Matcha batch:  {batch_2s/150:.1f}x")

print(f"\n{'='*65}")
print("RAW DATA")
print(json.dumps(res, indent=2))
