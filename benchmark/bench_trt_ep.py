#!/usr/bin/env python3
"""
Benchmark: FP32 model + TensorRT EP (auto FP16).
TRT EP auto-converts FP32 graphs to FP16 internally.
First run builds TRT engine (slow), subsequent runs are fast.
Tests one model at a time to avoid OOM.
"""
import time, gc, json, numpy as np, onnxruntime as ort, os

FP32 = "/tmp/qwen3-tts-bench/model"
INT8 = "/tmp/qwen3-tts-bench/model-int8"
H = 1024; W = 2; R = 5
CPU = ["CPUExecutionProvider"]
CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]

ENGINE_CACHE = "/tmp/qwen3-tts-bench/trt_cache"
os.makedirs(ENGINE_CACHE, exist_ok=True)

def sess_trt(path):
    """Create session with TRT EP (auto FP16)."""
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        ("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": ENGINE_CACHE,
            "trt_max_workspace_size": str(1 * 1024 * 1024 * 1024),  # 1GB
        }),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    return ort.InferenceSession(path, sess_options=o, providers=providers)

def sess_plain(path, provs):
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

def get_rss():
    try:
        with open("/proc/self/status") as f:
            for l in f:
                if l.startswith("VmRSS:"): return int(l.split()[1])//1024
    except: pass
    return 0

print("=" * 65)
print("TRT EP (auto FP16) vs Baseline")
print("=" * 65)
print(f"ORT {ort.__version__}")
print(f"Available: {ort.get_available_providers()}")
print(f"TRT EP: {'TensorrtExecutionProvider' in ort.get_available_providers()}")
print()

res = {}

# === 1. CODE PREDICTOR ===
print("[1/2] CODE PREDICTOR x15")
print("-" * 50)
cp_emb = np.random.randn(1, 1, H).astype(np.float32)

# Baseline: INT8 CPU
s = sess_plain(f"{INT8}/code_predictor_q.onnx", CPU)
for _ in range(W):
    for j in range(15):
        s.run(None, {"inputs_embeds": cp_emb, "generation_step": np.array([j], dtype=np.int64)})
ts = []
for _ in range(R):
    t0 = time.perf_counter()
    for j in range(15):
        s.run(None, {"inputs_embeds": cp_emb, "generation_step": np.array([j], dtype=np.int64)})
    ts.append((time.perf_counter()-t0)*1000)
m_base = round(np.mean(ts), 1)
print(f"  INT8 CPU (baseline):   {m_base:6.1f}ms ({m_base/15:.1f}ms/step)")
res["cp_int8_cpu"] = m_base
del s; gc.collect()

# FP32 code_predictor has extra inputs: past_keys, past_values
cp_pk = np.zeros((5, 1, 8, 0, 128), dtype=np.float32)
cp_pv = np.zeros((5, 1, 8, 0, 128), dtype=np.float32)
cp_feed_fp32 = lambda j: {
    "inputs_embeds": cp_emb,
    "generation_steps": np.array([j], dtype=np.int64),
    "past_keys": cp_pk,
    "past_values": cp_pv,
}

# FP32 CUDA
s = sess_plain(f"{FP32}/code_predictor.onnx", CUDA)
for _ in range(W):
    for j in range(15):
        s.run(None, cp_feed_fp32(j))
ts = []
for _ in range(R):
    t0 = time.perf_counter()
    for j in range(15):
        s.run(None, cp_feed_fp32(j))
    ts.append((time.perf_counter()-t0)*1000)
m_fp32 = round(np.mean(ts), 1)
print(f"  FP32 CUDA:             {m_fp32:6.1f}ms ({m_fp32/15:.1f}ms/step)")
res["cp_fp32_cuda"] = m_fp32
del s; gc.collect()

# TRT EP (auto FP16)
if "TensorrtExecutionProvider" in ort.get_available_providers():
    print(f"  Building TRT engine (first run, be patient)... RSS={get_rss()}MB")
    try:
        s = sess_trt(f"{FP32}/code_predictor.onnx")
        for j in range(15):
            s.run(None, cp_feed_fp32(j))
        print(f"  TRT engine built. RSS={get_rss()}MB")

        for _ in range(W):
            for j in range(15):
                s.run(None, cp_feed_fp32(j))
        ts = []
        for _ in range(R):
            t0 = time.perf_counter()
            for j in range(15):
                s.run(None, cp_feed_fp32(j))
            ts.append((time.perf_counter()-t0)*1000)
        m_trt = round(np.mean(ts), 1)
        print(f"  TRT EP (FP16):         {m_trt:6.1f}ms ({m_trt/15:.1f}ms/step)")
        res["cp_trt_fp16"] = m_trt
        del s; gc.collect()
    except Exception as e:
        print(f"  TRT EP failed: {e}")
        m_trt = 0
else:
    print("  TRT EP not available")
    m_trt = 0

# === 2. VOCODER ===
print(f"\n[2/2] VOCODER — 25 frames / 2s audio")
print("-" * 50)

# Baseline: FP32 CUDA
s = sess_plain(f"{FP32}/vocoder.onnx", CUDA)
inp = s.get_inputs()[0].name
codes = np.random.randint(0, 2048, (1, 16, 25), dtype=np.int64)
m_vbase, _ = bench(s, {inp: codes})
rtf = round(m_vbase/1000/2.0, 4)
print(f"  FP32 CUDA (baseline):  {m_vbase:6.1f}ms (RTF={rtf})")
res["voc_fp32_cuda"] = m_vbase
del s; gc.collect()

# TRT EP
if "TensorrtExecutionProvider" in ort.get_available_providers():
    print(f"  Building TRT engine... RSS={get_rss()}MB")
    try:
        s = sess_trt(f"{FP32}/vocoder.onnx")
        inp = s.get_inputs()[0].name
        # Warmup / engine build
        s.run(None, {inp: codes})
        print(f"  TRT engine built. RSS={get_rss()}MB")
        m_vtrt, _ = bench(s, {inp: codes})
        rtf = round(m_vtrt/1000/2.0, 4)
        print(f"  TRT EP (FP16):         {m_vtrt:6.1f}ms (RTF={rtf})")
        res["voc_trt_fp16"] = m_vtrt
        del s; gc.collect()
    except Exception as e:
        print(f"  TRT EP failed: {e}")
        m_vtrt = m_vbase

# === SUMMARY ===
talker = 35  # INT8 CPU baseline, unchanged
cp_best = min(m_base, m_fp32, m_trt if m_trt else 9999)
voc_best = min(m_vbase, res.get("voc_trt_fp16", 9999))

print(f"\n{'='*65}")
print("SUMMARY: Best Achievable vs Baseline vs Matcha")
print(f"{'='*65}")
print(f"""
                    Baseline    FP32 CUDA   TRT FP16    Matcha
Code pred x15       {m_base:>6.0f}ms    {m_fp32:>6.0f}ms    {m_trt if m_trt else 'N/A':>6}ms
Vocoder (2s)        {m_vbase:>6.0f}ms                    {res.get('voc_trt_fp16', 'N/A'):>6}ms    ~incl
Talker decode          {talker}ms       {talker}ms       {talker}ms      N/A

Per-step:           {talker+m_base:>6.0f}ms    {talker+m_fp32:>6.0f}ms    {talker+cp_best:>6.0f}ms      N/A
Audio/step:            80ms       80ms       80ms
RTF:                {(talker+m_base)/80:>5.2f}x     {(talker+m_fp32)/80:>5.2f}x     {(talker+cp_best)/80:>5.2f}x

TTFA (est):         ~284ms                  ~{142+talker+cp_best:.0f}ms     ~60ms
Batch 2s:           ~{142+25*(talker+m_base)+m_vbase:.0f}ms                ~{142+25*(talker+cp_best)+voc_best:.0f}ms    ~150ms
""")

streaming = (talker + cp_best) < 80
print(f"Real-time streaming: {'✅ YES' if streaming else '❌ NO'} (per-step={talker+cp_best:.0f}ms vs 80ms threshold)")
print(f"\n{json.dumps(res, indent=2)}")
