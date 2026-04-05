#!/usr/bin/env python3
"""
Diagnose WHY Qwen3-TTS is slow on Orin Nano.
Check: CUDA vs CPU fallback, FP16 vs INT8, operator profiling.
"""

import time
import gc
import numpy as np
import onnxruntime as ort

MODEL_DIR = "/tmp/qwen3-tts-bench/model-int8"
MODEL_DIR_FP32 = "/tmp/qwen3-tts-bench/model"
HIDDEN = 1024
NUM_LAYERS = 28
KV_HEADS = 8
HEAD_DIM = 128
WARMUP = 2
RUNS = 5


def get_mem():
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS:"):
                return int(l.split()[1]) // 1024
    return 0


def create_sess(path, providers, profile=False):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    if profile:
        opts.enable_profiling = True
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


def check_provider_actual(sess):
    """Check which provider each node actually runs on."""
    # The session's get_providers() shows requested, not actual per-node
    return sess.get_providers()


def bench_one(label, sess, feed, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        sess.run(None, feed)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    mean = np.mean(times)
    p50 = np.median(times)
    print(f"  {label}: {mean:.1f}ms (p50={p50:.1f})")
    return mean


def main():
    print("=" * 65)
    print("DIAGNOSTIC: Why is Qwen3-TTS slow on Orin Nano?")
    print("=" * 65)
    print(f"ORT {ort.__version__}")
    print(f"Available: {ort.get_available_providers()}\n")

    # ============================================================
    # TEST 1: Is the vocoder falling back to CPU?
    # ============================================================
    print("[TEST 1] VOCODER — CUDA vs CPU comparison")
    print("-" * 50)

    codes_25 = np.random.randint(0, 2048, (1, 25, 16), dtype=np.int64)

    # CUDA
    sess = create_sess(f"{MODEL_DIR}/tokenizer12hz_decode_q.onnx",
                       ["CUDAExecutionProvider", "CPUExecutionProvider"])
    cuda_t = bench_one("INT8 CUDA", sess, {"audio_codes": codes_25})
    del sess; gc.collect()

    # CPU only
    sess = create_sess(f"{MODEL_DIR}/tokenizer12hz_decode_q.onnx",
                       ["CPUExecutionProvider"])
    cpu_t = bench_one("INT8 CPU-only", sess, {"audio_codes": codes_25})
    del sess; gc.collect()

    if cuda_t > cpu_t * 0.8:
        print(f"  ⚠️  CUDA ({cuda_t:.0f}ms) not faster than CPU ({cpu_t:.0f}ms)")
        print(f"  → Vocoder INT8 ops likely falling back to CPU!")
    else:
        print(f"  CUDA {cuda_t/cpu_t:.1f}x faster than CPU")

    # FP32 vocoder comparison (if available)
    import os
    fp32_vocoder = f"{MODEL_DIR_FP32}/vocoder.onnx"
    if os.path.exists(fp32_vocoder):
        print()
        print("[TEST 1b] VOCODER FP32 vs INT8")
        print("-" * 50)

        # FP32 vocoder has different input format: [batch, 16, timesteps]
        sess_fp32 = create_sess(fp32_vocoder,
                                ["CUDAExecutionProvider", "CPUExecutionProvider"])
        inp_names = [i.name for i in sess_fp32.get_inputs()]
        inp_shapes = [i.shape for i in sess_fp32.get_inputs()]
        print(f"  FP32 vocoder inputs: {list(zip(inp_names, inp_shapes))}")

        # Adapt shape
        codes_fp32 = np.random.randint(0, 2048, (1, 16, 25), dtype=np.int64)
        fp32_t = bench_one("FP32 CUDA", sess_fp32, {inp_names[0]: codes_fp32})
        del sess_fp32; gc.collect()

        print(f"  INT8: {cuda_t:.0f}ms vs FP32: {fp32_t:.0f}ms → ", end="")
        if fp32_t < cuda_t:
            print(f"FP32 is {cuda_t/fp32_t:.1f}x FASTER! INT8 quantization hurts here.")
        else:
            print(f"INT8 is {fp32_t/cuda_t:.1f}x faster")

    # ============================================================
    # TEST 2: Talker decode — CUDA utilization
    # ============================================================
    print()
    print("[TEST 2] TALKER DECODE — CUDA vs CPU")
    print("-" * 50)

    embeds = np.random.randn(1, 1, HIDDEN).astype(np.float32)
    mask = np.ones((1, 101), dtype=np.int64)
    kv = {}
    for i in range(NUM_LAYERS):
        kv[f"past_key_{i}"] = np.zeros((1, KV_HEADS, 100, HEAD_DIM), dtype=np.float32)
        kv[f"past_value_{i}"] = np.zeros((1, KV_HEADS, 100, HEAD_DIM), dtype=np.float32)
    feed = {"inputs_embeds": embeds, "attention_mask": mask, **kv}

    sess = create_sess(f"{MODEL_DIR}/talker_decode_q.onnx",
                       ["CUDAExecutionProvider", "CPUExecutionProvider"])
    cuda_t = bench_one("INT8 CUDA", sess, feed)
    del sess; gc.collect()

    sess = create_sess(f"{MODEL_DIR}/talker_decode_q.onnx",
                       ["CPUExecutionProvider"])
    cpu_t = bench_one("INT8 CPU-only", sess, feed)
    del sess; gc.collect()

    print(f"  CUDA/CPU ratio: {cpu_t/cuda_t:.1f}x")
    if cuda_t > 80:
        print(f"  ⚠️  Still >80ms with CUDA. Need TensorRT or model optimization.")

    # ============================================================
    # TEST 3: Code predictor — batching potential
    # ============================================================
    print()
    print("[TEST 3] CODE PREDICTOR — single vs batch")
    print("-" * 50)

    sess = create_sess(f"{MODEL_DIR}/code_predictor_q.onnx",
                       ["CUDAExecutionProvider", "CPUExecutionProvider"])

    # Single step
    emb1 = np.random.randn(1, 1, HIDDEN).astype(np.float32)
    t_single = bench_one("1 step", sess, {
        "inputs_embeds": emb1, "generation_step": np.array([0], dtype=np.int64)
    })

    # Check if batching across steps is possible
    emb15 = np.random.randn(1, 15, HIDDEN).astype(np.float32)
    try:
        t_batch = bench_one("15 steps batched", sess, {
            "inputs_embeds": emb15, "generation_step": np.array([0], dtype=np.int64)
        })
        print(f"  Serial 15x: {t_single*15:.0f}ms vs Batched: {t_batch:.0f}ms")
    except Exception as e:
        print(f"  Batching failed: {e}")
        print(f"  Serial 15x: {t_single*15:.0f}ms (no batching possible)")

    del sess; gc.collect()

    # ============================================================
    # TEST 4: TensorRT potential
    # ============================================================
    print()
    print("[TEST 4] TENSORRT PROVIDER")
    print("-" * 50)

    if "TensorrtExecutionProvider" in ort.get_available_providers():
        # Test prefill with TRT
        sess = create_sess(f"{MODEL_DIR}/talker_prefill_q.onnx",
                           ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
        embeds_50 = np.random.randn(1, 50, HIDDEN).astype(np.float32)
        mask_50 = np.ones((1, 50), dtype=np.int64)
        print("  Building TRT engine (first run slow)...")
        try:
            trt_t = bench_one("Prefill TRT seq=50", sess,
                              {"inputs_embeds": embeds_50, "attention_mask": mask_50},
                              warmup=1, runs=3)
            print(f"  CUDA was 259ms, TRT is {trt_t:.0f}ms → {259/trt_t:.1f}x speedup")
        except Exception as e:
            print(f"  TRT failed: {e}")
        del sess; gc.collect()

        # Test vocoder with TRT
        try:
            sess = create_sess(f"{MODEL_DIR}/tokenizer12hz_decode_q.onnx",
                               ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
            print("  Building TRT engine for vocoder...")
            trt_voc = bench_one("Vocoder TRT 25fr", sess, {"audio_codes": codes_25},
                                warmup=1, runs=3)
            print(f"  CUDA was 7000ms, TRT is {trt_voc:.0f}ms → {7000/trt_voc:.1f}x speedup")
        except Exception as e:
            print(f"  Vocoder TRT failed: {e}")
        del sess; gc.collect()
    else:
        print("  TensorRT provider not available")

    # ============================================================
    # SUMMARY
    # ============================================================
    print()
    print("=" * 65)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 65)
    print("""
1. VOCODER: If INT8 is falling back to CPU, use FP32/FP16 instead
2. TENSORRT: Convert key models to TRT engines for 2-3x speedup
3. CODE PREDICTOR: If batching works, can cut 15x overhead significantly
4. KV CACHE: Use IO binding to avoid CPU↔GPU memcpy per decode step
5. STREAMING VOCODER: Decode audio incrementally (every N frames)
6. FP16 MODEL: INT8 may actually be slower on Jetson due to CPU fallback
""")


if __name__ == "__main__":
    main()
