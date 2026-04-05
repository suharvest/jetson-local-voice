#!/usr/bin/env python3
"""
Benchmark: Code Predictor IO Binding optimization.

Goal: Reduce 15-step code predictor loop from ~90ms to <45ms by using
ONNX Runtime IO Binding to eliminate CPU<->GPU data copies and reduce
Python call overhead.

Strategies tested:
  A) Baseline CPU (current best: ~6ms/step, 90ms total)
  B) CUDA naive session.run() — typically slower due to sync overhead
  C) CUDA + IO Binding — pre-allocated GPU buffers, no copies
  D) CUDA + IO Binding + pre-bound outputs (reuse output buffers)
  E) CUDA + IO Binding + OrtValue on GPU (zero-copy input construction)

Model: code_predictor_q.onnx (INT8) / code_predictor.onnx (FP32)
  Inputs:  inputs_embeds [1,1,1024] float32, generation_step [1] int64
  Outputs: logits [1,2048] float32
"""

import time
import gc
import json
import sys
import numpy as np
import onnxruntime as ort

# ── Paths ────────────────────────────────────────────────────────────
INT8_MODEL = "/tmp/qwen3-tts-bench/model-int8/code_predictor_q.onnx"
FP32_MODEL = "/tmp/qwen3-tts-bench/model/code_predictor.onnx"

HIDDEN = 1024
NUM_STEPS = 15   # codebook groups per TTS decode step
WARMUP = 5
RUNS = 20

CPU = ["CPUExecutionProvider"]
CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]


# ── Utilities ────────────────────────────────────────────────────────

def get_mem_mb():
    """RSS in MB (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return 0


def stats(times_ms):
    arr = np.array(times_ms)
    return {
        "mean": round(float(np.mean(arr)), 2),
        "std": round(float(np.std(arr)), 2),
        "p50": round(float(np.median(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "min": round(float(np.min(arr)), 2),
    }


def make_session(model_path, providers, threads=6):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads
    # Disable memory pattern for IO binding — allows buffer reuse
    opts.enable_mem_pattern = False
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


def has_cuda():
    return "CUDAExecutionProvider" in ort.get_available_providers()


# ── Strategy A: CPU baseline (session.run) ───────────────────────────

def bench_cpu_baseline(model_path, threads=6):
    """Current approach: 15 serial session.run() calls on CPU."""
    sess = make_session(model_path, CPU, threads)
    embeds = np.random.randn(1, 1, HIDDEN).astype(np.float32)
    steps = [np.array([i], dtype=np.int64) for i in range(NUM_STEPS)]

    # warmup
    for _ in range(WARMUP):
        for s in steps:
            sess.run(None, {"inputs_embeds": embeds, "generation_step": s})

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for s in steps:
            sess.run(None, {"inputs_embeds": embeds, "generation_step": s})
        times.append((time.perf_counter() - t0) * 1000)

    del sess; gc.collect()
    return stats(times)


# ── Strategy B: CUDA naive session.run ───────────────────────────────

def bench_cuda_naive(model_path):
    """CUDA with standard session.run() — has CPU<->GPU copy overhead."""
    if not has_cuda():
        return None
    sess = make_session(model_path, CUDA)
    embeds = np.random.randn(1, 1, HIDDEN).astype(np.float32)
    steps = [np.array([i], dtype=np.int64) for i in range(NUM_STEPS)]

    for _ in range(WARMUP):
        for s in steps:
            sess.run(None, {"inputs_embeds": embeds, "generation_step": s})

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for s in steps:
            sess.run(None, {"inputs_embeds": embeds, "generation_step": s})
        times.append((time.perf_counter() - t0) * 1000)

    del sess; gc.collect()
    return stats(times)


# ── Strategy C: CUDA + IO Binding (basic) ────────────────────────────

def bench_cuda_iobinding_basic(model_path):
    """
    IO Binding: bind numpy inputs on CPU, let ORT copy to GPU once,
    bind output on GPU to avoid copy-back.
    """
    if not has_cuda():
        return None
    sess = make_session(model_path, CUDA)
    embeds = np.random.randn(1, 1, HIDDEN).astype(np.float32)
    steps = [np.array([i], dtype=np.int64) for i in range(NUM_STEPS)]

    # warmup
    for _ in range(WARMUP):
        for s in steps:
            binding = sess.io_binding()
            binding.bind_cpu_input("inputs_embeds", embeds)
            binding.bind_cpu_input("generation_step", s)
            binding.bind_output("logits", "cuda")
            sess.run_with_iobinding(binding)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for s in steps:
            binding = sess.io_binding()
            binding.bind_cpu_input("inputs_embeds", embeds)
            binding.bind_cpu_input("generation_step", s)
            binding.bind_output("logits", "cuda")
            sess.run_with_iobinding(binding)
        times.append((time.perf_counter() - t0) * 1000)

    del sess; gc.collect()
    return stats(times)


# ── Strategy D: CUDA + IO Binding + pre-allocated OrtValues ──────────

def bench_cuda_iobinding_prealloc(model_path):
    """
    Pre-allocate GPU OrtValues for inputs AND outputs before the loop.
    Only update the generation_step value each iteration.
    The inputs_embeds stays constant across all 15 steps (same hidden state).
    """
    if not has_cuda():
        return None
    sess = make_session(model_path, CUDA)

    # Pre-create OrtValues on CUDA
    embeds_np = np.random.randn(1, 1, HIDDEN).astype(np.float32)

    # Create persistent GPU tensors
    embeds_gpu = ort.OrtValue.ortvalue_from_numpy(embeds_np, "cuda", 0)
    output_gpu = ort.OrtValue.ortvalue_from_shape_and_type(
        [1, 2048], np.float32, "cuda", 0
    )

    # Pre-create all 15 step tensors on GPU
    step_gpus = []
    for i in range(NUM_STEPS):
        step_np = np.array([i], dtype=np.int64)
        step_gpus.append(ort.OrtValue.ortvalue_from_numpy(step_np, "cuda", 0))

    # warmup
    for _ in range(WARMUP):
        for i in range(NUM_STEPS):
            binding = sess.io_binding()
            binding.bind_ortvalue_input("inputs_embeds", embeds_gpu)
            binding.bind_ortvalue_input("generation_step", step_gpus[i])
            binding.bind_ortvalue_output("logits", output_gpu)
            sess.run_with_iobinding(binding)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for i in range(NUM_STEPS):
            binding = sess.io_binding()
            binding.bind_ortvalue_input("inputs_embeds", embeds_gpu)
            binding.bind_ortvalue_input("generation_step", step_gpus[i])
            binding.bind_ortvalue_output("logits", output_gpu)
            sess.run_with_iobinding(binding)
        times.append((time.perf_counter() - t0) * 1000)

    del sess, embeds_gpu, output_gpu, step_gpus
    gc.collect()
    return stats(times)


# ── Strategy E: CUDA + IO Binding + reused binding object ────────────

def bench_cuda_iobinding_reuse_binding(model_path):
    """
    Reuse a SINGLE io_binding object across the loop, only rebinding
    the generation_step input each iteration. Minimizes Python overhead
    from constructing binding objects.
    """
    if not has_cuda():
        return None
    sess = make_session(model_path, CUDA)

    embeds_np = np.random.randn(1, 1, HIDDEN).astype(np.float32)
    embeds_gpu = ort.OrtValue.ortvalue_from_numpy(embeds_np, "cuda", 0)
    output_gpu = ort.OrtValue.ortvalue_from_shape_and_type(
        [1, 2048], np.float32, "cuda", 0
    )
    step_gpus = []
    for i in range(NUM_STEPS):
        step_gpus.append(
            ort.OrtValue.ortvalue_from_numpy(
                np.array([i], dtype=np.int64), "cuda", 0
            )
        )

    # Create 15 pre-bound binding objects (one per step)
    bindings = []
    for i in range(NUM_STEPS):
        b = sess.io_binding()
        b.bind_ortvalue_input("inputs_embeds", embeds_gpu)
        b.bind_ortvalue_input("generation_step", step_gpus[i])
        b.bind_ortvalue_output("logits", output_gpu)
        bindings.append(b)

    # warmup
    for _ in range(WARMUP):
        for b in bindings:
            sess.run_with_iobinding(b)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for b in bindings:
            sess.run_with_iobinding(b)
        times.append((time.perf_counter() - t0) * 1000)

    del sess, bindings, embeds_gpu, output_gpu, step_gpus
    gc.collect()
    return stats(times)


# ── Strategy F: CPU with pre-allocated numpy arrays ──────────────────

def bench_cpu_prealloc(model_path, threads=6):
    """
    CPU baseline but with pre-allocated feed dicts to reduce Python
    dict/array construction overhead.
    """
    sess = make_session(model_path, CPU, threads)
    embeds = np.random.randn(1, 1, HIDDEN).astype(np.float32)

    # Pre-build all 15 feed dicts
    feeds = []
    for i in range(NUM_STEPS):
        feeds.append({
            "inputs_embeds": embeds,
            "generation_step": np.array([i], dtype=np.int64),
        })

    for _ in range(WARMUP):
        for f in feeds:
            sess.run(None, f)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for f in feeds:
            sess.run(None, f)
        times.append((time.perf_counter() - t0) * 1000)

    del sess; gc.collect()
    return stats(times)


# ── Strategy G: CPU multithreaded sweep ──────────────────────────────

def bench_cpu_thread_sweep(model_path):
    """Test different intra_op_num_threads values on CPU."""
    results = {}
    for t in [1, 2, 4, 6, 8]:
        s = bench_cpu_baseline(model_path, threads=t)
        results[t] = s
        print(f"    threads={t}: {s['mean']:.1f}ms (p50={s['p50']:.1f})")
    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    # Pick model
    import os
    use_int8 = "--fp32" not in sys.argv
    model_path = INT8_MODEL if use_int8 else FP32_MODEL
    model_label = "INT8" if use_int8 else "FP32"

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        print("  Copy models to /tmp/qwen3-tts-bench/ first.")
        sys.exit(1)

    print("=" * 70)
    print(f"Code Predictor IO Binding Benchmark ({model_label})")
    print("=" * 70)
    print(f"ORT {ort.__version__} | CUDA: {has_cuda()}")
    print(f"Model: {model_path}")
    print(f"Config: {NUM_STEPS} steps, {WARMUP} warmup, {RUNS} runs")
    print(f"RSS: {get_mem_mb():.0f}MB")
    print()

    results = {}

    # --- A: CPU baseline ---
    print("[A] CPU baseline (session.run x15, threads=6)")
    s = bench_cpu_baseline(model_path)
    print(f"    {s['mean']:.1f}ms total, {s['mean']/NUM_STEPS:.1f}ms/step")
    results["A_cpu_baseline"] = s

    # --- A2: CPU with pre-allocated feeds ---
    print("\n[A2] CPU + pre-allocated feed dicts")
    s = bench_cpu_prealloc(model_path)
    print(f"    {s['mean']:.1f}ms total, {s['mean']/NUM_STEPS:.1f}ms/step")
    results["A2_cpu_prealloc"] = s

    if has_cuda():
        # --- B: CUDA naive ---
        print("\n[B] CUDA naive (session.run x15)")
        s = bench_cuda_naive(model_path)
        print(f"    {s['mean']:.1f}ms total, {s['mean']/NUM_STEPS:.1f}ms/step")
        results["B_cuda_naive"] = s

        # --- C: CUDA IO Binding basic ---
        print("\n[C] CUDA + IO Binding (bind_cpu_input, bind_output on cuda)")
        s = bench_cuda_iobinding_basic(model_path)
        print(f"    {s['mean']:.1f}ms total, {s['mean']/NUM_STEPS:.1f}ms/step")
        results["C_cuda_iobinding_basic"] = s

        # --- D: CUDA IO Binding pre-allocated ---
        print("\n[D] CUDA + IO Binding + pre-allocated OrtValues on GPU")
        s = bench_cuda_iobinding_prealloc(model_path)
        print(f"    {s['mean']:.1f}ms total, {s['mean']/NUM_STEPS:.1f}ms/step")
        results["D_cuda_iobinding_prealloc"] = s

        # --- E: CUDA IO Binding reuse binding ---
        print("\n[E] CUDA + IO Binding + pre-bound bindings (15 reused objects)")
        s = bench_cuda_iobinding_reuse_binding(model_path)
        print(f"    {s['mean']:.1f}ms total, {s['mean']/NUM_STEPS:.1f}ms/step")
        results["E_cuda_iobinding_reuse"] = s
    else:
        print("\n[B-E] Skipped (no CUDA)")

    # --- CPU thread sweep ---
    print(f"\n[F] CPU thread sweep (finding optimal thread count)")
    thread_results = bench_cpu_thread_sweep(model_path)
    best_t = min(thread_results, key=lambda t: thread_results[t]["mean"])
    print(f"    Best: threads={best_t} -> {thread_results[best_t]['mean']:.1f}ms")
    results["F_thread_sweep"] = {
        str(k): v for k, v in thread_results.items()
    }

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY: 15-step code predictor total latency (ms)")
    print(f"{'='*70}")
    print(f"{'Strategy':<50} {'Mean':>8} {'P50':>8} {'P95':>8}")
    print("-" * 70)

    summary_keys = [
        ("A_cpu_baseline",           "A) CPU baseline (6 threads)"),
        ("A2_cpu_prealloc",          "A2) CPU + pre-alloc feeds"),
        ("B_cuda_naive",             "B) CUDA naive session.run"),
        ("C_cuda_iobinding_basic",   "C) CUDA + IO Binding basic"),
        ("D_cuda_iobinding_prealloc","D) CUDA + IO Binding pre-alloc GPU"),
        ("E_cuda_iobinding_reuse",   "E) CUDA + IO Binding reuse bindings"),
    ]

    best_key = None
    best_mean = float("inf")
    for key, label in summary_keys:
        if key in results and results[key] is not None:
            r = results[key]
            print(f"{label:<50} {r['mean']:>8.1f} {r['p50']:>8.1f} {r['p95']:>8.1f}")
            if r["mean"] < best_mean:
                best_mean = r["mean"]
                best_key = label

    baseline = results.get("A_cpu_baseline", {}).get("mean", 90)
    print(f"\nBest: {best_key} at {best_mean:.1f}ms")
    print(f"Speedup vs CPU baseline: {baseline/best_mean:.2f}x")

    target = 45.0
    if best_mean <= target:
        print(f"TARGET MET: {best_mean:.1f}ms <= {target}ms")
    else:
        print(f"TARGET MISSED: {best_mean:.1f}ms > {target}ms")
        print(f"  Gap: {best_mean - target:.1f}ms")

    # ── Decode step budget ───────────────────────────────────────────
    audio_per_step_ms = 80.0  # 1000ms / 12.5 Hz
    print(f"\nDecode step budget (audio = {audio_per_step_ms:.0f}ms/step):")
    print(f"  Code predictor at best: {best_mean:.1f}ms")
    print(f"  Remaining for talker:   {audio_per_step_ms - best_mean:.1f}ms")

    # ── Integration guide ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("INTEGRATION GUIDE")
    print(f"{'='*70}")
    print("""
If CUDA IO Binding wins (Strategy D or E), integrate like this:

    # One-time setup (at session init):
    sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])

    # Pre-allocate GPU buffers
    embeds_gpu = ort.OrtValue.ortvalue_from_numpy(dummy_embeds, "cuda", 0)
    output_gpu = ort.OrtValue.ortvalue_from_shape_and_type([1, 2048], np.float32, "cuda", 0)
    step_gpus = [ort.OrtValue.ortvalue_from_numpy(np.array([i], dtype=np.int64), "cuda", 0) for i in range(15)]

    # Pre-bind 15 IO binding objects
    bindings = []
    for i in range(15):
        b = sess.io_binding()
        b.bind_ortvalue_input("inputs_embeds", embeds_gpu)
        b.bind_ortvalue_input("generation_step", step_gpus[i])
        b.bind_ortvalue_output("logits", output_gpu)
        bindings.append(b)

    # Per decode step (hot path):
    def predict_codes(hidden_state_np):
        # Update embeds on GPU (one copy)
        embeds_gpu.update_inplace(hidden_state_np)
        codes = []
        for b in bindings:
            sess.run_with_iobinding(b)
            logits = output_gpu.numpy()  # copy back only final result
            code = np.argmax(logits, axis=-1)
            codes.append(code)
        return codes

If CPU baseline wins, the bottleneck is purely compute-bound and
the optimization path is: (1) reduce model size, (2) use TensorRT,
or (3) batch the 15 steps into a single model call.
""")

    print(f"\n{'='*70}")
    print("RAW DATA")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
