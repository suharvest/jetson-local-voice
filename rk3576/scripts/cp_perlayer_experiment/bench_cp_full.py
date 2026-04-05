#!/usr/bin/env python3
"""Benchmark full 5-layer transformer RKNN on RK3576."""
import os
import time
import numpy as np

LAYER_DIR = os.path.expanduser("~/qwen3-tts-export/cp_layers")
NUM_STEPS = 15
WARMUP = 10
BENCH_ITERS = 100


def main():
    from rknnlite.api import RKNNLite

    # Load full transformer model
    print("Loading full transformer RKNN model...")
    rknn_path = os.path.join(LAYER_DIR, "cp_transformer_full_w4a16.rknn")
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(rknn_path)
    assert ret == 0, f"Failed to load {rknn_path}"
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1)
    assert ret == 0, "Failed to init runtime"
    print("  Model loaded OK")

    # Load lm_heads and codebooks
    print("Loading lm_heads and codebooks...")
    lm_heads = []
    codebooks = []
    for j in range(NUM_STEPS):
        lm_w = np.load(os.path.join(LAYER_DIR, f"lm_head_{j}.npy")).astype(np.float32)
        cb_w = np.load(os.path.join(LAYER_DIR, f"codebook_{j}.npy")).astype(np.float32)
        lm_heads.append(lm_w)
        codebooks.append(cb_w)

    # === Benchmark 1: Single inference (5 layers) ===
    print("\n=== Benchmark 1: Single inference (all 5 layers) ===")
    hidden = np.random.randn(1, 2, 1024).astype(np.float32)

    for _ in range(WARMUP):
        rknn.inference(inputs=[hidden])

    times = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        out = rknn.inference(inputs=[hidden])
        times.append((time.perf_counter() - t0) * 1000)

    avg = np.mean(times)
    med = np.median(times)
    p99 = np.percentile(times, 99)
    mn = np.min(times)
    print(f"  avg={avg:.2f}ms, med={med:.2f}ms, min={mn:.2f}ms, p99={p99:.2f}ms")

    # === Benchmark 2: Full 15-step pipeline ===
    print("\n=== Benchmark 2: Full 15-step code_predictor pipeline ===")

    def run_15_steps():
        accumulated = np.random.randn(1, 1024).astype(np.float32)
        latest = np.random.randn(1, 1024).astype(np.float32)
        codes = []

        for step in range(NUM_STEPS):
            ctx = np.stack([accumulated, latest], axis=1)  # [1, 2, 1024]

            # Run transformer (all 5 layers + final norm)
            h = rknn.inference(inputs=[ctx])[0]

            # Extract last token
            last_h = h[:, -1, :]  # [1, 1024]

            # lm_head on CPU
            logits = last_h @ lm_heads[step].T
            code = np.argmax(logits, axis=-1)
            codes.append(code[0])

            embed = codebooks[step][code[0]]
            accumulated = last_h
            latest = embed.reshape(1, -1)

        return codes

    # Warmup
    for _ in range(5):
        run_15_steps()

    # Bench
    times_full = []
    for _ in range(30):
        t0 = time.perf_counter()
        codes = run_15_steps()
        times_full.append((time.perf_counter() - t0) * 1000)

    avg_full = np.mean(times_full)
    med_full = np.median(times_full)
    p99_full = np.percentile(times_full, 99)
    mn_full = np.min(times_full)

    print(f"  avg={avg_full:.1f}ms, med={med_full:.1f}ms, min={mn_full:.1f}ms, p99={p99_full:.1f}ms")
    print(f"  Per step: {avg_full/15:.2f}ms")

    # === Benchmark 3: NPU-only vs CPU overhead breakdown ===
    print("\n=== Benchmark 3: Breakdown ===")
    # NPU only (15 inference calls)
    npu_times = []
    for _ in range(30):
        ctx = np.random.randn(1, 2, 1024).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(NUM_STEPS):
            rknn.inference(inputs=[ctx])
        npu_times.append((time.perf_counter() - t0) * 1000)
    avg_npu = np.mean(npu_times)

    # CPU only (lm_head + lookup)
    cpu_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        for step in range(NUM_STEPS):
            test_h = np.random.randn(1, 1024).astype(np.float32)
            logits = test_h @ lm_heads[step].T
            code = np.argmax(logits, axis=-1)
            embed = codebooks[step][code[0]]
        cpu_times.append((time.perf_counter() - t0) * 1000)
    avg_cpu = np.mean(cpu_times)

    print(f"  NPU (15x transformer): {avg_npu:.1f}ms ({avg_npu/15:.2f}ms/call)")
    print(f"  CPU (15x lm_head+lookup): {avg_cpu:.1f}ms ({avg_cpu/15:.2f}ms/call)")
    print(f"  Overhead (data transfer etc): {avg_full - avg_npu - avg_cpu:.1f}ms")

    # === Summary ===
    print("\n=== SUMMARY ===")
    print(f"Full transformer (5 layers, W4A16 + exSDP): {avg:.2f}ms per call")
    print(f"15-step pipeline total: {avg_full:.1f}ms")
    print(f"  NPU inference: {avg_npu:.1f}ms")
    print(f"  CPU lm_head:   {avg_cpu:.1f}ms")
    rknn_size = os.path.getsize(rknn_path) / 1e6
    print(f"Model size: {rknn_size:.1f} MB")
    print(f"\nComparison:")
    print(f"  Per-layer RKNN (5 separate models):  205.6ms (75 calls)")
    print(f"  Full transformer RKNN (1 model):     {avg_full:.1f}ms (15 calls)")
    print(f"  MatMul API approach:                 69ms")
    print(f"  RKNN sequential (old):               117ms")
    print(f"  RKLLM:                               120ms")

    rknn.release()


if __name__ == "__main__":
    main()
