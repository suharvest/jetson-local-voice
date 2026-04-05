#!/usr/bin/env python3
"""Benchmark per-layer RKNN code_predictor on RK3576."""
import os
import sys
import time
import numpy as np

LAYER_DIR = os.path.expanduser("~/qwen3-tts-export/cp_layers")
NUM_LAYERS = 5
NUM_STEPS = 15
WARMUP = 10
BENCH_ITERS = 100


def main():
    from rknnlite.api import RKNNLite

    # Load 5 layer models
    print("Loading RKNN models...")
    layers = []
    for i in range(NUM_LAYERS):
        rknn_path = os.path.join(LAYER_DIR, f"cp_layer{i}_w4a16.rknn")
        rknn = RKNNLite(verbose=False)
        ret = rknn.load_rknn(rknn_path)
        if ret != 0:
            print(f"ERROR: Failed to load {rknn_path}")
            return
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1)
        if ret != 0:
            print(f"ERROR: Failed to init runtime for layer {i}")
            return
        layers.append(rknn)
        print(f"  Layer {i}: loaded OK")

    # Load lm_heads and codebooks
    print("Loading lm_heads and codebooks...")
    lm_heads = []
    codebooks = []
    for j in range(NUM_STEPS):
        lm_w = np.load(os.path.join(LAYER_DIR, f"lm_head_{j}.npy"))  # [2048, 1024]
        cb_w = np.load(os.path.join(LAYER_DIR, f"codebook_{j}.npy"))  # [2048, 1024]
        lm_heads.append(lm_w.astype(np.float32))
        codebooks.append(cb_w.astype(np.float32))
    print(f"  Loaded {NUM_STEPS} lm_heads and codebooks")

    # === Benchmark 1: Single layer inference ===
    print("\n=== Benchmark 1: Single layer latency ===")
    hidden = np.random.randn(1, 2, 1024).astype(np.float32)

    for i in range(NUM_LAYERS):
        # Warmup
        for _ in range(WARMUP):
            layers[i].inference(inputs=[hidden])

        # Bench
        times = []
        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter()
            out = layers[i].inference(inputs=[hidden])
            times.append((time.perf_counter() - t0) * 1000)

        avg = np.mean(times)
        med = np.median(times)
        p99 = np.percentile(times, 99)
        print(f"  Layer {i}: avg={avg:.2f}ms, med={med:.2f}ms, p99={p99:.2f}ms")

    # === Benchmark 2: 5-layer forward pass ===
    print("\n=== Benchmark 2: 5-layer forward pass ===")

    for _ in range(WARMUP):
        h = hidden.copy()
        for i in range(NUM_LAYERS):
            h = layers[i].inference(inputs=[h])[0]

    times_5layer = []
    for _ in range(BENCH_ITERS):
        h = hidden.copy()
        t0 = time.perf_counter()
        for i in range(NUM_LAYERS):
            h = layers[i].inference(inputs=[h])[0]
        times_5layer.append((time.perf_counter() - t0) * 1000)

    avg5 = np.mean(times_5layer)
    med5 = np.median(times_5layer)
    print(f"  5 layers: avg={avg5:.2f}ms, med={med5:.2f}ms")

    # === Benchmark 3: Full 15-step pipeline ===
    print("\n=== Benchmark 3: Full 15-step code_predictor pipeline ===")

    # Simulate the full pipeline
    def run_15_steps():
        accumulated = np.random.randn(1, 1024).astype(np.float32)
        latest = np.random.randn(1, 1024).astype(np.float32)
        codes = []

        for step in range(NUM_STEPS):
            # Build 2-token context [1, 2, 1024]
            ctx = np.stack([accumulated, latest], axis=1)  # [1, 2, 1024]

            # Run 5 layers
            h = ctx
            for i in range(NUM_LAYERS):
                h = layers[i].inference(inputs=[h])[0]

            # Extract last token hidden state
            last_h = h[:, -1, :]  # [1, 1024]

            # lm_head on CPU
            logits = last_h @ lm_heads[step].T  # [1, 2048]
            code = np.argmax(logits, axis=-1)  # [1]
            codes.append(code[0])

            # Codebook lookup
            embed = codebooks[step][code[0]]  # [1024]

            # Update state
            accumulated = last_h
            latest = embed.reshape(1, -1)

        return codes

    # Warmup
    for _ in range(3):
        run_15_steps()

    # Bench
    times_full = []
    for _ in range(20):
        t0 = time.perf_counter()
        codes = run_15_steps()
        times_full.append((time.perf_counter() - t0) * 1000)

    avg_full = np.mean(times_full)
    med_full = np.median(times_full)
    p99_full = np.percentile(times_full, 99)

    print(f"  15 steps (5 layers each): avg={avg_full:.1f}ms, med={med_full:.1f}ms, p99={p99_full:.1f}ms")
    print(f"  Per step: {avg_full/15:.2f}ms")
    print(f"  Per layer call: {avg_full/75:.2f}ms")
    print(f"  Last codes: {codes}")

    # === Benchmark 4: CPU-only lm_head time ===
    print("\n=== Benchmark 4: CPU lm_head overhead ===")
    test_h = np.random.randn(1, 1024).astype(np.float32)
    times_lm = []
    for _ in range(1000):
        t0 = time.perf_counter()
        logits = test_h @ lm_heads[0].T
        code = np.argmax(logits, axis=-1)
        embed = codebooks[0][code[0]]
        times_lm.append((time.perf_counter() - t0) * 1000)
    avg_lm = np.mean(times_lm)
    print(f"  lm_head + argmax + lookup: {avg_lm:.3f}ms per step")
    print(f"  15 steps total: {avg_lm*15:.2f}ms")

    # === Summary ===
    print("\n=== SUMMARY ===")
    print(f"Single layer (W4A16 + exSDP): ~{np.mean(times_5layer)/5:.2f}ms")
    print(f"5 layers (one transformer pass): {avg5:.1f}ms")
    print(f"15 steps x 5 layers (code_predictor): {avg_full:.1f}ms")
    print(f"CPU lm_head overhead (15 steps): {avg_lm*15:.1f}ms")
    print(f"Total estimated: {avg_full:.1f}ms (NPU) + {avg_lm*15:.1f}ms (CPU) = {avg_full + avg_lm*15:.1f}ms")
    print(f"\nModel size: {NUM_LAYERS * 9.2:.1f} MB (5 x 9.2 MB RKNN)")
    print(f"\nComparison targets:")
    print(f"  MatMul API approach:    69ms")
    print(f"  RKNN sequential:       117ms")
    print(f"  RKLLM:                 120ms")

    # Cleanup
    for rknn in layers:
        rknn.release()


if __name__ == "__main__":
    main()
