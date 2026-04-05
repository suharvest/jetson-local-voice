#!/usr/bin/env python3
"""Test and benchmark the code predictor engine on RK3576.

Usage:
    python3 test_engine.py [--weight-dir /home/cat/cp_weights] [--cores 2]
"""

import argparse
import time
import numpy as np
import sys
import os

# Add engine dir to path for the wrapper
sys.path.insert(0, os.path.dirname(__file__))
from cp_engine_wrapper import CodePredictorEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-dir", default="/home/cat/cp_weights")
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--lib", default=None, help="Path to libcp_engine.so")
    parser.add_argument("--reps", type=int, default=20, help="Benchmark repetitions")
    args = parser.parse_args()

    print("=" * 60)
    print("Code Predictor Engine Test")
    print("=" * 60)

    # Initialize
    t0 = time.time()
    engine = CodePredictorEngine(
        args.weight_dir,
        num_npu_cores=args.cores,
        lib_path=args.lib,
    )
    init_time = time.time() - t0
    print(f"Init time: {init_time:.1f}s")

    # Random inputs (deterministic seed)
    np.random.seed(42)
    last_hidden = np.random.randn(1024).astype(np.float32) * 0.1
    primary_embed = np.random.randn(1024).astype(np.float32) * 0.1

    # Single run test
    print("\n--- Single run ---")
    t0 = time.time()
    codes, codec_sum = engine.run(last_hidden, primary_embed)
    single_ms = (time.time() - t0) * 1000
    print(f"Codes: {codes}")
    print(f"Codec sum range: [{codec_sum.min():.4f}, {codec_sum.max():.4f}]")
    print(f"Codec sum norm: {np.linalg.norm(codec_sum):.4f}")
    print(f"Time: {single_ms:.1f} ms")

    # Benchmark
    print(f"\n--- Benchmark ({args.reps} reps) ---")
    times = []
    for i in range(args.reps):
        t0 = time.time()
        codes, codec_sum = engine.run(last_hidden, primary_embed)
        elapsed = (time.time() - t0) * 1000
        times.append(elapsed)

    times = np.array(times)
    print(f"Mean:   {times.mean():.1f} ms")
    print(f"Std:    {times.std():.1f} ms")
    print(f"Min:    {times.min():.1f} ms")
    print(f"Max:    {times.max():.1f} ms")
    print(f"Median: {np.median(times):.1f} ms")
    print(f"P95:    {np.percentile(times, 95):.1f} ms")

    # Target comparison
    target = 69.0
    print(f"\n--- vs Target ---")
    print(f"Target:  {target:.0f} ms (matmul-only benchmark)")
    print(f"Actual:  {times.mean():.1f} ms")
    ratio = times.mean() / target
    if ratio <= 1.1:
        print(f"Status:  ON TARGET ({ratio:.2f}x)")
    else:
        print(f"Status:  {ratio:.2f}x slower than matmul-only target")
        print(f"         (expected due to CPU ops + FP16<->FP32 conversion)")

    # Determinism check
    print("\n--- Determinism check ---")
    codes1, sum1 = engine.run(last_hidden, primary_embed)
    codes2, sum2 = engine.run(last_hidden, primary_embed)
    codes_match = np.array_equal(codes1, codes2)
    sum_diff = np.max(np.abs(sum1 - sum2))
    print(f"Codes match: {codes_match}")
    print(f"Sum max diff: {sum_diff:.6f}")

    engine.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
