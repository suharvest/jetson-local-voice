#!/usr/bin/env python3
"""
Standalone CP profiling script — run inside the speech container.

Usage (inside container):
    python3 /tmp/profile_cp_trt.py

Requires qwen3_tts_engine .so on sys.path (already in /opt/speech/app/backends/).
"""

import sys
import time
import os

# Paths inside container
MODEL_BASE = os.environ.get("QWEN3_MODEL_BASE", "/opt/models/qwen3-tts")
SHERPA_DIR = os.path.join(MODEL_BASE, "onnx")
MODEL_DIR  = os.path.join(MODEL_BASE, "onnx")
ENGINES    = os.path.join(MODEL_BASE, "engines")
TALKER_ENGINE = os.path.join(ENGINES, "talker_decode_bf16.engine")
CP_ENGINE     = os.path.join(ENGINES, "cp_bf16.engine")

# .so is at /opt/speech/app/backends/
sys.path.insert(0, "/opt/speech/app/backends")

print(f"Loading qwen3_tts_engine...")
import qwen3_tts_engine as e

print(f"Creating pipeline from {MODEL_DIR}")
t0 = time.time()
p = e.Pipeline(MODEL_DIR, SHERPA_DIR, TALKER_ENGINE, CP_ENGINE)
print(f"Pipeline loaded in {time.time()-t0:.1f}s")

# Enable CUDA event profiling
p.enable_profiling(True)
print("Profiling enabled.\n")

# Warmup (1 run, not counted)
print("Warmup run...")
r = p.synthesize(text="hello", lang="english", token_ids=[9707])
print(f"  Warmup: {r.get('n_frames', 0)} frames, {r.get('per_step_ms', 0):.1f}ms/step")

# Reset stats after warmup
p.print_profiling_stats()  # prints + resets

# Measurement runs
print("\n=== MEASUREMENT RUNS ===")
for i in range(5):
    t_wall = time.time()
    r = p.synthesize(text="hello world, how are you today", lang="english",
                     token_ids=[9707, 1879, 11, 1246, 553, 481, 3432])
    dt = time.time() - t_wall
    n_frames = r.get("n_frames", 0)
    print(f"Run {i+1}: {n_frames} frames, wall={dt*1000:.0f}ms, "
          f"cp={r.get('per_step_ms', 0):.1f}ms/step, rtf={r.get('rtf', 0):.3f}")

print("\n=== PROFILING STATS (CUDA event breakdown) ===")
p.print_profiling_stats()

print("\nDone.")
