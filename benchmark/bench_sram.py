#!/usr/bin/env python3
"""Benchmark RKNN SRAM environment variables on RK3576 NPU.

Tests vocoder (decoder_ctx25_int8) and code_predictor under different
SRAM-related env var configurations to measure performance impact.

Usage (on cat-remote):
    /home/cat/rknn-venv/bin/python bench_sram.py [--runs 20] [--log-level-run]
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("ERROR: rknnlite not found. Install rknn-toolkit-lite2.")
    sys.exit(1)

MODEL_DIR = "/home/cat/qwen3-tts-rknn"

# Models to test: (name, rknn_file, inputs_factory)
MODELS = {
    "decoder_ctx25_int8": {
        "file": "decoder_ctx25_int8.rknn",
        "make_inputs": lambda: [np.random.randn(1, 512, 50).astype(np.float32)],
    },
    "code_predictor": {
        "file": "code_predictor.rknn",
        "make_inputs": lambda: [np.random.randn(1, 2, 1024).astype(np.float32)],
    },
}

# Env var configurations to test
CONFIGS = [
    {
        "name": "baseline (no SRAM env)",
        "env": {},
    },
    {
        "name": "RKNN_INTERNAL_MEM_TYPE=sram",
        "env": {"RKNN_INTERNAL_MEM_TYPE": "sram"},
    },
    {
        "name": "SEPARATE_WEIGHT+WEIGHT_MEM_TYPE=sram#256",
        "env": {
            "RKNN_SEPARATE_WEIGHT_MEM": "1",
            "RKNN_WEIGHT_MEM_TYPE": "sram#256",
        },
    },
    {
        "name": "both SRAM env vars",
        "env": {
            "RKNN_INTERNAL_MEM_TYPE": "sram",
            "RKNN_SEPARATE_WEIGHT_MEM": "1",
            "RKNN_WEIGHT_MEM_TYPE": "sram#256",
        },
    },
]

WARMUP = 5


def benchmark_one(model_name, model_info, env_config, runs, verbose=False):
    """Benchmark a single model with specific env vars.

    Returns dict with results or error string.
    """
    model_path = os.path.join(MODEL_DIR, model_info["file"])
    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}"}

    # Set env vars
    env_backup = {}
    for k, v in env_config["env"].items():
        env_backup[k] = os.environ.get(k)
        os.environ[k] = v

    try:
        rknn = RKNNLite(verbose=verbose)
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            return {"error": f"load_rknn failed: {ret}"}

        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1)
        if ret != 0:
            rknn.release()
            return {"error": f"init_runtime failed: {ret}"}

        inputs = model_info["make_inputs"]()

        # Warmup
        for _ in range(WARMUP):
            rknn.inference(inputs=inputs)

        # Benchmark
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            rknn.inference(inputs=inputs)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)

        rknn.release()

        times = np.array(times)
        return {
            "avg_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "median_ms": float(np.median(times)),
            "all_ms": [round(t, 2) for t in times],
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Restore env vars
        for k, orig in env_backup.items():
            if orig is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig


def run_log_level_capture(model_name, model_info):
    """Run vocoder once with RKNN_LOG_LEVEL=3 and capture SramHit logs."""
    print("\n" + "=" * 70)
    print(f"RKNN_LOG_LEVEL=3 capture for {model_name}")
    print("=" * 70)

    # We run in a subprocess so the log level doesn't pollute main output
    script = f'''
import os, sys, numpy as np
os.environ["RKNN_LOG_LEVEL"] = "3"
from rknnlite.api import RKNNLite

rknn = RKNNLite(verbose=True)
rknn.load_rknn("{os.path.join(MODEL_DIR, model_info['file'])}")
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1)
inputs = [{model_info["make_inputs"].__doc__ or "np.random.randn(1, 512, 50).astype(np.float32)"}]
rknn.inference(inputs=inputs)
rknn.release()
'''
    # Build a proper self-contained script
    script_path = "/tmp/rknn_log_capture.py"
    with open(script_path, "w") as f:
        f.write(f"""#!/usr/bin/env python3
import os, sys, numpy as np
os.environ["RKNN_LOG_LEVEL"] = "3"
from rknnlite.api import RKNNLite

model_path = "{os.path.join(MODEL_DIR, model_info['file'])}"
rknn = RKNNLite(verbose=True)
ret = rknn.load_rknn(model_path)
ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1)
""")
        # Write input creation based on model
        if model_name == "decoder_ctx25_int8":
            f.write("inputs = [np.random.randn(1, 512, 50).astype(np.float32)]\n")
        else:
            f.write("inputs = [np.random.randn(1, 2, 1024).astype(np.float32)]\n")
        f.write("""
rknn.inference(inputs=inputs)
rknn.release()
""")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=60,
        )
        full_output = result.stdout + result.stderr
        lines = full_output.split("\n")

        # Filter for SRAM-related lines
        sram_lines = [l for l in lines if any(kw in l.lower() for kw in
                      ["sram", "mem_type", "internal_mem", "weight_mem", "nbuf"])]

        print(f"Total log lines: {len(lines)}")
        print(f"SRAM-related lines: {len(sram_lines)}")
        for line in sram_lines:
            print(f"  {line}")

        if not sram_lines:
            print("  (no SRAM-related lines found in log output)")
            # Show last 30 lines for context
            print("\n  Last 30 lines of output:")
            for line in lines[-30:]:
                if line.strip():
                    print(f"    {line}")

    except subprocess.TimeoutExpired:
        print("  TIMEOUT after 60s")
    except Exception as e:
        print(f"  ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark RKNN SRAM env vars")
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark runs per config")
    parser.add_argument("--log-level-run", action="store_true",
                        help="Also run with RKNN_LOG_LEVEL=3 to capture SRAM logs")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Models to test (default: all)")
    args = parser.parse_args()

    print("=" * 70)
    print("RKNN SRAM Environment Variable Benchmark")
    print("=" * 70)
    print(f"Runs per config: {args.runs}, Warmup: {WARMUP}")
    print(f"NPU core: dual-core (CORE_0_1)")

    # Check NPU info
    try:
        with open("/sys/kernel/debug/rknpu/mm") as f:
            print(f"\nNPU memory:\n{f.read().strip()}")
    except Exception:
        pass

    models_to_test = args.models or list(MODELS.keys())
    all_results = {}

    for model_name in models_to_test:
        if model_name not in MODELS:
            print(f"\nSKIP: unknown model {model_name}")
            continue

        model_info = MODELS[model_name]
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name} ({model_info['file']})")
        print(f"{'='*70}")

        model_results = []

        for config in CONFIGS:
            print(f"\n  Config: {config['name']}")
            if config["env"]:
                for k, v in config["env"].items():
                    print(f"    {k}={v}")
            else:
                print(f"    (default)")

            result = benchmark_one(model_name, model_info, config, args.runs)

            if "error" in result:
                print(f"    ERROR: {result['error']}")
            else:
                print(f"    avg={result['avg_ms']:.2f}ms  std={result['std_ms']:.2f}ms  "
                      f"min={result['min_ms']:.2f}ms  max={result['max_ms']:.2f}ms  "
                      f"median={result['median_ms']:.2f}ms")

            model_results.append({"config": config["name"], **result})

        all_results[model_name] = model_results

    # Summary table
    print("\n\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    for model_name, results in all_results.items():
        print(f"\n  {model_name}:")
        print(f"  {'Config':<42s} {'Avg(ms)':>8s} {'Std':>7s} {'Min':>8s} {'Max':>8s} {'Status'}")
        print(f"  {'-'*85}")

        baseline_avg = None
        for r in results:
            if "error" in r:
                print(f"  {r['config']:<42s} {'':>8s} {'':>7s} {'':>8s} {'':>8s} ERROR: {r['error']}")
                continue

            if baseline_avg is None:
                baseline_avg = r["avg_ms"]

            delta = ""
            if baseline_avg and r["config"] != "baseline (no SRAM env)":
                pct = (r["avg_ms"] - baseline_avg) / baseline_avg * 100
                delta = f" ({pct:+.1f}%)"

            print(f"  {r['config']:<42s} {r['avg_ms']:8.2f} {r['std_ms']:6.2f} "
                  f"{r['min_ms']:8.2f} {r['max_ms']:8.2f} OK{delta}")

    # Log level capture
    if args.log_level_run:
        # Only capture for vocoder
        run_log_level_capture("decoder_ctx25_int8", MODELS["decoder_ctx25_int8"])


if __name__ == "__main__":
    main()
