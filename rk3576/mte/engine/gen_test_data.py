#!/usr/bin/env python3
"""Generate test input data and FP32 reference outputs for zipformer_layer_test.

For each projection, generates:
  - {name}_input.bin  : FP32 [M, K] random input
  - {name}_output.bin : FP32 [M, N] reference output (dequant matmul + bias)

The C program will load these and compare NPU results against them.
Run this on the host (or device) before running the C test.
"""

import numpy as np
import os
import sys

WEIGHT_DIR = "/tmp/jetson-voice-mte/rk3576/mte/weights/layer_0_w8a16"
REF_DIR    = "/tmp/jetson-voice-mte/rk3576/mte/reference"
M = 8   # must match TEST_M in the C program
SEED = 42

PROJECTIONS = [
    ("feed_forward1_in_proj",   256, 768),
    ("feed_forward1_out_proj",  768, 256),
    ("self_attn_in_proj",       256, 496),
    ("self_attn_out_proj",       96, 256),
    ("self_attn_out_proj2",      96, 256),
    ("feed_forward2_in_proj",   256, 768),
    ("feed_forward2_out_proj",  768, 256),
    ("feed_forward3_in_proj",   256, 768),
    ("feed_forward3_out_proj",  768, 256),
]

def load_weights(name, K, N):
    """Load INT8 weights, FP32 scales, and optional FP32 bias."""
    w_path = os.path.join(WEIGHT_DIR, f"{name}.int8.bin")
    s_path = os.path.join(WEIGHT_DIR, f"{name}.scales.bin")
    b_path = os.path.join(WEIGHT_DIR, f"{name}.bias.fp32.bin")

    w_int8 = np.fromfile(w_path, dtype=np.int8).reshape(K, N)
    scales = np.fromfile(s_path, dtype=np.float32)
    assert scales.shape[0] == N, f"scales shape mismatch: {scales.shape} vs N={N}"

    bias = None
    if os.path.exists(b_path):
        bias = np.fromfile(b_path, dtype=np.float32)
        assert bias.shape[0] == N, f"bias shape mismatch: {bias.shape} vs N={N}"

    return w_int8, scales, bias

def cpu_matmul_ref(A_f32, w_int8, scales, bias):
    """Compute reference: A @ dequant(W) + bias in FP32."""
    # Dequantize: W_f32[k,n] = w_int8[k,n] * scales[n]
    W_f32 = w_int8.astype(np.float32) * scales[None, :]
    out = A_f32 @ W_f32
    if bias is not None:
        out += bias[None, :]
    return out

def cpu_matmul_fp16_input_ref(A_f32, w_int8, scales, bias):
    """Reference matching what the NPU actually computes:
    FP16(input) * INT8(weight) -> FP32, then scale+bias on CPU.

    The NPU receives FP16 input, so we simulate the FP16 rounding.
    """
    A_fp16 = A_f32.astype(np.float16).astype(np.float32)
    # Raw matmul: FP16_input * INT8_weight (no scale yet)
    raw = A_fp16 @ w_int8.astype(np.float32)
    # Apply per-column scale + bias (this is done on CPU after NPU)
    out = raw * scales[None, :]
    if bias is not None:
        out += bias[None, :]
    return out

def main():
    weight_dir = WEIGHT_DIR
    ref_dir = REF_DIR

    if len(sys.argv) >= 2:
        weight_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        ref_dir = sys.argv[2]

    os.makedirs(ref_dir, exist_ok=True)

    rng = np.random.RandomState(SEED)
    print(f"Generating test data: M={M}, seed={SEED}")
    print(f"  Weights: {weight_dir}")
    print(f"  Output:  {ref_dir}")
    print()

    for name, K, N in PROJECTIONS:
        print(f"  {name} [{M},{K}] x [{K},{N}] ... ", end="", flush=True)

        w_int8, scales, bias = load_weights(name, K, N)

        # Generate input: small random values (typical after norm)
        input_f32 = (rng.randn(M, K) * 0.1).astype(np.float32)

        # Compute two references:
        # 1. "Ideal" FP32 reference (with FP32 input)
        ref_f32 = cpu_matmul_ref(input_f32, w_int8, scales, bias)

        # 2. "Expected NPU" reference (with FP16 input rounding)
        ref_fp16in = cpu_matmul_fp16_input_ref(input_f32, w_int8, scales, bias)

        # Save input
        in_path = os.path.join(ref_dir, f"{name}_input.bin")
        input_f32.tofile(in_path)

        # Save FP16-input reference (this is what the C program's CPU path computes)
        out_path = os.path.join(ref_dir, f"{name}_output.bin")
        ref_fp16in.tofile(out_path)

        # Report precision of FP16 rounding vs FP32
        diff = np.abs(ref_f32 - ref_fp16in)
        cos_sim = np.dot(ref_f32.ravel(), ref_fp16in.ravel()) / (
            np.linalg.norm(ref_f32) * np.linalg.norm(ref_fp16in) + 1e-30)

        print(f"ok (fp16_rounding: max_diff={diff.max():.6f}, cos={cos_sim:.8f})")

    print(f"\nDone. Generated {len(PROJECTIONS)} input/output pairs.")
    print("Transfer to device and run: ./zipformer_layer_test")

if __name__ == "__main__":
    main()
