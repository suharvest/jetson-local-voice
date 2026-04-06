#!/usr/bin/env python3
"""
Per-layer debug comparison: MTE C engine vs Python reference intermediates.

1. Loads the debug .so (FP16 + MTE_DEBUG_DUMP)
2. Runs one streaming chunk (first chunk, T=16) using the SAME input as the Python reference
3. The C engine dumps intermediate tensors to /tmp/mte/debug_dump/
4. This script compares those dumps against the Python reference intermediates

Usage:
    python test_debug_compare.py [--fp16] [--int8]

    --fp16: use libzipformer_encoder_fp16_debug.so (default)
    --int8: use libzipformer_encoder_debug.so
"""

import ctypes
import os
import sys
import numpy as np

# --- Paths ---
ENGINE_DIR = "/tmp/mte/engine"
WEIGHT_DIR = "/tmp/mte/weights"
REF_DIR = "/tmp/mte/reference/streaming_intermediates"
DUMP_DIR = "/tmp/mte/debug_dump"
INPUT_NPY = "/tmp/mte/reference/streaming_intermediates/__Transpose_output_0.npy"

# --- Cosine similarity ---
def cos_sim(a, b):
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)

def load_ref(name):
    """Load reference tensor. Name uses / as separator, stored with __ encoding."""
    safe = name.replace("/", "__").replace(":", "_")
    path = os.path.join(REF_DIR, f"{safe}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None

def compare(label, c_tensor, ref_name=None, ref_tensor=None):
    """Compare C dump tensor against reference."""
    if ref_tensor is None and ref_name is not None:
        ref_tensor = load_ref(ref_name)
    if ref_tensor is None:
        print(f"  [{label}] NO REFERENCE")
        return None
    cos = cos_sim(c_tensor, ref_tensor)
    shape_ok = c_tensor.shape == ref_tensor.shape
    status = "OK" if cos > 0.999 else ("WARN" if cos > 0.99 else "FAIL")
    print(f"  [{status}] {label}: cos={cos:.6f}  C_shape={c_tensor.shape}  ref_shape={ref_tensor.shape}"
          f"  C_range=[{c_tensor.min():.4f},{c_tensor.max():.4f}]"
          f"  ref_range=[{ref_tensor.min():.4f},{ref_tensor.max():.4f}]")
    if not shape_ok:
        print(f"    *** SHAPE MISMATCH ***")
    return cos


def main():
    use_fp16 = "--int8" not in sys.argv

    if use_fp16:
        lib_name = "libzipformer_encoder_fp16_debug.so"
    else:
        lib_name = "libzipformer_encoder_debug.so"

    lib_path = os.path.join(ENGINE_DIR, lib_name)
    print(f"Using library: {lib_path}")
    print(f"Weight dir: {WEIGHT_DIR}")
    print(f"Reference dir: {REF_DIR}")
    print(f"Dump dir: {DUMP_DIR}")

    # Ensure dump dir exists
    os.makedirs(DUMP_DIR, exist_ok=True)

    # Load library
    lib = ctypes.CDLL(lib_path)

    # Define function signatures
    lib.zipformer_encoder_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.zipformer_encoder_init.restype = ctypes.c_void_p

    lib.zipformer_state_create.argtypes = [ctypes.c_void_p]
    lib.zipformer_state_create.restype = ctypes.c_void_p

    lib.zipformer_state_reset.argtypes = [ctypes.c_void_p]
    lib.zipformer_state_reset.restype = None

    lib.zipformer_state_destroy.argtypes = [ctypes.c_void_p]
    lib.zipformer_state_destroy.restype = None

    lib.zipformer_encoder_run_chunk.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
    ]
    lib.zipformer_encoder_run_chunk.restype = ctypes.c_int

    lib.zipformer_encoder_destroy.argtypes = [ctypes.c_void_p]
    lib.zipformer_encoder_destroy.restype = None

    lib.zipformer_encoder_set_debug_dump.argtypes = [ctypes.c_char_p]
    lib.zipformer_encoder_set_debug_dump.restype = None

    # Load input from reference
    if os.path.exists(INPUT_NPY):
        x = np.load(INPUT_NPY)  # [16, 1, 256] from ONNX
        print(f"\nLoaded reference input: {INPUT_NPY}")
        print(f"  Shape: {x.shape}, range=[{x.min():.6f}, {x.max():.6f}]")
        # Remove batch dim: [16, 1, 256] -> [16, 256]
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
    else:
        print(f"\nWARNING: No reference input at {INPUT_NPY}, using random data")
        np.random.seed(42)
        x = (np.random.randn(16, 256) * 0.1).astype(np.float32)

    x = np.ascontiguousarray(x, dtype=np.float32)
    T, D = x.shape
    print(f"  Input to engine: [{T}, {D}]")

    # Initialize engine
    print("\nInitializing engine...")
    enc = lib.zipformer_encoder_init(WEIGHT_DIR.encode(), 64)
    if not enc:
        print("ERROR: Engine init failed!")
        return

    # Set debug dump directory
    lib.zipformer_encoder_set_debug_dump(DUMP_DIR.encode())

    # Create state and run one chunk
    state = lib.zipformer_state_create(enc)
    out_buf = np.zeros((16, 512), dtype=np.float32)
    out_T = ctypes.c_int(0)

    print("\nRunning streaming chunk...")
    ret = lib.zipformer_encoder_run_chunk(
        enc, state,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        T, D,
        out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(out_T),
    )
    print(f"Return code: {ret}, out_T: {out_T.value}")

    # Compare C engine dumps with Python reference
    print(f"\n{'='*80}")
    print("PER-STEP COMPARISON: C engine vs Python reference")
    print(f"{'='*80}")

    # Map from C dump names to Python reference checkpoint names
    # The C dumps are: s0l0_ff1_out, s0l0_after_ff1, s0l0_whitened, etc.
    # The Python reference uses ONNX node names like /Add_output_0, etc.
    checkpoints = [
        # (c_dump_name, ref_onnx_name, description)
        ("s0l0_ff1_out",       "/feed_forward1/out_proj/Add_output_0", "FF1 output (before residual)"),
        ("s0l0_after_ff1",     "/Add_output_0",                       "After x += ff1(x)"),
        ("s0l0_whitened",      "/Mul_2_output_0",                     "Whitened (cumsum * reciprocal)"),
        ("s0l0_whiten_proj",   "/proj/MatMul_output_0",               "Whiten projection"),
        ("s0l0_x_attn_in",     "/Add_5_output_0",                     "x_attn_in = x + whiten_proj"),
        ("s0l0_attn_in_proj",  "/in_proj/Add_output_0",               "Attention in_proj output"),
        ("s0l0_Q",             "/Slice_output_0",                     "Q (query)"),
        ("s0l0_V_new",         "/Slice_2_output_0",                   "V (value)"),
        ("s0l0_attn_out_proj", "/Add_9_output_0",                     "Attention out_proj"),
        ("s0l0_after_attn",    "/Add_10_output_0",                    "After attention residual"),
        ("s0l0_after_conv1",   "/Add_11_output_0",                    "After conv1 residual"),
        ("s0l0_after_ff2",     "/Add_12_output_0",                    "After FF2 residual"),
        ("s0l0_after_v2",      "/Add_14_output_0",                    "After V2 path residual"),
        ("s0l0_after_conv2",   "/Add_15_output_0",                    "After conv2 residual"),
        ("s0l0_after_ff3",     "/Add_16_output_0",                    "After FF3 residual"),
        ("s0l0_after_bypass",  "/Add_17_output_0",                    "Layer output (after bypass)"),
    ]

    results = []
    for c_name, ref_name, desc in checkpoints:
        c_path = os.path.join(DUMP_DIR, f"{c_name}.npy")
        if os.path.exists(c_path):
            c_tensor = np.load(c_path)
            # Reference might have batch dim [T, 1, D], C has [T, D]
            ref_tensor = load_ref(ref_name)
            if ref_tensor is not None and ref_tensor.ndim == 3 and ref_tensor.shape[1] == 1:
                ref_tensor = ref_tensor.squeeze(1)
            cos = compare(desc, c_tensor, ref_tensor=ref_tensor)
            results.append((desc, cos))
        else:
            print(f"  [SKIP] {desc} — no C dump at {c_path}")
            results.append((desc, None))

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    first_fail = None
    for desc, cos in results:
        if cos is None:
            continue
        status = "PASS" if cos and cos > 0.99 else "FAIL"
        if status == "FAIL" and first_fail is None:
            first_fail = desc
        print(f"  {status}  cos={cos:.6f}  {desc}")

    if first_fail:
        print(f"\n*** FIRST DIVERGENCE at: {first_fail} ***")
        print("Fix this operation first, then re-test.")
    else:
        print("\nAll checkpoints PASS (cos > 0.99)")

    # Cleanup
    lib.zipformer_state_destroy(state)
    lib.zipformer_encoder_destroy(enc)


if __name__ == "__main__":
    main()
