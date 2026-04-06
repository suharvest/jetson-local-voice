#!/usr/bin/env python3
"""Per-column INT8 quantization of ALL Zipformer FP16 matmul weights for W8A16.

Reads all .fp16.bin files from the weights directory tree,
applies per-column INT8 quantization, saves .int8.bin + .scales.bin alongside.
"""
import numpy as np
import os
import json
import glob

WEIGHTS_DIR = "/tmp/jetson-voice-mte/rk3576/mte/weights"
META_PATH = os.path.join(WEIGHTS_DIR, "meta.json")

# Known weight dimensions for all MatMul weights
WEIGHT_DIMS = {
    # Per-layer matmul weights (same dims for all 10 layers)
    "ff1_in": (256, 768),
    "ff1_out": (768, 256),
    "ff2_in": (256, 768),
    "ff2_out": (768, 256),
    "ff3_in": (256, 768),
    "ff3_out": (768, 256),
    "attn_in": (256, 496),
    "attn_out": (96, 256),
    "attn_out2": (96, 256),
    "attn_whiten": (256, 256),
    "attn_pos_bias": (256, 16),
    "attn_whiten2": (256, 96),
    # Encoder embed linear
    "linear": (2432, 256),
    # Encoder proj
    "encoder_proj": (256, 512),
}


def quantize_per_column_int8(W_fp16_path, K, N):
    """Per-column INT8 quantization."""
    raw = np.fromfile(W_fp16_path, dtype=np.float16)
    expected = K * N
    if raw.size != expected:
        print(f"  WARNING: {W_fp16_path}: expected {expected} elements, got {raw.size}")
        return None, None
    W = raw.reshape(K, N).astype(np.float32)

    col_amax = np.max(np.abs(W), axis=0)
    scales = col_amax / 127.0
    scales[scales == 0] = 1.0
    W_q = np.clip(np.round(W / scales[None, :]), -128, 127).astype(np.int8)

    return W_q, scales.astype(np.float32)


def main():
    # Find all FP16 weight files recursively
    fp16_files = sorted(glob.glob(os.path.join(WEIGHTS_DIR, "**", "*.fp16.bin"), recursive=True))

    if not fp16_files:
        print("No FP16 weight files found!")
        return

    print(f"Found {len(fp16_files)} FP16 weight files to quantize")
    print(f"{'='*70}")

    total_fp16_bytes = 0
    total_int8_bytes = 0
    total_scales_bytes = 0
    quantized_count = 0
    skipped = []
    all_quant_meta = {}

    for fp16_path in fp16_files:
        basename = os.path.basename(fp16_path)
        # Extract weight name: "ff1_in.fp16.bin" -> "ff1_in", "encoder_proj.fp16.bin" -> "encoder_proj"
        weight_name = basename.replace(".fp16.bin", "")
        dirpath = os.path.dirname(fp16_path)
        rel_dir = os.path.relpath(dirpath, WEIGHTS_DIR)

        # Look up dimensions
        if weight_name not in WEIGHT_DIMS:
            # Try to infer from file size
            file_size = os.path.getsize(fp16_path)
            skipped.append((fp16_path, f"unknown dims for '{weight_name}', {file_size} bytes"))
            continue

        K, N = WEIGHT_DIMS[weight_name]
        W_q, scales = quantize_per_column_int8(fp16_path, K, N)
        if W_q is None:
            skipped.append((fp16_path, "size mismatch"))
            continue

        # Save alongside the FP16 file
        int8_path = os.path.join(dirpath, f"{weight_name}.int8.bin")
        scales_path = os.path.join(dirpath, f"{weight_name}.scales.bin")
        W_q.tofile(int8_path)
        scales.tofile(scales_path)

        fp16_bytes = K * N * 2
        int8_bytes = W_q.nbytes
        sc_bytes = scales.nbytes
        total_fp16_bytes += fp16_bytes
        total_int8_bytes += int8_bytes
        total_scales_bytes += sc_bytes
        quantized_count += 1

        # Compute quantization error
        W_orig = np.fromfile(fp16_path, dtype=np.float16).reshape(K, N).astype(np.float32)
        W_recon = W_q.astype(np.float32) * scales[None, :]
        rmse = float(np.sqrt(np.mean((W_orig - W_recon) ** 2)))
        max_err = float(np.max(np.abs(W_orig - W_recon)))

        key = f"{rel_dir}/{weight_name}"
        all_quant_meta[key] = {
            "K": K, "N": N,
            "rmse": round(rmse, 7),
            "max_error": round(max_err, 7),
        }

        print(f"  {rel_dir}/{weight_name:20s}: [{K:4d},{N:4d}] "
              f"FP16={fp16_bytes/1024:6.1f}KB -> INT8={int8_bytes/1024:5.1f}KB+scales={sc_bytes/1024:4.1f}KB "
              f"RMSE={rmse:.6f}")

    # Summary
    total_quant_bytes = total_int8_bytes + total_scales_bytes
    print(f"\n{'='*70}")
    print(f"QUANTIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Quantized weights: {quantized_count}")
    if skipped:
        print(f"  Skipped: {len(skipped)}")
        for path, reason in skipped:
            print(f"    {path}: {reason}")
    print()
    print(f"  FP16 total:  {total_fp16_bytes:>10,} bytes ({total_fp16_bytes/1024:.1f} KB)")
    print(f"  INT8 total:  {total_int8_bytes:>10,} bytes ({total_int8_bytes/1024:.1f} KB)")
    print(f"  Scales total:{total_scales_bytes:>10,} bytes ({total_scales_bytes/1024:.1f} KB)")
    print(f"  INT8+scales: {total_quant_bytes:>10,} bytes ({total_quant_bytes/1024:.1f} KB)")
    if total_fp16_bytes > 0:
        ratio = total_quant_bytes / total_fp16_bytes
        print(f"  Compression: {ratio:.1%} of FP16 ({1/ratio:.2f}x smaller)")
        print(f"  Savings:     {(total_fp16_bytes - total_quant_bytes)/1024:.1f} KB")

    # Save quantization metadata
    quant_meta = {
        "quantization": "per_column_int8 (W8A16)",
        "quantized_count": quantized_count,
        "total_fp16_bytes": total_fp16_bytes,
        "total_int8_bytes": total_int8_bytes,
        "total_scales_bytes": total_scales_bytes,
        "total_quant_bytes": total_quant_bytes,
        "compression_ratio": f"{total_quant_bytes/total_fp16_bytes:.1%}" if total_fp16_bytes > 0 else "N/A",
        "weights": all_quant_meta,
    }
    quant_meta_path = os.path.join(WEIGHTS_DIR, "quant_meta.json")
    with open(quant_meta_path, "w") as f:
        json.dump(quant_meta, f, indent=2)
    print(f"\n  Quantization metadata saved to {quant_meta_path}")


if __name__ == "__main__":
    main()
