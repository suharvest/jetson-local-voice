#!/usr/bin/env python3
"""Per-column INT8 quantization of Zipformer layer 0 FP16 weights for W8A16 matmul.

Reads FP16 .bin files from weights/layer_0/,
saves .int8.bin + .scales.bin to weights/layer_0_w8a16/.
"""
import numpy as np
import os
import json
import glob

INPUT_DIR = "/tmp/jetson-voice-mte/rk3576/mte/weights/layer_0"
OUTPUT_DIR = "/tmp/jetson-voice-mte/rk3576/mte/weights/layer_0_w8a16"
META_PATH = "/tmp/jetson-voice-mte/rk3576/mte/weights/meta.json"

def quantize_per_column_int8(W_fp16_path, K, N):
    """Per-column INT8 quantization: same pattern as quantize_w8a16.py."""
    raw = np.fromfile(W_fp16_path, dtype=np.float16)
    assert raw.size == K * N, f"Expected {K*N} elements, got {raw.size} from {W_fp16_path}"
    W = raw.reshape(K, N).astype(np.float32)

    col_amax = np.max(np.abs(W), axis=0)
    scales = col_amax / 127.0
    scales[scales == 0] = 1.0
    W_q = np.clip(np.round(W / scales[None, :]), -128, 127).astype(np.int8)

    return W_q, scales.astype(np.float32)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all FP16 weight files
    fp16_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.weight.fp16.bin")))

    if not fp16_files:
        print("No FP16 weight files found!")
        return

    # Weight dimensions from the analysis
    weight_dims = {
        "self_attn_in_proj": (256, 496),
        "self_attn_out_proj": (96, 256),
        "self_attn_out_proj2": (96, 256),
        "feed_forward1_in_proj": (256, 768),
        "feed_forward1_out_proj": (768, 256),
        "feed_forward2_in_proj": (256, 768),
        "feed_forward2_out_proj": (768, 256),
        "feed_forward3_in_proj": (256, 768),
        "feed_forward3_out_proj": (768, 256),
        # Unmapped weights
        "unmapped_4844": (256, 256),
        "unmapped_4846": (256, 16),
        "unmapped_4868": (256, 96),
    }

    total_orig = 0
    total_int8 = 0
    quantized = {}

    for fp16_path in fp16_files:
        basename = os.path.basename(fp16_path)
        # Extract weight name: e.g. "self_attn_in_proj.weight.fp16.bin" -> "self_attn_in_proj"
        weight_name = basename.replace(".weight.fp16.bin", "")

        if weight_name not in weight_dims:
            print(f"  SKIP {weight_name}: unknown dimensions")
            continue

        K, N = weight_dims[weight_name]
        W_q, scales = quantize_per_column_int8(fp16_path, K, N)

        # Save quantized weight and scales
        int8_path = os.path.join(OUTPUT_DIR, f"{weight_name}.int8.bin")
        scales_path = os.path.join(OUTPUT_DIR, f"{weight_name}.scales.bin")
        W_q.tofile(int8_path)
        scales.tofile(scales_path)

        orig_bytes = K * N * 2  # FP16
        int8_bytes = W_q.nbytes + scales.nbytes
        total_orig += orig_bytes
        total_int8 += int8_bytes

        # Compute quantization error
        W_orig = np.fromfile(fp16_path, dtype=np.float16).reshape(K, N).astype(np.float32)
        W_recon = W_q.astype(np.float32) * scales[None, :]
        rmse = np.sqrt(np.mean((W_orig - W_recon) ** 2))
        max_err = np.max(np.abs(W_orig - W_recon))

        print(f"  {weight_name:30s}: [{K},{N}] -> {int8_bytes/1024:.0f}KB ({int8_bytes/orig_bytes:.1%}) "
              f"RMSE={rmse:.6f} MaxErr={max_err:.6f}")

        quantized[weight_name] = {
            "K": K, "N": N,
            "int8_file": f"{weight_name}.int8.bin",
            "scales_file": f"{weight_name}.scales.bin",
            "rmse": float(rmse),
            "max_error": float(max_err),
            "compression": f"{int8_bytes/orig_bytes:.1%}",
        }

    # Also copy bias files as-is (FP32)
    bias_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.bias.fp32.bin")))
    for bias_path in bias_files:
        basename = os.path.basename(bias_path)
        dst = os.path.join(OUTPUT_DIR, basename)
        data = np.fromfile(bias_path, dtype=np.float32)
        data.tofile(dst)
        print(f"  Copied bias: {basename} ({data.size} elements)")

    # Copy conv weights as-is (FP32)
    conv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.fp32.bin")))
    for conv_path in conv_files:
        basename = os.path.basename(conv_path)
        if basename in [os.path.basename(b) for b in bias_files]:
            continue  # already copied
        dst = os.path.join(OUTPUT_DIR, basename)
        data = np.fromfile(conv_path, dtype=np.uint8)
        data.tofile(dst)
        print(f"  Copied: {basename} ({data.size} bytes)")

    # Save quantization metadata
    quant_meta = {
        "quantization": "per_column_int8",
        "source": INPUT_DIR,
        "weights": quantized,
        "total_fp16_bytes": total_orig,
        "total_int8_bytes": total_int8,
        "compression_ratio": f"{total_int8/total_orig:.1%}" if total_orig > 0 else "N/A",
    }
    meta_path = os.path.join(OUTPUT_DIR, "quant_meta.json")
    with open(meta_path, "w") as f:
        json.dump(quant_meta, f, indent=2)

    print(f"\n  FP16 total: {total_orig/1024:.0f} KB -> INT8 total: {total_int8/1024:.0f} KB "
          f"({total_int8/total_orig:.1%})")
    print(f"\nSaved to {OUTPUT_DIR}")

    # List all output files
    print(f"\n--- All quantized files ---")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(path)
        print(f"  {path} ({size/1024:.1f} KB)")

if __name__ == "__main__":
    main()
