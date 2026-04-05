#!/usr/bin/env python3
"""Convert per-layer ONNX to RKNN with W4A16 quantization."""
import os
import sys
import numpy as np
import onnx

LAYER_DIR = os.path.expanduser("~/qwen3-tts-export/cp_layers")


def internalize_onnx(onnx_path, out_path):
    """Load ONNX with external data and save as single self-contained file."""
    model = onnx.load(onnx_path)
    onnx.save(model, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Internalized: {out_path} ({size_mb:.1f} MB)")
    return out_path


def make_dataset_file(layer_idx, num_samples=10):
    """Create a calibration dataset file for RKNN (list of .npy paths)."""
    dataset_dir = os.path.join(LAYER_DIR, f"calib_layer{layer_idx}")
    os.makedirs(dataset_dir, exist_ok=True)

    dataset_txt = os.path.join(LAYER_DIR, f"dataset_layer{layer_idx}.txt")
    lines = []
    for s in range(num_samples):
        npy_path = os.path.join(dataset_dir, f"sample_{s}.npy")
        data = np.random.randn(1, 2, 1024).astype(np.float32)
        np.save(npy_path, data)
        lines.append(npy_path)

    with open(dataset_txt, "w") as f:
        f.write("\n".join(lines))

    print(f"  Dataset: {dataset_txt} ({num_samples} samples)")
    return dataset_txt


def convert_layer(layer_idx):
    """Convert one layer ONNX -> RKNN W4A16."""
    from rknn.api import RKNN

    onnx_path = os.path.join(LAYER_DIR, f"cp_layer{layer_idx}.onnx")
    onnx_int_path = os.path.join(LAYER_DIR, f"cp_layer{layer_idx}_int.onnx")
    rknn_path = os.path.join(LAYER_DIR, f"cp_layer{layer_idx}_w4a16.rknn")

    print(f"\n=== Converting layer {layer_idx} ===")

    # Internalize external data
    internalize_onnx(onnx_path, onnx_int_path)

    # Create calibration dataset
    dataset_txt = make_dataset_file(layer_idx)

    rknn = RKNN(verbose=True)

    # Configure for RK3576, W4A16
    ret = rknn.config(
        target_platform="rk3576",
        quantized_dtype="w4a16",
        optimization_level=3,
    )
    if ret != 0:
        print(f"  ERROR: config failed ({ret})")
        return False

    # Load ONNX
    ret = rknn.load_onnx(model=onnx_int_path)
    if ret != 0:
        print(f"  ERROR: load_onnx failed ({ret})")
        return False

    # Build with W4A16 quantization + dataset
    ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    if ret != 0:
        print(f"  ERROR: build failed ({ret})")
        return False

    # Export
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"  ERROR: export failed ({ret})")
        return False

    size_mb = os.path.getsize(rknn_path) / 1e6
    print(f"  RKNN saved: {rknn_path} ({size_mb:.1f} MB)")

    rknn.release()

    # Clean up
    os.remove(onnx_int_path)

    return True


def main():
    results = {}
    for i in range(5):
        ok = convert_layer(i)
        results[i] = ok
        if not ok:
            print(f"Layer {i} FAILED, stopping")
            break

    print("\n=== Summary ===")
    for i, ok in results.items():
        status = "OK" if ok else "FAILED"
        rknn_path = os.path.join(LAYER_DIR, f"cp_layer{i}_w4a16.rknn")
        if ok and os.path.exists(rknn_path):
            size_mb = os.path.getsize(rknn_path) / 1e6
            print(f"  Layer {i}: {status} ({size_mb:.1f} MB)")
        else:
            print(f"  Layer {i}: {status}")


if __name__ == "__main__":
    main()
