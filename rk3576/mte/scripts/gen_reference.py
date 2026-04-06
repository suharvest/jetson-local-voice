#!/usr/bin/env python3
"""Generate reference input/output for the streaming Zipformer encoder using onnxruntime.

Creates known inputs (random features + zero states) and saves the encoder output
for validation of custom implementations.
"""
import onnxruntime as ort
import numpy as np
import os
import json

MODEL_PATH = "/tmp/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder-epoch-99-avg-1.onnx"
OUTPUT_DIR = "/tmp/jetson-voice-mte/rk3576/mte/reference"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_PATH}")
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    # Get all input metadata
    inputs_meta = sess.get_inputs()
    outputs_meta = sess.get_outputs()

    print(f"\n--- Model Inputs ({len(inputs_meta)}) ---")
    input_shapes = {}
    for inp in inputs_meta:
        print(f"  {inp.name}: shape={inp.shape} dtype={inp.type}")
        input_shapes[inp.name] = {
            "shape": [str(s) for s in inp.shape],
            "dtype": inp.type,
        }

    print(f"\n--- Model Outputs ({len(outputs_meta)}) ---")
    output_shapes = {}
    for out in outputs_meta:
        print(f"  {out.name}: shape={out.shape} dtype={out.type}")
        output_shapes[out.name] = {
            "shape": [str(s) for s in out.shape],
            "dtype": out.type,
        }

    # Build concrete input tensors
    # x: [N, 39, 80] -> use batch=1, T=39 (chunk size from model)
    # Note: task says T=71 for RKNN, but the ONNX model's native chunk is T=39
    batch_size = 1
    T = 39  # native chunk size from ONNX model input shape

    np.random.seed(42)  # reproducible

    feeds = {}
    dtype_map = {
        "tensor(float)": np.float32,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
    }

    for inp in inputs_meta:
        dtype = dtype_map.get(inp.type, np.float32)
        # Resolve shape: replace 'N' with batch_size, keep concrete dims
        concrete_shape = []
        for s in inp.shape:
            if isinstance(s, int):
                concrete_shape.append(s)
            elif isinstance(s, str):
                if s == "N":
                    concrete_shape.append(batch_size)
                else:
                    # Dynamic dim -- use the model's expected size
                    concrete_shape.append(T if "x" == inp.name else batch_size)
            else:
                concrete_shape.append(int(s))

        if inp.name == "x":
            # Input features: random in reasonable range
            data = np.random.randn(*concrete_shape).astype(dtype) * 0.1
        elif "cached_len" in inp.name:
            # Cache lengths: start at 0
            data = np.zeros(concrete_shape, dtype=dtype)
        else:
            # All state caches: initialize to zero
            data = np.zeros(concrete_shape, dtype=dtype)

        feeds[inp.name] = data
        print(f"  Created input: {inp.name} shape={data.shape} dtype={data.dtype}")

    # Save input features
    input_path = os.path.join(OUTPUT_DIR, "input_features.npy")
    np.save(input_path, feeds["x"])
    print(f"\nSaved input features to {input_path}")

    # Run inference
    print("\nRunning inference...")
    output_names = [out.name for out in outputs_meta]
    results = sess.run(output_names, feeds)

    # Save outputs
    for name, data in zip(output_names, results):
        safe_name = name.replace("/", "_")
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.npy")
        np.save(out_path, data)
        print(f"  Output: {name} shape={data.shape} dtype={data.dtype} "
              f"min={data.min():.6f} max={data.max():.6f} mean={data.mean():.6f}")

    # Save encoder_out specifically
    encoder_out_idx = output_names.index("encoder_out")
    encoder_out = results[encoder_out_idx]
    encoder_out_path = os.path.join(OUTPUT_DIR, "encoder_output.npy")
    np.save(encoder_out_path, encoder_out)
    print(f"\nSaved encoder output to {encoder_out_path}")
    print(f"  encoder_out shape: {encoder_out.shape}")
    print(f"  encoder_out range: [{encoder_out.min():.6f}, {encoder_out.max():.6f}]")

    # Save metadata
    meta = {
        "model": MODEL_PATH,
        "batch_size": batch_size,
        "chunk_size_T": T,
        "seed": 42,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "encoder_out_shape": list(encoder_out.shape),
        "encoder_out_stats": {
            "min": float(encoder_out.min()),
            "max": float(encoder_out.max()),
            "mean": float(encoder_out.mean()),
            "std": float(encoder_out.std()),
        },
        "all_state_inputs_zeroed": True,
        "notes": "Initial chunk inference with zero-initialized cache states"
    }
    meta_path = os.path.join(OUTPUT_DIR, "reference_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved reference metadata to {meta_path}")

    # List all saved files
    print(f"\n--- All reference files ---")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(path)
        print(f"  {path} ({size/1024:.1f} KB)")

if __name__ == "__main__":
    main()
