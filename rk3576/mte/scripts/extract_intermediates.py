#!/usr/bin/env python3
"""
Extract intermediate tensors from the ONNX streaming Zipformer encoder.

Uses onnxruntime with added output nodes to capture every intermediate tensor
for one streaming chunk. This is the ground truth that our Python reference
must match exactly.
"""
import onnxruntime as ort
import onnx
import onnx.numpy_helper
import numpy as np
import os
import json
import re

MODEL_PATH = "/tmp/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder-epoch-99-avg-1.onnx"
OUTPUT_DIR = "/tmp/jetson-voice-mte/rk3576/mte/reference/streaming_intermediates"


def add_all_intermediate_outputs(model_path):
    """Add all intermediate tensors as outputs so we can capture them."""
    model = onnx.load(model_path)
    graph = model.graph

    # Get existing output names
    existing_outputs = {o.name for o in graph.output}

    # Collect all intermediate tensor names (node outputs)
    intermediate_names = set()
    for node in graph.node:
        for output in node.output:
            if output and output not in existing_outputs:
                intermediate_names.add(output)

    # We can't add ALL intermediates (too many). Focus on key ones.
    # Strategy: add outputs for all named operations (MatMul, Add with bias, etc.)
    key_outputs = []
    for node in graph.node:
        # Keep all MatMul outputs, all named outputs, and key ops
        if node.op_type in ("MatMul", "Softmax", "Conv", "Split"):
            for out in node.output:
                if out and out not in existing_outputs:
                    key_outputs.append(out)
        # Keep Add outputs (residual connections)
        elif node.op_type == "Add":
            for out in node.output:
                if out and out not in existing_outputs:
                    key_outputs.append(out)
        # Keep activation outputs
        elif node.op_type in ("Sigmoid", "Mul", "Sub") and any(
            "activation" in o or "norm" in o or "swoosh" in o.lower()
            for o in node.output if o
        ):
            for out in node.output:
                if out and out not in existing_outputs:
                    key_outputs.append(out)
        # Keep CumSum, Reciprocal, ReduceMean (BasicNorm ops)
        elif node.op_type in ("CumSum", "Reciprocal", "ReduceMean", "Pow"):
            for out in node.output:
                if out and out not in existing_outputs:
                    key_outputs.append(out)
        # Keep Reshape, Transpose for attention shape tracking
        elif node.op_type in ("Reshape", "Transpose", "Slice", "Concat"):
            # Only keep the first few per layer to avoid explosion
            for out in node.output:
                if out and out not in existing_outputs:
                    key_outputs.append(out)

    # Actually, let's just add ALL intermediate outputs - onnxruntime can handle it
    all_outputs = []
    for node in graph.node:
        for out in node.output:
            if out and out not in existing_outputs:
                all_outputs.append(out)

    print(f"Adding {len(all_outputs)} intermediate outputs (from {len(graph.node)} nodes)")

    # Build shape inference info
    # We need to infer shapes to create proper ValueInfoProto
    # Easiest: use onnx shape inference
    try:
        model = onnx.shape_inference.infer_shapes(model)
        graph = model.graph

        # Build value_info map
        vi_map = {}
        for vi in graph.value_info:
            vi_map[vi.name] = vi

        added = 0
        for name in all_outputs:
            if name in vi_map:
                graph.output.append(vi_map[name])
                added += 1
            else:
                # Create a generic output (no shape info)
                from onnx import TensorProto
                new_out = onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
                graph.output.append(new_out)
                added += 1

        print(f"Added {added} intermediate outputs")
    except Exception as e:
        print(f"Shape inference failed: {e}, adding without shapes")
        for name in all_outputs:
            from onnx import TensorProto
            new_out = onnx.helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            graph.output.append(new_out)

    # Save modified model
    modified_path = "/tmp/encoder_with_intermediates.onnx"
    onnx.save(model, modified_path)
    print(f"Saved modified model to {modified_path}")
    return modified_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Add intermediate outputs
    modified_path = add_all_intermediate_outputs(MODEL_PATH)

    # Load with onnxruntime
    print(f"\nLoading modified model...")
    sess = ort.InferenceSession(modified_path, providers=["CPUExecutionProvider"])

    # Build input feeds (first chunk with zero states)
    inputs_meta = sess.get_inputs()
    batch_size = 1

    np.random.seed(42)

    feeds = {}
    dtype_map = {
        "tensor(float)": np.float32,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
    }

    for inp in inputs_meta:
        dtype = dtype_map.get(inp.type, np.float32)
        concrete_shape = []
        for s in inp.shape:
            if isinstance(s, int):
                concrete_shape.append(s)
            elif isinstance(s, str):
                if s == "N":
                    concrete_shape.append(batch_size)
                else:
                    concrete_shape.append(39 if inp.name == "x" else batch_size)
            else:
                concrete_shape.append(int(s))

        if inp.name == "x":
            data = np.random.randn(*concrete_shape).astype(dtype) * 0.1
        elif "cached_len" in inp.name:
            data = np.zeros(concrete_shape, dtype=dtype)
        else:
            data = np.zeros(concrete_shape, dtype=dtype)

        feeds[inp.name] = data

    # Run inference
    print("Running inference with intermediate captures...")
    output_names = [out.name for out in sess.get_outputs()]
    results = sess.run(output_names, feeds)

    # Save all results
    print(f"\nSaving {len(results)} tensors...")
    tensor_info = {}
    for name, data in zip(output_names, results):
        if data is None:
            continue
        safe_name = name.replace("/", "__").replace(":", "_")
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.npy")
        np.save(out_path, data)
        info = {
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "min": float(np.min(data)) if data.size > 0 else 0,
            "max": float(np.max(data)) if data.size > 0 else 0,
            "mean": float(np.mean(data)) if data.size > 0 else 0,
        }
        tensor_info[name] = info

    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, "tensor_info.json")
    with open(meta_path, "w") as f:
        json.dump(tensor_info, f, indent=2)

    # Print key tensors for layer 0
    print("\n=== Key Tensors for Stack 0, Layer 0 ===")
    key_patterns = [
        "Transpose_output",  # embed output / layer input
        "feed_forward1",
        "Add_output_0",  # post ff1 residual
        "CumSum",
        "Reciprocal",
        "Mul_2_output",  # whitened
        "proj/MatMul",  # whiten proj
        "Add_5",  # whiten residual
        "in_proj/",  # attention in_proj
        "linear_pos",
        "Slice_output_0",  # Q
        "Slice_1_output_0",  # K
        "Slice_2_output_0",  # V
        "Softmax",
        "out_proj/Add",  # attention out
        "out_proj2/Add",
        "pointwise_conv1/Conv",
        "depthwise_conv/Conv",
        "pointwise_conv2/Conv",
        "feed_forward2",
        "feed_forward3",
        "norm_final",
        "bypass_scale",
        "Add_17",  # layer output
    ]

    for pattern in key_patterns:
        for name, info in tensor_info.items():
            if pattern in name:
                print(f"  {name}: shape={info['shape']} range=[{info['min']:.6f}, {info['max']:.6f}] mean={info['mean']:.6f}")
                break

    # Print encoder output
    print("\n=== Encoder Output ===")
    enc_out = tensor_info.get("encoder_out", {})
    print(f"  encoder_out: shape={enc_out.get('shape')} range=[{enc_out.get('min', 0):.6f}, {enc_out.get('max', 0):.6f}]")

    print(f"\nTotal tensors saved: {len(tensor_info)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
