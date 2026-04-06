#!/usr/bin/env python3
"""Extract first encoder stack layer 0 weights from the Zipformer ONNX encoder.

Weight matrices (MatMul): transposed from ONNX [K, N] to [K, N] (already in row-major
K x N for rknn_matmul B matrix), saved as FP16.
Bias/norm/conv weights: saved as FP32.

Also traces the graph to map anonymous onnx::MatMul_XXXX tensors to their named biases.
"""
import onnx
import onnx.numpy_helper
import numpy as np
import os
import json
import re
from collections import defaultdict

MODEL_PATH = "/tmp/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder-epoch-99-avg-1.onnx"
OUTPUT_DIR = "/tmp/jetson-voice-mte/rk3576/mte/weights/layer_0"
META_PATH = "/tmp/jetson-voice-mte/rk3576/mte/weights/meta.json"

def main():
    print(f"Loading model with external data: {MODEL_PATH}")
    model = onnx.load(MODEL_PATH)  # load with data this time
    graph = model.graph

    # Build initializer name -> numpy array map
    init_data = {}
    for init in graph.initializer:
        init_data[init.name] = onnx.numpy_helper.to_array(init)

    # Build MatMul weight -> bias name mapping (same logic as analyze)
    matmul_to_bias = {}
    for node in graph.node:
        if node.op_type == "MatMul":
            weight_name = None
            for inp in node.input:
                if inp in init_data and inp.startswith("onnx::MatMul"):
                    weight_name = inp
            if weight_name is None:
                continue
            matmul_out = node.output[0]
            for node2 in graph.node:
                if node2.op_type == "Add" and matmul_out in node2.input:
                    for inp2 in node2.input:
                        if inp2 in init_data and not inp2.startswith("onnx::"):
                            matmul_to_bias[weight_name] = inp2

    # Also find MatMul weights that don't have bias (like whiten, pos projections)
    # by tracing which node outputs they connect to
    matmul_no_bias = {}
    for node in graph.node:
        if node.op_type == "MatMul":
            weight_name = None
            for inp in node.input:
                if inp in init_data and inp.startswith("onnx::MatMul"):
                    weight_name = inp
            if weight_name and weight_name not in matmul_to_bias:
                matmul_no_bias[weight_name] = init_data[weight_name].shape

    # Filter: stack 0, layer 0 weights
    # Named biases for stack 0 layer 0:
    #   encoder.encoders.0.layers.0.self_attn.in_proj.bias
    #   encoder.encoders.0.layers.0.self_attn.out_proj.bias
    #   encoder.encoders.0.layers.0.self_attn.out_proj2.bias
    #   encoder.encoders.0.layers.0.feed_forward{1,2,3}.{in,out}_proj.bias
    prefix = "encoder.encoders.0.layers.0."

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved_files = []

    # 1. Extract linear projection weights (MatMul weights mapped to biases)
    layer0_weights = {}
    for weight_name, bias_name in matmul_to_bias.items():
        if bias_name.startswith(prefix):
            param_type = bias_name[len(prefix):-len(".bias")]  # e.g. "self_attn.in_proj"
            safe_name = param_type.replace(".", "_")

            W = init_data[weight_name].astype(np.float32)
            # ONNX MatMul: X @ W where W is [K, N]
            # For rknn_matmul B matrix, we want [K, N] (K=in_features, N=out_features)
            # The ONNX weight is already [in_features, out_features] for MatMul
            # (NOT [out, in] like PyTorch nn.Linear)
            K, N = W.shape
            print(f"  {safe_name}: ONNX shape [{K}, {N}] -> save as FP16 [{K}, {N}]")

            # Save weight as FP16
            W_fp16 = W.astype(np.float16)
            w_path = os.path.join(OUTPUT_DIR, f"{safe_name}.weight.fp16.bin")
            W_fp16.tofile(w_path)
            saved_files.append(w_path)

            # Save bias as FP32
            bias = init_data[bias_name].astype(np.float32)
            b_path = os.path.join(OUTPUT_DIR, f"{safe_name}.bias.fp32.bin")
            bias.tofile(b_path)
            saved_files.append(b_path)

            layer0_weights[safe_name] = {"K": K, "N": N, "bias_size": len(bias)}

    # 2. Extract conv module weights for layer 0
    conv_prefix = prefix
    for name, data in init_data.items():
        if name.startswith(conv_prefix) and ("conv_module" in name):
            safe_name = name[len(conv_prefix):].replace(".", "_")
            path = os.path.join(OUTPUT_DIR, f"{safe_name}.fp32.bin")
            data.astype(np.float32).tofile(path)
            saved_files.append(path)
            print(f"  conv: {safe_name} shape={data.shape} -> FP32")

    # 3. Extract bypass_scale
    bypass_name = f"{prefix}bypass_scale"
    if bypass_name in init_data:
        path = os.path.join(OUTPUT_DIR, "bypass_scale.fp32.bin")
        init_data[bypass_name].astype(np.float32).tofile(path)
        saved_files.append(path)
        print(f"  bypass_scale: {init_data[bypass_name].item():.6f}")

    # 4. Also find and save the unmapped weights that belong to layer 0
    # These are things like self_attn.whiten (linear without bias)
    # onnx::MatMul_4844 [256, 256] and onnx::MatMul_4846 [256, 16]
    # We need to trace them more carefully. For now, save the ones right after
    # the layer 0 in_proj weight.
    # The in_proj weight for layer 0 is onnx::MatMul_4845 [256, 496]
    # So onnx::MatMul_4844 [256, 256] is likely the whiten projection
    # and onnx::MatMul_4846 [256, 16] is the pos_bias projection
    unmapped_candidates = []
    for wn in sorted(matmul_no_bias.keys()):
        shape = matmul_no_bias[wn]
        # Extract number from name
        m = re.match(r"onnx::MatMul_(\d+)", wn)
        if m:
            idx = int(m.group(1))
            unmapped_candidates.append((idx, wn, shape))

    # Layer 0 weights are around 4839-4875. The unmapped ones in this range:
    print("\n  Unmapped MatMul weights (no bias):")
    for idx, wn, shape in sorted(unmapped_candidates):
        print(f"    {wn}: {list(shape)}")
        if 4843 <= idx <= 4875:  # Layer 0 range
            safe_name = f"unmapped_{idx}"
            W = init_data[wn].astype(np.float16)
            path = os.path.join(OUTPUT_DIR, f"{safe_name}.weight.fp16.bin")
            W.tofile(path)
            saved_files.append(path)
            print(f"      -> saved as {safe_name} (layer 0 candidate)")

    # 5. Build meta.json
    meta = {
        "model": "zipformer-small-bilingual-zh-en",
        "stack": 0,
        "layer": 0,
        "hidden_dim": 256,
        "ffn_dim": 768,
        "attn_in_proj_dim": 496,
        "attn_out_dim": 96,  # out_proj input dim
        "key_dim": 192,
        "val_dim": 96,
        "num_stacks": 5,
        "layers_per_stack": 2,
        "encoder_output_dim": 512,
        "weights": layer0_weights,
        "conv_modules": ["conv_module1", "conv_module2"],
        "conv_kernel_size": 31,
        "feed_forward_count": 3,
        "notes": {
            "weight_layout": "ONNX MatMul [K, N] = [in_features, out_features]",
            "fp16_weights": "Linear projection weights saved as FP16",
            "fp32_biases": "All biases saved as FP32",
            "fp32_conv": "All conv weights saved as FP32",
        }
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    saved_files.append(META_PATH)

    print(f"\nSaved {len(saved_files)} files to {OUTPUT_DIR}")
    print(f"Meta saved to {META_PATH}")
    for f in sorted(saved_files):
        size = os.path.getsize(f)
        print(f"  {f} ({size/1024:.1f} KB)")

if __name__ == "__main__":
    main()
