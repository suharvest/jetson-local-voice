#!/usr/bin/env python3
"""Extract ALL Zipformer encoder weights from the ONNX model.

Extracts weights for:
- encoder_embed (input projection + 3 conv layers)
- 10 encoder layers (5 stacks x 2 layers): matmul, conv, bypass_scale
- encoder_proj (output projection)
- downsample/out_combiner/skip_modules between stacks

Weight layout:
- MatMul weights: saved as FP16 in [K, N] layout (same as ONNX)
- Biases/norms/conv: saved as FP32
"""
import onnx
import onnx.numpy_helper
import numpy as np
import os
import json
import re

MODEL_PATH = "/tmp/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder-epoch-99-avg-1.onnx"
WEIGHTS_DIR = "/tmp/jetson-voice-mte/rk3576/mte/weights"
META_PATH = os.path.join(WEIGHTS_DIR, "meta.json")

# Stack/layer prefixes in ONNX named initializers
# Stack 0 uses encoder.encoders.0.layers.{0,1}
# Stack 1-4 uses encoder.encoders.{1,2,3,4}.encoder.layers.{0,1}
def layer_prefix(stack, layer):
    if stack == 0:
        return f"encoder.encoders.0.layers.{layer}."
    else:
        return f"encoder.encoders.{stack}.encoder.layers.{layer}."

def layer_dir_name(stack, layer):
    return f"stack{stack}_layer{layer}"

# Mapping from bias path suffix to our short filename
MATMUL_SHORT_NAMES = {
    "feed_forward1.in_proj": "ff1_in",
    "feed_forward1.out_proj": "ff1_out",
    "feed_forward2.in_proj": "ff2_in",
    "feed_forward2.out_proj": "ff2_out",
    "feed_forward3.in_proj": "ff3_in",
    "feed_forward3.out_proj": "ff3_out",
    "self_attn.in_proj": "attn_in",
    "self_attn.out_proj": "attn_out",
    "self_attn.out_proj2": "attn_out2",
}

# Unmapped MatMul weights per layer (no bias): whiten [256,256], pos_bias [256,16], whiten2 [256,96]
UNMAPPED_SHAPES = {
    (256, 256): "attn_whiten",
    (256, 16): "attn_pos_bias",
    (256, 96): "attn_whiten2",
}

# Conv module sub-weights
CONV_PARTS = [
    "conv_module1.depthwise_conv.weight",
    "conv_module1.depthwise_conv.bias",
    "conv_module1.pointwise_conv1.weight",
    "conv_module1.pointwise_conv1.bias",
    "conv_module1.pointwise_conv2.weight",
    "conv_module1.pointwise_conv2.bias",
    "conv_module2.depthwise_conv.weight",
    "conv_module2.depthwise_conv.bias",
    "conv_module2.pointwise_conv1.weight",
    "conv_module2.pointwise_conv1.bias",
    "conv_module2.pointwise_conv2.weight",
    "conv_module2.pointwise_conv2.bias",
]

CONV_SHORT_NAMES = {
    "conv_module1.depthwise_conv.weight": "conv1_dw.fp32.bin",
    "conv_module1.depthwise_conv.bias": "conv1_dw_bias.fp32.bin",
    "conv_module1.pointwise_conv1.weight": "conv1_pw1.fp32.bin",
    "conv_module1.pointwise_conv1.bias": "conv1_pw1_bias.fp32.bin",
    "conv_module1.pointwise_conv2.weight": "conv1_pw2.fp32.bin",
    "conv_module1.pointwise_conv2.bias": "conv1_pw2_bias.fp32.bin",
    "conv_module2.depthwise_conv.weight": "conv2_dw.fp32.bin",
    "conv_module2.depthwise_conv.bias": "conv2_dw_bias.fp32.bin",
    "conv_module2.pointwise_conv1.weight": "conv2_pw1.fp32.bin",
    "conv_module2.pointwise_conv1.bias": "conv2_pw1_bias.fp32.bin",
    "conv_module2.pointwise_conv2.weight": "conv2_pw2.fp32.bin",
    "conv_module2.pointwise_conv2.bias": "conv2_pw2_bias.fp32.bin",
}


def main():
    print(f"Loading ONNX model: {MODEL_PATH}")
    model = onnx.load(MODEL_PATH)
    graph = model.graph

    # Build initializer map
    init_data = {}
    for init in graph.initializer:
        init_data[init.name] = onnx.numpy_helper.to_array(init)

    # Build MatMul weight -> bias mapping
    matmul_to_bias = {}
    for node in graph.node:
        if node.op_type == "MatMul":
            wn = None
            for inp in node.input:
                if inp in init_data and inp.startswith("onnx::MatMul"):
                    wn = inp
            if wn is None:
                continue
            out = node.output[0]
            for node2 in graph.node:
                if node2.op_type == "Add" and out in node2.input:
                    for inp2 in node2.input:
                        if inp2 in init_data and not inp2.startswith("onnx::"):
                            matmul_to_bias[wn] = inp2

    # Reverse map: bias_name -> weight_name
    bias_to_weight = {v: k for k, v in matmul_to_bias.items()}

    # Collect all unmapped MatMul weights with their indices
    unmapped = {}
    for name in init_data:
        if name.startswith("onnx::MatMul") and name not in matmul_to_bias:
            m = re.match(r"onnx::MatMul_(\d+)", name)
            if m:
                unmapped[int(m.group(1))] = name

    total_files = 0
    total_bytes = 0
    meta_layers = {}
    all_saved = []

    def save_fp16(data, path):
        nonlocal total_files, total_bytes
        arr = data.astype(np.float16)
        arr.tofile(path)
        total_files += 1
        total_bytes += arr.nbytes
        return arr.nbytes

    def save_fp32(data, path):
        nonlocal total_files, total_bytes
        arr = data.astype(np.float32)
        arr.tofile(path)
        total_files += 1
        total_bytes += arr.nbytes
        return arr.nbytes

    # =========================================================================
    # 1. encoder_embed: 3 conv layers + linear projection
    # =========================================================================
    print("\n=== encoder_embed ===")
    embed_dir = os.path.join(WEIGHTS_DIR, "encoder_embed")
    os.makedirs(embed_dir, exist_ok=True)

    # Conv layers: encoder.encoder_embed.conv.{0,3,6}.{weight,bias}
    for conv_idx in [0, 3, 6]:
        for suffix in ["weight", "bias"]:
            name = f"encoder.encoder_embed.conv.{conv_idx}.{suffix}"
            if name in init_data:
                fname = f"conv{conv_idx}_{suffix}.fp32.bin"
                path = os.path.join(embed_dir, fname)
                sz = save_fp32(init_data[name], path)
                shape = list(init_data[name].shape)
                print(f"  {name}: {shape} -> {fname} ({sz} bytes)")
                all_saved.append((path, shape))

    # Linear projection: onnx::MatMul_4839 [2432, 256] + encoder.encoder_embed.out.bias
    embed_weight_name = "onnx::MatMul_4839"
    embed_bias_name = "encoder.encoder_embed.out.bias"
    if embed_weight_name in init_data:
        W = init_data[embed_weight_name]
        path = os.path.join(embed_dir, "linear.fp16.bin")
        sz = save_fp16(W, path)
        print(f"  {embed_weight_name}: {list(W.shape)} -> linear.fp16.bin ({sz} bytes)")
        all_saved.append((path, list(W.shape)))
    if embed_bias_name in init_data:
        b = init_data[embed_bias_name]
        path = os.path.join(embed_dir, "linear_bias.fp32.bin")
        sz = save_fp32(b, path)
        print(f"  {embed_bias_name}: {list(b.shape)} -> linear_bias.fp32.bin ({sz} bytes)")
        all_saved.append((path, list(b.shape)))

    # =========================================================================
    # 2. All 10 encoder layers
    # =========================================================================
    for stack in range(5):
        for layer in range(2):
            prefix = layer_prefix(stack, layer)
            dirname = layer_dir_name(stack, layer)
            layer_dir = os.path.join(WEIGHTS_DIR, dirname)
            os.makedirs(layer_dir, exist_ok=True)
            print(f"\n=== {dirname} ({prefix[:-1]}) ===")

            layer_meta = {"matmul": {}, "conv": {}, "other": {}}

            # --- MatMul weights with biases (9 per layer) ---
            for param_suffix, short_name in MATMUL_SHORT_NAMES.items():
                bias_name = f"{prefix}{param_suffix}.bias"
                if bias_name not in bias_to_weight:
                    print(f"  WARNING: {bias_name} not found in bias_to_weight map!")
                    continue

                weight_onnx_name = bias_to_weight[bias_name]
                W = init_data[weight_onnx_name]
                K, N = W.shape

                # Save weight as FP16
                w_path = os.path.join(layer_dir, f"{short_name}.fp16.bin")
                save_fp16(W, w_path)

                # Save bias as FP32
                b = init_data[bias_name]
                b_path = os.path.join(layer_dir, f"{short_name}_bias.fp32.bin")
                save_fp32(b, b_path)

                print(f"  {short_name}: [{K}, {N}] + bias[{len(b)}]")
                layer_meta["matmul"][short_name] = {"K": K, "N": N, "bias_size": len(b)}
                all_saved.append((w_path, [K, N]))

            # --- Unmapped MatMul weights (whiten, pos_bias, whiten2) ---
            # Find them by proximity to the in_proj weight
            in_proj_bias = f"{prefix}self_attn.in_proj.bias"
            if in_proj_bias in bias_to_weight:
                in_proj_wn = bias_to_weight[in_proj_bias]
                m = re.match(r"onnx::MatMul_(\d+)", in_proj_wn)
                if m:
                    in_proj_idx = int(m.group(1))
                    # whiten is at idx-1, pos_bias is at idx+1, whiten2 is found
                    # by scanning near ff2_out_proj
                    # Pattern: whiten=[idx-1], in_proj=[idx], pos_bias=[idx+1]
                    # whiten2 is between ff2_out_proj and attn_out_proj2
                    for candidate_idx in sorted(unmapped.keys()):
                        wn = unmapped[candidate_idx]
                        shape = tuple(init_data[wn].shape)
                        if shape in UNMAPPED_SHAPES:
                            # Check proximity: within the range of this layer's weights
                            # Layer weights span from ff1_in to ff3_out
                            ff1_in_bias = f"{prefix}feed_forward1.in_proj.bias"
                            ff3_out_bias = f"{prefix}feed_forward3.out_proj.bias"
                            if ff1_in_bias in bias_to_weight and ff3_out_bias in bias_to_weight:
                                ff1_idx = int(re.match(r"onnx::MatMul_(\d+)", bias_to_weight[ff1_in_bias]).group(1))
                                ff3_idx = int(re.match(r"onnx::MatMul_(\d+)", bias_to_weight[ff3_out_bias]).group(1))
                                if ff1_idx <= candidate_idx <= ff3_idx:
                                    short_name = UNMAPPED_SHAPES[shape]
                                    W = init_data[wn]
                                    w_path = os.path.join(layer_dir, f"{short_name}.fp16.bin")
                                    save_fp16(W, w_path)
                                    print(f"  {short_name}: {list(shape)} (unmapped {wn})")
                                    layer_meta["matmul"][short_name] = {"K": shape[0], "N": shape[1]}
                                    all_saved.append((w_path, list(shape)))

            # --- Conv module weights ---
            for conv_part in CONV_PARTS:
                full_name = f"{prefix}{conv_part}"
                if full_name in init_data:
                    fname = CONV_SHORT_NAMES[conv_part]
                    data = init_data[full_name]
                    path = os.path.join(layer_dir, fname)
                    save_fp32(data, path)
                    shape = list(data.shape)
                    print(f"  {fname}: {shape}")
                    layer_meta["conv"][fname] = shape
                    all_saved.append((path, shape))

            # --- bypass_scale ---
            bypass_name = f"{prefix}bypass_scale"
            if bypass_name in init_data:
                path = os.path.join(layer_dir, "bypass_scale.fp32.bin")
                save_fp32(init_data[bypass_name], path)
                val = float(init_data[bypass_name])
                print(f"  bypass_scale: {val:.6f}")
                layer_meta["other"]["bypass_scale"] = val
                all_saved.append((path, [1]))

            meta_layers[dirname] = layer_meta

    # =========================================================================
    # 3. Inter-stack weights: downsample, out_combiner, skip_modules
    # =========================================================================
    print("\n=== Inter-stack weights ===")
    inter_dir = os.path.join(WEIGHTS_DIR, "inter_stack")
    os.makedirs(inter_dir, exist_ok=True)
    inter_meta = {}

    for name, data in sorted(init_data.items()):
        is_inter = False
        if "downsample" in name and not name.startswith("onnx::"):
            is_inter = True
        elif "out_combiner" in name:
            is_inter = True
        elif "skip_modules" in name:
            is_inter = True
        elif name == "encoder.downsample_output.query":
            is_inter = True

        if is_inter:
            safe_name = name.replace(".", "_").replace("encoder_", "")
            fname = f"{safe_name}.fp32.bin"
            path = os.path.join(inter_dir, fname)
            save_fp32(data, path)
            shape = list(data.shape)
            print(f"  {name}: {shape} -> {fname}")
            inter_meta[name] = shape
            all_saved.append((path, shape))

    # =========================================================================
    # 4. encoder_proj: output projection [256, 512]
    # =========================================================================
    print("\n=== encoder_proj ===")
    proj_weight_name = "onnx::MatMul_5221"
    proj_bias_name = "encoder_proj.bias"

    if proj_weight_name in init_data:
        W = init_data[proj_weight_name]
        path = os.path.join(WEIGHTS_DIR, "encoder_proj.fp16.bin")
        save_fp16(W, path)
        print(f"  {proj_weight_name}: {list(W.shape)} -> encoder_proj.fp16.bin")
        all_saved.append((path, list(W.shape)))
    if proj_bias_name in init_data:
        b = init_data[proj_bias_name]
        path = os.path.join(WEIGHTS_DIR, "encoder_proj_bias.fp32.bin")
        save_fp32(b, path)
        print(f"  {proj_bias_name}: {list(b.shape)} -> encoder_proj_bias.fp32.bin")
        all_saved.append((path, list(b.shape)))

    # =========================================================================
    # 5. Build comprehensive meta.json
    # =========================================================================
    meta = {
        "model": "zipformer-small-bilingual-zh-en",
        "source": MODEL_PATH,
        "architecture": {
            "num_stacks": 5,
            "layers_per_stack": 2,
            "total_layers": 10,
            "hidden_dim": 256,
            "ffn_dim": 768,
            "attn_in_proj_dim": 496,
            "attn_out_dim": 96,
            "key_dim": 192,
            "val_dim": 96,
            "encoder_input_dim": 2432,
            "encoder_output_dim": 512,
            "conv_kernel_size": 31,
            "feed_forward_count": 3,
            "num_heads_per_stack": [64, 32, 16, 8, 32],
        },
        "layers": meta_layers,
        "inter_stack": inter_meta,
        "format": {
            "matmul_weights": "FP16, [K, N] layout (in_features, out_features)",
            "biases": "FP32",
            "conv_weights": "FP32",
            "unmapped_attn_weights": "FP16, [K, N] layout",
        },
        "stats": {
            "total_files": total_files,
            "total_bytes": total_bytes,
            "total_kb": round(total_bytes / 1024, 1),
            "total_mb": round(total_bytes / (1024 * 1024), 2),
        },
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMeta saved to {META_PATH}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total files saved: {total_files}")
    print(f"  Total size: {total_bytes / 1024:.1f} KB ({total_bytes / (1024*1024):.2f} MB)")

    # Count by type
    matmul_count = sum(
        len(meta_layers[d]["matmul"]) for d in meta_layers
    )
    # Add encoder_embed linear + encoder_proj
    matmul_count += 2
    print(f"  MatMul projection weights: {matmul_count}")
    print(f"  (9 per layer x 10 layers + 3 unmapped/layer x 10 + embed + proj)")

    # Directory listing
    print(f"\n--- Directory structure ---")
    for dirpath, dirnames, filenames in sorted(os.walk(WEIGHTS_DIR)):
        level = dirpath.replace(WEIGHTS_DIR, "").count(os.sep)
        indent = "  " * level
        dirname = os.path.basename(dirpath)
        dir_size = sum(os.path.getsize(os.path.join(dirpath, f)) for f in filenames)
        print(f"{indent}{dirname}/ ({dir_size/1024:.1f} KB, {len(filenames)} files)")


if __name__ == "__main__":
    main()
