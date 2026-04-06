#!/usr/bin/env python3
"""Analyze streaming Zipformer ONNX encoder: list all initializers, trace MatMul weights
to named layers, identify architecture dimensions."""
import onnx
import numpy as np
import re
import sys
from collections import defaultdict, OrderedDict

MODEL_PATH = "/tmp/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder-epoch-99-avg-1.onnx"
OUTPUT_PATH = "/tmp/jetson-voice-mte/rk3576/mte/scripts/model_analysis.txt"

def shape_from_tensor(t):
    return [d for d in t.dims]

def dtype_name(dt):
    mapping = {1: "float32", 7: "int64", 10: "float16", 6: "int32", 9: "bool"}
    return mapping.get(dt, f"dtype_{dt}")

def main():
    print(f"Loading model: {MODEL_PATH}")
    model = onnx.load(MODEL_PATH, load_external_data=False)
    graph = model.graph

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 80)
    log("ONNX Model Analysis: Streaming Zipformer Encoder")
    log("=" * 80)

    # --- Graph inputs/outputs ---
    log("\n--- Graph Inputs ---")
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        log(f"  {inp.name}: {shape} ({dtype_name(inp.type.tensor_type.elem_type)})")

    log("\n--- Graph Outputs ---")
    for out in graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        log(f"  {out.name}: {shape} ({dtype_name(out.type.tensor_type.elem_type)})")

    # Build lookup: initializer name -> shape, dtype
    init_map = {}
    for init in graph.initializer:
        init_map[init.name] = (shape_from_tensor(init), init.data_type)

    # --- Trace MatMul nodes to find which bias/named tensor connects to which weight ---
    # Build output->node map and node output->consumers
    output_to_node = {}
    for node in graph.node:
        for o in node.output:
            output_to_node[o] = node

    # For each MatMul node, find its weight input (the initializer) and trace forward
    # to find the bias Add that follows, which has a named initializer
    matmul_to_bias = {}
    bias_to_matmul = {}

    for node in graph.node:
        if node.op_type == "MatMul":
            # One input should be an initializer (weight), one should be activation
            weight_name = None
            for inp in node.input:
                if inp in init_map and inp.startswith("onnx::MatMul"):
                    weight_name = inp
            if weight_name is None:
                continue

            # Trace forward: find the Add node that uses this MatMul's output + a named bias
            matmul_out = node.output[0]
            # Search all Add nodes for one that consumes this output
            for node2 in graph.node:
                if node2.op_type == "Add" and matmul_out in node2.input:
                    for inp2 in node2.input:
                        if inp2 in init_map and not inp2.startswith("onnx::"):
                            matmul_to_bias[weight_name] = inp2
                            bias_to_matmul[inp2] = weight_name

    log(f"\n--- MatMul Weight -> Bias Mapping ({len(matmul_to_bias)} pairs) ---")

    # Group by encoder stack and layer
    layer_weights = defaultdict(dict)  # (stack, layer) -> {param_type: (weight_name, shape)}

    for weight_name, bias_name in sorted(matmul_to_bias.items(), key=lambda x: x[1]):
        w_shape = init_map[weight_name][0]
        b_shape = init_map[bias_name][0]
        log(f"  {weight_name:30s} {str(w_shape):20s} <-> {bias_name} {b_shape}")

        # Parse bias name to get layer location
        # encoder.encoders.0.layers.0.self_attn.in_proj.bias -> stack=0, layer=0, type=self_attn.in_proj
        m = re.match(r"encoder\.encoders\.(\d+)\.(layers\.(\d+)|encoder\.layers\.(\d+))\.(.*?)\.bias", bias_name)
        if m:
            stack = int(m.group(1))
            layer = int(m.group(3)) if m.group(3) is not None else int(m.group(4))
            param_type = m.group(5)
            layer_weights[(stack, layer)][param_type] = (weight_name, w_shape, bias_name, b_shape)
        elif bias_name == "encoder_proj.bias":
            layer_weights[("proj", 0)]["encoder_proj"] = (weight_name, w_shape, bias_name, b_shape)

    # --- Architecture Summary Per Layer ---
    log("\n" + "=" * 80)
    log("Architecture Summary Per Encoder Layer")
    log("=" * 80)

    for (stack, layer) in sorted(layer_weights.keys(), key=lambda x: (str(x[0]), x[1])):
        params = layer_weights[(stack, layer)]
        log(f"\n  Stack {stack}, Layer {layer}:")
        for param_type in sorted(params.keys()):
            wn, ws, bn, bs = params[param_type]
            log(f"    {param_type:40s}: weight {ws} bias {bs}")

    # --- Extract architecture dimensions ---
    log("\n" + "=" * 80)
    log("Detected Architecture Dimensions")
    log("=" * 80)

    # From first layer (stack 0, layer 0):
    first_layer = layer_weights.get((0, 0), {})

    hidden_dim = None
    attention_dim = None
    num_heads = None
    ffn_dim = None

    # self_attn.in_proj: weight [256, 496] -> in=256 (hidden_dim), out=496 (= Q+K+V+pos?)
    if "self_attn.in_proj" in first_layer:
        _, ws, _, bs = first_layer["self_attn.in_proj"]
        in_proj_in = ws[0]   # ONNX MatMul: [K, N] means X @ W = [M,K] @ [K,N] = [M,N]
        in_proj_out = ws[1]
        hidden_dim = in_proj_in
        attention_dim = in_proj_out
        log(f"  self_attn.in_proj: [{in_proj_in}, {in_proj_out}]")
        log(f"    hidden_dim = {hidden_dim}")
        log(f"    in_proj output = {in_proj_out} (Q+K+V+pos combined)")

    if "self_attn.out_proj" in first_layer:
        _, ws, _, _ = first_layer["self_attn.out_proj"]
        log(f"  self_attn.out_proj: {ws}")

    if "self_attn.out_proj2" in first_layer:
        _, ws, _, _ = first_layer["self_attn.out_proj2"]
        log(f"  self_attn.out_proj2: {ws}")

    for ff_name in ["feed_forward1.in_proj", "feed_forward1.out_proj",
                     "feed_forward2.in_proj", "feed_forward2.out_proj",
                     "feed_forward3.in_proj", "feed_forward3.out_proj"]:
        if ff_name in first_layer:
            _, ws, _, bs = first_layer[ff_name]
            log(f"  {ff_name}: weight {ws} bias {bs}")
            if "in_proj" in ff_name:
                ffn_dim = ws[1]

    # Count stacks and layers
    stacks = defaultdict(set)
    for (s, l) in layer_weights.keys():
        if isinstance(s, int):
            stacks[s].add(l)

    log(f"\n  === Summary ===")
    log(f"  Number of encoder stacks: {len(stacks)}")
    for s in sorted(stacks.keys()):
        log(f"    Stack {s}: {len(stacks[s])} layers (indices {sorted(stacks[s])})")
    log(f"  hidden_dim: {hidden_dim}")
    log(f"  ffn_dim: {ffn_dim}")
    log(f"  attention in_proj output: {attention_dim}")

    # From cached_key shapes, infer num_heads per stack
    log(f"\n  --- Cached key/val shapes (num_heads per stack) ---")
    for inp in graph.input:
        if inp.name.startswith("cached_key_"):
            shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
            stack_idx = int(inp.name.split("_")[-1])
            num_heads_stack = shape[1]
            head_dim_k = shape[3]
            log(f"    Stack {stack_idx}: cached_key shape {shape} -> num_heads={num_heads_stack}, head_dim_k={head_dim_k}")
    for inp in graph.input:
        if inp.name.startswith("cached_val_") and not inp.name.startswith("cached_val2_"):
            shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
            stack_idx = int(inp.name.split("_")[-1])
            head_dim_v = shape[3]
            log(f"    Stack {stack_idx}: cached_val shape {shape} -> head_dim_v={head_dim_v}")

    # self_attn.in_proj output=496, if num_heads=4 for stack 0 (64 heads from cached_key):
    # Wait, cached_key_0 has 64 heads? That seems like left_context_len, not num_heads.
    # Let me check: cached_key_0: [2, 64, N, 192]
    # The 64 is the cache length, N is batch. head_dim_k = 192
    # Actually in Zipformer, cached_key shape is [num_left_parts, left_context_len, batch, attention_dim]
    # So 192 is the full attention dim for keys
    # in_proj output = 496, and we need to figure out the Q/K/V/pos split
    # For Zipformer: in_proj outputs [Q || K || V || pos_bias]
    # If head_dim_k = head_dim_v (from cached_val), and attention_dim = head_dim_k
    # Then: Q_dim + K_dim + V_dim + pos_dim = 496
    # From cached: K_dim = 192, V_dim = 96
    # So Q_dim + pos_dim = 496 - 192 - 96 = 208
    # In Zipformer, pos_bias has shape [num_heads, head_dim] and Q_dim = num_heads * head_dim
    # Actually the split might be: Q(=num_heads*head_dim) + K(=num_heads*head_dim) + V(=num_heads*head_dim) + pos(=num_heads*head_dim)
    # But K=192, V=96 doesn't make sense with same head_dim...
    # In Zipformer, K and V have different head dims! K has 3*head_dim and V has head_dim (with value_head_dim)
    # With 4 heads: K = 4 * 48 = 192, V = 4 * 24 = 96
    # Q = 4 * (48+48) = ... no. Let me just compute:
    # in_proj_out = 496. Let's see if 496 = Q + K + V + whidden
    # 192 + 96 = 288, 496 - 288 = 208
    # 208 could be Q (= num_heads * query_head_dim) + pos_bias_dim
    # With num_heads = 4: Q = 4*32 = 128, pos = 4*20 = 80? No...
    # Actually let me check the self_attn.linear_pos weight if it exists
    # Or check onnx::MatMul_4846: [256, 16] -- this is the pos_bias weight

    log(f"\n  --- Attention dimension analysis ---")
    # Stack 0: cached_key_0 [2, 64, N, 192] -- 64 is left_context, 192 is key_dim
    # Stack 0: cached_val_0 [2, 64, N, 96] -- 96 is value_dim
    # self_attn.in_proj: [256, 496]
    # 496 decomposition with key_dim=192, val_dim=96:
    # Remaining for Q + pos = 496 - 192 - 96 = 208

    # From Zipformer code: in_proj splits into [query, key, value, pos]
    # where pos has shape [num_heads, pos_head_dim]
    # Let's check the onnx::MatMul_4846 [256, 16] -- this could be pos_head_dim related
    # Actually 16 = 4 heads * 4 pos_head_dim or 8 heads * 2 pos_head_dim

    # Let me look at num_heads from the Reshape/Split patterns by checking
    # the in_proj bias = 496 = query_dim + key_dim + value_dim + pos_dim
    # With the Zipformer architecture, let's try: num_heads=4
    # key_dim = 192 = 4 * 48, val_dim = 96 = 4 * 24
    # query_dim = num_heads * (key_head_dim + val_head_dim) = 4 * (48 + 24) = 288? No, too big
    # Actually query_head_dim is separate.
    # 496 - 192 - 96 = 208 for query + pos
    # If num_heads = 4, pos_dim = 4 * 4 = 16 (from the [256,16] weight)
    # Then query_dim = 208 - 16 = 192 -> query_head_dim = 48
    # But wait, that [256,16] might not be pos for this layer specifically.
    # Let me just report what we know.

    for stack_idx in sorted(stacks.keys()):
        # Find cached dimensions
        for inp in graph.input:
            if inp.name == f"cached_key_{stack_idx}":
                key_shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
                left_ctx = key_shape[1]
                key_dim = key_shape[3]
            if inp.name == f"cached_val_{stack_idx}":
                val_shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
                val_dim = val_shape[3]

        fl = layer_weights.get((stack_idx, 0), {})
        if "self_attn.in_proj" in fl:
            _, ws, _, _ = fl["self_attn.in_proj"]
            in_proj_out = ws[1]
            remaining = in_proj_out - key_dim - val_dim
            log(f"    Stack {stack_idx}: in_proj_out={in_proj_out}, key_dim={key_dim}, val_dim={val_dim}, "
                f"remaining(Q+pos)={remaining}, left_context={left_ctx}")

    # --- Conv weights summary ---
    log(f"\n  --- Convolution Module Weights ---")
    for init in graph.initializer:
        name = init.name
        if "conv_module" in name and "weight" in name and "pointwise" in name:
            shape = shape_from_tensor(init)
            log(f"    {name}: {shape}")
            break  # Just show pattern

    log(f"\n  --- Embedding Conv Weights ---")
    for init in graph.initializer:
        name = init.name
        if "encoder_embed" in name:
            shape = shape_from_tensor(init)
            log(f"    {name}: {shape}")

    # Total parameter count
    total_params = sum(int(np.prod(shape_from_tensor(i))) for i in graph.initializer)
    log(f"\n  Total initializers: {len(graph.initializer)}")
    log(f"  Total parameters: {total_params:,} ({total_params*4/1024/1024:.1f} MB as FP32)")

    # --- Op type counts ---
    op_counts = defaultdict(int)
    for node in graph.node:
        op_counts[node.op_type] += 1
    log("\n--- Op Type Counts ---")
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1])[:20]:
        log(f"  {op}: {cnt}")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nAnalysis saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
