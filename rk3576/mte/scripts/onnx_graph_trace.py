#!/usr/bin/env python3
"""
Trace the ONNX graph of the streaming Zipformer encoder node-by-node.

Goal: Understand the EXACT computation flow so we can write a Python reference
implementation that matches onnxruntime output (cos>0.99 per frame).

This script:
1. Loads the ONNX model
2. Lists all nodes in topological order
3. Groups nodes by encoder layer (stack/layer)
4. Identifies the operation sequence within each layer
5. Finds all operations missing from the current C engine
"""
import onnx
import onnx.numpy_helper
import numpy as np
import re
import sys
from collections import defaultdict, OrderedDict

MODEL_PATH = "/tmp/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder-epoch-99-avg-1.onnx"


def shape_str(shape):
    return str([d for d in shape])


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = onnx.load(MODEL_PATH)
    graph = model.graph

    # Build initializer map: name -> (shape, data)
    init_map = {}
    for init in graph.initializer:
        shape = [d for d in init.dims]
        init_map[init.name] = shape

    # Build output->node map
    output_to_node = {}
    for node in graph.node:
        for o in node.output:
            output_to_node[o] = node

    # Build node output -> consumers
    consumers = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            consumers[inp].append(node)

    # === Phase 1: Identify all named initializers (weights/biases) ===
    print("\n" + "=" * 80)
    print("PHASE 1: Named Initializers (non-onnx:: prefix)")
    print("=" * 80)

    named_inits = {}
    for name, shape in init_map.items():
        if not name.startswith("onnx::"):
            named_inits[name] = shape
            # print(f"  {name}: {shape}")

    # Group by component
    components = defaultdict(list)
    for name in sorted(named_inits.keys()):
        parts = name.split(".")
        if "encoders" in name:
            # Parse stack and layer
            m = re.match(r"encoder\.encoders\.(\d+)\.(layers\.(\d+)|encoder\.layers\.(\d+))\.(.*)", name)
            if m:
                stack = int(m.group(1))
                layer = int(m.group(3)) if m.group(3) is not None else int(m.group(4))
                param = m.group(5)
                components[f"stack{stack}_layer{layer}"].append((param, named_inits[name]))
            else:
                # Inter-stack components
                m2 = re.match(r"encoder\.encoders\.(\d+)\.(.*)", name)
                if m2:
                    stack = int(m2.group(1))
                    param = m2.group(2)
                    components[f"stack{stack}_inter"].append((param, named_inits[name]))
        elif "encoder_embed" in name:
            components["encoder_embed"].append((name, named_inits[name]))
        elif "encoder_proj" in name:
            components["encoder_proj"].append((name, named_inits[name]))
        elif "downsample_output" in name:
            components["downsample_output"].append((name, named_inits[name]))
        else:
            components["other"].append((name, named_inits[name]))

    for comp in sorted(components.keys()):
        params = components[comp]
        print(f"\n  [{comp}] ({len(params)} params)")
        for param, shape in params:
            print(f"    {param}: {shape}")

    # === Phase 2: Trace nodes in order, annotate with layer membership ===
    print("\n" + "=" * 80)
    print("PHASE 2: All Nodes in Execution Order")
    print("=" * 80)

    # For each node, determine which layer it belongs to by tracing which
    # initializer weights it uses (directly or transitively)
    def find_init_inputs(node, depth=0):
        """Find all initializer names used by this node."""
        inits = set()
        for inp in node.input:
            if inp in init_map:
                inits.add(inp)
        return inits

    # Print all nodes with their init inputs
    node_count = len(graph.node)
    print(f"\nTotal nodes: {node_count}")

    # Group nodes into segments based on which initializer they use
    # We'll identify the layer structure by finding patterns

    # First, let's just print a summary of node types and count per region
    region_starts = []  # (node_idx, label)

    for i, node in enumerate(graph.node):
        inits = find_init_inputs(node)
        named = [n for n in inits if not n.startswith("onnx::")]
        onnx_mm = [n for n in inits if n.startswith("onnx::MatMul")]

        # Identify key anchor points
        for n in named:
            if "encoder_embed" in n or "encoder_proj" in n:
                region_starts.append((i, n))
            elif "bypass_scale" in n:
                region_starts.append((i, n))
            elif "self_attn" in n and "bias" in n:
                region_starts.append((i, n))

    # === Phase 3: Detailed trace of ONE layer (stack 0, layer 0) ===
    print("\n" + "=" * 80)
    print("PHASE 3: Detailed Trace of Stack 0 Layer 0")
    print("=" * 80)

    # Find the range of nodes belonging to stack0_layer0
    # Strategy: find the first node that uses a stack0_layer0 weight,
    # and the last node before the first stack0_layer1 weight is used
    def find_layer_nodes(stack, layer):
        """Find node indices that belong to a specific stack/layer."""
        if stack == 0:
            prefix = f"encoder.encoders.0.layers.{layer}."
        else:
            prefix = f"encoder.encoders.{stack}.encoder.layers.{layer}."

        # Find nodes that directly use weights from this layer
        layer_node_indices = set()
        layer_outputs = set()

        for i, node in enumerate(graph.node):
            for inp in node.input:
                if inp in init_map:
                    # Named init
                    if not inp.startswith("onnx::"):
                        if inp.startswith(prefix):
                            layer_node_indices.add(i)
                    # onnx::MatMul weights — trace to see if they connect to this layer's bias
                    # This is complex, so we'll use a different approach

        return sorted(layer_node_indices)

    # Better approach: find nodes between ff1_in_proj of this layer and ff1_in_proj of next layer
    # by looking at the onnx::MatMul indices

    # Build bias_name -> onnx_matmul_name mapping
    matmul_to_bias = {}
    for node in graph.node:
        if node.op_type == "MatMul":
            wn = None
            for inp in node.input:
                if inp in init_map and inp.startswith("onnx::MatMul"):
                    wn = inp
            if wn is None:
                continue
            out = node.output[0]
            for node2 in graph.node:
                if node2.op_type == "Add" and out in node2.input:
                    for inp2 in node2.input:
                        if inp2 in init_map and not inp2.startswith("onnx::"):
                            matmul_to_bias[wn] = inp2

    bias_to_matmul = {v: k for k, v in matmul_to_bias.items()}

    # Find all onnx::MatMul_ indices for stack0_layer0
    s0l0_prefix = "encoder.encoders.0.layers.0."
    s0l0_matmul_indices = []
    for bias_name, matmul_name in bias_to_matmul.items():
        if bias_name.startswith(s0l0_prefix):
            m = re.match(r"onnx::MatMul_(\d+)", matmul_name)
            if m:
                s0l0_matmul_indices.append(int(m.group(1)))
    s0l0_matmul_indices.sort()
    print(f"\n  Stack0 Layer0 MatMul indices: {s0l0_matmul_indices}")
    print(f"  Range: {min(s0l0_matmul_indices)} - {max(s0l0_matmul_indices)}")

    # Also find unmapped MatMuls in this range
    for name, shape in init_map.items():
        if name.startswith("onnx::MatMul_"):
            idx = int(re.match(r"onnx::MatMul_(\d+)", name).group(1))
            if min(s0l0_matmul_indices) <= idx <= max(s0l0_matmul_indices):
                bias = matmul_to_bias.get(name, "UNMAPPED")
                print(f"    MatMul_{idx}: {shape} -> {bias}")

    # Find all nodes that produce or consume outputs related to stack0_layer0
    # by tracing from the MatMul nodes
    s0l0_matmul_names = set()
    for bias_name, matmul_name in bias_to_matmul.items():
        if bias_name.startswith(s0l0_prefix):
            s0l0_matmul_names.add(matmul_name)
    # Add unmapped ones in the range
    for name in init_map:
        if name.startswith("onnx::MatMul_"):
            idx = int(re.match(r"onnx::MatMul_(\d+)", name).group(1))
            if min(s0l0_matmul_indices) <= idx <= max(s0l0_matmul_indices):
                s0l0_matmul_names.add(name)

    # Now find all nodes in the subgraph rooted at these MatMuls
    # and all named inits with the s0l0 prefix
    s0l0_init_names = set()
    for name in init_map:
        if not name.startswith("onnx::") and name.startswith(s0l0_prefix):
            s0l0_init_names.add(name)
    s0l0_init_names.update(s0l0_matmul_names)

    # Trace: find all nodes where at least one input is from s0l0 inits or
    # from a node that's already in our set
    s0l0_node_indices = set()
    s0l0_tensor_names = set(s0l0_init_names)

    # Forward trace from s0l0 weights
    for i, node in enumerate(graph.node):
        for inp in node.input:
            if inp in s0l0_tensor_names:
                s0l0_node_indices.add(i)
                for out in node.output:
                    s0l0_tensor_names.add(out)
                break

    # This gives us too many nodes (downstream effects). Let's be more precise.
    # Instead, let's just find the contiguous range of nodes between embed output
    # and the start of layer 1.

    # === Phase 4: Print all nodes in order with annotations ===
    print("\n" + "=" * 80)
    print("PHASE 4: Complete Node List (annotated)")
    print("=" * 80)

    # For brevity, group by regions
    current_region = "preamble"

    for i, node in enumerate(graph.node):
        # Determine region from init inputs
        inits_used = find_init_inputs(node)
        named_used = [n for n in inits_used if not n.startswith("onnx::")]

        region = None
        for n in named_used:
            if "encoder_embed" in n:
                region = "encoder_embed"
            elif "encoder_proj" in n:
                region = "encoder_proj"
            elif "downsample_output" in n:
                region = "downsample_output"
            else:
                # Parse layer info
                m = re.match(r"encoder\.encoders\.(\d+)\.(layers\.(\d+)|encoder\.layers\.(\d+))\.(.*)", n)
                if m:
                    stack = int(m.group(1))
                    layer = int(m.group(3)) if m.group(3) is not None else int(m.group(4))
                    param = m.group(5)
                    region = f"s{stack}l{layer}:{param}"
                else:
                    m2 = re.match(r"encoder\.encoders\.(\d+)\.(.*)", n)
                    if m2:
                        stack = int(m2.group(1))
                        region = f"s{stack}_inter:{m2.group(2)}"

        for n in inits_used:
            if n.startswith("onnx::MatMul_"):
                bias = matmul_to_bias.get(n, None)
                if bias:
                    m = re.match(r"encoder\.encoders\.(\d+)\.(layers\.(\d+)|encoder\.layers\.(\d+))\.(.*?)\.bias", bias)
                    if m:
                        stack = int(m.group(1))
                        layer = int(m.group(3)) if m.group(3) is not None else int(m.group(4))
                        param = m.group(5)
                        region = f"s{stack}l{layer}:{param}"
                    elif "encoder_embed" in bias:
                        region = "encoder_embed"
                    elif "encoder_proj" in bias:
                        region = "encoder_proj"
                else:
                    # Unmapped matmul - annotate by index
                    idx = int(re.match(r"onnx::MatMul_(\d+)", n).group(1))
                    region = f"matmul_{idx}"

        # Print region change
        if region and region != current_region:
            print(f"\n  --- [{region}] ---")
            current_region = region

        # Compact node description
        inputs_str = ", ".join(node.input[:3])
        if len(node.input) > 3:
            inputs_str += f", ... (+{len(node.input)-3})"
        outputs_str = ", ".join(node.output[:2])

        # Get attributes
        attrs = {}
        for attr in node.attribute:
            if attr.type == 1:  # float
                attrs[attr.name] = attr.f
            elif attr.type == 2:  # int
                attrs[attr.name] = attr.i
            elif attr.type == 7:  # ints
                attrs[attr.name] = list(attr.ints)
            elif attr.type == 3:  # string
                attrs[attr.name] = attr.s.decode()

        attr_str = ""
        if attrs:
            # Show key attrs
            interesting = {k: v for k, v in attrs.items()
                          if k in ("axis", "perm", "shape", "to", "alpha",
                                  "starts", "ends", "axes", "steps",
                                  "kernel_shape", "strides", "pads", "dilations",
                                  "group", "value")}
            if interesting:
                attr_str = " " + str(interesting)

        # Init inputs annotation
        init_info = []
        for inp in node.input:
            if inp in init_map:
                s = init_map[inp]
                short = inp.split(".")[-1] if not inp.startswith("onnx::") else inp
                init_info.append(f"{short}{s}")

        init_str = ""
        if init_info:
            init_str = f" W={init_info}"

        print(f"  [{i:4d}] {node.op_type:15s} {outputs_str[:40]:40s}{attr_str}{init_str}")

    # === Phase 5: Detailed operation sequence for one full layer ===
    print("\n" + "=" * 80)
    print("PHASE 5: Operation Sequence for Stack 0 Layer 0 (detailed)")
    print("=" * 80)

    # Find exact node range: from the first node using s0l0 weight to
    # the node producing the s0l0 bypass output
    s0l0_first = None
    s0l0_last = None

    for i, node in enumerate(graph.node):
        uses_s0l0 = False
        for inp in node.input:
            if inp in init_map:
                name = inp
                if not name.startswith("onnx::"):
                    if name.startswith(s0l0_prefix):
                        uses_s0l0 = True
                elif name in matmul_to_bias:
                    bias = matmul_to_bias[name]
                    if bias.startswith(s0l0_prefix):
                        uses_s0l0 = True
                else:
                    # Unmapped matmul - check index range
                    m = re.match(r"onnx::MatMul_(\d+)", name)
                    if m:
                        idx = int(m.group(1))
                        if min(s0l0_matmul_indices) <= idx <= max(s0l0_matmul_indices):
                            uses_s0l0 = True

        if uses_s0l0:
            if s0l0_first is None:
                s0l0_first = i
            s0l0_last = i

    print(f"\n  Node range: [{s0l0_first}, {s0l0_last}] ({s0l0_last - s0l0_first + 1} nodes)")

    # Print all nodes in this range + a few before/after for context
    start = max(0, s0l0_first - 5)
    end = min(len(graph.node), s0l0_last + 10)

    for i in range(start, end):
        node = graph.node[i]
        in_layer = s0l0_first <= i <= s0l0_last
        marker = ">>>" if in_layer else "   "

        inputs_str = []
        for inp in node.input:
            if inp in init_map:
                short = inp.split(".")[-1] if not inp.startswith("onnx::") else inp
                shape = init_map[inp]
                inputs_str.append(f"W:{short}{shape}")
            else:
                inputs_str.append(inp[:30])

        attrs = {}
        for attr in node.attribute:
            if attr.type == 1:
                attrs[attr.name] = round(attr.f, 6)
            elif attr.type == 2:
                attrs[attr.name] = attr.i
            elif attr.type == 7:
                v = list(attr.ints)
                if len(v) <= 6:
                    attrs[attr.name] = v
                else:
                    attrs[attr.name] = f"[{v[0]},...,{v[-1]}] len={len(v)}"

        attr_str = f" {attrs}" if attrs else ""
        out_str = node.output[0][:40] if node.output else ""

        print(f"  {marker} [{i:4d}] {node.op_type:15s} -> {out_str:40s} IN=[{', '.join(inputs_str[:4])}]{attr_str}")

    # === Phase 6: Extract operation sequence annotation ===
    print("\n" + "=" * 80)
    print("PHASE 6: Annotated Operation Sequence")
    print("=" * 80)

    # Walk through nodes of s0l0 and identify sub-module boundaries
    print("""
Expected Zipformer layer structure (from icefall source):
  1. whiten(x, cached_avg, cached_len)  — BasicNorm / running mean whitening
  2. x = x + 0.5 * ff1(x)              — feed_forward1
  3. x = x + self_attn(x)              — self-attention with positional bias
  4. x = x + conv_module1(x)           — conv module 1
  5. x = x + 0.5 * ff2(x)              — feed_forward2
  6. x = x + conv_module2(x)           — conv module 2
  7. x = x + 0.5 * ff3(x)              — feed_forward3
  8. x = x * bypass_scale + x_orig * (1 - bypass_scale)  — bypass

Current C engine layer structure:
  1. x = x + 0.5 * ff1(x)              (NO whiten/BasicNorm!)
  2. x = x + self_attn(x)              (pos bias IGNORED)
  3. x = x + conv1(x)
  4. x = x + 0.5 * ff2(x)
  5. x = x + conv2(x)
  6. x = x + 0.5 * ff3(x)
  7. x = x * bypass_scale               (missing bypass blend!)

MISSING OPERATIONS in C engine:
  - BasicNorm / whiten (running mean normalization)
  - Positional bias in attention
  - Correct bypass formula (should blend with original, not just scale)
  - attn_whiten [256,256] and attn_whiten2 [256,96] weights unused
  - attn_pos_bias [256,16] weight unused
""")

    # Count ops by type in s0l0 range
    op_counts = defaultdict(int)
    for i in range(s0l0_first, s0l0_last + 1):
        op_counts[graph.node[i].op_type] += 1

    print("  Op type counts in s0l0:")
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"    {op}: {cnt}")

    # === Phase 7: Trace the attention sub-module in detail ===
    print("\n" + "=" * 80)
    print("PHASE 7: Attention Sub-module Detail (look for pos bias, whiten)")
    print("=" * 80)

    # Find nodes using attn_in_proj, whiten, pos_bias
    attn_nodes = []
    for i in range(s0l0_first, s0l0_last + 1):
        node = graph.node[i]
        for inp in node.input:
            if inp in init_map:
                name = inp
                is_attn = False
                if not name.startswith("onnx::"):
                    if "self_attn" in name and name.startswith(s0l0_prefix):
                        is_attn = True
                elif name in matmul_to_bias:
                    bias = matmul_to_bias[name]
                    if "self_attn" in bias and bias.startswith(s0l0_prefix):
                        is_attn = True
                else:
                    # Check unmapped in attn range
                    m = re.match(r"onnx::MatMul_(\d+)", name)
                    if m:
                        idx = int(m.group(1))
                        # in_proj is 4845, whiten=4843, pos_bias=4846
                        if 4840 <= idx <= 4875:
                            is_attn = True

                if is_attn:
                    attn_nodes.append(i)
                    break

    print(f"\n  Attention-related nodes: {len(attn_nodes)}")
    for i in attn_nodes:
        node = graph.node[i]
        inputs_str = []
        for inp in node.input:
            if inp in init_map:
                short = inp if inp.startswith("onnx::") else inp.split(".")[-1]
                inputs_str.append(f"W:{short}{init_map[inp]}")
            else:
                inputs_str.append(inp[:25])

        attrs = {}
        for attr in node.attribute:
            if attr.type == 2:
                attrs[attr.name] = attr.i
            elif attr.type == 7:
                v = list(attr.ints)
                attrs[attr.name] = v[:6]

        attr_str = f" {attrs}" if attrs else ""
        out_str = node.output[0][:35] if node.output else ""

        print(f"    [{i:4d}] {node.op_type:15s} -> {out_str:35s} IN=[{', '.join(inputs_str[:4])}]{attr_str}")

    # === Phase 8: Find BasicNorm / whiten pattern ===
    print("\n" + "=" * 80)
    print("PHASE 8: BasicNorm / Whiten Pattern Search")
    print("=" * 80)

    # Look for nodes that use cached_avg, cached_len
    # These should be at the start of each layer
    for i, node in enumerate(graph.node):
        for inp in node.input:
            if "cached_avg" in inp or "cached_len" in inp:
                # Print this node and the next few
                print(f"\n  cached state usage at node {i}:")
                for j in range(i, min(i + 15, len(graph.node))):
                    n = graph.node[j]
                    inputs = []
                    for x in n.input:
                        if x in init_map:
                            inputs.append(f"W:{x[:30]}{init_map[x]}")
                        elif "cached" in x:
                            inputs.append(f"STATE:{x}")
                        else:
                            inputs.append(x[:25])
                    out = n.output[0][:35] if n.output else ""
                    print(f"    [{j:4d}] {n.op_type:15s} -> {out:35s} IN=[{', '.join(inputs[:4])}]")
                break  # Only show first occurrence per cached state input

    # === Phase 9: Inter-stack topology ===
    print("\n" + "=" * 80)
    print("PHASE 9: Inter-stack Topology (downsample, upsample, combiner)")
    print("=" * 80)

    for i, node in enumerate(graph.node):
        for inp in node.input:
            if inp in init_map and not inp.startswith("onnx::"):
                if any(kw in inp for kw in ["downsample", "out_combiner", "skip_modules"]):
                    print(f"\n  Inter-stack weight at node {i}: {inp} {init_map[inp]}")
                    for j in range(max(0, i - 3), min(i + 8, len(graph.node))):
                        n = graph.node[j]
                        inputs = []
                        for x in n.input:
                            if x in init_map:
                                inputs.append(f"W:{x.split('.')[-1]}{init_map[x]}")
                            else:
                                inputs.append(x[:25])
                        marker = ">>>" if j == i else "   "
                        out = n.output[0][:35] if n.output else ""
                        print(f"    {marker} [{j:4d}] {n.op_type:15s} -> {out:35s} IN=[{', '.join(inputs[:4])}]")

    print("\n\n=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()
