#!/usr/bin/env python3
"""
ONNX surgery: replace the `If` node in encoder.onnx with a Squeeze.

The original Qwen3-ASR encoder.onnx contains an `If` node whose two branches
output mismatched shapes ([128,T] vs [1,128,T]) — TensorRT 10.3 rejects this
with "IIfConditionalOutputLayer inputs must have the same shape".

The `If` checks whether batch_dim == 1 (always true for single-utterance ASR).
Both branches do the same work; only the shape differs. Replacing with a
Squeeze on dim 0 is mathematically equivalent for batch=1.

Usage:
    python3 scripts/surgery_encoder_onnx.py \\
        --in /path/to/encoder_fp16.onnx \\
        --out /path/to/encoder_fp16_trt.onnx
"""

import argparse
import sys

try:
    import onnx
except ImportError:
    print("ERROR: onnx package required. Install with: pip3 install onnx", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input encoder.onnx")
    ap.add_argument("--out", dest="out", required=True, help="Output surgeried .onnx")
    args = ap.parse_args()

    print(f"Loading {args.inp} ...")
    model = onnx.load(args.inp)
    g = model.graph

    # Find the If node — there should be exactly one in encoder.onnx
    if_nodes = [n for n in g.node if n.op_type == "If"]
    if len(if_nodes) != 1:
        print(f"ERROR: expected 1 If node, found {len(if_nodes)}", file=sys.stderr)
        sys.exit(1)

    if_node = if_nodes[0]
    print(f"Found If node: {if_node.name}, inputs={list(if_node.input)}, outputs={list(if_node.output)}")

    # The If node's then_branch should have a Squeeze that produces the [128,T] shape.
    # The else_branch keeps it as [1,128,T]. We want to use the else_branch's input directly.
    # Find the input tensor that comes IN to the If — that's our pre-If feature tensor.

    # Strategy: find what feeds the else_branch's output (it should be a Reshape or Identity
    # of an upstream tensor). Replace If with Squeeze of that upstream tensor on dim 0.

    # Inspect the else branch
    if not if_node.attribute:
        print("ERROR: If node has no attributes", file=sys.stderr)
        sys.exit(1)

    else_subgraph = None
    then_subgraph = None
    for attr in if_node.attribute:
        if attr.name == "else_branch":
            else_subgraph = attr.g
        elif attr.name == "then_branch":
            then_subgraph = attr.g

    if else_subgraph is None or then_subgraph is None:
        print("ERROR: missing else_branch or then_branch", file=sys.stderr)
        sys.exit(1)

    # The else branch's output should reference the input tensor (probably via Identity)
    # Walk the else subgraph to find the source tensor name.
    print(f"  Else branch nodes: {[(n.op_type, n.name) for n in else_subgraph.node]}")
    print(f"  Else branch outputs: {[o.name for o in else_subgraph.output]}")

    # Simplest case: else branch is just Identity of an outer tensor
    source_tensor = None
    if len(else_subgraph.node) == 1 and else_subgraph.node[0].op_type == "Identity":
        source_tensor = else_subgraph.node[0].input[0]
    elif len(else_subgraph.node) == 0 and len(else_subgraph.output) == 1:
        source_tensor = else_subgraph.output[0].name
    else:
        # Walk back from output through identity nodes
        out_name = else_subgraph.output[0].name
        for n in else_subgraph.node:
            if n.op_type == "Identity" and out_name in n.output:
                source_tensor = n.input[0]
                break
        if source_tensor is None:
            print(f"ERROR: cannot determine source tensor from else_branch", file=sys.stderr)
            print(f"       Manual inspection needed.", file=sys.stderr)
            sys.exit(1)

    print(f"Source tensor: {source_tensor}")

    # Replace If with Squeeze on axis 0
    if_idx = list(g.node).index(if_node)
    new_squeeze = onnx.helper.make_node(
        "Squeeze",
        inputs=[source_tensor, "_surgery_axis_0"],
        outputs=list(if_node.output),
        name=f"{if_node.name}_replaced_squeeze",
    )
    # axes initializer
    axes_init = onnx.helper.make_tensor("_surgery_axis_0", onnx.TensorProto.INT64, [1], [0])
    g.initializer.append(axes_init)

    g.node.pop(if_idx)
    g.node.insert(if_idx, new_squeeze)

    print(f"Saving {args.out} ...")
    onnx.save(model, args.out, save_as_external_data=True,
              all_tensors_to_one_file=True, location="encoder_weights.bin")
    print("Done.")
    print()
    print("Verify with:")
    print(f"  python3 -c \"import onnx; m=onnx.load('{args.out}'); print('If:', sum(1 for n in m.graph.node if n.op_type=='If'))\"")
    print("(should print 'If: 0')")


if __name__ == "__main__":
    main()
