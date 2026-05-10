#!/usr/bin/env python3
"""Replace Qwen3-TTS CP MLP subgraphs with Qwen3TtsCpMlpPlugin nodes.

This is an experimental performance path. It preserves the per-layer MLP
mathematics:

    down_proj(silu(gate_proj(x)) * up_proj(x))

and replaces the five CP MLP blocks with one plugin node each. The plugin keeps
the same output tensor name as the original down projection so the rest of the
graph remains untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


def _node_by_name(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {node.name: node for node in model.graph.node}


def _remove_and_insert(model: onnx.ModelProto, remove_names: set[str], insert_after: dict[str, onnx.NodeProto]) -> None:
    new_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in insert_after:
            new_nodes.append(insert_after[node.name])
        if node.name not in remove_names:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)


def transform(model: onnx.ModelProto, *, layers: int = 5) -> int:
    nodes = _node_by_name(model)
    remove: set[str] = set()
    insert_after: dict[str, onnx.NodeProto] = {}
    fused = 0

    for layer in range(layers):
        prefix = f"/backbone/layers.{layer}/mlp"
        gate = nodes[f"{prefix}/gate_proj/MatMul"]
        up = nodes[f"{prefix}/up_proj/MatMul"]
        sigmoid = nodes[f"{prefix}/act_fn/Sigmoid"]
        silu_mul = nodes[f"{prefix}/act_fn/Mul"]
        mul = nodes[f"{prefix}/Mul"]
        down = nodes[f"{prefix}/down_proj/MatMul"]

        if gate.input[0] != up.input[0]:
            raise ValueError(f"layer {layer}: gate/up inputs differ: {gate.input[0]} vs {up.input[0]}")
        if sigmoid.input[0] != gate.output[0] or silu_mul.input[0] != gate.output[0]:
            raise ValueError(f"layer {layer}: unsupported SiLU pattern")
        if up.output[0] not in mul.input or silu_mul.output[0] not in mul.input:
            raise ValueError(f"layer {layer}: unsupported gate/up multiply pattern")
        if down.input[0] != mul.output[0]:
            raise ValueError(f"layer {layer}: down input is not MLP multiply output")

        plugin = helper.make_node(
            "Qwen3TtsCpMlpPlugin",
            inputs=[gate.input[0], gate.input[1], up.input[1], down.input[1]],
            outputs=[down.output[0]],
            name=f"{prefix}/Qwen3TtsCpMlpPlugin",
            hidden_size=1024,
            ffn_size=3072,
        )
        remove.update({gate.name, up.name, sigmoid.name, silu_mul.name, mul.name, down.name})
        insert_after[down.name] = plugin
        fused += 1

    _remove_and_insert(model, remove, insert_after)
    return fused


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, default=5)
    args = parser.parse_args()

    model = onnx.load(args.input)
    fused = transform(model, layers=args.layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"fused_mlp={fused} output={args.output}")


if __name__ == "__main__":
    main()
