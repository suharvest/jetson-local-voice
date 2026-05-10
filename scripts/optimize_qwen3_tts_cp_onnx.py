#!/usr/bin/env python3
"""Small graph optimizations for the Qwen3-TTS CodePredictor ONNX.

Currently this only fuses each MLP gate/up projection pair:

    x @ W_gate, x @ W_up  ->  y = x @ concat(W_gate, W_up); slice(y)

The transformation preserves math and sampling semantics. It is meant to give
TensorRT a larger GEMM and one fewer MatMul launch per decoder layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _const_i64(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def fuse_mlp_gate_up(model: onnx.ModelProto) -> int:
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    nodes = list(graph.node)

    gate_by_layer: dict[str, onnx.NodeProto] = {}
    up_by_layer: dict[str, onnx.NodeProto] = {}
    for node in nodes:
        if node.op_type != "MatMul":
            continue
        name = node.name
        if "/mlp/gate_proj/MatMul" in name:
            layer = name.split("/mlp/gate_proj/MatMul", 1)[0]
            gate_by_layer[layer] = node
        elif "/mlp/up_proj/MatMul" in name:
            layer = name.split("/mlp/up_proj/MatMul", 1)[0]
            up_by_layer[layer] = node

    fused: dict[str, list[onnx.NodeProto]] = {}
    skip_ids: set[int] = set()
    new_initializers: list[onnx.TensorProto] = []

    for layer, gate in sorted(gate_by_layer.items()):
        up = up_by_layer.get(layer)
        if up is None:
            continue
        if gate.input[0] != up.input[0]:
            continue
        if gate.input[1] not in initializers or up.input[1] not in initializers:
            continue

        gate_w = numpy_helper.to_array(initializers[gate.input[1]])
        up_w = numpy_helper.to_array(initializers[up.input[1]])
        if gate_w.ndim != 2 or up_w.ndim != 2 or gate_w.shape[0] != up_w.shape[0] or gate_w.shape[1] != up_w.shape[1]:
            continue

        fused_w = np.concatenate([gate_w, up_w], axis=1)
        safe_layer = layer.strip("/").replace("/", "_").replace(".", "_")
        weight_name = f"{safe_layer}_mlp_gate_up_fused_weight"
        fused_out = f"{layer}/mlp/gate_up_fused/MatMul_output_0"
        gate_out = gate.output[0]
        up_out = up.output[0]
        hidden = int(gate_w.shape[1])

        new_initializers.append(numpy_helper.from_array(fused_w, name=weight_name))
        new_initializers.extend(
            [
                _const_i64(f"{safe_layer}_gate_slice_starts", [0]),
                _const_i64(f"{safe_layer}_gate_slice_ends", [hidden]),
                _const_i64(f"{safe_layer}_up_slice_starts", [hidden]),
                _const_i64(f"{safe_layer}_up_slice_ends", [2 * hidden]),
                _const_i64(f"{safe_layer}_gate_up_slice_axes", [-1]),
            ]
        )

        fused[layer] = [
            helper.make_node(
                "MatMul",
                [gate.input[0], weight_name],
                [fused_out],
                name=f"{layer}/mlp/gate_up_fused/MatMul",
            ),
            helper.make_node(
                "Slice",
                [
                    fused_out,
                    f"{safe_layer}_gate_slice_starts",
                    f"{safe_layer}_gate_slice_ends",
                    f"{safe_layer}_gate_up_slice_axes",
                ],
                [gate_out],
                name=f"{layer}/mlp/gate_proj/SliceFromFused",
            ),
            helper.make_node(
                "Slice",
                [
                    fused_out,
                    f"{safe_layer}_up_slice_starts",
                    f"{safe_layer}_up_slice_ends",
                    f"{safe_layer}_gate_up_slice_axes",
                ],
                [up_out],
                name=f"{layer}/mlp/up_proj/SliceFromFused",
            ),
        ]
        skip_ids.add(id(up))

    if not fused:
        return 0

    new_nodes: list[onnx.NodeProto] = []
    for node in nodes:
        if id(node) in skip_ids:
            continue
        layer = node.name.split("/mlp/gate_proj/MatMul", 1)[0] if "/mlp/gate_proj/MatMul" in node.name else None
        if layer in fused and node is gate_by_layer.get(layer):
            new_nodes.extend(fused[layer])
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_initializers)
    return len(fused)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    model = onnx.load(args.input)
    count = fuse_mlp_gate_up(model)
    if count == 0:
        raise SystemExit("no gate/up MLP pairs were fused")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.check:
        onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"fused_mlp_gate_up={count} output={args.output}")


if __name__ == "__main__":
    main()
