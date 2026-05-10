#!/usr/bin/env python3
"""Pre-transpose Qwen3-TTS CP stacked lm_head weights.

The native CP ONNX stores ``stacked_weights`` as [15, 2048, 1024], then does:

    Gather(stacked_weights, gen_step) -> [2048, 1024]
    Transpose -> [1024, 2048]
    MatMul(hidden, transposed_head)

That runtime Transpose costs about 0.10ms per decode on Orin Nano. This script
changes the initializer to [15, 1024, 2048] and wires the final MatMul directly
to the Gather output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input)
    init_by_name = {init.name: init for init in model.graph.initializer}
    stacked = init_by_name.get("stacked_weights")
    if stacked is None:
        raise ValueError("missing stacked_weights initializer")
    weights = numpy_helper.to_array(stacked)
    if weights.shape != (15, 2048, 1024):
        raise ValueError(f"unexpected stacked_weights shape {weights.shape}")
    stacked.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(weights.transpose(0, 2, 1)), "stacked_weights"))

    gather = None
    transpose = None
    matmul = None
    for node in model.graph.node:
        if node.name == "/Gather":
            gather = node
        elif node.name == "/Transpose":
            transpose = node
        elif node.name == "/MatMul":
            matmul = node
    if gather is None or transpose is None or matmul is None:
        raise ValueError("missing expected /Gather -> /Transpose -> /MatMul tail")
    if list(transpose.input) != [gather.output[0]]:
        raise ValueError(f"unexpected transpose input {list(transpose.input)}")
    if matmul.input[1] != transpose.output[0]:
        raise ValueError(f"unexpected final MatMul input {list(matmul.input)}")
    matmul.input[1] = gather.output[0]

    new_nodes = [node for node in model.graph.node if node.name != "/Transpose"]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"removed_runtime_lm_head_transpose=1 output={args.output}")


if __name__ == "__main__":
    main()
