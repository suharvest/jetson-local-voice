#!/usr/bin/env python3
"""Convert W8A16LinearPlugin ONNX weights from [K, N] to [N, K].

This is a layout-only transform for models that are already rewritten to use
``W8A16LinearPlugin``. It keeps the quantized values and scales unchanged, but
transposes each plugin qweight initializer and adds ``weight_layout=1`` to the
plugin node. The runtime plugin can then use a small-M GEMV path with contiguous
per-output weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--external-data-file", default=None)
    parser.add_argument("--size-threshold", type=int, default=1024)
    args = parser.parse_args()

    model = onnx.load_model(args.input, load_external_data=True)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    converted = 0
    for node in model.graph.node:
        if node.op_type != "W8A16LinearPlugin":
            continue
        qweight_name = node.input[1]
        qweight_tensor = initializers[qweight_name]
        qweight = numpy_helper.to_array(qweight_tensor)
        if qweight.ndim != 2:
            raise ValueError(f"{qweight_name}: expected 2D qweight, got {qweight.shape}")
        qweight_tensor.CopyFrom(numpy_helper.from_array(qweight.T.copy(), name=qweight_name))
        for attr in node.attribute:
            if attr.name == "weight_layout":
                attr.i = 1
                break
        else:
            node.attribute.append(helper.make_attribute("weight_layout", 1))
        converted += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        args.output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=args.external_data_file or f"{args.output.name}.data",
        size_threshold=args.size_threshold,
    )
    print(f"converted_w8a16_nodes={converted}")


if __name__ == "__main__":
    main()
