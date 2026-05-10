#!/usr/bin/env python3
"""Roll selected W8A16LinearPlugin nodes back to FP16 MatMul.

This is intended for layer-selective W8A16 experiments: keep most constant
weight MatMuls quantized, but restore selected projections so we can measure
quality, latency, and memory contribution without re-exporting the model.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import convert_model_to_external_data


def _attr_int(node: onnx.NodeProto, name: str, default: int = 0) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def _node_label(node: onnx.NodeProto) -> str:
    return " ".join([node.name, *node.input, *node.output])


def _projection_bucket(label: str) -> str:
    for key in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if key in label:
            return key
    if "self_attn" in label:
        return "self_attn_other"
    if "mlp" in label:
        return "mlp_other"
    return "other"


def _dequant_weight(qweight: np.ndarray, scales: np.ndarray, layout: int) -> np.ndarray:
    if qweight.ndim != 2:
        raise ValueError(f"expected 2D qweight, got {qweight.shape}")
    if scales.ndim != 1:
        scales = scales.reshape(-1)
    if layout == 0:
        if scales.shape[0] != qweight.shape[1]:
            raise ValueError(f"layout0 scales {scales.shape} do not match qweight {qweight.shape}")
        return (qweight.astype(np.float16) * scales.astype(np.float16).reshape(1, -1)).astype(np.float16)
    if layout == 1:
        if scales.shape[0] != qweight.shape[0]:
            raise ValueError(f"layout1 scales {scales.shape} do not match qweight {qweight.shape}")
        weight_nk = qweight.astype(np.float16) * scales.astype(np.float16).reshape(-1, 1)
        return weight_nk.T.copy().astype(np.float16)
    raise ValueError(f"unsupported W8A16 weight_layout={layout}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input ONNX model")
    parser.add_argument("--output", required=True, help="Output ONNX model")
    parser.add_argument(
        "--rollback-regex",
        action="append",
        default=[],
        help="Regex matched against node name, inputs, and outputs. May be repeated.",
    )
    parser.add_argument("--external-data-file", default="onnx_model.data")
    parser.add_argument("--all-to-fp16", action="store_true", help="Rollback every W8A16 node")
    args = parser.parse_args()

    if not args.rollback_regex and not args.all_to_fp16:
        raise SystemExit("provide --rollback-regex or --all-to-fp16")

    patterns = [re.compile(pattern) for pattern in args.rollback_regex]
    model = onnx.load_model(args.input, load_external_data=True)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}

    kept = Counter()
    rolled = Counter()
    total_w8 = 0
    added_initializers: list[onnx.TensorProto] = []

    for node in model.graph.node:
        if node.op_type != "W8A16LinearPlugin":
            continue
        total_w8 += 1
        label = _node_label(node)
        bucket = _projection_bucket(label)
        should_rollback = args.all_to_fp16 or any(pattern.search(label) for pattern in patterns)
        if not should_rollback:
            kept[bucket] += 1
            continue

        qweight_tensor = initializers.get(node.input[1])
        scales_tensor = initializers.get(node.input[2])
        if qweight_tensor is None or scales_tensor is None:
            raise ValueError(f"{node.name}: missing qweight/scales initializers")
        qweight = numpy_helper.to_array(qweight_tensor)
        scales = numpy_helper.to_array(scales_tensor)
        layout = _attr_int(node, "weight_layout", 0)
        weight = _dequant_weight(qweight, scales, layout)
        weight_name = f"{node.name or node.output[0]}_fp16_weight"
        added_initializers.append(numpy_helper.from_array(weight, name=weight_name))

        old_name = node.name
        activation_name = node.input[0]
        output_name = node.output[0]
        del node.attribute[:]
        del node.input[:]
        del node.output[:]
        node.op_type = "MatMul"
        node.domain = ""
        node.name = f"{old_name}_fp16_matmul" if old_name else f"{output_name}_fp16_matmul"
        node.input.extend([activation_name, weight_name])
        node.output.extend([output_name])
        rolled[bucket] += 1

    model.graph.initializer.extend(added_initializers)
    onnx.checker.check_model(model)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=args.external_data_file,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.save_model(model, output)

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"total_w8={total_w8}")
    print(f"rolled={sum(rolled.values())}")
    for key, value in sorted(rolled.items()):
        print(f"  rolled {key}: {value}")
    print(f"kept={sum(kept.values())}")
    for key, value in sorted(kept.items()):
        print(f"  kept {key}: {value}")


if __name__ == "__main__":
    main()
