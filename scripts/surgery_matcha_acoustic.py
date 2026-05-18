#!/usr/bin/env python3
"""ONNX surgery for Matcha acoustic model → TRT-compatible ONNX.

Two fixes applied:
1. RandomNormalLike → external noise input with internal Slice (TRT unsupported op)
2. length_scale: float32 → int64 (TRT requires int type for shape-tensor inputs)

Usage:
    python3 scripts/surgery_matcha_acoustic.py \\
        --input model-steps-3.onnx \\
        --output model-steps-3-trt.onnx \\
        --max-mel 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto


def make_int64_tensor(name: str, vals: list[int]) -> onnx.TensorProto:
    arr = np.array(vals, dtype=np.int64)
    return helper.make_tensor(name, TensorProto.INT64, arr.shape, arr.tobytes(), raw=True)


def make_float_tensor(name: str, vals: list[float]) -> onnx.TensorProto:
    arr = np.array(vals, dtype=np.float32)
    return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.tobytes(), raw=True)


def fix_random_normal_like(model: onnx.ModelProto, max_mel: int = 800):
    """Replace RandomNormalLike with max-size noise + internal Slice."""
    rnn_node = None
    shape_ref = None
    for n in model.graph.node:
        if n.op_type == "RandomNormalLike":
            rnn_node = n
            shape_ref = n.input[0]
            break

    if rnn_node is None:
        print("  No RandomNormalLike — skipping")
        return

    rnn_output = rnn_node.output[0]
    print(f"  RandomNormalLike: {rnn_node.name}, shape_ref={shape_ref}")

    # Find Mul consumer
    mul_node = None
    mul_idx = -1
    for n in model.graph.node:
        for idx, inp in enumerate(n.input):
            if inp == rnn_output:
                mul_node = n
                mul_idx = idx
                break

    if mul_node is None:
        raise RuntimeError("No Mul consumer for RandomNormalLike output")
    print(f"  Mul consumer: {mul_node.name}, noise_scale={mul_node.input[1 - mul_idx]}")

    # Add nodes: Shape, Cast, Slice before Mul
    shape_name = f"{rnn_node.name}_Shape"
    shape_out = f"{shape_name}_output_0"
    cast_name = f"{rnn_node.name}_Cast"
    cast_out = f"{cast_name}_output_0"
    slice_name = f"{rnn_node.name}_Slice"
    slice_out = f"{slice_name}_output_0"
    starts_name = f"{rnn_node.name}_slice_starts"
    axes_name = f"{rnn_node.name}_slice_axes"
    steps_name = f"{rnn_node.name}_slice_steps"

    new_nodes = [
        helper.make_node("Shape", [shape_ref], [shape_out], name=shape_name),
        helper.make_node("Cast", [shape_out], [cast_out],
                         to=int(TensorProto.INT64), name=cast_name),
        helper.make_node("Slice",
                         ["noise", starts_name, cast_out, axes_name, steps_name],
                         [slice_out], name=slice_name),
    ]
    new_initializers = [
        make_int64_tensor(starts_name, [0, 0, 0]),
        make_int64_tensor(axes_name, [0, 1, 2]),
        make_int64_tensor(steps_name, [1, 1, 1]),
    ]

    # Rewire Mul
    mul_node.input[mul_idx] = slice_out

    # Insert nodes before Mul
    model.graph.node.remove(rnn_node)
    insert_pos = list(model.graph.node).index(mul_node)
    for i, n in enumerate(new_nodes):
        model.graph.node.insert(insert_pos + i, n)
    for init in new_initializers:
        model.graph.initializer.append(init)

    # Add noise input [1, 80, max_mel]
    # Dynamic last dim — TRT will resolve it from the length_scale shape tensor chain
    noise_input = helper.make_tensor_value_info("noise", TensorProto.FLOAT, [1, 80, "L_mel"])
    model.graph.input.append(noise_input)
    print(f"  Added noise input [1, 80, {max_mel}] + {len(new_nodes)} nodes")


def fix_length_scale_type(model: onnx.ModelProto):
    """Convert length_scale to fixed-point INT64 (x1000) to appease TRT shape-tensor detection.

    TRT infers that length_scale is a shape tensor because it flows (through the
    duration predictor) to Range nodes. Shape-tensor inputs must be Int32/Int64.

    Codex-verified approach (2026-05-16):
    1. Change length_scale input from FLOAT to INT64 — represents round(ls * 1000)
    2. Insert Cast(INT64→FLOAT) + Div(1000.0) before every float consumer
    3. The path to Range already has Cast(FLOAT→INT64) nodes; the float now
       comes from the Div output, preserving fractional precision
    4. Runtime: pass int(round(length_scale * 1000)) as length_scale
    """

    ls_input = None
    for inp in model.graph.input:
        if inp.name == "length_scale":
            ls_input = inp
            break
    if ls_input is None:
        print("  length_scale input not found — skipping")
        return

    old_type = ls_input.type.tensor_type.elem_type
    ls_input.type.tensor_type.elem_type = TensorProto.INT64
    print(f"  Changed length_scale: {old_type} (FLOAT) → {TensorProto.INT64} (INT64, fixed-point x1000)")

    # Create the Div constant (1000.0) once
    div_const_name = "/length_scale_div_const"
    div_const = make_float_tensor(div_const_name, [1000.0])
    model.graph.initializer.append(div_const)

    # Find all direct consumers of length_scale and insert Cast(FLOAT) + Div(1000.0)
    float_consumers = []
    for n in model.graph.node:
        for idx, inp in enumerate(n.input):
            if inp == "length_scale":
                float_consumers.append((n, idx))

    for i, (node, idx) in enumerate(float_consumers):
        cast_name = f"/length_scale_cast_{i}"
        cast_out = f"{cast_name}_output_0"
        div_name = f"/length_scale_div_{i}"
        div_out = f"{div_name}_output_0"

        cast_node = helper.make_node(
            "Cast", ["length_scale"], [cast_out],
            to=int(TensorProto.FLOAT), name=cast_name
        )
        div_node = helper.make_node(
            "Div", [cast_out, div_const_name], [div_out], name=div_name
        )

        node.input[idx] = div_out
        insert_pos = list(model.graph.node).index(node)
        model.graph.node.insert(insert_pos, div_node)
        model.graph.node.insert(insert_pos, cast_node)
        print(f"  Inserted Cast+Div/1000 for {node.name}[{idx}]")

    print(f"  Fixed {len(float_consumers)} float consumer(s) with fixed-point scaling")


def fix_range_nodes(model: onnx.ModelProto, max_tokens: int = 64, max_mel: int = 1000):
    """Replace dynamic Range nodes with static constant + Slice.

    TRT 10.3 can't resolve dynamic Range limits when the limit value traces
    back through float computations to a float model input (length_scale).
    Replacing Range(0, N, 1) with Slice(static_arange, [0], [N], [0], [1])
    breaks the shape-tensor chain while preserving semantics.

    Three Range nodes are patched:
      /encoder/Range      → token positional encoding  (max N = max_tokens)
      /Range              → duration predictor index   (max N = max_mel)
      /Range_1            → duration predictor index   (max N = max_mel)
    """

    range_configs = [
        {"name": "/encoder/Range", "max": max_tokens},
        {"name": "/Range", "max": max_mel},
        {"name": "/Range_1", "max": max_mel},
    ]

    # Build map of value_info to check types
    value_types = {}
    for vi in model.graph.value_info:
        value_types[vi.name] = vi.type.tensor_type.elem_type

    replaced = 0
    for cfg in range_configs:
        rn = None
        for n in model.graph.node:
            if n.name == cfg["name"]:
                rn = n
                break
        if rn is None:
            print(f"  {cfg['name']}: not found, skipping")
            continue

        limit_input = rn.input[1]
        range_out = rn.output[0]
        max_n = cfg["max"]

        # If limit is float type (Cast outputs float), add Cast to int64 for Slice
        slice_limit = limit_input
        extra_nodes = []
        extra_inits = []
        # Check if the limit producer outputs float
        limit_is_float = False
        for p in model.graph.node:
            if limit_input in p.output:
                for attr in p.attribute:
                    if attr.name == "to" and attr.i == int(TensorProto.FLOAT):
                        limit_is_float = True
                break
        # Also check if it's a model input that's float
        for inp in model.graph.input:
            if inp.name == limit_input and inp.type.tensor_type.elem_type == TensorProto.FLOAT:
                limit_is_float = True

        # Slice requires ends to be 1-D. Range limits may be scalars.
        # Wrap scalar into [N] via Unsqueeze(axes=[0]) if needed.
        unsq_name = f"{rn.name}_unsq_limit"
        unsq_out = f"{unsq_name}_output_0"
        unsq_axes_name = f"{rn.name}_unsq_axes"
        slice_limit_1d = limit_input

        if limit_is_float:
            cast_name = f"{rn.name}_cast_int64"
            cast_out = f"{cast_name}_output_0"
            extra_nodes.append(
                helper.make_node("Cast", [limit_input], [cast_out],
                                 to=int(TensorProto.INT64), name=cast_name)
            )
            # Unsqueeze after Cast
            extra_nodes.append(
                helper.make_node("Unsqueeze", [cast_out, unsq_axes_name], [unsq_out], name=unsq_name)
            )
            extra_inits.append(make_int64_tensor(unsq_axes_name, [0]))
            slice_limit_1d = unsq_out
            print(f"  {cfg['name']}: added Cast(float→int64) + Unsqueeze for Slice limit")
        else:
            # Just Unsqueeze to ensure 1-D
            extra_nodes.append(
                helper.make_node("Unsqueeze", [limit_input, unsq_axes_name], [unsq_out], name=unsq_name)
            )
            extra_inits.append(make_int64_tensor(unsq_axes_name, [0]))
            slice_limit_1d = unsq_out

        # Create static range constant: [0, 1, ..., max_n-1]
        const_name = f"{rn.name}_static_range"
        const_vals = list(range(max_n))
        const_tensor = make_int64_tensor(const_name, const_vals)
        model.graph.initializer.append(const_tensor)

        # Slice: static_range[0:N]
        slice_name = f"{rn.name}_Slice"
        slice_out = f"{slice_name}_output_0"
        starts_name = f"{rn.name}_slice_starts"
        axes_name = f"{rn.name}_slice_axes"
        steps_name = f"{rn.name}_slice_steps"

        slice_node = helper.make_node(
            "Slice",
            [const_name, starts_name, slice_limit_1d, axes_name, steps_name],
            [slice_out],
            name=slice_name,
        )

        model.graph.initializer.append(make_int64_tensor(starts_name, [0]))
        model.graph.initializer.append(make_int64_tensor(axes_name, [0]))
        model.graph.initializer.append(make_int64_tensor(steps_name, [1]))
        for init in extra_inits:
            model.graph.initializer.append(init)

        # If limit was float, we added Cast(int64) — now Slice outputs int64.
        # But downstream may expect float. Insert Cast back to float if needed.
        final_out = slice_out
        if limit_is_float:
            cast_back_name = f"{rn.name}_cast_back_float"
            cast_back_out = f"{cast_back_name}_output_0"
            extra_nodes.append(
                helper.make_node("Cast", [slice_out], [cast_back_out],
                                 to=int(TensorProto.FLOAT), name=cast_back_name)
            )
            final_out = cast_back_out
            print(f"  {cfg['name']}: added Cast(int64→float) for downstream compat")

        # Replace all references to Range output with final output
        for n in model.graph.node:
            for idx, inp in enumerate(n.input):
                if inp == range_out:
                    n.input[idx] = final_out

        # Insert nodes in topological order:
        # 1. Cast(int64) for limit (depends on limit_input)
        # 2. Slice (depends on constant + starts + slice_limit + axes + steps)
        # 3. Cast(float) for output (depends on slice_out)
        pre_nodes = []   # nodes that depend on pre-existing tensors
        post_nodes = []  # nodes that depend on slice_out
        for en in extra_nodes:
            if slice_out in en.input:
                post_nodes.append(en)
            else:
                pre_nodes.append(en)

        insert_pos = len(model.graph.node)
        for i, n in enumerate(model.graph.node):
            if final_out in n.input:
                insert_pos = min(insert_pos, i)
        for pn in reversed(pre_nodes):
            model.graph.node.insert(insert_pos, pn)
        model.graph.node.insert(insert_pos + len(pre_nodes), slice_node)
        for pn in post_nodes:
            model.graph.node.insert(insert_pos + len(pre_nodes) + 1, pn)
        model.graph.node.remove(rn)

        replaced += 1
        print(f"  {cfg['name']}: Range(0,N,1) → Slice([0..{max_n-1}], 0, N)")

    print(f"  Replaced {replaced}/3 Range nodes")


def validate_model(model_path: str, max_mel: int):
    """Run ORT inference to verify the modified model works."""
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    tests = [
        (5, 1.0, 1.0),
        (10, 1.0, 1.0),
        (20, 1.0, 1.0),
        (40, 1.0, 1.0),
        (64, 1.0, 1.0),
        (10, 0.8, 1.5),
    ]

    all_ok = True
    for n_tok, ns_val, ls_val in tests:
        x = np.array([list(range(n_tok))], dtype=np.int64)
        x_len = np.array([n_tok], dtype=np.int64)
        ns = np.array([ns_val], dtype=np.float32)
        ls = np.array([int(ls_val)], dtype=np.int64)  # length_scale is now INT64
        noise = np.random.randn(1, 80, max_mel).astype(np.float32)

        try:
            mel = sess.run(None, {
                "x": x, "x_length": x_len,
                "noise_scale": ns, "length_scale": ls,
                "noise": noise,
            })[0]
            ok = not np.isnan(mel).any() and not np.isinf(mel).any()
            status = "OK" if ok else "FAIL"
            print(f"  tokens={n_tok:3d} ls={ls_val} → mel {mel.shape} {status}")
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"  tokens={n_tok:3d} ERROR: {e}")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Matcha ONNX → TRT-compatible ONNX")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-mel", type=int, default=800)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path}")
    model = onnx.load(str(input_path))
    n_nodes = len(model.graph.node)

    # Fix 1: Replace RandomNormalLike
    fix_random_normal_like(model, max_mel=args.max_mel)

    # Fix 2: Convert length_scale float32 → int64
    fix_length_scale_type(model)

    # Fix 3: Replace dynamic Range nodes with static + Slice
    fix_range_nodes(model, max_tokens=64, max_mel=args.max_mel)

    # Validate ONNX structure
    try:
        onnx.checker.check_model(model)
        print(f"  ONNX checker: valid ({len(model.graph.node)} nodes, was {n_nodes})")
    except onnx.checker.ValidationError as e:
        print(f"  ONNX checker FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"\nSaved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    # Show I/O
    print("\nModel inputs:")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        dtype_name = {1: "FLOAT", 7: "INT64", 6: "INT32"}.get(dtype, str(dtype))
        print(f"  {inp.name}: {shape} ({dtype_name})")
    print("Model outputs:")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")

    if args.validate:
        print("\n=== ORT validation ===")
        ok = validate_model(str(output_path), args.max_mel)
        if not ok:
            print("FAILED", file=sys.stderr)
            sys.exit(1)
        print("All tests passed.")


if __name__ == "__main__":
    main()
