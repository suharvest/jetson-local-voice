#!/usr/bin/env python3
"""Fix Matcha TTS acoustic model ONNX for RKNN conversion.

This script applies a 4-step ONNX graph surgery pipeline to make
model-steps-3.onnx (from sherpa-onnx matcha-icefall-zh-baker) compatible
with RKNN conversion:

Step 1: onnxsim with fixed shapes (SEQ_LEN=256, x_length=[1])
  - Resolves symbolic dims N and L
  - Folds static shape computations

Step 2: Replace Range nodes with Constants
  - Range nodes for positional encodings are replaced with baked constant tensors
  - Values extracted by running ORT with x_length=86

Step 3: Fix Slice_2 dynamic ends
  - /Unsqueeze_11_output_0 (mel_frames=769) is the dynamic ends input
  - Replaced with constant [769] initializer
  - Unsqueeze_11 node removed (its output is now statically provided)

Step 4: Replace Ceil with Neg(Floor(Neg(x)))
  - RKNN runtime doesn't support Ceil op on CPU
  - Mathematically equivalent: ceil(x) = -floor(-x)

Step 5: Replace RandomNormalLike with fixed constant noise tensor
  - ODE noise is baked at compile time (deterministic, seed=42)
  - RKNN runtime doesn't support RandomNormalLike on CPU

After surgery, RKNN conversion succeeds without "Unsupport CPU op" errors.

Usage:
  python fix_matcha_rknn.py --input model-steps-3.onnx --output model-steps-3-fixed.onnx

Then convert with RKNN toolkit2:
  from rknn.api import RKNN
  rknn = RKNN()
  rknn.config(target_platform='rk3576', optimization_level=0)
  rknn.load_onnx(model='model-steps-3-fixed.onnx')
  rknn.build(do_quantization=False)
  rknn.export_rknn('matcha-acoustic-fp16.rknn')

Note: The vocoder (vocos-vocoder.rknn) expects mel input of shape [1,80,2048],
pad the mel output from the acoustic model before feeding to vocoder.
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
import onnxruntime as ort

# Fixed parameters matching the existing sherpa-onnx RKNN model
SEQ_LEN = 256   # Max phoneme sequence length (padded)
X_LEN = 86      # Reference x_length (gives mel_frames=769, close to original 768)


def load_and_simplify(input_path: str) -> onnx.ModelProto:
    """Step 1: Load and simplify with onnxsim."""
    import onnxsim
    model = onnx.load(input_path)
    print(f"  Original nodes: {len(model.graph.node)}")
    simplified, ok = onnxsim.simplify(
        model,
        overwrite_input_shapes={
            'x': [1, SEQ_LEN],
            'x_length': [1],
            'noise_scale': [1],
            'length_scale': [1],
        },
    )
    print(f"  Simplified: ok={ok}, nodes={len(simplified.graph.node)}")
    return simplified


def probe_ort(model: onnx.ModelProto, extra_outputs: list[str], test_inputs: dict) -> dict:
    """Run ORT inference and capture specific intermediate tensor values."""
    m_probe = onnx.ModelProto()
    m_probe.CopyFrom(model)
    for name in extra_outputs:
        vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        m_probe.graph.output.append(vi)

    tmp = tempfile.mktemp(suffix='.onnx')
    onnx.save(m_probe, tmp)
    sess = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])
    all_out = sess.run(None, test_inputs)
    os.unlink(tmp)

    all_names = [o.name for o in m_probe.graph.output]
    return {name: val for name, val in zip(all_names, all_out)}


def fix_range_nodes(model: onnx.ModelProto, test_inputs: dict) -> onnx.ModelProto:
    """Step 2: Replace Range nodes with constant tensors."""
    range_nodes = [n for n in model.graph.node if n.op_type == 'Range']
    if not range_nodes:
        print("  No Range nodes found")
        return model

    # Probe to get Range output values
    m_probe = onnx.ModelProto()
    m_probe.CopyFrom(model)
    for rn in range_nodes:
        for out in rn.output:
            vi = helper.make_tensor_value_info(out, TensorProto.INT64, None)
            m_probe.graph.output.append(vi)

    tmp = tempfile.mktemp(suffix='.onnx')
    onnx.save(m_probe, tmp)
    sess = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])
    all_out = sess.run(None, test_inputs)
    os.unlink(tmp)
    all_names = [o.name for o in m_probe.graph.output]
    values = {n: v for n, v in zip(all_names, all_out) if any(rn.output[0] == n for rn in range_nodes)}

    range_outputs = set(values.keys())
    new_nodes = []
    for i, n in enumerate(model.graph.node):
        if n.op_type == 'Range' and n.output[0] in range_outputs:
            val = values[n.output[0]]
            arr = val.astype(np.float32) if val.dtype == np.float32 else val.astype(np.int64)
            const_node = helper.make_node(
                'Constant', inputs=[], outputs=[n.output[0]],
                name=f'const_range_{i}',
                value=numpy_helper.from_array(arr, name=n.output[0])
            )
            new_nodes.insert(0, const_node)
            print(f"  Replaced Range {n.output[0]}: shape={arr.shape}")
        else:
            new_nodes.append(n)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


def fix_dynamic_slice_ends(model: onnx.ModelProto, test_inputs: dict) -> onnx.ModelProto:
    """Step 3: Fix Slice nodes with dynamic index inputs."""
    init_names = {init.name for init in model.graph.initializer}
    const_names = {out for n in model.graph.node if n.op_type == 'Constant' for out in n.output}
    static_names = init_names | const_names
    graph_input_names = {inp.name for inp in model.graph.input}

    # Find Slice nodes with dynamic INDEX inputs (not data)
    dynamic_tensors = {}
    for n in model.graph.node:
        if n.op_type == 'Slice':
            for i in range(1, len(n.input)):
                inp = n.input[i]
                if inp and inp not in static_names and inp not in graph_input_names:
                    dynamic_tensors[inp] = None

    if not dynamic_tensors:
        print("  No dynamic Slice index inputs found")
        return model

    # Probe to get values
    m_probe = onnx.ModelProto()
    m_probe.CopyFrom(model)
    for name in dynamic_tensors:
        vi = helper.make_tensor_value_info(name, TensorProto.INT64, None)
        m_probe.graph.output.append(vi)
    tmp = tempfile.mktemp(suffix='.onnx')
    onnx.save(m_probe, tmp)
    sess = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])
    all_out = sess.run(None, test_inputs)
    os.unlink(tmp)
    for name, val in zip([o.name for o in m_probe.graph.output], all_out):
        if name in dynamic_tensors:
            dynamic_tensors[name] = val
            print(f"  {name} = {val}")

    # For each dynamic tensor, find its single producing node (if safe to remove)
    out_to_node = {out: n for n in model.graph.node for out in n.output}
    tensor_consumers = {}
    for n in model.graph.node:
        for inp in n.input:
            if inp:
                tensor_consumers.setdefault(inp, []).append(n)

    nodes_to_remove = set()
    for name, val in dynamic_tensors.items():
        if val is None:
            continue
        prod = out_to_node.get(name)
        consumers = tensor_consumers.get(name, [])
        # Add constant initializer
        model.graph.initializer.append(numpy_helper.from_array(val.astype(np.int64), name=name))
        # Remove producing node only if safe
        if (prod and len(prod.output) == 1
                and all(c.op_type in ('Slice',) for c in consumers)):
            nodes_to_remove.add(id(prod))
            print(f"  Removed {prod.op_type} {prod.name}")

    if nodes_to_remove:
        new_nodes = [n for n in model.graph.node if id(n) not in nodes_to_remove]
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)

    return model


def fix_ceil_ops(model: onnx.ModelProto) -> onnx.ModelProto:
    """Step 4: Replace Ceil(x) with Neg(Floor(Neg(x)))."""
    ceil_nodes = [n for n in model.graph.node if n.op_type == 'Ceil']
    if not ceil_nodes:
        print("  No Ceil nodes found")
        return model

    new_nodes = []
    for i, n in enumerate(model.graph.node):
        if n.op_type == 'Ceil':
            x_in, y_out = n.input[0], n.output[0]
            neg1 = f'{y_out}__neg1'
            flr = f'{y_out}__floor'
            new_nodes.extend([
                helper.make_node('Neg', [x_in], [neg1], name=f'{n.name}_neg1'),
                helper.make_node('Floor', [neg1], [flr], name=f'{n.name}_floor'),
                helper.make_node('Neg', [flr], [y_out], name=f'{n.name}_neg2'),
            ])
            print(f"  Replaced Ceil {n.name}")
        else:
            new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


def fix_random_normal_like(model: onnx.ModelProto, test_inputs: dict) -> onnx.ModelProto:
    """Step 5: Replace RandomNormalLike with fixed constant noise (seed=42)."""
    rn_nodes = [n for n in model.graph.node if n.op_type == 'RandomNormalLike']
    if not rn_nodes:
        print("  No RandomNormalLike nodes found")
        return model

    # Probe to get output shapes
    m_probe = onnx.ModelProto()
    m_probe.CopyFrom(model)
    for rn in rn_nodes:
        vi = helper.make_tensor_value_info(rn.output[0], TensorProto.FLOAT, None)
        m_probe.graph.output.append(vi)
    tmp = tempfile.mktemp(suffix='.onnx')
    onnx.save(m_probe, tmp)
    sess = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])
    all_out = sess.run(None, test_inputs)
    os.unlink(tmp)
    shapes = {name: val.shape for name, val in zip([o.name for o in m_probe.graph.output], all_out)
              if any(rn.output[0] == name for rn in rn_nodes)}

    rng = np.random.default_rng(42)
    nodes_to_remove = set()
    const_nodes = []
    for rn in rn_nodes:
        name = rn.output[0]
        if name in shapes:
            noise = rng.standard_normal(size=shapes[name]).astype(np.float32)
            const_nodes.append(helper.make_node(
                'Constant', inputs=[], outputs=[name],
                name=f'const_{rn.name}',
                value=numpy_helper.from_array(noise, name=name)
            ))
            nodes_to_remove.add(id(rn))
            print(f"  Replaced RandomNormalLike {name}: shape={shapes[name]}")

    new_nodes = [n for n in model.graph.node if id(n) not in nodes_to_remove]
    for cn in const_nodes:
        new_nodes.insert(0, cn)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='~/matcha-data/model-steps-3.onnx')
    parser.add_argument('--output', default='~/matcha-data/model-steps-3-rknn-ready.onnx')
    parser.add_argument('--x-len', type=int, default=X_LEN,
                        help=f'Reference x_length for baking (default: {X_LEN} -> mel_frames=769)')
    args = parser.parse_args()

    input_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)
    x_len = args.x_len

    print(f"=== Matcha ONNX RKNN Fix Pipeline ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"SEQ_LEN={SEQ_LEN}, X_LEN={x_len}")

    # Reference test inputs
    x = np.zeros((1, SEQ_LEN), dtype=np.int64)
    x[0, :x_len] = (np.arange(x_len) % 100) + 1
    test_inputs = {
        'x': x,
        'x_length': np.array([x_len], dtype=np.int64),
        'noise_scale': np.array([0.667], dtype=np.float32),
        'length_scale': np.array([1.0], dtype=np.float32),
    }

    # Step 1: onnxsim
    print("\n[1/5] onnxsim simplification...")
    model = load_and_simplify(input_path)

    # Step 2: Range nodes
    print("\n[2/5] Replacing Range nodes...")
    model = fix_range_nodes(model, test_inputs)

    # Step 3: Dynamic Slice ends
    print("\n[3/5] Fixing dynamic Slice index inputs...")
    model = fix_dynamic_slice_ends(model, test_inputs)

    # Step 4: Ceil ops
    print("\n[4/5] Replacing Ceil ops...")
    model = fix_ceil_ops(model)

    # Step 5: RandomNormalLike
    print("\n[5/5] Replacing RandomNormalLike...")
    model = fix_random_normal_like(model, test_inputs)

    # Shape inference
    print("\nRunning shape inference...")
    try:
        model = onnx.shape_inference.infer_shapes(model)
        print("  OK")
    except Exception as e:
        print(f"  Warning: {e}")

    # ORT verify
    print("\nVerifying with ORT...")
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
        out = sess.run(None, test_inputs)
        mel = out[0]
        print(f"  mel shape: {mel.shape}")
        print(f"  mel RMS: {float(np.sqrt(np.mean(mel**2))):.4f}")
        print("  ORT OK")
    except Exception as e:
        print(f"  ORT FAIL: {e}")
        sys.exit(1)

    # Check for remaining problematic ops
    for op in ['Range', 'Ceil', 'RandomNormalLike']:
        n = sum(1 for node in model.graph.node if node.op_type == op)
        if n:
            print(f"  WARNING: {n} {op} nodes remain")

    # Save
    onnx.save(model, output_path)
    sz = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({sz:.1f} MB)")
    print("Done.")


if __name__ == '__main__':
    main()
