"""Decompose ORT-specific fused ops into standard ONNX ops for TensorRT compatibility."""

import onnx
from onnx import helper, numpy_helper
import numpy as np
from collections import Counter

INPUT = "/opt/models/f5-tts-onnx/F5_Transformer_fixed.onnx"
OUTPUT = "/opt/models/f5-tts-onnx/F5_Transformer_trt.onnx"

m = onnx.load(INPUT)
new_nodes = []
new_inits = []
idx = 0

for node in m.graph.node:
    if node.op_type == "FastGelu" and node.domain == "com.microsoft":
        x = node.input[0]
        y = node.output[0]
        p = f"_fg{idx}_"
        idx += 1

        new_inits.append(numpy_helper.from_array(np.array(0.044715, dtype=np.float16), p+"c1"))
        new_inits.append(numpy_helper.from_array(np.array(0.7978845608028654, dtype=np.float16), p+"c2"))
        new_inits.append(numpy_helper.from_array(np.array(0.5, dtype=np.float16), p+"c3"))
        new_inits.append(numpy_helper.from_array(np.array(1.0, dtype=np.float16), p+"c4"))

        new_nodes.append(helper.make_node("Mul", [x, x], [p+"x2"]))
        new_nodes.append(helper.make_node("Mul", [p+"x2", x], [p+"x3"]))
        new_nodes.append(helper.make_node("Mul", [p+"x3", p+"c1"], [p+"cx3"]))
        new_nodes.append(helper.make_node("Add", [x, p+"cx3"], [p+"sum"]))
        new_nodes.append(helper.make_node("Mul", [p+"sum", p+"c2"], [p+"inner"]))
        new_nodes.append(helper.make_node("Tanh", [p+"inner"], [p+"tanh"]))
        new_nodes.append(helper.make_node("Add", [p+"tanh", p+"c4"], [p+"add1"]))
        new_nodes.append(helper.make_node("Mul", [p+"add1", p+"c3"], [p+"sig"]))
        new_nodes.append(helper.make_node("Mul", [x, p+"sig"], [y]))

    elif node.op_type == "QuickGelu" and node.domain == "com.microsoft":
        x = node.input[0]
        y = node.output[0]
        p = f"_qg{idx}_"
        idx += 1

        new_inits.append(numpy_helper.from_array(np.array(1.702, dtype=np.float16), p+"alpha"))
        new_nodes.append(helper.make_node("Mul", [x, p+"alpha"], [p+"ax"]))
        new_nodes.append(helper.make_node("Sigmoid", [p+"ax"], [p+"sig"]))
        new_nodes.append(helper.make_node("Mul", [x, p+"sig"], [y]))

    elif node.op_type == "SkipLayerNormalization" and node.domain == "com.microsoft":
        x = node.input[0]
        skip = node.input[1]
        gamma = node.input[2]
        beta = node.input[3] if len(node.input) > 3 else None
        y = node.output[0]
        p = f"_sln{idx}_"
        idx += 1

        skip_sum = p + "skip_sum"
        new_nodes.append(helper.make_node("Add", [x, skip], [skip_sum]))

        eps = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps = attr.f

        ln_inputs = [skip_sum, gamma]
        if beta:
            ln_inputs.append(beta)
        new_nodes.append(helper.make_node("LayerNormalization", ln_inputs, [y], epsilon=eps, axis=-1))

        # Handle skip_sum output (index 3)
        if len(node.output) > 3 and node.output[3] and not node.output[3].startswith("_unused_"):
            new_nodes.append(helper.make_node("Identity", [skip_sum], [node.output[3]]))
    else:
        new_nodes.append(node)

del m.graph.node[:]
m.graph.node.extend(new_nodes)
m.graph.initializer.extend(new_inits)

onnx.save(m, OUTPUT)

# Verify
m2 = onnx.load(OUTPUT)
ops = Counter(n.op_type for n in m2.graph.node)
custom = {op for op in ops if any(n.domain not in ("", "ai.onnx") for n in m2.graph.node if n.op_type == op)}
print(f"Saved: {OUTPUT}")
print(f"Nodes: {len(m2.graph.node)} (was 1258)")
print(f"Custom ops remaining: {custom if custom else 'NONE'}")
print(f"Top ops: {ops.most_common(15)}")
