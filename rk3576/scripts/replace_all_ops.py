"""Replace Snake activation ops + InstanceNorm with custom CPU ops in estimator ONNX.

1:1 op_type replacement — no graph topology change, no shape inference issues.

Snake chain per transformer block: Mul_1 → Sin → Pow → Mul_2 → Add_1
InstanceNorm: in ResnetBlock1D blocks
"""
import os, numpy as np
import onnx
from onnx import helper

def replace_ops(model_path, output_path):
    model = onnx.load(model_path)

    replacements = {
        'Sin': ('CstSin', 0),
        'Mul': ('CstMul', 0),  # only specific Muls
        'Pow': ('CstPow', 0),
        'Add': ('CstAdd', 0),
        'InstanceNormalization': ('CstInstanceNorm', 0),
    }

    count = {}
    for n in model.graph.node:
        # Snake activation ops in ff/net.0
        if "ff/net.0" in n.name:
            if n.op_type == "Sin":
                n.op_type = "CstSin"; n.domain = "custom"
                count['CstSin'] = count.get('CstSin', 0) + 1
            elif n.op_type == "Mul" and ("Mul_1" in n.name or "Mul_2" in n.name):
                n.op_type = "CstMul"; n.domain = "custom"
                count['CstMul'] = count.get('CstMul', 0) + 1
            elif n.op_type == "Pow":
                n.op_type = "CstPow"; n.domain = "custom"
                count['CstPow'] = count.get('CstPow', 0) + 1
            elif n.op_type == "Add" and "Add_1" in n.name:
                n.op_type = "CstAdd"; n.domain = "custom"
                count['CstAdd'] = count.get('CstAdd', 0) + 1

        # InstanceNormalization in ResnetBlock1D
        if n.op_type == "InstanceNormalization":
            n.op_type = "CstInstanceNorm"; n.domain = "custom"
            count['CstInstanceNorm'] = count.get('CstInstanceNorm', 0) + 1

    # Add custom domain
    has_custom = any(op.domain == 'custom' for op in model.opset_import)
    if not has_custom:
        custom_opset = onnx.OperatorSetIdProto()
        custom_opset.domain = 'custom'
        custom_opset.version = 1
        model.opset_import.append(custom_opset)

    print(f"Replacements: {count}")
    print(f"Total custom ops: {sum(count.values())}")

    onnx.save(model, output_path)
    print(f"Saved: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f}MB)")


def build_rknn(onnx_path, rknn_path):
    from rknn.api import RKNN

    # Register all custom ops for toolkit shape inference
    class CstSin:
        op_type = 'CstSin'
        def shape_infer(self, node, in_shapes, in_dtypes):
            return in_shapes.copy(), in_dtypes.copy()
        def compute(self, node, inputs):
            return [np.sin(inputs[0].astype(np.float32))]

    class CstMul:
        op_type = 'CstMul'
        def shape_infer(self, node, in_shapes, in_dtypes):
            # Output shape = broadcast of inputs
            return [in_shapes[0]], [in_dtypes[0]]
        def compute(self, node, inputs):
            return [(inputs[0].astype(np.float32) * inputs[1].astype(np.float32))]

    class CstPow:
        op_type = 'CstPow'
        def shape_infer(self, node, in_shapes, in_dtypes):
            return [in_shapes[0]], [in_dtypes[0]]
        def compute(self, node, inputs):
            return [np.power(inputs[0].astype(np.float32), inputs[1].astype(np.float32))]

    class CstAdd:
        op_type = 'CstAdd'
        def shape_infer(self, node, in_shapes, in_dtypes):
            return [in_shapes[0]], [in_dtypes[0]]
        def compute(self, node, inputs):
            return [(inputs[0].astype(np.float32) + inputs[1].astype(np.float32))]

    class CstInstanceNorm:
        op_type = 'CstInstanceNorm'
        def shape_infer(self, node, in_shapes, in_dtypes):
            return [in_shapes[0]], [in_dtypes[0]]
        def compute(self, node, inputs):
            from rknn.api.custom_op import get_node_attr
            x, scale, bias = inputs[0], inputs[1], inputs[2]
            eps = get_node_attr(node, 'epsilon') or 1e-5
            # InstanceNorm: per channel, across spatial
            N, C = x.shape[0], x.shape[1]
            spatial = x.reshape(N, C, -1)
            mean = spatial.mean(axis=2, keepdims=True)
            var = spatial.var(axis=2, keepdims=True)
            normed = (spatial - mean) / np.sqrt(var + eps)
            # scale/bias shape: [C]
            s = scale.reshape(1, C, 1)
            b = bias.reshape(1, C, 1)
            result = (normed * s + b).reshape(x.shape)
            return [result.astype(np.float32)]

    input_names = ["z", "mu", "mask"] + [f"time_emb_{i}" for i in range(6)]
    input_sizes = [[1,80,600], [1,80,600], [1,1,600]] + [[1,256,1]]*6

    rknn = RKNN(verbose=False)
    rknn.reg_custom_op(CstSin())
    rknn.reg_custom_op(CstMul())
    rknn.reg_custom_op(CstPow())
    rknn.reg_custom_op(CstAdd())
    rknn.reg_custom_op(CstInstanceNorm())

    rknn.config(target_platform="rk3576", optimization_level=3, float_dtype="float16")
    ret = rknn.load_onnx(model=onnx_path, inputs=input_names, input_size_list=input_sizes)
    print(f"load: {ret}")
    if ret != 0: rknn.release(); return

    ret = rknn.build(do_quantization=False)
    print(f"build: {ret}")
    if ret != 0: rknn.release(); return

    rknn.export_rknn(rknn_path)
    print(f"exported: {os.path.getsize(rknn_path)/1024/1024:.1f}MB")
    rknn.release()


if __name__ == "__main__":
    split_dir = os.path.expanduser("~/matcha-data/split")
    out_dir = os.path.expanduser("~/matcha-rknn-rebuild/all_custom")
    os.makedirs(out_dir, exist_ok=True)

    custom_onnx = os.path.join(out_dir, "matcha-estimator-allcustom.onnx")
    custom_rknn = os.path.join(out_dir, "matcha-estimator-allcustom.rknn")

    replace_ops(os.path.join(split_dir, "matcha-estimator.onnx"), custom_onnx)
    build_rknn(custom_onnx, custom_rknn)
