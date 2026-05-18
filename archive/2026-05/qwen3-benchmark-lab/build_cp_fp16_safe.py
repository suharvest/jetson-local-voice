#!/usr/bin/env python3
"""Build CP TRT engine with FP16 + selective FP32 for attention.

Uses OBEY_PRECISION_CONSTRAINTS with careful layer type filtering.
Only sets precision on layers that SUPPORT it (MatMul, Softmax, ElementWise).
Skips Shape, Constant, Gather, Concatenation, etc.

Run on Jetson host:
    docker stop $(docker ps -q)
    python3 /tmp/build_cp_fp16_safe.py
"""
import tensorrt as trt
import numpy as np
import os

ONNX_PATH = "/tmp/code_predictor.onnx"  # FP32 sherpa-style CP
ENGINE_PATH = "/tmp/cp_sherpa_fp16safe.engine"

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

print(f"Parsing {ONNX_PATH}...")
with open(ONNX_PATH, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f"  Error: {parser.get_error(i)}")
        raise RuntimeError("Parse failed")

n_layers = network.num_layers
print(f"Network: {n_layers} layers")

# Configure
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024)
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

# Dynamic shapes
profile = builder.create_optimization_profile()
profile.set_shape("context", (1, 2, 1024), (1, 9, 1024), (1, 17, 1024))
profile.set_shape("gen_step", (1,), (1,), (1,))
config.add_optimization_profile(profile)

# Only these layer types can safely have precision set
COMPUTE_TYPES = {
    trt.LayerType.MATRIX_MULTIPLY,
    trt.LayerType.SOFTMAX,
    trt.LayerType.ELEMENTWISE,
    trt.LayerType.UNARY,
    trt.LayerType.REDUCE,
    trt.LayerType.ACTIVATION,
    trt.LayerType.CONVOLUTION,
    trt.LayerType.DECONVOLUTION,
    trt.LayerType.SCALE,
    trt.LayerType.NORMALIZATION,
    trt.LayerType.POOLING,
}

# Force attention-related compute layers to FP32
n_fp32 = 0
n_fp16 = 0
n_skip = 0
for i in range(n_layers):
    layer = network.get_layer(i)
    name = layer.name.lower()
    lt = layer.type

    # Skip layers that don't support precision setting
    if lt not in COMPUTE_TYPES:
        n_skip += 1
        continue

    # Determine if this is an attention layer (needs FP32)
    is_attention = "self_attn" in name
    is_norm = "layernorm" in name or "layer_norm" in name or "norm" in name

    if True:
        if is_attention or is_norm:
            # Force FP32 for attention and normalization
            try:
                layer.precision = trt.float32
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float32)
                n_fp32 += 1
            except Exception as e:
                print(f"  Warning: can't set FP32 on {layer.name} ({lt}): {e}")
                n_skip += 1
        else:
            # FFN and other compute: let TRT use FP16
            try:
                layer.precision = trt.float16
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float16)
                n_fp16 += 1
            except Exception:
                n_skip += 1
    else:
        # Unknown type, skip
        n_skip += 1

print(f"FP32 (attention+norm): {n_fp32}")
print(f"FP16 (FFN+other): {n_fp16}")
print(f"Skipped: {n_skip}")

# Build
print("Building engine...")
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    raise RuntimeError("Build failed")

with open(ENGINE_PATH, "wb") as f:
    f.write(serialized)

size_mb = os.path.getsize(ENGINE_PATH) / 1024 / 1024
print(f"\nEngine saved: {ENGINE_PATH} ({size_mb:.0f}MB)")
