#!/usr/bin/env python3
"""Build CP TRT engine with mixed precision: FP16 matmul + FP32 layernorm/softmax.

Run on Jetson host with containers stopped for max memory:
    docker stop $(docker ps -q)
    python3 /tmp/build_cp_mixed.py
"""
import tensorrt as trt
import numpy as np
import os

ONNX_PATH = "/tmp/code_predictor.onnx"
ENGINE_PATH = "/tmp/cp_sherpa_mixed.engine"

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

print(f"Parsing {ONNX_PATH}...")
with open(ONNX_PATH, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f"  Parse error: {parser.get_error(i)}")
        raise RuntimeError("ONNX parse failed")

print(f"Network: {network.num_layers} layers, {network.num_inputs} inputs, {network.num_outputs} outputs")

# Configure builder
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024)  # 1GB
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)  # Must obey per-layer precision

# Set dynamic shapes
profile = builder.create_optimization_profile()
profile.set_shape("context", (1, 2, 1024), (1, 9, 1024), (1, 17, 1024))
profile.set_shape("gen_step", (1,), (1,), (1,))
config.add_optimization_profile(profile)

# Mixed precision: only force MatMul and Softmax in attention to FP32
# These are the layers that cause overflow; FFN MatMul is safe
SAFE_TYPES = {trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.SOFTMAX,
              trt.LayerType.REDUCE, trt.LayerType.UNARY, trt.LayerType.ELEMENTWISE}
n_forced = 0
n_total = network.num_layers
for i in range(n_total):
    layer = network.get_layer(i)
    # Only force compute layers (not Shape, Constant, Gather, etc.)
    if layer.type not in SAFE_TYPES:
        continue
    name_lower = layer.name.lower()
    force_fp32 = False
    # Force all Softmax
    if layer.type == trt.LayerType.SOFTMAX:
        force_fp32 = True
    # Force attention MatMul (QK^T) - identified by "self_attn" in name
    elif layer.type == trt.LayerType.MATRIX_MULTIPLY and "self_attn" in name_lower:
        force_fp32 = True
    # Force LayerNorm ops (Reduce, Pow, Sqrt)
    elif any(kw in name_lower for kw in ["layernorm", "layer_norm"]):
        force_fp32 = True

    if force_fp32:
        layer.precision = trt.float32
        for j in range(layer.num_outputs):
            layer.set_output_type(j, trt.float32)
        n_forced += 1

print(f"Forced {n_forced}/{n_total} layers to FP32 (attention + norm), rest FP16")

# Build engine
print("Building engine (this may take 1-2 minutes)...")
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    raise RuntimeError("Engine build failed")

with open(ENGINE_PATH, "wb") as f:
    f.write(serialized)

size_mb = os.path.getsize(ENGINE_PATH) / 1024 / 1024
print(f"Engine saved: {ENGINE_PATH} ({size_mb:.0f}MB)")

# Quick benchmark
print("\nBenchmarking...")
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized)
context = engine.create_execution_context()

import pycuda.driver as cuda
import pycuda.autoinit

stream = cuda.Stream()
ctx_np = np.random.randn(1, 9, 1024).astype(np.float32)
gs_np = np.array([4], dtype=np.int64)

# Allocate buffers
d_ctx = cuda.mem_alloc(ctx_np.nbytes)
d_gs = cuda.mem_alloc(gs_np.nbytes)
out_np = np.empty((1, 1, 2048), dtype=np.float32)
d_out = cuda.mem_alloc(out_np.nbytes)

cuda.memcpy_htod(d_ctx, ctx_np)
cuda.memcpy_htod(d_gs, gs_np)

context.set_input_shape("context", ctx_np.shape)
context.set_input_shape("gen_step", gs_np.shape)
context.set_tensor_address("context", int(d_ctx))
context.set_tensor_address("gen_step", int(d_gs))
context.set_tensor_address("logits", int(d_out))

# Warm up
for _ in range(5):
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()

# Benchmark
import time
times = []
for _ in range(50):
    t0 = time.perf_counter()
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

cuda.memcpy_dtoh(out_np, d_out)
print(f"Latency: {np.mean(times):.2f}ms avg (min={np.min(times):.2f}, max={np.max(times):.2f})")
print(f"NaN check: {np.isnan(out_np).sum()} NaN values, mean={out_np.mean():.4f}")
print(f"15 steps theoretical: {np.mean(times)*15:.0f}ms")
