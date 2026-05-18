#!/usr/bin/env python3
"""Build dual-profile BF16 TensorRT engine for cp_unified.onnx.

Profile 0 (prefill): seq_len=2, past_len=0  (used once per codec frame)
Profile 1 (decode):  seq_len=1, past_len=2..17 (used 14 times per frame)

Separating prefill and decode into different profiles eliminates TRT's
shape-change overhead when switching between seq_len=2 and seq_len=1
within a single context.

Usage (on Jetson):
    python3 build_cp_unified_dual.py \
        --onnx /path/to/cp_unified.onnx \
        --output /path/to/cp_unified_bf16.engine
"""
import argparse
import tensorrt as trt


def build(onnx_path: str, engine_path: str):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.BF16)

    n_layers = 5

    # Profile 0: Prefill — seq_len=2, past_len=0
    p0 = builder.create_optimization_profile()
    p0.set_shape("inputs_embeds", (1, 2, 1024), (1, 2, 1024), (1, 2, 1024))
    p0.set_shape("cache_position", (2,), (2,), (2,))
    for i in range(n_layers):
        p0.set_shape(f"past_key_{i}", (1, 8, 0, 128), (1, 8, 0, 128), (1, 8, 0, 128))
        p0.set_shape(f"past_value_{i}", (1, 8, 0, 128), (1, 8, 0, 128), (1, 8, 0, 128))
    config.add_optimization_profile(p0)

    # Profile 1: Decode — seq_len=1, past_len=2..17
    p1 = builder.create_optimization_profile()
    p1.set_shape("inputs_embeds", (1, 1, 1024), (1, 1, 1024), (1, 1, 1024))
    p1.set_shape("cache_position", (1,), (1,), (1,))
    for i in range(n_layers):
        p1.set_shape(f"past_key_{i}", (1, 8, 2, 128), (1, 8, 9, 128), (1, 8, 20, 128))
        p1.set_shape(f"past_value_{i}", (1, 8, 2, 128), (1, 8, 9, 128), (1, 8, 20, 128))
    config.add_optimization_profile(p1)

    print(f"Building dual-profile BF16 engine from {onnx_path} ...")
    print("  Profile 0: prefill (seq=2, past=0)")
    print("  Profile 1: decode  (seq=1, past=2..20)")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    data = bytes(engine_bytes)
    with open(engine_path, "wb") as f:
        f.write(data)
    print(f"Saved: {engine_path} ({len(data) / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="/tmp/cp_unified.onnx")
    p.add_argument("--output", default="/tmp/cp_unified_bf16.engine")
    a = p.parse_args()
    build(a.onnx, a.output)
