#!/usr/bin/env python3
"""Build BF16 TensorRT engine for Qwen3-TTS code predictor.

Key: config.set_flag(trt.BuilderFlag.BF16) — solves FP16 NaN issues.
BF16 has same exponent range as FP32 (max=3.4e38), so no overflow.

Usage:
    python3 build_cp_bf16.py --onnx /tmp/code_predictor.onnx --output /tmp/cp_bf16.engine
"""
import argparse
import tensorrt as trt

def build(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

    # BF16 — the key line
    config.set_flag(trt.BuilderFlag.BF16)

    # Dynamic shapes: context [1, 2-17, 1024], gen_step [1]
    profile = builder.create_optimization_profile()
    profile.set_shape("context", (1, 2, 1024), (1, 9, 1024), (1, 17, 1024))
    profile.set_shape("gen_step", (1,), (1,), (1,))
    config.add_optimization_profile(profile)

    print(f"Building BF16 engine from {onnx_path} ...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    data = bytes(engine_bytes)
    with open(engine_path, "wb") as f:
        f.write(data)
    print(f"Saved: {engine_path} ({len(data) / 1e6:.0f} MB)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="/tmp/code_predictor.onnx")
    p.add_argument("--output", default="/tmp/cp_bf16.engine")
    a = p.parse_args()
    build(a.onnx, a.output)
