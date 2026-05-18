#!/usr/bin/env python3
"""Build BF16 TensorRT engine for cp_single_head.onnx.

Single-head CP: takes gen_step input, only computes 1 lm_head per step.
Output: logits [1, vocab] instead of logits_all [15, vocab].

Usage (on Jetson):
    python3 build_cp_single_head.py \
        --onnx /tmp/cp_single_head.onnx \
        --output /tmp/cp_single_head_bf16.engine
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

    # Single profile: seq_len 1-2, past_len 0-20, gen_step 0-14
    p = builder.create_optimization_profile()
    p.set_shape("inputs_embeds", (1, 1, 1024), (1, 1, 1024), (1, 2, 1024))
    p.set_shape("cache_position", (1,), (1,), (2,))
    # gen_step is a scalar (0-D tensor) — TRT needs min/opt/max as 0-D
    p.set_shape_input("gen_step", (0,), (7,), (14,))
    for i in range(n_layers):
        p.set_shape(f"past_key_{i}",
                    (1, 8, 0, 128), (1, 8, 8, 128), (1, 8, 20, 128))
        p.set_shape(f"past_value_{i}",
                    (1, 8, 0, 128), (1, 8, 8, 128), (1, 8, 20, 128))
    config.add_optimization_profile(p)

    print(f"Building BF16 engine from {onnx_path} ...")
    print("  seq_len: 1-2, past_len: 0-20, gen_step: 0-14")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Engine build failed")

    data = bytes(engine_bytes)
    with open(engine_path, "wb") as f:
        f.write(data)
    print(f"Saved: {engine_path} ({len(data) / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="/tmp/cp_single_head.onnx")
    p.add_argument("--output", default="/tmp/cp_single_head_bf16.engine")
    a = p.parse_args()
    build(a.onnx, a.output)
