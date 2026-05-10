#!/usr/bin/env python3
"""Build a TensorRT engine for Qwen3-TTS native CodePredictor ONNX.

Run this on the target Jetson, not on a different Orin SKU, so TensorRT tactic
selection matches the deployment device.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


SIDEcar_FILES = (
    "config.json",
    "cp_embed_fp32.bin",
    "codec_embeddings.safetensors",
    "lm_heads.safetensors",
    "small_to_mtp_projection.safetensors",
)


def link_sidecars(output_dir: Path, sidecar_dir: Path) -> None:
    for name in SIDEcar_FILES:
        src = sidecar_dir / name
        dst = output_dir / name
        if not src.exists() or dst.exists():
            continue
        os.symlink(src, dst)
    llm = output_dir / "llm.engine"
    if not llm.exists():
        os.symlink("qwen3_tts_cp.engine", llm)


def build_engine(args: argparse.Namespace) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(args.onnx.read_bytes()):
        for i in range(parser.num_errors):
            print(f"ONNX parse error: {parser.get_error(i)}")
        raise RuntimeError("ONNX parse failed")

    if args.bf16_io:
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            if tensor.name.startswith(("past_key_", "past_value_")):
                tensor.dtype = trt.DataType.BF16
        for i in range(network.num_outputs):
            tensor = network.get_output(i)
            if tensor.name == "logits" or tensor.name.startswith(("new_past_key_", "new_past_value_")):
                tensor.dtype = trt.DataType.BF16

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb << 20)
    config.set_flag(trt.BuilderFlag.BF16)
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if args.max_aux_streams is not None and hasattr(config, "max_aux_streams"):
        config.max_aux_streams = args.max_aux_streams
    if hasattr(config, "profiling_verbosity"):
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if args.builder_opt_level is not None and hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = args.builder_opt_level

    profile = builder.create_optimization_profile()
    profile.set_shape("inputs_embeds", (1, 1, 1024), (1, 1, 1024), (1, 2, 1024))
    profile.set_shape("cache_position", (1,), (1,), (2,))
    profile.set_shape("gen_step", (), (), ())
    for i in range(args.layers):
        profile.set_shape(f"past_key_{i}", (1, 8, 0, 128), (1, 8, args.opt_past, 128), (1, 8, args.max_past, 128))
        profile.set_shape(f"past_value_{i}", (1, 8, 0, 128), (1, 8, args.opt_past, 128), (1, 8, args.max_past, 128))
    config.add_optimization_profile(profile)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine_path = args.output_dir / "qwen3_tts_cp.engine"
    print(
        "Building Qwen3-TTS CP engine:",
        f"onnx={args.onnx}",
        f"output={engine_path}",
        f"bf16_io={args.bf16_io}",
        f"workspace_mb={args.workspace_mb}",
        f"opt_past={args.opt_past}",
        f"max_past={args.max_past}",
        f"builder_opt_level={args.builder_opt_level}",
        f"max_aux_streams={args.max_aux_streams}",
    )
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("Engine build failed")
    engine_path.write_bytes(bytes(engine))
    if args.sidecar_dir:
        link_sidecars(args.output_dir, args.sidecar_dir)
    print(f"Saved {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path)
    parser.add_argument("--workspace-mb", type=int, default=512)
    parser.add_argument("--builder-opt-level", type=int, default=3)
    parser.add_argument("--opt-past", type=int, default=8)
    parser.add_argument("--max-past", type=int, default=20)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--max-aux-streams", type=int)
    parser.add_argument("--bf16-io", action="store_true", help="Use BF16 for CP KV inputs, logits, and new KV outputs.")
    parser.add_argument("--fp16", action="store_true", help="Also enable FP16 tactics; not quality-safe by default.")
    args = parser.parse_args()
    build_engine(args)


if __name__ == "__main__":
    main()
