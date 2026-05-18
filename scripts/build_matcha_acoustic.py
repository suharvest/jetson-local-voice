#!/usr/bin/env python3
"""Build Matcha acoustic TRT engine using Python API.

Workaround for trtexec shape-tensor auto-detection bug: TRT 10.3 incorrectly marks
the float32 ``length_scale`` input as a shape tensor because it flows into a Cast->Range
path. The Python API lets us explicitly control this.

Usage (on Jetson):
    python3 build_matcha_acoustic.py \
        --onnx /tmp/model-steps-3-trt.onnx \
        --output /tmp/matcha_acoustic_fp16.engine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt


def build_engine(onnx_path: str, output_path: str, fp16: bool = True,
                 workspace_mb: int = 1024):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Build profile with explicit shape tensor annotations.
    # CRITICAL: TRT auto-detects length_scale as a shape tensor (float->Cast->Range),
    # but it's actually a regular value input. We force it to be a non-shape input
    # by providing it in the profile as a fixed-shape value tensor.
    profile = builder.create_optimization_profile()

    # x: [1, L_tokens] int64
    profile.add("x", min=(1, 4), opt=(1, 32), max=(1, 64))
    # x_length: [1] int64
    profile.add("x_length", min=(1,), opt=(1,), max=(1,))
    # noise_scale: [1] float32 — scalar value, NOT shape tensor
    profile.add("noise_scale", min=(1,), opt=(1,), max=(1,))
    # length_scale: [1] float32 — scalar value, NOT shape tensor
    profile.add("length_scale", min=(1,), opt=(1,), max=(1,))
    # noise: [1, 80, L] float32 — dynamic last dim
    profile.add("noise", min=(1, 80, 72), opt=(1, 80, 256), max=(1, 80, 800))

    config.add_optimization_profile(profile)

    print("Building engine (this may take 10-30 minutes on Orin Nano)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed — returned None")

    with open(output_path, "wb") as f:
        f.write(serialized)

    print(f"Engine saved: {output_path} ({len(serialized) / 1e6:.1f} MB)")
    return output_path


def validate_engine(engine_path: str, onnx_path: str):
    """Quick sanity check: load engine and run a test inference."""
    import onnxruntime as ort
    from cuda import cudart

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    ctx = engine.create_execution_context()

    # Determine I/O shapes
    n_io = engine.num_io_tensors
    print(f"\nEngine I/O ({n_io} tensors):")
    for i in range(n_io):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"  {'IN' if mode == trt.TensorIOMode.INPUT else 'OUT'}: {name} {shape} {dtype}")

    # Run a quick inference to verify
    test_tokens = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int64)
    test_len = np.array([10], dtype=np.int64)
    test_ns = np.array([1.0], dtype=np.float32)
    test_ls = np.array([1.0], dtype=np.float32)
    test_noise = np.random.randn(1, 80, 256).astype(np.float32)

    # Allocate device memory
    bufs = {}
    for i in range(n_io):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        # Replace -1 dims with actual
        actual_shape = []
        for d in shape:
            actual_shape.append(d if d > 0 else 256)
        nbytes = int(np.prod(actual_shape)) * 4  # float32
        err, ptr = cudart.cudaMalloc(nbytes)
        if err != 0:
            raise RuntimeError(f"cudaMalloc {nbytes}B failed: {err}")
        bufs[name] = (ptr, tuple(actual_shape))

    # Copy inputs
    def copy_to(name, arr):
        ptr, shape = bufs[name]
        cudart.cudaMemcpy(ptr, arr.ctypes.data, arr.nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    copy_to("x", test_tokens)
    copy_to("x_length", test_len)
    copy_to("noise_scale", test_ns)
    copy_to("length_scale", test_ls)
    copy_to("noise", test_noise)

    # Set shapes and addresses
    ctx.set_input_shape("x", (1, 10))
    ctx.set_input_shape("noise", (1, 80, 256))
    for name, (ptr, shape) in bufs.items():
        ctx.set_tensor_address(name, int(ptr))

    err, stream = cudart.cudaStreamCreate()
    ok = ctx.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)

    if not ok:
        print("WARNING: execute_async_v3 returned False")

    # Get mel output
    mel_name = "mel"
    mel_ptr, mel_shape = bufs[mel_name]
    mel = np.empty(mel_shape, dtype=np.float32)
    cudart.cudaMemcpy(mel.ctypes.data, mel_ptr, mel.nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    print(f"\nTest inference: mel shape = {mel.shape}, range=[{mel.min():.3f}, {mel.max():.3f}]")
    print(f"No NaN: {not np.isnan(mel).any()}, No Inf: {not np.isinf(mel).any()}")

    # Cleanup
    cudart.cudaStreamDestroy(stream)
    for ptr, _ in bufs.values():
        cudart.cudaFree(ptr)

    # Compare with ORT
    print("\nComparing with ORT reference...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    mel_ort = sess.run(None, {
        "x": test_tokens, "x_length": test_len,
        "noise_scale": test_ns, "length_scale": test_ls,
        "noise": test_noise,
    })[0]
    print(f"  ORT mel shape: {mel_ort.shape}")
    # TRT produces output at actual L_mel (from duration predictor), may differ from input noise size
    # Compare only the overlapping region
    min_frames = min(mel.shape[2], mel_ort.shape[2])
    diff = np.abs(mel[:, :, :min_frames] - mel_ort[:, :, :min_frames])
    print(f"  Max abs diff (TRT vs ORT): {diff.max():.6f}")
    print(f"  Mean abs diff: {diff.mean():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--workspace", type=int, default=1024)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if not Path(args.onnx).exists():
        print(f"ONNX not found: {args.onnx}", file=sys.stderr)
        sys.exit(1)

    build_engine(args.onnx, args.output, fp16=args.fp16, workspace_mb=args.workspace)

    if args.validate:
        validate_engine(args.output, args.onnx)


if __name__ == "__main__":
    main()
