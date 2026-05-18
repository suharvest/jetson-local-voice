#!/usr/bin/env python3
"""Build Matcha acoustic TRT engine v2 — explicit shape tensor annotations.

The v3 ONNX surgery replaced all Range nodes with static Constant+Slice, and
converted length_scale to INT64 fixed-point. ORT validation passes. But TRT's
ONNX parser marks length_scale as "unused at runtime" because it can't trace
the data-dependent shape chain through the duration predictor.

Fix: use Python TRT API to explicitly mark length_scale as a SHAPE_TENSOR input.
This forces TRT to include it in shape analysis, allowing the noise Slice ends
to be resolved from the length_scale→duration chain.

Usage (on Jetson):
    python3 build_matcha_acoustic_v2.py \\
        --onnx /tmp/model-steps-3-v3.onnx \\
        --output /tmp/matcha_acoustic_fp16.engine
"""

import argparse, sys, numpy as np
from pathlib import Path
import tensorrt as trt


def build(onnx_path, output_path, fp16=True, ws_mb=1024):
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
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_mb * 1024 * 1024)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()

    # All inputs use set_shape in TRT 10.x
    profile.set_shape("x", (1, 4), (1, 32), (1, 64))
    profile.set_shape("x_length", (1,), (1,), (1,))
    profile.set_shape("noise_scale", (1,), (1,), (1,))

    # length_scale: INT64, explicitly marked as SHAPE tensor input.
    # This tells TRT to keep it in the shape analysis graph, preventing
    # the "unused" dead-code elimination that breaks the duration predictor.
    profile.set_shape_input("length_scale", (1,), (1,), (1,))
    print("  Marked length_scale as shape tensor input")

    # noise: dynamic last dim, same as mel output
    profile.set_shape("noise", (1, 80, 72), (1, 80, 256), (1, 80, 800))

    config.add_optimization_profile(profile)

    print(f"Building engine (WS={ws_mb}MB, fp16={fp16})...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build returned None")

    with open(output_path, "wb") as f:
        f.write(serialized)
    print(f"Saved: {output_path} ({len(serialized)/1e6:.1f}MB)")
    return output_path


def validate(engine_path, onnx_path):
    """Run a quick inference and compare with ORT."""
    import onnxruntime as ort
    from cuda import cudart

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    ctx = engine.create_execution_context()

    test_tokens = np.array([list(range(10))], dtype=np.int64)
    test_len = np.array([10], dtype=np.int64)
    test_ns = np.array([1.0], dtype=np.float32)
    test_ls = np.array([1000], dtype=np.int64)  # fixed-point: 1.0 * 1000
    test_noise = np.random.randn(1, 80, 256).astype(np.float32)

    # Allocate and run
    bufs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        actual = tuple(d if d > 0 else 256 for d in shape)
        nbytes = int(np.prod(actual)) * 4
        err, ptr = cudart.cudaMalloc(nbytes)
        if err != 0:
            raise RuntimeError(f"cudaMalloc {nbytes} failed: {err}")
        bufs[name] = (ptr, actual)

    def copy_h2d(name, arr):
        ptr, shape = bufs[name]
        cudart.cudaMemcpy(ptr, arr.ctypes.data, arr.nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    copy_h2d("x", test_tokens)
    copy_h2d("x_length", test_len)
    copy_h2d("noise_scale", test_ns)
    copy_h2d("length_scale", test_ls)
    copy_h2d("noise", test_noise)

    ctx.set_input_shape("x", (1, 10))
    ctx.set_input_shape("noise", (1, 80, 256))
    for name, (ptr, shape) in bufs.items():
        ctx.set_tensor_address(name, int(ptr))

    err, stream = cudart.cudaStreamCreate()
    ok = ctx.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)

    if not ok:
        print("  execute_async_v3 returned False")
        return

    mel_ptr, mel_shape = bufs["mel"]
    mel = np.empty(mel_shape, dtype=np.float32)
    cudart.cudaMemcpy(mel.ctypes.data, mel_ptr, mel.nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    print(f"  TRT mel: {mel.shape}, range=[{mel.min():.3f},{mel.max():.3f}], NaN={np.isnan(mel).any()}")

    # ORT reference
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    mel_ort = sess.run(None, {"x": test_tokens, "x_length": test_len,
                               "noise_scale": test_ns, "length_scale": test_ls,
                               "noise": test_noise})[0]
    print(f"  ORT mel: {mel_ort.shape}")
    min_f = min(mel.shape[2], mel_ort.shape[2])
    diff = np.abs(mel[:,:,:min_f] - mel_ort[:,:,:min_f])
    print(f"  Max abs diff: {diff.max():.6f}, Mean: {diff.mean():.6f}")

    cudart.cudaStreamDestroy(stream)
    for ptr, _ in bufs.values():
        cudart.cudaFree(ptr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--workspace", type=int, default=1024)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()

    if not Path(args.onnx).exists():
        print(f"ONNX not found: {args.onnx}"); sys.exit(1)

    build(args.onnx, args.output, fp16=args.fp16, ws_mb=args.workspace)
    if args.validate:
        validate(args.output, args.onnx)


if __name__ == "__main__":
    main()
