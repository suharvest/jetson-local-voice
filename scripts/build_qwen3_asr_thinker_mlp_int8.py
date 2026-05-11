#!/usr/bin/env python3
"""Build an experimental Qwen3-ASR thinker engine with MLP-only INT8.

This is a feasibility builder for W8A8/PTQ, not a production recipe.
It keeps attention and lm_head conservative while forcing only
``/mlp/{gate,up,down}_proj/MatMul`` layers to INT8 and FP16 output.

The ASR thinker ONNX uses external data, so the parser must read from the
model path instead of bytes.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import time
from pathlib import Path

import numpy as np
import tensorrt as trt


N_LAYERS = 28
N_HEADS = 8
HEAD_DIM = 128
HIDDEN_DIM = 1024

_libcudart = ctypes.CDLL("libcudart.so")
_libcudart.cudaMalloc.restype = ctypes.c_int
_libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_libcudart.cudaFree.restype = ctypes.c_int
_libcudart.cudaFree.argtypes = [ctypes.c_void_p]
_libcudart.cudaMemcpy.restype = ctypes.c_int
_libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
CUDA_MEMCPY_HOST_TO_DEVICE = 1


def _cuda_malloc(nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    nbytes = max(256, ((int(nbytes) + 255) // 256) * 256)
    err = _libcudart.cudaMalloc(ctypes.byref(ptr), nbytes)
    if err != 0:
        raise RuntimeError(f"cudaMalloc({nbytes}) failed: {err}")
    return int(ptr.value)


def _cuda_free(ptr: int) -> None:
    if ptr:
        _libcudart.cudaFree(ctypes.c_void_p(ptr))


def _cuda_copy_to_device(dptr: int, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    err = _libcudart.cudaMemcpy(
        ctypes.c_void_p(dptr),
        arr.ctypes.data_as(ctypes.c_void_p),
        arr.nbytes,
        CUDA_MEMCPY_HOST_TO_DEVICE,
    )
    if err != 0:
        raise RuntimeError(f"cudaMemcpy H2D failed: {err}")


class SyntheticAsrCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, input_specs, cache_file: str, batches: int, opt_input_len: int, opt_past: int):
        super().__init__()
        self.input_specs = input_specs
        self.cache_file = cache_file
        self.batch_idx = 0
        self.rng = np.random.RandomState(20260511)
        self.host_batches = [
            self._make_batch(i, opt_input_len=opt_input_len, opt_past=opt_past) for i in range(batches)
        ]
        self.device_ptrs = {}
        max_sizes = {}
        for batch in self.host_batches:
            for name, arr in batch.items():
                max_sizes[name] = max(max_sizes.get(name, 0), arr.nbytes)
        for name, size in max_sizes.items():
            self.device_ptrs[name] = _cuda_malloc(size)
        print(f"calibrator: {batches} synthetic batches, {len(self.device_ptrs)} inputs")

    def __del__(self):
        for ptr in getattr(self, "device_ptrs", {}).values():
            _cuda_free(ptr)

    def _make_batch(self, index: int, opt_input_len: int, opt_past: int):
        past_len = [0, 16, 64, 128, opt_past][index % 5]
        seq_len = [1, 8, 32, opt_input_len][index % 4]
        batch = {}
        for name, shape, dtype in self.input_specs:
            resolved = []
            for d in shape:
                if d != -1:
                    resolved.append(int(d))
                elif name == "inputs_embeds":
                    resolved.append(seq_len)
                elif name.startswith("past_key_values_"):
                    resolved.append(past_len)
                elif name == "rope_rotary_cos_sin":
                    resolved.append(max(256, past_len + seq_len + 8))
                elif name == "kvcache_start_index":
                    resolved.append(1)
                else:
                    resolved.append(1)

            if dtype == trt.DataType.HALF:
                if name == "inputs_embeds":
                    arr = self.rng.normal(0, 0.35, resolved).astype(np.float16)
                else:
                    arr = self.rng.normal(0, 0.02, resolved).astype(np.float16)
            elif dtype == trt.DataType.FLOAT:
                if name == "rope_rotary_cos_sin":
                    arr = self.rng.normal(0, 0.5, resolved).astype(np.float32)
                else:
                    arr = self.rng.normal(0, 0.1, resolved).astype(np.float32)
            elif dtype == trt.DataType.INT32:
                if name == "context_lengths":
                    arr = np.array([past_len + seq_len], dtype=np.int32)
                elif name == "kvcache_start_index":
                    arr = np.array([0], dtype=np.int32)
                else:
                    arr = np.zeros(resolved, dtype=np.int32)
            elif dtype == trt.DataType.INT64:
                arr = np.zeros(resolved, dtype=np.int64)
            else:
                raise TypeError(f"unsupported calibrator dtype for {name}: {dtype}")
            batch[name] = np.ascontiguousarray(arr)
        return batch

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.batch_idx >= len(self.host_batches):
            return None
        batch = self.host_batches[self.batch_idx]
        self.batch_idx += 1
        ptrs = []
        for name in names:
            arr = batch[name]
            _cuda_copy_to_device(self.device_ptrs[name], arr)
            ptrs.append(self.device_ptrs[name])
        if self.batch_idx == 1 or self.batch_idx % 10 == 0:
            print(f"calib batch {self.batch_idx}/{len(self.host_batches)}", flush=True)
        return ptrs

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            return Path(self.cache_file).read_bytes()
        return None

    def write_calibration_cache(self, cache):
        Path(self.cache_file).write_bytes(cache)
        print(f"wrote calibration cache {self.cache_file} ({len(cache) / 1024:.1f} KiB)")


def parse_network(onnx_path: str, network, logger) -> None:
    parser = trt.OnnxParser(network, logger)
    if hasattr(parser, "parse_from_file"):
        ok = parser.parse_from_file(onnx_path)
    else:
        ok = parser.parse_from_file(onnx_path.encode())
    if not ok:
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")


def is_mlp_matmul(layer) -> bool:
    return layer.type == trt.LayerType.MATRIX_MULTIPLY and "/mlp/" in layer.name


def set_precisions(network) -> tuple[int, int]:
    mlp_int8 = 0
    conservative = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        try:
            if is_mlp_matmul(layer):
                layer.precision = trt.DataType.INT8
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.HALF)
                mlp_int8 += 1
            else:
                # Keep the rest of the plain MatMul layers out of INT8 tactic
                # selection. Do not constrain AttentionPlugin here; its plugin
                # format contract is narrower than TensorRT's generic layer API.
                if layer.type == trt.LayerType.MATRIX_MULTIPLY:
                    layer.precision = trt.DataType.HALF
                    for j in range(layer.num_outputs):
                        if layer.get_output(j).dtype in (trt.DataType.FLOAT, trt.DataType.HALF):
                            layer.set_output_type(j, trt.DataType.HALF)
                    conservative += 1
                elif layer.type == trt.LayerType.PLUGIN_V2 and "AttentionPlugin" in layer.name:
                    layer.precision = trt.DataType.HALF
                    conservative += 1
        except Exception as exc:
            print(f"precision skip {i} {layer.name}: {exc}")
    return mlp_int8, conservative


def add_profile(builder, network, config, max_input_len: int, max_kv: int, opt_input_len: int, opt_kv: int):
    profile = builder.create_optimization_profile()
    seen = set()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        name = inp.name
        seen.add(name)
        if name == "inputs_embeds":
            profile.set_shape(name, (1, 1, HIDDEN_DIM), (1, opt_input_len, HIDDEN_DIM), (1, max_input_len, HIDDEN_DIM))
        elif name.startswith("past_key_values_"):
            profile.set_shape(
                name,
                (1, 2, N_HEADS, 0, HEAD_DIM),
                (1, 2, N_HEADS, opt_kv, HEAD_DIM),
                (1, 2, N_HEADS, max_kv, HEAD_DIM),
            )
        elif name == "rope_rotary_cos_sin":
            profile.set_shape(name, (1, 1, HEAD_DIM), (1, max_kv, HEAD_DIM), (1, max_kv + max_input_len, HEAD_DIM))
        elif name == "context_lengths":
            profile.set_shape(name, (1,), (1,), (1,))
        elif name == "last_token_ids":
            profile.set_shape(name, (1, 1), (1, 1), (1, 1))
        elif name == "kvcache_start_index":
            profile.set_shape(name, (1,), (1,), (1,))
        else:
            raise KeyError(f"unhandled input {name} shape={inp.shape}")
    config.add_optimization_profile(profile)
    print(f"profile: max_input_len={max_input_len} opt_input_len={opt_input_len} max_kv={max_kv} opt_kv={opt_kv}")
    print(f"profile inputs: {sorted(seen)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    # NOTE: --onnx, --out-dir defaults reflect the 2026-05-11 build session and
    # may not exist on a fresh checkout. Pass --onnx/--out-dir explicitly.
    # --plugin default points at the canonical source-build location; override
    # via EDGE_LLM_ASR_PLUGIN_PATH or pass --plugin if your build lives elsewhere.
    ap.add_argument("--onnx", default="/home/harvest/tensorrt-edgellm-workspace/Qwen3-ASR-0.6B/onnx-src/thinker/model.onnx")
    ap.add_argument("--out-dir", default="/tmp/qwen3_asr_thinker_mlp_int8_0511")
    ap.add_argument(
        "--plugin",
        default=os.environ.get(
            "EDGE_LLM_ASR_PLUGIN_PATH",
            os.path.expanduser(
                "~/project/tensorrt-edge-llm/build_sm87/libNvInfer_edgellm_plugin.so"
            ),
        ),
    )
    ap.add_argument("--workspace-mb", type=int, default=1024)
    ap.add_argument("--max-input-len", type=int, default=128)
    ap.add_argument("--max-kv", type=int, default=256)
    ap.add_argument("--opt-input-len", type=int, default=39)
    ap.add_argument("--opt-kv", type=int, default=128)
    ap.add_argument("--calib-batches", type=int, default=40)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plugin:
        ctypes.CDLL(args.plugin, mode=ctypes.RTLD_GLOBAL)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parse_network(args.onnx, network, logger)
    print(f"parsed network: {network.num_layers} layers, {network.num_inputs} inputs, {network.num_outputs} outputs")

    mlp_int8, conservative = set_precisions(network)
    print(f"precision: mlp_int8={mlp_int8}, conservative_half={conservative}")
    if mlp_int8 != 84:
        raise RuntimeError(f"expected 84 MLP MatMul layers, got {mlp_int8}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb << 20)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    add_profile(builder, network, config, args.max_input_len, args.max_kv, args.opt_input_len, args.opt_kv)

    input_specs = []
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        input_specs.append((inp.name, [int(d) for d in inp.shape], inp.dtype))
        print(f"input {inp.name} dtype={inp.dtype} shape={inp.shape}")
    config.int8_calibrator = SyntheticAsrCalibrator(
        input_specs,
        str(out_dir / "mlp_int8_synth.cache"),
        args.calib_batches,
        args.opt_input_len,
        args.opt_kv,
    )

    print("building engine...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build returned None")
    elapsed = time.time() - t0
    engine_path = out_dir / "llm.engine"
    engine_path.write_bytes(bytes(serialized))
    print(f"built {engine_path} size={engine_path.stat().st_size / 1024 / 1024:.1f} MiB time={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
