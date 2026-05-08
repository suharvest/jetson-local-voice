#!/usr/bin/env python3
"""Build Qwen3-TTS Talker INT8 engine with BF16 attention layers.

Produces:
  talker_decode_int8.engine  — dual-profile (prefill + decode), W8A16
  talker_prefill_int8.engine — batch prefill, W8A16

Attention MatMul layers pinned to BF16 to prevent FP16/INT8 QK^T overflow.
Other layers are INT8 with entropy calibration (synthetic data).

No pycuda required — uses ctypes to call libcuda.so directly.

Usage (on Jetson Nano):
  SKIP_PREFILL=1 python3 build_talker_int8_engine.py

Environment variables:
  ONNX_DECODE  — path to talker_decode.onnx
  ONNX_PREFILL — path to talker_prefill.onnx
  OUT_DIR      — output directory (default: .../engines/)
  WS_MB        — workspace MiB (default: 512)
  SKIP_PREFILL — 1 to skip prefill
  SKIP_DECODE  — 1 to skip decode
  CALIB_NPZ    — optional real calibration batches from bench/calib/gen_calib_data.py
"""

import os, sys, time, ctypes
import numpy as np
import tensorrt as trt

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_DIR = "/home/harvest/voice_test/models/qwen3-tts"
ONNX_DECODE = os.environ.get("ONNX_DECODE", f"{MODEL_DIR}/onnx/talker_decode_dynamic.onnx")
ONNX_PREFILL = os.environ.get("ONNX_PREFILL", f"{MODEL_DIR}/onnx/talker_prefill_dynamic.onnx")
OUT_DIR = os.environ.get("OUT_DIR", f"{MODEL_DIR}/engines")
WS_MB = int(os.environ.get("WS_MB", "512"))
SKIP_PREFILL = os.environ.get("SKIP_PREFILL", "0") == "1"
SKIP_DECODE = os.environ.get("SKIP_DECODE", "0") == "1"
CALIB_BATCHES = 50
CALIB_NPZ = os.environ.get("CALIB_NPZ", "")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
HIDDEN_DIM = 1024
N_LAYERS = 28
N_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 3072
MAX_SEQ = 200
MAX_PAST = 200

# ── CUDA runtime via ctypes ─────────────────────────────────────────────────
# Use CUDA Runtime API (libcudart.so) — no explicit context init needed.
_libcudart = ctypes.CDLL("libcudart.so")

cudaError_t = ctypes.c_int
cudaMemcpyHostToDevice = 1

_libcudart.cudaMalloc.restype = cudaError_t
_libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]

_libcudart.cudaFree.restype = cudaError_t
_libcudart.cudaFree.argtypes = [ctypes.c_void_p]

_libcudart.cudaMemcpy.restype = cudaError_t
_libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


class CudaMem:
    """Minimal CUDA memory allocator via libcudart.so (runtime API)."""

    def alloc(self, nbytes):
        ptr = ctypes.c_void_p()
        # Round up to 256-byte alignment
        nbytes = ((nbytes + 255) // 256) * 256
        if nbytes == 0:
            nbytes = 256
        err = _libcudart.cudaMalloc(ctypes.byref(ptr), nbytes)
        if err != 0:
            raise RuntimeError(f"cudaMalloc({nbytes}) failed: {err}")
        return ptr.value

    def free(self, ptr):
        if ptr is not None and ptr != 0:
            _libcudart.cudaFree(ctypes.c_void_p(ptr))

    def copy_to_device(self, dptr, host_arr):
        nbytes = host_arr.nbytes
        if nbytes == 0:
            return
        err = _libcudart.cudaMemcpy(
            ctypes.c_void_p(dptr),
            host_arr.ctypes.data_as(ctypes.c_void_p),
            nbytes,
            cudaMemcpyHostToDevice,
        )
        if err != 0:
            raise RuntimeError(f"cudaMemcpy H2D failed: {err}")


_cuda = CudaMem()


# ── Calibrator ─────────────────────────────────────────────────────────────
class CtypesCalibrator(trt.IInt8EntropyCalibrator2):
    """Entropy calibrator — synthetic data, no pycuda dependency."""

    def __init__(self, input_specs, n_batches=50, cache_file="/tmp/talker_calib.cache",
                 calib_data=None):
        super().__init__()
        self.cache_file = cache_file
        self.n_batches = n_batches
        self.batch_idx = 0
        self._d_ptrs = {}
        self.input_specs = input_specs
        print(f"Calibrator: {len(self.input_specs)} inputs:", flush=True)
        for name, dims in self.input_specs:
            print(f"  {name}: {dims}", flush=True)

        # Prefer real calibration batches from the product text projection path.
        # Synthetic random tensors are a fallback only; they are too far from
        # real Talker residual/KV distributions for quality-sensitive builds.
        if calib_data:
            self._batches = self._load_real_batches(calib_data)
            print(f"  Loaded {len(self._batches)} real calibration batches from {calib_data}", flush=True)
        else:
            self._batches = self._make_synthetic_batches(n_batches)

        # Pre-allocate GPU buffers
        max_sizes = {}
        for batch in self._batches:
            for name, arr in batch.items():
                max_sizes[name] = max(max_sizes.get(name, 0), arr.nbytes)
        for name, sz in max_sizes.items():
            self._d_ptrs[name] = _cuda.alloc(max(sz, 256))
        print(f"  Allocated GPU buffers for {len(self._d_ptrs)} inputs", flush=True)

    def _load_real_batches(self, calib_data):
        data = np.load(calib_data)
        batch_ids = sorted(
            {
                int(key.split("_", 1)[0][5:])
                for key in data.files
                if key.startswith("batch") and "_" in key
            }
        )
        batches = []
        required = [name for name, _ in self.input_specs]
        for batch_id in batch_ids:
            batch = {}
            prefix = f"batch{batch_id}_"
            for name in required:
                key = prefix + name
                if key not in data:
                    raise KeyError(f"real calibration batch {batch_id} missing {name}")
                batch[name] = np.ascontiguousarray(data[key])
            batches.append(batch)
        if not batches:
            raise ValueError(f"no batches found in {calib_data}")
        return batches

    def _make_synthetic_batches(self, n_batches):
        rng = np.random.RandomState(42)
        batches = []
        for b in range(n_batches):
            batch = {}
            past_len = rng.randint(0, MAX_PAST + 1)
            for name, dims in self.input_specs:
                resolved = []
                for d in dims:
                    v = d
                    if d == -1:
                        if "past_key" in name or "past_value" in name:
                            v = past_len
                        elif "attention_mask" in name:
                            v = past_len + 1
                        else:
                            v = 1
                    resolved.append(v)
                # Generate data
                if "attention_mask" in name:
                    arr = np.ones(resolved, dtype=np.int64)
                elif "position_ids" in name:
                    # position_ids shape: (1, seq_len), values = [past_len, past_len+1, ...]
                    seq_len = resolved[1]
                    arr = np.arange(past_len, past_len + seq_len, dtype=np.int64).reshape(1, seq_len)
                elif any(p in name for p in ["past_key", "past_value"]):
                    arr = rng.randn(*resolved).astype(np.float32) * 0.02
                elif "inputs_embeds" in name:
                    arr = rng.randn(*resolved).astype(np.float32) * 0.5
                else:
                    arr = rng.randn(*resolved).astype(np.float32) * 0.1
                batch[name] = np.ascontiguousarray(arr)
            batches.append(batch)
        print(f"  Generated {len(batches)} synthetic calib batches, past_len 0..{MAX_PAST}", flush=True)
        return batches

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.batch_idx >= len(self._batches):
            return None
        batch = self._batches[self.batch_idx]
        self.batch_idx += 1

        ptrs = []
        for name in names:
            arr = batch[name]
            _cuda.copy_to_device(self._d_ptrs[name], arr)
            ptrs.append(self._d_ptrs[name])
        if self.batch_idx % 10 == 0 or self.batch_idx == 1:
            print(f"  Calib batch {self.batch_idx}/{len(self._batches)}", flush=True)
        return ptrs

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"  Calib cache: {self.cache_file} ({len(cache)/1024:.1f} KB)")


# ── Build helper ───────────────────────────────────────────────────────────
def find_attention_matmul_layers(network):
    """Find attention-score MatMul layers (QK^T, score*V) via shape heuristic.

    Attention MatMuls have two 4D inputs: [B,H,S,D] x [B,H,D,T].
    Weight-projection MatMuls have at least one 2D/3D input.
    Only the former need BF16 pinning to prevent FP16/INT8 overflow.
    """
    attention_matmul = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type != trt.LayerType.MATRIX_MULTIPLY:
            continue
        # Both inputs are 4D → attention score MatMul (QK^T or score*V)
        if all(len(layer.get_input(j).shape) == 4 for j in range(layer.num_inputs)):
            attention_matmul.append(i)
    return attention_matmul


def build_engine(onnx_path, out_path, engine_name, profiles_fn):
    """Build a TRT engine from ONNX with INT8 + BF16-attention."""

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    print(f"\n{'='*60}")
    print(f"Building {engine_name}")
    print(f"  ONNX: {onnx_path}")
    print(f"  OUT:  {out_path}")
    print(f"  WS:   {WS_MB} MiB")
    print(f"{'='*60}")

    # Parse ONNX
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        data = f.read()
    if not parser.parse(data):
        for i in range(parser.num_errors):
            print(f"  ONNX parse error: {parser.get_error(i)}")
        raise RuntimeError("ONNX parse failed")

    print(f"  Parsed: {network.num_layers} layers, {network.num_inputs} inputs, "
          f"{network.num_outputs} outputs")

    for j in range(min(network.num_inputs, 5)):
        inp = network.get_input(j)
        print(f"  IN:  {inp.name} shape={inp.shape} dtype={inp.dtype}")
    if network.num_inputs > 5:
        print(f"  ... ({network.num_inputs - 5} more inputs)")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WS_MB << 20)
    # Precision: INT8 weights, BF16 fallback
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    # Pin only attention-score MatMul layers to BF16 (QK^T, score*V).
    # Weight-projection MatMuls (Constant input) are left for INT8 quantization.
    attn_indices = find_attention_matmul_layers(network)
    print(f"\n  Pinning {len(attn_indices)} attention-score MatMul layers to BF16...")
    pinned = 0
    for idx in attn_indices:
        layer = network.get_layer(idx)
        try:
            layer.precision = trt.DataType.BF16
            pinned += 1
        except Exception as e:
            pass
    print(f"  Pinned: {pinned} MatMul layers")

    # Set optimization profiles
    profiles_fn(builder, network, config)

    # Calibrator — extract input specs from TRT network (avoid 2nd onnx.load of 1.7GB)
    input_specs = []
    for j in range(network.num_inputs):
        inp = network.get_input(j)
        dims = [int(d) for d in inp.shape]  # -1 for dynamic
        input_specs.append((inp.name, dims))
    calib_tag = ""
    if CALIB_NPZ:
        calib_tag = "_" + os.path.basename(CALIB_NPZ).replace(".", "_")
    calib_cache = f"/tmp/{engine_name}_calib{calib_tag}.cache"
    calibrator = CtypesCalibrator(input_specs, n_batches=CALIB_BATCHES,
                                  cache_file=calib_cache,
                                  calib_data=CALIB_NPZ or None)
    config.int8_calibrator = calibrator

    # Build
    print(f"\n  Building with {CALIB_BATCHES} calibration batches (this may take 20-40 min)...")
    t0 = time.time()

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build returned None")

    elapsed = time.time() - t0
    engine_bytes = bytes(serialized)
    sz_mb = len(engine_bytes) / 1e6

    with open(out_path, "wb") as f:
        f.write(engine_bytes)

    print(f"\n  Built in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Engine: {out_path} ({sz_mb:.1f} MB)")

    # Verify
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    print(f"  Verified: {engine.num_io_tensors} I/O tensors, "
          f"{engine.num_optimization_profiles} profiles")

    # Print I/O summary
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        mark = "IN" if mode == trt.TensorIOMode.INPUT else "OUT"
        print(f"    {name}: {mark} dtype={dtype} shape={shape}")

    return out_path, sz_mb


# ── Profile definitions ────────────────────────────────────────────────────
def setup_decode_profiles(builder, network, config):
    """Single profile: past=0..200 dynamic, inputs_embeds seq=1 fixed.

    Use single profile so C++ loads separate prefill engine
    (talker_prefill_bf16.engine) instead of using this engine for prefill.
    """
    p0 = builder.create_optimization_profile()
    p0.set_shape("inputs_embeds",  (1, 1, HIDDEN_DIM), (1, 1, HIDDEN_DIM), (1, 1, HIDDEN_DIM))
    p0.set_shape("attention_mask", (1, 1),              (1, MAX_PAST // 2),  (1, MAX_PAST + 1))
    p0.set_shape("position_ids",  (1, 1),              (1, 1),              (1, 1))
    for i in range(N_LAYERS):
        p0.set_shape(f"past_key_{i}",   (1, N_HEADS, 0, HEAD_DIM), (1, N_HEADS, MAX_PAST // 2, HEAD_DIM), (1, N_HEADS, MAX_PAST, HEAD_DIM))
        p0.set_shape(f"past_value_{i}", (1, N_HEADS, 0, HEAD_DIM), (1, N_HEADS, MAX_PAST // 2, HEAD_DIM), (1, N_HEADS, MAX_PAST, HEAD_DIM))
    config.add_optimization_profile(p0)
    print("  Profile 0 (decode): emb=1 fixed, past=0..200")


def setup_prefill_profiles(builder, network, config):
    """Single profile: seq_len dynamic, no KV inputs."""
    p0 = builder.create_optimization_profile()
    p0.set_shape("inputs_embeds",  (1, 1, HIDDEN_DIM), (1, 60, HIDDEN_DIM), (1, MAX_SEQ, HIDDEN_DIM))
    p0.set_shape("attention_mask", (1, 1),              (1, 60),              (1, MAX_SEQ))
    p0.set_shape("position_ids",  (1, 1),              (1, 60),              (1, MAX_SEQ))
    config.add_optimization_profile(p0)
    print("  Profile 0: emb=1..200 + pos_ids")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    if not SKIP_DECODE:
        out_decode = os.path.join(OUT_DIR, "talker_decode_int8.engine")
        build_engine(ONNX_DECODE, out_decode, "talker_decode_int8",
                     setup_decode_profiles)
        os.system(f"ls -lh {out_decode}")
        os.system(f"md5sum {out_decode}")

    if not SKIP_PREFILL:
        out_prefill = os.path.join(OUT_DIR, "talker_prefill_int8.engine")
        build_engine(ONNX_PREFILL, out_prefill, "talker_prefill_int8",
                     setup_prefill_profiles)
        os.system(f"ls -lh {out_prefill}")
        os.system(f"md5sum {out_prefill}")

    total = time.time() - t_start
    print(f"\nTotal build time: {total:.0f}s ({total/60:.1f}m)")


if __name__ == "__main__":
    main()
