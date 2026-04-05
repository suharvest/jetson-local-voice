#!/usr/bin/env python3
"""Qwen3-TTS — Optimized TRT pipeline with GPU-resident KV cache.

- Prefill: ORT CUDA EP (runs once)
- Talker decode: TRT FP16 with double-buffered GPU-resident KV (no KV memcpy per step)
- Code predictor: ORT CUDA EP with IOBinding (pre-allocated GPU buffers)
- Embeddings: ORT CUDA EP (small models)
- Vocoder: ORT CUDA EP
"""
import argparse, json, os, time, wave
import numpy as np
import onnxruntime as ort

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    print("WARNING: tensorrt/pycuda not available, falling back to ORT")

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "/tmp/qwen3-tts-bench/model")
SHERPA_DIR = os.environ.get("TTS_SHERPA_DIR", "/tmp/qwen3-sherpa")
TRT_TALKER = os.environ.get("TRT_TALKER", "/tmp/talker_decode_sherpa_fp16.engine")

with open(f"{SHERPA_DIR}/config.json") as f:
    CFG = json.load(f)

D = CFG["hidden_size"]
N_LAYERS = CFG["num_hidden_layers"]
N_GROUPS = CFG["num_code_groups"]
TALKER_VOCAB = CFG["vocab_size"]
CP_VOCAB = 2048
TTS_BOS = CFG["tts_bos_token_id"]
TTS_EOS = CFG["tts_eos_token_id"]
TTS_PAD = CFG["tts_pad_token_id"]
CODEC_BOS = CFG["codec_bos_id"]
CODEC_EOS = CFG["codec_eos_token_id"]
CODEC_PAD = CFG["codec_pad_id"]
CODEC_NOTHINK = CFG["codec_nothink_id"]
CODEC_THINK_BOS = CFG["codec_think_bos_id"]
CODEC_THINK_EOS = CFG["codec_think_eos_id"]
LANG_IDS = CFG["codec_language_id"]


# ---------------------------------------------------------------------------
# GPU-Resident Talker TRT Engine (double-buffered KV cache)
# ---------------------------------------------------------------------------
class TRTTalkerEngine:
    """TRT 10.x talker decode with GPU-resident KV cache.

    Only copies inputs_embeds (4KB) to GPU and logits+hidden (16KB) back per step.
    KV cache stays on GPU via double-buffered pointer swapping.
    """
    def __init__(self, engine_path, max_seq=200):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.max_seq = max_seq
        self.n_layers = N_LAYERS

        # Discover I/O dtypes
        self.dtypes = {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            self.dtypes[n] = trt.nptype(self.engine.get_tensor_dtype(n))

        # Determine KV dtype and element size
        kv_dtype = self.dtypes.get("past_key_0", np.float32)
        kv_itemsize = np.dtype(kv_dtype).itemsize

        # Double-buffer: A and B sets of 56 KV tensors
        # Each: [1, 8, max_seq, 128]
        kv_nbytes = 1 * 8 * max_seq * 128 * kv_itemsize
        self.kv_A = {}
        self.kv_B = {}
        for i in range(self.n_layers):
            for prefix in ["past_key_", "past_value_"]:
                self.kv_A[f"{prefix}{i}"] = cuda.mem_alloc(kv_nbytes)
                self.kv_B[f"{prefix}{i}"] = cuda.mem_alloc(kv_nbytes)

        # Small fixed I/O buffers
        emb_dtype = self.dtypes.get("inputs_embeds", np.float32)
        logits_dtype = self.dtypes.get("logits", np.float32)
        hidden_dtype = self.dtypes.get("last_hidden", np.float32)
        self.d_emb = cuda.mem_alloc(1 * 1 * D * np.dtype(emb_dtype).itemsize)
        self.d_logits = cuda.mem_alloc(1 * 1 * TALKER_VOCAB * np.dtype(logits_dtype).itemsize)
        self.d_hidden = cuda.mem_alloc(1 * 1 * D * np.dtype(hidden_dtype).itemsize)

        # Pre-allocate host output arrays
        self._h_logits = np.empty((1, 1, TALKER_VOCAB), dtype=logits_dtype)
        self._h_hidden = np.empty((1, 1, D), dtype=hidden_dtype)

        self.seq_len = 0
        self.parity = 0  # 0 = read A write B, 1 = read B write A

    def seed_kv(self, kv_dict):
        """One-time: copy prefill KV (numpy) to GPU buffer A."""
        for name, arr in kv_dict.items():
            if name in self.kv_A:
                a = np.ascontiguousarray(arr.astype(self.dtypes.get(name, np.float32)))
                cuda.memcpy_htod_async(self.kv_A[name], a, self.stream)
        self.stream.synchronize()
        self.parity = 0

    def decode_step(self, inputs_embeds):
        """One decode step. Only copies 4KB in + 16KB out. KV stays on GPU."""
        ctx = self.context
        read = self.kv_A if self.parity == 0 else self.kv_B
        write = self.kv_B if self.parity == 0 else self.kv_A

        # Copy inputs_embeds to GPU (4KB)
        emb_dtype = self.dtypes.get("inputs_embeds", np.float32)
        ie = np.ascontiguousarray(inputs_embeds.astype(emb_dtype))
        cuda.memcpy_htod_async(self.d_emb, ie, self.stream)

        # Bind inputs_embeds
        ctx.set_input_shape("inputs_embeds", (1, 1, D))
        ctx.set_tensor_address("inputs_embeds", int(self.d_emb))

        # Bind KV cache — pointer swap only, no memcpy
        for i in range(self.n_layers):
            for prefix in ["past_key_", "past_value_"]:
                name = f"{prefix}{i}"
                ctx.set_input_shape(name, (1, 8, self.seq_len, 128))
                ctx.set_tensor_address(name, int(read[name]))
                ctx.set_tensor_address(f"new_{name}", int(write[name]))

        # Bind outputs
        ctx.set_tensor_address("logits", int(self.d_logits))
        ctx.set_tensor_address("last_hidden", int(self.d_hidden))

        # Execute
        ctx.execute_async_v3(stream_handle=self.stream.handle)

        # Copy only logits + hidden back (16KB)
        cuda.memcpy_dtoh_async(self._h_logits, self.d_logits, self.stream)
        cuda.memcpy_dtoh_async(self._h_hidden, self.d_hidden, self.stream)
        self.stream.synchronize()

        self.seq_len += 1
        self.parity ^= 1
        return self._h_logits.astype(np.float32), self._h_hidden.astype(np.float32)


# ---------------------------------------------------------------------------
# ORT IOBinding Code Predictor
# ---------------------------------------------------------------------------
class CPIOBinding:
    """Code predictor with pre-allocated GPU buffers via ORT IOBinding."""
    def __init__(self, sess):
        self.sess = sess
        self.ctx_gpus = []
        for j in range(N_GROUPS - 1):
            ctx_len = j + 2
            self.ctx_gpus.append(
                ort.OrtValue.ortvalue_from_shape_and_type([1, ctx_len, D], np.float32, "cuda", 0))
        self.step_gpus = [
            ort.OrtValue.ortvalue_from_numpy(np.array([j], dtype=np.int64), "cuda", 0)
            for j in range(N_GROUPS - 1)]
        self.logits_gpu = ort.OrtValue.ortvalue_from_shape_and_type(
            [1, 1, CP_VOCAB], np.float32, "cuda", 0)
        self.bindings = []
        for j in range(N_GROUPS - 1):
            b = sess.io_binding()
            b.bind_ortvalue_input("context", self.ctx_gpus[j])
            b.bind_ortvalue_input("gen_step", self.step_gpus[j])
            b.bind_ortvalue_output("logits", self.logits_gpu)
            self.bindings.append(b)

    def predict(self, ctx_np, step):
        self.ctx_gpus[step].update_inplace(ctx_np)
        self.sess.run_with_iobinding(self.bindings[step])
        return self.logits_gpu.numpy()


# ---------------------------------------------------------------------------
# TRT Code Predictor (INT8 or FP32)
# ---------------------------------------------------------------------------
class CPTRT:
    """CP with TRT engine, pre-allocated GPU buffers."""
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        # Pre-allocate GPU buffers at max size
        max_ctx_bytes = 1 * 17 * D * 4  # [1, 17, 1024] float32
        self.d_ctx = cuda.mem_alloc(max_ctx_bytes)
        self.d_gs = cuda.mem_alloc(8)  # int64
        self.d_out = cuda.mem_alloc(1 * 1 * CP_VOCAB * 4)
        self._h_out = np.empty((1, 1, CP_VOCAB), dtype=np.float32)

    def predict(self, ctx_np, step):
        """Run CP. ctx_np: [1, step+2, D] float32."""
        ctx_np = np.ascontiguousarray(ctx_np)
        gs_np = np.array([step], dtype=np.int64)
        cuda.memcpy_htod_async(self.d_ctx, ctx_np, self.stream)
        cuda.memcpy_htod_async(self.d_gs, gs_np, self.stream)
        self.ctx.set_input_shape("context", ctx_np.shape)
        self.ctx.set_input_shape("gen_step", gs_np.shape)
        self.ctx.set_tensor_address("context", int(self.d_ctx))
        self.ctx.set_tensor_address("gen_step", int(self.d_gs))
        self.ctx.set_tensor_address("logits", int(self.d_out))
        self.ctx.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self._h_out, self.d_out, self.stream)
        self.stream.synchronize()
        return self._h_out.copy()


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
print("Loading models...")
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]

text_proj_s = ort.InferenceSession(f"{SHERPA_DIR}/text_project.onnx", so, providers=CUDA)
codec_emb_s = ort.InferenceSession(f"{SHERPA_DIR}/codec_embed.onnx", so, providers=CUDA)
cp_emb_s = ort.InferenceSession(f"{SHERPA_DIR}/code_predictor_embed.onnx", so, providers=CUDA)
prefill_s = ort.InferenceSession(f"{SHERPA_DIR}/talker_prefill.onnx", so, providers=CUDA)
voc_s = ort.InferenceSession(f"{MODEL_DIR}/vocoder.onnx", so, providers=CUDA) if os.path.exists(f"{MODEL_DIR}/vocoder.onnx") else None

# Talker: TRT with GPU-resident KV, or ORT CUDA fallback
talker_trt = None
if HAS_TRT and os.path.exists(TRT_TALKER):
    print("  Loading talker TRT engine (GPU-resident KV)...")
    talker_trt = TRTTalkerEngine(TRT_TALKER)
if not talker_trt:
    decode_s = ort.InferenceSession(f"{SHERPA_DIR}/talker_decode.onnx", so, providers=CUDA)

# CP: TRT INT8/FP32 if available, otherwise ORT CUDA with IOBinding
TRT_CP = os.environ.get("TRT_CP", "/tmp/cp_sherpa_int8.engine")
cp_trt_engine = None
if HAS_TRT and os.path.exists(TRT_CP):
    print(f"  Loading CP TRT engine: {TRT_CP}")
    cp_trt_engine = CPTRT(TRT_CP)
cp_s = ort.InferenceSession(f"{SHERPA_DIR}/code_predictor.onnx", so, providers=CUDA)
cp_iob = CPIOBinding(cp_s)

pf_outs = [o.name for o in prefill_s.get_outputs()]

# Tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
tok = Tokenizer(BPE(f"{MODEL_DIR}/tokenizer/vocab.json", f"{MODEL_DIR}/tokenizer/merges.txt"))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
def tokenize(text): return tok.encode(text).ids

print("Models loaded.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def text_proj(ids):
    return text_proj_s.run(None, {"input_ids": np.array([ids], dtype=np.int64)})[0]

def codec_emb(ids):
    return codec_emb_s.run(None, {"token_ids": np.array([ids], dtype=np.int64)})[0]

def cp_embed(token_id, layer_idx):
    return cp_emb_s.run(None, {
        "token_id": np.array([[token_id]], dtype=np.int64),
        "layer_idx": np.array(layer_idx, dtype=np.int64),
    })[0]

def sample(logits, vocab_size, k=50, t=0.9, suppress_eos=False, eos_id=None):
    l = logits.flatten()[:vocab_size].astype(np.float64)
    if suppress_eos and eos_id is not None:
        l[eos_id] = -1e9
    l = l / max(t, 1e-6)
    if 0 < k < len(l):
        threshold = np.partition(l, -k)[-k]
        l[l < threshold] = -1e9
    l = l - l.max()
    p = np.exp(l)
    p = p / p.sum()
    return int(np.random.choice(len(p), p=p))


# ---------------------------------------------------------------------------
# Build prefill
# ---------------------------------------------------------------------------
def build_prefill(text, lang="english"):
    role_emb = text_proj([151644, 77091, 198])
    special_emb = text_proj([TTS_BOS, TTS_EOS, TTS_PAD])
    tts_bos_e = special_emb[:, 0:1, :]
    tts_eos_e = special_emb[:, 1:2, :]
    tts_pad_e = special_emb[:, 2:3, :]
    codec_prefix = codec_emb([CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS, CODEC_PAD, CODEC_BOS])
    text_ids = tokenize(text)
    body_emb = text_proj(text_ids)
    codec_pad_e = codec_emb([CODEC_PAD])

    prefill = np.zeros((1, 8, D), dtype=np.float32)
    prefill[0, :3] = role_emb[0, :3]
    for i in range(3):
        prefill[0, 3+i] = tts_pad_e[0, 0] + codec_prefix[0, i]
    prefill[0, 6] = tts_bos_e[0, 0] + codec_prefix[0, 3]
    prefill[0, 7] = body_emb[0, 0] + codec_prefix[0, 4]

    trailing = []
    for i in range(1, len(text_ids)):
        trailing.append(body_emb[0, i] + codec_pad_e[0, 0])
    trailing.append(tts_eos_e[0, 0] + codec_pad_e[0, 0])

    return prefill, trailing, tts_pad_e, codec_pad_e


# ---------------------------------------------------------------------------
# Synthesize
# ---------------------------------------------------------------------------
def synthesize(text, lang="english", output="/tmp/tts_sherpa_trt.wav", max_frames=200, seed=42):
    np.random.seed(seed)
    print(f"\nSynth: \"{text}\" ({lang})")
    t_total = time.perf_counter()

    prefill_emb, trailing_text, tts_pad_e, codec_pad_e = build_prefill(text, lang)

    # --- Prefill (ORT CUDA) ---
    t0 = time.perf_counter()
    pf_result = prefill_s.run(None, {"inputs_embeds": prefill_emb})
    pf_map = dict(zip(pf_outs, pf_result))
    logits = pf_map["logits"]
    last_hidden = pf_map["last_hidden"]
    kv = {k: v for k, v in pf_map.items() if k.startswith("past_")}
    pf_ms = (time.perf_counter() - t0) * 1000

    # Seed TRT KV cache from prefill output (one-time copy)
    if talker_trt:
        talker_trt.seed_kv(kv)
        talker_trt.seq_len = prefill_emb.shape[1]
    print(f"  Prefill: {pf_ms:.0f}ms ({prefill_emb.shape[1]} tokens)")

    # --- Decode loop ---
    all_codes = []
    dt_times = []
    ct_times = []

    for step in range(max_frames):
        primary_code = sample(logits[0, -1, :], TALKER_VOCAB,
                              suppress_eos=(step < 2), eos_id=CODEC_EOS)
        if primary_code == CODEC_EOS:
            print(f"  EOS at step {step}")
            break

        # --- Code predictor with IOBinding ---
        t_cp = time.perf_counter()
        primary_e = codec_emb([primary_code])
        lh_last = last_hidden[:, -1:, :]
        cp_ctx = np.concatenate([lh_last, primary_e], axis=1).astype(np.float32)
        codec_sum = primary_e[0, 0, :].copy()
        frame_codes = [primary_code]

        for j in range(N_GROUPS - 1):
            cp_logits = cp_trt_engine.predict(cp_ctx, j) if cp_trt_engine else cp_iob.predict(cp_ctx, j)
            rc = sample(cp_logits, CP_VOCAB)
            frame_codes.append(rc)
            re = cp_embed(rc, j)
            cp_ctx = np.concatenate([cp_ctx, re], axis=1).astype(np.float32)
            codec_sum += re[0, 0, :]
        ct_times.append(time.perf_counter() - t_cp)
        all_codes.append(frame_codes)

        # --- Next talker input ---
        if step < len(trailing_text):
            text_e = trailing_text[step]
        else:
            text_e = tts_pad_e[0, 0, :] + codec_pad_e[0, 0, :]
        next_emb = (codec_sum + text_e).reshape(1, 1, D).astype(np.float32)

        # --- Talker decode ---
        t_d = time.perf_counter()
        if talker_trt:
            logits, last_hidden = talker_trt.decode_step(next_emb)
        else:
            feeds = {"inputs_embeds": next_emb}
            feeds.update(kv)
            dc_outs = [o.name for o in decode_s.get_outputs()]
            dc_result = decode_s.run(None, feeds)
            dc_map = dict(zip(dc_outs, dc_result))
            logits = dc_map["logits"]
            last_hidden = dc_map["last_hidden"]
            kv = {k.replace("new_", ""): v for k, v in dc_map.items()
                  if k.startswith("new_past_")}
        dt_times.append(time.perf_counter() - t_d)

        if (step + 1) % 10 == 0:
            print(f"  Frame {step + 1}")

    n = len(all_codes)
    if n == 0:
        print("  No frames!")
        return

    dur = n / 12.5
    da = np.mean(dt_times) * 1000
    ca = np.mean(ct_times) * 1000

    # Vocoder
    if voc_s:
        t0 = time.perf_counter()
        codes_arr = np.array(all_codes, dtype=np.int64)
        vi = voc_s.get_inputs()[0].name
        wav = voc_s.run(None, {vi: codes_arr.T[np.newaxis, :, :]})[0].flatten()
        tv = (time.perf_counter() - t0) * 1000
        with wave.open(output, "w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
            wf.writeframes((wav * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
    else:
        tv = 0
        np.save("/tmp/tts_codes.npy", np.array(all_codes, dtype=np.int64))

    total = (time.perf_counter() - t_total) * 1000
    trt_label = "TRT-GPU" if talker_trt else "CUDA"
    print(f"\n  === TIMING ({n} frames, {dur:.1f}s audio) ===")
    print(f"  Prefill:     {pf_ms:>7.0f}ms (CUDA)")
    print(f"  Talker/step: {da:>7.1f}ms ({trt_label})")
    cp_label = "TRT-INT8" if cp_trt_engine else "IOBind"
    print(f"  CP/step:     {ca:>7.1f}ms ({cp_label})")
    print(f"  Per-step:    {da+ca:>7.1f}ms  RTF={(da+ca)/80:.2f}")
    if voc_s:
        print(f"  Vocoder:     {tv:>7.0f}ms (CUDA)")
    print(f"  Total:       {total:>7.0f}ms  RTF={total/1000/dur:.2f}")
    if voc_s:
        print(f"  Saved: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="Hello, welcome to the voice synthesis system.")
    p.add_argument("--lang", default="english", choices=list(LANG_IDS.keys()))
    p.add_argument("--output", default="/tmp/tts_sherpa_trt.wav")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    synthesize(a.text, a.lang, a.output, a.max_frames, a.seed)
