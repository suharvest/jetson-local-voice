"""Matcha TTS backend via TensorRT (Jetson iGPU).

Supports: BASIC_TTS, STREAMING
Models: encoder + estimator (N=3) + vocos, all BF16/FP16 TRT engines.

Uses pycuda-style cuda-python bindings initialized AFTER TRT loads.
"""

from __future__ import annotations

import ctypes
import io
import logging
import os
import struct
import time
import numpy as np
from typing import Optional

from tts_backend import TTSBackend, TTSCapability

logger = logging.getLogger(__name__)

# Paths
_LANGUAGE_MODE = os.environ.get("LANGUAGE_MODE", "zh_en")
_MODEL_BASE = os.environ.get("MATCHA_MODEL_BASE", "/opt/models/matcha-icefall-zh-en")
MATCHA_ENCODER_ENGINE = os.environ.get(
    "MATCHA_ENCODER_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "matcha_encoder_bf16.engine")
)
MATCHA_ESTIMATOR_ENGINE = os.environ.get(
    "MATCHA_ESTIMATOR_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "matcha_estimator_n3_bf16.engine")
)
VOCOS_ENGINE = os.environ.get(
    "VOCOS_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "vocos_fp16.engine")
)
LEXICON_PATH = os.environ.get("LEXICON_PATH", os.path.join(_MODEL_BASE, "lexicon.txt"))
TOKENS_PATH = os.environ.get("TOKENS_PATH", os.path.join(_MODEL_BASE, "tokens.txt"))

# Audio constants
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256

# Model constants
MAX_MEL_FRAMES = 600
MEL_DIM = 80
TIME_EMB_DIM = 256
N_TIME_BLOCKS = 6
N_ODE_STEPS = 3
ODE_DT = 1.0 / N_ODE_STEPS

MEL_SIGMA = 5.446792
MEL_MEAN = -2.9521978


def _samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 samples to WAV bytes."""
    buf = io.BytesIO()
    num_samples = len(samples)
    data_size = num_samples * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    arr = np.clip(samples, -1.0, 1.0)
    buf.write((arr * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def _compute_time_embedding(t: float, dim: int = TIME_EMB_DIM) -> np.ndarray:
    """Compute sinusoidal time embedding for ODE step."""
    half_dim = dim // 2
    emb = np.log(10000.0) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype=np.float32) * -emb)
    emb = t * emb
    emb = np.concatenate([np.sin(emb), np.cos(emb)], dtype=np.float32)
    return emb.reshape(1, dim, 1)


# Periodic Hann window matches torch.hann_window(N_FFT, periodic=True) exactly.
_HANN_PERIODIC = np.hanning(N_FFT + 1)[:-1].astype(np.float32)


def _istft(mag: np.ndarray, x: np.ndarray, y: np.ndarray, length: Optional[int] = None) -> np.ndarray:
    """torch.istft-equivalent NumPy ISTFT (center=True, periodic Hann, COLA).

    mag/x/y: [513, T] float32 (Vocos outputs).
    """
    spec = (mag * (x + 1j * y)).astype(np.complex64)  # [F, T]
    n_frames = spec.shape[1]
    # IRFFT per time frame: [F, T] -> [N_FFT, T]
    frames = np.fft.irfft(spec, n=N_FFT, axis=0).astype(np.float32)
    # Apply window in time domain
    frames = frames * _HANN_PERIODIC[:, None]
    sq_window = (_HANN_PERIODIC ** 2).astype(np.float32)

    # OLA reconstruction
    output_len = (n_frames - 1) * HOP_LENGTH + N_FFT
    audio = np.zeros(output_len, dtype=np.float32)
    win_sum = np.zeros(output_len, dtype=np.float32)
    for i in range(n_frames):
        start = i * HOP_LENGTH
        audio[start:start + N_FFT] += frames[:, i]
        win_sum[start:start + N_FFT] += sq_window
    audio = audio / np.maximum(win_sum, 1e-11)

    # center=True: trim N_FFT//2 from each side
    pad = N_FFT // 2
    audio = audio[pad:output_len - pad]

    if length is not None:
        if len(audio) > length:
            audio = audio[:length]
        elif len(audio) < length:
            audio = np.pad(audio, (0, length - len(audio)))

    return audio


class MatchaTRTBackend(TTSBackend):
    """Matcha TTS via TensorRT (encoder + estimator + vocos)."""

    def __init__(self):
        self._encoder_engine = None
        self._encoder_ctx = None
        self._estimator_engine = None
        self._estimator_ctx = None
        self._vocos_engine = None
        self._vocos_ctx = None
        self._cuda_pool = None  # CudaMemoryPool instance
        self._lexicon = None
        self._token_to_id = None
        self._ready = False

    @property
    def name(self) -> str:
        return "matcha_trt"

    @property
    def capabilities(self) -> set[TTSCapability]:
        return {TTSCapability.BASIC_TTS}

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        self._load_lexicon()
        self._load_engines()
        self._warmup()
        self._ready = True

    def _load_lexicon(self):
        """Load lexicon.txt and tokens.txt."""
        self._lexicon = {}
        if os.path.exists(LEXICON_PATH):
            with open(LEXICON_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self._lexicon[parts[0]] = parts[1:]
            logger.info("Loaded %d lexicon entries from %s", len(self._lexicon), LEXICON_PATH)

        self._token_to_id = {}
        if os.path.exists(TOKENS_PATH):
            with open(TOKENS_PATH, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        self._token_to_id[parts[0]] = i + 1
            logger.info("Loaded %d tokens from %s", len(self._token_to_id), TOKENS_PATH)

    def _load_engines(self):
        """Load TRT engines FIRST, then initialize CUDA memory pool."""
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)

        def load_engine(path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Engine not found: {path}")
            with open(path, "rb") as f:
                runtime = trt.Runtime(trt_logger)
                engine = runtime.deserialize_cuda_engine(f.read())
            return engine

        # Load TRT engines first (this initializes CUDA context via runtime API)
        t0 = time.time()
        self._encoder_engine = load_engine(MATCHA_ENCODER_ENGINE)
        self._encoder_ctx = self._encoder_engine.create_execution_context()
        logger.info("Encoder loaded: %s (%.1fs)", MATCHA_ENCODER_ENGINE, time.time() - t0)

        t0 = time.time()
        self._estimator_engine = load_engine(MATCHA_ESTIMATOR_ENGINE)
        self._estimator_ctx = self._estimator_engine.create_execution_context()
        logger.info("Estimator loaded: %s (%.1fs)", MATCHA_ESTIMATOR_ENGINE, time.time() - t0)

        t0 = time.time()
        self._vocos_engine = load_engine(VOCOS_ENGINE)
        self._vocos_ctx = self._vocos_engine.create_execution_context()
        logger.info("Vocos loaded: %s (%.1fs)", VOCOS_ENGINE, time.time() - t0)

        # Now initialize CUDA memory pool (after TRT has initialized CUDA)
        self._cuda_pool = CudaMemoryPool()

    def _warmup(self):
        """Warmup inference."""
        texts = ["你好", "你好世界"]
        start = time.time()
        for t in texts:
            self.synthesize(t)
        logger.info("Warmup: %.1fs", time.time() - start)

    STRESS_MARKS = "ˈˌː"  # Primary stress, secondary stress, length mark

    def _phonemize_english(self, text: str) -> list[str]:
        """Phonemize English via piper-phonemize (same lib as sherpa-onnx)."""
        import piper_phonemize
        sentences = piper_phonemize.phonemize_espeak(text, "en-us")
        if not sentences:
            logger.warning("piper-phonemize returned empty for: %r", text)
            return []
        out = []
        for phoneme_list in sentences:
            for ph in phoneme_list:
                if ph == " ":
                    continue
                ph_stripped = ph.lstrip(self.STRESS_MARKS)
                if not ph_stripped:
                    continue
                if ph_stripped in self._token_to_id:
                    out.append(ph_stripped)
                else:
                    for ch in ph_stripped:
                        if ch in self._token_to_id:
                            out.append(ch)
                        else:
                            logger.warning("Unknown phoneme: %r (from %r)", ch, ph)
        return out

    def _text_to_tokens(self, text: str) -> list[int]:
        """Convert text to token IDs via lexicon (zh) + piper-phonemize (en)."""
        import re
        tokens = []

        segments = re.findall(
            r'[一-鿿]+|[A-Za-z][A-Za-z\' ]*[A-Za-z]|[A-Za-z]|[^一-鿿A-Za-z]+',
            text
        )

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            if re.match(r'^[一-鿿]+$', seg):
                tokens.extend(self._chinese_to_tokens(seg))
            elif re.match(r'^[A-Za-z]', seg):
                phonemes = self._phonemize_english(seg)
                for p in phonemes:
                    tokens.append(self._token_to_id[p])
                if not phonemes:
                    logger.warning("Empty phonemes for English seg %r", seg)

        return tokens

    def _chinese_to_tokens(self, text: str) -> list[int]:
        """Convert Chinese text via lexicon lookup."""
        tokens = []
        i = 0
        while i < len(text):
            found = False
            for length in range(min(4, len(text) - i), 0, -1):
                word = text[i:i+length]
                if word in self._lexicon:
                    phonemes = self._lexicon[word]
                    for p in phonemes:
                        if p in self._token_to_id:
                            tokens.append(self._token_to_id[p])
                    i += length
                    found = True
                    break
            if not found:
                char = text[i]
                if char in self._lexicon:
                    phonemes = self._lexicon[char]
                    for p in phonemes:
                        if p in self._token_to_id:
                            tokens.append(self._token_to_id[p])
                i += 1
        return tokens

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if speed is None:
            speed = 1.0

        pool = self._cuda_pool
        t_start = time.time()

        # Step 1: text → tokens
        t0 = time.time()
        tokens = self._text_to_tokens(text)
        text_ms = (time.time() - t0) * 1000
        if len(tokens) == 0:
            logger.warning("No tokens for text: %r", text)
            silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
            return _samples_to_wav(silence, SAMPLE_RATE), {"duration": 0.1, "inference_time": 0.0}

        num_tokens = min(len(tokens), 80)
        x = np.zeros((1, 80), dtype=np.int32)
        x[0, :num_tokens] = tokens[:num_tokens]
        x_length = np.array([num_tokens], dtype=np.int32)
        noise_scale = np.array([0.667], dtype=np.float32)
        length_scale = np.array([1.0 / speed], dtype=np.float32)
        z0 = np.random.randn(1, 80, 600).astype(np.float32) * noise_scale[0]

        # Allocate and copy
        def alloc(arr):
            ptr = pool.allocate(arr.nbytes)
            pool.copy_htod(arr, ptr)
            return ptr

        t0 = time.time()
        d_noise = alloc(noise_scale)
        d_len = alloc(length_scale)
        d_z0 = alloc(z0)
        d_x = alloc(x)
        d_xlen = alloc(x_length)

        mu = np.zeros((1, 80, 600), dtype=np.float32)
        mask = np.zeros((1, 1, 600), dtype=np.float32)
        z = np.zeros((1, 80, 600), dtype=np.float32)

        d_mu = pool.allocate(mu.nbytes)
        d_mask = pool.allocate(mask.nbytes)
        d_z_out = pool.allocate(z.nbytes)

        # Encoder
        self._encoder_ctx.set_tensor_address("noise_scale", d_noise)
        self._encoder_ctx.set_tensor_address("length_scale", d_len)
        self._encoder_ctx.set_tensor_address("z0_noise", d_z0)
        self._encoder_ctx.set_tensor_address("x", d_x)
        self._encoder_ctx.set_tensor_address("x_length", d_xlen)
        self._encoder_ctx.set_tensor_address("/Transpose_3_output_0", d_mu)
        self._encoder_ctx.set_tensor_address("/Cast_3_output_0", d_mask)
        self._encoder_ctx.set_tensor_address("/decoder/Mul_output_0", d_z_out)
        self._encoder_ctx.execute_async_v3(pool.stream_handle())
        pool.synchronize()
        pool.copy_dtoh(d_mu, mu)
        pool.copy_dtoh(d_mask, mask)
        pool.copy_dtoh(d_z_out, z)
        encoder_ms = (time.time() - t0) * 1000

        # Estimator ODE loop
        t0 = time.time()
        for step in range(N_ODE_STEPS):
            t_val = step * ODE_DT
            time_embs = [_compute_time_embedding(t_val + i * ODE_DT / N_TIME_BLOCKS)
                         for i in range(N_TIME_BLOCKS)]

            d_z_in = alloc(z)
            d_mu_in = alloc(mu)
            d_mask_in = alloc(mask)

            self._estimator_ctx.set_tensor_address("z", d_z_in)
            self._estimator_ctx.set_tensor_address("mu", d_mu_in)
            self._estimator_ctx.set_tensor_address("mask", d_mask_in)

            d_time_embs = [alloc(te) for te in time_embs]
            for i, d_te in enumerate(d_time_embs):
                self._estimator_ctx.set_tensor_address(f"time_emb_{i}", d_te)

            velocity = np.zeros((1, 80, 600), dtype=np.float32)
            d_vel = pool.allocate(velocity.nbytes)
            self._estimator_ctx.set_tensor_address("velocity", d_vel)
            self._estimator_ctx.execute_async_v3(pool.stream_handle())
            pool.synchronize()
            pool.copy_dtoh(d_vel, velocity)
            z = z + ODE_DT * velocity

        estimator_ms = (time.time() - t0) * 1000

        # Denormalize + clip mel (matches rkvoice matcha.py:429-430)
        mel = z * MEL_SIGMA + MEL_MEAN
        mel = np.clip(mel, -25.0, 8.0)

        # Derive mel_frames from encoder mask (1=valid, 0=padding).
        # Heuristic 11.9·num_tokens+51 saturates at 600 for English long inputs.
        mask_valid = int(mask[0, 0, :].astype(np.float32).sum() + 0.5)
        if mask_valid <= 0:
            est_frames = int((11.9 * num_tokens + 51) * length_scale[0] * 1.2 + 0.5)
            mel_frames = min(est_frames, MAX_MEL_FRAMES)
            logger.warning("matcha mask sum=0, falling back to heuristic mel_frames=%d", mel_frames)
        else:
            mel_frames = min(mask_valid, MAX_MEL_FRAMES)
        logger.debug("matcha frames: tokens=%d mask=%d mel_frames=%d (~%.2fs)",
                     num_tokens, mask_valid, mel_frames, mel_frames * HOP_LENGTH / SAMPLE_RATE)

        # Vocos
        t0 = time.time()
        mel_input = mel[:, :, :mel_frames].astype(np.float32)
        d_mel = alloc(mel_input)
        self._vocos_ctx.set_tensor_address("mels", d_mel)
        self._vocos_ctx.set_input_shape("mels", (1, MEL_DIM, mel_frames))

        mag = np.zeros((1, 513, mel_frames), dtype=np.float32)
        out_x = np.zeros((1, 513, mel_frames), dtype=np.float32)
        out_y = np.zeros((1, 513, mel_frames), dtype=np.float32)

        d_mag = pool.allocate(mag.nbytes)
        d_x_out = pool.allocate(out_x.nbytes)
        d_y_out = pool.allocate(out_y.nbytes)

        self._vocos_ctx.set_tensor_address("mag", d_mag)
        self._vocos_ctx.set_tensor_address("x", d_x_out)
        self._vocos_ctx.set_tensor_address("y", d_y_out)
        self._vocos_ctx.execute_async_v3(pool.stream_handle())
        pool.synchronize()

        pool.copy_dtoh(d_mag, mag)
        pool.copy_dtoh(d_x_out, out_x)
        pool.copy_dtoh(d_y_out, out_y)
        vocos_ms = (time.time() - t0) * 1000

        # ISTFT (length matches mel_frames * HOP_LENGTH)
        audio = _istft(mag[0], out_x[0], out_y[0], length=mel_frames * HOP_LENGTH)

        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.95

        elapsed = time.time() - t_start
        duration = len(audio) / SAMPLE_RATE
        wav_bytes = _samples_to_wav(audio.astype(np.float32), SAMPLE_RATE)

        meta = {
            "duration": round(duration, 3),
            "inference_time": round(elapsed, 3),
            "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
            "sample_rate": SAMPLE_RATE,
            "num_tokens": num_tokens,
            "text_ms": round(text_ms, 1),
            "encoder_ms": round(encoder_ms, 1),
            "estimator_ms": round(estimator_ms, 1),
            "vocos_ms": round(vocos_ms, 1),
        }
        return wav_bytes, meta


class CudaMemoryPool:
    """CUDA memory pool using cuda-python runtime API (initialized after TRT loads)."""

    @staticmethod
    def _cuda_err(result):
        """Normalize cuda-python return value to cudaError_t."""
        if isinstance(result, tuple):
            return result[0]
        return result

    def __init__(self):
        self._stream = None
        self._allocations = []
        self._initialized = False

    def _init_cuda(self):
        """Initialize CUDA runtime after TRT has loaded."""
        if self._initialized:
            return

        from cuda import cudart

        # TRT has already initialized CUDA runtime context
        # Just create a stream using runtime API
        err, self._stream = cudart.cudaStreamCreate()
        if self._cuda_err(err) != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamCreate failed: {err}")

        self._initialized = True
        logger.info("CudaMemoryPool initialized with stream %d", int(self._stream))

    def allocate(self, size_bytes: int) -> int:
        """Allocate device memory."""
        self._init_cuda()
        from cuda import cudart
        err, ptr = cudart.cudaMalloc(size_bytes)
        if self._cuda_err(err) != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc({size_bytes}) failed: {err}")
        self._allocations.append(ptr)
        return int(ptr)

    def copy_htod(self, host_arr: np.ndarray, dev_ptr: int):
        """Copy host to device."""
        self._init_cuda()
        from cuda import cudart
        err = cudart.cudaMemcpy(
            dev_ptr, host_arr.ctypes.data, host_arr.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        if self._cuda_err(err) != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpy H2D failed: {err}")

    def copy_dtoh(self, dev_ptr: int, host_arr: np.ndarray):
        """Copy device to host."""
        self._init_cuda()
        from cuda import cudart
        err = cudart.cudaMemcpy(
            host_arr.ctypes.data, dev_ptr, host_arr.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        if self._cuda_err(err) != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpy D2H failed: {err}")

    def synchronize(self):
        """Synchronize stream."""
        if self._stream is not None:
            from cuda import cudart
            err = cudart.cudaStreamSynchronize(self._stream)
            if self._cuda_err(err) != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaStreamSynchronize failed: {err}")

    def stream_handle(self) -> int:
        """Return stream handle as int for TRT."""
        self._init_cuda()
        return int(self._stream)