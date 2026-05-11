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

from app.core.tts_backend import TTSBackend, TTSCapability

logger = logging.getLogger(__name__)

# Paths
_LANGUAGE_MODE = os.environ.get("LANGUAGE_MODE", "zh_en")
_MODEL_BASE = os.environ.get("MATCHA_MODEL_BASE", "/opt/models/matcha-icefall-zh-en")
VOCOS_ENGINE = os.environ.get(
    "VOCOS_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "vocos_fp16.engine")  # actually BF16 — kept filename for compat
)
ACOUSTIC_ONNX = os.environ.get(
    "ACOUSTIC_ONNX",
    os.path.join(_MODEL_BASE, "model-steps-3.onnx")
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


_HANN_PERIODIC = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic Hann, matches sherpa vocos-vocoder.cc:92-140


def _istft(mag: np.ndarray, x: np.ndarray, y: np.ndarray, length: Optional[int] = None) -> np.ndarray:
    """ISTFT matching sherpa-onnx vocos pipeline (knf::StftConfig center=1).

    mag/x/y: [513, T] float32 (Vocos outputs). complex = mag * (cos + j sin).
    center=True: trim N_FFT//2 from each end of OLA output (matches sherpa
    vocos-vocoder.cc:161-172).
    """
    complex_spec = (mag * (x + 1j * y)).astype(np.complex64)  # [F, T]
    n_frames = complex_spec.shape[1]
    output_len = (n_frames - 1) * HOP_LENGTH + N_FFT

    audio = np.zeros(output_len, dtype=np.float32)
    win_sum = np.zeros(output_len, dtype=np.float32)
    sq_window = (_HANN_PERIODIC ** 2).astype(np.float32)
    for i in range(n_frames):
        frame = np.fft.irfft(complex_spec[:, i], n=N_FFT).astype(np.float32) * _HANN_PERIODIC
        start = i * HOP_LENGTH
        audio[start:start + N_FFT] += frame
        win_sum[start:start + N_FFT] += sq_window
    audio = audio / np.maximum(win_sum, 1e-8)

    # center=True: trim N_FFT//2 padding from each end
    pad = N_FFT // 2
    audio = audio[pad:-pad] if pad > 0 and len(audio) > 2 * pad else audio

    if length is not None:
        if len(audio) > length:
            audio = audio[:length]
        elif len(audio) < length:
            audio = np.pad(audio, (0, length - len(audio)))
    return audio


class MatchaTRTBackend(TTSBackend):
    """Matcha TTS — ORT-CPU acoustic (model-steps-3.onnx) + TRT vocos.

    The original split (encoder TRT + estimator TRT + ODE loop) produced
    subtly wrong mel due to time_emb npy mismatch with current model
    variant. We use the baked acoustic ONNX via ORT-CPU directly
    (sherpa-equivalent path), then BF16 TRT vocos for final audio.
    """

    def __init__(self):
        self._acoustic_ort = None
        self._vocos_engine = None
        self._vocos_ctx = None
        self._cuda_pool = None
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
        self._load_acoustic_ort()
        self._load_engines()
        self._warmup()
        self._ready = True

    def _load_acoustic_ort(self):
        """Load baked model-steps-3.onnx via ORT with CUDA EP.

        Bypasses split TRT estimator due to time_emb npy mismatch with
        current model variant. Single-shot ORT call with CUDA EP gives
        GPU acceleration without the TRT split accuracy risk.
        """
        import onnxruntime as ort
        path = os.path.join(_MODEL_BASE, "model-steps-3.onnx")
        ep_override = os.environ.get("MATCHA_ACOUSTIC_EP", "").upper()
        if ep_override == "CPU":
            providers = ["CPUExecutionProvider"]
        else:
            providers = [
                ("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"}),
                "CPUExecutionProvider",
            ]
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._acoustic_ort = ort.InferenceSession(path, sess_opt, providers=providers)
        logger.info("Acoustic ORT loaded (%s): %s",
                     self._acoustic_ort.get_providers()[0], path)

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
                for line in f:
                    raw = line.rstrip("\n").rstrip("\r")
                    if not raw:
                        continue
                    # Format: "<token><ws><id>". Token may itself be a single space.
                    rsep = max(raw.rfind(" "), raw.rfind("\t"))
                    if rsep < 0:
                        continue
                    tok = raw[:rsep] or " "  # leading-whitespace line → space token
                    try:
                        tid = int(raw[rsep + 1:])
                    except ValueError:
                        continue
                    self._token_to_id[tok] = tid
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

        t0 = time.time()
        self._vocos_engine = load_engine(VOCOS_ENGINE)
        self._vocos_ctx = self._vocos_engine.create_execution_context()
        logger.info("Vocos loaded: %s (%.1fs)", VOCOS_ENGINE, time.time() - t0)

        self._cuda_pool = CudaMemoryPool()

    def _warmup(self):
        """Warmup inference."""
        texts = ["你好", "你好世界"]
        start = time.time()
        for t in texts:
            self.synthesize(t)
        logger.info("Warmup: %.1fs", time.time() - start)

    # English IPA → tokens.txt phoneme replacement table.
    # Mirrors sherpa-onnx matcha-tts-lexicon.cc:44-57 (MatchaTtsLexicon for zh+en).
    # Applied as ordered string replace BEFORE per-codepoint token lookup.
    # Diphthongs map to ASCII letters that exist in tokens.txt as single tokens.
    _IPA_REPLACEMENTS = [
        ("eɪ", "A"), ("aɪ", "I"), ("ɔɪ", "Y"),
        ("oʊ", "O"), ("əʊ", "O"), ("aʊ", "W"),
        ("tʃ", "ʧ"), ("dʒ", "ʤ"),
        ("ɝ", "ɜɹ"), ("ɚ", "əɹ"),
        ("g", "ɡ"), ("r", "ɹ"), ("e", "ɛ"),
        ("ː", ""),  # length mark deleted (not in vocab)
    ]

    def _phonemize_english(self, text: str) -> list[str]:
        """Phonemize English via piper-phonemize, then sherpa replacement table.

        sherpa-onnx MatchaTtsLexicon (matcha-tts-lexicon.cc) joins IPA
        codepoints, applies _IPA_REPLACEMENTS, then splits per Unicode
        codepoint and looks up each in tokens.txt (silently skip unknowns).
        Stress marks ˈˌ are NOT in the table — kept and looked up directly.
        """
        import piper_phonemize
        sentences = piper_phonemize.phonemize_espeak(text, "en-us")
        if not sentences:
            logger.warning("piper-phonemize returned empty for: %r", text)
            return []
        out = []
        for sent_idx, phoneme_list in enumerate(sentences):
            if sent_idx > 0 and " " in self._token_to_id:
                out.append(" ")
            joined = "".join(p for p in phoneme_list if p)
            for src, dst in self._IPA_REPLACEMENTS:
                joined = joined.replace(src, dst)
            for cp in joined:
                if cp in self._token_to_id:
                    out.append(cp)
                # else: silently skip (sherpa behavior)
        return out

    def _text_to_tokens(self, text: str) -> list[int]:
        """Convert text to token IDs via lexicon (zh) + piper-phonemize (en).

        Inserts space token between consecutive English words (sherpa
        matcha-tts-lexicon.cc:283-287 inserts ' ' before next word when
        previous word started with ASCII alpha).
        """
        import re
        tokens: list[int] = []
        space_id = self._token_to_id.get(" ")
        prev_was_english = False

        # Full-width → half-width punctuation (tokens.txt has ASCII
        # punctuation only; CJK variants must be mapped or the model
        # gets no prosody cues).
        _FW_PUNCT = {
            "，": ",", "。": ".", "！": "!", "？": "?",
            "、": ",", "；": ";", "：": ":",
            "（": "(", "）": ")", "［": "[", "］": "]",
            "【": "[", "】": "]", "〈": "<", "〉": ">",
            "《": "<", "》": ">",
        }

        segments = re.findall(
            r'[一-鿿]+|[A-Za-z][A-Za-z\' ]*[A-Za-z]|[A-Za-z]|[^一-鿿A-Za-z]+',
            text,
        )

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            if re.match(r'^[一-鿿]+$', seg):
                tokens.extend(self._chinese_to_tokens(seg))
                prev_was_english = False
            elif re.match(r'^[A-Za-z]', seg):
                if prev_was_english and space_id is not None:
                    tokens.append(space_id)
                phonemes = self._phonemize_english(seg)
                for p in phonemes:
                    tid = self._token_to_id.get(p)
                    if tid is not None:
                        tokens.append(tid)
                if not phonemes:
                    logger.warning("Empty phonemes for English seg %r", seg)
                prev_was_english = True
            else:
                # Punctuation / whitespace / other: map full-width to
                # half-width and look up each character in token table.
                for ch in seg:
                    mapped = _FW_PUNCT.get(ch, ch)
                    tid = self._token_to_id.get(mapped)
                    if tid is not None:
                        tokens.append(tid)

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
        # Use ORT-CPU acoustic (baked model-steps-3.onnx) — sherpa-equivalent.
        # TRT split estimator is bypassed because rkvoice-extracted time_emb npy
        # is from a different model variant; using it produces correct-shape but
        # subtly wrong mel ("字串"/字错位 in listening tests).
        t0 = time.time()
        x = np.array([tokens[:num_tokens]], dtype=np.int64)
        x_length = np.array([num_tokens], dtype=np.int64)
        noise_scale = np.array([1.0], dtype=np.float32)
        length_scale = np.array([1.0 / speed], dtype=np.float32)
        ao = self._acoustic_ort.run(None, {
            "x": x, "x_length": x_length,
            "noise_scale": noise_scale, "length_scale": length_scale,
        })
        mel = ao[0]  # [1, 80, T_mel] already denormalized (denorm baked into graph)
        encoder_ms = (time.time() - t0) * 1000
        estimator_ms = 0.0
        mel_frames = mel.shape[2]
        # Pad to MAX_MEL_FRAMES for vocos engine compat (drop excess if any)
        if mel.shape[2] > MAX_MEL_FRAMES:
            mel = mel[:, :, :MAX_MEL_FRAMES]
            mel_frames = MAX_MEL_FRAMES
        mask = None
        mask_valid = mel_frames

        def alloc(arr):
            ptr = pool.allocate(arr.nbytes)
            pool.copy_htod(arr, ptr)
            return ptr
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

        # No peak normalize — sherpa returns raw ISTFT (offline-tts-impl.cc:88-102 only int16-scales).
        # Clip to int16 range to prevent overflow on rare loud frames.
        audio = np.clip(audio, -1.0, 1.0)

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