"""Matcha TTS backend via TensorRT (Jetson iGPU).

Supports: BASIC_TTS, STREAMING (chunked PCM from synthesized audio)
Models: ORT encoder + estimator (N=3) TRT + vocos TRT in split mode.

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

from app.core.language import detect_zh_en
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
SPLIT_ENCODER_ONNX = os.environ.get(
    "MATCHA_SPLIT_ENCODER_ONNX",
    os.path.join(_MODEL_BASE, "onnx", "matcha_encoder_trt.onnx"),
)
SPLIT_ESTIMATOR_ENGINE = os.environ.get(
    "MATCHA_SPLIT_ESTIMATOR_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "matcha_estimator_step0_bf16.engine"),
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
ODE_DT = 1.0 / 3.0
N_ODE_STEPS = 3
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
    """Matcha TTS backend.

    Default mode uses full acoustic ONNX via ORT plus TRT Vocos. Split mode
    keeps the data-dependent duration/encoder path in ORT and runs the three
    high-compute ODE estimator steps as TensorRT engines.
    """

    def __init__(self):
        self._acoustic_ort = None
        self._split_encoder_ort = None
        self._split_estimator_engines = []
        self._split_estimator_ctxs = []
        self._acoustic_mode = "full_ort"
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
        return {
            TTSCapability.BASIC_TTS,
            TTSCapability.STREAMING,
            TTSCapability.MULTI_LANGUAGE,
        }

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
        """Load acoustic frontend.

        Modes:
        - MATCHA_ACOUSTIC_EP=CPU: full acoustic ONNX on ORT CPU.
        - MATCHA_ACOUSTIC_EP=SPLIT_TRT: duration/encoder ONNX on ORT CPU,
          ODE estimator on TensorRT. This avoids TRT's data-dependent mel
          shape limitation while moving the heavy estimator block to an engine.
        """
        import onnxruntime as ort
        ep_override = os.environ.get("MATCHA_ACOUSTIC_EP", "").upper()
        if ep_override in ("SPLIT_TRT", "TRT_SPLIT", "HYBRID_TRT"):
            self._acoustic_mode = "split_trt"
            self._ensure_split_onnx()
            if not os.path.exists(SPLIT_ENCODER_ONNX):
                raise FileNotFoundError(
                    f"Split Matcha encoder ONNX not found: {SPLIT_ENCODER_ONNX}. "
                    "Generate it with scripts/split_matcha_trt.py."
                )
            sess_opt = ort.SessionOptions()
            sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._split_encoder_ort = ort.InferenceSession(
                SPLIT_ENCODER_ONNX, sess_opt, providers=["CPUExecutionProvider"]
            )
            logger.info("Split Matcha encoder ORT loaded: %s", SPLIT_ENCODER_ONNX)
            return

        path = os.path.join(_MODEL_BASE, "model-steps-3.onnx")
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

    def _ensure_split_onnx(self) -> None:
        """Generate fixed-path split ONNX artifacts from the full Matcha model."""
        estimator0 = os.path.join(
            os.path.dirname(SPLIT_ENCODER_ONNX),
            "matcha_estimator_step0_trt.onnx",
        )
        if os.path.exists(SPLIT_ENCODER_ONNX) and os.path.exists(estimator0):
            return

        full_onnx = os.environ.get("ACOUSTIC_ONNX") or os.path.join(_MODEL_BASE, "model-steps-3.onnx")
        if not os.path.exists(full_onnx):
            logger.warning("Cannot generate split Matcha ONNX; full model missing: %s", full_onnx)
            return

        out_dir = os.path.dirname(SPLIT_ENCODER_ONNX)
        os.makedirs(out_dir, exist_ok=True)
        logger.info("Generating split Matcha ONNX at %s from %s", out_dir, full_onnx)
        try:
            from scripts.split_matcha_trt import split_onnx
            from pathlib import Path

            split_onnx(Path(full_onnx), Path(out_dir))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate split Matcha ONNX at {out_dir} from {full_onnx}: {exc}"
            ) from exc

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

        if self._acoustic_mode == "split_trt":
            t0 = time.time()
            base_dir = os.path.dirname(SPLIT_ESTIMATOR_ENGINE)
            self._split_estimator_engines = []
            self._split_estimator_ctxs = []
            names = []
            for step in range(N_ODE_STEPS):
                path = os.path.join(base_dir, f"matcha_estimator_step{step}_bf16.engine")
                engine = load_engine(path)
                self._split_estimator_engines.append(engine)
                self._split_estimator_ctxs.append(engine.create_execution_context())
                names.append([engine.get_tensor_name(i) for i in range(engine.num_io_tensors)])
            logger.info(
                "Split Matcha estimator TRT loaded from %s (%.1fs, tensors=%s)",
                base_dir, time.time() - t0, names,
            )

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
        detected_language = detect_zh_en(text, language)

        pool = self._cuda_pool
        t_start = time.time()

        # Step 1: text → tokens
        t0 = time.time()
        tokens = self._text_to_tokens(text)
        text_ms = (time.time() - t0) * 1000
        if len(tokens) == 0:
            logger.warning("No tokens for text: %r", text)
            silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
            return _samples_to_wav(silence, SAMPLE_RATE), {
                "duration": 0.1,
                "inference_time": 0.0,
                "language": detected_language,
            }

        num_tokens = min(len(tokens), 80)
        t0 = time.time()
        x = np.array([tokens[:num_tokens]], dtype=np.int64)
        x_length = np.array([num_tokens], dtype=np.int64)
        noise_scale = np.array([1.0], dtype=np.float32)
        length_scale = np.array([1.0 / speed], dtype=np.float32)
        if self._acoustic_mode == "split_trt":
            mel = self._run_split_acoustic(x, x_length, noise_scale, length_scale)
        else:
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
            "language": detected_language,
        }
        pool.free_all()
        return wav_bytes, meta

    def generate_streaming(self, text: str, **kwargs):
        """Yield raw PCM int16 chunks.

        This is chunk-level streaming: synthesize the current sentence/segment
        with the same Matcha path used by /tts, strip the WAV header, then
        yield fixed-duration PCM chunks. The FastAPI layer already splits long
        text into sentences before calling this backend, so long requests can
        start playback after the first sentence is synthesized while preserving
        the offline Matcha/Vocos quality path.
        """
        synth_kwargs = {
            "speaker_id": kwargs.get("speaker_id", kwargs.get("sid")),
            "speed": kwargs.get("speed"),
            "pitch_shift": kwargs.get("pitch_shift", kwargs.get("pitch")),
            "language": kwargs.get("language"),
        }
        wav_bytes, _meta = self.synthesize(text, **synth_kwargs)
        if len(wav_bytes) <= 44:
            return

        try:
            chunk_ms = int(os.environ.get("MATCHA_STREAM_CHUNK_MS", "40"))
        except ValueError:
            chunk_ms = 40
        chunk_ms = max(10, min(200, chunk_ms))
        bytes_per_sample = 2
        chunk_bytes = max(
            bytes_per_sample,
            int(SAMPLE_RATE * chunk_ms / 1000) * bytes_per_sample,
        )

        pcm = wav_bytes[44:]
        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset:offset + chunk_bytes]
            if chunk:
                yield chunk

    def _run_split_acoustic(
        self,
        x: np.ndarray,
        x_length: np.ndarray,
        noise_scale: np.ndarray,
        length_scale: np.ndarray,
    ) -> np.ndarray:
        """Run Matcha acoustic as ORT encoder + TRT estimator ODE loop."""
        mu, mask, z = self._split_encoder_ort.run(None, {
            "x": x,
            "x_length": x_length,
            "noise_scale": noise_scale,
            "length_scale": length_scale,
        })
        mu = np.ascontiguousarray(mu.astype(np.float32))
        mask = np.ascontiguousarray(mask.astype(np.float32))
        z = np.ascontiguousarray(z.astype(np.float32))
        if z.shape[2] > MAX_MEL_FRAMES:
            mu = mu[:, :, :MAX_MEL_FRAMES]
            mask = mask[:, :, :MAX_MEL_FRAMES]
            z = z[:, :, :MAX_MEL_FRAMES]

        valid_frames = int(np.clip(np.rint(mask.sum()), 1, MAX_MEL_FRAMES))
        for step in range(N_ODE_STEPS):
            feeds = {"z": z, "mu": mu, "mask": mask}
            velocity = self._run_estimator_trt(step, feeds)
            z = z + ODE_DT * velocity

        mel = z[:, :, :valid_frames] * MEL_SIGMA + MEL_MEAN
        return mel.astype(np.float32)

    def _run_estimator_trt(self, step: int, feeds: dict[str, np.ndarray]) -> np.ndarray:
        pool = self._cuda_pool
        ctx = self._split_estimator_ctxs[step]

        def alloc_input(name: str, arr: np.ndarray) -> int:
            arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
            ptr = pool.allocate(arr.nbytes)
            pool.copy_htod(arr, ptr)
            ctx.set_tensor_address(name, ptr)
            try:
                ctx.set_input_shape(name, tuple(arr.shape))
            except Exception:
                pass
            return ptr

        for name, arr in feeds.items():
            alloc_input(name, arr)

        frames = int(feeds["z"].shape[2])
        velocity = np.empty((1, MEL_DIM, frames), dtype=np.float32)
        d_velocity = pool.allocate(velocity.nbytes)
        ctx.set_tensor_address("velocity", d_velocity)
        ok = ctx.execute_async_v3(pool.stream_handle())
        if not ok:
            raise RuntimeError("Matcha estimator TRT execute_async_v3 returned False")
        pool.synchronize()
        pool.copy_dtoh(d_velocity, velocity)
        return velocity


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

    def free_all(self):
        """Free per-request device allocations while keeping the stream warm."""
        if not self._allocations:
            return
        from cuda import cudart
        for ptr in self._allocations:
            cudart.cudaFree(ptr)
        self._allocations.clear()

    def stream_handle(self) -> int:
        """Return stream handle as int for TRT."""
        self._init_cuda()
        return int(self._stream)
