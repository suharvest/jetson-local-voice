"""Qwen3-ASR RK3576 backend: wraps qwen3asr library as ASRBackend.

NPU access (encoder + decoder) is serialized via a module-level lock
shared with the TTS backend to prevent contention on RK3576.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
from typing import Optional

import numpy as np

# Import from parent package (app/)
from asr_backend import ASRBackend, ASRCapability, ASRStream, TranscriptionResult

logger = logging.getLogger(__name__)

# Shared NPU lock — imported by TTS backend too when available
_npu_lock: Optional[threading.Lock] = None


def get_npu_lock() -> threading.Lock:
    """Return the shared NPU lock, creating it on first call."""
    global _npu_lock
    if _npu_lock is None:
        _npu_lock = threading.Lock()
    return _npu_lock


class Qwen3ASRRKBackend(ASRBackend):
    """ASR backend using Qwen3-ASR RKNN/RKLLM on RK3576."""

    def __init__(self):
        self._engine = None
        self._ready = False

    @property
    def name(self) -> str:
        return "qwen3_asr_rk"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.STREAMING, ASRCapability.MULTI_LANGUAGE}

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready and self._engine is not None

    def preload(self) -> None:
        # Ensure qwen3asr package is findable (lives in app/)
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)

        from qwen3asr import Qwen3ASREngine

        model_dir = os.environ.get("ASR_MODEL_DIR", "/opt/asr/models")
        logger.info("Loading Qwen3-ASR engine from %s", model_dir)

        # lib_path: librkllmrt.so is installed in /usr/lib/ inside the container
        lib_path = os.environ.get("RKLLM_LIB_PATH", "/usr/lib/librkllmrt.so")

        self._engine = Qwen3ASREngine(
            model_dir=model_dir,
            platform="rk3576",
            lib_path=lib_path,
            decoder_quant="w4a16",      # decoder_hf.w4a16.rk3576.rkllm
            encoder_sizes=[4],          # 4s encoder only — saves NPU memory
            enabled_cpus=2,
            repeat_penalty=1.15,
            compact_suffix=True,
            verbose=True,
        )
        self._ready = True
        logger.info("Qwen3-ASR RK backend ready.")

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if not self.is_ready():
            raise RuntimeError("ASR backend not ready")

        audio = self._decode_audio(audio_bytes)
        # language=None (auto-detect) generates EOS immediately with this decoder;
        # fall back to Chinese for now as default when auto is requested.
        lang_hint = "Chinese" if language == "auto" else language

        lock = get_npu_lock()
        with lock:
            result = self._engine.transcribe(
                audio=audio,
                language=lang_hint,
                chunk_size=4.0,
                memory_num=2,
                rollback_tokens=2,
            )

        return TranscriptionResult(
            text=result["text"],
            language=result.get("language"),
            rtf=result.get("stats", {}).get("rtf"),
            enc_ms=result.get("stats", {}).get("enc_ms"),
            llm_ms=result.get("stats", {}).get("llm_ms"),
        )

    def create_stream(self, language: str = "auto") -> ASRStream:
        if not self.is_ready():
            raise RuntimeError("ASR backend not ready")

        lang_hint = "Chinese" if language == "auto" else language
        stream_session = self._engine.create_stream(
            language=lang_hint,
            chunk_size=4.0,
            memory_num=2,
            rollback_tokens=2,
        )
        return Qwen3ASRRKStream(stream_session)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_audio(audio_bytes: bytes) -> np.ndarray:
        """Decode audio_bytes (WAV/FLAC/etc.) to 16kHz float32 mono numpy."""
        import soundfile as sf

        buf = io.BytesIO(audio_bytes)
        try:
            audio, sr = sf.read(buf, dtype="float32")
        except Exception as exc:
            raise ValueError(f"Cannot decode audio: {exc}") from exc

        # Mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed (simple linear interpolation)
        if sr != 16000:
            logger.warning("Input sample rate %d != 16000, resampling.", sr)
            audio = _resample(audio, sr, 16000)

        return audio.astype(np.float32)


class Qwen3ASRRKStream(ASRStream):
    """Wraps StreamSession as ASRStream interface."""

    def __init__(self, stream_session):
        self._stream = stream_session
        self._lock = get_npu_lock()

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        """Feed float32 audio (already in [-1,1]) into stream."""
        audio = samples.astype(np.float32)

        if samples.ndim > 1:
            audio = audio.mean(axis=1)

        if sample_rate != 16000:
            audio = _resample(audio, sample_rate, 16000)

        with self._lock:
            self._stream.feed_audio(audio)

    def finalize(self) -> str:
        with self._lock:
            result = self._stream.finish()
        return result["text"]

    def get_partial(self) -> tuple[str, bool]:
        result = self._stream.get_result()
        return result["text"], False


# ------------------------------------------------------------------
# Simple resampler (no librosa dependency)
# ------------------------------------------------------------------

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1-D float32 audio array using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(round(duration * target_sr))
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, audio).astype(np.float32)
