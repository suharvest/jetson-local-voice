"""Qwen3-ASR backend — ORT encoder/prefill + TRT decoder.

Supports: OFFLINE, MULTI_LANGUAGE, LANGUAGE_ID
Models loaded once at preload(), stays resident.
"""

from __future__ import annotations

import io
import logging
import os
import struct
import time
import wave
from typing import Optional

import numpy as np

from asr_backend import ASRBackend, ASRCapability, TranscriptionResult

logger = logging.getLogger(__name__)

_BASE = os.environ.get("QWEN3_ASR_MODEL_BASE", "/opt/models/qwen3-asr-v2")


class Qwen3ASRBackend(ASRBackend):

    def __init__(self):
        self._pipeline = None
        self._ready = False

    @property
    def name(self) -> str:
        return "qwen3_asr"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.MULTI_LANGUAGE, ASRCapability.LANGUAGE_ID}

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        import sys
        # Add benchmark dir to path for the pipeline module
        benchmark_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark")
        if os.path.exists(benchmark_dir):
            sys.path.insert(0, benchmark_dir)

        # Import inline to avoid top-level deps
        from asr_qwen3_trt import Qwen3ASRPipeline

        logger.info("Loading Qwen3-ASR from %s", _BASE)
        t0 = time.time()
        self._pipeline = Qwen3ASRPipeline(_BASE)
        logger.info("Qwen3-ASR loaded in %.1fs", time.time() - t0)
        self._ready = True

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        audio = self._bytes_to_float(audio_bytes)
        lang = language if language != "auto" else None
        text, meta = self._pipeline.transcribe(audio, language=lang)
        return TranscriptionResult(
            text=text,
            language=meta.get("language"),
            **meta,
        )

    @staticmethod
    def _bytes_to_float(audio_bytes: bytes) -> np.ndarray:
        """Convert WAV bytes to float32 numpy array at 16kHz."""
        try:
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        except ImportError:
            bio = io.BytesIO(audio_bytes)
            with wave.open(bio) as w:
                sr = w.getframerate()
                raw = w.readframes(w.getnframes())
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(indices, np.arange(len(audio)), audio)
        return audio.astype(np.float32)
