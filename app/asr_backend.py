"""ASR backend abstraction with capability discovery.

Mirrors the TTS backend pattern (tts_backend.py).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ASRCapability(str, Enum):
    OFFLINE = "offline"
    STREAMING = "streaming"
    TIMESTAMPS = "timestamps"
    MULTI_LANGUAGE = "multi_language"
    LANGUAGE_ID = "language_id"


class TranscriptionResult:
    def __init__(self, text: str, language: Optional[str] = None, **meta):
        self.text = text
        self.language = language
        self.meta = meta


class ASRStream(ABC):
    """A streaming ASR session that accumulates audio and produces text."""

    @abstractmethod
    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        """Feed audio samples (float32, [-1,1]) into the stream."""
        ...

    @abstractmethod
    def finalize(self) -> str:
        """Signal end-of-audio and return final transcription text."""
        ...

    def get_partial(self) -> tuple[str, bool]:
        """Return (partial_text, is_endpoint). Default: no partial results."""
        return "", False

    def prepare_finalize(self) -> None:
        """Pre-encode remaining audio buffer so finalize() only runs decoder.

        Optional optimization — finalize() works without calling this first.
        """
        pass


class ASRBackend(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def capabilities(self) -> set[ASRCapability]: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    def preload(self) -> None: ...

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult: ...

    def create_stream(self, language: str = "auto") -> ASRStream:
        """Create a streaming ASR session. Requires STREAMING capability."""
        raise NotImplementedError(f"{self.name} does not support streaming")

    def has_capability(self, cap: ASRCapability) -> bool:
        return cap in self.capabilities


def create_asr_backend(backend_name: Optional[str] = None) -> ASRBackend:
    """Factory: create ASR backend by name.

    Auto-detect logic:
        - If LANGUAGE_MODE=multilanguage → qwen3 (52 languages)
        - Otherwise, use ASR_BACKEND env var (default: sherpa)
    """
    if backend_name is None:
        # Check LANGUAGE_MODE for automatic backend selection
        language_mode = os.environ.get("LANGUAGE_MODE", "zh_en")
        if language_mode == "multilanguage":
            backend_name = "qwen3"
            logger.info("LANGUAGE_MODE=multilanguage → using qwen3 ASR backend")
        else:
            backend_name = os.environ.get("ASR_BACKEND", "sherpa")

    if backend_name == "sherpa":
        from backends.sherpa_asr import SherpaASRBackend
        return SherpaASRBackend()
    elif backend_name == "qwen3":
        # Try importing from standalone package first, fallback to local
        try:
            from jetson_qwen3_speech import Qwen3ASRBackend
            logger.info("Using Qwen3ASRBackend from jetson-qwen3-speech package")
        except ImportError:
            from backends.qwen3_asr import Qwen3ASRBackend
            logger.info("Using Qwen3ASRBackend from local backends/")
        return Qwen3ASRBackend()
    else:
        raise ValueError(f"Unknown ASR backend: {backend_name}")
