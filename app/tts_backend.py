"""TTS backend abstraction with capability discovery.

Backends expose different capabilities (voice clone, streaming, etc.).
Clients check capabilities before calling optional endpoints.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TTSCapability(str, Enum):
    """Capabilities a TTS backend may support."""
    BASIC_TTS = "basic_tts"
    VOICE_CLONE = "voice_clone"
    VOICE_CLONE_ICL = "voice_clone_icl"
    STREAMING = "streaming"
    MULTI_SPEAKER = "multi_speaker"
    MULTI_LANGUAGE = "multi_language"


class TTSBackend(ABC):
    """Base class for all TTS backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g. 'sherpa', 'qwen3_trt')."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> set[TTSCapability]:
        """Set of supported capabilities."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def preload(self) -> None:
        """Load models and warm up. Called once at startup."""
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Synthesize text to WAV bytes. Returns (wav_bytes, metadata)."""
        ...

    def clone_voice(
        self,
        text: str,
        speaker_embedding: bytes,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Synthesize with voice cloning. Requires VOICE_CLONE capability."""
        raise NotImplementedError(
            f"Backend '{self.name}' does not support voice cloning"
        )

    def extract_speaker_embedding(self, audio_wav_bytes: bytes) -> bytes:
        """Extract speaker embedding from WAV audio. Requires VOICE_CLONE capability."""
        raise NotImplementedError(
            f"Backend '{self.name}' does not support speaker embedding extraction"
        )

    def generate_streaming(self, text: str, **kwargs):
        """Generator yielding PCM chunks. Requires STREAMING capability."""
        raise NotImplementedError(
            f"Backend '{self.name}' does not support streaming"
        )

    def has_capability(self, cap: TTSCapability) -> bool:
        return cap in self.capabilities


def create_backend(backend_name: Optional[str] = None) -> TTSBackend:
    """Factory: create TTS backend by name.

    Args:
        backend_name: 'sherpa', 'qwen3_trt', or None for auto-detect.

    Auto-detect logic:
        - If LANGUAGE_MODE=multilanguage → qwen3_trt (52 languages)
        - Otherwise, use TTS_BACKEND env var (default: sherpa)
    """
    if backend_name is None:
        # Check LANGUAGE_MODE for automatic backend selection
        language_mode = os.environ.get("LANGUAGE_MODE", "zh_en")
        if language_mode == "multilanguage":
            backend_name = "qwen3_trt"
            logger.info("LANGUAGE_MODE=multilanguage → using qwen3_trt backend")
        else:
            backend_name = os.environ.get("TTS_BACKEND", "sherpa")

    if backend_name == "sherpa":
        from backends.sherpa import SherpaBackend
        return SherpaBackend()
    elif backend_name == "qwen3_trt":
        # Try importing from standalone package first, fallback to local
        try:
            from jetson_qwen3_speech import Qwen3TRTBackend
            logger.info("Using Qwen3TRTBackend from jetson-qwen3-speech package")
        except ImportError:
            from backends.qwen3_trt import Qwen3TRTBackend
            logger.info("Using Qwen3TRTBackend from local backends/")
        return Qwen3TRTBackend()
    else:
        raise ValueError(f"Unknown TTS backend: {backend_name}")
