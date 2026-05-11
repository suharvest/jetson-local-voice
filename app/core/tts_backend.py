"""TTS backend abstraction with capability discovery.

Backends expose different capabilities (voice clone, streaming, etc.).
Clients check capabilities before calling optional endpoints.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple

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

    def unload(self) -> None:
        """Release GPU/NPU resources. See ASRBackend.unload() for semantics."""
        pass


_TTS_REGISTRY: Dict[str, Tuple[str, str]] = {
    "jetson.trt_edge_llm": ("app.backends.jetson.trt_edge_llm_tts", "TRTEdgeLLMTTSBackend"),
    "jetson.matcha_trt":   ("app.backends.jetson.matcha_trt",       "MatchaTRTBackend"),
    "jetson.qwen3_trt":    ("app.backends.jetson.qwen3_trt",        "Qwen3TRTBackend"),
    "cpu.sherpa":          ("app.backends.cpu.sherpa",              "SherpaBackend"),
}


def _lazy_import(module_path: str, attr: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def create_tts_backend() -> TTSBackend:
    """Factory: instantiate the TTS backend declared by the loaded profile.

    Reads ``tts_backend`` from app.core.profile_loader.current_profile().
    Raises ValueError when no profile is loaded, the key is missing, or
    the value is not in the registry.
    """
    from app.core.profile_loader import current_profile
    spec = current_profile().get("tts_backend")
    if not spec:
        raise ValueError("Profile must declare 'tts_backend'")
    if spec not in _TTS_REGISTRY:
        raise ValueError(f"Unknown tts_backend: {spec!r}")
    module_path, cls_name = _TTS_REGISTRY[spec]
    logger.info("Creating TTS backend %s (%s.%s)", spec, module_path, cls_name)
    return _lazy_import(module_path, cls_name)()
