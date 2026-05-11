"""ASR backend abstraction with capability discovery.

Mirrors the TTS backend pattern (tts_backend.py).
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Tuple

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

    def unload(self) -> None:
        """Release GPU/NPU resources. Override in backends that hold shared
        hardware so the BackendCoordinator's 'exclusive' mode can hand the
        device to another backend. Default is a no-op — backends without
        an unload() stay resident, which is fine for 'concurrent' and
        'serialized' modes."""
        pass


_ASR_REGISTRY: Dict[str, Tuple[str, str]] = {
    "jetson.trt_edge_llm":   ("app.backends.jetson.trt_edge_llm_asr", "TRTEdgeLLMASRBackend"),
    "jetson.paraformer_trt": ("app.backends.jetson.paraformer_trt",   "ParaformerTRTBackend"),
    "jetson.qwen3_asr":      ("app.backends.jetson.qwen3_asr",        "Qwen3ASRBackend"),
    "cpu.sherpa_asr":        ("app.backends.cpu.sherpa_asr",          "SherpaASRBackend"),
    "rk.asr":                ("app.backends.rk.asr",                  "RKASRBackend"),
}


def _lazy_import(module_path: str, attr: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def create_asr_backend() -> ASRBackend:
    """Factory: instantiate the ASR backend declared by the loaded profile.

    Reads ``asr_backend`` from app.core.profile_loader.current_profile().
    The value must be a registry key (e.g. ``jetson.trt_edge_llm``). Raises
    ValueError if no profile is loaded, the key is missing, or unknown.
    """
    from app.core.profile_loader import current_profile
    spec = current_profile().get("asr_backend")
    if not spec:
        raise ValueError("Profile must declare 'asr_backend'")
    if spec not in _ASR_REGISTRY:
        raise ValueError(f"Unknown asr_backend: {spec!r}")
    module_path, cls_name = _ASR_REGISTRY[spec]
    logger.info("Creating ASR backend %s (%s.%s)", spec, module_path, cls_name)
    return _lazy_import(module_path, cls_name)()
