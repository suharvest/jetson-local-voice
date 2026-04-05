"""ASR backend abstraction with capability discovery.

Mirrors the TTS backend pattern (tts_backend.py).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

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

    def has_capability(self, cap: ASRCapability) -> bool:
        return cap in self.capabilities


def create_asr_backend(backend_name: Optional[str] = None) -> ASRBackend:
    if backend_name is None:
        backend_name = os.environ.get("ASR_BACKEND", "sherpa")

    if backend_name == "sherpa":
        from backends.sherpa_asr import SherpaASRBackend
        return SherpaASRBackend()
    elif backend_name == "qwen3":
        from backends.qwen3_asr import Qwen3ASRBackend
        return Qwen3ASRBackend()
    else:
        raise ValueError(f"Unknown ASR backend: {backend_name}")
