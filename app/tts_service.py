"""TTS service — backend-agnostic thin proxy.

Backend is selected via TTS_BACKEND env var:
  - "sherpa" (default): Matcha/Kokoro via sherpa-onnx
  - "qwen3_trt": Qwen3-TTS via C++ TRT native engine

All backends implement TTSBackend (see tts_backend.py).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from tts_backend import TTSBackend, TTSCapability, create_backend

logger = logging.getLogger(__name__)

_backend: Optional[TTSBackend] = None


def get_backend() -> TTSBackend:
    global _backend
    if _backend is None:
        _backend = create_backend()
    return _backend


def preload() -> None:
    """Pre-load TTS model and warmup."""
    backend = get_backend()
    logger.info("TTS backend: %s (capabilities: %s)",
                backend.name, [c.value for c in backend.capabilities])
    backend.preload()


def synthesize(
    text: str,
    speaker_id: Optional[int] = None,
    speed: Optional[float] = None,
    pitch_shift: Optional[float] = None,
    language: Optional[str] = None,
    **kwargs,
) -> tuple[bytes, dict]:
    """Synthesize text to WAV bytes. Returns (wav_bytes, metadata)."""
    return get_backend().synthesize(
        text=text, speaker_id=speaker_id, speed=speed,
        pitch_shift=pitch_shift, language=language, **kwargs
    )


def clone_voice(
    text: str,
    speaker_embedding: bytes,
    language: Optional[str] = None,
    **kwargs,
) -> tuple[bytes, dict]:
    """Synthesize with voice cloning. Raises NotImplementedError if unsupported."""
    return get_backend().clone_voice(
        text=text, speaker_embedding=speaker_embedding,
        language=language, **kwargs
    )


def extract_speaker_embedding(audio_wav_bytes: bytes) -> bytes:
    """Extract speaker embedding from WAV. Raises NotImplementedError if unsupported."""
    return get_backend().extract_speaker_embedding(audio_wav_bytes)


def get_sample_rate() -> int:
    return get_backend().sample_rate


def capabilities() -> set[TTSCapability]:
    return get_backend().capabilities


def has_capability(cap: TTSCapability) -> bool:
    return get_backend().has_capability(cap)


def backend_name() -> str:
    return get_backend().name


def is_ready() -> bool:
    return _backend is not None and _backend.is_ready()
