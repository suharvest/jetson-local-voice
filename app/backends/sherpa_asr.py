"""Sherpa-onnx ASR backend (Paraformer streaming + SenseVoice offline).

Wraps existing streaming_asr_service.py and asr_service.py.
Supports: OFFLINE, STREAMING
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from asr_backend import ASRBackend, ASRCapability, TranscriptionResult

logger = logging.getLogger(__name__)


class SherpaASRBackend(ASRBackend):

    def __init__(self):
        self._ready = False

    @property
    def name(self) -> str:
        return "sherpa_asr"

    @property
    def capabilities(self) -> set[ASRCapability]:
        caps = {ASRCapability.OFFLINE}
        try:
            import streaming_asr_service
            if streaming_asr_service.is_ready():
                caps.add(ASRCapability.STREAMING)
        except ImportError:
            pass
        return caps

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        try:
            import streaming_asr_service
            streaming_asr_service.preload()
            logger.info("Sherpa streaming ASR loaded")
        except Exception as e:
            logger.info("Streaming ASR not available: %s", e)
        self._ready = True

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        import asr_service
        text = asr_service.transcribe_audio(audio_bytes, language=language)
        return TranscriptionResult(text=text, language=language)
