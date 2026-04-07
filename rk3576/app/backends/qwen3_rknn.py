"""Qwen3-TTS RKNN/RKLLM backend for RK3576.

Wraps the existing tts_service.py TTSService into the TTSBackend interface.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from tts_backend import TTSBackend

logger = logging.getLogger(__name__)


class Qwen3RKNNBackend(TTSBackend):
    """Qwen3-TTS pipeline using RKNN NPU + RKLLM talker on RK3576."""

    def __init__(self):
        self._service = None

    @property
    def name(self) -> str:
        return "qwen3_rknn"

    def is_ready(self) -> bool:
        return self._service is not None and self._service.is_ready()

    def preload(self) -> None:
        from tts_service import TTSService

        model_dir = os.environ.get("MODEL_DIR", "/opt/tts/models")
        logger.info("Loading Qwen3-TTS RKNN models from %s", model_dir)
        self._service = TTSService(model_dir)
        self._service.load()
        logger.info("Qwen3-TTS RKNN backend ready")

    def synthesize(
        self,
        text: str,
        speaker_id: int = 0,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if self._service is None:
            raise RuntimeError("Backend not loaded — call preload() first")

        # Serialize NPU access with ASR backend if it is loaded
        try:
            from backends.qwen3_asr_rk import get_npu_lock
            lock = get_npu_lock()
        except ImportError:
            lock = None

        if lock is not None:
            with lock:
                return self._service.synthesize(
                    text=text,
                    speaker_id=speaker_id,
                    speed=speed or 1.0,
                )
        else:
            return self._service.synthesize(
                text=text,
                speaker_id=speaker_id,
                speed=speed or 1.0,
            )

    def get_sample_rate(self) -> int:
        from tts_service import SAMPLE_RATE
        return SAMPLE_RATE
