"""RK TTS adapter — wraps rkvoice_stream.create_tts() output.

rkvoice-stream's TTSBackend ABC is smaller than ours (no `capabilities`,
no `language` arg, `speaker_id` is int with default 0); the adapter
forwards everything the seeed-local-voice contract requires and exposes
a conservative default capability set.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from app.core.tts_backend import TTSBackend, TTSCapability


# rkvoice-stream's TTSBackend doesn't expose a capability set. The shipped
# backends (matcha_rknn, piper_rknn, qwen3_rknn) all do basic + streaming
# TTS, so declare that as the floor. The wire layer feature-detects optional
# things (voice clone, etc.) via has_capability().
_DEFAULT_RK_TTS_CAPS = {TTSCapability.BASIC_TTS, TTSCapability.STREAMING}


class RKTTSBackend(TTSBackend):
    """Adapter around rkvoice_stream.create_tts(). Backend selection is
    delegated to rkvoice-stream via the ``TTS_BACKEND`` env var (set in
    the rk3576/rk3588 profile)."""

    def __init__(self):
        from rkvoice_stream import create_tts
        self._inner = create_tts()

    @property
    def name(self) -> str:
        return f"rk:{self._inner.name}"

    @property
    def capabilities(self) -> set[TTSCapability]:
        return set(_DEFAULT_RK_TTS_CAPS)

    @property
    def sample_rate(self) -> int:
        return self._inner.get_sample_rate()

    def is_ready(self) -> bool:
        return self._inner.is_ready()

    def preload(self) -> None:
        self._inner.preload()

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        # rkvoice-stream's synthesize() doesn't take `language`; pass it
        # through kwargs only when explicitly set so backends that ignore it
        # are unaffected.
        if language is not None:
            kwargs.setdefault("language", language)
        return self._inner.synthesize(
            text=text,
            speaker_id=speaker_id if speaker_id is not None else 0,
            speed=speed,
            pitch_shift=pitch_shift,
            **kwargs,
        )

    def generate_streaming(self, text: str, **kwargs):
        """Bridge our base-class generate_streaming() to rkvoice-stream's
        synthesize_stream()."""
        speaker_id = kwargs.pop("speaker_id", 0) or 0
        speed = kwargs.pop("speed", None)
        pitch_shift = kwargs.pop("pitch_shift", None)
        language = kwargs.pop("language", None)
        if language is not None:
            kwargs.setdefault("language", language)
        yield from self._inner.synthesize_stream(
            text=text,
            speaker_id=speaker_id,
            speed=speed,
            pitch_shift=pitch_shift,
            **kwargs,
        )

    def synthesize_stream(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        if language is not None:
            kwargs.setdefault("language", language)
        yield from self._inner.synthesize_stream(
            text=text,
            speaker_id=speaker_id if speaker_id is not None else 0,
            speed=speed,
            pitch_shift=pitch_shift,
            **kwargs,
        )
