"""RK TTS adapter — wraps rkvoice_stream.create_tts() output.

rkvoice-stream's TTSBackend ABC is smaller than ours (no `capabilities`,
no `language` arg, `speaker_id` is int with default 0); the adapter
forwards everything the OpenVoiceStream contract requires and exposes
a conservative default capability set.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from app.core.language import detect_zh_en
from app.core.tts_backend import TTSBackend, TTSCapability


# rkvoice-stream's TTSBackend doesn't expose a capability set. The shipped
# backends (matcha_rknn, piper_rknn, qwen3_rknn) all do basic + streaming
# TTS, so declare that as the floor. The wire layer feature-detects optional
# things (voice clone, etc.) via has_capability().
_DEFAULT_RK_TTS_CAPS = {
    TTSCapability.BASIC_TTS,
    TTSCapability.STREAMING,
    TTSCapability.MULTI_LANGUAGE,
}


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
        language = detect_zh_en(text, language)
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
        synthesize_stream().

        rkvoice-stream yields ``(audio, metadata)`` tuples where ``audio`` is
        either float32 [-1,1], int16 PCM, or raw bytes. The wire layer
        (`/tts/stream`) expects int16 PCM bytes per chunk, so coerce here
        — starlette's StreamingResponse calls ``.encode()`` on non-bytes
        items and explodes on tuples (`'tuple' object has no attribute
        'encode'`).
        """
        speaker_id = kwargs.pop("speaker_id", 0) or 0
        speed = kwargs.pop("speed", None)
        pitch_shift = kwargs.pop("pitch_shift", None)
        language = detect_zh_en(text, kwargs.pop("language", None))
        kwargs.setdefault("language", language)
        for item in self._inner.synthesize_stream(
            text=text,
            speaker_id=speaker_id,
            speed=speed,
            pitch_shift=pitch_shift,
            **kwargs,
        ):
            audio = item[0] if isinstance(item, tuple) else item
            if audio is None:
                continue
            if isinstance(audio, (bytes, bytearray)):
                if len(audio) == 0:
                    continue
                yield bytes(audio)
                continue
            if isinstance(audio, np.ndarray):
                if audio.size == 0:
                    continue
                if audio.dtype == np.int16:
                    yield audio.tobytes()
                else:
                    a = np.asarray(audio, dtype=np.float32)
                    a = np.clip(a, -1.0, 1.0)
                    yield (a * 32767.0).astype(np.int16).tobytes()
                continue
            # Unknown payload — skip rather than poison the stream.
            continue

    def synthesize_stream(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        language = detect_zh_en(text, language)
        kwargs.setdefault("language", language)
        yield from self._inner.synthesize_stream(
            text=text,
            speaker_id=speaker_id if speaker_id is not None else 0,
            speed=speed,
            pitch_shift=pitch_shift,
            **kwargs,
        )
