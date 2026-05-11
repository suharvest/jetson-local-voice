"""RK ASR adapter — wraps rkvoice_stream.create_asr() output to fit the
seeed-local-voice ASRBackend interface.

The two ABCs (ours in app.core.asr_backend and theirs in
rkvoice_stream.engine.asr) are intentionally near-identical; this module
just bridges the capability enum and forwards every method.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from app.core.asr_backend import (
    ASRBackend,
    ASRCapability,
    ASRStream,
    TranscriptionResult,
)


class _RKASRStreamAdapter(ASRStream):
    def __init__(self, inner):
        self._inner = inner

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        self._inner.accept_waveform(sample_rate, samples)

    def finalize(self) -> str:
        return self._inner.finalize()

    def prepare_finalize(self) -> None:
        self._inner.prepare_finalize()

    def cancel_and_finalize(self) -> None:
        # rkvoice-stream's ASRStream defines cancel_and_finalize with a no-op
        # default; our ABC doesn't declare it but the WS layer calls it. Forward
        # defensively in case a backend overrides it.
        fn = getattr(self._inner, "cancel_and_finalize", None)
        if fn is not None:
            fn()

    def get_partial(self) -> tuple[str, bool]:
        return self._inner.get_partial()


# Map rkvoice_stream.engine.asr.ASRCapability values (strings) onto ours.
_CAP_MAP = {
    "offline": ASRCapability.OFFLINE,
    "streaming": ASRCapability.STREAMING,
    "multi_language": ASRCapability.MULTI_LANGUAGE,
}


class RKASRBackend(ASRBackend):
    """Adapter around rkvoice_stream.create_asr().

    Backend selection is delegated to rkvoice-stream itself via the
    ``ASR_BACKEND`` env var (set in the rk3576/rk3588 profile). We pass
    no kwargs to keep that the single source of truth.
    """

    def __init__(self):
        from rkvoice_stream import create_asr
        self._inner = create_asr()
        self._platform = os.environ.get("RK_PLATFORM", "rk3576")

    @property
    def name(self) -> str:
        return f"rk:{self._inner.name}"

    @property
    def capabilities(self) -> set[ASRCapability]:
        out: set[ASRCapability] = set()
        for cap in self._inner.capabilities:
            value = cap.value if hasattr(cap, "value") else str(cap)
            mapped = _CAP_MAP.get(value)
            if mapped is not None:
                out.add(mapped)
        return out

    @property
    def sample_rate(self) -> int:
        return self._inner.sample_rate

    def is_ready(self) -> bool:
        return self._inner.is_ready()

    def preload(self) -> None:
        self._inner.preload()

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        result = self._inner.transcribe(audio_bytes, language=language)
        meta = getattr(result, "meta", {}) or {}
        return TranscriptionResult(
            text=result.text,
            language=result.language,
            **meta,
        )

    def create_stream(self, language: str = "auto") -> ASRStream:
        return _RKASRStreamAdapter(self._inner.create_stream(language=language))
