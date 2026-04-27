"""Tests for Qwen3StreamingASRStream._offline_final_text.

Covers: happy path, empty buffer, repeated calls, and large audio
(>6s triggering the _transcribe_segmented path inside the backend).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asr_backend import TranscriptionResult  # noqa: E402
from backends.qwen3_asr import Qwen3StreamingASRStream  # noqa: E402


# ── Test double ────────────────────────────────────────────────────────

class _FakeBackend:
    """Stand-in that records every transcribe_audio call."""

    def __init__(self, return_text: str = "hello world"):
        self._return_text = return_text
        self.last_audio: Optional[np.ndarray] = None
        self.last_language: Optional[str] = None
        self.call_count = 0
        # Stub attributes the stream touches on init.
        self._encoder = None
        self._decoder = None
        self._decoder_ort = None
        self._embed_tokens = None
        self._tokenizer = None

    def transcribe_audio(self, audio: np.ndarray, language: str = "auto"):
        self.last_audio = np.asarray(audio).copy()
        self.last_language = language
        self.call_count += 1
        return TranscriptionResult(
            text=self._return_text,
            duration=len(audio) / 16000,
            inference_time=0.0,
            rtf=0.0,
            n_tokens=len(self._return_text),
            per_token_ms=0.0,
            backend="fake",
        )


def _make_stream(return_text: str = "hello world"):
    backend = _FakeBackend(return_text=return_text)
    stream = Qwen3StreamingASRStream(backend, language="auto")
    # Stub encoder-bound path so we can test buffer logic without a
    # real ONNX model.
    stream._process_streaming_chunk = lambda: setattr(
        stream, "_processed_samples", len(stream._audio_buf)
    )
    return stream, backend


def _speech_chunk(seed: int, n_samples: int = 3200) -> np.ndarray:
    """Deterministic non-silent float32 chunk (~200ms @ 16kHz)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float32) / 16000.0
    tone = 0.4 * np.sin(2 * np.pi * (200 + 50 * seed) * t).astype(np.float32)
    noise = rng.standard_normal(n_samples).astype(np.float32) * 0.05
    return tone + noise


# ── Tests ──────────────────────────────────────────────────────────────

class TestOfflineFinalText:
    """_offline_final_text unit tests."""

    def test_normal_flow(self):
        """Feed audio chunks, call _offline_final_text, get mock text."""
        stream, backend = _make_stream(return_text="the transcript")
        stream.accept_waveform(16000, _speech_chunk(1))
        stream.accept_waveform(16000, _speech_chunk(2))

        result = stream._offline_final_text()

        assert result == "the transcript"
        assert backend.call_count == 1
        assert backend.last_audio is not None
        # Both chunks must be concatenated and passed to the backend.
        assert len(backend.last_audio) >= 6400  # 2 × 3200

    def test_empty_buffer(self):
        """Call _offline_final_text with empty buffer → empty string."""
        stream, _ = _make_stream()
        assert stream._offline_final_text() == ""

    def test_multiple_calls(self):
        """Repeated calls both return the correct text (no state pollution)."""
        stream, backend = _make_stream(return_text="same text")
        stream.accept_waveform(16000, _speech_chunk(10))

        r1 = stream._offline_final_text()
        r2 = stream._offline_final_text()

        assert r1 == "same text"
        assert r2 == "same text"
        # Each call must invoke the backend independently.
        assert backend.call_count == 2

    def test_long_audio_passed_to_backend(self):
        """~7s of audio — _offline_final_text passes full buffer to backend.

        The backend.r transcribe_audio method internally dispatches to
        _transcribe_segmented when audio exceeds 6s.  This test verifies
        that _offline_final_text correctly concatenates and passes the
        full buffer regardless of length.
        """
        stream, backend = _make_stream(return_text="long audio result")

        n_chunks = int(7.0 * 16000 / 3200) + 1  # >7s of 200ms chunks
        for i in range(n_chunks):
            stream.accept_waveform(16000, _speech_chunk(i))

        result = stream._offline_final_text()

        assert result == "long audio result"
        assert backend.call_count == 1
        # Total audio must be at least ~7s.
        min_samples = int(7.0 * 16000 * 0.95)
        assert len(backend.last_audio) >= min_samples, (
            f"expected >={min_samples} samples for ~7s audio, "
            f"got {len(backend.last_audio)}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
