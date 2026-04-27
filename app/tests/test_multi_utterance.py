"""Regression test for multi-utterance buffer-drop bug.

Bug (pre-M1): in Qwen3StreamingASRStream.accept_waveform, when the previous
utterance had been finalised (_episode_final=True), a new chunk's VAD
detection used to clear `_utterance_audio_buffer` *after* the chunk had
already been appended to it — dropping the first chunk of the next
utterance and causing the leading word to be truncated.

Fix (M1): _check_new_utterance_resume() runs *before* the append, so the
buffer starts empty and the new chunk becomes its first sample.

This test mocks transcribe_audio to capture the audio that the stream
passes in, then asserts the second utterance's transcribe call sees the
first chunk's samples.
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asr_backend import TranscriptionResult  # noqa: E402
from backends.qwen3_asr import Qwen3ASRBackend, Qwen3StreamingASRStream  # noqa: E402


# ── Test doubles ────────────────────────────────────────────────────────

class _FakeBackend:
    """Stand-in for Qwen3ASRBackend that records every transcribe_audio call."""

    def __init__(self, texts: List[str]):
        self._texts = list(texts)
        self.calls: List[np.ndarray] = []  # captured audio per call
        # Stub attributes Qwen3StreamingASRStream may touch on init/finalize.
        self._encoder = None
        self._decoder = None
        self._decoder_ort = None
        self._embed_tokens = None
        self._tokenizer = None

    def transcribe_audio(self, audio: np.ndarray, language: str = "auto"):
        self.calls.append(np.asarray(audio).copy())
        text = self._texts.pop(0) if self._texts else ""
        return TranscriptionResult(
            text=text,
            duration=len(audio) / 16000,
            inference_time=0.0,
            rtf=0.0,
            n_tokens=len(text),
            per_token_ms=0.0,
            backend="fake",
        )


def _make_stream(texts):
    backend = _FakeBackend(texts)
    stream = Qwen3StreamingASRStream(backend, language="auto")
    # Skip the encoder-bound partial-decode path; the offline finaliser
    # (used by force_endpoint) reads _utterance_audio_buffer directly,
    # which is what this test cares about. Without this stub we would
    # need a full ONNX encoder + mel extractor in CI.
    stream._process_streaming_chunk = lambda: setattr(
        stream, "_processed_samples", len(stream._audio_buf)
    )
    return stream, backend


# Distinct waveforms so we can detect them inside the captured audio buffer.
def _speech_chunk(seed: int, n_samples: int = 3200) -> np.ndarray:
    """Return a deterministic non-silent float32 chunk (~200ms @ 16kHz).

    Sized below Qwen3StreamingASRStream._chunk_size_samples (6400) so that
    accept_waveform does NOT invoke the encoder-bound streaming path; the
    test exercises only the buffer-management / helper logic.
    """
    rng = np.random.default_rng(seed)
    # Strong tone + noise → RMS well above the energy fallback threshold and
    # webrtcvad will reliably classify as speech.
    t = np.arange(n_samples, dtype=np.float32) / 16000.0
    tone = 0.4 * np.sin(2 * np.pi * (200 + 50 * seed) * t).astype(np.float32)
    noise = rng.standard_normal(n_samples).astype(np.float32) * 0.05
    return tone + noise


def _silence_chunk(n_samples: int = 3200) -> np.ndarray:
    return np.zeros(n_samples, dtype=np.float32)


# ── Tests ───────────────────────────────────────────────────────────────

class TestMultiUtteranceBufferRetention:
    def test_second_utterance_first_chunk_is_kept(self):
        """The first chunk of utterance 2 must reach transcribe_audio."""
        stream, backend = _make_stream(["话头一", "话头二"])

        # ── Utterance 1: feed one speech chunk, then force-finalise. ──
        u1 = _speech_chunk(seed=1)
        stream.accept_waveform(16000, u1)
        # Force the offline finaliser path (mirrors VAD endpoint) without
        # needing precise silence timing.
        text1 = stream.force_endpoint()
        assert text1 == "话头一"
        assert stream._episode_final is True

        # transcribe_audio call #1 must have seen u1's samples.
        assert len(backend.calls) == 1
        seen1 = backend.calls[0]
        assert seen1.size >= u1.size
        # Sanity: u1 sits at the start of the captured audio.
        assert np.allclose(seen1[: u1.size], u1, atol=1e-5)

        # ── Utterance 2: distinct speech chunk arrives next. ──
        u2_first = _speech_chunk(seed=2)
        stream.accept_waveform(16000, u2_first)

        # The new-utterance helper must have reset state and accepted the
        # chunk, NOT discarded it.
        assert stream._episode_final is False
        assert len(stream._utterance_audio_buffer) >= 1
        assert np.allclose(stream._utterance_audio_buffer[0], u2_first,
                           atol=1e-5)

        text2 = stream.force_endpoint()
        assert text2 == "话头二"

        # transcribe_audio call #2 must have seen u2_first's samples —
        # this is the regression assertion. Pre-fix, _utterance_audio_buffer
        # was cleared *after* the chunk was appended, so the second call
        # would see an empty / silence buffer.
        assert len(backend.calls) == 2
        seen2 = backend.calls[1]
        assert seen2.size >= u2_first.size, (
            f"second transcribe got {seen2.size} samples, expected >= "
            f"{u2_first.size} (first chunk was dropped — bug regressed)"
        )
        assert np.allclose(seen2[: u2_first.size], u2_first, atol=1e-5), (
            "second transcribe did not start with utterance-2 first chunk"
        )

    def test_helper_returns_false_when_not_finalised(self):
        """No state reset until the previous episode is marked final."""
        stream, _ = _make_stream([])
        stream._episode_final = False
        # Even with speech, helper is a no-op while we are still mid-utterance.
        assert stream._check_new_utterance_resume(_speech_chunk(3)) is False

    def test_helper_returns_false_on_silence_chunk(self):
        """Silence after finalisation must NOT reset (no new utterance yet)."""
        stream, _ = _make_stream([])
        stream._episode_final = True
        stream._archive_text = "previous"
        stream._utterance_audio_buffer = [np.zeros(10, dtype=np.float32)]
        result = stream._check_new_utterance_resume(_silence_chunk())
        assert result is False
        # State preserved.
        assert stream._episode_final is True
        assert stream._archive_text == "previous"

    def test_helper_resets_full_state_on_new_utterance(self):
        stream, _ = _make_stream([])
        stream._episode_final = True
        stream._archive_text = "old text"
        stream._partial_text = "partial"
        stream._partial_token_ids = [1, 2, 3]
        stream._committed_token_ids = [4, 5]
        stream._encoder_frames = [np.zeros((1, 5, 1024), dtype=np.float32)]
        stream._total_encoder_frames = 5
        stream._vad_speech_samples = 9999
        stream._vad_silence_samples = 9999
        stream._utterance_audio_buffer = [np.zeros(10, dtype=np.float32)]

        did_reset = stream._check_new_utterance_resume(_speech_chunk(7))
        assert did_reset is True
        assert stream._episode_final is False
        assert stream._archive_text == ""
        assert stream._partial_text == ""
        assert stream._partial_token_ids == []
        assert stream._committed_token_ids == []
        assert stream._encoder_frames == []
        assert stream._total_encoder_frames == 0
        assert stream._vad_speech_samples == 0
        assert stream._vad_silence_samples == 0
        assert stream._utterance_audio_buffer == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
