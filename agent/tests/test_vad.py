"""Tests for client-side VAD + BaseApp asr_eos integration."""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from openvoicestream_agent.vad import EnergyVAD, create_vad


# ── EnergyVAD unit tests ────────────────────────────────────────────────


def test_energy_vad_silence() -> None:
    vad = EnergyVAD(threshold=0.01)
    silence = np.zeros(1600, dtype=np.int16).tobytes()
    assert vad.is_speech(silence) is False


def test_energy_vad_loud() -> None:
    vad = EnergyVAD(threshold=0.01)
    loud = np.random.randint(
        -10000, 10000, size=1600, dtype=np.int16
    ).tobytes()
    assert vad.is_speech(loud) is True


def test_energy_vad_empty() -> None:
    vad = EnergyVAD(threshold=0.01)
    assert vad.is_speech(b"") is False


def test_create_vad_energy_explicit() -> None:
    v = create_vad("energy")
    assert v.name == "energy"


def test_create_vad_auto_falls_back() -> None:
    # On a clean test env silero-vad likely isn't installed; auto must
    # not raise and must yield *some* VAD.
    v = create_vad("auto")
    assert v.name in ("silero", "energy")


# ── BaseApp asr_eos integration ─────────────────────────────────────────


class _FakeSLV:
    """Minimal SLV stand-in capturing send_audio + asr_eos calls."""

    def __init__(self) -> None:
        self.audio_chunks: list[bytes] = []
        self.asr_eos_count = 0

    async def send_audio(self, pcm: bytes) -> None:
        self.audio_chunks.append(pcm)

    async def asr_eos(self) -> None:
        self.asr_eos_count += 1


class _FakeAudio:
    """Yields pre-canned mic chunks then halts."""

    def __init__(self, chunks: list[bytes], chunk_ms: int = 100) -> None:
        self._chunks = chunks
        self.chunk_ms = chunk_ms

    async def start_capture(self):
        for c in self._chunks:
            yield c
            await asyncio.sleep(0)


class _AlwaysSpeechThenSilenceVAD:
    """Deterministic VAD that reports N speech chunks then silence."""

    name = "test"
    threshold = 0.5

    def __init__(self, speech_n: int) -> None:
        self._left = speech_n

    def is_speech(self, pcm: bytes) -> bool:
        if self._left > 0:
            self._left -= 1
            return True
        return False

    def reset(self) -> None:
        pass


@pytest.mark.asyncio
async def test_mic_pump_fires_asr_eos_on_speech_end() -> None:
    """speech (>=200ms) followed by 600ms silence → exactly one asr_eos."""
    # Construct app without invoking __init__ (avoids LLM/audio setup).
    from openvoicestream_agent.app_base import BaseApp

    app = BaseApp.__new__(BaseApp)

    class _Cfg:
        client_vad_backend = "off"  # we'll inject our own
        client_vad_threshold = None
        client_vad_speech_min_ms = 200
        client_vad_silence_ms = 600
        client_vad_drive_eos = True  # explicit for this test
        audio_input_sample_rate = 16000

    app.config = _Cfg()
    app.slv = _FakeSLV()
    # 100ms chunks: 3 speech (=300ms speech, crosses 200ms min), then 7
    # silence (=700ms, crosses 600ms silence threshold).
    chunk = b"\x00\x00" * 1600  # bytes content irrelevant to fake VAD
    chunks = [chunk] * 10
    app.audio = _FakeAudio(chunks, chunk_ms=100)
    app._client_vad = _AlwaysSpeechThenSilenceVAD(speech_n=3)
    app._vad_state = "idle"
    app._vad_speech_ms = 0
    app._vad_silence_ms = 0
    app._vad_eos_sent = False

    await app._mic_pump()

    assert app.slv.asr_eos_count == 1, (
        f"expected exactly 1 asr_eos, got {app.slv.asr_eos_count}"
    )
    # mic_pump only sends audio when VAD state is "speech" (with the
    # idle pre-roll flushed at speech-start). It also stops sending the
    # moment VAD transitions back to idle on speech-end. With 3 speech
    # chunks + 7 silence chunks and speech_min=200ms, silence=600ms:
    #   chunk 1: idle, buffered
    #   chunk 2: state→speech, flush preroll(1) + send(1) = 2 sends
    #   chunks 3..8: state=speech, send each = 6 sends
    #   chunk 9: silence_ms hits 600 → asr_eos, state→idle, NOT sent
    #   chunk 10: idle, buffered
    # → 8 sends total
    assert len(app.slv.audio_chunks) == 8
