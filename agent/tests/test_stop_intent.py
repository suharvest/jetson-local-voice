"""Tests for stop-intent detection and dispatch behaviour."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openvoicestream_agent.app_base import BaseApp
from openvoicestream_agent.config import Config
from openvoicestream_agent.event_bus import EventBus
from openvoicestream_agent.state import ConvState
from openvoicestream_agent.slv_client import ASRFinal


def _stub_app() -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.config = Config()
    app.events = EventBus()
    app.plugins = []
    app._state = ConvState.IDLE
    app._slv_reconnect_count = 0
    app._llm_turn_task = None
    app._first_tts_seen = False
    app._vad_state = "idle"
    app._vad_speech_ms = 0
    app._vad_silence_ms = 0
    app._vad_eos_sent = False
    app._client_vad = None
    return app


def test_is_stop_intent_matches_chinese():
    app = _stub_app()
    assert app._is_stop_intent("停") is True
    assert app._is_stop_intent("停下") is True
    assert app._is_stop_intent("停下来") is True
    assert app._is_stop_intent("停下。") is True  # trailing punctuation
    assert app._is_stop_intent("别说了") is True


def test_is_stop_intent_matches_english_case_insensitive():
    app = _stub_app()
    assert app._is_stop_intent("Stop") is True
    assert app._is_stop_intent("stop") is True
    assert app._is_stop_intent("STOP.") is True
    assert app._is_stop_intent("shut up please") is True  # word-boundary prefix
    assert app._is_stop_intent("be quiet") is True


def test_is_stop_intent_rejects_negatives():
    app = _stub_app()
    # Chinese must be full-string equality
    assert app._is_stop_intent("停一下吧") is False
    assert app._is_stop_intent("我不想停") is False
    # English must respect word boundaries
    assert app._is_stop_intent("stopwatch") is False
    assert app._is_stop_intent("stopping") is False
    assert app._is_stop_intent("") is False
    assert app._is_stop_intent("   ") is False


@pytest.mark.asyncio
async def test_stop_intent_aborts_and_skips_llm():
    """ASRFinal containing a stop-word must NOT call on_user_utterance,
    MUST abort SLV + stop playback, MUST transition to IDLE, and MUST NOT
    append to session.history."""
    app = _stub_app()
    # Mocks for SLV / audio / session.
    app.slv = SimpleNamespace(
        abort=AsyncMock(),
        reconnect=AsyncMock(),
    )
    app.audio = SimpleNamespace(
        is_playing=True,
        stop_playback=AsyncMock(),
        play=AsyncMock(),
    )
    app.session = SimpleNamespace(history=[])

    # State pretends we were speaking.
    app._set_state(ConvState.SPEAKING)

    # Sentinel: any LLM-route call MUST NOT happen.
    on_utt_called = False

    async def on_user_utterance(text):  # type: ignore[no-redef]
        nonlocal on_utt_called
        on_utt_called = True

    app.on_user_utterance = on_user_utterance  # type: ignore[assignment]

    evt = ASRFinal(text="停下", session_complete=True)
    await app._dispatch_one(evt)

    # LLM router not invoked.
    assert on_utt_called is False
    # No history append.
    assert app.session.history == []
    # Abort + stop_playback were called.
    app.slv.abort.assert_awaited()
    app.audio.stop_playback.assert_awaited()
    # State landed in IDLE.
    assert app._state == ConvState.IDLE
