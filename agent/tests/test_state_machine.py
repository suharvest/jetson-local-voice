"""Tests for ConvState transitions in BaseApp."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from openvoicestream_agent.app_base import BaseApp
from openvoicestream_agent.state import ConvState
from openvoicestream_agent.event_bus import EventBus


def _fresh_app() -> BaseApp:
    """Build a BaseApp without touching audio/llm/slv (via __new__)."""
    app = BaseApp.__new__(BaseApp)
    app.events = EventBus()
    app.plugins = []
    app._state = ConvState.IDLE
    app._slv_reconnect_count = 0
    return app


@pytest.mark.asyncio
async def test_set_state_only_fires_on_change():
    app = _fresh_app()
    seen: list[dict] = []

    class P:
        name = "p"
        async def on_state_change(self, data):  # noqa: ANN001
            seen.append(data)

    app.plugins.append(P())

    app._set_state(ConvState.IDLE)  # no-op
    await asyncio.sleep(0.01)
    assert seen == []

    app._set_state(ConvState.LISTENING)
    await asyncio.sleep(0.01)
    assert len(seen) == 1
    assert seen[0]["state"] == "listening"
    assert seen[0]["prev"] == "idle"


@pytest.mark.asyncio
async def test_golden_path_transitions():
    app = _fresh_app()
    transitions: list[tuple[str, str]] = []

    def cb(d):
        transitions.append((d["prev"], d["state"]))

    app.events.subscribe("state_change", cb)

    for s in (
        ConvState.LISTENING,
        ConvState.THINKING,
        ConvState.SPEAKING,
        ConvState.IDLE,
    ):
        app._set_state(s)

    assert transitions == [
        ("idle", "listening"),
        ("listening", "thinking"),
        ("thinking", "speaking"),
        ("speaking", "idle"),
    ]


@pytest.mark.asyncio
async def test_barge_in_transition():
    app = _fresh_app()
    app._set_state(ConvState.SPEAKING)
    app._set_state(ConvState.BARGED_IN)
    app._set_state(ConvState.THINKING)
    assert app._state == ConvState.THINKING


@pytest.mark.asyncio
async def test_barge_in_cancels_llm_turn():
    """ASRPartial during SPEAKING must cancel the in-flight LLM task,
    abort SLV TTS, stop playback, and re-arm _first_tts_seen.
    """
    from openvoicestream_agent.slv_client import ASRPartial

    app = _fresh_app()
    # Minimal stubs for the barge-in branch.
    abort_calls: list[int] = []
    stop_calls: list[int] = []

    class _Audio:
        is_playing = True
        async def stop_playback(self):
            stop_calls.append(1)

    class _SLV:
        async def abort(self):
            abort_calls.append(1)

    app.audio = _Audio()
    app.slv = _SLV()
    app._first_tts_seen = True
    app._state = ConvState.SPEAKING

    # Long-running fake LLM turn that should be cancelled.
    cancelled = asyncio.Event()
    async def _llm_turn():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    app._llm_turn_task = asyncio.create_task(_llm_turn(), name="llm-turn")
    # Give it one loop tick to actually start.
    await asyncio.sleep(0)

    await app._dispatch_one(ASRPartial(text="打断一下"))

    assert cancelled.is_set(), "LLM turn was not cancelled by barge-in"
    assert abort_calls == [1]
    assert stop_calls == [1]
    assert app._first_tts_seen is False
    assert app._state == ConvState.BARGED_IN


@pytest.mark.asyncio
async def test_slv_error_in_sleeping_stays_sleeping():
    """Regression: a transport error while SLEEPING must NOT
    force the FSM back to IDLE — that would hot-mic the agent in
    wake_word / push_to_talk mode."""
    from openvoicestream_agent.slv_client import SLVError

    app = _fresh_app()
    app._llm_turn_task = None
    app._state = ConvState.SLEEPING

    await app._dispatch_one(SLVError(message="ws ping timeout"))
    assert app._state == ConvState.SLEEPING


@pytest.mark.asyncio
async def test_slv_error_in_thinking_drops_to_idle():
    """Non-sleeping SLVError still resets FSM to IDLE (status quo)."""
    from openvoicestream_agent.slv_client import SLVError

    app = _fresh_app()
    app._llm_turn_task = None
    app._state = ConvState.THINKING

    await app._dispatch_one(SLVError(message="boom"))
    assert app._state == ConvState.IDLE


@pytest.mark.asyncio
async def test_sleeping_drops_asr_partial_and_endpoint():
    """Regression: late-arriving ASR events after /sleep must
    not be processed — would otherwise leak through to on_user_utterance
    and silently wake the agent."""
    from openvoicestream_agent.slv_client import ASRPartial, ASREndpoint, ASRFinal

    app = _fresh_app()
    app._state = ConvState.SLEEPING
    app._llm_turn_task = None
    app._first_tts_seen = False
    app._slv_reconnect_count = 0

    broadcasts: list[tuple] = []
    async def _br(name, *args):
        broadcasts.append((name, args))
    app.broadcast = _br
    app._broadcast = _br

    class _Audio:
        is_playing = False
        async def stop_playback(self):
            pass

    class _SLV:
        reconnected = 0
        async def reconnect(self):
            type(self).reconnected += 1
        async def abort(self):
            pass

    app.audio = _Audio()
    app.slv = _SLV()

    # ASRPartial dropped silently.
    await app._dispatch_one(ASRPartial(text="hello"))
    assert all(n != "on_user_partial" for n, _ in broadcasts)
    # ASREndpoint dropped silently.
    await app._dispatch_one(ASREndpoint())
    assert all(n != "on_user_speech_start" for n, _ in broadcasts)
    # ASRFinal with session_complete=True triggers reconnect but does
    # NOT broadcast on_user_utterance and does NOT spawn LLM turn.
    await app._dispatch_one(ASRFinal(
        text="real text", duplicate_of_streamed=False, session_complete=True
    ))
    assert _SLV.reconnected == 1
    assert all(n != "on_user_utterance" for n, _ in broadcasts), (
        f"on_user_utterance leaked through SLEEPING gate: {broadcasts}"
    )
    assert app._llm_turn_task is None
    assert app._state == ConvState.SLEEPING


def test_conv_state_values():
    assert ConvState.IDLE.value == "idle"
    assert ConvState.SPEAKING.value == "speaking"
    assert ConvState.BARGED_IN.value == "barged_in"
    assert list(ConvState) == [
        ConvState.IDLE,
        ConvState.LISTENING,
        ConvState.THINKING,
        ConvState.SPEAKING,
        ConvState.BARGED_IN,
        ConvState.SLEEPING,
    ]
    assert ConvState.SLEEPING.value == "sleeping"
