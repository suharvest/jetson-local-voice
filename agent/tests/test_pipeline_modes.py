"""Unit tests for pipeline_mode (wake_word / push_to_talk) plumbing.

These tests build BaseApp via __new__ so they exercise wake/sleep
machinery without spinning up SLV / LLM / audio.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openvoicestream_agent.app_base import BaseApp
from openvoicestream_agent.config import Config
from openvoicestream_agent.event_bus import EventBus
from openvoicestream_agent.state import ConvState
from openvoicestream_agent.wake_sources import HTTPWakeSource


# ── Config-level checks ────────────────────────────────────────────


def test_config_default_always_on():
    c = Config()
    assert c.pipeline_mode == "always_on"


def test_config_validates_pipeline_mode():
    with pytest.raises(ValueError):
        Config(pipeline_mode="bogus")


def test_config_accepts_wake_word_and_ptt():
    Config(pipeline_mode="wake_word")
    Config(pipeline_mode="push_to_talk")


# ── helpers ────────────────────────────────────────────────────────


def _fresh_app(pipeline_mode: str = "always_on", sleep_timeout_s: float = 30.0) -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.events = EventBus()
    app.plugins = []
    app.config = SimpleNamespace(
        pipeline_mode=pipeline_mode,
        sleep_timeout_s=sleep_timeout_s,
    )
    if pipeline_mode == "always_on":
        app._state = ConvState.IDLE
    else:
        app._state = ConvState.SLEEPING
    app._sleep_task = None
    app._slv_reconnect_count = 0
    app._llm_turn_task = None
    app._first_tts_seen = False
    # Stub slv + audio with async no-ops.
    slv = MagicMock()
    slv.abort = AsyncMock()
    slv.asr_eos = AsyncMock()
    app.slv = slv
    audio = MagicMock()
    audio.stop_playback = AsyncMock()
    audio.is_playing = False
    app.audio = audio
    return app


# ── tests ──────────────────────────────────────────────────────────


def test_always_on_default_state_is_idle():
    app = _fresh_app("always_on")
    assert app._state == ConvState.IDLE


def test_wake_word_default_state_is_sleeping():
    app = _fresh_app("wake_word")
    assert app._state == ConvState.SLEEPING


def test_push_to_talk_default_state_is_sleeping():
    app = _fresh_app("push_to_talk")
    assert app._state == ConvState.SLEEPING


@pytest.mark.asyncio
async def test_wake_transitions_sleeping_to_idle():
    app = _fresh_app("wake_word", sleep_timeout_s=999)
    await app.wake(source="test")
    assert app._state == ConvState.IDLE
    # Sleep task armed.
    assert app._sleep_task is not None
    app._sleep_task.cancel()
    try:
        await app._sleep_task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_wake_noop_when_not_sleeping():
    app = _fresh_app("wake_word", sleep_timeout_s=999)
    app._state = ConvState.LISTENING
    await app.wake(source="test")
    # Still LISTENING, not bumped to IDLE.
    assert app._state == ConvState.LISTENING


@pytest.mark.asyncio
async def test_sleep_cancels_llm_and_aborts_slv():
    app = _fresh_app("wake_word", sleep_timeout_s=999)
    app._state = ConvState.SPEAKING

    async def _long():
        await asyncio.sleep(10)

    app._llm_turn_task = asyncio.create_task(_long())
    await asyncio.sleep(0)  # let it start

    await app.sleep()
    assert app._state == ConvState.SLEEPING
    assert app._llm_turn_task.cancelled() or app._llm_turn_task.done()
    app.slv.abort.assert_awaited()
    app.audio.stop_playback.assert_awaited()


@pytest.mark.asyncio
async def test_sleep_idempotent():
    app = _fresh_app("wake_word")
    assert app._state == ConvState.SLEEPING
    await app.sleep()
    assert app._state == ConvState.SLEEPING


@pytest.mark.asyncio
async def test_reset_sleep_timer_noop_for_always_on():
    app = _fresh_app("always_on")
    app._reset_sleep_timer()
    assert app._sleep_task is None


@pytest.mark.asyncio
async def test_reset_sleep_timer_cancels_old_on_recall():
    app = _fresh_app("wake_word", sleep_timeout_s=10)
    app._state = ConvState.IDLE
    app._reset_sleep_timer()
    first = app._sleep_task
    assert first is not None
    app._reset_sleep_timer()
    second = app._sleep_task
    assert second is not first
    # Let the cancel propagate.
    try:
        await first
    except (asyncio.CancelledError, Exception):
        pass
    assert first.cancelled() or first.done()
    second.cancel()
    try:
        await second
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_sleep_after_only_fires_when_idle():
    app = _fresh_app("wake_word", sleep_timeout_s=0.05)
    app._state = ConvState.THINKING  # mid-turn — should NOT sleep
    app._reset_sleep_timer()
    await asyncio.sleep(0.15)
    assert app._state == ConvState.THINKING


@pytest.mark.asyncio
async def test_sleep_after_fires_when_idle():
    app = _fresh_app("wake_word", sleep_timeout_s=0.05)
    app._state = ConvState.IDLE
    app._reset_sleep_timer()
    await asyncio.sleep(0.2)
    assert app._state == ConvState.SLEEPING


def test_http_wake_source_basic_lifecycle():
    app = _fresh_app("wake_word")
    ws = HTTPWakeSource(app)
    assert ws.name == "http"
    assert ws.setup() is True
    # start/stop are simple awaitables.
    asyncio.get_event_loop().run_until_complete(ws.start())
    asyncio.get_event_loop().run_until_complete(ws.stop())


@pytest.mark.asyncio
async def test_on_wake_hook_fires():
    app = _fresh_app("wake_word", sleep_timeout_s=999)
    seen: list[dict] = []

    class P:
        name = "p"
        async def on_wake(self, data):  # noqa: ANN001
            seen.append(data)

    app.plugins.append(P())
    await app.wake(source="unittest")
    assert seen == [{"source": "unittest"}]
    if app._sleep_task is not None:
        app._sleep_task.cancel()


@pytest.mark.asyncio
async def test_on_sleep_hook_fires():
    app = _fresh_app("wake_word")
    app._state = ConvState.IDLE
    seen: list = []

    class P:
        name = "p"
        async def on_sleep(self, data):  # noqa: ANN001
            seen.append(data)

    app.plugins.append(P())
    await app.sleep()
    assert seen == [None]


# ── PTT double-EOS dedupe ------------------------------------------


@pytest.mark.asyncio
async def test_send_asr_eos_once_is_idempotent_within_turn():
    app = _fresh_app("push_to_talk")
    app._eos_sent_this_turn = False

    sent1 = await app.send_asr_eos_once()
    sent2 = await app.send_asr_eos_once()
    assert sent1 is True
    assert sent2 is False
    # Underlying slv.asr_eos called exactly once.
    assert app.slv.asr_eos.await_count == 1


@pytest.mark.asyncio
async def test_send_asr_eos_once_resets_on_new_turn():
    """After a turn ends (next PTT start or ASRFinal), the flag clears
    so the next turn can fire asr_eos again."""
    app = _fresh_app("push_to_talk")
    app._eos_sent_this_turn = False

    await app.send_asr_eos_once()
    # Simulate new turn boundary — PTT start clears the flag.
    app._eos_sent_this_turn = False
    await app.send_asr_eos_once()
    assert app.slv.asr_eos.await_count == 2


# ── shutdown cancels sleep_task ------------------------------------


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_sleep_task():
    app = _fresh_app("wake_word", sleep_timeout_s=30.0)
    app._llm_turn_task = None
    app._mic_task = None
    app._dispatch_task = None
    app.audio.is_playing = False
    app.audio.close = AsyncMock()
    app.slv.close = AsyncMock()
    app.llm = MagicMock()
    app.llm.aclose = AsyncMock()
    # Arm the sleep timer.
    app._reset_sleep_timer()
    assert app._sleep_task is not None
    task_ref = app._sleep_task
    await app.shutdown()
    assert task_ref.cancelled() or task_ref.done()
    assert app._sleep_task is None
