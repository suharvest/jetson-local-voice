"""Fail-fast integration: when LLMAvailability says DOWN,
run_default_dialogue_turn must raise LLMUnavailable immediately — NOT wait
for the LLM first-token timeout.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from openvoicestream_agent import Config, Session
from openvoicestream_agent.app_base import BaseApp
from openvoicestream_agent.app_mode import ModeContext
from openvoicestream_agent.event_bus import EventBus
from openvoicestream_agent.plugins.llm_availability import (
    AvailabilityState,
    LLMAvailabilityPlugin,
    LLMUnavailable,
)
from openvoicestream_agent.state import ConvState


class _SlowLLM:
    """Would block forever — if fail-fast works we never touch it."""
    last_cache_metrics = None

    async def stream(self, messages, **kw):
        await asyncio.sleep(60)
        yield "never"  # pragma: no cover

    async def aclose(self):  # pragma: no cover
        pass


def _fresh_app() -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.events = EventBus()
    app.plugins = []
    app._state = ConvState.THINKING
    app._slv_reconnect_count = 0
    app._sleep_task = None
    app.config = Config(pipeline_mode="always_on", llm_first_token_timeout_s=15.0)
    app.llm_availability = None
    app._llm_turn_task = None
    app._mic_task = None
    app._dispatch_task = None
    return app


def _make_down_plugin(app) -> LLMAvailabilityPlugin:
    p = LLMAvailabilityPlugin(app)
    # Force DOWN state without running probes.
    p.state = AvailabilityState.DOWN
    p.consecutive_failures = 3
    app.llm_availability = p
    return p


@pytest.mark.asyncio
async def test_run_default_dialogue_turn_fails_fast_when_down():
    """ctx.run_default_dialogue_turn → LLMUnavailable in well under the
    LLM timeout (which is 15s by default)."""
    app = _fresh_app()
    _make_down_plugin(app)

    class _Bus:
        def emit(self, *a, **k):
            pass

    ctx = ModeContext(
        config=app.config,
        slv=None,
        llm=_SlowLLM(),
        session=Session(),
        audio=None,
        events=_Bus(),
        broadcast=app.broadcast,  # bound — gives ctx access to app.llm_availability
    )

    t0 = time.monotonic()
    with pytest.raises(LLMUnavailable):
        await ctx.run_default_dialogue_turn("hello")
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"fail-fast took {elapsed:.2f}s (expected <0.5s)"


@pytest.mark.asyncio
async def test_app_base_catches_llm_unavailable_and_returns_idle():
    """BaseApp._run_user_utterance catches LLMUnavailable, broadcasts
    on_error('LLM 不可用：...'), and returns FSM to IDLE."""
    app = _fresh_app()
    _make_down_plugin(app)

    captured: list[BaseException] = []

    class _CaptureErrorsPlugin:
        name = "capture"

        async def on_error(self, exc):
            captured.append(exc)

    app.plugins.append(_CaptureErrorsPlugin())

    async def _fail(text):
        raise LLMUnavailable("LLM is DOWN (consecutive failures: 3)")

    app.on_user_utterance = _fail  # type: ignore[assignment]
    await app._run_user_utterance("hi")

    assert app._state == ConvState.IDLE
    assert len(captured) == 1
    msg = str(captured[0])
    assert "LLM 不可用" in msg


@pytest.mark.asyncio
async def test_healthy_state_does_not_fail_fast():
    """When state is HEALTHY, run_default_dialogue_turn proceeds normally."""
    app = _fresh_app()
    p = LLMAvailabilityPlugin(app)
    p.state = AvailabilityState.HEALTHY
    app.llm_availability = p

    class _OK:
        last_cache_metrics = None

        async def stream(self, messages, **kw):
            yield "hi"

        async def aclose(self):  # pragma: no cover
            pass

    class _SLV:
        def __init__(self):
            self.frames = []

        async def send_text(self, t):
            self.frames.append(t)

        async def flush_tts(self):
            pass

    class _Bus:
        def emit(self, *a, **k):
            pass

    slv = _SLV()
    ctx = ModeContext(
        config=app.config,
        slv=slv,
        llm=_OK(),
        session=Session(),
        audio=None,
        events=_Bus(),
        broadcast=app.broadcast,
    )
    await ctx.run_default_dialogue_turn("hello")
    assert slv.frames == ["hi"]
