"""Tests for LLM timeout watchdog in ModeContext.run_default_dialogue_turn
and the BaseApp recovery path that resets FSM to IDLE + fires on_error.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from openvoicestream_agent import Config, Session
from openvoicestream_agent.app_base import BaseApp
from openvoicestream_agent.app_mode import LLMTimeoutError, ModeContext
from openvoicestream_agent.event_bus import EventBus
from openvoicestream_agent.state import ConvState


# ── shared fakes -----------------------------------------------------


class FakeSLV:
    def __init__(self) -> None:
        self.text_frames: list[str] = []
        self.flushed: int = 0

    async def send_text(self, text: str) -> None:
        self.text_frames.append(text)

    async def flush_tts(self) -> None:
        self.flushed += 1


class _LLMBase:
    last_cache_metrics: dict | None = None

    async def aclose(self) -> None:
        pass


def _make_ctx(llm, cfg=None):
    cfg = cfg or Config(
        llm_first_token_timeout_s=0.2,
        llm_stream_idle_timeout_s=0.2,
    )
    slv = FakeSLV()
    session = Session()
    broadcasts: list[tuple] = []

    async def _br(name, *args):
        broadcasts.append((name, args))

    events = type("E", (), {"emit": lambda *a, **k: None})()
    ctx = ModeContext(
        config=cfg, slv=slv, llm=llm, session=session, audio=None,
        events=events, broadcast=_br,
    )
    return ctx, slv, broadcasts


# ── config validation -----------------------------------------------


def test_config_rejects_nonpositive_first_token_timeout():
    with pytest.raises(ValueError):
        Config(llm_first_token_timeout_s=0)
    with pytest.raises(ValueError):
        Config(llm_first_token_timeout_s=-1)


def test_config_rejects_nonpositive_idle_timeout():
    with pytest.raises(ValueError):
        Config(llm_stream_idle_timeout_s=0)


def test_config_defaults_present():
    cfg = Config()
    assert cfg.llm_first_token_timeout_s == 15.0
    assert cfg.llm_stream_idle_timeout_s == 30.0


# ── stream watchdog --------------------------------------------------


@pytest.mark.asyncio
async def test_first_token_timeout():
    """LLM that never yields → LLMTimeoutError(kind=first_token, partial='')."""

    class NeverYieldLLM(_LLMBase):
        async def stream(self, messages, **kw):
            await asyncio.sleep(60)
            yield "should not arrive"  # pragma: no cover

    ctx, _slv, _ = _make_ctx(NeverYieldLLM())
    with pytest.raises(LLMTimeoutError) as ei:
        await ctx.run_default_dialogue_turn("hi")
    assert ei.value.kind == "first_token"
    assert ei.value.partial_text == ""
    assert ei.value.timeout_s == pytest.approx(0.2, abs=1e-3)


@pytest.mark.asyncio
async def test_stream_idle_timeout():
    """LLM yields one token then hangs → kind=stream_idle, partial='first'."""

    class OneThenHangLLM(_LLMBase):
        async def stream(self, messages, **kw):
            yield "first"
            await asyncio.sleep(60)
            yield "never"  # pragma: no cover

    ctx, slv, _ = _make_ctx(OneThenHangLLM())
    with pytest.raises(LLMTimeoutError) as ei:
        await ctx.run_default_dialogue_turn("hi")
    assert ei.value.kind == "stream_idle"
    assert ei.value.partial_text == "first"
    # The first token did reach SLV before the hang.
    assert slv.text_frames == ["first"]
    # finally-block still flushes TTS so SLV isn't left buffering.
    assert slv.flushed == 1


@pytest.mark.asyncio
async def test_normal_stream_no_timeout():
    """Quick complete stream must NOT raise."""

    class QuickLLM(_LLMBase):
        async def stream(self, messages, **kw):
            yield "hello"
            yield " world"

    # Use generous timeout to confirm fast streams don't trip the watchdog.
    cfg = Config(
        llm_first_token_timeout_s=5.0,
        llm_stream_idle_timeout_s=5.0,
    )
    ctx, slv, _ = _make_ctx(QuickLLM(), cfg=cfg)
    await ctx.run_default_dialogue_turn("hi")
    assert slv.text_frames == ["hello", " world"]
    assert slv.flushed == 1
    # Assistant message appended to history.
    assert ctx.session.history[-1]["role"] == "assistant"
    assert ctx.session.history[-1]["content"] == "hello world"


# ── BaseApp recovery -------------------------------------------------


def _fresh_app() -> BaseApp:
    app = BaseApp.__new__(BaseApp)
    app.events = EventBus()
    app.plugins = []
    app._state = ConvState.THINKING
    app._slv_reconnect_count = 0
    app._sleep_task = None
    app.config = Config(pipeline_mode="always_on")
    return app


@pytest.mark.asyncio
async def test_llm_timeout_resets_to_idle():
    """_run_user_utterance must catch LLMTimeoutError, broadcast on_error
    with a human-readable str, and reset state to IDLE."""
    app = _fresh_app()

    errors: list[BaseException] = []

    class _ErrorPlugin:
        name = "errp"
        async def on_error(self, exc):
            errors.append(exc)

    app.plugins.append(_ErrorPlugin())

    async def boom(text: str) -> None:
        raise LLMTimeoutError("first_token", 15.0)

    app.on_user_utterance = boom  # type: ignore[assignment]
    await app._run_user_utterance("hi")

    assert app._state == ConvState.IDLE
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    msg = str(errors[0])
    assert "LLM 响应超时" in msg
    assert "first_token" in msg


@pytest.mark.asyncio
async def test_generic_exception_also_resets_idle_and_resets_sleep():
    """Regression: non-timeout exceptions must still leave FSM IDLE."""
    app = _fresh_app()

    async def boom(text: str) -> None:
        raise RuntimeError("oops")

    app.on_user_utterance = boom  # type: ignore[assignment]
    await app._run_user_utterance("hi")
    assert app._state == ConvState.IDLE


# ── dashboard error formatting --------------------------------------


@pytest.mark.asyncio
async def test_debug_dashboard_on_error_prefers_str():
    """Dashboard.on_error should display str(exc) (clean Chinese msg) not
    repr(exc) when str is non-empty."""
    from openvoicestream_agent.plugins.debug_dashboard import DebugDashboardPlugin

    p = DebugDashboardPlugin.__new__(DebugDashboardPlugin)
    p._errors = []
    p._broadcast = lambda *a, **k: _noop()  # type: ignore

    exc = RuntimeError("LLM 响应超时（first_token, >15s）。可能 edge-llm 服务挂了或输入太长。")
    await p.on_error(exc)
    assert len(p._errors) == 1
    assert p._errors[0]["msg"].startswith("LLM 响应超时")
    assert "RuntimeError(" not in p._errors[0]["msg"]


async def _noop():
    return None
