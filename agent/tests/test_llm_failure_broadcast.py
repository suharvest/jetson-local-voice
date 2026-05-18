"""A3 — non-timeout LLM exceptions must broadcast `on_error` to plugins
so the dashboard surfaces them (previously only LLMTimeoutError did)."""
from __future__ import annotations

import pytest

from openvoicestream_agent import Config
from openvoicestream_agent.app_base import BaseApp
from openvoicestream_agent.event_bus import EventBus
from openvoicestream_agent.state import ConvState


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
async def test_generic_llm_failure_broadcasts_on_error():
    """A RuntimeError from on_user_utterance must:
      1) Reset FSM to IDLE,
      2) Fire on_error on plugins with a RuntimeError whose message
         identifies the original exception class.
    """
    app = _fresh_app()

    errors: list[BaseException] = []

    class _ErrPlugin:
        name = "errp"

        async def on_error(self, exc):
            errors.append(exc)

    app.plugins.append(_ErrPlugin())

    async def boom(text: str) -> None:
        raise RuntimeError("edge-llm down")

    app.on_user_utterance = boom  # type: ignore[assignment]
    await app._run_user_utterance("hi")

    assert app._state == ConvState.IDLE
    assert len(errors) == 1
    msg = str(errors[0])
    assert "LLM 调用失败" in msg
    assert "RuntimeError" in msg
    assert "edge-llm down" in msg


@pytest.mark.asyncio
async def test_llm_stream_error_broadcasts_on_error():
    """A LLMStreamError (SSE mid-stream upstream failure) must also be
    surfaced, not silently swallowed."""
    from openvoicestream_agent.llm import LLMStreamError

    app = _fresh_app()
    errors: list[BaseException] = []

    class _ErrPlugin:
        name = "errp"

        async def on_error(self, exc):
            errors.append(exc)

    app.plugins.append(_ErrPlugin())

    async def boom(text: str) -> None:
        raise LLMStreamError("finish_reason=error")

    app.on_user_utterance = boom  # type: ignore[assignment]
    await app._run_user_utterance("hi")

    assert app._state == ConvState.IDLE
    assert len(errors) == 1
    msg = str(errors[0])
    assert "LLM 调用失败" in msg
    assert "LLMStreamError" in msg
