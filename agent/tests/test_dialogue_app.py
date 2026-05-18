"""Default dialogue turn: tokens stream directly to SLV; no client-side batching.

Historically this was DialogueApp.on_user_utterance; the same logic now
lives in ModeContext.run_default_dialogue_turn (invoked by ChatMode).
"""
from __future__ import annotations

from typing import Any

import pytest

from openvoicestream_agent import Config, Session
from openvoicestream_agent.app_mode import ModeContext, ModeManager
from openvoicestream_agent.apps_dialogue_shim import DialogueApp  # back-compat alias
from openvoicestream_agent.llm.base import LLMBackend
from openvoicestream_agent.modes import ChatMode


class FakeSLV:
    def __init__(self) -> None:
        self.text_frames: list[str] = []
        self.flushed: int = 0
        self.aborted: int = 0

    async def send_text(self, text: str) -> None:
        self.text_frames.append(text)

    async def flush_tts(self) -> None:
        self.flushed += 1

    async def abort(self) -> None:
        self.aborted += 1


class FakeLLM(LLMBackend):
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.last_messages: list[dict[str, str]] | None = None
        self.last_session: Any = None

    async def stream(self, messages, **kw):  # type: ignore[override]
        self.last_messages = list(messages)
        self.last_session = kw.get("session")
        for t in self.tokens:
            yield t


async def _noop_broadcast(*args, **kwargs):
    return None


@pytest.mark.asyncio
async def test_default_dialogue_turn_streams_tokens_directly_to_slv():
    cfg = Config(system_prompt="SYS")
    slv = FakeSLV()
    llm = FakeLLM(["你", "好", "，", "世界。"])
    session = Session()
    events = type("E", (), {"emit": lambda *a, **k: None})()

    ctx = ModeContext(
        config=cfg, slv=slv, llm=llm, session=session, audio=None,
        events=events, broadcast=_noop_broadcast,
    )
    mgr = ModeManager(lambda: ctx)
    mgr.register(ChatMode())
    await mgr.start("chat")

    await mgr.current.on_user_utterance(ctx, "hi")

    # Every LLM token forwarded individually (no batching/joining).
    assert slv.text_frames == ["你", "好", "，", "世界。"]
    # flush_tts called exactly once after stream ends.
    assert slv.flushed == 1
    # History has user + assistant entries.
    assert session.history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "你好，世界。"},
    ]
    # LLM saw full messages including the configured system prompt.
    assert llm.last_messages[0] == {"role": "system", "content": "SYS"}
    assert llm.last_messages[-1] == {"role": "user", "content": "hi"}
    # session was passed through to LLM (for prefix-cache control).
    assert llm.last_session is session


@pytest.mark.asyncio
async def test_multi_mode_app_class_is_back_compat_dialogue_shim():
    """The legacy `DialogueApp` import path now resolves to MultiModeApp."""
    from apps.multi_mode.app import MultiModeApp

    assert DialogueApp is MultiModeApp
