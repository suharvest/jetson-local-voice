"""Streaming LLM cache metrics plumbing."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from openvoicestream_agent import Config, Session
from openvoicestream_agent.app_mode import ModeContext
from openvoicestream_agent.llm.edge_llm import EdgeLLMBackend
from openvoicestream_agent.llm.openai_compat import OpenAICompatBackend


class _Delta:
    def __init__(self, content: str | None):
        self.content = content


class _Choice:
    def __init__(self, content: str | None):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content: str | None, cache_metrics=None):
        self.choices = [_Choice(content)]
        self.model_extra = {}
        if cache_metrics is not None:
            self.model_extra["cache_metrics"] = cache_metrics


class _Completions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return _AsyncChunks()


class _AsyncChunks:
    def __aiter__(self):
        self._chunks = iter([
            _Chunk("你"),
            _Chunk("好"),
            _Chunk(None, {"prefill": {"reused_tokens": 6, "computed_tokens": 14}}),
        ])
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_Completions())

    async def close(self):
        return None


def _backend() -> OpenAICompatBackend:
    backend = OpenAICompatBackend(
        base_url="http://example.invalid/v1",
        api_key="EMPTY",
        model="fake",
    )
    backend.client = _FakeClient()
    return backend


@pytest.mark.asyncio
async def test_openai_compat_captures_final_streaming_cache_metrics():
    backend = _backend()
    tokens = [t async for t in backend.stream([{"role": "user", "content": "hi"}])]
    assert tokens == ["你", "好"]
    assert backend.last_cache_metrics == {
        "prefill": {"reused_tokens": 6, "computed_tokens": 14}
    }


@pytest.mark.asyncio
async def test_edge_llm_requests_streaming_cache_metrics_for_cold_and_warm_session():
    backend = EdgeLLMBackend("http://example.invalid/v1", "EMPTY", "fake")
    backend.client = _FakeClient()
    session = Session()

    _ = [t async for t in backend.stream([{"role": "user", "content": "hi"}], session=session)]
    first_extra = backend.client.chat.completions.kwargs["extra_body"]
    assert first_extra["save_system_prompt_kv_cache"] is True
    assert first_extra["return_cache_metrics"] is True
    assert session.cache_warmed is True

    _ = [t async for t in backend.stream([{"role": "user", "content": "again"}], session=session)]
    second_extra = backend.client.chat.completions.kwargs["extra_body"]
    assert second_extra["prefix_cache"] is True
    assert second_extra["return_cache_metrics"] is True


@pytest.mark.asyncio
async def test_mode_context_broadcasts_cache_metrics_after_stream_drains():
    class FakeSLV:
        async def send_text(self, _text):
            return None

        async def flush_tts(self):
            return None

    backend = _backend()
    broadcasts = []

    async def broadcast(name, *args):
        broadcasts.append((name, args))

    ctx = ModeContext(
        config=Config(system_prompt="SYS"),
        slv=FakeSLV(),
        llm=backend,
        session=Session(),
        audio=None,
        events=SimpleNamespace(emit=lambda *args, **kwargs: None),
        broadcast=broadcast,
    )

    await ctx.run_default_dialogue_turn("hi")

    assert ("on_llm_cache_metrics", (backend.last_cache_metrics,)) in broadcasts
