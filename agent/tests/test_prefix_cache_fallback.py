"""Tests for A4 — prefix_cache failure fallback (independent flag).

Covers:
  • Successful prefix_cache call → flag stays False
  • prefix_cache failure → flag latches, retry without prefix_cache, success
  • Subsequent call on warm+disabled session → omits prefix_cache
  • Non-prefix-cache APIError → propagates, flag untouched
  • Event bus receives ``on_prefix_cache_disabled`` exactly once
  • Session.reset() clears the latch (and ``cache_warmed``)
  • cache_warmed=True + prefix_cache_disabled=True still skips prefix_cache
    (regression guard — codex's primary concern)
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIError

from openvoicestream_agent import Session
from openvoicestream_agent.llm.edge_llm import EdgeLLMBackend


# ── upstream fakes (mirrors tests/test_llm_retry.py style) ──────────────


def _req() -> httpx.Request:
    return httpx.Request("POST", "http://example.invalid/v1/chat/completions")


def _make_api_error(status: int, msg: str = "boom") -> APIError:
    e = APIError(msg, request=_req(), body=None)
    e.status_code = status  # type: ignore[attr-defined]
    return e


class _Delta:
    def __init__(self, content: str | None):
        self.content = content


class _Choice:
    def __init__(self, content: str | None, finish_reason: str | None = None):
        self.delta = _Delta(content)
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, content: str | None, finish_reason: str | None = None):
        self.choices = [_Choice(content, finish_reason)]
        self.model_extra: dict[str, Any] = {}


class _AsyncChunks:
    def __init__(self, script: list[Any]):
        self._script = list(script)
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._script):
            raise StopAsyncIteration
        item = self._script[self._i]
        self._i += 1
        if isinstance(item, BaseException):
            raise item
        return item


class _ScriptedCompletions:
    def __init__(self, plan: list[Any]):
        self._plan = list(plan)
        self.calls = 0
        self.kwargs_history: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.kwargs_history.append(kwargs)
        if not self._plan:
            raise RuntimeError("scripted plan exhausted")
        nxt = self._plan.pop(0)
        self.calls += 1
        if isinstance(nxt, BaseException):
            raise nxt
        if isinstance(nxt, list):
            return _AsyncChunks(nxt)
        return nxt


class _FakeClient:
    def __init__(self, plan: list[Any]):
        self.chat = SimpleNamespace(completions=_ScriptedCompletions(plan))

    async def close(self):
        return None


def _backend(plan: list[Any]) -> EdgeLLMBackend:
    b = EdgeLLMBackend(
        base_url="http://example.invalid/v1",
        api_key="EMPTY",
        model="fake",
        retry_on_transient=0,
        retry_backoff_s=0.0,
    )
    b.client = _FakeClient(plan)  # type: ignore[assignment]
    return b


def _extras(b: EdgeLLMBackend) -> list[dict[str, Any]]:
    return [
        kw.get("extra_body", {})
        for kw in b.client.chat.completions.kwargs_history  # type: ignore[attr-defined]
    ]


# ── tests ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prefix_cache_success_keeps_flag_false():
    """Happy path: warm session → prefix_cache=True request succeeds →
    flag stays False, only one upstream call."""
    session = Session()
    session.cache_warmed = True  # pretend we already warmed

    b = _backend([[_Chunk("你"), _Chunk("好")]])
    toks = [
        t
        async for t in b.stream(
            [{"role": "user", "content": "hi"}], session=session
        )
    ]
    assert toks == ["你", "好"]
    assert session.prefix_cache_disabled is False
    assert b.client.chat.completions.calls == 1  # type: ignore[attr-defined]
    extras = _extras(b)
    assert extras[0].get("prefix_cache") is True


@pytest.mark.asyncio
async def test_prefix_cache_failure_sets_flag_and_retries():
    """First call (with prefix_cache=True) raises APIError mentioning
    prefix_cache → backend retries without prefix_cache → success.
    Flag must latch True, upstream must have been hit twice."""
    session = Session()
    session.cache_warmed = True

    plan = [
        _make_api_error(400, "prefix_cache requires prefix_messages"),
        [_Chunk("ok")],
    ]
    b = _backend(plan)
    toks = [
        t
        async for t in b.stream(
            [{"role": "user", "content": "hi"}], session=session
        )
    ]
    assert toks == ["ok"]
    assert session.prefix_cache_disabled is True
    assert b.client.chat.completions.calls == 2  # type: ignore[attr-defined]
    extras = _extras(b)
    assert extras[0].get("prefix_cache") is True
    assert "prefix_cache" not in extras[1]


@pytest.mark.asyncio
async def test_subsequent_call_skips_prefix_cache():
    """After flag latches, a *new* stream() on the same warm session must
    not send prefix_cache — even though cache_warmed is still True."""
    session = Session()
    session.cache_warmed = True

    plan = [
        _make_api_error(400, "prefix_cache miss"),
        [_Chunk("first")],
        [_Chunk("second")],
    ]
    b = _backend(plan)

    # Turn 1: triggers fallback, latches flag.
    _ = [t async for t in b.stream([{"role": "user", "content": "u1"}], session=session)]
    assert session.prefix_cache_disabled is True

    # Turn 2: must NOT send prefix_cache, even though cache_warmed=True
    # was re-set by the successful retry drain.
    assert session.cache_warmed is True
    _ = [t async for t in b.stream([{"role": "user", "content": "u2"}], session=session)]

    extras = _extras(b)
    # 3 upstream calls total: u1 attempt, u1 retry, u2.
    assert b.client.chat.completions.calls == 3  # type: ignore[attr-defined]
    assert "prefix_cache" not in extras[2]
    # Cold-path flag is the fallback (still asks server to cache the
    # system prompt KV and report metrics).
    assert extras[2].get("save_system_prompt_kv_cache") is True


@pytest.mark.asyncio
async def test_non_prefix_cache_error_propagates():
    """An unrelated 4xx (e.g. rate limit) must NOT trigger fallback —
    flag stays False, exception propagates so A3 / fail-fast can handle."""
    session = Session()
    session.cache_warmed = True

    plan = [
        _make_api_error(429, "rate limit exceeded"),
        [_Chunk("would-not-be-reached")],
    ]
    b = _backend(plan)
    with pytest.raises(APIError):
        _ = [
            t
            async for t in b.stream(
                [{"role": "user", "content": "hi"}], session=session
            )
        ]
    assert session.prefix_cache_disabled is False
    assert b.client.chat.completions.calls == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_event_emitted_on_disable():
    """Fallback must emit ``on_prefix_cache_disabled`` exactly once."""

    events: list[tuple[str, dict[str, Any]]] = []

    class _Bus:
        def emit(self, name: str, payload: dict[str, Any]) -> None:
            events.append((name, payload))

    session = Session()
    session.cache_warmed = True
    session.event_bus = _Bus()

    plan = [
        _make_api_error(400, "kv cache mismatch"),
        [_Chunk("ok")],
    ]
    b = _backend(plan)
    _ = [t async for t in b.stream([{"role": "user", "content": "hi"}], session=session)]

    assert len(events) == 1
    name, payload = events[0]
    assert name == "on_prefix_cache_disabled"
    assert payload["sid"] == session.sid
    assert "kv cache" in payload["reason"].lower()


@pytest.mark.asyncio
async def test_session_reset_clears_flag():
    """Session.reset() must clear both prefix_cache_disabled and
    cache_warmed so the next dialogue can re-warm normally."""
    session = Session()
    session.cache_warmed = True
    session.prefix_cache_disabled = True
    session.add_user("u")
    session.add_assistant("a")

    session.reset()

    assert session.prefix_cache_disabled is False
    assert session.cache_warmed is False
    assert session.history == []


@pytest.mark.asyncio
async def test_cache_warmed_true_with_disabled_does_not_send_prefix():
    """Regression: codex's primary concern. A warm session with the latch
    set must, on EVERY call, omit prefix_cache. No alternation, no loop
    back into prefix_cache=True even after successful drains keep
    re-setting cache_warmed."""
    session = Session()
    session.cache_warmed = True
    session.prefix_cache_disabled = True

    plan = [
        [_Chunk("a")],
        [_Chunk("b")],
        [_Chunk("c")],
    ]
    b = _backend(plan)

    for _ in range(3):
        _ = [
            t
            async for t in b.stream(
                [{"role": "user", "content": "x"}], session=session
            )
        ]

    extras = _extras(b)
    assert len(extras) == 3
    for e in extras:
        assert "prefix_cache" not in e
        assert e.get("save_system_prompt_kv_cache") is True
    # And the latch never spontaneously cleared.
    assert session.prefix_cache_disabled is True
