"""Tests for A3 — transparent LLM retry + SSE error-frame detection.

Covers:
  • 5xx APIError, APIConnectionError, APITimeoutError → retried
  • 4xx APIError → NOT retried
  • Retry budget exhausted → final exception raised
  • Backoff respected (asyncio.sleep called with configured interval)
  • Mid-stream failure (after a token was yielded) → NOT retried
  • First-chunk-only failure → retried and succeeds
  • SSE finish_reason="error" frame → raises LLMStreamError, NOT retried
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIConnectionError, APIError, APITimeoutError

from openvoicestream_agent.llm.openai_compat import (
    LLMStreamError,
    OpenAICompatBackend,
)


# ── upstream chunk fakes -------------------------------------------------


def _req() -> httpx.Request:
    return httpx.Request("POST", "http://example.invalid/v1/chat/completions")


def _make_api_error(status: int, msg: str = "boom") -> APIError:
    e = APIError(msg, request=_req(), body=None)
    # APIError doesn't set status_code by default; our retry classifier
    # inspects it duck-typed, so attach it explicitly.
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
    def __init__(
        self,
        content: str | None,
        finish_reason: str | None = None,
    ):
        self.choices = [_Choice(content, finish_reason)]
        self.model_extra: dict[str, Any] = {}


class _AsyncChunks:
    """Async iterator over a script of chunks-or-exceptions."""

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
    """Each call returns the next scripted response (or raises)."""

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
        # `nxt` is either a list of chunks → wrap into _AsyncChunks
        # or an already-iterable object.
        if isinstance(nxt, list):
            return _AsyncChunks(nxt)
        return nxt


class _FakeClient:
    def __init__(self, plan: list[Any]):
        self.chat = SimpleNamespace(completions=_ScriptedCompletions(plan))

    async def close(self):
        return None


def _backend(plan: list[Any], **kw: Any) -> OpenAICompatBackend:
    b = OpenAICompatBackend(
        base_url="http://example.invalid/v1",
        api_key="EMPTY",
        model="fake",
        **kw,
    )
    b.client = _FakeClient(plan)  # type: ignore[assignment]
    return b


# ── retry classification ------------------------------------------------


@pytest.mark.asyncio
async def test_transient_5xx_retried_once():
    """First call 503 → second call returns tokens. Total upstream calls = 2."""
    plan = [
        _make_api_error(503),
        [_Chunk("你"), _Chunk("好")],
    ]
    b = _backend(plan, retry_on_transient=1, retry_backoff_s=0.0)
    toks = [t async for t in b.stream([{"role": "user", "content": "hi"}])]
    assert toks == ["你", "好"]
    assert b.client.chat.completions.calls == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_4xx_not_retried():
    """400 must NOT trigger retry."""
    plan = [
        _make_api_error(400),
        [_Chunk("nope")],  # would only be reached if retry happened
    ]
    b = _backend(plan, retry_on_transient=3, retry_backoff_s=0.0)
    with pytest.raises(APIError) as ei:
        _ = [t async for t in b.stream([{"role": "user", "content": "hi"}])]
    assert ei.value.status_code == 400  # type: ignore[attr-defined]
    assert b.client.chat.completions.calls == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_connection_error_retried():
    plan = [
        APIConnectionError(request=_req()),
        [_Chunk("ok")],
    ]
    b = _backend(plan, retry_on_transient=1, retry_backoff_s=0.0)
    toks = [t async for t in b.stream([{"role": "user", "content": "hi"}])]
    assert toks == ["ok"]
    assert b.client.chat.completions.calls == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_timeout_retried():
    plan = [
        APITimeoutError(request=_req()),
        [_Chunk("ok")],
    ]
    b = _backend(plan, retry_on_transient=1, retry_backoff_s=0.0)
    toks = [t async for t in b.stream([{"role": "user", "content": "hi"}])]
    assert toks == ["ok"]
    assert b.client.chat.completions.calls == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_retry_exhausted_raises():
    """All attempts fail → the *last* exception propagates."""
    plan = [
        APIConnectionError(request=_req()),
        _make_api_error(502),
        _make_api_error(500),
    ]
    b = _backend(plan, retry_on_transient=2, retry_backoff_s=0.0)
    with pytest.raises(APIError) as ei:
        _ = [t async for t in b.stream([{"role": "user", "content": "hi"}])]
    assert ei.value.status_code == 500  # type: ignore[attr-defined]
    assert b.client.chat.completions.calls == 3  # type: ignore[attr-defined]


# ── SSE error frame -----------------------------------------------------


@pytest.mark.asyncio
async def test_sse_finish_reason_error_raises_llm_stream_error():
    """An upstream chunk with finish_reason='error' must surface as
    LLMStreamError and NOT be retried (we've already yielded text)."""
    plan = [
        [_Chunk("partial"), _Chunk(None, finish_reason="error")],
        [_Chunk("would-be-retry")],
    ]
    b = _backend(plan, retry_on_transient=1, retry_backoff_s=0.0)
    yielded: list[str] = []
    with pytest.raises(LLMStreamError):
        async for tok in b.stream([{"role": "user", "content": "hi"}]):
            yielded.append(tok)
    assert yielded == ["partial"]
    assert b.client.chat.completions.calls == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_sse_finish_reason_error_first_frame_not_retried():
    """Even if the error frame is the *first* frame upstream sends, we
    still don't retry — by spec LLMStreamError is a 'stream gave up'
    signal, not a connect-level fault."""
    plan = [
        [_Chunk(None, finish_reason="error")],
        [_Chunk("would-be-retry")],
    ]
    b = _backend(plan, retry_on_transient=1, retry_backoff_s=0.0)
    with pytest.raises(LLMStreamError):
        _ = [t async for t in b.stream([{"role": "user", "content": "hi"}])]
    assert b.client.chat.completions.calls == 1  # type: ignore[attr-defined]


# ── mid-stream vs pre-stream failures -----------------------------------


@pytest.mark.asyncio
async def test_mid_stream_failure_not_retried():
    """Token already yielded → failure must NOT be retried."""

    class _BrokenAfterFirst:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not hasattr(self, "_done"):
                self._done = True
                return _Chunk("first")
            raise APIConnectionError(request=_req())

    plan = [
        _BrokenAfterFirst(),
        [_Chunk("would-be-retry")],
    ]
    b = _backend(plan, retry_on_transient=3, retry_backoff_s=0.0)
    seen: list[str] = []
    with pytest.raises(APIConnectionError):
        async for tok in b.stream([{"role": "user", "content": "hi"}]):
            seen.append(tok)
    assert seen == ["first"]
    assert b.client.chat.completions.calls == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_first_chunk_failure_retried():
    """If the upstream call itself raises (before any token), retry."""
    plan = [
        APIConnectionError(request=_req()),
        [_Chunk("hello")],
    ]
    b = _backend(plan, retry_on_transient=1, retry_backoff_s=0.0)
    toks = [t async for t in b.stream([{"role": "user", "content": "hi"}])]
    assert toks == ["hello"]
    assert b.client.chat.completions.calls == 2  # type: ignore[attr-defined]


# ── backoff -------------------------------------------------------------


@pytest.mark.asyncio
async def test_backoff_respected(monkeypatch):
    """asyncio.sleep should be invoked with the configured backoff between
    retries (we patch sleep so the test stays fast)."""
    sleeps: list[float] = []

    import asyncio as _asyncio

    async def _fake_sleep(delay):
        sleeps.append(float(delay))

    monkeypatch.setattr(
        "openvoicestream_agent.llm.openai_compat.asyncio.sleep",
        _fake_sleep,
    )

    plan = [
        APIConnectionError(request=_req()),
        APIConnectionError(request=_req()),
        [_Chunk("ok")],
    ]
    b = _backend(plan, retry_on_transient=2, retry_backoff_s=0.5)
    toks = [t async for t in b.stream([{"role": "user", "content": "hi"}])]
    assert toks == ["ok"]
    assert sleeps == [0.5, 0.5]
    # Sanity: our patched sleep is the only sleep that ran here
    assert _asyncio is not None
