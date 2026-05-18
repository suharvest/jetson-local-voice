"""MED-1: A3 (transient retry) × A4 (prefix_cache fallback) must not stack.

Scenario: the upstream surfaces a prefix_cache failure as a 5xx (regression
case — normally it's 400, which A3 doesn't retry, so the issue lies dormant).

Before the fix:
    attempt 1 (prefix_cache=True, 5xx)
    A3 retry → attempt 2 (prefix_cache=True, 5xx)
    raise → A4 catches, latches prefix_cache_disabled, retries
        attempt 3 (no prefix_cache, 5xx)
        A3 retry → attempt 4 (no prefix_cache, success)
    = 4 upstream hits per turn.

After the fix:
    attempt 1 (prefix_cache=True, 5xx)
    A3 retry → attempt 2 (prefix_cache=True, 5xx)
    raise → A4 catches, retries with _retry_disabled=True
        attempt 3 (no prefix_cache, success)
    = 3 upstream hits per turn.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIError

from openvoicestream_agent.llm.edge_llm import EdgeLLMBackend
from openvoicestream_agent.session import Session


def _req() -> httpx.Request:
    return httpx.Request("POST", "http://example.invalid/v1/chat/completions")


def _make_api_error(status: int, msg: str) -> APIError:
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


def _backend(plan: list[Any], *, retry_on_transient: int = 1) -> EdgeLLMBackend:
    b = EdgeLLMBackend(
        base_url="http://example.invalid/v1",
        api_key="EMPTY",
        model="fake",
        retry_on_transient=retry_on_transient,
        retry_backoff_s=0.0,
    )
    b.client = _FakeClient(plan)  # type: ignore[assignment]
    return b


@pytest.mark.asyncio
async def test_a4_fallback_does_not_double_retry_via_a3():
    """5xx prefix_cache failure: A3 retries once, then A4 fallback runs
    a single attempt (no inner A3 retry) → 3 total upstream calls, not 4.
    """
    session = Session()
    session.cache_warmed = True  # so prefix_cache=True is on the first call

    # 1st call (prefix_cache=True): 5xx with prefix_cache marker
    # 2nd call (A3 retry, prefix_cache=True): same 5xx
    # 3rd call (A4 fallback, prefix_cache omitted): success
    # MUST NOT see a 4th call (would be A3 retrying the A4 fallback).
    plan = [
        _make_api_error(503, "prefix_cache: server overloaded"),
        _make_api_error(503, "prefix_cache: server overloaded"),
        [_Chunk("ok")],
    ]
    b = _backend(plan, retry_on_transient=1)

    toks = [
        t
        async for t in b.stream(
            [{"role": "user", "content": "hi"}], session=session
        )
    ]
    assert toks == ["ok"]
    # The critical assertion: exactly 3 calls, not 4.
    assert b.client.chat.completions.calls == 3, (  # type: ignore[attr-defined]
        f"expected 3 upstream calls (A3 retry + A4 fallback w/o nested retry), "
        f"got {b.client.chat.completions.calls}"  # type: ignore[attr-defined]
    )
    # And the flag is latched.
    assert session.prefix_cache_disabled is True

    # Verify the 3rd call was made WITHOUT prefix_cache.
    third_extra = b.client.chat.completions.kwargs_history[2].get("extra_body", {})  # type: ignore[attr-defined]
    assert "prefix_cache" not in third_extra
    assert third_extra.get("save_system_prompt_kv_cache") is True


@pytest.mark.asyncio
async def test_a4_fallback_5xx_no_retry_propagates():
    """If even the A4 fallback gets a 5xx, we should propagate after one
    attempt — not silently retry again, which would explode the budget."""
    session = Session()
    session.cache_warmed = True

    plan = [
        _make_api_error(503, "prefix_cache: server overloaded"),
        _make_api_error(503, "prefix_cache: server overloaded"),
        # A4 fallback call — also 5xx, but no prefix_cache marker. With
        # _retry_disabled=True, this should NOT trigger another A3 retry.
        _make_api_error(503, "still overloaded"),
    ]
    b = _backend(plan, retry_on_transient=1)

    with pytest.raises(APIError):
        async for _ in b.stream(
            [{"role": "user", "content": "hi"}], session=session
        ):
            pass

    assert b.client.chat.completions.calls == 3, (  # type: ignore[attr-defined]
        f"expected 3 calls (A3 retry + A4 fallback w/o nested retry), "
        f"got {b.client.chat.completions.calls}"  # type: ignore[attr-defined]
    )


@pytest.mark.asyncio
async def test_retry_disabled_kwarg_bypasses_a3_loop():
    """Direct unit test on OpenAICompatBackend: passing _retry_disabled=True
    via kwargs makes a 5xx propagate immediately (no retries)."""
    from openvoicestream_agent.llm.openai_compat import OpenAICompatBackend

    plan = [_make_api_error(503, "boom")]
    b = OpenAICompatBackend(
        base_url="http://example.invalid/v1",
        api_key="EMPTY",
        model="fake",
        retry_on_transient=3,  # would normally retry 3 times
        retry_backoff_s=0.0,
    )
    b.client = _FakeClient(plan)  # type: ignore[assignment]

    with pytest.raises(APIError):
        async for _ in b.stream(
            [{"role": "user", "content": "hi"}], _retry_disabled=True
        ):
            pass

    assert b.client.chat.completions.calls == 1, (  # type: ignore[attr-defined]
        f"with _retry_disabled=True, expected 1 attempt, "
        f"got {b.client.chat.completions.calls}"  # type: ignore[attr-defined]
    )


@pytest.mark.asyncio
async def test_retry_disabled_not_leaked_to_request():
    """The _retry_disabled internal kwarg must be popped before being sent
    to OpenAI's chat.completions.create — otherwise the upstream sees an
    unknown parameter."""
    from openvoicestream_agent.llm.openai_compat import OpenAICompatBackend

    plan = [[_Chunk("ok")]]
    b = OpenAICompatBackend(
        base_url="http://example.invalid/v1",
        api_key="EMPTY",
        model="fake",
        retry_on_transient=0,
        retry_backoff_s=0.0,
    )
    b.client = _FakeClient(plan)  # type: ignore[assignment]

    toks = [
        t
        async for t in b.stream(
            [{"role": "user", "content": "hi"}], _retry_disabled=True
        )
    ]
    assert toks == ["ok"]
    sent_kwargs = b.client.chat.completions.kwargs_history[0]  # type: ignore[attr-defined]
    assert "_retry_disabled" not in sent_kwargs
