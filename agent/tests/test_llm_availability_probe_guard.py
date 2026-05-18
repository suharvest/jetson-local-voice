"""Issue #5 — LLM probe must distinguish guard-rejected 400s from real failures.

If an input-validation middleware (e.g. SLV's input_too_long guard with a
low test threshold) rejects the probe's tiny ``content="."`` payload with
400 + structured error code, that's *not* evidence that the LLM is DOWN —
the LLM never even saw the request. Counting it as a failure would silently
drive the breaker toward DOWN on every cycle.

Verify:
  • 400 + {"error": {"code": "input_too_long"}} → probe returns None (unknown).
  • 400 + {"error": {"code": "invalid_request"}} → probe returns None.
  • 400 with unstructured body (no recognised error.code) → still counts as failure.
  • Repeated guard rejections must NOT advance the state machine to DEGRADED
    (they only count toward consecutive_unknowns → UNKNOWN).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from openvoicestream_agent.plugins.llm_availability import (
    AvailabilityState,
    LLMAvailabilityPlugin,
)


# ── helpers (mirrors tests/test_llm_availability.py) ───────────────────


def _make_app() -> Any:
    cfg = SimpleNamespace(
        llm_base_url="http://localhost:8000/v1",
        llm_api_key="",
        llm_model="qwen-test",
        llm_availability_enabled=True,
        llm_availability_probe_interval_s=0.05,
        llm_availability_probe_timeout_s=0.5,
        llm_availability_failures_to_down=3,
    )

    class _EventBus:
        def __init__(self) -> None:
            self.emitted: list[tuple[str, Any]] = []

        def emit(self, event: str, data: Any = None) -> None:
            self.emitted.append((event, data))

    class _App:
        def __init__(self) -> None:
            self.config = cfg
            self.events = _EventBus()
            self.llm_availability = None

        async def broadcast(self, hook: str, data: Any) -> None:
            return None

    return _App()


def _make_plugin(app) -> LLMAvailabilityPlugin:
    p = LLMAvailabilityPlugin(app)
    p._wake_evt = asyncio.Event()
    p._probe_lock = asyncio.Lock()
    return p


class _MockResp:
    def __init__(self, status: int, body: Any = None):
        self.status_code = status
        self._body = body

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _FakeClient:
    def __init__(self, *, response: Any, **_kwargs) -> None:
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url: str, *, json=None, headers=None):
        return self._response


def _patch_httpx(monkeypatch, *, response):
    def factory(*args, **kwargs):
        return _FakeClient(response=response, **kwargs)
    monkeypatch.setattr(
        "openvoicestream_agent.plugins.llm_availability.httpx.AsyncClient",
        factory,
    )


# ── tests ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_probe_guard_400_input_too_long_treated_as_unknown(monkeypatch):
    """The SLV input_too_long guard returning 400 must classify as unknown,
    not a failure."""
    app = _make_app()
    p = _make_plugin(app)
    _patch_httpx(monkeypatch, response=_MockResp(
        400, {"error": {"code": "input_too_long", "message": "too long"}}
    ))

    result = await p._probe()
    assert result is None, f"expected unknown (None), got {result!r}"


@pytest.mark.asyncio
async def test_probe_guard_400_invalid_request_treated_as_unknown(monkeypatch):
    app = _make_app()
    p = _make_plugin(app)
    _patch_httpx(monkeypatch, response=_MockResp(
        400, {"error": {"code": "invalid_request", "message": "bad payload"}}
    ))

    result = await p._probe()
    assert result is None, f"expected unknown (None), got {result!r}"


@pytest.mark.asyncio
async def test_probe_400_unstructured_still_counts_as_failure(monkeypatch):
    """A 400 with no recognised error.code (or no error object at all)
    is still a real failure — could be a misconfigured endpoint, auth,
    etc. We can't safely treat all 400s as unknown."""
    app = _make_app()
    p = _make_plugin(app)
    # Unstructured: just an error string, no .error.code.
    _patch_httpx(monkeypatch, response=_MockResp(400, {"detail": "bad request"}))

    result = await p._probe()
    assert result is False, f"expected failure (False), got {result!r}"


@pytest.mark.asyncio
async def test_probe_400_unknown_code_still_counts_as_failure(monkeypatch):
    """A structured 400 with an unrecognised error.code is still a failure
    — we only whitelist the codes we know to be guard rejections."""
    app = _make_app()
    p = _make_plugin(app)
    _patch_httpx(monkeypatch, response=_MockResp(
        400, {"error": {"code": "rate_limited", "message": "slow down"}}
    ))

    result = await p._probe()
    assert result is False


@pytest.mark.asyncio
async def test_guard_400_does_not_advance_state_to_degraded(monkeypatch):
    """Three consecutive guard-rejected probes must NOT push the state
    machine into DEGRADED (which is what the bug looked like in prod).
    Instead they should land in UNKNOWN — same semantic as a connection
    timeout — leaving room for a real success to flip back to HEALTHY."""
    app = _make_app()
    p = _make_plugin(app)
    _patch_httpx(monkeypatch, response=_MockResp(
        400, {"error": {"code": "input_too_long"}}
    ))

    # Lower the unknown threshold so this test is cheap.
    p.unknowns_to_unknown_state = 3

    for _ in range(3):
        result = await p._probe()
        assert result is None
        # mimic run-loop accounting
        p.consecutive_unknowns += 1
        p._maybe_enter_unknown()

    assert p.state == AvailabilityState.UNKNOWN, (
        f"guard rejections leaked into the failure path: {p.state}"
    )
    assert p.consecutive_failures == 0, (
        f"guard rejections must NOT increment failures: {p.consecutive_failures}"
    )


@pytest.mark.asyncio
async def test_force_probe_with_guard_400_is_unknown(monkeypatch):
    """The dashboard's manual ping button (force_probe) must also benefit
    from the guard-aware classification."""
    app = _make_app()
    p = _make_plugin(app)
    _patch_httpx(monkeypatch, response=_MockResp(
        400, {"error": {"code": "input_too_long"}}
    ))

    result = await p.force_probe()
    assert result is None
    assert p.consecutive_failures == 0
    assert p.state == AvailabilityState.HEALTHY  # one unknown is below threshold
