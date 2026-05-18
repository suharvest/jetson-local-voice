"""Tests for the combined health-probe + circuit-breaker state machine.

Covers:
  • Initial state
  • Single-fail → DEGRADED, recovery → HEALTHY
  • N consecutive fails → DOWN
  • DOWN → RECOVERING → HEALTHY requires two successes
  • Probe timeout is "unknown" (does not advance state)
  • State change emits event; no-change does not
  • report_request_failure / report_request_success
  • force_probe bypasses interval
  • Probe uses POST /v1/chat/completions, NOT GET /v1/models
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from openvoicestream_agent.plugins.llm_availability import (
    AvailabilityState,
    LLMAvailabilityPlugin,
    LLMUnavailable,
)


# ── helpers ────────────────────────────────────────────────────────────


def _make_app(events_sink: list[tuple[str, dict]] | None = None) -> Any:
    """Build a minimal app stub with .config, .events, .broadcast."""
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

    bus = _EventBus()

    class _App:
        def __init__(self) -> None:
            self.config = cfg
            self.events = bus
            self.llm_availability = None

        async def broadcast(self, hook: str, data: Any) -> None:
            if events_sink is not None:
                events_sink.append((hook, data))

    return _App()


def _make_plugin(app, **overrides) -> LLMAvailabilityPlugin:
    p = LLMAvailabilityPlugin(app, **overrides)
    # Manually init the start()-time bits without spinning the run task.
    p._wake_evt = asyncio.Event()
    p._probe_lock = asyncio.Lock()
    return p


class _MockResp:
    def __init__(self, status: int = 200, body: dict | None = None):
        self.status_code = status
        self._body = body if body is not None else {"choices": [{"delta": {}}]}

    def json(self) -> dict:
        return self._body


class _FakeClient:
    """Drop-in for ``httpx.AsyncClient`` recording calls."""

    last_calls: list[dict] = []

    def __init__(self, *, response: Any = None, raise_exc: Exception | None = None,
                 **_kwargs) -> None:
        self._response = response if response is not None else _MockResp()
        self._raise = raise_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url: str, *, json=None, headers=None):
        _FakeClient.last_calls.append({
            "url": url, "json": json, "headers": headers, "method": "POST",
        })
        if self._raise is not None:
            raise self._raise
        return self._response


@pytest.fixture(autouse=True)
def _reset_fake_calls():
    _FakeClient.last_calls = []
    yield
    _FakeClient.last_calls = []


def _patch_httpx(monkeypatch, *, response=None, raise_exc=None):
    def factory(*args, **kwargs):
        return _FakeClient(response=response, raise_exc=raise_exc, **kwargs)
    monkeypatch.setattr(
        "openvoicestream_agent.plugins.llm_availability.httpx.AsyncClient",
        factory,
    )


# ── tests ──────────────────────────────────────────────────────────────


def test_initial_state_healthy():
    p = _make_plugin(_make_app())
    assert p.state == AvailabilityState.HEALTHY
    assert p.consecutive_failures == 0
    assert p.last_ok_ts is None


@pytest.mark.asyncio
async def test_probe_success_keeps_healthy(monkeypatch):
    _patch_httpx(monkeypatch, response=_MockResp(200, {"choices": [{}]}))
    p = _make_plugin(_make_app())
    res = await p._probe()
    assert res is True
    p._advance(res)
    assert p.state == AvailabilityState.HEALTHY
    assert p.last_ok_ts is not None


@pytest.mark.asyncio
async def test_single_failure_to_degraded(monkeypatch):
    _patch_httpx(monkeypatch, response=_MockResp(500, {}))
    p = _make_plugin(_make_app())
    p._advance(await p._probe())
    assert p.state == AvailabilityState.DEGRADED
    assert p.consecutive_failures == 1


@pytest.mark.asyncio
async def test_consecutive_failures_to_down(monkeypatch):
    _patch_httpx(monkeypatch, response=_MockResp(500, {}))
    p = _make_plugin(_make_app())
    for _ in range(3):
        p._advance(await p._probe())
    assert p.state == AvailabilityState.DOWN
    assert p.consecutive_failures == 3


@pytest.mark.asyncio
async def test_recovery_requires_two_consecutive_successes(monkeypatch):
    p = _make_plugin(_make_app())
    # Drive to DOWN with 3 fails.
    _patch_httpx(monkeypatch, response=_MockResp(500, {}))
    for _ in range(3):
        p._advance(await p._probe())
    assert p.state == AvailabilityState.DOWN

    # First success → RECOVERING (not yet HEALTHY).
    _patch_httpx(monkeypatch, response=_MockResp(200, {"choices": [{}]}))
    p._advance(await p._probe())
    assert p.state == AvailabilityState.RECOVERING

    # Second success → HEALTHY.
    p._advance(await p._probe())
    assert p.state == AvailabilityState.HEALTHY
    assert p.consecutive_failures == 0


@pytest.mark.asyncio
async def test_probe_timeout_does_not_count(monkeypatch):
    _patch_httpx(monkeypatch, raise_exc=httpx.ConnectTimeout("slow"))
    p = _make_plugin(_make_app())
    res = await p._probe()
    assert res is None  # unknown
    # Per run() loop contract: None results are skipped.
    # Confirm: calling _advance only when not-None means state is unchanged.
    if res is not None:
        p._advance(res)
    assert p.state == AvailabilityState.HEALTHY
    assert p.consecutive_failures == 0


@pytest.mark.asyncio
async def test_state_change_emits_event(monkeypatch):
    sink: list[tuple[str, dict]] = []
    app = _make_app(events_sink=sink)
    _patch_httpx(monkeypatch, response=_MockResp(500, {}))
    p = _make_plugin(app)
    p._advance(await p._probe())  # HEALTHY → DEGRADED
    # The broadcast goes via asyncio.create_task; let it run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    # Event-bus emission is synchronous; check it.
    assert any(name == "on_llm_availability_change" for name, _ in app.events.emitted)
    payload = next(d for n, d in app.events.emitted if n == "on_llm_availability_change")
    assert payload["state"] == "degraded"
    assert payload["consecutive_failures"] == 1
    # The broadcast hook should have fired too.
    assert any(h == "on_llm_availability_change" for h, _ in sink)


@pytest.mark.asyncio
async def test_no_event_when_state_unchanged(monkeypatch):
    app = _make_app()
    _patch_httpx(monkeypatch, response=_MockResp(200, {"choices": [{}]}))
    p = _make_plugin(app)
    # Two successes from HEALTHY → still HEALTHY → no events.
    p._advance(await p._probe())
    p._advance(await p._probe())
    names = [n for n, _ in app.events.emitted]
    assert "on_llm_availability_change" not in names


@pytest.mark.asyncio
async def test_report_request_failure_advances_state():
    p = _make_plugin(_make_app())
    p.report_request_failure()
    assert p.state == AvailabilityState.DEGRADED
    p.report_request_failure()
    p.report_request_failure()
    assert p.state == AvailabilityState.DOWN


@pytest.mark.asyncio
async def test_report_request_success_resets():
    p = _make_plugin(_make_app())
    p.report_request_failure()
    assert p.state == AvailabilityState.DEGRADED
    p.report_request_success()
    assert p.state == AvailabilityState.HEALTHY
    assert p.consecutive_failures == 0


@pytest.mark.asyncio
async def test_force_probe_bypasses_interval(monkeypatch):
    _patch_httpx(monkeypatch, response=_MockResp(500, {}))
    p = _make_plugin(_make_app(), interval_s=60.0)  # long interval
    # force_probe runs *now*, not in 60s.
    t0 = asyncio.get_event_loop().time()
    res = await p.force_probe()
    elapsed = asyncio.get_event_loop().time() - t0
    assert res is False
    assert elapsed < 0.5  # certainly did not wait the interval
    assert p.state == AvailabilityState.DEGRADED
    # And the wake event is set so a sleeping run() loop wakes up.
    assert p._wake_evt is not None and p._wake_evt.is_set()


@pytest.mark.asyncio
async def test_probe_uses_real_inference_not_models(monkeypatch):
    _patch_httpx(monkeypatch, response=_MockResp(200, {"choices": [{}]}))
    p = _make_plugin(_make_app())
    await p._probe()
    assert len(_FakeClient.last_calls) == 1
    call = _FakeClient.last_calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/v1/chat/completions")
    assert "/models" not in call["url"]
    # And the payload requests minimal generation (max_tokens=1, stream=False).
    assert call["json"]["max_tokens"] == 1
    assert call["json"]["stream"] is False
    assert call["json"]["messages"] == [{"role": "user", "content": "."}]


# ── MED-3: UNKNOWN state for network partitions ─────────────────────


@pytest.mark.asyncio
async def test_three_consecutive_timeouts_transition_to_unknown(monkeypatch):
    """MED-3: probe returning None (timeout / connect error) N times in
    a row must surface UNKNOWN — not silently stay HEALTHY."""
    _patch_httpx(monkeypatch, raise_exc=httpx.ConnectTimeout("net down"))
    p = _make_plugin(_make_app())
    assert p.state == AvailabilityState.HEALTHY

    # Simulate the main loop logic for three unknown results.
    for _ in range(3):
        result = await p._probe()
        assert result is None
        p.consecutive_unknowns += 1
        p._maybe_enter_unknown()

    assert p.state == AvailabilityState.UNKNOWN, (
        f"after 3 unknowns expected UNKNOWN, got {p.state}"
    )
    assert p.consecutive_unknowns == 3


@pytest.mark.asyncio
async def test_unknowns_below_threshold_no_state_change(monkeypatch):
    """MED-3: 2 unknowns (threshold=3) must NOT trip UNKNOWN — would be
    too jittery for a real production deploy."""
    _patch_httpx(monkeypatch, raise_exc=httpx.ConnectTimeout("blip"))
    p = _make_plugin(_make_app())
    for _ in range(2):
        result = await p._probe()
        assert result is None
        p.consecutive_unknowns += 1
        p._maybe_enter_unknown()
    assert p.state == AvailabilityState.HEALTHY
    assert p.consecutive_unknowns == 2


@pytest.mark.asyncio
async def test_success_resets_unknown_counter(monkeypatch):
    """MED-3: UNKNOWN + concrete success → straight back to HEALTHY,
    counter reset."""
    p = _make_plugin(_make_app())
    # Build up to UNKNOWN.
    _patch_httpx(monkeypatch, raise_exc=httpx.ConnectTimeout("net down"))
    for _ in range(3):
        result = await p._probe()
        p.consecutive_unknowns += 1
        p._maybe_enter_unknown()
    assert p.state == AvailabilityState.UNKNOWN

    # Now the network comes back.
    _patch_httpx(monkeypatch, response=_MockResp(200, {"choices": [{}]}))
    result = await p._probe()
    assert result is True
    p.consecutive_unknowns = 0
    p._advance(result)
    assert p.state == AvailabilityState.HEALTHY
    assert p.consecutive_unknowns == 0


@pytest.mark.asyncio
async def test_failure_after_unknowns_advances_to_degraded(monkeypatch):
    """MED-3: UNKNOWN + concrete probe failure → DEGRADED (LLM really is
    broken, not just unreachable). Starts the normal DOWN progression."""
    p = _make_plugin(_make_app())
    _patch_httpx(monkeypatch, raise_exc=httpx.ConnectTimeout("net down"))
    for _ in range(3):
        await p._probe()
        p.consecutive_unknowns += 1
        p._maybe_enter_unknown()
    assert p.state == AvailabilityState.UNKNOWN

    _patch_httpx(monkeypatch, response=_MockResp(500, {}))
    result = await p._probe()
    assert result is False
    p.consecutive_unknowns = 0
    p._advance(result)
    assert p.state == AvailabilityState.DEGRADED


@pytest.mark.asyncio
async def test_unknown_does_not_clobber_down(monkeypatch):
    """MED-3: if we're already in DOWN (a stronger signal), unknown
    probes must NOT downgrade us back to UNKNOWN."""
    p = _make_plugin(_make_app())
    # Drive to DOWN with three concrete failures.
    _patch_httpx(monkeypatch, response=_MockResp(500, {}))
    for _ in range(3):
        p._advance(await p._probe())
    assert p.state == AvailabilityState.DOWN

    # Now network goes wonky.
    _patch_httpx(monkeypatch, raise_exc=httpx.ConnectTimeout("net down"))
    for _ in range(5):
        await p._probe()
        p.consecutive_unknowns += 1
        p._maybe_enter_unknown()
    assert p.state == AvailabilityState.DOWN, (
        "UNKNOWN must not override DOWN — DOWN is the stronger diagnosis"
    )


@pytest.mark.asyncio
async def test_unknown_emits_state_change_event(monkeypatch):
    """MED-3: transition into UNKNOWN must fire on_llm_availability_change
    so the dashboard updates immediately."""
    sink: list[tuple[str, dict]] = []
    app = _make_app(events_sink=sink)
    _patch_httpx(monkeypatch, raise_exc=httpx.ConnectTimeout("net down"))
    p = _make_plugin(app)

    for _ in range(3):
        await p._probe()
        p.consecutive_unknowns += 1
        p._maybe_enter_unknown()

    # Let the broadcast task run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    payloads = [d for n, d in app.events.emitted if n == "on_llm_availability_change"]
    assert payloads, "expected at least one on_llm_availability_change emission"
    # The latest emitted state should be "unknown".
    assert payloads[-1]["state"] == "unknown"


@pytest.mark.asyncio
async def test_run_loop_increments_unknown_counter(monkeypatch):
    """MED-3: integration — main run() loop must do the consecutive
    unknown bookkeeping (not the test driving it manually)."""
    _patch_httpx(monkeypatch, raise_exc=httpx.ConnectTimeout("net down"))
    p = _make_plugin(_make_app(), interval_s=0.005)
    # Run a few iterations of the loop.
    task = asyncio.create_task(p.run())
    # Give the loop time to do >3 probes.
    await asyncio.sleep(0.2)
    p._stopped = True
    if p._wake_evt is not None:
        p._wake_evt.set()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()
    assert p.state == AvailabilityState.UNKNOWN
    assert p.consecutive_unknowns >= 3


@pytest.mark.asyncio
async def test_probe_200_with_no_choices_is_failure(monkeypatch):
    # /v1/models would return {} or {data: [...]} with no choices — probe
    # must NOT treat that as healthy.
    _patch_httpx(monkeypatch, response=_MockResp(200, {"data": ["foo"]}))
    p = _make_plugin(_make_app())
    res = await p._probe()
    assert res is False
