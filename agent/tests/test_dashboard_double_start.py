"""Issue #4 — DebugDashboardPlugin must be idempotent across start()/stop().

Without this guard, restarting the plugin without an intervening clean
``stop()`` (e.g. tests, supervisor relaunch) double-subscribes to the
EventBus and each event fans out twice — the chat history would show
every message twice, prefix-cache toggles fire twice, etc.

Also covers the analogous guard on LLMAvailabilityPlugin.start().
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from openvoicestream_agent.event_bus import EventBus
from openvoicestream_agent.plugins.debug_dashboard import DebugDashboardPlugin
from openvoicestream_agent.plugins.llm_availability import LLMAvailabilityPlugin


# ── DebugDashboardPlugin ───────────────────────────────────────────────


def _bare_dashboard_app(port: int = 0) -> SimpleNamespace:
    """Build the minimum app surface DebugDashboardPlugin needs to start()."""
    cfg = SimpleNamespace(metadata={"dashboard_port": port})
    return SimpleNamespace(config=cfg, events=EventBus())


@pytest.mark.asyncio
async def test_debug_dashboard_double_start_does_not_double_subscribe(caplog):
    """Two start() calls without stop() ⇒ a single set of EventBus subscribers."""
    pytest.importorskip("aiohttp")
    app = _bare_dashboard_app(port=0)
    p = DebugDashboardPlugin(app)
    p.setup()
    try:
        with caplog.at_level(logging.WARNING):
            await p.start()
            sub_count_after_first = len(
                app.events._subscribers.get("on_session_trimmed", [])
            )
            await p.start()  # second start — should be a no-op + warning.
            sub_count_after_second = len(
                app.events._subscribers.get("on_session_trimmed", [])
            )
        assert sub_count_after_first == 1, (
            f"first start should subscribe once: {sub_count_after_first}"
        )
        assert sub_count_after_second == 1, (
            "second start must NOT double-subscribe; "
            f"got {sub_count_after_second} subscribers"
        )
        # The warning log proves the guard fired (and gives operators a hint).
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("already" in r.message.lower() or
                   "double" in r.message.lower() or
                   "duplicate" in r.message.lower() or
                   "twice" in r.message.lower()
                   for r in warnings), (
            f"expected a warning log for double-start; got {warnings}"
        )
    finally:
        await p.stop()


@pytest.mark.asyncio
async def test_debug_dashboard_stop_then_start_resubscribes_cleanly():
    """Guard must NOT make the plugin unstartable after a real stop()."""
    pytest.importorskip("aiohttp")
    app = _bare_dashboard_app(port=0)
    p = DebugDashboardPlugin(app)
    p.setup()
    await p.start()
    assert len(app.events._subscribers.get("on_session_trimmed", [])) == 1
    await p.stop()
    assert len(app.events._subscribers.get("on_session_trimmed", [])) == 0
    # Re-start should work and re-subscribe.
    await p.start()
    try:
        assert len(app.events._subscribers.get("on_session_trimmed", [])) == 1
    finally:
        await p.stop()


# ── LLMAvailabilityPlugin ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_availability_double_start_keeps_single_task(caplog):
    """Calling start() twice must not leak a second probe task."""
    cfg = SimpleNamespace(
        llm_base_url="http://localhost:8000",
        llm_api_key="",
        llm_model="dummy",
        llm_availability_enabled=True,
        llm_availability_probe_interval_s=3600.0,  # never auto-probes during test
        llm_availability_probe_timeout_s=1.0,
        llm_availability_failures_to_down=3,
    )

    class _App:
        def __init__(self) -> None:
            self.config = cfg
            self.events = EventBus()
            self.llm_availability = None

        async def broadcast(self, *_a, **_kw):  # pragma: no cover - unused
            return None

    app = _App()
    p = LLMAvailabilityPlugin(app)

    with caplog.at_level(logging.WARNING):
        await p.start()
        first_task = p._task
        await p.start()  # double-start — must be a no-op.
        second_task = p._task

    try:
        assert first_task is second_task, (
            "double-start created a new probe task; "
            f"old={first_task!r} new={second_task!r}"
        )
        assert first_task is not None
        assert not first_task.done()
    finally:
        await p.stop()
