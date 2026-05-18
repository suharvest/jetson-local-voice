"""Dashboard LLM health card integration (Task #18 A6).

Covers:
  - initial snapshot includes ``llm_availability`` + ``prefix_cache_disabled``
  - snapshot reflects current availability state (not a default)
  - late-connecting client sees DOWN if availability has been DOWN
  - ``on_llm_availability_change`` plugin hook relays to WS clients
  - ``on_session_trimmed`` / ``on_prefix_cache_disabled`` bus events relay
  - ``POST /api/llm/probe`` calls ``force_probe`` and returns current state
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from openvoicestream_agent.app_mode import AppMode, ModeManager
from openvoicestream_agent.config import Config
from openvoicestream_agent.event_bus import EventBus
from openvoicestream_agent.plugins.debug_dashboard import DebugDashboardPlugin
from openvoicestream_agent.plugins.llm_availability import AvailabilityState


class _Chat(AppMode):
    name = "chat"
    display_name = "对话"
    icon = "💬"
    description = "test"

    async def on_user_utterance(self, ctx, text):
        return None


def _mk_app(port: int, *, availability=None, session_prefix_disabled: bool = False):
    cfg = Config(metadata={"dashboard_port": port})
    app = MagicMock()
    app.config = cfg
    app.events = EventBus()
    # Force these to plain values so MagicMock's auto-attribute behaviour
    # doesn't produce un-JSON-serialisable MagicMock instances inside the
    # initial snapshot payload.
    app._state = None
    app._slv_reconnect_count = 0
    app.audio = SimpleNamespace(_in_queue=None, stop_playback=AsyncMock())
    app.slv = SimpleNamespace(
        _ws=None, reconnect=AsyncMock(), abort=AsyncMock(), send_text=AsyncMock()
    )
    app.session = SimpleNamespace(
        history=[], prefix_cache_disabled=session_prefix_disabled
    )
    app.broadcast = AsyncMock()
    mgr = ModeManager(lambda: None)
    chat = _Chat()
    mgr.register(chat)
    mgr._current = chat
    app.modes = mgr
    app.llm_availability = availability
    # plugins list (so app.broadcast emulation can iterate if needed)
    app.plugins = []
    return app


def _fake_availability(state: AvailabilityState, *, last_ok_ts=None,
                       consecutive_failures=0, interval_s=30.0):
    """Minimal stand-in for LLMAvailabilityPlugin (only fields the dashboard reads)."""
    avail = SimpleNamespace(
        state=state,
        last_ok_ts=last_ok_ts,
        consecutive_failures=consecutive_failures,
        interval_s=interval_s,
        force_probe=AsyncMock(return_value=True),
    )
    return avail


async def _recv_snapshot(ws):
    """Read messages until we get the snapshot (skipping any racing stats frames)."""
    deadline = asyncio.get_event_loop().time() + 2.0
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise AssertionError("never received snapshot")
        msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
        if msg.type != aiohttp.WSMsgType.TEXT:
            continue
        data = json.loads(msg.data)
        if data.get("event") == "snapshot":
            return data["data"]


# ── snapshot field presence ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_snapshot_includes_llm_availability(unused_tcp_port):
    avail = _fake_availability(AvailabilityState.HEALTHY, interval_s=15.0)
    app = _mk_app(unused_tcp_port, availability=avail)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        url = f"http://127.0.0.1:{unused_tcp_port}/ws"
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(url) as ws:
                snap = await _recv_snapshot(ws)
        assert "llm_availability" in snap
        assert "prefix_cache_disabled" in snap
        assert snap["llm_availability"]["state"] == "healthy"
        assert snap["llm_availability"]["probe_interval_s"] == 15.0
        assert snap["prefix_cache_disabled"] is False
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_snapshot_state_reflects_current(unused_tcp_port):
    avail = _fake_availability(
        AvailabilityState.DOWN, last_ok_ts=12345.0, consecutive_failures=5
    )
    app = _mk_app(unused_tcp_port, availability=avail, session_prefix_disabled=True)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        url = f"http://127.0.0.1:{unused_tcp_port}/ws"
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(url) as ws:
                snap = await _recv_snapshot(ws)
        assert snap["llm_availability"]["state"] == "down"
        assert snap["llm_availability"]["last_ok_ts"] == 12345.0
        assert snap["llm_availability"]["consecutive_failures"] == 5
        assert snap["prefix_cache_disabled"] is True
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_late_client_sees_current_state(unused_tcp_port):
    """A client that connects *after* availability went DOWN must see DOWN."""
    avail = _fake_availability(AvailabilityState.HEALTHY)
    app = _mk_app(unused_tcp_port, availability=avail)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        # Simulate availability transitioning to DOWN well after plugin start.
        await asyncio.sleep(0.05)
        avail.state = AvailabilityState.DOWN
        avail.consecutive_failures = 3

        url = f"http://127.0.0.1:{unused_tcp_port}/ws"
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(url) as ws:
                snap = await _recv_snapshot(ws)
        # Critical: the late-connecting browser must NOT see the default "unknown"
        # nor the stale "healthy" — it must see the live state machine value.
        assert snap["llm_availability"]["state"] == "down"
        assert snap["llm_availability"]["consecutive_failures"] == 3
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_snapshot_without_availability_plugin(unused_tcp_port):
    """No availability plugin attached → snapshot still has the fields with defaults."""
    app = _mk_app(unused_tcp_port, availability=None)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        url = f"http://127.0.0.1:{unused_tcp_port}/ws"
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(url) as ws:
                snap = await _recv_snapshot(ws)
        assert snap["llm_availability"]["state"] == "unknown"
        assert snap["llm_availability"]["last_ok_ts"] is None
        assert snap["llm_availability"]["consecutive_failures"] == 0
        assert snap["prefix_cache_disabled"] is False
    finally:
        await plugin.stop()


# ── relay broadcasts ────────────────────────────────────────────────


async def _drain_until(ws, event_name, *, timeout=2.0):
    """Read WS frames until we see ``event_name`` or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError(f"never saw event {event_name!r}")
        msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
        if msg.type != aiohttp.WSMsgType.TEXT:
            continue
        data = json.loads(msg.data)
        if data.get("event") == event_name:
            return data


@pytest.mark.asyncio
async def test_relay_availability_change_event(unused_tcp_port):
    avail = _fake_availability(AvailabilityState.HEALTHY)
    app = _mk_app(unused_tcp_port, availability=avail)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        url = f"http://127.0.0.1:{unused_tcp_port}/ws"
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(url) as ws:
                # Skip initial snapshot.
                await _recv_snapshot(ws)
                # Drive the plugin hook directly (simulating BaseApp.broadcast).
                payload = {
                    "state": "degraded",
                    "last_ok_ts": 9999.0,
                    "consecutive_failures": 1,
                }
                await plugin.on_llm_availability_change(payload)
                msg = await _drain_until(ws, "on_llm_availability_change")
        assert msg["data"]["state"] == "degraded"
        assert msg["data"]["consecutive_failures"] == 1
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_relay_prefix_cache_disabled_event(unused_tcp_port):
    app = _mk_app(unused_tcp_port)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        url = f"http://127.0.0.1:{unused_tcp_port}/ws"
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(url) as ws:
                await _recv_snapshot(ws)
                # Fire the bus event the way edge_llm does it.
                app.events.emit("on_prefix_cache_disabled", {"reason": "test"})
                msg = await _drain_until(ws, "on_prefix_cache_disabled")
        assert msg["data"]["reason"] == "test"
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_relay_session_trimmed_event(unused_tcp_port):
    app = _mk_app(unused_tcp_port)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        url = f"http://127.0.0.1:{unused_tcp_port}/ws"
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(url) as ws:
                await _recv_snapshot(ws)
                app.events.emit(
                    "on_session_trimmed",
                    {"dropped_turns": 2, "kept_turns": 5, "approx_tokens": 1024},
                )
                msg = await _drain_until(ws, "on_session_trimmed")
        assert msg["data"]["dropped_turns"] == 2
        assert msg["data"]["kept_turns"] == 5
    finally:
        await plugin.stop()


# ── /api/llm/probe endpoint ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_api_llm_probe_calls_force_probe(unused_tcp_port):
    avail = _fake_availability(
        AvailabilityState.RECOVERING, last_ok_ts=42.0, consecutive_failures=1
    )
    app = _mk_app(unused_tcp_port, availability=avail)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(base + "/api/llm/probe")
            assert r.status == 200
            data = await r.json()
        assert data["state"] == "recovering"
        assert data["last_ok_ts"] == 42.0
        assert data["consecutive_failures"] == 1
        avail.force_probe.assert_awaited_once()
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_api_llm_probe_without_plugin_returns_503(unused_tcp_port):
    app = _mk_app(unused_tcp_port, availability=None)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(base + "/api/llm/probe")
            assert r.status == 503
    finally:
        await plugin.stop()
