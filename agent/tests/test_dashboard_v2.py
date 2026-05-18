"""Tests for DebugDashboardPlugin v2: control endpoints + state broadcast."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from openvoicestream_agent.event_bus import EventBus
from openvoicestream_agent.plugins.debug_dashboard import DebugDashboardPlugin


def _mk_app(port):
    app_mock = MagicMock()
    app_mock.config = SimpleNamespace(
        metadata={"dashboard_port": port},
        system_prompt="You are a helpful assistant.",
    )
    app_mock.events = EventBus()
    app_mock.audio = SimpleNamespace(
        _in_queue=None,
        stop_playback=AsyncMock(),
    )
    app_mock.slv = SimpleNamespace(
        _ws=None,
        reconnect=AsyncMock(),
        abort=AsyncMock(),
        send_text=AsyncMock(),
    )
    app_mock.session = SimpleNamespace(
        history=[
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好呀"},
        ]
    )
    return app_mock


@pytest.mark.asyncio
async def test_control_endpoints(unused_tcp_port):
    app_mock = _mk_app(unused_tcp_port)
    plugin = DebugDashboardPlugin(app_mock)
    assert plugin.setup() is True
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(base + "/api/control/reconnect")
            assert r.status == 200
            assert (await r.json())["ok"] is True
            app_mock.slv.reconnect.assert_awaited()

            r = await s.post(base + "/api/control/abort")
            assert r.status == 200
            app_mock.slv.abort.assert_awaited()
            app_mock.audio.stop_playback.assert_awaited()

            r = await s.post(
                base + "/api/control/send_text",
                json={"text": "hi"},
            )
            assert r.status == 200
            app_mock.slv.send_text.assert_awaited_with("hi")

            # missing text → 400
            r = await s.post(base + "/api/control/send_text", json={})
            assert r.status == 400
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_session_history_endpoint(unused_tcp_port):
    app_mock = _mk_app(unused_tcp_port)
    plugin = DebugDashboardPlugin(app_mock)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.get(base + "/api/session/history")
            assert r.status == 200
            items = await r.json()
        assert items[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert items[1] == {"role": "user", "content": "你好"}
        assert items[2] == {"role": "assistant", "content": "你好呀"}
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_state_change_reaches_browser(unused_tcp_port):
    app_mock = _mk_app(unused_tcp_port)
    plugin = DebugDashboardPlugin(app_mock)
    plugin.setup()
    await plugin.start()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                f"http://127.0.0.1:{unused_tcp_port}/ws"
            ) as ws:
                # Wait for server-side registration.
                for _ in range(50):
                    if plugin._browser_clients:
                        break
                    await asyncio.sleep(0.01)

                await plugin.on_state_change({"state": "listening", "prev": "idle"})

                # Drain until we see the state_change event (skip stats noise).
                got = None
                for _ in range(20):
                    msg = await ws.receive(timeout=2.0)
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    payload = json.loads(msg.data)
                    if payload["event"] == "on_state_change":
                        got = payload
                        break
                assert got is not None
                assert got["data"]["state"] == "listening"
                assert got["data"]["prev"] == "idle"
    finally:
        await plugin.stop()
