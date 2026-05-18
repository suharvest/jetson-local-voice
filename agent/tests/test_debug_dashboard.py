"""Tests for the DebugDashboardPlugin: a browser WS client should receive
broadcasted hook events as JSON."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import aiohttp
import pytest
from aiohttp import web

from openvoicestream_agent.event_bus import EventBus
from openvoicestream_agent.plugins.debug_dashboard import DebugDashboardPlugin


@pytest.mark.asyncio
async def test_browser_receives_hook_broadcast(unused_tcp_port):
    app_mock = MagicMock()
    app_mock.config = SimpleNamespace(metadata={"dashboard_port": unused_tcp_port})
    app_mock.events = EventBus()
    app_mock.audio = SimpleNamespace(_in_queue=None)
    app_mock.slv = SimpleNamespace(_ws=None)

    plugin = DebugDashboardPlugin(app_mock)
    assert plugin.setup() is True
    await plugin.start()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"http://127.0.0.1:{unused_tcp_port}/ws") as ws:
                # Wait until the server registers the client.
                for _ in range(50):
                    if plugin._browser_clients:
                        break
                    import asyncio
                    await asyncio.sleep(0.01)

                await plugin.on_user_utterance("你好")

                msg = await ws.receive(timeout=2.0)
                assert msg.type == aiohttp.WSMsgType.TEXT
                payload = json.loads(msg.data)
                assert payload["event"] == "on_user_utterance"
                assert payload["data"] == "你好"
                assert isinstance(payload["ts"], int)
    finally:
        await plugin.stop()
