"""Per-mode override editor API: GET/POST /api/modes/<name>/overrides."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from openvoicestream_agent.app_mode import AppMode, ModeManager
from openvoicestream_agent.config import Config, load_config
from openvoicestream_agent.event_bus import EventBus
from openvoicestream_agent.plugins.debug_dashboard import DebugDashboardPlugin


class _Chat(AppMode):
    name = "chat"
    display_name = "对话"
    icon = "💬"
    description = "test chat"
    system_prompt = "DEFAULT_CHAT_PROMPT"
    temperature = 0.7

    async def on_user_utterance(self, ctx, text):
        return None


def _mk_app(port: int, config: Config):
    app = MagicMock()
    app.config = config
    app.events = EventBus()
    app.audio = SimpleNamespace(_in_queue=None, stop_playback=AsyncMock())
    app.slv = SimpleNamespace(_ws=None, reconnect=AsyncMock(), abort=AsyncMock(), send_text=AsyncMock())
    app.session = SimpleNamespace(history=[])
    app.broadcast = AsyncMock()

    # Real ModeManager with one mode registered + active.
    ctx_holder = {"ctx": None}
    def _factory():
        return ctx_holder["ctx"]
    mgr = ModeManager(_factory)
    chat = _Chat()
    mgr.register(chat)
    mgr._current = chat  # skip async start for unit test
    app.modes = mgr
    return app


@pytest.mark.asyncio
async def test_get_overrides_returns_class_default(unused_tcp_port):
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.get(base + "/api/modes/chat/overrides")
            assert r.status == 200
            data = await r.json()
        assert data["name"] == "chat"
        assert data["display_name"] == "对话"
        assert data["class_default"]["system_prompt"] == "DEFAULT_CHAT_PROMPT"
        assert data["class_default"]["temperature"] == 0.7
        # No override yet → effective == class_default.
        assert data["current_override"] == {}
        assert data["effective"]["system_prompt"] == "DEFAULT_CHAT_PROMPT"
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_get_overrides_unknown_mode(unused_tcp_port):
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.get(base + "/api/modes/nope/overrides")
            assert r.status == 404
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_post_overrides_updates_config_and_clears(unused_tcp_port):
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            # Set a new system_prompt + temperature.
            r = await s.post(
                base + "/api/modes/chat/overrides",
                json={"system_prompt": "NEW", "temperature": 0.3},
            )
            assert r.status == 200
            data = await r.json()
        assert cfg.mode_overrides["chat"]["system_prompt"] == "NEW"
        assert cfg.mode_overrides["chat"]["temperature"] == 0.3
        assert data["effective"]["system_prompt"] == "NEW"
        # Persistence skipped because _source_path is None.
        assert data["persisted"] is False
        # Broadcast was called.
        app.broadcast.assert_any_call(
            "on_mode_override_change",
            {"name": "chat", "override": {"system_prompt": "NEW", "temperature": 0.3}, "persisted": False},
        )

        # Now clear system_prompt with null; temperature unchanged.
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                base + "/api/modes/chat/overrides",
                json={"system_prompt": None},
            )
            assert r.status == 200
            data = await r.json()
        assert "system_prompt" not in cfg.mode_overrides["chat"]
        assert cfg.mode_overrides["chat"]["temperature"] == 0.3
        assert data["effective"]["system_prompt"] == "DEFAULT_CHAT_PROMPT"

        # Clear remaining → entry removed entirely.
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                base + "/api/modes/chat/overrides",
                json={"temperature": None},
            )
            assert r.status == 200
        assert "chat" not in cfg.mode_overrides
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_post_overrides_persists_to_yaml(tmp_path, unused_tcp_port):
    src = tmp_path / "config.yaml"
    src.write_text(
        "slv_url: ws://x/y\n"
        "system_prompt: GLOBAL\n"
        "default_mode: chat\n",
        encoding="utf-8",
    )
    cfg = load_config(src)
    cfg.metadata = {"dashboard_port": unused_tcp_port}
    assert cfg._source_path == src

    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                base + "/api/modes/chat/overrides",
                json={"system_prompt": "PERSISTED"},
            )
            assert r.status == 200
            data = await r.json()
        assert data["persisted"] is True

        # Re-load and check that mode_overrides is in the yaml file.
        import yaml
        with src.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        assert raw["mode_overrides"]["chat"]["system_prompt"] == "PERSISTED"
        # Other fields preserved.
        assert raw["system_prompt"] == "GLOBAL"
        assert raw["default_mode"] == "chat"

        # Now clear it; yaml mode_overrides key should disappear.
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                base + "/api/modes/chat/overrides",
                json={"system_prompt": None},
            )
            assert r.status == 200
        with src.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        assert "mode_overrides" not in raw
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_post_overrides_empty_string_system_prompt_is_preserved(unused_tcp_port):
    """Regression: POST {system_prompt: ""} must store and
    return the empty string — not collapse it to "use the default"."""
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                base + "/api/modes/chat/overrides",
                json={"system_prompt": ""},
            )
            assert r.status == 200
            data = await r.json()
        # Storage layer: empty string preserved (not popped).
        assert cfg.mode_overrides["chat"]["system_prompt"] == ""
        # GET-back: current_override carries the empty string.
        assert data["current_override"].get("system_prompt") == ""
        # Effective: empty string wins over class default.
        assert data["effective"]["system_prompt"] == ""
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_post_overrides_bad_json(unused_tcp_port):
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                base + "/api/modes/chat/overrides",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert r.status == 400
    finally:
        await plugin.stop()
