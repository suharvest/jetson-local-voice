"""Dashboard /api/errors/clear and /api/agent/settings endpoints + ModeManager
late-registration broadcast."""
from __future__ import annotations

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

    async def on_user_utterance(self, ctx, text):
        return None


class _Other(AppMode):
    name = "other"
    display_name = "其他"
    icon = "🔧"
    description = "another mode"

    async def on_user_utterance(self, ctx, text):
        return None


def _mk_app(port: int, config: Config):
    app = MagicMock()
    app.config = config
    app.events = EventBus()
    app.audio = SimpleNamespace(_in_queue=None, stop_playback=AsyncMock())
    app.slv = SimpleNamespace(
        _ws=None, reconnect=AsyncMock(), abort=AsyncMock(), send_text=AsyncMock()
    )
    app.session = SimpleNamespace(history=[])
    app.broadcast = AsyncMock()
    mgr = ModeManager(lambda: None)
    chat = _Chat()
    mgr.register(chat)
    mgr._current = chat
    app.modes = mgr
    return app


@pytest.mark.asyncio
async def test_errors_clear_endpoint(unused_tcp_port):
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    # Seed some errors directly.
    plugin._errors.extend([{"ts": 1, "msg": "boom"}, {"ts": 2, "msg": "kaboom"}])
    assert len(plugin._errors) == 2
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(base + "/api/errors/clear")
            assert r.status == 200
            data = await r.json()
            assert data == {"ok": True}
        assert plugin._errors == []
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_agent_settings_get_returns_three_fields(unused_tcp_port):
    cfg = Config(
        metadata={"dashboard_port": unused_tcp_port},
        pipeline_mode="wake_word",
        sleep_timeout_s=42.5,
        stop_words=["停", "stop"],
    )
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.get(base + "/api/agent/settings")
            assert r.status == 200
            data = await r.json()
        assert data["pipeline_mode"] == "wake_word"
        assert data["sleep_timeout_s"] == 42.5
        assert data["stop_words"] == ["停", "stop"]
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_agent_settings_post_updates_config(unused_tcp_port):
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                base + "/api/agent/settings",
                json={
                    "pipeline_mode": "wake_word",
                    "sleep_timeout_s": 12.0,
                    "stop_words": ["a", "b ", "  "],
                },
            )
            assert r.status == 200
            data = await r.json()
        assert cfg.pipeline_mode == "wake_word"
        assert cfg.sleep_timeout_s == 12.0
        assert cfg.stop_words == ["a", "b"]  # blank dropped, trimmed
        # Persistence skipped because no _source_path.
        assert data["persisted"] is False
        # Broadcast fired.
        called = [c for c in app.broadcast.call_args_list
                  if c.args[0] == "on_agent_settings_change"]
        assert called, "broadcast should have fired on_agent_settings_change"
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_agent_settings_post_persists_to_yaml(tmp_path, unused_tcp_port):
    src = tmp_path / "config.yaml"
    src.write_text(
        "slv_url: ws://x/y\n"
        "system_prompt: GLOBAL\n"
        "pipeline_mode: always_on\n"
        "sleep_timeout_s: 30.0\n",
        encoding="utf-8",
    )
    cfg = load_config(src)
    cfg.metadata = {"dashboard_port": unused_tcp_port}
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                base + "/api/agent/settings",
                json={"sleep_timeout_s": 99.0, "stop_words": ["x", "y"]},
            )
            assert r.status == 200
            data = await r.json()
        assert data["persisted"] is True

        import yaml
        with src.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        assert raw["sleep_timeout_s"] == 99.0
        assert raw["stop_words"] == ["x", "y"]
        assert raw["system_prompt"] == "GLOBAL"
        assert raw["pipeline_mode"] == "always_on"
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_agent_settings_post_validates_pipeline_mode(unused_tcp_port):
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    await plugin.start()
    try:
        base = f"http://127.0.0.1:{unused_tcp_port}"
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                base + "/api/agent/settings",
                json={"pipeline_mode": "garbage"},
            )
            assert r.status == 400
    finally:
        await plugin.stop()


@pytest.mark.asyncio
async def test_mode_manager_register_silent_before_start():
    """Registering modes before start() must NOT broadcast on_mode_registered."""
    broadcasts: list = []

    async def _bcast(name, payload):
        broadcasts.append((name, payload))

    def _factory():
        return SimpleNamespace(broadcast=_bcast, mode_manager=None)

    mgr = ModeManager(_factory)
    mgr.register(_Chat())
    mgr.register(_Other())
    assert broadcasts == []
    assert mgr._started is False


@pytest.mark.asyncio
async def test_mode_manager_register_broadcasts_after_start():
    """Registering AFTER start() must broadcast on_mode_registered."""
    broadcasts: list = []

    async def _bcast(name, payload):
        broadcasts.append((name, payload))

    def _factory():
        return SimpleNamespace(broadcast=_bcast, mode_manager=None)

    mgr = ModeManager(_factory)
    mgr.register(_Chat())
    await mgr.start("chat")
    assert mgr._started is True
    # on_mode_change fired from switch().
    assert any(b[0] == "on_mode_change" for b in broadcasts)

    n_before = len(broadcasts)
    mgr.register(_Other())

    # Late-registration broadcast is fire-and-forget; yield to let it run.
    import asyncio
    await asyncio.sleep(0.01)
    new_b = broadcasts[n_before:]
    reg_events = [b for b in new_b if b[0] == "on_mode_registered"]
    assert len(reg_events) == 1
    assert reg_events[0][1]["name"] == "other"
    assert reg_events[0][1]["display_name"] == "其他"


@pytest.mark.asyncio
async def test_tts_metrics_emitted_on_audio_frame(unused_tcp_port):
    """on_tts_audio_frame should accumulate bytes; on_assistant_done snapshots."""
    cfg = Config(metadata={"dashboard_port": unused_tcp_port})
    app = _mk_app(unused_tcp_port, cfg)
    plugin = DebugDashboardPlugin(app)
    plugin.setup()
    # Don't start HTTP server — just drive hooks directly.
    await plugin.on_user_utterance("hi")
    await plugin.on_assistant_sentence("hello there.")
    await plugin.on_tts_audio_frame({"sample_rate": 24000, "frame_len": 48000})
    await plugin.on_tts_audio_frame({"sample_rate": 24000, "frame_len": 24000})
    assert plugin._tts_sentence_count == 1
    assert plugin._tts_bytes_current == 72000
    await plugin.on_assistant_done()
    assert plugin._tts_bytes_last == 72000
    assert plugin._tts_bytes_current == 0
