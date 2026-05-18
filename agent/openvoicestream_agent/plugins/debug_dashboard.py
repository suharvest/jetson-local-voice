"""Debug dashboard plugin v2: live agent observability + manual controls.

Browse to http://localhost:<dashboard_port>/ during local dev to see:
  - ConvState pill, latency cards, mic RMS chart, TTS indicator, errors
  - Filterable event log
  - Chat / Events / History tabs
  - Manual control buttons (reconnect / abort / send text)

Strictly a debugging aid. No auth, no TLS, no CORS handling.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

from ..plugin import Plugin
from ..state import ConvState

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).with_name("static")
_INDEX_HTML = _STATIC_DIR / "dashboard.html"


class DebugDashboardPlugin(Plugin):
    name = "debug_dashboard"

    def __init__(self, app) -> None:  # noqa: ANN001
        super().__init__(app)
        self._port: int = 18000
        self._runner = None  # aiohttp AppRunner
        self._site = None
        self._browser_clients: set = set()
        self._stats_task: asyncio.Task | None = None

        # ── per-turn timing scratch ──
        self._turn_id: int = 0
        self._last_speech_end_ts_ms: int | None = None
        self._last_utterance_ts_ms: int | None = None
        self._first_token_ts_ms: int | None = None
        self._first_tts_audio_ts_ms: int | None = None
        # ── rolling latency history (most-recent 3) ──
        self._latency_history: dict[str, deque] = {
            "asr": deque(maxlen=3),
            "ttft": deque(maxlen=3),
            "ttfa": deque(maxlen=3),
            "rtt": deque(maxlen=3),
        }
        # ── persistent error log (browser-side filters can hide them) ──
        self._errors: list[dict[str, Any]] = []
        # ── session start (used by browser via stats; here for completeness) ──
        self._session_started_ms: int = int(time.time() * 1000)
        # ── rolling cache-hit percentages (most-recent 20) ──
        self._cache_pct_history: deque = deque(maxlen=20)
        # ── per-turn TTS metrics ──
        self._tts_sentence_count: int = 0
        self._tts_bytes_current: int = 0
        self._tts_bytes_last: int = 0
        self._tts_last_sample_rate: int = 24000

    # ── lifecycle ─────────────────────────────────────────────────

    def setup(self) -> bool:
        try:
            meta = getattr(self.app.config, "metadata", {}) or {}
            self._port = int(meta.get("dashboard_port", 18000))
        except Exception:
            self._port = 18000
        return True

    async def start(self) -> None:
        try:
            from aiohttp import web
        except ImportError:
            logger.error("aiohttp not installed -- debug_dashboard disabled")
            return

        web_app = web.Application()
        web_app.router.add_get("/", self._handle_index)
        web_app.router.add_get("/ws", self._handle_ws)

        # Control endpoints (POST).
        web_app.router.add_post("/api/control/reconnect", self._api_reconnect)
        web_app.router.add_post("/api/control/abort", self._api_abort)
        web_app.router.add_post("/api/control/send_text", self._api_send_text)
        web_app.router.add_get("/api/session/history", self._api_session_history)
        # AppMode framework endpoints.
        web_app.router.add_get("/api/modes", self._api_modes_list)
        web_app.router.add_post("/api/control/mode", self._api_mode_switch)
        # Per-mode override editor (system_prompt etc.).
        web_app.router.add_get(
            "/api/modes/{name}/overrides", self._api_mode_overrides_get
        )
        web_app.router.add_post(
            "/api/modes/{name}/overrides", self._api_mode_overrides_post
        )
        # pipeline_mode endpoints (wake_word / push_to_talk).
        web_app.router.add_post("/api/control/wake", self._api_wake)
        web_app.router.add_post("/api/control/sleep", self._api_sleep)
        web_app.router.add_post("/api/control/ptt/start", self._api_ptt_start)
        web_app.router.add_post("/api/control/ptt/end", self._api_ptt_end)
        # Errors management + agent settings.
        web_app.router.add_post("/api/errors/clear", self._api_errors_clear)
        web_app.router.add_get("/api/agent/settings", self._api_agent_settings_get)
        web_app.router.add_post("/api/agent/settings", self._api_agent_settings_post)
        web_app.router.add_post("/api/llm/probe", self._api_llm_probe)

        # Static assets (css/js).
        if _STATIC_DIR.exists():
            web_app.router.add_static("/static/", str(_STATIC_DIR), show_index=False)

        self._runner = web.AppRunner(web_app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "0.0.0.0", self._port)
        await self._site.start()
        logger.info("debug_dashboard listening on http://0.0.0.0:%d", self._port)

        self._stats_task = asyncio.create_task(self._stats_loop(), name="dashboard-stats")

        # Subscribe to EventBus signals that don't have a plugin-hook bridge
        # (session.trim_history / edge_llm prefix_cache disable). Relay them
        # to browser clients so the LLM-health card can reflect them live.
        bus = getattr(self.app, "events", None)
        if bus is not None:
            try:
                bus.subscribe("on_session_trimmed", self._on_bus_session_trimmed)
                bus.subscribe(
                    "on_prefix_cache_disabled", self._on_bus_prefix_cache_disabled
                )
            except Exception:  # pragma: no cover - defensive
                logger.debug("event bus subscribe failed", exc_info=True)

        await super().start()

    async def stop(self) -> None:
        bus = getattr(self.app, "events", None)
        if bus is not None:
            try:
                bus.unsubscribe("on_session_trimmed", self._on_bus_session_trimmed)
                bus.unsubscribe(
                    "on_prefix_cache_disabled", self._on_bus_prefix_cache_disabled
                )
            except Exception:  # pragma: no cover - defensive
                pass

        if self._stats_task is not None:
            self._stats_task.cancel()
            try:
                await self._stats_task
            except (asyncio.CancelledError, Exception):
                pass
            self._stats_task = None

        for ws in list(self._browser_clients):
            try:
                await ws.close()
            except Exception:
                pass
        self._browser_clients.clear()

        if self._site is not None:
            try:
                await self._site.stop()
            except Exception:
                pass
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
            self._runner = None

        await super().stop()

    # ── HTTP / WS handlers ───────────────────────────────────────

    async def _handle_index(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            text = _INDEX_HTML.read_text(encoding="utf-8")
        except FileNotFoundError:
            return web.Response(status=500, text="dashboard.html not found")
        return web.Response(text=text, content_type="text/html")

    async def _handle_ws(self, request):  # noqa: ANN001
        from aiohttp import web
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._browser_clients.add(ws)
        # Send initial snapshot so late-connecting browsers don't see
        # stale "–" placeholders until the next event happens to fire.
        try:
            state_val = "idle"
            try:
                s = getattr(self.app, "_state", None)
                if s is not None:
                    state_val = getattr(s, "value", str(s))
            except Exception:
                state_val = "idle"
            mode_name = None
            mode_list: list[dict] = []
            try:
                mgr = getattr(self.app, "modes", None)
                if mgr is not None:
                    mode_name = mgr.current_name
                    mode_list = mgr.list_all()
            except Exception:
                mode_name = None
            pipeline_mode = "always_on"
            sleep_timeout_s = 30.0
            try:
                cfg = getattr(self.app, "config", None)
                if cfg is not None:
                    pipeline_mode = getattr(cfg, "pipeline_mode", "always_on")
                    sleep_timeout_s = float(getattr(cfg, "sleep_timeout_s", 30.0))
            except Exception:
                pass
            snapshot = {
                "event": "snapshot",
                "ts": int(time.time() * 1000),
                "data": {
                    "state": state_val,
                    "reconnect_count": getattr(self.app, "_slv_reconnect_count", 0),
                    "session_started_ms": self._session_started_ms,
                    "errors": list(self._errors),
                    "mode": mode_name,
                    "modes": mode_list,
                    "pipeline_mode": pipeline_mode,
                    "sleep_timeout_s": sleep_timeout_s,
                    "llm_availability": self._build_llm_availability_payload(),
                    "prefix_cache_disabled": self._read_prefix_cache_disabled(),
                },
            }
            await ws.send_str(json.dumps(snapshot, ensure_ascii=False))
        except Exception:
            logger.debug("snapshot send failed", exc_info=True)
        try:
            async for _msg in ws:
                pass  # ignore inbound
        finally:
            self._browser_clients.discard(ws)
        return ws

    # ── /api/control/* ──────────────────────────────────────────

    async def _api_reconnect(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            try:
                await self.app.audio.stop_playback()
            except Exception:
                pass
            try:
                self.app._first_tts_seen = False
            except Exception:
                pass
            try:
                self.app._set_state(ConvState.IDLE)
            except Exception:
                pass
            await self.app.slv.reconnect()
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    async def _api_abort(self, request):  # noqa: ANN001
        from aiohttp import web
        errs: list[str] = []
        try:
            await self.app.slv.abort()
        except Exception as e:
            errs.append(f"abort: {e}")
        try:
            await self.app.audio.stop_playback()
        except Exception as e:
            errs.append(f"stop_playback: {e}")
        if errs:
            return web.json_response({"ok": False, "errors": errs}, status=500)
        return web.json_response({"ok": True})

    async def _api_send_text(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "bad json"}, status=400)
        text = (body or {}).get("text")
        if not isinstance(text, str) or not text.strip():
            return web.json_response({"ok": False, "error": "missing text"}, status=400)
        try:
            await self.app.slv.send_text(text)
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    async def _api_modes_list(self, request):  # noqa: ANN001
        from aiohttp import web
        mgr = getattr(self.app, "modes", None)
        if mgr is None:
            return web.json_response([])
        try:
            return web.json_response(mgr.list_all())
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    async def _api_mode_switch(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "bad json"}, status=400)
        name = (body or {}).get("name")
        if not isinstance(name, str) or not name.strip():
            return web.json_response({"ok": False, "error": "missing name"}, status=400)
        mgr = getattr(self.app, "modes", None)
        if mgr is None:
            return web.json_response(
                {"ok": False, "error": "app has no ModeManager"}, status=400
            )
        try:
            await mgr.switch(name)
            return web.json_response({"ok": True, "current": mgr.current_name})
        except KeyError:
            return web.json_response(
                {"ok": False, "error": f"unknown mode: {name}"}, status=404
            )
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    # ── per-mode override editor ────────────────────────────────
    _OVERRIDE_KEYS = ("system_prompt", "temperature", "max_history", "barge_in_enabled")
    _AGENT_SETTINGS_KEYS = ("pipeline_mode", "sleep_timeout_s", "stop_words")
    _PIPELINE_MODES = ("always_on", "wake_word", "push_to_talk")

    def _build_override_payload(self, mode) -> dict[str, Any]:
        class_default = {k: getattr(mode, k, None) for k in self._OVERRIDE_KEYS}
        overrides = getattr(self.app.config, "mode_overrides", None) or {}
        current = {}
        if isinstance(overrides, dict):
            mo = overrides.get(mode.name) or {}
            if isinstance(mo, dict):
                current = {k: mo[k] for k in mo if k in self._OVERRIDE_KEYS}
        effective = dict(class_default)
        for k, v in current.items():
            effective[k] = v
        return {
            "name": mode.name,
            "display_name": mode.display_name,
            "icon": mode.icon,
            "effective": effective,
            "class_default": class_default,
            "current_override": current,
        }

    async def _api_mode_overrides_get(self, request):  # noqa: ANN001
        from aiohttp import web
        name = request.match_info.get("name", "")
        mgr = getattr(self.app, "modes", None)
        if mgr is None:
            return web.json_response(
                {"ok": False, "error": "app has no ModeManager"}, status=400
            )
        mode = mgr.get(name)
        if mode is None:
            return web.json_response(
                {"ok": False, "error": f"unknown mode: {name}"}, status=404
            )
        return web.json_response(self._build_override_payload(mode))

    async def _api_mode_overrides_post(self, request):  # noqa: ANN001
        from aiohttp import web
        name = request.match_info.get("name", "")
        mgr = getattr(self.app, "modes", None)
        if mgr is None:
            return web.json_response(
                {"ok": False, "error": "app has no ModeManager"}, status=400
            )
        mode = mgr.get(name)
        if mode is None:
            return web.json_response(
                {"ok": False, "error": f"unknown mode: {name}"}, status=404
            )
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "bad json"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "body must be object"}, status=400)
        # Ensure mode_overrides dict exists and is mutable.
        cfg = self.app.config
        overrides = getattr(cfg, "mode_overrides", None)
        if not isinstance(overrides, dict):
            overrides = {}
            try:
                cfg.mode_overrides = overrides
            except Exception:
                pass
        mo = overrides.get(name)
        if not isinstance(mo, dict):
            mo = {}
        for k in self._OVERRIDE_KEYS:
            if k not in body:
                continue
            v = body[k]
            if v is None:
                mo.pop(k, None)
            else:
                mo[k] = v
        if mo:
            overrides[name] = mo
        else:
            overrides.pop(name, None)
        # Try to persist to YAML.
        persisted = False
        persist_error: str | None = None
        src = getattr(cfg, "_source_path", None)
        if src is not None:
            try:
                self._persist_overrides_to_yaml(Path(src), overrides)
                persisted = True
            except Exception as e:  # pragma: no cover - defensive
                persist_error = str(e)
                logger.exception("persist mode_overrides failed")
        # Broadcast plugin hook.
        try:
            await self.app.broadcast(
                "on_mode_override_change",
                {"name": name, "override": mo, "persisted": persisted},
            )
        except Exception:
            logger.exception("on_mode_override_change broadcast failed")
        payload = self._build_override_payload(mode)
        payload["persisted"] = persisted
        if persist_error is not None:
            payload["persist_error"] = persist_error
        return web.json_response(payload)

    @staticmethod
    def _persist_overrides_to_yaml(path: Path, overrides: dict[str, Any]) -> None:
        """Re-write the YAML config file, replacing only mode_overrides.

        Loads with safe_load, mutates, dumps with default_flow_style=False.
        Comments in the file will be lost (pyyaml limitation) — this is
        acceptable for a developer-tool endpoint.
        """
        import yaml

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config root not a mapping: {type(raw).__name__}")
        if overrides:
            raw["mode_overrides"] = overrides
        else:
            raw.pop("mode_overrides", None)
        # Write atomically: tmp + rename.
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                raw, f,
                default_flow_style=False, allow_unicode=True, sort_keys=False,
            )
        tmp.replace(path)

    async def _api_wake(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            await self.app.wake(source="dashboard")
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    async def _api_sleep(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            await self.app.sleep()
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    async def _api_ptt_start(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            # Wake first (no-op if already awake), then jump straight to
            # LISTENING — bypassing the client-VAD speech_min wait so the
            # very first PCM chunk after the button-press is streamed.
            await self.app.wake(source="ptt")
            try:
                self.app._ptt_explicit_eos_pending = True
            except Exception:
                pass
            try:
                self.app._set_state(ConvState.LISTENING)
            except Exception:
                pass
            # Force VAD into "speech" mode so mic_pump starts streaming
            # immediately — without this, _update_vad would still wait
            # for client_vad_speech_min_ms of energy before transmitting.
            try:
                self.app._vad_state = "speech"
                self.app._vad_speech_ms = 0
                self.app._vad_silence_ms = 0
                self.app._vad_eos_sent = False
                # Allow asr_eos for the new turn (dedupe is per-turn).
                self.app._eos_sent_this_turn = False
            except Exception:
                pass
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    async def _api_ptt_end(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            # Explicit end: send asr_eos to SLV (Paraformer will finalise
            # the buffered audio) and let the normal dispatch path take
            # care of THINKING → SPEAKING → IDLE → SLEEPING via the
            # auto-sleep timer.
            # Use the dedupe helper so a client VAD silence trigger +
            # this explicit end can't both fire asr_eos and race the
            # SLV state machine.
            try:
                helper = getattr(self.app, "send_asr_eos_once", None)
                if callable(helper):
                    await helper()
                else:
                    await self.app.slv.asr_eos()
            except Exception:
                logger.debug("ptt/end asr_eos send failed", exc_info=True)
            try:
                self.app._ptt_explicit_eos_pending = False
            except Exception:
                pass
            try:
                self.app._set_state(ConvState.THINKING)
            except Exception:
                pass
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    # ── errors / agent-settings endpoints ───────────────────────

    async def _api_errors_clear(self, request):  # noqa: ANN001
        from aiohttp import web
        self._errors.clear()
        try:
            await self._broadcast("errors_cleared", None)
        except Exception:
            pass
        return web.json_response({"ok": True})

    def _build_agent_settings_payload(self) -> dict[str, Any]:
        cfg = getattr(self.app, "config", None)
        pipeline_mode = "always_on"
        sleep_timeout_s = 30.0
        stop_words: list[str] = []
        if cfg is not None:
            pipeline_mode = getattr(cfg, "pipeline_mode", "always_on")
            try:
                sleep_timeout_s = float(getattr(cfg, "sleep_timeout_s", 30.0))
            except Exception:
                sleep_timeout_s = 30.0
            raw_sw = getattr(cfg, "stop_words", None) or []
            if isinstance(raw_sw, (list, tuple)):
                stop_words = [str(x) for x in raw_sw]
        return {
            "pipeline_mode": pipeline_mode,
            "sleep_timeout_s": sleep_timeout_s,
            "stop_words": stop_words,
        }

    async def _api_agent_settings_get(self, request):  # noqa: ANN001
        from aiohttp import web
        return web.json_response(self._build_agent_settings_payload())

    async def _api_agent_settings_post(self, request):  # noqa: ANN001
        from aiohttp import web
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "bad json"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "body must be object"}, status=400)
        cfg = getattr(self.app, "config", None)
        if cfg is None:
            return web.json_response({"ok": False, "error": "no config"}, status=500)
        updated: dict[str, Any] = {}
        if "pipeline_mode" in body:
            v = body["pipeline_mode"]
            if v not in self._PIPELINE_MODES:
                return web.json_response(
                    {"ok": False, "error": f"pipeline_mode must be one of {list(self._PIPELINE_MODES)}"},
                    status=400,
                )
            # Persist + update config, but do NOT switch live state machine
            # — change takes effect on next agent restart.
            cfg.pipeline_mode = v
            updated["pipeline_mode"] = v
        if "sleep_timeout_s" in body:
            try:
                f = float(body["sleep_timeout_s"])
                if f < 0:
                    raise ValueError("must be >= 0")
            except (TypeError, ValueError) as e:
                return web.json_response(
                    {"ok": False, "error": f"sleep_timeout_s: {e}"}, status=400
                )
            cfg.sleep_timeout_s = f
            updated["sleep_timeout_s"] = f
        if "stop_words" in body:
            sw_raw = body["stop_words"]
            if not isinstance(sw_raw, list):
                return web.json_response(
                    {"ok": False, "error": "stop_words must be a list"}, status=400
                )
            sw = [str(x).strip() for x in sw_raw if str(x).strip()]
            cfg.stop_words = sw
            updated["stop_words"] = sw
        # Persist to YAML if we have a source path.
        persisted = False
        persist_error: str | None = None
        src = getattr(cfg, "_source_path", None)
        if src is not None and updated:
            try:
                self._persist_agent_settings_to_yaml(Path(src), updated)
                persisted = True
            except Exception as e:  # pragma: no cover - defensive
                persist_error = str(e)
                logger.exception("persist agent_settings failed")
        # Broadcast plugin hook.
        try:
            await self.app.broadcast(
                "on_agent_settings_change",
                {**self._build_agent_settings_payload(), "persisted": persisted, "changed": list(updated.keys())},
            )
        except Exception:
            logger.exception("on_agent_settings_change broadcast failed")
        payload = self._build_agent_settings_payload()
        payload["persisted"] = persisted
        payload["changed"] = list(updated.keys())
        if persist_error is not None:
            payload["persist_error"] = persist_error
        return web.json_response(payload)

    @staticmethod
    def _persist_agent_settings_to_yaml(path: Path, updated: dict[str, Any]) -> None:
        """Re-write the YAML config replacing only the changed agent-setting keys."""
        import yaml

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config root not a mapping: {type(raw).__name__}")
        for k, v in updated.items():
            raw[k] = v
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                raw, f,
                default_flow_style=False, allow_unicode=True, sort_keys=False,
            )
        tmp.replace(path)

    async def _api_session_history(self, request):  # noqa: ANN001
        from aiohttp import web
        system_prompt = getattr(self.app.config, "system_prompt", "")
        history = list(getattr(self.app.session, "history", []) or [])
        items: list[dict[str, str]] = []
        if system_prompt:
            items.append({"role": "system", "content": system_prompt})
        items.extend(history)
        return web.json_response(items)

    # ── broadcast helpers ────────────────────────────────────────

    def _schedule_broadcast(self, event: str, data: Any) -> None:
        try:
            asyncio.get_running_loop().create_task(self._broadcast(event, data))
        except RuntimeError:
            pass

    async def _broadcast(self, event: str, data: Any = None) -> None:
        if not self._browser_clients:
            return
        try:
            payload = json.dumps(
                {"event": event, "data": _safe(data), "ts": int(time.time() * 1000)},
                ensure_ascii=False,
            )
        except Exception:
            return
        for ws in list(self._browser_clients):
            try:
                if ws.closed:
                    self._browser_clients.discard(ws)
                    continue
                await ws.send_str(payload)
            except Exception:
                self._browser_clients.discard(ws)

    async def _emit_latency(self, kind: str, ms: float) -> None:
        if ms < 0:
            return
        self._latency_history[kind].append(float(ms))
        await self._broadcast("latency", {"kind": kind, "ms": int(ms)})

    async def _stats_loop(self) -> None:
        try:
            while True:
                mic_qsize = None
                try:
                    q = getattr(self.app.audio, "_in_queue", None)
                    if q is not None:
                        mic_qsize = q.qsize()
                except Exception:
                    mic_qsize = None

                ws_state = "closed"
                try:
                    slv = getattr(self.app, "slv", None)
                    raw_ws = getattr(slv, "_ws", None) if slv is not None else None
                    if raw_ws is None:
                        ws_state = "closed"
                    elif getattr(raw_ws, "closed", False):
                        ws_state = "closed"
                    else:
                        ws_state = "open"
                except Exception:
                    ws_state = "closed"

                await self._broadcast(
                    "stats",
                    {"mic_queue_depth": mic_qsize, "slv_ws_state": ws_state},
                )
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise

    # ── plugin observer hooks (forward to browser + compute latency) ─

    async def on_user_speech_start(self) -> None:
        await self._broadcast("on_user_speech_start", None)

    async def on_user_partial(self, text: str) -> None:
        await self._broadcast("on_user_partial", text)

    async def on_user_speech_end_client(self, data: dict) -> None:
        try:
            self._last_speech_end_ts_ms = int((data or {}).get("ts") or time.time() * 1000)
        except Exception:
            self._last_speech_end_ts_ms = int(time.time() * 1000)
        await self._broadcast("on_user_speech_end_client", data)

    async def on_user_utterance(self, text: str) -> None:
        now_ms = int(time.time() * 1000)
        self._turn_id += 1
        self._last_utterance_ts_ms = now_ms
        self._first_token_ts_ms = None
        self._first_tts_audio_ts_ms = None
        # Reset per-turn TTS counters at the start of a new user turn.
        self._tts_sentence_count = 0
        self._tts_bytes_current = 0
        # ASR latency: speech end → final utterance.
        if self._last_speech_end_ts_ms is not None:
            await self._emit_latency("asr", now_ms - self._last_speech_end_ts_ms)
        await self._broadcast("on_user_utterance", text)

    async def on_user_stop_intent(self, data: str) -> None:
        await self._broadcast("on_user_stop_intent", data)

    async def on_assistant_token(self, token: str) -> None:
        if self._first_token_ts_ms is None and self._last_utterance_ts_ms is not None:
            self._first_token_ts_ms = int(time.time() * 1000)
            await self._emit_latency(
                "ttft", self._first_token_ts_ms - self._last_utterance_ts_ms
            )
        await self._broadcast("on_assistant_token", token)

    async def on_assistant_sentence(self, sentence: str) -> None:
        self._tts_sentence_count += 1
        await self._broadcast(
            "on_assistant_sentence",
            sentence,
        )
        await self._broadcast(
            "tts_metrics",
            {
                "sentence_count": self._tts_sentence_count,
                "bytes_current": self._tts_bytes_current,
                "bytes_last": self._tts_bytes_last,
                "sample_rate": self._tts_last_sample_rate,
            },
        )

    async def on_assistant_sentence_start(self, sentence: str) -> None:
        await self._broadcast("on_assistant_sentence_start", sentence)

    async def on_assistant_done(self) -> None:
        # Snapshot per-turn TTS metrics into "last", then reset current.
        self._tts_bytes_last = self._tts_bytes_current
        duration_s = 0.0
        if self._tts_last_sample_rate > 0:
            # 16-bit PCM mono → 2 bytes per sample.
            duration_s = self._tts_bytes_last / float(self._tts_last_sample_rate) / 2.0
        await self._broadcast(
            "tts_metrics",
            {
                "sentence_count": self._tts_sentence_count,
                "bytes_current": 0,
                "bytes_last": self._tts_bytes_last,
                "last_duration_s": round(duration_s, 2),
                "sample_rate": self._tts_last_sample_rate,
            },
        )
        self._tts_bytes_current = 0
        await self._broadcast("on_assistant_done", None)

    async def on_tts_audio_frame(self, data: dict) -> None:
        if self._first_tts_audio_ts_ms is None:
            self._first_tts_audio_ts_ms = int(time.time() * 1000)
            if self._first_token_ts_ms is not None:
                await self._emit_latency(
                    "ttfa", self._first_tts_audio_ts_ms - self._first_token_ts_ms
                )
            if self._last_speech_end_ts_ms is not None:
                await self._emit_latency(
                    "rtt", self._first_tts_audio_ts_ms - self._last_speech_end_ts_ms
                )
        # Track bytes for the per-turn TTS metrics card.
        try:
            d = data or {}
            frame_len = int(d.get("frame_len") or 0)
            if frame_len > 0:
                self._tts_bytes_current += frame_len
            sr = int(d.get("sample_rate") or 0)
            if sr > 0:
                self._tts_last_sample_rate = sr
        except Exception:
            pass
        await self._broadcast("on_tts_audio_frame", data)

    async def on_error(self, exc: BaseException) -> None:
        # Prefer the exception's str() representation when it carries a
        # human-readable message (e.g. our LLMTimeoutError-derived
        # RuntimeError); fall back to repr() so empty/bare exceptions
        # still produce something useful.
        try:
            s = str(exc)
        except Exception:
            s = ""
        msg = s if s else repr(exc)
        entry = {"ts": int(time.time() * 1000), "msg": msg}
        self._errors.append(entry)
        if len(self._errors) > 50:
            self._errors.pop(0)
        await self._broadcast("on_error", msg)

    async def on_state_change(self, data: dict) -> None:
        await self._broadcast("on_state_change", data)

    async def on_slv_reconnect(self, data: dict) -> None:
        await self._broadcast("on_slv_reconnect", data)

    async def on_mic_rms(self, data: dict) -> None:
        await self._broadcast("on_mic_rms", data)

    async def on_mode_change(self, data: dict) -> None:
        await self._broadcast("on_mode_change", data)

    async def on_mode_override_change(self, data: dict) -> None:
        await self._broadcast("on_mode_override_change", data)

    async def on_mode_registered(self, data: dict) -> None:
        await self._broadcast("mode_registered", data)

    async def on_agent_settings_change(self, data: dict) -> None:
        await self._broadcast("on_agent_settings_change", data)

    async def on_transcribed(self, data: dict) -> None:
        await self._broadcast("on_transcribed", data)

    async def on_wake(self, data: dict) -> None:
        await self._broadcast("on_wake", data)

    async def on_sleep(self, data) -> None:
        await self._broadcast("on_sleep", data)

    # ── LLM availability / session-trim / prefix-cache relays ──────

    def _build_llm_availability_payload(self) -> dict[str, Any]:
        """Snapshot of current LLM availability for late-connecting clients."""
        payload: dict[str, Any] = {
            "state": "unknown",
            "last_ok_ts": None,
            "consecutive_failures": 0,
            "probe_interval_s": None,
        }
        avail = getattr(self.app, "llm_availability", None) if self.app else None
        if avail is not None:
            try:
                state = getattr(avail, "state", None)
                payload["state"] = (
                    state.value if hasattr(state, "value") else str(state or "unknown")
                )
                payload["last_ok_ts"] = getattr(avail, "last_ok_ts", None)
                payload["consecutive_failures"] = int(
                    getattr(avail, "consecutive_failures", 0) or 0
                )
                interval = getattr(avail, "interval_s", None)
                payload["probe_interval_s"] = (
                    float(interval) if interval is not None else None
                )
            except Exception:  # pragma: no cover - defensive
                logger.debug("availability snapshot failed", exc_info=True)
        return payload

    def _read_prefix_cache_disabled(self) -> bool:
        try:
            sess = getattr(self.app, "session", None) if self.app else None
            if sess is None:
                return False
            return bool(getattr(sess, "prefix_cache_disabled", False))
        except Exception:  # pragma: no cover - defensive
            return False

    async def on_llm_availability_change(self, data: dict) -> None:
        await self._broadcast("on_llm_availability_change", data)

    def _on_bus_session_trimmed(self, data: Any) -> None:
        self._schedule_broadcast("on_session_trimmed", data)

    def _on_bus_prefix_cache_disabled(self, data: Any) -> None:
        self._schedule_broadcast("on_prefix_cache_disabled", data)

    async def _api_llm_probe(self, request):  # noqa: ANN001
        from aiohttp import web

        avail = getattr(self.app, "llm_availability", None) if self.app else None
        if avail is None:
            return web.json_response(
                {"error": "llm_availability plugin not active"}, status=503
            )
        try:
            await avail.force_probe()
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
        return web.json_response(self._build_llm_availability_payload())

    async def on_llm_cache_metrics(self, data: dict) -> None:
        # edge-llm-chat-service payload shape:
        #   {"prefill": {"reused_tokens": N, "computed_tokens": M}, ...}
        # Older / flat shapes:
        #   {"hit_tokens": N, "total_tokens": T} or {"cache_hit_tokens": N}
        if not isinstance(data, dict):
            await self._broadcast("on_llm_cache_metrics", data)
            return
        prefill = data.get("prefill") or {}
        if not isinstance(prefill, dict):
            prefill = {}
        reused = (
            prefill.get("reused_tokens")
            or data.get("hit_tokens")
            or data.get("cache_hit_tokens")
            or 0
        )
        computed = prefill.get("computed_tokens") or 0
        if reused or computed:
            total = int(reused) + int(computed)
        else:
            total = int(data.get("total_tokens") or data.get("input_tokens") or 0)
        reused_i = int(reused)
        pct: float | None = (
            round(100.0 * reused_i / total, 1) if total else None
        )
        if pct is not None:
            self._cache_pct_history.append(pct)
        avg_pct: float | None = (
            round(sum(self._cache_pct_history) / len(self._cache_pct_history), 1)
            if self._cache_pct_history
            else None
        )
        await self._broadcast(
            "on_llm_cache_metrics",
            {
                "hit_pct": pct,
                "avg_pct": avg_pct,
                "reused": reused_i,
                "total": total,
                "raw": data,
            },
        )


def _safe(value: Any) -> Any:
    """Coerce to a JSON-serialisable form (best effort)."""
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


__all__ = ["DebugDashboardPlugin"]
