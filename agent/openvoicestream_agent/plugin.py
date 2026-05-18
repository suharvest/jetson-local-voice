"""Plugin base class for the OpenVoiceStream Agent.

Lifecycle:
  1. __init__(app) -- store reference to shared BaseApp context
  2. setup() -- check deps/hardware, return False to skip gracefully
  3. async start() -- run alongside the app event loop (no-op default)
  4. async stop() -- shutdown (called in reverse registration order)

Semantic hooks (observer broadcasts -- they do NOT route the
conversation; BaseApp.on_user_utterance is the single router):
  - async on_user_speech_start()
  - async on_user_partial(text: str)
  - async on_user_utterance(text: str)
  - async on_assistant_token(token: str)
  - async on_assistant_sentence(sentence: str)
  - async on_assistant_done()
  - async on_error(exc: BaseException)

All hooks default to no-op; override only what you need.
"""
from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app_base import BaseApp

logger = logging.getLogger(__name__)


class Plugin(ABC):
    name: str = "unnamed"

    def __init__(self, app: "BaseApp") -> None:
        self.app = app
        self._running = False

    def setup(self) -> bool:
        """Sync prerequisite check. Return False to skip the plugin."""
        return True

    async def start(self) -> None:
        """Optional async startup. Default: no-op."""
        self._running = True

    async def stop(self) -> None:
        """Optional async shutdown. Default: no-op."""
        self._running = False

    # ── observer hooks ─────────────────────────────────────────────

    async def on_user_speech_start(self) -> None:
        """VAD detected the user started talking. Default: no-op."""

    async def on_user_partial(self, text: str) -> None:
        """ASR partial transcript update. Default: no-op."""

    async def on_user_utterance(self, text: str) -> None:
        """ASR final transcript for one utterance. Default: no-op."""

    async def on_assistant_token(self, token: str) -> None:
        """One LLM streaming token. Default: no-op."""

    async def on_assistant_sentence(self, sentence: str) -> None:
        """SLV finished synthesizing one sentence. Default: no-op."""

    async def on_assistant_done(self) -> None:
        """SLV emitted tts_done. Default: no-op."""

    async def on_error(self, exc: BaseException) -> None:
        """Any V2V transport/protocol error. Default: no-op."""

    # ── v2 hooks (state machine + dashboard instrumentation) ───────

    async def on_state_change(self, data: dict) -> None:
        """ConvState transition. data = {"state": str, "prev": str}."""

    async def on_user_stop_intent(self, data: str) -> None:
        """User said a stop-word; assistant was silenced. data = matched text."""

    async def on_user_speech_end_client(self, data: dict) -> None:
        """Client-side VAD detected end of speech. data = {"ts": int, "drove_eos": bool}."""

    async def on_slv_reconnect(self, data: dict) -> None:
        """SLV WebSocket was reconnected. data = {"count": int}."""

    async def on_mic_rms(self, data: dict) -> None:
        """Per-chunk mic RMS sample. data = {"rms": float, "threshold": float, "state": str}."""

    async def on_llm_cache_metrics(self, data: dict) -> None:
        """Per-turn LLM prefix-cache stats. data = {"hit_tokens": int, "total_tokens": int}."""

    async def on_tts_audio_frame(self, data: dict) -> None:
        """First TTS audio frame of a turn. data = {"sample_rate": int, "frame_len": int}."""

    async def on_mode_change(self, data: dict) -> None:
        """AppMode switched. data = {"name": str, "display_name": str, "icon": str, "prev": str | None}."""

    async def on_mode_override_change(self, data: dict) -> None:
        """Per-mode override (system_prompt/temperature) was edited via the
        dashboard. data = {"name": str, "override": dict, "persisted": bool}."""

    async def on_transcribed(self, data: dict) -> None:
        """TranscribeMode emitted a finalised user utterance. data = {"text": str}."""

    # ── pipeline_mode hooks (wake_word / push_to_talk) ────────────

    async def on_wake(self, data: dict) -> None:
        """Agent transitioned SLEEPING → IDLE. data = {"source": str}."""

    async def on_sleep(self, data) -> None:
        """Agent transitioned (any) → SLEEPING. data is None."""

    # ── runtime config / mode-registry hooks ──────────────────────

    async def on_agent_settings_change(self, data: dict) -> None:
        """Agent-level settings (pipeline_mode / sleep_timeout_s /
        stop_words) were edited via the dashboard. data carries the
        updated fields, e.g. {"pipeline_mode": str, "sleep_timeout_s":
        float, "stop_words": list[str], "persisted": bool}. Default: no-op."""

    async def on_mode_registered(self, data: dict) -> None:
        """A new AppMode was registered after ModeManager.start() — the
        dashboard listens to refresh its dropdown. data = {"name": str,
        "display_name": str, "icon": str, "description": str}. Default: no-op."""
