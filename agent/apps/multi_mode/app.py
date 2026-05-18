"""MultiModeApp — the standard voice app with runtime-switchable Modes.

Replaces the old DialogueApp. The pipeline lives in BaseApp; the
per-turn behavior is delegated to whatever AppMode is currently active.

INVARIANT: tokens stream DIRECTLY to SLV. No client-side sentence batching.
"""
from __future__ import annotations

import logging

from openvoicestream_agent import BaseApp
from openvoicestream_agent.app_mode import ModeContext, ModeManager
from openvoicestream_agent.state import ConvState
from openvoicestream_agent.modes import (
    ChatMode,
    InterpreterMode,
    MonologueMode,
    TranscribeMode,
)
from openvoicestream_agent.plugins.debug_dashboard import DebugDashboardPlugin
from openvoicestream_agent.plugins.llm_availability import LLMAvailabilityPlugin
from openvoicestream_agent.wake_sources import (
    HTTPWakeSource,
    LocalKeywordWakeSource,
    MQTTWakeSource,
    SerialWakeSource,
)

_WAKE_SOURCE_REGISTRY = {
    "http": HTTPWakeSource,
    "mqtt": MQTTWakeSource,
    "serial": SerialWakeSource,
    "local_keyword": LocalKeywordWakeSource,
}

logger = logging.getLogger(__name__)


class MultiModeApp(BaseApp):
    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        self.register(DebugDashboardPlugin(self))
        self.register(LLMAvailabilityPlugin(self))

        # Register WakeSources for non-always_on pipeline modes.
        if getattr(config, "pipeline_mode", "always_on") != "always_on":
            for ws_name in getattr(config, "wake_sources", ["http"]) or []:
                cls = _WAKE_SOURCE_REGISTRY.get(ws_name)
                if cls is None:
                    logger.warning("unknown wake_source: %r (skipped)", ws_name)
                    continue
                try:
                    self.register(cls(self))
                except Exception:
                    logger.exception("wake_source %s registration failed", ws_name)

        # Build ModeManager with a context factory so each hook
        # invocation sees a fresh ModeContext bound to current state.
        self.modes = ModeManager(self._make_mode_ctx)

        # Register built-in modes. Users may register more after
        # construction or subclass to swap the default set.
        self.modes.register(ChatMode())
        self.modes.register(InterpreterMode())
        self.modes.register(MonologueMode())
        self.modes.register(TranscribeMode())

    def _make_mode_ctx(self) -> ModeContext:
        ctx = ModeContext(
            config=self.config,
            slv=self.slv,
            llm=self.llm,
            session=self.session,
            audio=self.audio,
            events=self.events,
            broadcast=self.broadcast,
        )
        # Wire the ModeManager so `_resolve_system_prompt` can find
        # the current mode's class default + per-mode config overrides.
        ctx.mode_manager = getattr(self, "modes", None)
        return ctx

    async def run(self) -> None:
        default = getattr(self.config, "default_mode", "chat") or "chat"
        await self.modes.start(default)
        await super().run()

    def _restore_idle_after_silent_turn(self) -> None:
        """Pull FSM out of THINKING when no TTS turn will run.

        BaseApp._dispatch_one sets THINKING before invoking the LLM
        turn. The SPEAKING→IDLE return path only fires on TTSDone, so
        modes that produce no TTS (transcribe) or drop the utterance
        entirely (monologue.preprocess) would leave the FSM stuck.
        """
        if getattr(self, "_state", ConvState.IDLE) == ConvState.THINKING:
            self._set_state(ConvState.IDLE)
            try:
                self._reset_sleep_timer()
            except Exception:  # pragma: no cover - defensive
                logger.debug("_reset_sleep_timer failed", exc_info=True)

    async def on_user_utterance(self, text: str) -> None:
        mode = self.modes.current
        try:
            preprocessed = mode.preprocess_user_text(text)
        except Exception:
            logger.exception(
                "mode %s preprocess_user_text crashed; using raw text", mode.name
            )
            preprocessed = text
        if preprocessed is None:
            logger.info("mode %s dropped utterance via preprocess", mode.name)
            self._restore_idle_after_silent_turn()
            return
        logger.info(
            "on_user_utterance: dispatching to mode=%s text=%r", mode.name, preprocessed
        )
        await mode.on_user_utterance(self._make_mode_ctx(), preprocessed)
        # If the mode declared it doesn't produce TTS, the SPEAKING→IDLE
        # transition will never fire from TTSDone — restore IDLE here.
        if not getattr(mode, "produces_tts", True):
            self._restore_idle_after_silent_turn()

    async def broadcast(self, hook_name: str, *args) -> None:
        # Default plugin fan-out from BaseApp.
        await super().broadcast(hook_name, *args)
        # Also invoke the active Mode's matching hook when it has one.
        # Only a small set of hooks are forwarded — the ones a Mode is
        # documented to override. This keeps the surface narrow.
        if hook_name == "on_assistant_done":
            mode = getattr(self, "modes", None)
            if mode is None or mode._current is None:
                return
            try:
                await mode.current.on_assistant_done(self._make_mode_ctx())
            except Exception:
                logger.exception("mode %s on_assistant_done failed", mode.current.name)


__all__ = ["MultiModeApp"]
