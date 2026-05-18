"""MonologueMode — proactive periodic broadcasts; ignores user input."""
from __future__ import annotations

import asyncio
import logging

from ..app_mode import AppMode, ModeContext

logger = logging.getLogger(__name__)


class MonologueMode(AppMode):
    name = "monologue"
    display_name = "独白"
    icon = "📢"
    description = "定时主动说话，忽略用户输入"
    interval_s: float = 60.0

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None

    async def enter(self, ctx: ModeContext) -> None:
        # Capture ctx for the broadcast loop; the factory always
        # returns a fresh ctx but the slv/session/etc references it
        # points to are stable for the app lifetime.
        self._task = asyncio.create_task(
            self._broadcast_loop(ctx), name="monologue-loop"
        )

    async def exit(self, ctx: ModeContext) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    def preprocess_user_text(self, text: str) -> str | None:
        # Monologue ignores user input by design — drop the utterance at
        # preprocess so MultiModeApp.on_user_utterance takes the
        # "dropped" branch and restores IDLE (otherwise the FSM stays
        # stuck in THINKING because no TTS turn ever runs).
        return None

    async def on_user_utterance(self, ctx: ModeContext, text: str) -> None:
        # Unreachable in practice (preprocess returns None), but kept
        # for protocol compliance / direct-call tests.
        return

    async def _broadcast_loop(self, ctx: ModeContext) -> None:
        try:
            while True:
                await asyncio.sleep(self.interval_s)
                try:
                    await ctx.run_default_dialogue_turn(
                        "Pick a fresh topic and share an interesting "
                        "one-line fact in Chinese.",
                        system_prompt_override="你是百科助手。",
                    )
                except Exception:
                    logger.exception("monologue broadcast turn failed")
        except asyncio.CancelledError:
            raise


__all__ = ["MonologueMode"]
