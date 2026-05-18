"""ChatMode — the classic multi-turn voice chat behavior."""
from __future__ import annotations

from ..app_mode import AppMode, ModeContext


class ChatMode(AppMode):
    name = "chat"
    display_name = "对话"
    icon = "💬"
    description = "标准多轮对话，记忆上下文"

    async def on_user_utterance(self, ctx: ModeContext, text: str) -> None:
        await ctx.run_default_dialogue_turn(text)


__all__ = ["ChatMode"]
