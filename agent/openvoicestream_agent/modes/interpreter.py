"""InterpreterMode — stateless real-time translator (zh → en)."""
from __future__ import annotations

from ..app_mode import AppMode, ModeContext


class InterpreterMode(AppMode):
    name = "interpreter"
    display_name = "同传"
    icon = "🌐"
    description = "把听到的中文翻译成英文，不维护对话上下文"
    system_prompt = (
        "You are a real-time interpreter. Translate the user's "
        "Chinese into natural spoken English. Output ONLY the "
        "translation, no preface or explanation."
    )
    max_history = 0  # interpreter is stateless

    async def on_user_utterance(self, ctx: ModeContext, text: str) -> None:
        # Clear history before each turn — interpreter is stateless.
        ctx.session.history.clear()
        # No explicit system_prompt_override: let _resolve_system_prompt
        # walk mode_overrides → class .system_prompt, so dashboard edits
        # to mode_overrides["interpreter"] actually take effect.
        await ctx.run_default_dialogue_turn(text)


__all__ = ["InterpreterMode"]
