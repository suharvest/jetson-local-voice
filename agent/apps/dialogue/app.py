"""DialogueApp -- the canonical Phase-1 App: simple voice chat.

INVARIANT: tokens stream DIRECTLY to SLV. No client-side sentence batching.
"""
from __future__ import annotations

from openvoicestream_agent import BaseApp


class DialogueApp(BaseApp):
    async def on_user_utterance(self, text: str) -> None:
        if not text.strip():
            return
        self.session.add_user(text)
        chunks: list[str] = []
        try:
            async for token in self.llm.stream(
                self.session.messages(self.config.system_prompt),
                session=self.session,
            ):
                chunks.append(token)
                self.events.emit("assistant_token", token)
                await self.broadcast("on_assistant_token", token)
                await self.slv.send_text(token)
        finally:
            # Always flush so SLV stops the TTS pipeline cleanly even if
            # the LLM stream raised mid-turn, and persist whatever tokens
            # we did receive so the conversation stays consistent.
            try:
                await self.slv.flush_tts()
            except Exception:  # pragma: no cover - best effort
                pass
            if chunks:
                self.session.add_assistant("".join(chunks))


__all__ = ["DialogueApp"]
