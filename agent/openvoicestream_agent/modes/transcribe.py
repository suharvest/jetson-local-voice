"""TranscribeMode — pass-through ASR with no LLM and no TTS."""
from __future__ import annotations

from ..app_mode import AppMode, ModeContext


class TranscribeMode(AppMode):
    name = "transcribe"
    display_name = "转录"
    icon = "📝"
    description = "只记录用户语音转文字，不调 LLM 不播音"
    barge_in_enabled = False
    # Transcribe never produces TTS — dispatcher must restore IDLE after
    # each utterance, otherwise FSM gets stuck in THINKING permanently.
    produces_tts = False

    async def on_user_utterance(self, ctx: ModeContext, text: str) -> None:
        # Just broadcast for downstream consumers (diary plugin etc).
        # Do not touch session.history, do not call LLM, do not synth TTS.
        await ctx.broadcast("on_transcribed", {"text": text})


__all__ = ["TranscribeMode"]
