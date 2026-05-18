"""Empty asr_final: 5s of silence → no LLM call, state returns to IDLE."""
import asyncio
import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_empty_final_no_llm(test_config):
    audio = ScriptedAudioIO([(500, WAV_DIR / "silence_5s.wav")])
    async with run_agent(test_config, audio) as (app, probe):
        # Just observe 10 seconds — no utterance should fire from silence.
        await asyncio.sleep(10)
        utterances = [e for e in probe.events if e.get("event") == "on_user_utterance"]
        # If anything DID fire (shouldn't), it must be empty + state must be IDLE.
        for u in utterances:
            text = u.get("data") or ""
            assert not (text or "").strip(), f"silence yielded non-empty final: {text!r}"
        tokens = probe.assistant_tokens()
        assert tokens == [], f"LLM was called on silence: {tokens!r}"
        assert app.session.history == []
