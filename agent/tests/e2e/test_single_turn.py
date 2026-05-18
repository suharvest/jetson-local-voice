"""Single utterance: ASR → LLM → TTS roundtrip."""
import asyncio
import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_single_turn(test_config):
    audio = ScriptedAudioIO([(800, WAV_DIR / "hello.wav")])
    async with run_agent(test_config, audio) as (app, probe):
        await probe.wait_event("on_user_utterance", timeout=20)
        await probe.wait_state("speaking", timeout=25)
        await probe.wait_state("idle", timeout=30)
        # Give DialogueApp's finally-block a chance to append the assistant
        # message after the LLM stream completes (it runs after TTSDone).
        for _ in range(50):
            if app.session.history and app.session.history[-1]["role"] == "assistant":
                break
            await asyncio.sleep(0.1)

        assert len(audio.captured_tts) > 1000, (
            f"expected TTS PCM bytes > 1000, got {len(audio.captured_tts)}"
        )
        assert app.session.history, "session history should not be empty"
        assert app.session.history[-1]["role"] == "assistant", (
            f"expected last msg to be assistant; history: {app.session.history}"
        )
        assert any(s == "thinking" for _, s in probe.state_history), (
            f"expected a THINKING transition, got: {probe.state_history}"
        )
