"""Regression guard: pipeline_mode work must NOT change always_on behaviour.

Mirrors the structure of test_single_turn so any drift in the always_on
audio pipeline (mic_pump, VAD, dispatch) breaks here first.
"""
import asyncio
import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_always_on_default_unchanged(test_config):
    # test_config defaults to pipeline_mode=always_on; assert it explicitly.
    assert test_config.pipeline_mode == "always_on"

    audio = ScriptedAudioIO([(800, WAV_DIR / "hello.wav")])
    async with run_agent(test_config, audio) as (app, probe):
        # SLEEPING must never appear in the always_on path.
        await probe.wait_event("on_user_utterance", timeout=20)
        await probe.wait_state("speaking", timeout=25)
        await probe.wait_state("idle", timeout=30)

        # No sleep transitions expected.
        assert not any(s == "sleeping" for _, s in probe.state_history), (
            f"unexpected SLEEPING transition in always_on: {probe.state_history}"
        )

        for _ in range(50):
            if app.session.history and app.session.history[-1]["role"] == "assistant":
                break
            await asyncio.sleep(0.1)

        assert len(audio.captured_tts) > 1000
        assert app.session.history[-1]["role"] == "assistant"
