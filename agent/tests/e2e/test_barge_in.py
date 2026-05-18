"""Barge-in: while TTS is playing, a new utterance must cancel it.

Sync model: the test scripts ONLY the story-request WAV. Once we see
TTS bytes start landing in the fake sink (= agent is actively
playing), we dynamically inject the barge-in WAV via
`ScriptedAudioIO.inject()`. This avoids races where a too-fast LLM
response finishes TTS before a fixed pre-delay elapses.
"""
import asyncio
import time
import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_barge_in_cancels_tts(test_config):
    # Use a longer-response trigger: explicit instruction to produce
    # a detailed answer increases the odds of a multi-sentence TTS.
    audio = ScriptedAudioIO([
        (500, WAV_DIR / "story_request.wav"),
    ])
    async with run_agent(test_config, audio) as (app, probe):
        # Step 1: wait for TTS bytes to start flowing.
        t_dl = time.monotonic() + 30
        while time.monotonic() < t_dl and len(audio.captured_tts) == 0:
            await asyncio.sleep(0.05)
        assert len(audio.captured_tts) > 0, "TTS never started"

        # Step 2: give TTS a brief head-start so the agent is firmly
        # in SPEAKING state, then inject the barge-in WAV.
        await asyncio.sleep(0.4)
        bytes_at_inject = len(audio.captured_tts)
        speaking_seen = any(s == "speaking" for _, s in probe.state_history)
        assert speaking_seen, (
            f"expected SPEAKING before barge-in; saw {probe.state_history}"
        )
        audio.inject(WAV_DIR / "barge_in.wav")

        # Step 3: wait for barged_in state.
        t_dl = time.monotonic() + 15
        while time.monotonic() < t_dl:
            if any(s == "barged_in" for _, s in probe.state_history):
                break
            await asyncio.sleep(0.05)
        else:
            partials = [
                e.get("data") for e in probe.events
                if e.get("event") == "on_user_partial"
            ]
            pytest.fail(
                f"barged_in state never reached. "
                f"state_history={probe.state_history!r} "
                f"partials={partials!r}"
            )

        # Step 4: TTS bytes should stop growing within ~500 ms after
        # barge-in. Allow a small grace (one frame ≈ 32-64 KB at 24kHz).
        await asyncio.sleep(0.6)
        bytes_after = len(audio.captured_tts)
        grew = bytes_after - bytes_at_inject
        # Allow ~150 KB grace for in-flight chunks (≈ 3s of 24kHz mono
        # int16). If TTS keeps streaming a full LLM response after
        # barge-in, growth would be much larger.
        assert grew < 150_000, (
            f"TTS kept streaming after barge-in: "
            f"+{grew} bytes (inject@{bytes_at_inject}, after={bytes_after})"
        )
