"""E2E: pipeline_mode=push_to_talk.

Boots SLEEPING → POST /api/control/ptt/start → LISTENING → inject audio
→ POST /api/control/ptt/end → THINKING → SPEAKING → IDLE → SLEEPING.
"""
import asyncio
from dataclasses import replace

import aiohttp
import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_push_to_talk_full_cycle(test_config):
    cfg = replace(test_config, pipeline_mode="push_to_talk", sleep_timeout_s=30.0)

    # Big initial delay (3s) ensures the WAV is still queued when PTT/start
    # arrives. Without this, ScriptedAudioIO would have already streamed
    # hello.wav into the SLEEPING-gated mic_pump (which drops it).
    audio = ScriptedAudioIO([(3000, WAV_DIR / "hello.wav")])

    async with run_agent(cfg, audio) as (app, probe):
        await asyncio.sleep(0.5)
        assert app._state.value == "sleeping"

        port = cfg.metadata["dashboard_port"]
        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"http://127.0.0.1:{port}/api/control/ptt/start") as r:
                assert r.status == 200, await r.text()

            # Should reach LISTENING.
            try:
                await probe.wait_state("listening", timeout=5)
            except TimeoutError:
                raise AssertionError(
                    f"listening not reached; state_history={probe.state_history}; "
                    f"app._state={app._state}"
                )

            # Wait for the scripted audio to actually be injected + flushed.
            await asyncio.sleep(4.0)

            async with sess.post(f"http://127.0.0.1:{port}/api/control/ptt/end") as r:
                assert r.status == 200

        # Now expect the normal post-EOS pipeline.
        try:
            await probe.wait_event("on_user_utterance", timeout=25)
        except TimeoutError:
            uniq = []
            seen_set = set()
            for e in probe.events:
                ev = e.get("event")
                if ev not in seen_set:
                    seen_set.add(ev)
                    uniq.append(ev)
            raise AssertionError(
                f"no user_utterance. unique events: {uniq}; "
                f"state_history={probe.state_history}; state={app._state}"
            )
        await probe.wait_state("speaking", timeout=25)
        await probe.wait_state("idle", timeout=30)

        # Captured TTS should be non-trivial.
        assert len(audio.captured_tts) > 1000, (
            f"expected TTS PCM > 1000, got {len(audio.captured_tts)}"
        )
