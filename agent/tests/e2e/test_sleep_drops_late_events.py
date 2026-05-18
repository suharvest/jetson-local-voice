"""E2E regression: /sleep must drop late-arriving ASR events.

Boots wake_word agent → wake → speak first utterance → mid-turn POST
/sleep → inject more audio → assert NO second on_user_utterance fires
(SLEEPING gate at dispatch boundary must drop ASRFinal text).
"""
import asyncio
from dataclasses import replace

import aiohttp
import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_sleep_drops_late_asr_events(test_config):
    cfg = replace(test_config, pipeline_mode="wake_word", sleep_timeout_s=60.0)

    # Two utterances back-to-back; the SECOND one is injected after
    # we POST /sleep, so its ASRFinal should NOT trigger an LLM turn.
    audio = ScriptedAudioIO([
        (1500, WAV_DIR / "hello.wav"),
        (4000, WAV_DIR / "weather.wav"),
    ])

    async with run_agent(cfg, audio) as (app, probe):
        await asyncio.sleep(0.5)
        assert app._state.value == "sleeping"

        port = cfg.metadata["dashboard_port"]
        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"http://127.0.0.1:{port}/api/control/wake") as r:
                assert r.status == 200
            await probe.wait_event("on_wake", timeout=5)

            # Let the first utterance complete a turn.
            await probe.wait_event("on_user_utterance", timeout=25)
            first_utterance_count = sum(
                1 for e in probe.events if e.get("event") == "on_user_utterance"
            )
            assert first_utterance_count == 1

            # Sleep BEFORE the second WAV is finished being decoded by SLV.
            await asyncio.sleep(0.5)
            async with sess.post(f"http://127.0.0.1:{port}/api/control/sleep") as r:
                assert r.status == 200
            await probe.wait_event("on_sleep", timeout=5)
            assert app._state.value == "sleeping"

        # Give SLV plenty of time to produce any in-flight ASRFinal from
        # the second WAV; the SLEEPING gate must drop it.
        await asyncio.sleep(8.0)

        # Core regression assertion: no second on_user_utterance leaked
        # through the SLEEPING gate. (The FSM may settle to IDLE if a
        # TTSDone for the cancelled first turn lands after sleep, but
        # no new LLM turn must be spawned.)
        final_utterance_count = sum(
            1 for e in probe.events if e.get("event") == "on_user_utterance"
        )
        assert final_utterance_count == 1, (
            "second utterance leaked through SLEEPING gate; "
            f"saw {final_utterance_count} on_user_utterance events"
        )
