"""E2E: pipeline_mode=wake_word.

Boots SLEEPING → POST /api/control/wake → IDLE → utterance runs → IDLE
→ sleep_timeout_s elapses → SLEEPING again.
"""
import asyncio
from dataclasses import replace

import aiohttp
import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_wake_word_full_cycle(test_config):
    cfg = replace(test_config, pipeline_mode="wake_word", sleep_timeout_s=8.0)

    # Audio script: short pre-wake gap (during SLEEPING — VAD won't fire
    # since chunks are dropped), then "hello" after wake.
    audio = ScriptedAudioIO([(1500, WAV_DIR / "hello.wav")])

    async with run_agent(cfg, audio) as (app, probe):
        # Initial state must be SLEEPING.
        await asyncio.sleep(0.5)
        assert app._state.value == "sleeping", f"expected SLEEPING, got {app._state}"

        # POST wake.
        port = cfg.metadata["dashboard_port"]
        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"http://127.0.0.1:{port}/api/control/wake") as r:
                assert r.status == 200

        # on_wake hook should have fired.
        await probe.wait_event("on_wake", timeout=5)
        # State went to IDLE.
        assert any(s == "idle" for _, s in probe.state_history)

        # Inject hello.wav (the second script entry should now be reachable).
        # Wait for the full turn to complete.
        await probe.wait_event("on_user_utterance", timeout=25)
        await probe.wait_state("speaking", timeout=25)
        await probe.wait_state("idle", timeout=30)

        # Now wait sleep_timeout_s + buffer for auto-sleep.
        await probe.wait_state("sleeping", timeout=cfg.sleep_timeout_s + 5)
