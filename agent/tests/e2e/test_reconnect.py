"""SLV reconnect: SLV closes WS after each asr_eos; agent must reconnect."""
import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_reconnect_per_turn(test_config):
    audio = ScriptedAudioIO([
        (800,  WAV_DIR / "hello.wav"),
        (8000, WAV_DIR / "hello.wav"),
    ])
    async with run_agent(test_config, audio) as (app, probe):
        await probe.wait_event("on_user_utterance", timeout=20)
        # First reconnect (after first turn's asr_eos).
        await probe.wait_event("on_slv_reconnect", timeout=20)
        # Wait for second turn.
        cnt = sum(1 for e in probe.events if e.get("event") == "on_user_utterance")
        deadline_ev = "on_user_utterance"
        # Wait until we see >=2 user utterances and >=1 reconnect.
        import asyncio, time
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            u = sum(1 for e in probe.events if e.get("event") == "on_user_utterance")
            r = sum(1 for e in probe.events if e.get("event") == "on_slv_reconnect")
            if u >= 2 and r >= 2:
                break
            await asyncio.sleep(0.2)
        u = sum(1 for e in probe.events if e.get("event") == "on_user_utterance")
        r = sum(1 for e in probe.events if e.get("event") == "on_slv_reconnect")
        assert u >= 2, f"expected ≥2 utterances, got {u}"
        assert r >= 2, f"expected ≥2 reconnects, got {r}"
        # Reconnect counter reflected on the app.
        assert getattr(app, "_slv_reconnect_count", 0) >= 2
