"""Idle stability: 30s of silence → no exceptions, WS still open, state IDLE."""
import asyncio
import pytest

from .conftest import run_agent
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_idle_30s(test_config):
    # Empty script → only silence is yielded forever.
    audio = ScriptedAudioIO([])
    async with run_agent(test_config, audio) as (app, probe):
        await asyncio.sleep(30)
        # No errors raised on the bus.
        assert probe.errors == [], f"errors during idle: {probe.errors}"
        # SLV WS state via stats events.
        stats = [e for e in probe.events if e.get("event") == "stats"]
        # Last stats event must show open WS.
        assert stats, "no stats events received in 30s"
        last_state = (stats[-1].get("data") or {}).get("slv_ws_state")
        assert last_state == "open", f"WS state at end: {last_state}"
        # No spurious utterances.
        u = [e for e in probe.events if e.get("event") == "on_user_utterance"]
        # Allow empty finals (filtered out before on_user_utterance fires).
        assert u == [], f"spurious utterances during silence: {[e.get('data') for e in u]}"
