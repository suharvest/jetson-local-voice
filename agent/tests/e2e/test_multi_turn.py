"""Multi-turn: 3 utterances → session history preserved + reconnects happen."""
import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_multi_turn(test_config):
    audio = ScriptedAudioIO([
        (800,  WAV_DIR / "hello.wav"),
        (8000, WAV_DIR / "weather.wav"),
        (8000, WAV_DIR / "hello.wav"),
    ])
    async with run_agent(test_config, audio) as (app, probe):
        # Wait for 3 user utterances + final idle.
        for i in range(3):
            await probe.wait_event("on_user_utterance", timeout=30)
            await probe.wait_state("speaking", timeout=30)
            # Wait for that turn's TTS to finish before checking the next.
            # Track via assistant_done events.
            await _wait_assistant_done_count(probe, i + 1, timeout=30)

        # History: 3 user + 3 assistant = 6 messages.
        assert len(app.session.history) == 6, (
            f"expected 6 messages, got {len(app.session.history)}: "
            f"{[(m['role'], m['content'][:20]) for m in app.session.history]}"
        )
        # At least 2 reconnects observed via the slv_reconnect event.
        reconnects = [e for e in probe.events if e.get("event") == "on_slv_reconnect"]
        assert len(reconnects) >= 2, f"expected ≥2 reconnects, saw {len(reconnects)}"


async def _wait_assistant_done_count(probe, n: int, timeout: float = 30) -> None:
    import asyncio, time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        cnt = sum(1 for e in probe.events if e.get("event") == "on_assistant_done")
        if cnt >= n:
            return
        await asyncio.sleep(0.1)
    raise TimeoutError(f"expected {n} on_assistant_done, only saw "
                       f"{sum(1 for e in probe.events if e.get('event') == 'on_assistant_done')}")
