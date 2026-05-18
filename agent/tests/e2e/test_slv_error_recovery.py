"""SLVError recovery: force an SLV error mid-THINKING → state→IDLE, LLM cancelled."""
import asyncio
import pytest

from openvoicestream_agent.slv_client import SLVError

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_slv_error_resets_state(test_config):
    audio = ScriptedAudioIO([(800, WAV_DIR / "story_request.wav")])
    async with run_agent(test_config, audio) as (app, probe):
        # Wait until LLM turn is in flight (THINKING fired or assistant tokens flowing).
        try:
            await probe.wait_state("thinking", timeout=20)
        except TimeoutError:
            pytest.fail("never reached THINKING state")

        # Inject an SLVError into the dispatch path by calling _dispatch_one directly.
        await app._dispatch_one(SLVError(message="injected fault"))

        # Within a couple seconds we should land on IDLE.
        deadline = asyncio.get_event_loop().time() + 5
        while asyncio.get_event_loop().time() < deadline:
            if any(s == "idle" for _, s in probe.state_history[-5:]):
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail(
                f"state did not return to IDLE after SLVError. history: {probe.state_history}"
            )

        # LLM turn task should be done (cancelled or finished).
        if app._llm_turn_task is not None:
            assert app._llm_turn_task.done(), "LLM turn was not cancelled after SLVError"
        # on_error event surfaced to dashboard. Wait for the WS round-trip
        # to deliver it; the _dispatch_one broadcast is awaited but the
        # probe's reader runs in a separate task and hasn't necessarily
        # consumed the frame by the time the IDLE poll returns.
        try:
            await probe.wait_event("on_error", timeout=3)
        except TimeoutError:
            pytest.fail(
                f"on_error event never reached dashboard probe. "
                f"events seen: {[e.get('event') for e in probe.events[-20:]]}"
            )
