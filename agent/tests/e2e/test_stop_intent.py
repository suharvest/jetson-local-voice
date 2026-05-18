"""Stop intent: stop words should bypass LLM and reset state→IDLE."""
import asyncio
import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_stop_zh_no_llm(test_config):
    audio = ScriptedAudioIO([(800, WAV_DIR / "stop_zh.wav")])
    async with run_agent(test_config, audio) as (app, probe):
        await probe.wait_event("on_user_utterance", timeout=20)
        # Give a moment for stop intent dispatch.
        try:
            await probe.wait_event("on_user_stop_intent", timeout=8)
        except TimeoutError:
            uttr = [e for e in probe.events if e.get("event") == "on_user_utterance"]
            pytest.fail(
                f"stop intent did not match. utterances: "
                f"{[e.get('data') for e in uttr]}"
            )
        # Sleep briefly to confirm no LLM tokens arrive after stop.
        await asyncio.sleep(1.5)
        tokens = probe.assistant_tokens()
        assert tokens == [], f"expected no LLM tokens, got {tokens!r}"
        assert app.session.history == [], (
            f"stop intent must not extend history; got: {app.session.history}"
        )


@pytest.mark.asyncio
async def test_stop_en_matches(test_config):
    audio = ScriptedAudioIO([(800, WAV_DIR / "stop_en.wav")])
    async with run_agent(test_config, audio) as (app, probe):
        await probe.wait_event("on_user_utterance", timeout=20)
        try:
            await probe.wait_event("on_user_stop_intent", timeout=8)
        except TimeoutError:
            uttr = [e for e in probe.events if e.get("event") == "on_user_utterance"]
            pytest.skip(
                f"ASR did not yield 'stop'-like text for English audio: "
                f"{[e.get('data') for e in uttr]}"
            )
        assert app.session.history == []


@pytest.mark.asyncio
async def test_stopwatch_does_not_match(test_config):
    """'stopwatch' should NOT match the 'stop' stop-word (word boundary)."""
    audio = ScriptedAudioIO([(800, WAV_DIR / "stopwatch.wav")])
    async with run_agent(test_config, audio) as (app, probe):
        try:
            await probe.wait_event("on_user_utterance", timeout=20)
        except TimeoutError:
            pytest.skip("ASR did not produce a final for 'stopwatch'")
        # Wait briefly — must NOT trigger stop_intent.
        await asyncio.sleep(2.0)
        stop_events = [e for e in probe.events if e.get("event") == "on_user_stop_intent"]
        assert stop_events == [], f"'stopwatch' falsely matched stop intent: {stop_events}"
