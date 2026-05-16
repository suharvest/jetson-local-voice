"""P0-A regression: dispatch loop must not block during an LLM turn.

While `on_user_utterance` streams tokens, the dispatch loop must still
process queued TTSAudio (so the speaker plays) and ASRPartial (so
barge-in fires `slv.abort()`).
"""
from __future__ import annotations

import asyncio

import pytest

from openvoicestream_agent.app_base import BaseApp
from openvoicestream_agent.slv_client import ASRFinal, ASRPartial, TTSAudio


class _FakeAudio:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.is_playing = False

    async def play(self, pcm: bytes) -> None:
        self.is_playing = True
        self.played.append(pcm)

    async def stop_playback(self) -> None:
        self.is_playing = False

    def set_output_sample_rate(self, sr: int) -> None:
        pass


class _FakeSLV:
    def __init__(self) -> None:
        self.aborted = 0

    async def abort(self) -> None:
        self.aborted += 1


@pytest.mark.asyncio
async def test_dispatch_does_not_block_on_llm_turn():
    app = BaseApp.__new__(BaseApp)
    app.audio = _FakeAudio()
    app.slv = _FakeSLV()
    app.plugins = []
    app._first_tts_seen = False
    app._llm_turn_task = None

    llm_started = asyncio.Event()
    llm_release = asyncio.Event()
    seen_during_llm: dict[str, bool] = {"audio": False, "abort": False}

    async def slow_on_user_utterance(text: str) -> None:
        llm_started.set()
        await llm_release.wait()  # block until test releases

    app.on_user_utterance = slow_on_user_utterance  # type: ignore[assignment]

    # 1. ASRFinal -> spawns LLM turn task (non-blocking).
    await app._dispatch_one(ASRFinal(text="hi", duplicate_of_streamed=False))
    await asyncio.wait_for(llm_started.wait(), timeout=1.0)

    # 2. While LLM is "running", dispatch a TTSAudio -- must reach speaker.
    await app._dispatch_one(TTSAudio(pcm=b"\x01\x00" * 8, sample_rate=24000))
    seen_during_llm["audio"] = len(app.audio.played) == 1

    # 3. While LLM is "running", dispatch ASRPartial -- must trigger abort.
    app.audio.is_playing = True
    await app._dispatch_one(ASRPartial(text="wait"))
    seen_during_llm["abort"] = app.slv.aborted == 1

    assert not app._llm_turn_task.done(), "LLM turn finished prematurely"
    assert seen_during_llm["audio"], "TTSAudio not played while LLM streaming"
    assert seen_during_llm["abort"], "ASRPartial did not trigger abort while LLM streaming"

    llm_release.set()
    await asyncio.wait_for(app._llm_turn_task, timeout=1.0)
