"""Mode switching: chat → transcribe → chat round-trip via dashboard API.

This validates the AppMode framework end-to-end against the real
remote SLV + LLM stack. Drives:
  1. Default chat mode → standard utterance produces TTS + history.
  2. POST /api/control/mode {name:"transcribe"} → wait on_mode_change.
  3. New utterance → NO TTS, NO LLM call, but on_transcribed event fires.
  4. POST /api/control/mode {name:"chat"} → standard turn behavior again.
"""
from __future__ import annotations

import asyncio

import aiohttp
import pytest

from .conftest import run_agent, WAV_DIR
from .fake_audio import ScriptedAudioIO


async def _switch_mode(port: int, name: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://127.0.0.1:{port}/api/control/mode",
            json={"name": name},
        ) as r:
            return await r.json()


async def _list_modes(port: int) -> list[dict]:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://127.0.0.1:{port}/api/modes") as r:
            return await r.json()


@pytest.mark.asyncio
async def test_mode_switch_chat_transcribe_chat(test_config):
    # Three identical short utterances; behavior differs by current mode.
    audio = ScriptedAudioIO([
        (800,  WAV_DIR / "hello.wav"),
        (8000, WAV_DIR / "hello.wav"),
        (8000, WAV_DIR / "hello.wav"),
    ])
    port = test_config.metadata["dashboard_port"]

    async with run_agent(test_config, audio) as (app, probe):
        # ── Sanity: /api/modes lists all four built-ins, chat current ──
        modes = await _list_modes(port)
        names = {m["name"] for m in modes}
        assert {"chat", "interpreter", "monologue", "transcribe"} <= names, (
            f"missing built-in modes: got {names}"
        )
        current = [m for m in modes if m["current"]]
        assert current and current[0]["name"] == "chat"

        # Helper: count assistant_done events seen so far.
        def _assistant_done_count() -> int:
            return sum(1 for e in probe.events if e.get("event") == "on_assistant_done")

        # ── Turn 1: chat mode → standard TTS + history ──
        await probe.wait_event("on_user_utterance", timeout=25)
        # Wait until Turn 1's TTS finishes (on_assistant_done arrives).
        deadline = asyncio.get_event_loop().time() + 35
        while _assistant_done_count() < 1 and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.1)
        assert _assistant_done_count() >= 1, "Turn 1 never finished TTS"
        for _ in range(50):
            if app.session.history and app.session.history[-1]["role"] == "assistant":
                break
            await asyncio.sleep(0.1)
        assert len(audio.captured_tts) > 1000, (
            "chat turn should produce TTS audio"
        )
        chat_history_len = len(app.session.history)
        assert chat_history_len >= 2

        # ── Switch to transcribe ──
        r = await _switch_mode(port, "transcribe")
        assert r.get("ok") is True and r.get("current") == "transcribe"
        evt = await probe.wait_event("on_mode_change", timeout=5)
        assert (evt.get("data") or {}).get("name") == "transcribe"

        # ── Turn 2: transcribe mode → no LLM, no TTS, on_transcribed event ──
        # Snapshot AFTER mode switch + a small drain pause so any tail
        # TTS bytes from Turn 1 have flushed through audio.play().
        await asyncio.sleep(1.0)
        tts_bytes_before = len(audio.captured_tts)
        history_before = list(app.session.history)
        # Wait for the next user utterance event (script step 1).
        await audio.step_done(1).wait()
        # Allow agent + SLV roundtrip to fire on_user_utterance.
        transcribed = await probe.wait_event("on_transcribed", timeout=25)
        assert (transcribed.get("data") or {}).get("text"), (
            "on_transcribed missing text payload"
        )
        # Give SLV time to confirm no TTS arrives.
        await asyncio.sleep(2.0)
        assert len(audio.captured_tts) == tts_bytes_before, (
            f"transcribe mode must NOT produce TTS; "
            f"got {len(audio.captured_tts) - tts_bytes_before} extra bytes"
        )
        # History must not have grown — transcribe doesn't touch it.
        assert app.session.history == history_before, (
            f"transcribe mode must not append to history; "
            f"before={len(history_before)} after={len(app.session.history)}"
        )

        # ── Switch back to chat ──
        r = await _switch_mode(port, "chat")
        assert r.get("ok") is True
        # Wait until an on_mode_change event with name=="chat" appears.
        async def _wait_mode(target: str, timeout: float = 5.0):
            import time as _t
            deadline = _t.monotonic() + timeout
            while _t.monotonic() < deadline:
                for e in probe.events:
                    if e.get("event") == "on_mode_change":
                        d = e.get("data") or {}
                        if d.get("name") == target:
                            return e
                await asyncio.sleep(0.05)
            raise TimeoutError(f"on_mode_change(name={target!r}) not seen")

        evt = await _wait_mode("chat", timeout=5)
        assert (evt.get("data") or {}).get("name") == "chat"

        # ── Turn 3: chat mode again → TTS resumes, history grows ──
        await audio.step_done(2).wait()
        # Wait up to 30s for chat history to grow past where it was at
        # the start of the transcribe section (transcribe didn't grow it).
        for _ in range(300):
            if len(app.session.history) > chat_history_len:
                break
            await asyncio.sleep(0.1)
        assert len(app.session.history) > chat_history_len, (
            "chat mode resumed should append history"
        )
        # Wait for new TTS bytes to arrive (LLM may take a beat after
        # history append before tokens flush all the way to TTS sink).
        for _ in range(150):
            if len(audio.captured_tts) > tts_bytes_before:
                break
            await asyncio.sleep(0.1)
        assert len(audio.captured_tts) > tts_bytes_before, (
            "chat mode resumed should produce more TTS"
        )
