"""E2E: per-mode system_prompt override applied to the live agent.

POST /api/modes/chat/overrides → drive one ASR turn → assert the LLM
was invoked with the new system prompt as its first message.
Asserting on the message-list rather than the model's text output makes
the test deterministic (otherwise we'd be testing the LLM's
instruction-following compliance, not the framework).
"""
from __future__ import annotations

import aiohttp
import pytest

from .conftest import WAV_DIR, run_agent
from .fake_audio import ScriptedAudioIO


@pytest.mark.asyncio
async def test_prompt_editor_runtime_override(test_config):
    audio = ScriptedAudioIO([(800, WAV_DIR / "hello.wav")])
    async with run_agent(test_config, audio) as (app, probe):
        port = test_config.metadata["dashboard_port"]
        base = f"http://127.0.0.1:{port}"

        # Capture every (messages, kw) pair the LLM is called with so we
        # can assert on the system prompt actually sent.
        captured: list[list[dict]] = []
        real_stream = app.llm.stream

        async def spy_stream(messages, **kw):
            captured.append(list(messages))
            async for tok in real_stream(messages, **kw):
                yield tok

        app.llm.stream = spy_stream

        # Push a new system_prompt before the first user utterance.
        new_prompt = "TEST_OVERRIDE_PROMPT_FOR_E2E_CHECK"
        async with aiohttp.ClientSession() as s:
            r = await s.post(
                base + "/api/modes/chat/overrides",
                json={"system_prompt": new_prompt},
            )
            assert r.status == 200
            data = await r.json()
        # Persistence is skipped (Config built in code → _source_path None).
        assert data["persisted"] is False
        assert app.config.mode_overrides["chat"]["system_prompt"] == new_prompt
        assert data["effective"]["system_prompt"] == new_prompt

        # Drive the ASR turn.
        await probe.wait_event("on_user_utterance", timeout=20)
        await probe.wait_state("speaking", timeout=25)
        await probe.wait_state("idle", timeout=40)

        # The LLM should have been called at least once with the override
        # as its system message.
        assert captured, "LLM was never invoked"
        sys_msg = captured[0][0]
        assert sys_msg["role"] == "system"
        assert sys_msg["content"] == new_prompt, (
            f"expected system_prompt override to be sent; got: {sys_msg!r}"
        )
