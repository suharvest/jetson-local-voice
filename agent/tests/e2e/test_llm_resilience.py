"""A7 — end-to-end LLM-failure-resilience tests.

These tests verify the *combined* behavior of the agent-side hardening
landed in A1-A6 (token-aware trim, A3 retry + SSE-error detect, the
LLMAvailability state machine, A4 prefix-cache fallback, A6 dashboard
integration) when the real LLM upstream misbehaves.

Design choices that keep this CI-friendly:

* We start a tiny aiohttp **mock LLM server** (`mock_llm_server.py`) that
  speaks the OpenAI streaming dialect and accepts scripted scenarios.
* We bypass SLV entirely — there is no audio capture / playback in these
  tests. Instead we instantiate ``MultiModeApp`` with stubbed
  ``slv`` / ``audio`` and drive a turn by calling
  ``app.on_user_utterance(text)`` directly. This is the same code path
  the dispatch loop would take, minus the speech recognition front-end.
* The ``DebugDashboardPlugin`` runs as in production so we can probe its
  WS contract using ``AgentProbe`` (same fixture used by the rest of the
  e2e suite).
"""
from __future__ import annotations

import asyncio
import socket
import time
from contextlib import asynccontextmanager
from typing import Any

import pytest
import pytest_asyncio

from .mock_llm_server import MockLLMServer
from .probe import AgentProbe


# ── stub slv / audio ────────────────────────────────────────────────


class _StubSLV:
    """Records what the mode layer tries to push to SLV — no transport."""

    def __init__(self) -> None:
        self.sent_text: list[str] = []
        self.flush_calls: int = 0
        self.abort_calls: int = 0
        self.closed: bool = False
        # The dispatch loop reads ``slv.events()`` — return an iterator
        # that yields nothing then sleeps forever. BaseApp.run() never
        # runs in these tests so this is purely a safety net.
        self._stop = asyncio.Event()

    async def connect(self) -> None: pass
    async def reconnect(self) -> None: pass
    async def close(self) -> None: self.closed = True
    async def send_audio(self, pcm: bytes) -> None: pass
    async def send_text(self, text: str) -> None: self.sent_text.append(text)
    async def flush_tts(self) -> None: self.flush_calls += 1
    async def abort(self) -> None: self.abort_calls += 1
    async def asr_eos(self) -> None: pass

    async def events(self):  # pragma: no cover - never iterated in tests
        await self._stop.wait()
        if False:
            yield None


class _StubAudio:
    """No-op AudioIO replacement — never opens a device."""

    def __init__(self) -> None:
        self.is_playing = False
        self.chunk_ms = 100

    def set_output_sample_rate(self, sr: int) -> None: pass
    def mark_playback_done(self) -> None: self.is_playing = False
    async def play(self, pcm: bytes) -> None: pass
    async def stop_playback(self) -> None: self.is_playing = False
    async def close(self) -> None: pass
    async def start_capture(self):  # pragma: no cover - never iterated
        if False:
            yield b""


# ── harness ─────────────────────────────────────────────────────────


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _make_config(
    llm_base_url: str,
    dashboard_port: int,
    *,
    availability_enabled: bool = True,
):
    from openvoicestream_agent.config import Config, _default_slv_config
    slv_cfg = _default_slv_config()
    slv_cfg.update({"vad": "none", "asr_language": "auto"})
    llm_base_url = llm_base_url.rstrip("/")
    if not llm_base_url.endswith("/v1"):
        llm_base_url = f"{llm_base_url}/v1"
    return Config(
        slv_url="ws://127.0.0.1:1/disabled",
        slv_config=slv_cfg,
        llm_backend="edge_llm",
        llm_base_url=llm_base_url,
        llm_api_key="EMPTY",
        llm_model="mock",
        system_prompt="you are a mock assistant.",
        audio_input_sample_rate=16000,
        audio_output_sample_rate=24000,
        client_vad_backend="off",
        # Make probe fast for tests.
        llm_availability_enabled=availability_enabled,
        llm_availability_probe_interval_s=0.25,
        llm_availability_probe_timeout_s=1.5,
        llm_availability_failures_to_down=3,
        # Keep first-token timeout small so fail-fast assertions are tight.
        llm_first_token_timeout_s=2.0,
        llm_stream_idle_timeout_s=3.0,
        # No transparent retries by default; individual tests opt in.
        llm_retry_on_transient=1,
        llm_retry_backoff_s=0.05,
        metadata={"dashboard_port": dashboard_port},
    )


@asynccontextmanager
async def run_mock_agent(llm_base_url: str, *, availability_enabled: bool = True):
    """Spin up MultiModeApp + DebugDashboardPlugin + LLMAvailabilityPlugin
    without ever touching SLV / audio / signal handlers.
    """
    from apps.multi_mode.app import MultiModeApp

    port = _free_port()
    cfg = _make_config(
        llm_base_url,
        port,
        availability_enabled=availability_enabled,
    )
    app = MultiModeApp(cfg)

    # Inject stubs *before* run-time so plugins / modes see them.
    app.slv = _StubSLV()
    app.audio = _StubAudio()

    # Spin up plugins by hand (skip BaseApp.run — it would try to connect
    # to SLV + register signal handlers).
    started: list[Any] = []
    for p in app.plugins:
        setup = getattr(p, "setup", None)
        if callable(setup) and not setup():
            continue
        try:
            await p.start()
            started.append(p)
        except Exception:
            pass

    # Activate default mode (chat).
    await app.modes.start("chat")

    probe = AgentProbe(port=port)
    try:
        await probe.connect()
        yield app, probe
    finally:
        try:
            await probe.close()
        except Exception:
            pass
        # Stop plugins in reverse order.
        for p in reversed(started):
            try:
                await p.stop()
            except Exception:
                pass
        try:
            await app.llm.aclose()
        except Exception:
            pass


# ── helpers ─────────────────────────────────────────────────────────


async def _wait_state_value(app, value: str, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        avail = getattr(app, "llm_availability", None)
        if avail is not None and avail.state.value == value:
            return
        await asyncio.sleep(0.05)
    cur = (getattr(app, "llm_availability", None) and
           app.llm_availability.state.value)
    raise AssertionError(
        f"LLMAvailability never reached state={value!r}; last={cur!r}"
    )


# ── tests ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_down_before_request_fails_fast() -> None:
    """Scenario 1 — LLM has been DOWN before the user spoke.

    The probe should detect DOWN within a few probe intervals, and the
    user turn should fail-fast (< 0.5s, not waiting for the 2s first-token
    timeout).
    """
    server = MockLLMServer()
    base = await server.start()
    await server.stop()  # immediately close — connection refused

    async with run_mock_agent(base) as (app, probe):
        # Wait for probe state machine to reach DOWN (needs 3 fails).
        await _wait_state_value(app, "down", timeout=6.0)

        t0 = time.monotonic()
        await app._run_user_utterance("hello")
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, (
            f"fail-fast violated: turn took {elapsed*1000:.0f}ms, "
            f"expected <500ms"
        )

        err_evt = await probe.wait_event("on_error", timeout=2.0)
        msg = err_evt.get("data") or err_evt.get("message") or ""
        # on_error broadcasts the exception's str() — RuntimeError("LLM 不可用...")
        flat = str(msg)
        assert "不可用" in flat or "LLM" in flat, (
            f"on_error payload missing LLM-unavailable signal: {err_evt!r}"
        )

        # state must come to rest at IDLE
        from openvoicestream_agent.state import ConvState
        assert app._state == ConvState.IDLE

        # Print debug timeline for EVIDENCE.
        print(
            f"\n[scenario1] probe DOWN detected; "
            f"user_turn fail-fast in {elapsed*1000:.1f}ms "
            f"(timeout config=2000ms)"
        )


@pytest.mark.asyncio
async def test_mid_stream_fail_then_retry_succeeds() -> None:
    """Scenario 2 — 502 before first token, then success.

    A3's retry policy says: if NO tokens have been yielded yet, transparently
    retry. Caller should see a clean stream; mock should record 2 requests
    and no on_error should be broadcast.
    """
    server = MockLLMServer()
    base = await server.start()
    try:
        server.enqueue_502()
        server.enqueue_success(tokens=["hel", "lo"])

        async with run_mock_agent(base) as (app, probe):
            await app._run_user_utterance("hi there")

            # SLV stub should have received both tokens.
            assert app.slv.sent_text == ["hel", "lo"], app.slv.sent_text
            # Two upstream attempts.
            assert len(server.chat_requests) == 2, server.chat_requests
            # No error broadcast.
            assert not probe.errors, [e for e in probe.errors]
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_sse_error_frame_propagates_without_retry() -> None:
    """Scenario 3 — finish_reason=error mid-stream → no retry, error to UI."""
    server = MockLLMServer()
    base = await server.start()
    try:
        server.enqueue_finish_reason_error(n_tokens_before=3)
        # Even if A3 attempted a retry (it must not) a second success would
        # leak through — queue a second success so the assertion is sharp.
        server.enqueue_success(tokens=["should_not_appear"])

        async with run_mock_agent(base, availability_enabled=False) as (app, probe):
            await app._run_user_utterance("tell me a story")

            # Mid-stream error must NOT be retried.
            assert len(server.chat_requests) == 1, (
                f"A3 retried mid-stream error: "
                f"{len(server.chat_requests)} requests"
            )
            # User got the partial tokens.
            assert app.slv.sent_text == ["t0", "t1", "t2"], app.slv.sent_text
            # And an on_error event was broadcast.
            err_evt = await probe.wait_event("on_error", timeout=2.0)
            payload = err_evt.get("data") or err_evt.get("message") or ""
            flat = str(payload)
            assert ("LLM" in flat or "LLMStreamError" in flat
                    or "调用失败" in flat or "error" in flat.lower()), (
                f"unexpected on_error payload: {err_evt!r}"
            )

            from openvoicestream_agent.state import ConvState
            assert app._state == ConvState.IDLE
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_models_endpoint_green_but_chat_red() -> None:
    """Scenario 4 — /v1/models 200 but /v1/chat/completions always 500.

    The probe must NOT be tricked into HEALTHY by the metadata endpoint —
    it pings real inference. After ``failures_to_down`` 500s it should
    advance to DOWN.
    """
    server = MockLLMServer()
    server.models_endpoint_always_ok = True
    server.probes_always_500 = True
    base = await server.start()
    try:
        # Scenarios cover *real* user turns; probes are served by the
        # dedicated probe path (returning 500 because probes_always_500).
        for _ in range(5):
            server.enqueue_500()

        async with run_mock_agent(base) as (app, probe):
            await _wait_state_value(app, "down", timeout=6.0)

            # Sanity: /v1/models endpoint really is green.
            import httpx
            async with httpx.AsyncClient(timeout=2.0) as cli:
                r = await cli.get(f"{base}/v1/models")
                assert r.status_code == 200, r.text

            # User turn should fail-fast now.
            t0 = time.monotonic()
            await app._run_user_utterance("hello")
            elapsed = time.monotonic() - t0
            assert elapsed < 0.5, (
                f"fail-fast violated post-DOWN: {elapsed*1000:.0f}ms"
            )

            print(
                f"\n[scenario4] /v1/models stayed 200 OK, but "
                f"availability correctly reached DOWN via real inference "
                f"probe; fail-fast={elapsed*1000:.1f}ms"
            )
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_late_dashboard_client_sees_current_down_state() -> None:
    """Scenario 5 — A late-connecting dashboard ws receives the current
    DOWN state in its initial snapshot, not the default healthy."""
    server = MockLLMServer()
    base = await server.start()
    await server.stop()

    from apps.multi_mode.app import MultiModeApp

    port = _free_port()
    cfg = _make_config(base, port)
    app = MultiModeApp(cfg)
    app.slv = _StubSLV()
    app.audio = _StubAudio()

    started: list[Any] = []
    for p in app.plugins:
        try:
            await p.start()
            started.append(p)
        except Exception:
            pass
    await app.modes.start("chat")

    try:
        await _wait_state_value(app, "down", timeout=6.0)

        # Now connect the dashboard ws.
        probe = AgentProbe(port=port)
        await probe.connect()
        try:
            snap = await probe.wait_event("snapshot", timeout=3.0)
        finally:
            await probe.close()

        data = snap.get("data") or {}
        avail = data.get("llm_availability")
        assert avail is not None, snap
        assert avail.get("state") == "down", avail
    finally:
        for p in reversed(started):
            try:
                await p.stop()
            except Exception:
                pass
        try:
            await app.llm.aclose()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_prefix_cache_disabled_persists_across_turns() -> None:
    """Scenario 6 — A4 prefix_cache fallback latches across multiple turns.

    Turn 1: cold (no prefix_cache flag) → success → cache_warmed=True
    Turn 2: warm → request *will* carry prefix_cache → mock rejects → A4
            retries without it, latches session.prefix_cache_disabled=True
    Turn 3: warm + disabled → must NOT carry prefix_cache → success
    """
    server = MockLLMServer()
    base = await server.start()
    try:
        # Turn 1 (cold): plain success.
        server.enqueue_success(tokens=["a"])
        # Turn 2 (warm): the 400-when-prefix-cache scenario auto-handles
        # both the failed prefix_cache attempt AND the fallback retry —
        # it always returns success when prefix_cache is absent. So one
        # scenario covers both inbound requests by being applied twice.
        server.enqueue_400_prefix_cache()
        server.enqueue_400_prefix_cache()
        # Turn 3 (warm + disabled): one more success.
        server.enqueue_400_prefix_cache()

        async with run_mock_agent(base) as (app, probe):
            # Turn 1
            await app._run_user_utterance("turn 1")
            assert app.session.cache_warmed is True
            assert app.session.prefix_cache_disabled is False
            req1 = server.chat_requests[0]
            # Cold path uses save_system_prompt_kv_cache, NOT prefix_cache.
            assert _has_prefix_cache(req1) is False, req1

            # Turn 2
            await app._run_user_utterance("turn 2")
            # We should have seen two requests for turn 2: first with
            # prefix_cache=True (rejected), second without.
            req2_first = server.chat_requests[1]
            req2_retry = server.chat_requests[2]
            assert _has_prefix_cache(req2_first) is True, req2_first
            assert _has_prefix_cache(req2_retry) is False, req2_retry
            assert app.session.prefix_cache_disabled is True

            # Turn 3 — must skip prefix_cache from the start.
            await app._run_user_utterance("turn 3")
            req3 = server.chat_requests[3]
            assert _has_prefix_cache(req3) is False, req3

            # Dashboard should have seen the on_prefix_cache_disabled relay.
            evt = await probe.wait_event("on_prefix_cache_disabled", timeout=2.0)
            assert evt is not None
    finally:
        await server.stop()


def _has_prefix_cache(req: dict) -> bool:
    """OpenAI client may either flatten extra_body into top-level or send
    it nested — accept either."""
    if "prefix_cache" in req:
        return bool(req["prefix_cache"])
    eb = req.get("extra_body") or {}
    return bool(eb.get("prefix_cache"))
