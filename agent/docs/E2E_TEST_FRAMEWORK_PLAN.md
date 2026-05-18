# OpenVoiceStream Agent — E2E Test Framework Plan

This document records the current state of the agent layer and defines
an autonomous end-to-end test framework so iteration on bugs no longer
needs a human at the mic.

## 1. Current state snapshot (2026-05-17)

### What's built (15 commits, ~3000 lines under `agent/`)

| Layer | Status | Where |
|---|---|---|
| Plugin / EventBus / ConvState | done | `plugin.py`, `event_bus.py`, `state.py` |
| SLVClient (v2v WS, reconnect, asr_eos) | done | `slv_client.py` |
| AudioIO (sounddevice, half-duplex mic) | done | `audio_io.py` |
| LLMBackend (OpenAI-compat + EdgeLLM cache extras) | done | `llm/` |
| BaseApp orchestrator (dispatch, state machine, stop intent, barge-in) | done | `app_base.py` |
| DialogueApp (LLM token → SLV text → TTS) | done | `apps/dialogue/app.py` |
| Client-side VAD (silero + energy) | done | `vad.py` |
| Debug Dashboard v2 (state pill, latency, mic chart, controls, tabs) | done | `plugins/debug_dashboard.py`, `static/` |
| Self-driven WAV-replay test | done | `tests/test_e2e_orin.py` |

### Verified working (against orin-nx SLV + edge-llm)

- ✅ Single utterance: ASR → LLM → TTS roundtrip (~1-2s total)
- ✅ Multi-turn (2+ turns) with session history preserved
- ✅ SLVClient auto-reconnect after each `asr_eos` (SLV closes WS by design)
- ✅ Client VAD speech_min=200ms / silence=600ms triggers correctly
- ✅ Empty `asr_final` filtered, state recovers to IDLE
- ✅ Stop intent ("停下来" / "stop") cancels LLM + aborts TTS
- ✅ Dashboard chat/events/history tabs, state pill in Chinese
- ✅ Bidirectional WS keeps alive under half-duplex (idle silence not transmitted)

### Known gaps (need autonomous verification)

1. **Barge-in correctness under sustained TTS** — code paths in
   place (LLM cancel + slv.abort + stop_playback + first_tts reset)
   but the user can't reliably trigger because of:
   - ~800ms physical floor (200ms VAD speech_min + 610ms Paraformer
     first-decode). Short interjections never fire barge-in.
   - Acoustic feedback when not wearing headphones causes false barge-ins.
   We've never observed `BARGE-IN fired` in a log against a real long-TTS
   scenario without a human; need scripted verification.

2. **Cache hit metrics** — confirmed edge-llm doesn't emit
   `cache_metrics` in streaming mode (only non-streaming). Dashboard
   correctly shows "N/A (streaming)". Not a fixable agent bug —
   server-side limitation.

3. **State machine completeness** — `_set_state` wired into
   ASRPartial/ASRFinal/TTSAudio/TTSDone/SLVError. Untested in
   isolation:
   - SLVError path during THINKING (does state reset cleanly?)
   - Stop intent during BARGED_IN (does it also reset?)
   - Reconnect mid-LLM (does session history survive?)

4. **Long-session stability** — verified up to ~5 turns. Memory leaks,
   WS connection degradation over 30+ turns or 30+ minutes idle: unknown.

5. **Stop intent edge cases**:
   - "停一下" should NOT match "停" (currently doesn't, but unverified)
   - "stop, please" / "stop." / "stop?" should match (per code, but unverified)

6. **Dashboard correctness** — UI rendering of latency cards, mic RMS
   chart updates, chat bubble ordering, error panel persistence:
   verified visually once, never automated.

7. **Audio output integrity** — TTS PCM is played via sounddevice but
   we've never inspected the actual bytes (is it the right rate? clipped?
   silenced? truncated?).

---

## 2. Autonomous E2E framework design

### Goals

- Run a full conversation scenario from a Python test without any
  human speaking, listening, or clicking.
- Assert on agent state transitions, ASR text, LLM responses, TTS
  audio bytes, dashboard rendering.
- Repeatable (same WAV input → same observable behavior, modulo LLM
  non-determinism which we'll handle via prompt seeding or fuzzy
  asserts on response shape).
- No mock SLV / no mock LLM — talk to the real orin-nx services.
  We're testing the full pipeline.
- Optionally launch the dashboard in headless Chromium and screenshot
  /snapshot it for visual asserts.

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  agent/tests/e2e/                                                │
│                                                                  │
│   conftest.py     ─ pytest fixtures: spawn agent, fake audio    │
│   inject.py       ─ scripted audio source (WAV → AudioIO queue) │
│   sink.py         ─ TTS PCM capture → file or in-memory         │
│   probe.py        ─ subscribe to dashboard WS, await events     │
│   playwright.py   ─ headless dashboard inspection wrappers      │
│   fixtures/                                                      │
│     wav/                                                         │
│       hello.wav           — "你好"                              │
│       weather.wav         — "今天天气怎么样"                    │
│       story_request.wav   — "给我讲一个长一点的故事"            │
│       stop_zh.wav         — "停下来"                            │
│       barge_in.wav        — "等一下我要换个话题" (1.5s)         │
│       silence_5s.wav      — pure silence for VAD bait           │
│     prompts.yaml          — expected LLM response shapes        │
│                                                                  │
│   test_single_turn.py     ─ 1 utterance → 1 response            │
│   test_multi_turn.py      ─ 3 utterances, session history works │
│   test_stop_intent.py     ─ "停下来" → no LLM, state→IDLE       │
│   test_barge_in.py        ─ long TTS, interrupt with WAV        │
│   test_empty_final.py     ─ silence → empty final → state→IDLE  │
│   test_reconnect.py       ─ verify WS recycles each turn        │
│   test_dashboard_ui.py    ─ Playwright: state pill, tabs, cards │
│   test_idle_stability.py  ─ 60s idle → no crash, no spurious   │
└──────────────────────────────────────────────────────────────────┘
```

### Key building block: `ScriptedAudioIO`

Replaces real `AudioIO` for tests. Async iterator that yields PCM
chunks from a list of (delay_ms, wav_path) tuples, and writes
playback PCM to an in-memory `bytearray` (or rolling WAV file).

```python
class ScriptedAudioIO:
    def __init__(self, script: list[tuple[float, str | bytes]]):
        """script = [(delay_ms_before_yield, wav_path or pcm_bytes), ...]"""
        ...
    chunk_ms = 100
    input_sr = 16000

    async def start_capture(self):
        for delay_ms, source in self.script:
            await asyncio.sleep(delay_ms / 1000)
            for chunk in self._wav_chunks(source):
                yield chunk
                await asyncio.sleep(0.1)
        # idle silence forever until shutdown
        while True:
            yield bytes(3200)
            await asyncio.sleep(0.1)

    async def play(self, pcm): self.captured_tts.extend(pcm)
    async def stop_playback(self): self._stopped_at = time.time()
    def mark_playback_done(self): self._is_playing = False
    # ... is_playing, set_output_sample_rate, close
```

Tests construct: `script = [(2000, "hello.wav"), (8000, "weather.wav")]`
("Wait 2s then say 'hello'; then wait 8s and say 'weather'").

### Key building block: `AgentProbe`

Connects to dashboard's `/ws` and accumulates events. Tests `await
probe.wait_event("on_user_utterance", text="你好", timeout=5)` to
synchronize.

```python
class AgentProbe:
    async def connect(self, port: int = 18000): ...
    async def wait_event(self, name: str, timeout: float = 10, **fields) -> dict
    async def wait_state(self, state: str, timeout: float = 10) -> None
    def history(self) -> list[dict]
    def state_transitions(self) -> list[tuple[str, str]]
    def latencies(self) -> dict[str, list[float]]
    def tts_pcm_total_bytes(self) -> int  # via TTSAudioFrame events
```

### Key building block: pytest fixture `agent`

```python
@pytest.fixture
async def agent(scripted_audio, env):
    """Spawn agent process with ScriptedAudioIO, return AgentProbe."""
    cfg = make_test_config(...)
    app = DialogueApp(cfg)
    app.audio = scripted_audio   # monkey-patch
    task = asyncio.create_task(app.run())
    probe = AgentProbe()
    await probe.connect(cfg.metadata["dashboard_port"])
    yield app, probe
    app.request_shutdown()
    await task
    await probe.close()
```

### Playwright integration

Use the Claude `playwright-cli` skill. After agent runs through
scenarios, the test invokes:

```python
def test_dashboard_after_turn(agent, page):
    await drive_turn(agent, "hello.wav")
    snap = page.snapshot()  # ARIA tree
    assert "状态: 待机" in snap
    assert "你好" in snap  # chat bubble
    page.screenshot("/tmp/dashboard_after_turn.png")
```

Playwright runs in headless Chromium against `http://localhost:18000`.
The skill exposes `navigate / screenshot / snapshot / click / fill`
which is all we need.

### Scenario catalog (initial 8 tests)

| # | Test | Asserts |
|---|---|---|
| 1 | single_turn | 1 utterance → on_user_utterance + on_assistant_done; TTS bytes > 0 |
| 2 | multi_turn | 3 utterances → session.history has 6 entries; reconnect count = 3 |
| 3 | stop_intent_zh | "停下来" → on_user_stop_intent fired; state→IDLE; NO on_assistant_token; session.history unchanged |
| 4 | stop_intent_en | "stop" / "stop please" / "stop?" all trigger; "stopwatch" does NOT |
| 5 | barge_in | Inject long story request → wait 1.5s into TTS → inject "等一下我要换话题" → assert on state=barged_in + LLM turn cancelled + TTS bytes stops growing within 200ms of partial |
| 6 | empty_final | Inject 100ms noise burst (sub-VAD-threshold) → asr_final empty → state→IDLE; no LLM call |
| 7 | idle_stability | Inject 60s silence → no exceptions logged; WS still open; state stays IDLE |
| 8 | dashboard_ui | After test 1, Playwright snapshot shows expected chat bubbles + state pill + non-empty latency cards |

### Running

```bash
# Pre-req: orin-nx accessible at 100.82.225.102 with SLV + edge-llm up
cd agent
uv run python -m pytest tests/e2e/ -v --tb=short

# Single scenario
uv run python -m pytest tests/e2e/test_barge_in.py -v

# Visual diff (with Playwright screenshots)
uv run python -m pytest tests/e2e/test_dashboard_ui.py -v
```

Each test takes ~5-30s. Full suite ≤ 5 min.

### Out of scope (Phase 2)

- Mocking orin-nx services for offline CI (needs MockSLV + MockLLM)
- Multi-language LLM response testing (currently zh-only fixtures)
- Performance regression tracking (latency budgets over time)
- Stress test: 100 concurrent turns, memory profiling
- Wake-word integration tests (depends on wake plugin landing first)

---

## 3. Iteration loop after framework lands

```
While bugs remain:
  1. Run `uv run pytest tests/e2e/ -v`
  2. Read failures; identify root cause
  3. Patch code (myself or via general-purpose agent)
  4. Re-run failed scenarios
  5. When all green: commit
```

No human needed in the loop. Browser inspection automated via
Playwright snapshots that I can read directly.
