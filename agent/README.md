# openvoicestream-agent

Agent-layer client on top of [OpenVoiceStream (SLV)](../README.md).
SLV provides `/v2v/stream` (ASR + TTS + VAD + barge-in) — this package
adds the LLM, the session, the plugin system, and the audio I/O.

## Architecture

```
+----------+   PCM       +-----------+   text     +---------+
|   mic    | -----------> |    SLV    | <--------- |  LLM    |
+----------+              |  /v2v/    |  tokens    | (edge-  |
                          |  stream   | ---------> |  llm)   |
+----------+   PCM        |           |            +---------+
| speaker  | <----------- |           |
+----------+              +-----------+
```

ONE persistent WebSocket to SLV per App lifetime. SLV does the
ASR / VAD / sentence splitting / TTS server-side. The agent only
orchestrates LLM streaming and barge-in.

## Hard Invariants

1. **Single persistent WS** to `/v2v/stream`, opened with
   `multi_utterance: true`. NOT a new connection per turn.
2. **LLM tokens go DIRECTLY to SLV** as `text` frames. The agent does
   NOT do client-side sentence buffering — SLV's `SentenceBuffer` runs
   server-side and is the single source of truth.
3. **Session history is sent FULL** to the LLM, no client-side
   trimming or summarization. Edge-LLM's prefix cache is the
   optimization.
4. **Barge-in**: on `asr_partial` while TTS is playing, send
   `{"type":"abort"}` to SLV and drain the local playback queue.
5. **Plugin hooks are observer broadcasts**, not routers.
   `MultiModeApp.on_user_utterance` is the single router; modes decide
   behavior behind that stable entrypoint.
6. **Protocol constants come from `app.core.v2v`** (SLV's module).
   Never redeclare them in the agent.
7. **Prefix-cache metrics are streaming-safe**. The EdgeLLM backend asks
   for `return_cache_metrics` on every turn and the dashboard updates
   when the final SSE chunk carries `cache_metrics`.

## Layout

```
agent/
├── pyproject.toml
├── openvoicestream_agent/
│   ├── __init__.py        # sys.path shim so `from app.core.v2v ...` resolves
│   ├── app_base.py        # BaseApp orchestrator
│   ├── app_mode.py        # AppMode / ModeManager strategy framework
│   ├── audio_io.py        # sounddevice mic + speaker
│   ├── cli.py             # `ovs-agent run [app]`
│   ├── config.py          # YAML loader, env var substitution
│   ├── event_bus.py       # pub/sub
│   ├── modes/             # chat / interpreter / monologue / transcribe
│   ├── plugin.py          # Plugin ABC with observer hooks
│   ├── session.py         # OpenAI-format history (no trimming)
│   ├── slv_client.py      # persistent WS to /v2v/stream
│   └── llm/
│       ├── base.py
│       ├── openai_compat.py
│       └── edge_llm.py    # prefix-cache flags + streaming cache metrics
└── apps/
    ├── multi_mode/
        ├── app.py         # MultiModeApp -- runtime-switchable voice modes
        └── config.yaml
    └── companion_robot/
        ├── app.py         # CompanionRobotApp -- robot-oriented scaffold
        └── config.yaml
```

## Quick start

Prereqs:

- SLV running locally with `/v2v/stream` exposed (default `ws://localhost:8621`)
- edge-llm-chat-service running at `http://localhost:8000/v1` (OpenAI compatible)

```bash
cd agent
uv sync
uv run pytest tests/ -v
uv run ovs-agent run
```

Env overrides:

```bash
export OVS_SLV_URL="ws://192.168.1.100:8621/v2v/stream"
export OVS_LLM_URL="http://192.168.1.100:8000/v1"
export OVS_LLM_MODEL="qwen2.5-3b-instruct"
uv run ovs-agent run multi_mode
```

## Built-In Modes

`MultiModeApp` is the default app. It keeps one SLV connection open and
switches behavior through AppMode strategies:

| Mode | Purpose |
|---|---|
| `chat` | Normal multi-turn voice chat. |
| `interpreter` | Stateless Chinese-to-English interpretation. |
| `monologue` | Periodic proactive speech, ignoring user turns. |
| `transcribe` | ASR pass-through only; no LLM and no TTS. |

Set `default_mode` in `apps/multi_mode/config.yaml`, or switch at runtime
from the debug dashboard's mode menu.

The dashboard can also edit the current mode's `system_prompt` and
`temperature`. Overrides take effect on the next turn and are written back to
the app YAML when the app was started from a config file.

## Companion Robot App

`CompanionRobotApp` is a product scaffold for embodied voice projects. It keeps
the same SLV streaming pipeline and AppMode lifecycle, then adds a stable place
for robot-facing plugins such as wake control, pose/action dispatch, or device
telemetry.

```bash
cd agent
uv run ovs-agent run companion_robot
```

Use this when the downstream product is a robot-style assistant and should keep
project-specific wiring out of the generic `multi_mode` app.

## Why the `sys.path` shim?

SLV has no `pyproject.toml`, so we can't `pip install` it.
`openvoicestream_agent/__init__.py` prepends the SLV repo root to
`sys.path` at import time so `from app.core.v2v import CLIENT_TEXT, ...`
works without restructuring SLV. The Docker image accomplishes the same
thing by copying SLV's `app/` directory next to the agent and setting
`PYTHONPATH=/opt/slv`.

## Writing a plugin

```python
from openvoicestream_agent import Plugin

class LoggerPlugin(Plugin):
    name = "logger"

    async def on_user_utterance(self, text: str) -> None:
        print(f"user said: {text}")

    async def on_assistant_sentence(self, sentence: str) -> None:
        print(f"assistant said: {sentence}")
```

Register before `app.run()`:

```python
app = DialogueApp(config)
app.register(LoggerPlugin(app))
await app.run()
```

## Writing a new App

For most behavior changes, create an `AppMode` and register it with
`MultiModeApp`:

```python
from openvoicestream_agent.app_mode import AppMode, ModeContext

class MyMode(AppMode):
    name = "my_mode"
    display_name = "My Mode"

    async def on_user_utterance(self, ctx: ModeContext, text: str) -> None:
        await ctx.run_default_dialogue_turn(text)
```

For a completely different app, subclass `BaseApp`, drop a `config.yaml`
next to `app.py` under `apps/<name>/`, and run with `ovs-agent run <name>`.

See [`docs/extending_with_custom_modes.md`](docs/extending_with_custom_modes.md)
for the full AppMode lifecycle, override order, and a custom mode example.
