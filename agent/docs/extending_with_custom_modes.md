# Extending the agent with custom AppModes

The agent's voice pipeline (mic → ASR → LLM → TTS → speaker) is owned
by `BaseApp`. The decision of **what to do** with each recognized user
utterance is owned by an `AppMode`. Modes are the Strategy pattern:
plug a different one in and the same pipeline does a different job —
chat, interpreter, transcribe-only, recipe assistant, language tutor,
guided meditation, ...

Switching modes is runtime — click the mode dropdown in the dashboard
or `POST /api/control/mode {"name": "..."}`.

## 50-line quick-start: RecipeHelper

```python
# my_pkg/recipe_helper.py
from openvoicestream_agent.app_mode import AppMode, ModeContext


class RecipeHelperMode(AppMode):
    name = "recipe"
    display_name = "菜谱助手"
    icon = "🍳"
    description = "step-by-step cooking guide; one step per turn"
    system_prompt = (
        "You are a friendly cooking assistant. The user will name a "
        "dish; reply with one cooking step at a time. Wait for the "
        "user to say 'next' before continuing. Keep replies short."
    )

    async def enter(self, ctx: ModeContext) -> None:
        # Reset state and greet the user.
        ctx.session.history.clear()
        await ctx.run_default_dialogue_turn(
            "用户刚刚切到了菜谱助手模式。请用一句话欢迎并请用户说出他想做的菜。",
            system_prompt_override=self.system_prompt,
        )

    async def on_user_utterance(self, ctx: ModeContext, text: str) -> None:
        # Standard turn — system prompt is inherited from this mode class.
        await ctx.run_default_dialogue_turn(text)
```

Register it from your app subclass:

```python
from apps.multi_mode.app import MultiModeApp
from my_pkg.recipe_helper import RecipeHelperMode


class MyApp(MultiModeApp):
    def __init__(self, config):
        super().__init__(config)
        self.modes.register(RecipeHelperMode())
```

The dashboard's mode dropdown will list it automatically (it reads
`GET /api/modes`).

## Lifecycle of a mode

```
ModeManager.start("chat") → mode.enter(ctx)
                                 │
                user utterance → mode.preprocess_user_text(text)
                                     │ (None → drop)
                                 mode.on_user_utterance(ctx, text)
                                     │
                                 (LLM streams, TTS plays)
                                     │
                                 mode.on_assistant_done(ctx)
                                 ...
ModeManager.switch("interpreter") → mode.exit(ctx)
                                  → new_mode.enter(ctx)
```

All hooks default to no-op except `on_user_utterance`, which is
abstract. Modes never see `BaseApp` directly — they get a fresh
`ModeContext` per call, which exposes only the slots a mode needs:
`config`, `slv`, `llm`, `session`, `audio`, `events`,
`broadcast(hook_name, *args)`.

`ctx.run_default_dialogue_turn(text, system_prompt_override=None)` is
the building block for the standard chat-style turn (append user →
stream LLM tokens to SLV → flush → append assistant). Use it for any
mode whose behavior is "talk to LLM with these specifics".

## System-prompt resolution

When `ctx.run_default_dialogue_turn` builds the LLM messages, it
chooses the system prompt in this order:

1. The `system_prompt_override` argument, if non-None.
2. `config.mode_overrides[<current mode>].system_prompt`, if set.
3. The current mode class's `system_prompt` class attribute, if set.
4. `config.system_prompt` (the global default).

So a deployment can fine-tune individual modes through `config.yaml`
without touching code:

```yaml
mode_overrides:
  chat:
    system_prompt: "你是一个严肃的助手。"
  interpreter:
    system_prompt: "Translate Chinese to fluent English."
```

## When to subclass AppMode vs Plugin

| Question | AppMode | Plugin |
|----------|---------|--------|
| Changes how a user turn is handled? | ✅ | ❌ |
| Adds a side-effect (log to file, push to HA, ...)? | ❌ | ✅ |
| Needs to suppress LLM / TTS? | ✅ | ❌ |
| Wants to react to events without changing flow? | ❌ | ✅ |
| Runtime-switchable from the dashboard? | ✅ | usually no |

A diary plugin that logs every transcription is a `Plugin`. A
"transcribe only" mode that **skips** the LLM call is an `AppMode`.
You can run both at the same time — `TranscribeMode` broadcasts
`on_transcribed` which the diary plugin can subscribe to.

## Interaction with `pipeline_mode` (Task 11)

Modes and pipelines are **orthogonal axes**:

- *Pipeline mode* is about transport — server-side TTS in one WS vs
  client-side hybrid. It controls how bytes flow.
- *App mode* is about behavior — what to say back. It controls what
  the LLM is asked.

A single `pipeline_mode=server` deployment can host all four built-in
`AppMode`s simultaneously. Switching mode never touches the SLV
connection. If you later add a `client_pipeline` mode that pre-buffers
sentences on the agent side, the existing `AppMode`s keep working
unchanged.
