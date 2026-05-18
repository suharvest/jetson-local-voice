# companion_robot

Voice-agent entry point for **embodied** robots (Reachy Mini, etc.).

## What this is

`CompanionRobotApp` is a thin subclass of `MultiModeApp` that adds a few
shared-state slots robot plugins are expected to read or fill in:

| Attribute             | Default          | Filled in by         |
|-----------------------|------------------|----------------------|
| `app.reachy`          | `None`           | `ReachyClientPlugin` |
| `app.head_target_bus` | no-op stub       | `FaceTrackerPlugin`  |
| `app.current_emotion` | `"neutral"`      | `MoodPlugin`         |
| `app.motor_enabled`   | `True`           | Dashboard toggle     |

That is **all** this folder contains. The actual robot wiring (SDK init,
servo calibration, vision pipeline, motion calculators) is *not* part of
`openvoicestream/agent`; it lives in the consuming robot project, which
imports from here and registers its own plugins on top.

## Why migrate

`clawd-reachy-mini` currently ships a ~1700-line `ConversationPlugin`
that re-implements ASR streaming, LLM token streaming, TTS handoff,
state-machine, etc. — duplicating exactly what `openvoicestream-agent`
already does. The migration plan replaces that duplicate work with a
single `CompanionRobotApp(config)` and keeps only the things that are
actually robot-unique (motors, face tracking, emotion mapping).

## Migration plan

### Phase A — skeleton (this commit)

Nothing to do in `clawd-reachy-mini` yet. `CompanionRobotApp` exists as
the future target. You can already run

```bash
uv run ovs-agent run companion_robot
```

and it behaves identically to `multi_mode`.

### Phase B — cutover (separate future task in clawd-reachy-mini)

1. `pip install -e ../openvoicestream/agent` (or add as path dep).
2. Replace `ReachyClawApp` base with `CompanionRobotApp`:

   ```python
   from apps.companion_robot.app import CompanionRobotApp

   class ReachyClawApp(CompanionRobotApp):
       def __init__(self, config):
           super().__init__(config)
           self.register(ReachyClientPlugin(self))
           self.register(MotionPlugin(self))
           self.register(FaceTrackerPlugin(self))
           self.register(RestPlugin(self))
           self.register(VisionClientPlugin(self))
           self.register(DiaryPlugin(self))
           self.register(HealthcheckPlugin(self))
   ```

3. **Delete** these modules (they duplicate openvoicestream-agent):
   `ConversationPlugin`, `STT`, `TTS`, `VAD`, `LLM`, mode/state shim,
   audio-IO, session/history manager.
4. **Keep + adapt** these (they are robot-unique):
   `MotionPlugin`, `FaceTrackerPlugin`, `RestPlugin`,
   `VisionClientPlugin`, `DiaryPlugin`, `HealthcheckPlugin` — convert
   to subclasses of `openvoicestream_agent.plugin.Plugin`. The
   signature change is minor (no return value from `setup`, async
   `start/stop`, lifecycle hooks are async).
5. **Dashboard**: replace the 1775-line `DashboardPlugin` with
   `openvoicestream-agent`'s `DebugDashboardPlugin` + a robot-tab
   extension hook (see Phase B dashboard plan below).

## Phase B dashboard plan

clawd-reachy-mini's robot panels (motors / emotion overlay / vision
feeds / diary timeline) should stay — but as **tabs contributed by
robot plugins**, not as a wholly separate dashboard.

This requires a small enhancement to `debug_dashboard.py`:

```python
# Proposed API on DebugDashboardPlugin
dashboard.add_tab(
    name="motors",
    label="Motors",
    html_url="/plugins/motion/index.html",
    ws_topics=["on_motion_state"],
)
```

Robot plugins register their tabs in `setup()`; the dashboard injects
them into the left-column tab strip and proxies static assets. **This
is out of scope for the current commit** — list it as a follow-up task
("Dashboard plugin-tab extension") before starting Phase B in
`clawd-reachy-mini`.

## What is *not* in scope here

- Actual `reachy-sdk` initialisation
- Motor calibration / safety limits
- MediaPipe vision pipeline
- Mode-to-emotion mapping table
- Diary entity wiring

All of those continue to live in `clawd-reachy-mini`.

## Suggested CompanionMode (future built-in)

A natural fit for `CompanionRobotApp` is a `CompanionMode` that behaves
exactly like `ChatMode` but uses the `on_assistant_token` /
`on_assistant_sentence` hooks to drive motion + emotion overlays:

```python
from openvoicestream_agent.app_mode import AppMode, ModeContext


class CompanionMode(AppMode):
    name = "companion"
    display_name = "陪伴"
    icon = "🤖"
    description = "Chat + motion + emotion overlay"
    system_prompt = (
        "You are a friendly desk companion. Keep replies to 1-2 sentences."
    )

    async def on_user_utterance(self, ctx: ModeContext, text: str) -> None:
        # Forwards directly to the standard streaming turn; per-token
        # motion is driven by broadcast hooks observed by MotionPlugin.
        await ctx.run_default_dialogue_turn(text)

    async def on_assistant_done(self, ctx: ModeContext) -> None:
        # Settle head pose / return to neutral idle wobble.
        bus = getattr(ctx, "head_target_bus", None)
        if bus is not None:
            bus.publish(None)
```

This would be added to `CompanionRobotApp.__init__()` once Phase B
lands and the motion plugin exists to observe the events.
