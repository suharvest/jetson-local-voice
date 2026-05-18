"""CompanionRobotApp — entry point for embodied voice agents (Reachy, etc.).

Extends `MultiModeApp` with hooks for robot-specific concerns:

- head/face tracking bus
- motion / emotion plugin slots
- companion-specific built-in modes (future: a CompanionMode that overlays
  motion + emotion onto a standard chat turn)

This module is intentionally a *skeleton*. The actual robot wiring (Reachy
SDK init, motion calculators, vision pipeline) lives in the consuming
robot project (e.g. ``clawd-reachy-mini``), which imports from here and
fills in the plugin slots.

See ``README.md`` next to this file for the migration plan.
"""
from __future__ import annotations

from apps.multi_mode.app import MultiModeApp


class _NullHeadTargetBus:
    """Placeholder head-target fusion bus.

    Lets robot plugins always call ``app.head_target_bus.publish(...)`` /
    ``get_fused_target()`` without nullable checks. Replaced at runtime by
    a real fusion bus once a vision / face-tracker plugin registers.
    """

    def publish(self, target) -> None:  # noqa: ANN001
        return None

    def get_fused_target(self):
        return None


class CompanionRobotApp(MultiModeApp):
    """``MultiModeApp`` + shared state slots for robot plugins.

    Robot-specific plugins (``MotionPlugin``, ``FaceTrackerPlugin``,
    ``MoodPlugin`` …) plug in via the standard ``Plugin`` protocol. They
    read/write ``head_target_bus``, ``current_emotion``, ``motor_enabled``
    etc. — no special hook surface required.

    The default plugin set is unchanged from ``MultiModeApp`` (just the
    DebugDashboardPlugin and standard built-in modes). Robot projects are
    expected to subclass and ``self.register(...)`` their motion / vision
    plugins on top.
    """

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        # Reachy / motor handle: set by a robot plugin if present.
        self.reachy = None
        # Shared head-target fusion bus (face / DOA / nudges).
        self.head_target_bus = _NullHeadTargetBus()
        # Current expressed emotion (string slug consumed by motion plugin).
        # Modes can update this from on_assistant_sentence; motion plugin
        # animates accordingly.
        self.current_emotion: str = "neutral"
        # Runtime kill-switch for physical motors (dashboard toggle etc.)
        # without having to mutate the config / restart.
        self.motor_enabled: bool = True


__all__ = ["CompanionRobotApp"]
App = CompanionRobotApp  # for cli.py dynamic loader
