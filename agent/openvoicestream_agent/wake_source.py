"""WakeSource plugin ABC for pipeline_mode = wake_word / push_to_talk.

A WakeSource is a Plugin that listens to some external signal (HTTP
endpoint, MQTT topic, GPIO edge, ONNX keyword spotter, …) and calls
``self.app.wake(source=self.name)`` when the wake condition is met.

Subclasses override start() to begin listening and stop() to tear down.
The base class is intentionally minimal — wake fan-out and state
gating live in BaseApp.wake() / BaseApp.sleep().
"""
from __future__ import annotations

import logging

from .plugin import Plugin

logger = logging.getLogger(__name__)


class WakeSource(Plugin):
    """Base for plugins that fire wake events into the agent.

    Subclasses should:
      1. set ``name`` to a stable identifier (used in on_wake payload).
      2. override start() to open the listener (background task, HTTP
         route registration, MQTT subscribe, GPIO interrupt, etc.).
      3. override stop() to close the listener.
      4. call ``await self.app.wake(source=self.name)`` on detection.
    """

    name = "unnamed_wake"

    async def start(self) -> None:  # pragma: no cover - default no-op
        await super().start()

    async def stop(self) -> None:  # pragma: no cover - default no-op
        await super().stop()


__all__ = ["WakeSource"]
