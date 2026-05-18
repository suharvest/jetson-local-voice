"""HTTPWakeSource — wake fired by POST to the dashboard /api/control/wake.

This source needs no extra listener: the debug_dashboard plugin's wake
endpoint already calls ``app.wake(source="dashboard")``. Registering
HTTPWakeSource simply documents the dependency and is a no-op at
runtime — but it's still a Plugin so ``app.plugins`` reflects which
wake transports are enabled (used by tests / observability).
"""
from __future__ import annotations

import logging

from ..wake_source import WakeSource

logger = logging.getLogger(__name__)


class HTTPWakeSource(WakeSource):
    name = "http"

    def setup(self) -> bool:
        return True

    async def start(self) -> None:
        logger.info("HTTPWakeSource active (dashboard /api/control/wake)")
        await super().start()

    async def stop(self) -> None:
        await super().stop()


__all__ = ["HTTPWakeSource"]
