"""Stub WakeSource implementations.

These are not wired by default — they exist as scaffolding so users
can `from openvoicestream_agent.wake_sources import MQTTWakeSource`
and subclass / replace start()/stop() with their concrete transport.

Each stub raises NotImplementedError from start() with a clear
pointer to what it would do, so accidentally registering one without
overriding fails fast at startup rather than silently dropping wakes.
"""
from __future__ import annotations

import logging

from ..wake_source import WakeSource

logger = logging.getLogger(__name__)


class MQTTWakeSource(WakeSource):
    """TODO: subscribe to an MQTT topic and call app.wake() on message.

    Skeleton:
        import asyncio_mqtt
        async with asyncio_mqtt.Client(broker) as client:
            await client.subscribe(topic)
            async with client.messages() as messages:
                async for msg in messages:
                    await self.app.wake(source=self.name)
    """

    name = "mqtt"

    async def start(self) -> None:
        raise NotImplementedError(
            "MQTTWakeSource is a stub — subclass and implement start() to "
            "subscribe to your MQTT broker. See module docstring."
        )


class SerialWakeSource(WakeSource):
    """TODO: poll serial / GPIO for a wake edge.

    Skeleton:
        import serial_asyncio
        reader, _ = await serial_asyncio.open_serial_connection(...)
        while True:
            line = await reader.readline()
            if line.strip() == b"WAKE":
                await self.app.wake(source=self.name)
    """

    name = "serial"

    async def start(self) -> None:
        raise NotImplementedError(
            "SerialWakeSource is a stub — subclass and implement start() to "
            "open your serial / GPIO transport. See module docstring."
        )


class LocalKeywordWakeSource(WakeSource):
    """TODO: pipe mic chunks through silero / Porcupine / openWakeWord ONNX.

    Skeleton:
        kws = SileroVAD.load(model_path)
        async for chunk in self.app.audio.start_capture_tap():
            if kws.detect(chunk):
                await self.app.wake(source=self.name)
    """

    name = "local_keyword"

    async def start(self) -> None:
        raise NotImplementedError(
            "LocalKeywordWakeSource is a stub — subclass and implement "
            "start() to run your wake-word model. See module docstring."
        )


__all__ = [
    "MQTTWakeSource",
    "SerialWakeSource",
    "LocalKeywordWakeSource",
]
