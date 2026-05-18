"""AgentProbe -- subscribe to dashboard /ws and accumulate events for assertions."""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict


class AgentProbe:
    def __init__(self, port: int) -> None:
        self.port = port
        self.events: list[dict] = []
        self.state_history: list[tuple[str, str]] = []  # (prev, new)
        self.latencies: dict[str, list[float]] = defaultdict(list)
        self.errors: list[dict] = []
        self._ws = None
        self._session = None
        self._task: asyncio.Task | None = None

    async def connect(self, timeout: float = 15.0) -> None:
        import aiohttp

        self._session = aiohttp.ClientSession()
        deadline = time.monotonic() + timeout
        last_exc: Exception | None = None
        delay = 0.1
        while time.monotonic() < deadline:
            try:
                self._ws = await self._session.ws_connect(
                    f"http://127.0.0.1:{self.port}/ws", timeout=2.0
                )
                self._task = asyncio.create_task(self._reader(), name="probe-reader")
                return
            except Exception as e:
                last_exc = e
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 1.0)
        raise TimeoutError(
            f"AgentProbe could not connect to dashboard on port {self.port}: {last_exc!r}"
        )

    async def _reader(self) -> None:
        import aiohttp

        try:
            async for msg in self._ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    evt = json.loads(msg.data)
                except Exception:
                    continue
                self.events.append(evt)
                ev_name = evt.get("event")
                data = evt.get("data") or {}
                if ev_name == "on_state_change" and isinstance(data, dict):
                    self.state_history.append(
                        (str(data.get("prev")), str(data.get("state")))
                    )
                elif ev_name == "snapshot" and isinstance(data, dict):
                    s = data.get("state")
                    if s is not None:
                        self.state_history.append(("(snapshot)", str(s)))
                elif ev_name == "latency" and isinstance(data, dict):
                    kind = data.get("kind")
                    ms = data.get("ms")
                    if kind is not None and ms is not None:
                        self.latencies[str(kind)].append(float(ms))
                elif ev_name in ("on_error", "error"):
                    self.errors.append(evt)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def wait_event(self, name: str, timeout: float = 15.0, **fields) -> dict:
        """Wait until an event with event==name and matching data fields arrives."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for e in self.events:
                if e.get("event") != name:
                    continue
                d = e.get("data") if isinstance(e.get("data"), dict) else {}
                ok = True
                for k, v in fields.items():
                    if (d or {}).get(k) != v and e.get(k) != v:
                        ok = False
                        break
                if ok:
                    return e
            await asyncio.sleep(0.05)
        raise TimeoutError(
            f"wait_event({name!r}, {fields}) timed out after {timeout}s; "
            f"saw events: {[e.get('event') for e in self.events][-30:]}"
        )

    async def wait_state(self, state: str, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for _, s in self.state_history:
                if s == state:
                    return
            await asyncio.sleep(0.05)
        raise TimeoutError(
            f"wait_state({state!r}) timed out after {timeout}s; "
            f"history: {self.state_history}"
        )

    def assistant_tokens(self) -> list[str]:
        out: list[str] = []
        for e in self.events:
            if e.get("event") == "on_assistant_token":
                d = e.get("data")
                if isinstance(d, str):
                    out.append(d)
        return out

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None


__all__ = ["AgentProbe"]
