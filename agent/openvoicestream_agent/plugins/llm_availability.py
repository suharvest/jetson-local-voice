"""LLMAvailability — health probe + circuit breaker as a single state machine.

Combines the responsibilities of a health probe and a circuit breaker into
one component to avoid split-brain (health says green / breaker says OPEN).

State machine:
    HEALTHY ──probe fail──► DEGRADED ──N consecutive fails──► DOWN
       ▲                       │                                │
       │                       └────probe pass──┐               │
       │                                        ▼               │
       └────────────────────────────────── RECOVERING ◄─probe pass
                                                │
                                                └─probe fail──► DOWN

Probes hit `/v1/chat/completions` with `max_tokens=1` — NOT `/v1/models`,
which only returns metadata and would return green even when generation
is broken (api_server.py:86-96).

Probe timeouts return ``None`` (unknown) so a slow LLM doesn't get
mis-classified as DOWN by probe cadence alone.

External failures (a real user request that raised) feed back through
``report_request_failure()`` so the state advances within seconds rather
than waiting for the next probe interval.
"""
from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import TYPE_CHECKING

import httpx

from ..plugin import Plugin

if TYPE_CHECKING:
    from ..app_base import BaseApp

logger = logging.getLogger(__name__)


class AvailabilityState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    RECOVERING = "recovering"
    # MED-3: distinct from DEGRADED/DOWN. Reached when the probe has
    # returned "unknown" (timeout / connect error treated as unknown)
    # for N consecutive cycles — the LLM may be fine, we just can't
    # tell, so the dashboard should NOT show green. We don't transition
    # to DEGRADED here because that would mis-attribute a network
    # partition to the LLM and start counting toward DOWN.
    UNKNOWN = "unknown"


class LLMUnavailable(RuntimeError):
    """Raised when the LLM is in DOWN state. Not an APIError → A3 won't retry."""


class LLMAvailabilityPlugin(Plugin):
    """Continuously probe the LLM and expose a state machine.

    Hook events emitted via ``app.broadcast``:
      ``on_llm_availability_change(data)``
        data = {"state": str, "last_ok_ts": float | None,
                "consecutive_failures": int}
    """

    name = "llm_availability"

    def __init__(
        self,
        app: "BaseApp",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        interval_s: float | None = None,
        probe_timeout_s: float | None = None,
        failures_to_down: int | None = None,
    ) -> None:
        super().__init__(app)
        cfg = app.config
        # Allow explicit overrides (mainly for tests); fall back to config.
        self.base_url = (base_url or cfg.llm_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else cfg.llm_api_key
        self.model_name = model_name or cfg.llm_model
        self.interval_s = float(
            interval_s
            if interval_s is not None
            else getattr(cfg, "llm_availability_probe_interval_s", 30.0)
        )
        self.probe_timeout_s = float(
            probe_timeout_s
            if probe_timeout_s is not None
            else getattr(cfg, "llm_availability_probe_timeout_s", 5.0)
        )
        self.failures_to_down = int(
            failures_to_down
            if failures_to_down is not None
            else getattr(cfg, "llm_availability_failures_to_down", 3)
        )
        # MED-3: how many consecutive "unknown" probe results before we
        # surface UNKNOWN state. Defaults to 3 so a single transient
        # network blip doesn't flip the dashboard.
        self.unknowns_to_unknown_state = int(
            getattr(cfg, "llm_availability_unknowns_to_unknown_state", 3)
        )

        # State.
        self.state: AvailabilityState = AvailabilityState.HEALTHY
        self.consecutive_failures: int = 0
        self.last_ok_ts: float | None = None
        # MED-3: counts consecutive ``None`` probe results (timeout /
        # connect error). Reset on any concrete True/False outcome.
        self.consecutive_unknowns: int = 0

        self._stopped: bool = False
        self._task: asyncio.Task | None = None
        # Wake event for force_probe — sleep() races against this.
        self._wake_evt: asyncio.Event | None = None
        # Serialise probes (force_probe + main loop should not overlap).
        self._probe_lock: asyncio.Lock | None = None

    # ── plugin lifecycle ───────────────────────────────────────────

    def setup(self) -> bool:  # pragma: no cover - trivial
        return bool(getattr(self.app.config, "llm_availability_enabled", True))

    async def start(self) -> None:
        await super().start()
        self._stopped = False
        self._wake_evt = asyncio.Event()
        self._probe_lock = asyncio.Lock()
        # Expose ourselves on the app so app_mode/app_base can read state.
        self.app.llm_availability = self
        self._task = asyncio.create_task(self.run(), name="llm-availability")

    async def stop(self) -> None:
        self._stopped = True
        if self._wake_evt is not None:
            self._wake_evt.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        # Detach from app so post-shutdown code never queries a stale plugin.
        if getattr(self.app, "llm_availability", None) is self:
            self.app.llm_availability = None
        await super().stop()

    # ── probe ──────────────────────────────────────────────────────

    async def _probe(self) -> bool | None:
        """Run a minimal inference request.

        Returns:
            True  — got a 200 with at least one choice (real inference path OK)
            False — got an HTTP error / connection error / 200 with no choices
            None  — probe timed out (treat as unknown, do not advance state)
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            async with httpx.AsyncClient(timeout=self.probe_timeout_s) as client:
                r = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": "."}],
                        "max_tokens": 1,
                        "stream": False,
                    },
                    headers=headers,
                )
                if r.status_code == 200:
                    try:
                        data = r.json()
                    except Exception:
                        logger.warning("LLM probe: 200 but non-JSON body")
                        return False
                    return bool(data.get("choices"))
                logger.warning("LLM probe: HTTP %s", r.status_code)
                return False
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            # Network partition / DNS failure / connect refused — we genuinely
            # don't know if the LLM is up. Treat as unknown, not as a confirmed
            # failure (otherwise a partition gets mis-labeled DOWN and recovery
            # requires double-confirmation we shouldn't need).
            logger.info("LLM probe unreachable (%s: %s); treated as unknown",
                        type(e).__name__, e)
            return None
        except Exception as e:
            logger.warning("LLM probe error: %s", e)
            return False

    # ── main loop ──────────────────────────────────────────────────

    async def run(self) -> None:
        try:
            while not self._stopped:
                async with (self._probe_lock or asyncio.Lock()):
                    result = await self._probe()
                if result is None:
                    # MED-3: track consecutive unknowns. When the threshold
                    # is reached, transition to UNKNOWN — better than
                    # silently staying HEALTHY during a network partition.
                    self.consecutive_unknowns += 1
                    self._maybe_enter_unknown()
                else:
                    self.consecutive_unknowns = 0
                    self._advance(result)
                # Sleep with early-wake support for force_probe.
                if self._wake_evt is not None:
                    try:
                        await asyncio.wait_for(
                            self._wake_evt.wait(), timeout=self.interval_s
                        )
                        self._wake_evt.clear()
                    except asyncio.TimeoutError:
                        pass
                else:  # pragma: no cover - defensive
                    await asyncio.sleep(self.interval_s)
        except asyncio.CancelledError:
            raise

    # ── state machine ──────────────────────────────────────────────

    def _emit_change(self, prev: AvailabilityState) -> None:
        if prev == self.state:
            return
        payload = {
            "state": self.state.value,
            "last_ok_ts": self.last_ok_ts,
            "consecutive_failures": self.consecutive_failures,
        }
        logger.info(
            "LLM availability: %s → %s (failures=%d)",
            prev.value, self.state.value, self.consecutive_failures,
        )
        bus = getattr(self.app, "events", None)
        if bus is not None:
            try:
                bus.emit("on_llm_availability_change", payload)
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            broadcast = getattr(self.app, "broadcast", None)
            if broadcast is not None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(broadcast("on_llm_availability_change", payload))
                except RuntimeError:
                    pass
        except Exception:  # pragma: no cover - defensive
            pass

    def _advance(self, ok: bool) -> None:
        """Advance the state machine using a probe / report result."""
        prev = self.state
        if ok:
            self.last_ok_ts = time.time()
            if prev == AvailabilityState.HEALTHY:
                self.consecutive_failures = 0
            elif prev == AvailabilityState.DEGRADED:
                self.state = AvailabilityState.HEALTHY
                self.consecutive_failures = 0
            elif prev == AvailabilityState.DOWN:
                # One success isn't enough — need confirmation.
                self.state = AvailabilityState.RECOVERING
            elif prev == AvailabilityState.RECOVERING:
                self.state = AvailabilityState.HEALTHY
                self.consecutive_failures = 0
            elif prev == AvailabilityState.UNKNOWN:
                # MED-3: a clean success from UNKNOWN means the partition
                # cleared and the LLM is fine — straight back to HEALTHY.
                self.state = AvailabilityState.HEALTHY
                self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            if prev == AvailabilityState.HEALTHY:
                self.state = AvailabilityState.DEGRADED
            elif prev == AvailabilityState.DEGRADED:
                if self.consecutive_failures >= self.failures_to_down:
                    self.state = AvailabilityState.DOWN
            elif prev == AvailabilityState.RECOVERING:
                self.state = AvailabilityState.DOWN
            elif prev == AvailabilityState.UNKNOWN:
                # MED-3: probe got a concrete failure — the LLM really is
                # broken (not just unreachable). Enter DEGRADED, start the
                # normal failure-count progression to DOWN.
                self.state = AvailabilityState.DEGRADED
            # DOWN + fail → still DOWN.
        self._emit_change(prev)

    def _maybe_enter_unknown(self) -> None:
        """MED-3: transition into UNKNOWN once the consecutive-unknown
        threshold is crossed. We don't disturb DOWN (it's a stronger
        signal — keep it), and we don't transition out of UNKNOWN here
        (that's _advance's job)."""
        if self.consecutive_unknowns < self.unknowns_to_unknown_state:
            return
        if self.state in (AvailabilityState.UNKNOWN, AvailabilityState.DOWN):
            return
        prev = self.state
        self.state = AvailabilityState.UNKNOWN
        logger.warning(
            "LLM probe returned unknown %d times in a row; "
            "transitioning %s → UNKNOWN (network partition or hung server?)",
            self.consecutive_unknowns,
            prev.value,
        )
        self._emit_change(prev)

    # ── external signals ───────────────────────────────────────────

    def report_request_failure(self) -> None:
        """A real user-driven LLM request just failed (post A3 retries).

        Feeds the state machine without waiting for the next probe — a
        single failed turn is more informative than 30s of probe silence.
        """
        self._advance(False)

    def report_request_success(self) -> None:
        """A real user-driven LLM request just succeeded — reset failures."""
        self._advance(True)

    async def force_probe(self) -> bool | None:
        """Run a probe immediately (bypassing the interval).

        Returns the probe result (same tri-state as ``_probe``). Used by
        the dashboard to give operators a manual "ping" button.
        """
        async with (self._probe_lock or asyncio.Lock()):
            result = await self._probe()
        if result is None:
            # MED-3: same accounting as the main loop — operator-driven
            # probes also count toward the unknown threshold.
            self.consecutive_unknowns += 1
            self._maybe_enter_unknown()
        else:
            self.consecutive_unknowns = 0
            self._advance(result)
        # Wake the main loop so it doesn't immediately probe again on
        # top of ours, and so its sleep cycle re-aligns from "now".
        if self._wake_evt is not None:
            self._wake_evt.set()
        return result


__all__ = [
    "AvailabilityState",
    "LLMAvailabilityPlugin",
    "LLMUnavailable",
]
