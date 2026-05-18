"""AppMode: Strategy-pattern extension point for voice apps.

A `BaseApp` owns the pipeline + transport + state machine. An `AppMode`
plugged into it decides what to do with each user utterance: standard
chat, interpreter, monologue, transcribe, recipe assistant, ...

Modes interact with the agent exclusively through `ModeContext` — they
never hold a back-ref to `BaseApp`. This forbids feature-creep and keeps
the surface area testable.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class LLMTimeoutError(RuntimeError):
    """LLM 调用超时（首 token 或流式 idle）。"""

    def __init__(self, kind: str, timeout_s: float, partial_text: str = ""):
        super().__init__(f"LLM {kind} timeout after {timeout_s:.1f}s")
        self.kind = kind  # "first_token" | "stream_idle"
        self.timeout_s = timeout_s
        self.partial_text = partial_text  # 已收到的 token 拼接


@dataclass
class ModeContext:
    """Dependency-injection bundle handed to every Mode method.

    Constructed fresh per call by `MultiModeApp._make_mode_ctx()`, so
    Modes always observe the current state of pipeline objects.
    """

    config: Any
    slv: Any
    llm: Any
    session: Any
    audio: Any
    events: Any
    broadcast: Callable[..., Awaitable[None]]
    # Optional handle to the owning ModeManager so modes can introspect /
    # request a switch (e.g. a "switch to chat" voice command). Set by
    # ModeManager when the context is created.
    mode_manager: "ModeManager | None" = None

    async def run_default_dialogue_turn(
        self,
        text: str,
        system_prompt_override: str | None = None,
    ) -> None:
        """Standard chat turn: append user → stream LLM tokens to SLV →
        flush → append assistant to history.

        Reused by ChatMode, InterpreterMode, MonologueMode self-prompts,
        RecipeHelper, etc.

        System-prompt resolution order:
          1. `system_prompt_override` arg
          2. `config.mode_overrides[current_mode.name].system_prompt`
          3. `current_mode.system_prompt`
          4. `config.system_prompt`
        """
        if not text or not text.strip():
            logger.warning("run_default_dialogue_turn: empty text — skipping")
            return

        # Fail-fast on a known-DOWN LLM. The breaker lives on the app
        # (set by LLMAvailabilityPlugin.start). The ctx doesn't carry an
        # app handle, but the event bus does via its parent — instead we
        # stash the plugin on the ModeContext indirectly through
        # ctx.events._app if present. Simpler: read it off the events bus
        # via a dedicated attribute. The plugin attaches itself to
        # ``app.llm_availability``; ``ctx.events`` has no app pointer.
        # We resolve via the broadcast closure: the broadcast callable is
        # bound to ``app.broadcast`` → its ``__self__`` is the app.
        avail = None
        bc = getattr(self.broadcast, "__self__", None)
        if bc is not None:
            avail = getattr(bc, "llm_availability", None)
        if avail is not None:
            # Local import to avoid a cycle at module load.
            from .plugins.llm_availability import AvailabilityState, LLMUnavailable
            if avail.state == AvailabilityState.DOWN:
                raise LLMUnavailable(
                    f"LLM is DOWN (consecutive failures: "
                    f"{avail.consecutive_failures})"
                )

        system_prompt = self._resolve_system_prompt(system_prompt_override)
        temperature = self._resolve_mode_value("temperature")

        logger.info(
            "run_default_dialogue_turn: text=%r (history=%d msgs, sp_len=%d)",
            text, len(self.session.history) + 1, len(system_prompt),
        )

        self.session.add_user(text)
        chunks: list[str] = []
        first_token_received = False
        try:
            llm_kwargs = {"session": self.session}
            if temperature is not None:
                llm_kwargs["temperature"] = temperature
            cfg = self.config
            first_timeout = float(getattr(cfg, "llm_first_token_timeout_s", 15.0))
            idle_timeout = float(getattr(cfg, "llm_stream_idle_timeout_s", 30.0))

            stream = self.llm.stream(
                self.session.messages(system_prompt),
                **llm_kwargs,
            )
            # Manual async-iteration so we can wrap each __anext__ in
            # asyncio.wait_for and distinguish first-token vs stream-idle
            # timeouts.
            it = stream.__aiter__()
            while True:
                wait_timeout = first_timeout if not first_token_received else idle_timeout
                try:
                    token = await asyncio.wait_for(it.__anext__(), timeout=wait_timeout)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    kind = "first_token" if not first_token_received else "stream_idle"
                    raise LLMTimeoutError(kind, wait_timeout, "".join(chunks))
                first_token_received = True
                chunks.append(token)
                try:
                    self.events.emit("assistant_token", token)
                except Exception:  # pragma: no cover - defensive
                    pass
                await self.broadcast("on_assistant_token", token)
                await self.slv.send_text(token)
        finally:
            try:
                await self.slv.flush_tts()
            except Exception:  # pragma: no cover - best effort
                pass
            if chunks:
                self.session.add_assistant("".join(chunks))
            cm = getattr(self.llm, "last_cache_metrics", None)
            if cm:
                try:
                    await self.broadcast("on_llm_cache_metrics", cm)
                except Exception:  # pragma: no cover - defensive
                    pass

    def _resolve_system_prompt(self, override: str | None) -> str:
        if override is not None:
            return override
        mode_name = None
        if self.mode_manager is not None:
            mode_name = self.mode_manager.current_name
        # Per-mode overrides from config.mode_overrides[<name>].
        # Use `in` + `is not None` (NOT truthiness) so an explicit
        # empty-string override is honoured — users may legitimately
        # want "no system message" for a mode.
        overrides = getattr(self.config, "mode_overrides", None) or {}
        if mode_name and isinstance(overrides, dict):
            mo = overrides.get(mode_name) or {}
            if isinstance(mo, dict) and "system_prompt" in mo and mo["system_prompt"] is not None:
                return str(mo["system_prompt"])
        # Mode-class-declared system prompt (None means inherit;
        # empty string would also be honoured here).
        if self.mode_manager is not None and self.mode_manager._current is not None:
            sp = self.mode_manager._current.system_prompt
            if sp is not None:
                return sp
        # Fall back to global app system_prompt.
        return getattr(self.config, "system_prompt", "") or ""

    def _resolve_mode_value(self, key: str) -> Any:
        """Resolve a non-prompt AppMode value from config override → class default."""
        mode = self.mode_manager._current if self.mode_manager is not None else None
        mode_name = mode.name if mode is not None else None
        overrides = getattr(self.config, "mode_overrides", None) or {}
        if mode_name and isinstance(overrides, dict):
            mo = overrides.get(mode_name) or {}
            if isinstance(mo, dict) and key in mo:
                return mo[key]
        if mode is not None:
            return getattr(mode, key, None)
        return None


class AppMode(ABC):
    """Base class for a voice-app mode (Strategy pattern).

    Subclasses MUST implement `on_user_utterance`. All other hooks have
    no-op defaults and are optional.

    Class-level attributes (`system_prompt`, `temperature`, ...) are
    declarative defaults. They can be overridden per-deployment via
    `config.mode_overrides[<mode_name>] = {...}`.
    """

    name: str = "unnamed"
    display_name: str = "Unnamed"
    icon: str = "•"
    description: str = ""

    # Config overrides (None = inherit from app config).
    system_prompt: str | None = None
    temperature: float | None = None
    max_history: int | None = None
    barge_in_enabled: bool | None = None

    # If False, this mode does NOT produce TTS in response to user
    # utterances. The dispatcher uses this to restore IDLE after a turn
    # so the SPEAKING→IDLE path (which never fires for silent modes)
    # isn't required to advance the FSM. Default True — most modes talk.
    produces_tts: bool = True

    async def enter(self, ctx: ModeContext) -> None:
        """Called when this mode becomes active. Override for setup."""

    async def exit(self, ctx: ModeContext) -> None:
        """Called when this mode is being switched away from."""

    @abstractmethod
    async def on_user_utterance(self, ctx: ModeContext, text: str) -> None:
        """REQUIRED: how to handle the user's recognized text."""

    async def on_assistant_done(self, ctx: ModeContext) -> None:
        """Optional: called after the assistant finished speaking."""

    async def on_session_idle(self, ctx: ModeContext, idle_seconds: float) -> None:
        """Optional: periodic tick during long IDLE. monologue uses this."""

    def preprocess_user_text(self, text: str) -> str | None:
        """Optional: transform/filter ASR text before on_user_utterance.

        Return None to drop the turn entirely (e.g., wake-word filtering).
        """
        return text


class ModeManager:
    """Owns the registry of AppModes and the currently-active mode.

    Constructed with a `ctx_factory` so a fresh ModeContext is created
    for each hook invocation (avoids stale references to session/slv).
    """

    def __init__(self, ctx_factory: Callable[[], ModeContext]) -> None:
        self._ctx_factory = ctx_factory
        self._modes: dict[str, AppMode] = {}
        self._current: AppMode | None = None
        # Set True after start(); used to suppress on_mode_registered
        # broadcasts during initial bulk registration (would spam plugins).
        self._started: bool = False

    def register(self, mode: AppMode) -> None:
        if not isinstance(mode, AppMode):
            raise TypeError(f"register() expected AppMode, got {type(mode).__name__}")
        if not mode.name or mode.name == "unnamed":
            raise ValueError(f"AppMode subclass {type(mode).__name__} must set .name")
        if mode.name in self._modes:
            raise ValueError(f"mode {mode.name!r} already registered")
        self._modes[mode.name] = mode
        logger.info("registered mode %r (%s)", mode.name, mode.display_name)
        # Late registration (after start) → fire-and-forget broadcast so the
        # dashboard can refresh its mode dropdown without polling.
        if self._started:
            payload = {
                "name": mode.name,
                "display_name": mode.display_name,
                "icon": mode.icon,
                "description": mode.description,
            }
            try:
                ctx = self._make_ctx()
            except Exception:
                logger.exception("on_mode_registered: ctx build failed")
                return
            import asyncio as _asyncio
            try:
                loop = _asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                async def _bcast():
                    try:
                        await ctx.broadcast("on_mode_registered", payload)
                    except Exception:
                        logger.exception("on_mode_registered broadcast failed")
                loop.create_task(_bcast())

    @property
    def current(self) -> AppMode:
        if self._current is None:
            raise RuntimeError("No mode active — call start() first")
        return self._current

    @property
    def current_name(self) -> str | None:
        return self._current.name if self._current is not None else None

    def get(self, name: str) -> AppMode | None:
        return self._modes.get(name)

    def list_all(self) -> list[dict]:
        out: list[dict] = []
        cur = self.current_name
        for m in self._modes.values():
            out.append({
                "name": m.name,
                "display_name": m.display_name,
                "icon": m.icon,
                "description": m.description,
                "current": (m.name == cur),
            })
        return out

    def _make_ctx(self) -> ModeContext:
        ctx = self._ctx_factory()
        ctx.mode_manager = self
        return ctx

    async def switch(self, name: str) -> None:
        """Exit current (if any) → set new → enter new. No-op if same."""
        if name not in self._modes:
            raise KeyError(f"unknown mode: {name!r}")
        if self._current is not None and self._current.name == name:
            return
        prev = self._current
        next_mode = self._modes[name]
        ctx = self._make_ctx()
        if prev is not None:
            try:
                await prev.exit(ctx)
            except Exception:
                logger.exception("mode %s exit() failed", prev.name)
        self._current = next_mode
        try:
            await next_mode.enter(ctx)
        except Exception:
            logger.exception("mode %s enter() failed", next_mode.name)
        # Broadcast mode_change hook.
        payload = {
            "name": next_mode.name,
            "display_name": next_mode.display_name,
            "icon": next_mode.icon,
            "prev": prev.name if prev is not None else None,
        }
        try:
            await ctx.broadcast("on_mode_change", payload)
        except Exception:
            logger.exception("on_mode_change broadcast failed")

    async def start(self, default_name: str = "chat") -> None:
        """Initial switch to the default mode."""
        if default_name not in self._modes:
            # Fall back to first registered mode if default missing.
            if not self._modes:
                raise RuntimeError("ModeManager.start: no modes registered")
            default_name = next(iter(self._modes))
            logger.warning(
                "default mode not registered, falling back to %r", default_name
            )
        await self.switch(default_name)
        self._started = True


__all__ = ["AppMode", "ModeContext", "ModeManager", "LLMTimeoutError"]
