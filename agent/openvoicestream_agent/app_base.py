"""BaseApp orchestrator -- wires SLV + LLM + Audio + Plugins.

Lifecycle:
  1. `await slv.connect()` (one persistent WS).
  2. Spawn `_mic_pump_task` (mic -> WS binary) and `_slv_dispatch_task`
     (WS events -> hooks / on_user_utterance routing).
  3. Call each registered plugin's `start()`.
  4. Wait on shutdown event.
  5. `shutdown()` reverses everything.

Plugin hook dispatch is parallel via `asyncio.gather(return_exceptions=True)`
so observers don't block one another or the dispatch loop.
"""
from __future__ import annotations

import asyncio
import logging
import signal
from typing import TYPE_CHECKING

from .audio_io import AudioIO
from .config import Config
from .event_bus import EventBus
from .llm import EdgeLLMBackend, LLMBackend, OpenAICompatBackend
from .session import Session
from .slv_client import (
    ASREndpoint,
    ASRFinal,
    ASRPartial,
    SLVClient,
    SLVError,
    TTSAudio,
    TTSDone,
    TTSSentenceDone,
    TTSStarted,
)

if TYPE_CHECKING:
    from .plugin import Plugin

logger = logging.getLogger(__name__)


def _build_llm(config: Config) -> LLMBackend:
    backend = config.llm_backend.lower()
    if backend == "edge_llm":
        return EdgeLLMBackend(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
        )
    if backend in ("openai_compat", "openai"):
        return OpenAICompatBackend(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
        )
    raise ValueError(f"Unknown llm_backend: {config.llm_backend!r}")


class BaseApp:
    """Subclass and implement `on_user_utterance` to define an App."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.events = EventBus()
        self.slv = SLVClient(config.slv_url, config.slv_config)
        self.audio = AudioIO(
            input_device=config.audio_input_device,
            output_device=config.audio_output_device,
            input_sr=config.audio_input_sample_rate,
            output_sr=config.audio_output_sample_rate,
        )
        self.llm: LLMBackend = _build_llm(config)
        self.session = Session(
            locale=str(config.slv_config.get("asr_language", "zh")).lower()[:2]
        )
        self.plugins: list["Plugin"] = []
        self._shutdown_evt: asyncio.Event | None = None
        self._mic_task: asyncio.Task | None = None
        self._dispatch_task: asyncio.Task | None = None
        self._llm_turn_task: asyncio.Task | None = None
        self._first_tts_seen = False

    # ── public API ──────────────────────────────────────────────────

    def register(self, plugin: "Plugin") -> bool:
        if not plugin.setup():
            logger.info("plugin %s setup() returned False -- skipped", plugin.name)
            return False
        self.plugins.append(plugin)
        return True

    async def on_user_utterance(self, text: str) -> None:
        """Subclasses MUST override. Default raises."""
        raise NotImplementedError("Subclass BaseApp and implement on_user_utterance")

    async def run(self) -> None:
        self._shutdown_evt = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown_evt.set)
            except (NotImplementedError, RuntimeError):
                # Windows / non-main thread -- caller is responsible.
                pass

        await self.slv.connect()
        self._mic_task = asyncio.create_task(self._mic_pump(), name="mic-pump")
        self._dispatch_task = asyncio.create_task(self._slv_dispatch(), name="slv-dispatch")

        for p in self.plugins:
            try:
                await p.start()
            except Exception:
                logger.exception("plugin %s start() failed", p.name)

        try:
            await self._shutdown_evt.wait()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        # 0. cancel any in-flight LLM turn
        if self._llm_turn_task is not None and not self._llm_turn_task.done():
            self._llm_turn_task.cancel()
            try:
                await self._llm_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        # 1. stop mic capture
        if self._mic_task is not None:
            self._mic_task.cancel()
        # 2. cancel TTS if any
        if self.audio.is_playing:
            try:
                await self.slv.abort()
            except Exception:  # pragma: no cover
                pass
        # 3. stop plugins in reverse registration order
        for p in reversed(self.plugins):
            try:
                await p.stop()
            except Exception:
                logger.exception("plugin %s stop() failed", p.name)
        # 4. cancel dispatch
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
        for t in (self._mic_task, self._dispatch_task):
            if t is None:
                continue
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        # 5. close transport
        try:
            await self.slv.close()
        except Exception:  # pragma: no cover
            pass
        # 6. drain playback
        try:
            await self.audio.stop_playback()
            await self.audio.close()
        except Exception:  # pragma: no cover
            pass

    def request_shutdown(self) -> None:
        if self._shutdown_evt is not None:
            self._shutdown_evt.set()

    # ── internal pumps ──────────────────────────────────────────────

    async def _mic_pump(self) -> None:
        try:
            async for chunk in self.audio.start_capture():
                await self.slv.send_audio(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("mic pump crashed")

    async def _slv_dispatch(self) -> None:
        try:
            async for evt in self.slv.events():
                try:
                    await self._dispatch_one(evt)
                except Exception:
                    logger.exception("dispatch error on %r", evt)
        except asyncio.CancelledError:
            raise

    async def _dispatch_one(self, evt) -> None:  # noqa: ANN001
        if isinstance(evt, ASRPartial):
            # Barge-in: user spoke while we were playing.
            if self.audio.is_playing:
                try:
                    await self.slv.abort()
                except Exception:  # pragma: no cover
                    pass
                await self.audio.stop_playback()
            await self._broadcast("on_user_partial", evt.text)
            return

        if isinstance(evt, ASREndpoint):
            await self._broadcast("on_user_speech_start")
            return

        if isinstance(evt, ASRFinal):
            if evt.duplicate_of_streamed:
                return
            await self._broadcast("on_user_utterance", evt.text)
            # Spawn the LLM turn as a tracked task so the dispatch loop
            # stays free to handle queued TTSAudio (playback) and
            # ASRPartial (barge-in) while the model streams.
            if self._llm_turn_task is not None and not self._llm_turn_task.done():
                self._llm_turn_task.cancel()
                try:
                    await self._llm_turn_task
                except (asyncio.CancelledError, Exception):
                    pass
            self._llm_turn_task = asyncio.create_task(
                self._run_user_utterance(evt.text), name="llm-turn"
            )
            return

        if isinstance(evt, TTSStarted):
            await self._broadcast("on_assistant_sentence_start", evt.sentence)
            return

        if isinstance(evt, TTSAudio):
            if not self._first_tts_seen:
                self._first_tts_seen = True
                self.audio.set_output_sample_rate(evt.sample_rate)
            await self.audio.play(evt.pcm)
            return

        if isinstance(evt, TTSSentenceDone):
            await self._broadcast("on_assistant_sentence", evt.sentence)
            return

        if isinstance(evt, TTSDone):
            await self._broadcast("on_assistant_done")
            return

        if isinstance(evt, SLVError):
            await self._broadcast("on_error", RuntimeError(evt.message))
            return

    async def _run_user_utterance(self, text: str) -> None:
        """Wrap on_user_utterance so a crashing LLM turn doesn't kill the task silently."""
        try:
            await self.on_user_utterance(text)
        except NotImplementedError:
            logger.error("BaseApp.on_user_utterance not overridden -- text dropped")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("on_user_utterance failed")

    async def _broadcast(self, hook_name: str, *args) -> None:
        if not self.plugins:
            return
        coros = []
        for p in self.plugins:
            fn = getattr(p, hook_name, None)
            if fn is None:
                continue
            coros.append(_safe_call(p.name, hook_name, fn, *args))
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)


async def _safe_call(plugin_name: str, hook: str, fn, *args) -> None:  # noqa: ANN001
    try:
        result = fn(*args)
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        logger.exception("plugin %s.%s failed", plugin_name, hook)


__all__ = ["BaseApp"]
