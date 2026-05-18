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
import time
from typing import TYPE_CHECKING


class TypedLLMError(RuntimeError):
    """RuntimeError subclass carrying a structured payload for the dashboard.

    Keeps full backward compatibility with the old on_error contract
    (``isinstance(exc, RuntimeError)`` + ``str(exc)`` continue to work),
    but exposes ``.payload`` so plugins like debug_dashboard can render
    typed/coloured errors instead of opaque strings.
    """

    def __init__(
        self,
        type_: str,
        message: str,
        *,
        exc_class: str = "",
        **extra,
    ) -> None:
        super().__init__(message)
        self.payload: dict = {
            "type": type_,
            "message": message,
            "exc_class": exc_class,
            "timestamp": time.time(),
            **extra,
        }

from .app_mode import LLMTimeoutError
from .audio_io import AudioIO
from .config import Config
from .event_bus import EventBus
from .llm import EdgeLLMBackend, LLMBackend, LLMStreamError, OpenAICompatBackend
from .plugins.llm_availability import LLMUnavailable
from .session import Session
from .state import ConvState
from .vad import create_vad
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
            retry_on_transient=config.llm_retry_on_transient,
            retry_backoff_s=config.llm_retry_backoff_s,
        )
    if backend in ("openai_compat", "openai"):
        return OpenAICompatBackend(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
            retry_on_transient=config.llm_retry_on_transient,
            retry_backoff_s=config.llm_retry_backoff_s,
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
            locale=str(config.slv_config.get("asr_language", "zh")).lower()[:2],
            max_input_tokens=getattr(config, "session_max_input_tokens", None),
            tokenizer_model=getattr(
                config, "session_tokenizer_model", "Qwen/Qwen3-4B-AWQ"
            ),
            event_bus=self.events,
        )
        self.plugins: list["Plugin"] = []
        # Set by LLMAvailabilityPlugin.start(); read by app_mode to fail-fast
        # when the LLM is DOWN instead of hitting the 15s first-token timeout.
        self.llm_availability = None
        self._shutdown_evt: asyncio.Event | None = None
        self._mic_task: asyncio.Task | None = None
        self._dispatch_task: asyncio.Task | None = None
        self._llm_turn_task: asyncio.Task | None = None
        self._first_tts_seen = False
        # Client-side VAD state machine. Drives manual asr_eos to SLV when
        # server-side VAD is disabled (slv_config.vad == "none"), so the
        # ASR model gets a chance to accumulate enough audio before being
        # asked to finalize.
        self._client_vad = None
        if getattr(config, "client_vad_backend", "off") != "off":
            try:
                self._client_vad = create_vad(
                    config.client_vad_backend,
                    sample_rate=config.audio_input_sample_rate,
                    threshold=getattr(config, "client_vad_threshold", None),
                )
                logger.info(
                    "client VAD: %s (threshold=%s)",
                    self._client_vad.name,
                    self._client_vad.threshold,
                )
            except Exception as e:
                logger.warning("client VAD init failed (%s); disabled", e)
                self._client_vad = None
        self._vad_state = "idle"  # "idle" | "speech"
        self._vad_speech_ms = 0
        self._vad_silence_ms = 0
        self._vad_eos_sent = False
        # ── v2: conversation state machine + observability ──
        # Initial state depends on pipeline_mode: always_on boots IDLE
        # (legacy), wake_word / push_to_talk boot SLEEPING.
        if getattr(config, "pipeline_mode", "always_on") == "always_on":
            self._state: ConvState = ConvState.IDLE
        else:
            self._state = ConvState.SLEEPING
        self._slv_reconnect_count: int = 0
        # Auto-sleep timer (only armed when pipeline_mode != always_on).
        self._sleep_task: asyncio.Task | None = None
        # Push-to-talk: when True, the next asr_final is the explicit
        # close of a PTT turn — used to short-circuit empty-final guards
        # that would otherwise drop a clipped PTT utterance.
        self._ptt_explicit_eos_pending: bool = False
        # Per-turn EOS dedupe: VAD silence and PTT/end can both want to
        # send asr_eos. Send at most one per turn. Cleared on every
        # ASRFinal, on PTT/start (next turn), and on reconnect.
        self._eos_sent_this_turn: bool = False
        # Watchdog: SLV in `always_on` pipeline mode does NOT emit
        # asr_final when ASR yields empty text (it filters server-side
        # to avoid noise turns). Without this watchdog the state machine
        # would stay THINKING forever after the very first VAD trigger
        # on mic noise. Started in send_asr_eos_once, cancelled in the
        # ASRFinal / SLVError / reconnect paths.
        self._asr_watchdog_task: asyncio.Task | None = None
        # Rate-limited stop-word matcher cache (compiled per Config update).
        self._stop_words_cache: tuple[list[str], list[str]] | None = None

    # ── v2: state machine + stop intent ─────────────────────────────

    def _set_state(self, new: ConvState) -> None:
        """Transition the conversation state. Logs + emits hook/event on change.

        Safe to call from any coroutine in the same event loop. Tests build
        BaseApp via __new__ without invoking __init__, so default missing
        attributes to IDLE rather than crashing.
        """
        old = getattr(self, "_state", ConvState.IDLE)
        if new == old:
            return
        self._state = new
        logger.info("ConvState: %s → %s", old.value, new.value)
        bus = getattr(self, "events", None)
        if bus is not None:
            try:
                bus.emit("state_change", {"state": new.value, "prev": old.value})
            except Exception:  # pragma: no cover - defensive
                logger.debug("EventBus state_change emit failed", exc_info=True)
        try:
            asyncio.get_running_loop().create_task(
                self._broadcast(
                    "on_state_change", {"state": new.value, "prev": old.value}
                )
            )
        except RuntimeError:
            pass

    def _normalise_for_stop(self, text: str) -> str:
        """Lowercase + strip whitespace and trailing punctuation."""
        if not text:
            return ""
        s = text.strip()
        # Strip trailing CJK + ASCII sentence punctuation.
        while s and s[-1] in "。，！？.!?,;:":
            s = s[:-1]
        return s.strip().lower()

    def _is_stop_intent(self, text: str) -> bool:
        """Match per spec: Chinese -> exact full-string; English -> case-
        insensitive whole-utterance OR word-boundary prefix (>= 2 chars).
        """
        norm = self._normalise_for_stop(text)
        if not norm:
            return False
        # Partition stop_words by ASCII-ness.
        cfg = getattr(self, "config", None)
        words = (getattr(cfg, "stop_words", []) if cfg is not None else []) or []
        for w in words:
            if not w:
                continue
            wn = w.strip().lower()
            if not wn:
                continue
            is_ascii = wn.isascii()
            if not is_ascii:
                # CJK / unicode: full-string equality only.
                if norm == wn:
                    return True
            else:
                # English: whole-utterance equality OR
                # word-boundary prefix (matched word is at least 2 chars).
                if norm == wn:
                    return True
                if len(wn) >= 2 and (
                    norm.startswith(wn + " ")
                    or norm.startswith(wn + ",")
                    or norm.startswith(wn + "!")
                    or norm.startswith(wn + "?")
                    or norm.startswith(wn + ".")
                ):
                    return True
        return False

    # ── pipeline_mode: wake / sleep / sleep-timer ──────────────────

    async def wake(self, source: str = "external") -> None:
        """Transition SLEEPING → IDLE and (re-)arm the sleep timer.

        No-op if not currently SLEEPING. Broadcasts on_wake with source.
        """
        if getattr(self, "_state", ConvState.IDLE) != ConvState.SLEEPING:
            return
        logger.info("wake from %s", source)
        try:
            await self._broadcast("on_wake", {"source": source})
        except Exception:
            logger.exception("on_wake broadcast failed")
        self._set_state(ConvState.IDLE)
        self._reset_sleep_timer()

    async def sleep(self) -> None:
        """Forcibly transition to SLEEPING — cancel LLM turn, abort SLV,
        drop playback. Idempotent if already SLEEPING."""
        if getattr(self, "_state", ConvState.IDLE) == ConvState.SLEEPING:
            return
        logger.info("sleep")
        try:
            await self._broadcast("on_sleep", None)
        except Exception:
            logger.exception("on_sleep broadcast failed")
        if self._llm_turn_task is not None and not self._llm_turn_task.done():
            self._llm_turn_task.cancel()
            try:
                await self._llm_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await self.slv.abort()
        except Exception:
            pass
        try:
            await self.audio.stop_playback()
        except Exception:
            pass
        self._first_tts_seen = False
        self._set_state(ConvState.SLEEPING)
        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()
        self._sleep_task = None

    def _reset_sleep_timer(self) -> None:
        """(Re-)start the auto-sleep countdown. No-op for always_on."""
        if getattr(self.config, "pipeline_mode", "always_on") == "always_on":
            return
        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()
        timeout = float(getattr(self.config, "sleep_timeout_s", 30.0))
        try:
            self._sleep_task = asyncio.create_task(
                self._sleep_after(timeout), name="sleep-timer"
            )
        except RuntimeError:
            # No running loop (called from sync context like tests).
            self._sleep_task = None

    async def _sleep_after(self, timeout: float) -> None:
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        # Only sleep if still IDLE — an in-flight turn delays.
        if getattr(self, "_state", ConvState.IDLE) == ConvState.IDLE:
            await self.sleep()

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
        # 0a. cancel auto-sleep timer too — otherwise a pending
        # _sleep_after coroutine can fire mid-shutdown, racing with
        # the rest of the cleanup (and emitting on_sleep after plugins
        # have already stopped).
        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()
            try:
                await self._sleep_task
            except (asyncio.CancelledError, Exception):
                pass
        self._sleep_task = None
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
        # 7. release LLM client resources (HTTP connection pool, etc.)
        try:
            await self.llm.aclose()
        except Exception:  # pragma: no cover
            pass

    def request_shutdown(self) -> None:
        if self._shutdown_evt is not None:
            self._shutdown_evt.set()

    # ── internal pumps ──────────────────────────────────────────────

    async def _mic_pump(self) -> None:
        """Mic capture loop. When client VAD is enabled, only forwards audio
        to SLV during (and just before) actual speech — pre-roll buffer
        ensures the first ~300ms of an utterance isn't lost while the VAD
        is still confirming speech-start. Idle silence is never sent.

        Why: streaming background noise for minutes at a time saturates the
        WS write pipeline and starves websockets' keepalive ping coroutine,
        triggering 1011 keepalive ping timeout. Dropping idle chunks keeps
        the connection mostly quiet between turns.
        """
        from collections import deque

        try:
            chunk_ms = getattr(self.audio, "chunk_ms", 100)
            preroll_max = max(1, 400 // max(chunk_ms, 1))  # ~400ms
            preroll: deque[bytes] = deque(maxlen=preroll_max)
            import numpy as _np
            # Rate-limit on_mic_rms broadcast: 10Hz is overkill for a
            # dashboard sparkline, and awaiting every plugin every 100ms
            # was starving the mic queue during TTS playback — VAD never
            # saw the burst of audio when the user spoke, so barge-in
            # never fired. Broadcast at most every ~200ms (every 2nd
            # chunk at chunk_ms=100, every 4th at chunk_ms=50).
            rms_broadcast_every = max(1, 200 // max(chunk_ms, 1))
            rms_chunk_counter = 0
            async for chunk in self.audio.start_capture():
                # pipeline_mode gating: drop audio entirely while SLEEPING.
                # WS stays connected so wake-time reconnect cost is zero.
                if getattr(self, "_state", ConvState.IDLE) == ConvState.SLEEPING:
                    # Also clear pre-roll so we don't leak pre-sleep audio
                    # into the next wake's first utterance.
                    preroll.clear()
                    continue
                # Per-chunk mic RMS for the dashboard. Rate-limited so a
                # slow WS client doesn't backpressure the mic queue and
                # starve VAD (which kills barge-in detection during TTS).
                rms_chunk_counter = (rms_chunk_counter + 1) % rms_broadcast_every
                if rms_chunk_counter == 0:
                    try:
                        arr = _np.frombuffer(chunk, dtype=_np.int16)
                        if arr.size:
                            rms = float(_np.sqrt(_np.mean((arr.astype(_np.float32) / 32768.0) ** 2)))
                        else:
                            rms = 0.0
                        thr = float(getattr(self.config, "client_vad_threshold", None) or 0.012)
                        await self._broadcast(
                            "on_mic_rms",
                            {"rms": rms, "threshold": thr, "state": self._vad_state},
                        )
                    except Exception:  # pragma: no cover - defensive
                        pass

                if self._client_vad is None:
                    # No VAD configured: stream everything (legacy behaviour).
                    await self.slv.send_audio(chunk)
                    continue

                # Update VAD first; it may transition idle→speech this chunk.
                try:
                    await self._update_vad(chunk, chunk_ms)
                except Exception:
                    logger.exception("client VAD update failed")

                if self._vad_state == "speech":
                    # Drain the pre-roll buffer at speech onset, then stream
                    # this chunk plus subsequent ones in real time.
                    if preroll:
                        for buffered in preroll:
                            await self.slv.send_audio(buffered)
                        preroll.clear()
                    await self.slv.send_audio(chunk)
                else:
                    # Idle: keep a short rolling buffer but don't transmit.
                    preroll.append(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("mic pump crashed")

    async def send_asr_eos_once(self) -> bool:
        """Send asr_eos to SLV at most once per turn.

        Returns True if this call actually sent the EOS, False if it
        was a duplicate (already sent this turn). The flag is reset on
        ASRFinal / PTT-start / SLV reconnect.

        Arms an `_asr_final_watchdog` so the state machine self-recovers
        if SLV doesn't echo an ASR final back (always_on pipeline mode
        silently drops empty finals — without this, the FSM stays
        THINKING forever after a noise-triggered turn).
        """
        if getattr(self, "_eos_sent_this_turn", False):
            return False
        self._eos_sent_this_turn = True
        try:
            await self.slv.asr_eos()
        except Exception:
            logger.exception("asr_eos send failed")
            # Don't clear the flag — even on failure we don't want to
            # retry a second time within the same turn and risk the SLV
            # state machine getting into an inconsistent state.
        # Arm watchdog (cancels any stale one from a prior failed turn).
        self._cancel_asr_watchdog()
        self._asr_watchdog_task = asyncio.create_task(
            self._asr_final_watchdog(),
            name="asr-final-watchdog",
        )
        return True

    def _cancel_asr_watchdog(self) -> None:
        """Cancel any pending asr_final watchdog (idempotent)."""
        task = getattr(self, "_asr_watchdog_task", None)
        if task is not None and not task.done():
            task.cancel()
        self._asr_watchdog_task = None

    async def _asr_final_watchdog(self) -> None:
        """Force state back to IDLE if asr_final never arrives after asr_eos.

        SLV's always_on pipeline filters empty-text finals server-side, so
        an EOS triggered by mic noise produces no client-visible final and
        the FSM would stay in THINKING forever. Real finals cancel this
        task before it fires.
        """
        timeout = float(getattr(self.config, "asr_final_timeout_s", 3.0))
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            return
        # Only act if (a) we still believe an EOS is outstanding and
        # (b) the FSM hasn't moved on (e.g. via SLVError, a late final
        # that arrived just before us, or a barge-in).
        if not getattr(self, "_eos_sent_this_turn", False):
            return
        if getattr(self, "_state", ConvState.IDLE) != ConvState.THINKING:
            return
        logger.warning(
            "asr_final not received within %.1fs after asr_eos; "
            "assuming empty/dropped final — resetting to IDLE", timeout,
        )
        self._eos_sent_this_turn = False
        self._set_state(ConvState.IDLE)

    async def _update_vad(self, chunk: bytes, chunk_ms: int) -> None:
        """Client-side speech-end detector. Sends asr_eos to SLV after a
        period of silence following speech, so Paraformer has accumulated
        enough audio to produce a non-empty final."""
        assert self._client_vad is not None
        # Gate while SLEEPING: don't update counters or fire eos.
        if getattr(self, "_state", ConvState.IDLE) == ConvState.SLEEPING:
            return
        # PTT mode with explicit-eos-only: skip VAD silence accumulation
        # entirely so the only EOS path is /api/control/ptt/end.
        cfg = getattr(self, "config", None)
        if (
            cfg is not None
            and getattr(cfg, "pipeline_mode", "always_on") == "push_to_talk"
            and getattr(cfg, "push_to_talk_no_vad_silence", True)
        ):
            return
        is_speech = self._client_vad.is_speech(chunk)
        if self._vad_state == "idle":
            if is_speech:
                self._vad_speech_ms += chunk_ms
                if self._vad_speech_ms >= self.config.client_vad_speech_min_ms:
                    self._vad_state = "speech"
                    self._vad_silence_ms = 0
                    self._vad_eos_sent = False
                    logger.info("client VAD: speech started")
                    # If TTS is currently playing, this is a barge-in.
                    # Transition straight to BARGED_IN so the dispatch
                    # loop's later ASRPartial check (which races SLV's
                    # ~610ms first-decode latency) doesn't miss the
                    # transition. mic_pump fires first because client
                    # VAD detects speech the moment we send chunks.
                    if self.audio.is_playing:
                        self._set_state(ConvState.BARGED_IN)
                        # Also cancel + abort + stop immediately —
                        # don't wait for the SLV partial roundtrip.
                        if self._llm_turn_task is not None and not self._llm_turn_task.done():
                            self._llm_turn_task.cancel()
                            try:
                                await self._llm_turn_task
                            except (asyncio.CancelledError, Exception):
                                pass
                        try:
                            await self.slv.abort()
                        except Exception:
                            pass
                        await self.audio.stop_playback()
                        self._first_tts_seen = False
                    else:
                        self._set_state(ConvState.LISTENING)
            else:
                self._vad_speech_ms = 0
        elif self._vad_state == "speech":
            if not is_speech:
                self._vad_silence_ms += chunk_ms
                if self._vad_silence_ms >= self.config.client_vad_silence_ms:
                    if not self._vad_eos_sent:
                        import time as _t
                        drove_eos = bool(getattr(self.config, "client_vad_drive_eos", False))
                        if drove_eos:
                            logger.info("client VAD: speech ended -> asr_eos")
                            await self._broadcast(
                                "on_user_speech_end_client",
                                {"ts": int(_t.time() * 1000), "drove_eos": True},
                            )
                            # Dedup: PTT/end may also try to send. Only
                            # one asr_eos per turn — race protection.
                            await self.send_asr_eos_once()
                        else:
                            logger.debug(
                                "client VAD: speech ended (paraformer-endpoint mode, no asr_eos)"
                            )
                            await self._broadcast(
                                "on_user_speech_end_client",
                                {"ts": int(_t.time() * 1000), "drove_eos": False},
                            )
                        self._set_state(ConvState.THINKING)
                        self._vad_eos_sent = True
                    self._vad_state = "idle"
                    self._vad_speech_ms = 0
                    self._vad_silence_ms = 0
                    self._client_vad.reset()
            else:
                self._vad_silence_ms = 0

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
        # ── pipeline_mode SLEEPING gate ─────────────────────────────
        # SLEEPING means the user explicitly silenced the agent (or it
        # auto-slept). The mic pump already drops audio, but events
        # already queued by SLV (partial / endpoint / final) may still
        # arrive after the sleep call. Honour the gate at the dispatch
        # boundary so a late asr_final can't wake the agent and trigger
        # a new LLM turn.
        if getattr(self, "_state", ConvState.IDLE) == ConvState.SLEEPING:
            if isinstance(evt, (ASRPartial, ASREndpoint)):
                return
            if isinstance(evt, ASRFinal):
                # SLV closed its WS on its side after asr_eos — must
                # still reconnect or the next user turn fails silently
                # with "WS closed mid-send". But DON'T broadcast the
                # utterance or spawn an LLM turn.
                if evt.session_complete:
                    try:
                        await self.slv.reconnect()
                        self._slv_reconnect_count = getattr(self, "_slv_reconnect_count", 0) + 1
                        self._first_tts_seen = False
                        await self._broadcast(
                            "on_slv_reconnect", {"count": self._slv_reconnect_count}
                        )
                    except Exception:
                        logger.exception("SLV reconnect failed (sleeping)")
                return
            # TTS frames during SLEEPING shouldn't normally arrive (we
            # aborted on sleep), but if they do, ignore — playback is
            # stopped anyway. Fall through for SLVError so we still log.

        if isinstance(evt, ASRPartial):
            # SLV's silero VAD can fire spurious empty endpoints from breath/
            # ambient noise; ignore empty partials so we don't trigger
            # bogus barge-ins or noise the dashboard.
            if not (evt.text or "").strip():
                return
            # Barge-in: user spoke (real text) while we were playing.
            if self.audio.is_playing:
                logger.info(
                    "BARGE-IN fired (state=%s, partial=%r)",
                    self._state.value, evt.text[:40]
                )
                if self._state == ConvState.SPEAKING:
                    self._set_state(ConvState.BARGED_IN)
                # Cancel any in-flight LLM turn FIRST: otherwise it keeps
                # streaming tokens to SLV which immediately restarts TTS
                # and undoes our barge-in stop_playback below.
                if self._llm_turn_task is not None and not self._llm_turn_task.done():
                    self._llm_turn_task.cancel()
                    try:
                        await self._llm_turn_task
                    except (asyncio.CancelledError, Exception):
                        pass
                try:
                    await self.slv.abort()
                except Exception:  # pragma: no cover
                    pass
                await self.audio.stop_playback()
                # Re-arm first-TTS-frame detector so the NEXT turn re-emits
                # the THINKING→SPEAKING transition; otherwise the dashboard
                # state pill stays stuck after barge-in.
                self._first_tts_seen = False
            await self._broadcast("on_user_partial", evt.text)
            return

        if isinstance(evt, ASREndpoint):
            await self._broadcast("on_user_speech_start")
            return

        if isinstance(evt, ASRFinal):
            # A real final arrived — disarm the watchdog so it doesn't
            # later reset state out from under whatever dispatch we're
            # about to run.
            self._cancel_asr_watchdog()
            # Clear the per-turn EOS dedupe flag for ALL final paths
            # (duplicate-of-streamed, empty, and real). Previously it was
            # only reset in the non-empty branch below, so a duplicate or
            # empty final would leave the flag set and the next turn's
            # send_asr_eos_once would early-return → SLV never receives
            # EOS → no final → state stuck THINKING forever (worse than
            # the empty-final bug the watchdog was designed to catch,
            # because the watchdog never even arms).
            self._eos_sent_this_turn = False
            if evt.duplicate_of_streamed:
                return
            # SLV closes the WS after every asr_eos-triggered final
            # (session_complete=True), regardless of whether the final
            # text is empty. Reconnect FIRST, then decide whether the
            # text was worth an LLM turn. If we skipped reconnect on
            # empty finals, the next user utterance would silently fail
            # with "send_json: WS closed mid-send, dropping asr_eos".
            if evt.session_complete:
                try:
                    await self.slv.reconnect()
                    logger.debug("SLV reconnected after session_complete final")
                    self._slv_reconnect_count = getattr(self, "_slv_reconnect_count", 0) + 1
                    # Reset first-TTS-frame flag so the next turn re-emits the
                    # THINKING→SPEAKING transition cleanly after reconnect.
                    self._first_tts_seen = False
                    await self._broadcast(
                        "on_slv_reconnect", {"count": self._slv_reconnect_count}
                    )
                except Exception:
                    logger.exception("SLV reconnect failed")
            # Drop empty finals — clawd's proven pattern. SLV's server-side
            # VAD or a too-short utterance produces empty text. Treating
            # those as real utterances would call the LLM with no input.
            if not (evt.text or "").strip():
                logger.info("empty asr_final ignored (likely VAD noise trigger)")
                # State must NOT stay stuck in THINKING when no real text
                # arrives. Reset back to IDLE so the next user turn can
                # transition cleanly via LISTENING.
                self._set_state(ConvState.IDLE)
                self._reset_sleep_timer()
                return
            logger.info("asr_final received: %r", evt.text)
            # New utterance round about to begin — clear client VAD state so
            # the next speech_start fires fresh. (getattr-guarded so tests
            # that build BaseApp via __new__ don't have to set every field.)
            self._vad_state = "idle"
            self._vad_speech_ms = 0
            self._vad_silence_ms = 0
            self._vad_eos_sent = False
            # Allow the NEXT turn to send asr_eos again.
            self._eos_sent_this_turn = False
            _cv = getattr(self, "_client_vad", None)
            if _cv is not None:
                try:
                    _cv.reset()
                except Exception:  # pragma: no cover
                    pass
            await self._broadcast("on_user_utterance", evt.text)
            # (reconnect already happened above, before the empty-text guard)
            # Stop-intent: user said "停下" / "stop" — cancel everything,
            # do NOT route to LLM and do NOT extend session.history (the
            # user asked for quiet, not for more conversation).
            if self._is_stop_intent(evt.text):
                logger.info("stop intent matched: %r", evt.text)
                if self._llm_turn_task is not None and not self._llm_turn_task.done():
                    self._llm_turn_task.cancel()
                    try:
                        await self._llm_turn_task
                    except (asyncio.CancelledError, Exception):
                        pass
                try:
                    await self.slv.abort()
                except Exception:  # pragma: no cover - best effort
                    pass
                try:
                    await self.audio.stop_playback()
                except Exception:  # pragma: no cover - best effort
                    pass
                self._set_state(ConvState.IDLE)
                self._reset_sleep_timer()
                await self._broadcast("on_user_stop_intent", evt.text)
                return
            # Spawn the LLM turn as a tracked task so the dispatch loop
            # stays free to handle queued TTSAudio (playback) and
            # ASRPartial (barge-in) while the model streams.
            if self._llm_turn_task is not None and not self._llm_turn_task.done():
                self._llm_turn_task.cancel()
                try:
                    await self._llm_turn_task
                except (asyncio.CancelledError, Exception):
                    pass
            # Ensure THINKING fires on the server-VAD path (where
            # _update_vad never runs and no client-side transition has
            # set it). Idempotent for client-VAD path which already set
            # THINKING in _update_vad.
            self._set_state(ConvState.THINKING)
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
                self._set_state(ConvState.SPEAKING)
                await self._broadcast(
                    "on_tts_audio_frame",
                    {"sample_rate": evt.sample_rate, "frame_len": len(evt.pcm)},
                )
            await self.audio.play(evt.pcm)
            return

        if isinstance(evt, TTSSentenceDone):
            await self._broadcast("on_assistant_sentence", evt.sentence)
            return

        if isinstance(evt, TTSDone):
            # Reset first-frame flag so the NEXT turn re-emits SPEAKING.
            self._first_tts_seen = False
            # Authoritative is_playing reset (audio_io stopped doing this on
            # transient empty queue to keep barge-in checks reliable).
            mark = getattr(self.audio, "mark_playback_done", None)
            if callable(mark):
                mark()
            self._set_state(ConvState.IDLE)
            self._reset_sleep_timer()
            await self._broadcast("on_assistant_done")
            return

        if isinstance(evt, SLVError):
            # Transport died — any pending asr_final watchdog is moot;
            # SLVError handling below already drives state back to IDLE.
            self._cancel_asr_watchdog()
            old_state = getattr(self, "_state", ConvState.IDLE)
            await self._broadcast(
                "on_error",
                TypedLLMError(
                    "slv_error",
                    evt.message,
                    exc_class="SLVError",
                ),
            )
            # Don't leave the FSM stuck in THINKING/SPEAKING after a transport
            # error — cancel any in-flight LLM turn.
            if self._llm_turn_task is not None and not self._llm_turn_task.done():
                self._llm_turn_task.cancel()
                try:
                    await self._llm_turn_task
                except (asyncio.CancelledError, Exception):
                    pass
            # If we were SLEEPING when the transport error fired, stay
            # SLEEPING — a transport hiccup must never wake the agent
            # (would hot-mic in wake_word mode).
            if old_state != ConvState.SLEEPING:
                self._set_state(ConvState.IDLE)
            else:
                logger.info("SLVError while SLEEPING; staying SLEEPING")
            return

    async def _run_user_utterance(self, text: str) -> None:
        """Wrap on_user_utterance so a crashing LLM turn doesn't kill the task silently."""
        try:
            await self.on_user_utterance(text)
            # Success path: tell the availability plugin so a transient
            # failure that earlier flipped us to DEGRADED gets cleared.
            avail = getattr(self, "llm_availability", None)
            if avail is not None:
                try:
                    avail.report_request_success()
                except Exception:  # pragma: no cover - defensive
                    pass
        except NotImplementedError:
            logger.error("BaseApp.on_user_utterance not overridden -- text dropped")
            self._set_state(ConvState.IDLE)
        except asyncio.CancelledError:
            # Cancellation happens on barge-in / shutdown / stop-intent;
            # caller already drove the appropriate state transition.
            raise
        except LLMUnavailable as e:
            # Fail-fast path: the availability state machine already
            # decided the LLM is DOWN. Don't bother A3 retry — surface
            # to the dashboard and return to IDLE immediately.
            logger.warning("LLM unavailable, fail-fast: %s", e)
            try:
                await self._broadcast(
                    "on_error",
                    TypedLLMError(
                        "llm_unavailable",
                        f"LLM 不可用：{e}",
                        exc_class=type(e).__name__,
                    ),
                )
            except Exception:
                pass
            self._set_state(ConvState.IDLE)
            try:
                self._reset_sleep_timer()
            except Exception:
                pass
            return
        except LLMTimeoutError as e:
            logger.warning(
                "LLM %s timeout after %.1fs (partial=%r)",
                e.kind, e.timeout_s, e.partial_text[:80],
            )
            # Real-world failure — push the state machine forward without
            # waiting for the next probe.
            avail = getattr(self, "llm_availability", None)
            if avail is not None:
                try:
                    avail.report_request_failure()
                except Exception:  # pragma: no cover - defensive
                    pass
            msg = (
                f"LLM 响应超时（{e.kind}, >{e.timeout_s:.0f}s）。"
                "可能 edge-llm 服务挂了或输入太长。"
            )
            try:
                await self._broadcast(
                    "on_error",
                    TypedLLMError(
                        "llm_timeout",
                        msg,
                        exc_class="LLMTimeoutError",
                        kind=e.kind,
                        timeout_s=e.timeout_s,
                    ),
                )
            except Exception:
                pass
            self._set_state(ConvState.IDLE)
            try:
                self._reset_sleep_timer()
            except Exception:
                pass
        except Exception as e:
            logger.exception("on_user_utterance failed")
            # Real-world failure — feed back to the availability machine
            # (only for LLM-class errors; other exceptions might be local
            # bugs and shouldn't poison the breaker).
            try:
                from openai import APIError as _APIError
                _is_llm_err = isinstance(e, (_APIError, LLMStreamError))
            except Exception:  # pragma: no cover - defensive
                _is_llm_err = isinstance(e, LLMStreamError)
            if _is_llm_err:
                avail = getattr(self, "llm_availability", None)
                if avail is not None:
                    try:
                        avail.report_request_failure()
                    except Exception:  # pragma: no cover - defensive
                        pass
            # A3: surface non-timeout LLM failures to the dashboard so
            # operators see *something* when edge-llm crashes or returns
            # a 4xx. Wrap the original exception's repr into a clean
            # RuntimeError (the on_error contract already accepts a
            # BaseException and prefers str()).
            try:
                exc_class = type(e).__name__
                msg = f"LLM 调用失败（{exc_class}）：{e}"
                err_type = (
                    "llm_stream_error"
                    if isinstance(e, LLMStreamError)
                    else "llm_failure"
                )
                await self._broadcast(
                    "on_error",
                    TypedLLMError(err_type, msg, exc_class=exc_class),
                )
            except Exception:
                pass
            self._set_state(ConvState.IDLE)
            try:
                self._reset_sleep_timer()
            except Exception:
                pass

    async def broadcast(self, hook_name: str, *args) -> None:
        """Public hook broadcaster -- call from subclasses to fan out events.

        Used by DialogueApp.on_user_utterance to fan out per-token deltas
        (`on_assistant_token`) since the dispatch loop has no access to
        the LLM's token stream.
        """
        plugins = getattr(self, "plugins", None)
        if not plugins:
            return
        coros = []
        for p in plugins:
            fn = getattr(p, hook_name, None)
            if fn is None:
                continue
            coros.append(_safe_call(p.name, hook_name, fn, *args))
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

    # Backwards-compatible alias used internally by the dispatch loop.
    _broadcast = broadcast


async def _safe_call(plugin_name: str, hook: str, fn, *args) -> None:  # noqa: ANN001
    try:
        result = fn(*args)
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        logger.exception("plugin %s.%s failed", plugin_name, hook)


__all__ = ["BaseApp"]
