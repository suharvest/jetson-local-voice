"""Async client for SLV's /v2v/stream WebSocket.

Single persistent connection per App lifetime (invariant 1). All public
send methods are serialized through an internal lock so frames never
interleave. Reader task decodes JSON vs binary frames and pushes typed
V2VEvent values onto a queue exposed via `events()`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
from dataclasses import dataclass
from typing import Any, AsyncIterator

import websockets
from websockets.asyncio.client import connect as ws_connect

# Re-use SLV's protocol constants (invariant 6: never redeclare).
from app.core.v2v import (  # type: ignore[import-not-found]
    CLIENT_ABORT,
    CLIENT_ASR_EOS,
    CLIENT_CONFIG,
    CLIENT_TEXT,
    CLIENT_TTS_FLUSH,
    SERVER_ASR_ENDPOINT,
    SERVER_ASR_FINAL,
    SERVER_ASR_PARTIAL,
    SERVER_ERROR,
    SERVER_TTS_DONE,
    SERVER_TTS_SENTENCE_DONE,
    SERVER_TTS_STARTED,
)

logger = logging.getLogger(__name__)


# ── Typed events ──────────────────────────────────────────────────────


class V2VEvent:
    """Base class for events emitted by SLVClient."""


@dataclass
class ASRPartial(V2VEvent):
    text: str
    is_stable: bool = False


@dataclass
class ASREndpoint(V2VEvent):
    pass


@dataclass
class ASRFinal(V2VEvent):
    text: str
    session_complete: bool = True
    duplicate_of_streamed: bool = False


@dataclass
class TTSStarted(V2VEvent):
    sentence: str


@dataclass
class TTSSentenceDone(V2VEvent):
    sentence: str


@dataclass
class TTSDone(V2VEvent):
    pass


@dataclass
class TTSAudio(V2VEvent):
    pcm: bytes
    sample_rate: int


@dataclass
class SLVError(V2VEvent):
    message: str


# ── Client ───────────────────────────────────────────────────────────


class SLVClient:
    """One persistent WS to /v2v/stream for the entire App lifetime."""

    def __init__(self, url: str, config: dict[str, Any]) -> None:
        self.url = url
        self.config = dict(config)
        # Make sure multi_utterance is on (invariant 1).
        self.config["multi_utterance"] = True

        self._ws: Any | None = None
        self._reader_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()
        self._queue: asyncio.Queue[V2VEvent] = asyncio.Queue()
        self._tts_sample_rate: int | None = None
        self._closed = False
        # Set when reader exits for any reason; events() uses this to
        # break out of `await queue.get()` instead of hanging forever.
        self._reader_done: asyncio.Event = asyncio.Event()

    # ── lifecycle ───────────────────────────────────────────────────

    async def connect(self) -> None:
        if self._ws is not None:
            raise RuntimeError("SLVClient.connect() called twice -- one WS per lifetime")
        self._ws = await ws_connect(self.url, max_size=None)
        await self._ws.send(json.dumps({"type": CLIENT_CONFIG, **self.config}))
        self._reader_task = asyncio.create_task(self._reader_loop(), name="slv-reader")

    async def close(self) -> None:
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._ws = None

    # ── send helpers ────────────────────────────────────────────────

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("SLVClient not connected")
        async with self._send_lock:
            await self._ws.send(json.dumps(payload))

    async def send_audio(self, pcm: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("SLVClient not connected")
        async with self._send_lock:
            await self._ws.send(pcm)

    async def send_text(self, text: str) -> None:
        await self._send_json({"type": CLIENT_TEXT, "text": text})

    async def flush_tts(self) -> None:
        await self._send_json({"type": CLIENT_TTS_FLUSH})

    async def abort(self) -> None:
        await self._send_json({"type": CLIENT_ABORT})

    async def asr_eos(self) -> None:
        await self._send_json({"type": CLIENT_ASR_EOS})

    # ── reader ──────────────────────────────────────────────────────

    async def events(self) -> AsyncIterator[V2VEvent]:
        while True:
            # Drain anything already queued first so SLVError emitted on
            # reader exit is still surfaced to the consumer.
            if not self._queue.empty():
                yield self._queue.get_nowait()
                continue
            if self._reader_done.is_set() or self._closed:
                return
            get_task = asyncio.create_task(self._queue.get())
            done_task = asyncio.create_task(self._reader_done.wait())
            try:
                done, pending = await asyncio.wait(
                    {get_task, done_task}, return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                get_task.cancel()
                done_task.cancel()
                raise
            for t in pending:
                t.cancel()
            if get_task in done:
                yield get_task.result()
            else:
                # Reader finished; flush any final items it pushed.
                while not self._queue.empty():
                    yield self._queue.get_nowait()
                return

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if isinstance(msg, (bytes, bytearray)):
                    await self._handle_binary(bytes(msg))
                else:
                    await self._handle_json(msg)
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed as e:
            logger.info("SLV WS closed: %s", e)
            await self._queue.put(SLVError(f"connection closed: {e}"))
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("SLV reader crashed")
            await self._queue.put(SLVError(str(e)))
        finally:
            # Wake any consumer blocked on events().
            self._reader_done.set()

    async def _handle_binary(self, data: bytes) -> None:
        if self._tts_sample_rate is None:
            if len(data) < 4:
                await self._queue.put(SLVError("first binary frame < 4 bytes"))
                return
            (sr,) = struct.unpack("<I", data[:4])
            self._tts_sample_rate = sr
            pcm = data[4:]
            if pcm:
                await self._queue.put(TTSAudio(pcm=pcm, sample_rate=sr))
            return
        await self._queue.put(TTSAudio(pcm=data, sample_rate=self._tts_sample_rate))

    async def _handle_json(self, raw: str) -> None:
        try:
            evt = json.loads(raw)
        except json.JSONDecodeError as e:
            await self._queue.put(SLVError(f"bad json: {e}"))
            return
        t = evt.get("type")
        if t == SERVER_ASR_PARTIAL:
            await self._queue.put(
                ASRPartial(text=evt.get("text", ""), is_stable=bool(evt.get("is_stable", False)))
            )
        elif t == SERVER_ASR_ENDPOINT:
            await self._queue.put(ASREndpoint())
        elif t == SERVER_ASR_FINAL:
            await self._queue.put(
                ASRFinal(
                    text=evt.get("text", ""),
                    session_complete=bool(evt.get("session_complete", True)),
                    duplicate_of_streamed=bool(evt.get("duplicate_of_streamed", False)),
                )
            )
        elif t == SERVER_TTS_STARTED:
            await self._queue.put(TTSStarted(sentence=evt.get("sentence", "")))
        elif t == SERVER_TTS_SENTENCE_DONE:
            await self._queue.put(TTSSentenceDone(sentence=evt.get("sentence", "")))
        elif t == SERVER_TTS_DONE:
            await self._queue.put(TTSDone())
        elif t == SERVER_ERROR:
            await self._queue.put(SLVError(evt.get("error", "unknown")))
        else:
            logger.debug("Unknown SLV message type: %r", t)
