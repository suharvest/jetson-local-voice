"""sounddevice-backed mic capture + speaker playback.

Single persistent InputStream + OutputStream; sounddevice callbacks run
on background threads and push into asyncio.Queue via run_coroutine_threadsafe.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError) as _sd_exc:  # pragma: no cover - hardware-dependent
    sd = None  # type: ignore[assignment]
    _SD_IMPORT_ERR: Exception | None = _sd_exc
else:
    _SD_IMPORT_ERR = None

logger = logging.getLogger(__name__)


class AudioIO:
    """Mic in / speaker out, backed by sounddevice."""

    def __init__(
        self,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
        input_sr: int = 16000,
        output_sr: int = 24000,
        chunk_ms: int = 32,
    ) -> None:
        self.input_device = input_device
        self.output_device = output_device
        self.input_sr = input_sr
        self.output_sr = output_sr
        self.chunk_ms = chunk_ms
        self._chunk_frames = int(input_sr * chunk_ms / 1000)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._in_queue: asyncio.Queue[bytes] | None = None
        self._out_queue: asyncio.Queue[bytes] | None = None
        self._input_stream: "sd.RawInputStream | None" = None
        self._output_stream: "sd.RawOutputStream | None" = None
        self._playback_task: asyncio.Task | None = None
        self._is_playing = False

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    # ── capture ─────────────────────────────────────────────────────

    async def start_capture(self) -> AsyncIterator[bytes]:
        if sd is None:  # pragma: no cover - hardware-dependent
            raise RuntimeError(
                f"sounddevice unavailable: {_SD_IMPORT_ERR!r}. Install libportaudio2."
            )
        self._loop = asyncio.get_running_loop()
        self._in_queue = asyncio.Queue(maxsize=64)

        def _cb(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                logger.debug("input status: %s", status)
            buf = bytes(indata)
            try:
                assert self._loop is not None and self._in_queue is not None
                # IMPORTANT: schedule _safe_put on the loop thread, not
                # put_nowait directly -- QueueFull would otherwise be
                # raised on the loop thread where this try/except cannot
                # catch it.
                self._loop.call_soon_threadsafe(self._safe_put, buf)
            except Exception as e:  # pragma: no cover
                logger.warning("mic cb error: %s", e)

        self._input_stream = sd.RawInputStream(
            samplerate=self.input_sr,
            blocksize=self._chunk_frames,
            device=self.input_device,
            channels=1,
            dtype="int16",
            callback=_cb,
        )
        self._input_stream.start()

        try:
            while True:
                chunk = await self._in_queue.get()
                yield chunk
        finally:
            self._stop_input_stream()

    def _safe_put(self, data: bytes) -> None:
        """Runs on the asyncio loop thread; drops the chunk if the queue is full."""
        if self._in_queue is None:
            return
        try:
            self._in_queue.put_nowait(data)
        except asyncio.QueueFull:
            logger.warning("mic queue full -- dropping chunk")

    def _stop_input_stream(self) -> None:
        if self._input_stream is not None:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception:  # pragma: no cover
                pass
            self._input_stream = None

    # ── playback ────────────────────────────────────────────────────

    def _ensure_output(self) -> None:
        if sd is None:  # pragma: no cover - hardware-dependent
            raise RuntimeError(
                f"sounddevice unavailable: {_SD_IMPORT_ERR!r}. Install libportaudio2."
            )
        if self._output_stream is not None:
            return
        self._output_stream = sd.RawOutputStream(
            samplerate=self.output_sr,
            blocksize=0,
            device=self.output_device,
            channels=1,
            dtype="int16",
        )
        self._output_stream.start()
        if self._out_queue is None:
            self._out_queue = asyncio.Queue()
        self._playback_task = asyncio.create_task(self._playback_loop(), name="audio-playback")

    async def _playback_loop(self) -> None:
        assert self._out_queue is not None
        try:
            while True:
                pcm = await self._out_queue.get()
                if pcm is None:
                    self._is_playing = False
                    continue
                self._is_playing = True
                try:
                    if self._output_stream is not None:
                        self._output_stream.write(pcm)
                except Exception as e:  # pragma: no cover
                    logger.warning("playback write error: %s", e)
                if self._out_queue.empty():
                    self._is_playing = False
        except asyncio.CancelledError:
            raise

    async def play(self, pcm: bytes) -> None:
        self._ensure_output()
        assert self._out_queue is not None
        self._is_playing = True
        await self._out_queue.put(pcm)

    def set_output_sample_rate(self, sr: int) -> None:
        if sr == self.output_sr and self._output_stream is not None:
            return
        self.output_sr = sr
        # Reconfigure: drop existing stream, recreate lazily next play().
        if self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:  # pragma: no cover
                pass
            self._output_stream = None
        if self._playback_task is not None:
            self._playback_task.cancel()
            self._playback_task = None
        self._out_queue = None

    async def stop_playback(self) -> None:
        """Drain queued audio (barge-in)."""
        if self._out_queue is not None:
            while not self._out_queue.empty():
                try:
                    self._out_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        self._is_playing = False

    async def close(self) -> None:
        self._stop_input_stream()
        if self._playback_task is not None:
            self._playback_task.cancel()
            try:
                await self._playback_task
            except (asyncio.CancelledError, Exception):
                pass
            self._playback_task = None
        if self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:  # pragma: no cover
                pass
            self._output_stream = None


__all__ = ["AudioIO"]


# Helper to keep numpy import alive (some platforms need it loaded for
# sounddevice's CFFI bindings to find PortAudio's int16 path).
_ = np.zeros(1, dtype=np.int16)
