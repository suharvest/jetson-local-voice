"""ScriptedAudioIO -- drop-in replacement for AudioIO used by E2E tests.

Plays back scripted WAV files into the agent's mic pump, and captures
TTS PCM bytes that the agent would have written to the speaker.

Surface compatibility with `openvoicestream_agent.audio_io.AudioIO`:
  properties: chunk_ms, input_sr, output_sr, is_playing
  async:      start_capture, play, stop_playback, close
  sync:       set_output_sample_rate, mark_playback_done
"""
from __future__ import annotations

import asyncio
import time
import wave
from pathlib import Path
from typing import AsyncIterator

import numpy as np


def _read_wav_pcm16_mono(path: str | Path, target_sr: int = 16000) -> bytes:
    """Read a WAV file, downmix to mono, resample to target_sr, return int16 PCM bytes."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sw != 2:
        # Coerce 8-bit / 24-bit / 32-bit floats to int16. Best-effort only.
        if sw == 1:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8
        elif sw == 4:
            arr = (np.frombuffer(raw, dtype=np.int32) >> 16).astype(np.int16)
        else:
            arr = np.frombuffer(raw, dtype=np.int16)
    else:
        arr = np.frombuffer(raw, dtype=np.int16)
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1).astype(np.int16)
    if sr != target_sr and arr.size:
        n_out = int(len(arr) * target_sr / sr)
        x0 = np.linspace(0, 1, len(arr), endpoint=False)
        x1 = np.linspace(0, 1, n_out, endpoint=False)
        arr = np.interp(x1, x0, arr.astype(np.float32)).astype(np.int16)
    return arr.tobytes()


def _iter_chunks(pcm: bytes, chunk_bytes: int) -> list[bytes]:
    return [pcm[i:i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes) if pcm[i:i + chunk_bytes]]


class ScriptedAudioIO:
    """Scripted mic + TTS sink.

    `script` is a list of (delay_ms_after_prev_step, source) where source
    is either a WAV path (str/Path) or raw int16 PCM bytes (16kHz mono).

    After the script exhausts, yields silence forever until close().
    """

    def __init__(
        self,
        script,
        input_sr: int = 16000,
        chunk_ms: int = 100,
        output_sr: int = 16000,
    ) -> None:
        self.script = list(script)
        self.input_sr = input_sr
        self.chunk_ms = chunk_ms
        self.output_sr = output_sr
        self._is_playing = False
        self.captured_tts = bytearray()
        self.tts_first_frame_ts_ms: int | None = None
        self.tts_sr: int | None = None
        self._closed = False
        self.script_step: int = -1
        self._step_done_events: list[asyncio.Event] = []
        # Runtime injection: tests can call `inject(wav_path)` to feed an
        # extra utterance immediately, useful for barge-in scenarios where
        # the trigger moment must align with a real-time event (first TTS
        # byte). Drained ahead of the silent tail.
        self._inject_queue: asyncio.Queue[bytes] = asyncio.Queue()

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    async def start_capture(self) -> AsyncIterator[bytes]:
        chunk_bytes = (self.input_sr * self.chunk_ms // 1000) * 2
        silence = bytes(chunk_bytes)
        for i, (delay_ms, source) in enumerate(self.script):
            if self._closed:
                return
            # CRITICAL: inter-step delays must yield SILENCE CHUNKS instead
            # of blocking on asyncio.sleep. Otherwise mic_pump stalls, the
            # client VAD never sees speech-end between WAVs, and two
            # consecutive utterances get stitched into one. Tested: the
            # old "await asyncio.sleep(delay_ms/1000)" caused multi_turn
            # to merge hello.wav + weather.wav into a single asr_final.
            n_silence = max(0, int(delay_ms / self.chunk_ms))
            for _ in range(n_silence):
                if self._closed:
                    return
                yield silence
                await asyncio.sleep(self.chunk_ms / 1000)
            if isinstance(source, (bytes, bytearray)):
                pcm = bytes(source)
            else:
                pcm = _read_wav_pcm16_mono(source, self.input_sr)
            for chunk in _iter_chunks(pcm, chunk_bytes):
                if self._closed:
                    return
                # Pad final short chunk to fixed size so VAD framing stays stable.
                if len(chunk) < chunk_bytes:
                    chunk = chunk + bytes(chunk_bytes - len(chunk))
                yield chunk
                await asyncio.sleep(self.chunk_ms / 1000)
            self.script_step = i
            while len(self._step_done_events) <= i:
                self._step_done_events.append(asyncio.Event())
            self._step_done_events[i].set()
        # Idle tail: yield silence forever, but drain any runtime-injected
        # WAVs first so tests can react to live events (e.g. barge-in
        # after observing TTS bytes).
        while not self._closed:
            if not self._inject_queue.empty():
                pcm = self._inject_queue.get_nowait()
                for chunk in _iter_chunks(pcm, chunk_bytes):
                    if self._closed:
                        return
                    if len(chunk) < chunk_bytes:
                        chunk = chunk + bytes(chunk_bytes - len(chunk))
                    yield chunk
                    await asyncio.sleep(self.chunk_ms / 1000)
                continue
            yield silence
            await asyncio.sleep(self.chunk_ms / 1000)

    def inject(self, source) -> None:
        """Queue an extra audio source to be streamed during the idle tail.
        Called by tests to synchronize barge-in with real agent state."""
        if isinstance(source, (bytes, bytearray)):
            pcm = bytes(source)
        else:
            pcm = _read_wav_pcm16_mono(source, self.input_sr)
        self._inject_queue.put_nowait(pcm)

    async def play(self, pcm: bytes) -> None:
        if self.tts_first_frame_ts_ms is None:
            self.tts_first_frame_ts_ms = int(time.time() * 1000)
        self._is_playing = True
        self.captured_tts.extend(pcm)

    async def stop_playback(self) -> None:
        self._is_playing = False

    def mark_playback_done(self) -> None:
        self._is_playing = False

    def set_output_sample_rate(self, sr: int) -> None:
        self.tts_sr = sr
        self.output_sr = sr

    async def close(self) -> None:
        self._closed = True

    def step_done(self, i: int) -> asyncio.Event:
        """Returns event that fires when script step `i` finishes streaming."""
        while len(self._step_done_events) <= i:
            self._step_done_events.append(asyncio.Event())
        return self._step_done_events[i]


__all__ = ["ScriptedAudioIO", "_read_wav_pcm16_mono"]
