"""ASR / TTS clients with millisecond-level instrumentation.

Single source of truth for perf timing. Every runner uses these — keeps
timestamp semantics consistent across asr/tts/v2v/concurrent.
"""
from __future__ import annotations
import io, json, time, urllib.parse, wave
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import requests
import websocket  # websocket-client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_text(value) -> str:
    """ASR servers vary in `text` field type. Coerce to a single string.
    - str: pass through
    - dict: try common per-language keys, then fall back to first string value
    - None / other: empty string
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in ("zh", "en", "text", "result", "transcript"):
            v = value.get(k)
            if isinstance(v, str) and v:
                return v
        for v in value.values():
            if isinstance(v, str) and v:
                return v
        return ""
    if value is None:
        return ""
    return str(value)


def wav_duration_s(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        return wf.getnframes() / wf.getframerate()


def wav_to_pcm_chunks(wav_bytes: bytes, chunk_ms: int = 250) -> tuple[list[bytes], int]:
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16)
    chunk_n = int(sr * chunk_ms / 1000)
    return [samples[i:i + chunk_n].tobytes() for i in range(0, len(samples), chunk_n)], sr


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

@dataclass
class ASRResult:
    text: str
    audio_dur_s: float
    processing_ms: float          # wall-clock from request start to final text
    tfd_ms: float | None = None   # streaming only: first PCM sent -> first partial
    eos_to_final_ms: float | None = None  # streaming only
    rtf: float = 0.0              # wall-clock RTF: processing_ms / audio_dur
                                  # NOTE: in --realtime streaming mode this is ≥ 1.0
                                  # by construction (client sleeps between chunks).
    finalize_rtf: float | None = None  # compute-bound RTF for streaming:
                                       # eos_to_final_ms / audio_dur. This is the
                                       # cross-device-comparable number — independent
                                       # of how the client paces chunks.
    eos_mode: str = "vad"
    vad_silence_ms: float | None = None           # EOS → VAD endpoint detection
    asr_finalize_compute_ms: float | None = None  # VAD endpoint → ASR final text

    @property
    def as_dict(self) -> dict:
        return {**self.__dict__}


@dataclass
class TTSResult:
    audio_bytes: bytes
    audio_dur_s: float            # synthesized audio duration
    tfd_ms: float                 # request -> first audio chunk
    total_ms: float               # request -> last audio chunk
    rtf: float = 0.0              # total_ms / audio_dur

    @property
    def as_dict(self) -> dict:
        return {
            "audio_dur_s": self.audio_dur_s,
            "tfd_ms": self.tfd_ms,
            "total_ms": self.total_ms,
            "rtf": self.rtf,
        }


# ---------------------------------------------------------------------------
# ASR client
# ---------------------------------------------------------------------------

class ASRClient:
    def __init__(self, base_url: str, ws_url: str | None = None,
                 chunk_ms: int = 250, realtime: bool = True,
                 timeout: int = 120, vad_backend: str | None = "silero",
                 vad_silence_ms: int = 400):
        self.base_url = base_url.rstrip("/")
        self.ws_url = (ws_url or base_url).replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
        self.chunk_ms = chunk_ms
        self.realtime = realtime
        self.timeout = timeout
        self.vad_backend = vad_backend
        self.vad_silence_ms = vad_silence_ms

    # ----- offline POST /asr -----
    def transcribe_offline(self, wav_bytes: bytes, language: str = "Chinese") -> ASRResult:
        dur = wav_duration_s(wav_bytes)
        t0 = time.monotonic()
        resp = requests.post(
            f"{self.base_url}/asr",
            files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
            data={"language": language},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = resp.json().get("text", "").strip()
        proc_ms = (time.monotonic() - t0) * 1000
        return ASRResult(
            text=text, audio_dur_s=dur, processing_ms=proc_ms,
            rtf=proc_ms / (dur * 1000) if dur else 0.0,
        )

    # ----- streaming WS /asr/stream -----
    def transcribe_streaming(self, wav_bytes: bytes, language: str = "Chinese",
                             eos_mode: str = "vad") -> ASRResult:
        """
        eos_mode:
          - "forced": send b"" after last PCM chunk (immediate finalize)
          - "vad":    don't send b""; let server/backend VAD trigger finalize
          - "eou":    send {"type":"eou"} after last PCM chunk (dialogue-manager EOU)
        """
        assert eos_mode in ("forced", "vad", "eou")
        chunks, sr = wav_to_pcm_chunks(wav_bytes, self.chunk_ms)
        dur = wav_duration_s(wav_bytes)
        chunk_dur = self.chunk_ms / 1000.0
        query = {"language": language, "sample_rate": str(sr)}
        if eos_mode == "vad" and self.vad_backend:
            query["vad"] = self.vad_backend
            query["vad_silence_ms"] = str(self.vad_silence_ms)
        elif eos_mode in ("forced", "eou"):
            query["vad"] = "none"
        qs = urllib.parse.urlencode(query)

        ws = websocket.create_connection(
            f"{self.ws_url}/asr/stream?{qs}",
            timeout=self.timeout,
        )
        t_first_send = time.monotonic()
        t_first_partial: float | None = None
        final_text = ""
        final_received = False

        # Pump audio in a thread so we can read partials concurrently
        # Simpler approach: send each chunk, then non-blocking poll for partials
        # via short ws timeout. websocket-client doesn't make this easy with
        # a single connection — we just send everything, then read until final.
        # That's good enough for TFD measurement because partials arrive
        # eagerly anyway; we just sample TFD by checking right after each
        # send for a backlog message.
        ws.settimeout(0.001)
        t_vad_endpoint = None
        for c in chunks:
            ws.send_binary(c)
            try:
                msg = ws.recv()
                if not msg:
                    continue
                data = json.loads(msg)
                if data.get("type") == "vad_endpoint":
                    t_vad_endpoint = time.monotonic()
                elif data.get("type") == "final" or data.get("is_final") is True:
                    final_text = _coerce_text(data.get("text", "")).strip()
                    final_received = True
                    break
                elif data.get("text") and t_first_partial is None:
                    t_first_partial = time.monotonic()
            except (websocket.WebSocketTimeoutException, json.JSONDecodeError):
                pass
            if self.realtime:
                time.sleep(chunk_dur)
        ws.settimeout(self.timeout)

        t_eos = time.monotonic()
        if final_received:
            # VAD detected end during real audio — server already sent final.
            # Jump straight to computing result.
            t_final = time.monotonic()  # already received, approximate
            ws.close()
            eos_to_final_ms = 0.0  # final arrived before t_eos
            return ASRResult(
                text=final_text, audio_dur_s=dur,
                processing_ms=(t_final - t_first_send) * 1000,
                tfd_ms=((t_first_partial - t_first_send) * 1000) if t_first_partial else None,
                eos_to_final_ms=0.0, rtf=0.0, finalize_rtf=None,
                eos_mode=eos_mode,
                vad_silence_ms=None, asr_finalize_compute_ms=None,
            )
        vad_timeout_s = (self.vad_silence_ms + 30000) / 1000.0  # silence tail + 30s buffer for slow backends
        if eos_mode == "forced":
            ws.send_binary(b"")
        elif eos_mode == "eou":
            ws.send(json.dumps({"type": "eou"}))
        elif eos_mode == "vad":
            # File replay has no microphone tail. Send enough trailing silence
            # for server-side VAD to observe the configured hangover while
            # keeping t_eos anchored at the end of the speech file.
            silence_ms = max(self.vad_silence_ms + self.chunk_ms, self.chunk_ms)
            silence_chunks = int(np.ceil(silence_ms / self.chunk_ms))
            frames_per_chunk = int(sr * self.chunk_ms / 1000)
            silence = np.zeros(frames_per_chunk, dtype=np.int16).tobytes()
            ws.settimeout(0.001)
            for _ in range(silence_chunks):
                ws.send_binary(silence)
                try:
                    msg = ws.recv()
                    data = json.loads(msg)
                    text_str = _coerce_text(data.get("text", ""))
                    if t_first_partial is None and text_str:
                        t_first_partial = time.monotonic()
                    if data.get("type") == "vad_endpoint":
                        t_vad_endpoint = time.monotonic()
                    if data.get("type") == "final" or data.get("is_final") is True:
                        final_text = text_str.strip()
                        final_received = True
                        break
                except websocket.WebSocketTimeoutException:
                    pass
                if self.realtime:
                    time.sleep(chunk_dur)
            ws.settimeout(self.timeout)

        while not final_received:
            # Timeout guard: if VAD/silence mode and no final within deadline
            if eos_mode == "vad" and (time.monotonic() - t_eos) > vad_timeout_s:
                ws.close()
                return ASRResult(
                    text="<timeout>",
                    audio_dur_s=dur,
                    processing_ms=(time.monotonic() - t_first_send) * 1000,
                    eos_to_final_ms=None,
                    rtf=0.0,
                    finalize_rtf=None,
                    eos_mode=eos_mode,
                    vad_silence_ms=None,
                    asr_finalize_compute_ms=None,
                )
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not raw:
                raise RuntimeError("server closed WebSocket without a final frame (likely backend error)")
            data = json.loads(raw)
            if data.get("type") == "error":
                raise RuntimeError(f"server error: {data.get('error', '(no detail)')}")
            if data.get("type") == "vad_endpoint":
                t_vad_endpoint = time.monotonic()
                continue
            text_field = data.get("text", final_text)
            text_str = _coerce_text(text_field)
            if t_first_partial is None and text_str:
                t_first_partial = time.monotonic()
            if data.get("type") == "final" or data.get("is_final") is True:
                final_text = text_str.strip()
                final_received = True
                break
            final_text = text_str.strip()
        t_final = time.monotonic()
        ws.close()

        eos_to_final_ms = (t_final - t_eos) * 1000
        vad_silence = ((t_vad_endpoint - t_eos) * 1000) if t_vad_endpoint else None
        asr_finalize_compute = ((t_final - t_vad_endpoint) * 1000) if t_vad_endpoint else None
        return ASRResult(
            text=final_text,
            audio_dur_s=dur,
            processing_ms=(t_final - t_first_send) * 1000,
            tfd_ms=((t_first_partial - t_first_send) * 1000) if t_first_partial else None,
            eos_to_final_ms=eos_to_final_ms,
            rtf=((t_final - t_first_send) * 1000) / (dur * 1000) if dur else 0.0,
            finalize_rtf=eos_to_final_ms / (dur * 1000) if dur else None,
            eos_mode=eos_mode,
            vad_silence_ms=vad_silence,
            asr_finalize_compute_ms=asr_finalize_compute,
        )


# ---------------------------------------------------------------------------
# TTS client
# ---------------------------------------------------------------------------

class TTSClient:
    def __init__(self, base_url: str, timeout: int = 120,
                 stream: bool = True, voice: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.stream = stream
        self.voice = voice

    def synthesize(self, text: str, language: str = "zh") -> TTSResult:
        payload: dict = {"text": text}
        if self.voice:
            payload["voice"] = self.voice
        endpoint = "/tts/stream" if self.stream else "/tts"

        t0 = time.monotonic()
        resp = requests.post(
            f"{self.base_url}{endpoint}",
            json=payload, stream=self.stream, timeout=self.timeout,
        )
        resp.raise_for_status()

        buf = bytearray()
        t_first: float | None = None
        if self.stream:
            for chunk in resp.iter_content(4096):
                if not chunk:
                    continue
                if t_first is None:
                    t_first = time.monotonic()
                buf.extend(chunk)
        else:
            t_first = time.monotonic()
            buf.extend(resp.content)
        t_end = time.monotonic()

        audio = bytes(buf)
        try:
            dur = wav_duration_s(audio)
        except Exception:
            # Server may return raw PCM; estimate from byte size at 16k/16-bit mono
            dur = len(audio) / (16000 * 2)
        total_ms = (t_end - t0) * 1000
        tfd_ms = ((t_first or t_end) - t0) * 1000
        return TTSResult(
            audio_bytes=audio, audio_dur_s=dur,
            tfd_ms=tfd_ms, total_ms=total_ms,
            rtf=total_ms / (dur * 1000) if dur else 0.0,
        )


# ---------------------------------------------------------------------------
# V2V composite
# ---------------------------------------------------------------------------

@dataclass
class V2VResult:
    audio_dur_s: float
    asr_text: str
    tts_audio_dur_s: float
    eos_to_first_audio_ms: float
    asr_finalize_ms: float          # EOS → ASR final (total, for backwards compat)
    vad_silence_ms: float = 0.0     # EOS → VAD endpoint
    asr_finalize_compute_ms: float = 0.0  # VAD endpoint → ASR final
    llm_delay_ms: float = 0.0
    tts_tfd_ms: float = 0.0
    tts_total_ms: float = 0.0

    @property
    def as_dict(self) -> dict:
        return self.__dict__.copy()


def run_v2v(asr: ASRClient, tts: TTSClient, wav_bytes: bytes,
            language_asr: str = "Chinese", language_tts: str = "zh",
            eos_mode: str = "vad", llm_delay_ms: float = 0.0) -> V2VResult:
    """End-to-end voice-to-voice. LLM stage is a sleep placeholder."""
    asr_res = asr.transcribe_streaming(wav_bytes, language_asr, eos_mode)
    t_after_asr = time.monotonic()
    if llm_delay_ms > 0:
        time.sleep(llm_delay_ms / 1000.0)
    tts_res = tts.synthesize(asr_res.text, language_tts)
    return V2VResult(
        audio_dur_s=asr_res.audio_dur_s,
        asr_text=asr_res.text,
        tts_audio_dur_s=tts_res.audio_dur_s,
        # client-side EOS-to-first-audio: ASR finalize + LLM delay + TTS TFD
        eos_to_first_audio_ms=(asr_res.eos_to_final_ms or 0) + llm_delay_ms + tts_res.tfd_ms,
        asr_finalize_ms=asr_res.eos_to_final_ms or 0,
        vad_silence_ms=asr_res.vad_silence_ms or 0,
        asr_finalize_compute_ms=asr_res.asr_finalize_compute_ms or 0,
        llm_delay_ms=llm_delay_ms,
        tts_tfd_ms=tts_res.tfd_ms,
        tts_total_ms=tts_res.total_ms,
    )


# ---------------------------------------------------------------------------
# V2V stream protocol benchmark (ASR-only mode)
# ---------------------------------------------------------------------------

@dataclass
class V2VStreamASRResult:
    """Timings from a /v2v/stream ASR-only session.

    Unlike the composite run_v2v() which chains ASRClient + TTSClient
    through separate connections, this drives the actual /v2v/stream
    protocol end-to-end: config frame → audio chunks → asr_endpoint →
    asr_final. The server naturally emits two messages (endpoint then
    final), giving us the VAD-front-end / ASR-compute split for free.
    """
    text: str
    audio_dur_s: float
    tfd_ms: float | None = None               # first send → first partial
    endpoint_latency_ms: float = 0.0           # EOS → asr_endpoint (VAD front-end)
    asr_finalize_ms: float = 0.0               # asr_endpoint → asr_final (ASR compute)
    total_latency_ms: float = 0.0              # EOS → asr_final (sum of above)
    error: str | None = None

    @property
    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "error"}


def run_v2v_stream_asr(
    base_url: str,
    wav_bytes: bytes,
    language: str = "Chinese",
    chunk_ms: int = 250,
    vad_backend: str = "silero",
    vad_silence_ms: int = 400,
    realtime: bool = True,
    timeout: int = 120,
) -> V2VStreamASRResult:
    """Drive /v2v/stream in ASR-only mode and measure split timings.

    Protocol:
      1. Open WS → send config frame
      2. Pump audio chunks (with realtime pacing)
      3. Send trailing silence for VAD
      4. Record t_endpoint on asr_endpoint message
      5. Record t_final on asr_final message
    """
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    chunks, sr = wav_to_pcm_chunks(wav_bytes, chunk_ms)
    dur = wav_duration_s(wav_bytes)
    chunk_dur = chunk_ms / 1000.0

    ws = websocket.create_connection(
        f"{ws_url}/v2v/stream", timeout=timeout,
    )
    # Stage 1: config frame
    config = {
        "type": "config",
        "asr_language": language,
        "vad": vad_backend,
        "vad_silence_ms": vad_silence_ms,
        "sample_rate": sr,
    }
    ws.send(json.dumps(config))

    t_first_send = time.monotonic()
    t_first_partial: float | None = None

    # Stage 2: pump audio chunks
    ws.settimeout(0.001)
    for c in chunks:
        ws.send_binary(c)
        if t_first_partial is None:
            try:
                msg = ws.recv()
                data = json.loads(msg)
                if data.get("type") == "asr_partial" and data.get("text"):
                    t_first_partial = time.monotonic()
            except websocket.WebSocketTimeoutException:
                pass
        if realtime:
            time.sleep(chunk_dur)

    t_eos = time.monotonic()

    # Stage 3: trailing silence for VAD
    silence_ms = max(vad_silence_ms + chunk_ms, chunk_ms)
    silence_chunks = int(np.ceil(silence_ms / chunk_ms))
    frames_per_chunk = int(sr * chunk_ms / 1000)
    silence = np.zeros(frames_per_chunk, dtype=np.int16).tobytes()

    t_endpoint: float | None = None
    t_final: float | None = None
    final_text = ""
    done = False

    for _ in range(silence_chunks):
        ws.send_binary(silence)
        try:
            msg = ws.recv()
            data = json.loads(msg)
            if data.get("type") == "asr_endpoint":
                t_endpoint = time.monotonic()
            elif data.get("type") == "asr_final":
                t_final = time.monotonic()
                final_text = _coerce_text(data.get("text", ""))
                done = True
                break
        except websocket.WebSocketTimeoutException:
            pass
        if realtime:
            time.sleep(chunk_dur)

    # Stage 4: drain remaining messages
    ws.settimeout(timeout)
    deadline = t_eos + (vad_silence_ms + 30000) / 1000.0
    while not done:
        if time.monotonic() > deadline:
            ws.close()
            return V2VStreamASRResult(
                text="<timeout>", audio_dur_s=dur,
                tfd_ms=((t_first_partial - t_first_send) * 1000) if t_first_partial else None,
                endpoint_latency_ms=(t_endpoint - t_eos) * 1000 if t_endpoint else 0,
                error="timeout waiting for asr_final",
            )
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        if not raw:
            ws.close()
            return V2VStreamASRResult(
                text="<closed>", audio_dur_s=dur,
                error="server closed without asr_final",
            )
        data = json.loads(raw)
        if data.get("type") == "asr_endpoint":
            t_endpoint = time.monotonic()
        elif data.get("type") == "asr_final":
            t_final = time.monotonic()
            final_text = _coerce_text(data.get("text", ""))
            done = True
            break

    ws.close()

    endpoint_latency = (t_endpoint - t_eos) * 1000 if t_endpoint else 0.0
    asr_finalize = (t_final - t_endpoint) * 1000 if (t_final and t_endpoint) else 0.0
    return V2VStreamASRResult(
        text=final_text,
        audio_dur_s=dur,
        tfd_ms=((t_first_partial - t_first_send) * 1000) if t_first_partial else None,
        endpoint_latency_ms=endpoint_latency,
        asr_finalize_ms=asr_finalize,
        total_latency_ms=(t_final - t_eos) * 1000 if t_final else 0.0,
    )
