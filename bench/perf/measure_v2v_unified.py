"""Measure end-to-end latency on the unified WS `/v2v/stream` endpoint.

Complements `bench/measure_v2v.py`, which measures the legacy split
pipeline (POST /asr/stream → POST /tts/stream). This script exercises
the new unified WebSocket and reports:

  • tfd_ms                — first PCM chunk sent → first asr_partial
  • stop_to_endpoint_ms   — last PCM chunk → asr_endpoint  (VAD pre-fire)
  • stop_to_final_ms      — last PCM chunk → asr_final
  • final_to_tts_audio_ms — asr_final → first TTS PCM frame
  • stop_to_tts_audio_ms  — last PCM chunk → first TTS PCM frame (E2E)

Multi-utterance mode (`--multi`) measures per-utterance stop→final by
playing the same WAV N times back-to-back with synthesized silence
between them. Each utterance must produce its own mid-session
asr_final with session_complete=false; the final asr_eos triggers
session_complete=true.

Usage:
    uv run --with websocket-client --with soundfile --with numpy \
        python bench/measure_v2v_unified.py \
        --host 100.89.94.11:8621 \
        --wav tests/data/short_zh.wav \
        --runs 5 \
        --tts            # enable round-trip TTS measurement
        --multi 3        # play wav 3x in one session

Output: JSON to stdout (one record per run) + summary table to stderr.
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import soundfile as sf
import websocket


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_wav_16k_i16(path: str) -> tuple[bytes, float]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        ratio = 16000 / sr
        new_len = int(len(audio) * ratio)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        )
    duration = len(audio) / 16000
    i16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    return i16, duration


def make_silence(ms: int) -> bytes:
    n = int(16000 * ms / 1000)
    return b"\x00" * (n * 2)


# ---------------------------------------------------------------------------
# Per-run result
# ---------------------------------------------------------------------------

@dataclass
class Utterance:
    """Latencies for one utterance within a session."""
    tfd_ms: float | None = None
    stop_to_endpoint_ms: float | None = None
    stop_to_final_ms: float | None = None
    final_text: str = ""
    session_complete: bool | None = None


@dataclass
class RunResult:
    audio_dur_s: float
    utterances: list[Utterance] = field(default_factory=list)
    final_to_tts_audio_ms: float | None = None
    stop_to_tts_audio_ms: float | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_once(
    host: str,
    wav_pcm: bytes,
    audio_dur_s: float,
    *,
    language: str = "Chinese",
    tts_enabled: bool = False,
    tts_language: str = "auto",
    multi_count: int = 1,
    silence_ms: int = 700,
    realtime: bool = True,
    chunk_ms: int = 100,
    timeout: float = 30.0,
) -> RunResult:
    """One V2V session. If multi_count > 1, plays the wav N times with
    silence_ms gaps between, expecting N mid-session finals + 1 closing
    final."""
    multi_utterance = multi_count > 1
    cfg: dict[str, Any] = {
        "type": "config",
        "asr_language": language,
        "sample_rate": 16000,
        "vad": "silero",
        "vad_silence_ms": 500,
        "multi_utterance": multi_utterance,
    }
    if tts_enabled:
        cfg["tts_language"] = tts_language

    result = RunResult(audio_dur_s=audio_dur_s * multi_count)

    ws_url = f"ws://{host}/v2v/stream"
    ws = websocket.create_connection(ws_url, timeout=timeout)
    try:
        ws.send(json.dumps(cfg))

        chunk_bytes = int(16000 * chunk_ms / 1000) * 2
        silence_pcm = make_silence(silence_ms)
        chunk_dur_s = chunk_ms / 1000.0

        # Per-utterance state for sender side
        utt_pending = [Utterance() for _ in range(multi_count)]
        utt_idx_send = 0   # which utterance is currently being sent
        t_stops: list[float] = []        # filled as each utterance's audio ends
        t_first_send: float | None = None
        t_session_stop: float | None = None  # last byte of last utterance

        # Sender loop: send all utterances, recording stop time per utterance.
        # The reader runs in the main thread interleaved with sends via short
        # socket timeouts (same pattern as bench/perf/client.py).
        def send_audio(pcm: bytes, mark_stop_for: int | None = None):
            nonlocal t_first_send
            for off in range(0, len(pcm), chunk_bytes):
                chunk = pcm[off:off + chunk_bytes]
                if not chunk:
                    break
                ws.send_binary(chunk)
                if t_first_send is None:
                    t_first_send = time.monotonic()
                if realtime:
                    time.sleep(chunk_dur_s)
                # Opportunistic read for partial / endpoint frames while
                # streaming audio (so tfd / endpoint measurements include
                # mid-stream events).
                _drain_nonblocking(ws, t_first_send, utt_pending, utt_idx_send,
                                   t_stops, result)
            if mark_stop_for is not None:
                t_stops.append(time.monotonic())

        for i in range(multi_count):
            utt_idx_send = i
            send_audio(wav_pcm, mark_stop_for=i)
            if i < multi_count - 1:
                # Inject silence so VAD fires SPEECH_END between utterances.
                # No stop mark here — silence is not "user said stop".
                send_audio(silence_pcm, mark_stop_for=None)

        t_session_stop = t_stops[-1]

        # End the session.
        ws.send(json.dumps({"type": "asr_eos"}))
        if tts_enabled:
            # Wait for asr_final, then echo it to TTS to measure round-trip.
            pass  # handled in reader

        # Drain everything until session_complete (multi) or final (single)
        # plus optional TTS audio.
        ws.settimeout(timeout)
        got_session_complete = False     # set True on asr_final (single mode)
                                         # or session_complete=true (multi mode)
        utt_idx_recv = 0
        t_final_for_tts: float | None = None
        tts_audio_first: float | None = None
        tts_done = not tts_enabled

        while not (got_session_complete and tts_done):
            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                result.error = "timeout"
                break
            if msg is None or msg == "":
                result.error = "server closed"
                break
            if isinstance(msg, bytes):
                if tts_enabled and tts_audio_first is None and len(msg) > 4:
                    # First TTS binary frame: 4-byte sample rate header.
                    tts_audio_first = time.monotonic()
                    if t_final_for_tts is not None:
                        result.final_to_tts_audio_ms = (tts_audio_first - t_final_for_tts) * 1000
                    if t_session_stop is not None:
                        result.stop_to_tts_audio_ms = (tts_audio_first - t_session_stop) * 1000
                continue
            try:
                data = json.loads(msg)
            except (ValueError, TypeError):
                continue
            t = data.get("type")
            now = time.monotonic()
            if t == "asr_partial":
                u = utt_pending[utt_idx_recv]
                if u.tfd_ms is None and t_first_send is not None:
                    u.tfd_ms = (now - t_first_send) * 1000
            elif t == "asr_endpoint":
                u = utt_pending[utt_idx_recv]
                if utt_idx_recv < len(t_stops):
                    u.stop_to_endpoint_ms = (now - t_stops[utt_idx_recv]) * 1000
            elif t == "asr_final":
                u = utt_pending[utt_idx_recv]
                u.final_text = data.get("text", "") or ""
                u.session_complete = data.get("session_complete")
                if utt_idx_recv < len(t_stops):
                    u.stop_to_final_ms = (now - t_stops[utt_idx_recv]) * 1000
                if t_final_for_tts is None:
                    # First final triggers TTS in our test harness (when
                    # tts_enabled): echo it back as text.
                    t_final_for_tts = now
                    if tts_enabled and u.final_text:
                        ws.send(json.dumps({"type": "text", "text": u.final_text}))
                        ws.send(json.dumps({"type": "tts_flush"}))
                if multi_utterance:
                    if u.session_complete:
                        got_session_complete = True
                    else:
                        utt_idx_recv += 1
                else:
                    got_session_complete = True
            elif t == "tts_done":
                tts_done = True
            elif t == "error":
                result.error = data.get("error", "(no detail)")
                break

        # Truncate to actually-received utterances (the trailing
        # session_complete=true final lands on the last pending slot).
        result.utterances = utt_pending[:max(1, utt_idx_recv + 1)]
    finally:
        try: ws.close()
        except Exception: pass

    return result


def _drain_nonblocking(ws, t_first_send, utt_pending, utt_idx_send,
                       t_stops, result):
    """Read whatever the server has queued without blocking. Updates
    utt_pending[utt_idx_send] in place for tfd / endpoint observed mid-stream.
    """
    ws.settimeout(0.001)
    try:
        while True:
            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                return
            if isinstance(msg, bytes):
                continue   # TTS audio handled by main loop
            if not msg:
                return
            try:
                data = json.loads(msg)
            except (ValueError, TypeError):
                continue
            t = data.get("type")
            now = time.monotonic()
            if t == "asr_partial":
                u = utt_pending[utt_idx_send]
                if u.tfd_ms is None and t_first_send is not None:
                    u.tfd_ms = (now - t_first_send) * 1000
            elif t == "asr_endpoint":
                u = utt_pending[utt_idx_send]
                if utt_idx_send < len(t_stops):
                    u.stop_to_endpoint_ms = (now - t_stops[utt_idx_send]) * 1000
    finally:
        # Caller resets timeout when needed.
        pass


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def print_summary(records: list[RunResult], tts_enabled: bool):
    metrics: dict[str, list[float]] = {
        "tfd_ms": [],
        "stop_to_endpoint_ms": [],
        "stop_to_final_ms": [],
    }
    if tts_enabled:
        metrics["final_to_tts_audio_ms"] = []
        metrics["stop_to_tts_audio_ms"] = []

    for r in records:
        if r.error:
            continue
        for u in r.utterances:
            if u.tfd_ms is not None:                metrics["tfd_ms"].append(u.tfd_ms)
            if u.stop_to_endpoint_ms is not None:   metrics["stop_to_endpoint_ms"].append(u.stop_to_endpoint_ms)
            if u.stop_to_final_ms is not None:      metrics["stop_to_final_ms"].append(u.stop_to_final_ms)
        if tts_enabled:
            if r.final_to_tts_audio_ms is not None: metrics["final_to_tts_audio_ms"].append(r.final_to_tts_audio_ms)
            if r.stop_to_tts_audio_ms is not None:  metrics["stop_to_tts_audio_ms"].append(r.stop_to_tts_audio_ms)

    print("\n=== /v2v/stream latency summary ===", file=sys.stderr)
    print(f"{'metric':<26} {'n':>3} {'mean':>8} {'p50':>8} {'min':>8} {'max':>8}",
          file=sys.stderr)
    for k, vs in metrics.items():
        s = summarize(vs)
        if not s:
            print(f"{k:<26}  -- no samples --", file=sys.stderr)
            continue
        print(f"{k:<26} {s['n']:>3} {s['mean']:>7.0f}ms {s['p50']:>7.0f}ms "
              f"{s['min']:>7.0f}ms {s['max']:>7.0f}ms", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True, help="host:port (no scheme)")
    ap.add_argument("--wav", required=True, help="WAV file (mono, any sample rate)")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1, help="discard first N runs")
    ap.add_argument("--tts", action="store_true", help="enable TTS round-trip")
    ap.add_argument("--tts-language", default="auto")
    ap.add_argument("--multi", type=int, default=1,
                    help="utterances per session (>1 enables multi_utterance)")
    ap.add_argument("--silence-ms", type=int, default=700,
                    help="silence injected between utterances (multi mode)")
    ap.add_argument("--no-realtime", action="store_true",
                    help="send audio as fast as possible (default: real-time pacing)")
    args = ap.parse_args()

    wav_pcm, dur = load_wav_16k_i16(args.wav)
    print(f"wav: {args.wav}  dur={dur:.2f}s  size={len(wav_pcm)} bytes  "
          f"multi={args.multi}  tts={args.tts}", file=sys.stderr)

    total = args.runs + args.warmup
    records: list[RunResult] = []
    for i in range(total):
        tag = "warmup" if i < args.warmup else "run"
        try:
            r = run_once(
                args.host, wav_pcm, dur,
                language=args.language,
                tts_enabled=args.tts,
                tts_language=args.tts_language,
                multi_count=args.multi,
                silence_ms=args.silence_ms,
                realtime=not args.no_realtime,
            )
        except Exception as e:
            r = RunResult(audio_dur_s=dur, error=f"{type(e).__name__}: {e}")
        print(f"[{tag} {i+1}/{total}] err={r.error or '-'} "
              f"utts={len(r.utterances)} "
              f"first_final={r.utterances[0].stop_to_final_ms if r.utterances else None}",
              file=sys.stderr)
        if i >= args.warmup:
            records.append(r)
        # Emit per-run JSON to stdout
        sys.stdout.write(json.dumps(asdict(r)) + "\n")
        sys.stdout.flush()

    print_summary(records, args.tts)


if __name__ == "__main__":
    main()
