"""Raspberry Pi 5 benchmark for Paraformer streaming ASR + Matcha TTS.

Modes:
  asr  -- run streaming ASR over a fixed wav N times, report RTF.
  tts  -- run Matcha TTS over fixed prompts N times, report RTF.
  v2v  -- end-to-end: WS push wav -> wait final -> POST /tts/stream -> first audio byte latency.

All output is structured JSON to stdout. Designed to run inside the docker container
(or against http://localhost:8621 from inside it).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

# Make app modules importable when run inside the container.
sys.path.insert(0, "/opt/speech/app")


def _make_test_wav(path: str, duration_s: float = 6.0, sr: int = 16000) -> str:
    """Synthesize a sine-wave wav file (mono int16) for ASR benchmarking."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    # Mix two tones so paraformer at least has frequency content.
    sig = 0.2 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 660.0 * t)
    pcm = (sig * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def _load_wav_pcm(path: str) -> tuple[np.ndarray, int]:
    """Load a wav file into mono float32 in [-1,1], return (samples, sample_rate)."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported sample width: {sw}")
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1)
    return arr, sr


def _stats(times: list[float]) -> dict:
    return {
        "n": len(times),
        "mean": round(statistics.fmean(times), 4),
        "p50": round(statistics.median(times), 4),
        "p95": round(sorted(times)[max(0, int(len(times) * 0.95) - 1)], 4)
        if len(times) >= 2
        else round(times[0], 4),
        "min": round(min(times), 4),
        "max": round(max(times), 4),
    }


def bench_asr(n: int = 5) -> dict:
    from backends.sherpa_asr import SherpaASRBackend  # type: ignore

    wav_path = "/tmp/rpi_bench_asr.wav"
    duration_s = 6.0
    if not os.path.exists(wav_path):
        _make_test_wav(wav_path, duration_s=duration_s, sr=16000)
    samples, sr = _load_wav_pcm(wav_path)
    audio_dur = len(samples) / sr

    be = SherpaASRBackend()
    be.preload()

    # Warmup
    stream = be.create_stream(language="auto")
    stream.accept_waveform(sr, samples[: sr // 2])
    stream.prepare_finalize()
    _ = stream.finalize()

    decode_times = []
    rtfs = []
    finals = []
    for _ in range(n):
        stream = be.create_stream(language="auto")
        t0 = time.perf_counter()
        # Feed in 100ms chunks to mimic streaming use.
        chunk = int(sr * 0.1)
        for off in range(0, len(samples), chunk):
            stream.accept_waveform(sr, samples[off : off + chunk])
            stream.get_partial()
        stream.prepare_finalize()
        text = stream.finalize()
        elapsed = time.perf_counter() - t0
        decode_times.append(elapsed)
        rtfs.append(elapsed / audio_dur)
        finals.append(text)

    return {
        "mode": "asr",
        "audio_duration_s": round(audio_dur, 3),
        "decode_seconds": _stats(decode_times),
        "rtf": _stats(rtfs),
        "final_sample": finals[0] if finals else "",
    }


def bench_tts(n: int = 5) -> dict:
    from backends.sherpa import SherpaBackend  # type: ignore

    be = SherpaBackend()
    be.preload()

    prompts = [
        ("zh", "今天天气真好，我们一起出去走走"),
        ("en", "Hello, this is a test of the text to speech system"),
    ]

    results = {}
    for lang, text in prompts:
        synth_times = []
        rtfs = []
        audio_durs = []
        for _ in range(n):
            t0 = time.perf_counter()
            wav_bytes, meta = be.synthesize(text, language=lang)
            elapsed = time.perf_counter() - t0
            # Decode wav header to get duration.
            try:
                with wave.open(io.BytesIO(wav_bytes), "rb") as w:
                    dur = w.getnframes() / float(w.getframerate())
            except Exception:
                dur = float(meta.get("duration", 0)) if isinstance(meta, dict) else 0.0
            synth_times.append(elapsed)
            audio_durs.append(dur)
            if dur > 0:
                rtfs.append(elapsed / dur)
        results[lang] = {
            "text": text,
            "synth_seconds": _stats(synth_times),
            "audio_duration_s": _stats(audio_durs),
            "rtf": _stats(rtfs) if rtfs else None,
        }
    return {"mode": "tts", "results": results}


async def _v2v_once(host: str, wav_path: str) -> dict:
    import websockets  # type: ignore
    import requests  # type: ignore

    samples, sr = _load_wav_pcm(wav_path)
    pcm16 = (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()
    chunk_ms = 100
    chunk_bytes = int(sr * chunk_ms / 1000) * 2  # int16 = 2 bytes/sample

    ws_url = f"ws://{host}/asr/stream?sample_rate={sr}"
    final_text = None
    t_start = time.perf_counter()
    t_eos = None
    async with websockets.connect(ws_url, max_size=None) as ws:
        async def feeder():
            for off in range(0, len(pcm16), chunk_bytes):
                await ws.send(pcm16[off : off + chunk_bytes])
                await asyncio.sleep(chunk_ms / 1000.0)
            await ws.send(b"")  # EOS

        feed_task = asyncio.create_task(feeder())
        try:
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    continue
                data = json.loads(msg)
                if data.get("type") == "final" or data.get("is_final"):
                    final_text = data.get("text", "") or ""
                    t_eos = time.perf_counter()
                    break
        except Exception as e:
            # WS may close right after sending final.
            if final_text is None:
                return {"error": f"ws recv failed before final: {e}"}
        feed_task.cancel()
        try:
            await feed_task
        except BaseException:
            pass

    if not final_text:
        return {"error": "no final text from ASR", "elapsed_s": time.perf_counter() - t_start}

    # POST /tts/stream and time first audio byte after the 4-byte sample_rate header.
    t_post = time.perf_counter()
    # Pass explicit sid/speed; sherpa generate_streaming() does not coerce None.
    r = requests.post(
        f"http://{host}/tts/stream",
        json={"text": final_text, "sid": 0, "speed": 1.0, "pitch": 0.0},
        stream=True,
        timeout=60,
    )
    r.raise_for_status()
    first_byte_t = None
    header_consumed = 0
    for raw in r.iter_content(chunk_size=512):
        if not raw:
            continue
        if header_consumed < 4:
            header_consumed += len(raw)
            # If first chunk already includes audio bytes past the 4B SR header,
            # treat now as first audio time. Otherwise wait for next chunk.
            if header_consumed > 4:
                first_byte_t = time.perf_counter()
                break
            continue
        first_byte_t = time.perf_counter()
        break
    r.close()

    return {
        "asr_final_text": final_text,
        "eos_to_first_audio_ms": round((first_byte_t - t_eos) * 1000, 2)
        if first_byte_t and t_eos
        else None,
        "post_to_first_audio_ms": round((first_byte_t - t_post) * 1000, 2)
        if first_byte_t
        else None,
        "asr_total_ms": round((t_eos - t_start) * 1000, 2) if t_eos else None,
    }


def bench_v2v(n: int = 3, host: str = "localhost:8621") -> dict:
    # Prefer a real speech sample shipped with paraformer test_wavs.
    candidates = [
        "/opt/models/paraformer-streaming/test_wavs/0.wav",
        "/opt/models/paraformer-streaming/test_wavs/1.wav",
        "/tmp/rpi_bench_v2v.wav",
    ]
    wav_path = next((p for p in candidates if os.path.exists(p)), candidates[-1])
    if not os.path.exists(wav_path):
        _make_test_wav(wav_path, duration_s=4.0, sr=16000)

    runs = []
    for i in range(n):
        try:
            res = asyncio.run(_v2v_once(host, wav_path))
        except Exception as e:
            res = {"error": str(e)}
        runs.append(res)
        time.sleep(0.5)

    eos_lat = [r["eos_to_first_audio_ms"] for r in runs if r.get("eos_to_first_audio_ms") is not None]
    return {
        "mode": "v2v",
        "n": n,
        "runs": runs,
        "eos_to_first_audio_ms_stats": _stats(eos_lat) if eos_lat else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["asr", "tts", "v2v"], required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--host", default="localhost:8621")
    args = p.parse_args()

    if args.mode == "asr":
        out = bench_asr(n=args.n)
    elif args.mode == "tts":
        out = bench_tts(n=args.n)
    else:
        out = bench_v2v(n=args.n, host=args.host)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
