#!/usr/bin/env python3
"""Call /tts/stream and write the returned PCM stream as a WAV file."""

from __future__ import annotations

import argparse
import json
import urllib.request
import wave
from pathlib import Path


def stream_tts(base_url: str, text: str, timeout: float) -> tuple[int, bytes]:
    payload = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/tts/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        prefix = resp.read(4)
        if len(prefix) != 4:
            raise RuntimeError("missing 4-byte sample-rate prefix")
        sample_rate = int.from_bytes(prefix, "little")
        pcm = resp.read()
    return sample_rate, pcm


def write_wav(path: Path, sample_rate: int, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8621")
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", type=Path, default=Path("tts.wav"))
    parser.add_argument("--timeout-sec", type=float, default=180)
    args = parser.parse_args()

    sample_rate, pcm = stream_tts(args.url, args.text, args.timeout_sec)
    if len(pcm) < 1000:
        raise RuntimeError(f"TTS stream returned too little audio: {len(pcm)} bytes")
    write_wav(args.out, sample_rate, pcm)
    duration = len(pcm) / 2 / sample_rate
    print(f"Wrote {args.out} ({sample_rate} Hz, {duration:.2f} s, {len(pcm)} PCM bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
