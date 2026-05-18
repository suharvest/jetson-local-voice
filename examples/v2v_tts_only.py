#!/usr/bin/env python3
"""Minimal /v2v/stream TTS-only WebSocket client.

Install dependency:
    uv run --with websockets python examples/v2v_tts_only.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

import websockets


def chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def write_wav(path: Path, sample_rate: int, pcm_parts: list[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(pcm_parts))


async def run(args: argparse.Namespace) -> None:
    sample_rate: int | None = None
    pcm_parts: list[bytes] = []
    async with websockets.connect(args.url, open_timeout=args.timeout_sec) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "config",
                    "tts_language": args.language,
                    "sample_rate": args.sample_rate,
                    "multi_utterance": False,
                }
            )
        )
        for part in chunks(args.text, args.chunk_chars):
            await ws.send(json.dumps({"type": "text", "text": part}, ensure_ascii=False))
        await ws.send(json.dumps({"type": "tts_flush"}))

        async for msg in ws:
            if isinstance(msg, bytes):
                if sample_rate is None:
                    if len(msg) < 4:
                        raise RuntimeError("missing sample-rate prefix")
                    sample_rate = int.from_bytes(msg[:4], "little")
                    if len(msg) > 4:
                        pcm_parts.append(msg[4:])
                else:
                    pcm_parts.append(msg)
                continue
            event = json.loads(msg)
            if event.get("type") == "error":
                raise RuntimeError(event.get("error") or event)
            if event.get("type") == "tts_done":
                break

    if sample_rate is None:
        raise RuntimeError("server returned no audio sample-rate prefix")
    pcm = b"".join(pcm_parts)
    if len(pcm) < 1000:
        raise RuntimeError(f"TTS stream returned too little audio: {len(pcm)} bytes")
    write_wav(args.out, sample_rate, pcm_parts)
    duration = len(pcm) / 2 / sample_rate
    print(f"Wrote {args.out} ({sample_rate} Hz, {duration:.2f} s, {len(pcm)} PCM bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8621/v2v/stream")
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", type=Path, default=Path("v2v-tts.wav"))
    parser.add_argument("--language", default="zh")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-chars", type=int, default=12)
    parser.add_argument("--timeout-sec", type=float, default=30)
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
