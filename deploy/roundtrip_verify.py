#!/usr/bin/env python3
"""TTS -> ASR round-trip verifier for deployed OpenVoiceStream services."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def normalize(text: str) -> str:
    drop = set(" \t\r\n。，、！？；：,.!?;:\"'()[]{}<>《》“”‘’")
    return "".join(ch.lower() for ch in text if ch not in drop)


def lcs_similarity(a: str, b: str) -> float:
    a = normalize(a)
    b = normalize(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1] / max(len(a), len(b))


def request_json(url: str, timeout: float) -> dict:
    with OPENER.open(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict, timeout: float) -> bytes:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with OPENER.open(req, timeout=timeout) as resp:
        return resp.read()


def post_multipart_file(url: str, field: str, path: Path, timeout: float) -> dict:
    boundary = f"openvoicestream-{time.time_ns()}"
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + path.read_bytes() + footer
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8621")
    parser.add_argument("--text", default="你好，今天天气真不错。")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--min-sim", type=float, default=0.2)
    parser.add_argument("--timeout-sec", type=float, default=120)
    parser.add_argument("--keep-audio", type=Path)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    try:
        health = request_json(f"{base}/health", args.timeout_sec)
        if not health.get("tts"):
            raise RuntimeError(f"TTS is not ready: {health}")
        if not health.get("asr"):
            raise RuntimeError(f"ASR is not ready: {health}")

        wav = post_json(f"{base}/tts", {"text": args.text}, args.timeout_sec)
        if len(wav) < 1000:
            raise RuntimeError(f"TTS returned too little audio: {len(wav)} bytes")

        if args.keep_audio:
            args.keep_audio.write_bytes(wav)
            wav_path = args.keep_audio
            cleanup = False
        else:
            tmp = NamedTemporaryFile(prefix="openvoicestream-roundtrip-", suffix=".wav", delete=False)
            tmp.write(wav)
            tmp.close()
            wav_path = Path(tmp.name)
            cleanup = True

        try:
            asr = post_multipart_file(f"{base}/asr?language={args.language}", "file", wav_path, args.timeout_sec)
        finally:
            if cleanup:
                wav_path.unlink(missing_ok=True)

        text = str(asr.get("text", "")).strip()
        sim = lcs_similarity(args.text, text)
        result = {
            "url": base,
            "tts_bytes": len(wav),
            "expected": args.text,
            "asr_text": text,
            "similarity": round(sim, 4),
            "health": health,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not text:
            raise RuntimeError("ASR returned empty text")
        if sim < args.min_sim:
            raise RuntimeError(f"similarity {sim:.4f} < required {args.min_sim:.4f}")
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"roundtrip verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
