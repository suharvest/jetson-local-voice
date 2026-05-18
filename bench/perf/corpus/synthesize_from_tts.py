#!/usr/bin/env python3
"""Bootstrap corpus by synthesizing each transcript via the live TTS endpoint.

Run once on a stable device (Jetson with voice_clone preset) to generate
20 deterministic WAV files, then commit the resulting SHA256 fingerprints
to manifest.json. Every other device fetches the same bytes via fetch.py.

Usage:
  python synthesize_from_tts.py --base-url http://localhost:8621
"""
from __future__ import annotations
import argparse, json, sys, wave
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"


def synthesize(base_url: str, text: str, lang: str, voice: str | None) -> bytes:
    payload = {"text": text, "language": lang}
    if voice:
        payload["voice"] = voice
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/tts",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8621")
    ap.add_argument("--voice", default=None, help="TTS voice id (optional)")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    n = 0
    for entry in manifest["files"]:
        dst = ROOT / entry["filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {entry['id']:14s}  {entry['transcript'][:40]}...")
        try:
            audio = synthesize(args.base_url, entry["transcript"], entry["lang"], args.voice)
        except Exception as e:
            print(f"    FAIL: {e}")
            continue
        dst.write_bytes(audio)
        # Patch duration in manifest from actual WAV header
        try:
            with wave.open(str(dst), "rb") as wf:
                entry["duration_s"] = round(wf.getnframes() / wf.getframerate(), 2)
        except Exception:
            pass
        n += 1

    print(f"\nSynthesized {n}/{len(manifest['files'])} files.")
    print("Next: python fetch.py --recompute-hashes  &&  git add -p manifest.json")
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
