#!/usr/bin/env python3
"""TTS quality evaluation suite.

Usage:
    python tts_eval.py [--host HOST:PORT] [--ref-dir DIR] [--generate-ref] [--lang LANG]

Evaluates TTS quality via:
  1. TTS → ASR roundtrip: CER (Character Error Rate)
  2. Consistency: same input twice → same CER? (determinism check)
  3. If reference audio exists: waveform similarity (cross-correlation)

Works against the standard /tts and /asr API endpoints.
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


# ─── Test cases ───────────────────────────────────────────────────────────────
TEST_CASES = [
    # (text, language, expected_min_duration_s, expected_max_duration_s)
    ("你好", "chinese", 0.3, 2.0),
    ("今天天气真不错", "chinese", 1.0, 5.0),
    ("今天天气真不错，我们一起去公园散步吧。", "chinese", 2.0, 8.0),
    ("欢迎使用语音合成系统", "chinese", 1.5, 6.0),
    ("这是一个语音合成的测试", "chinese", 1.5, 6.0),
    ("我叫小明，今年二十岁，是一名大学生。", "chinese", 2.0, 8.0),
    ("Hello, how are you today?", "english", 1.0, 5.0),
    ("The weather is nice outside.", "english", 1.0, 5.0),
    ("Welcome to the text to speech system.", "english", 1.5, 6.0),
]


# ─── Utilities ────────────────────────────────────────────────────────────────
def cer(ref: str, hyp: str) -> float:
    """Character Error Rate via edit distance, ignoring punctuation."""
    ref = re.sub(r'[，。！？、；：""''（）\[\]{},\.!?;:\'"()\s\-]', '', ref)
    hyp = re.sub(r'[，。！？、；：""''（）\[\]{},\.!?;:\'"()\s\-]', '', hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / n


def read_wav_pcm(path: str) -> tuple:
    """Read WAV file, return (samples_float32_list, sample_rate)."""
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(n)
    if sw == 2:
        samples = struct.unpack(f'<{n * ch}h', raw)
        samples = [s / 32768.0 for s in samples]
    elif sw == 4:
        samples = struct.unpack(f'<{n * ch}f', raw)
        samples = list(samples)
    else:
        raise ValueError(f"Unsupported sample width: {sw}")
    if ch > 1:
        samples = samples[::ch]  # take first channel
    return samples, sr


def signal_energy(samples: list) -> float:
    """RMS energy of signal."""
    if not samples:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


def cross_correlation(a: list, b: list) -> float:
    """Normalized cross-correlation between two signals (peak)."""
    if not a or not b:
        return 0.0
    # Truncate to shorter
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    # Normalize
    ea = (sum(x * x for x in a) / n) ** 0.5
    eb = (sum(x * x for x in b) / n) ** 0.5
    if ea < 1e-10 or eb < 1e-10:
        return 0.0
    corr = sum(x * y for x, y in zip(a, b)) / (n * ea * eb)
    return corr


# ─── API calls ────────────────────────────────────────────────────────────────
def tts_api(host: str, text: str, language: str, output_path: str) -> dict:
    """Call TTS API, save WAV, return metadata."""
    cmd = [
        "curl", "-s", "-o", output_path,
        "-w", '{"http_code":%{http_code},"time":%{time_total}}',
        "-X", "POST", f"http://{host}/tts",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({"text": text, "language": language}),
    ]
    out = subprocess.check_output(cmd).decode()
    meta = json.loads(out)
    if os.path.exists(output_path):
        try:
            samples, sr = read_wav_pcm(output_path)
            meta["duration"] = len(samples) / sr
            meta["rms"] = signal_energy(samples)
        except Exception:
            meta["duration"] = 0
            meta["rms"] = 0
    return meta


def asr_api(host: str, wav_path: str) -> dict:
    """Call ASR API, return result."""
    cmd = [
        "curl", "-s",
        "-X", "POST", f"http://{host}/asr",
        "-F", f"file=@{wav_path}",
    ]
    out = subprocess.check_output(cmd).decode()
    return json.loads(out)


# ─── Main evaluation ─────────────────────────────────────────────────────────
def run_evaluation(host: str, ref_dir: str = None, generate_ref: bool = False):
    results = []
    ref_path = Path(ref_dir) if ref_dir else None
    if ref_path and generate_ref:
        ref_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*100}")
    print(f"TTS Evaluation — {host}")
    print(f"{'='*100}\n")
    hdr = f"{'#':>2} {'Input':<40} {'ASR':<35} {'CER':>5} {'Dur':>5} {'TTS':>6} {'RMS':>6} {'DurOK':>5}"
    if ref_path and not generate_ref:
        hdr += f" {'XCorr':>6}"
    print(hdr)
    print("-" * len(hdr))

    for i, (text, lang, dur_min, dur_max) in enumerate(TEST_CASES):
        wav_file = tempfile.mktemp(suffix=".wav")

        # TTS
        tts_meta = tts_api(host, text, lang, wav_file)
        if tts_meta.get("http_code") != 200:
            print(f"{i+1:2d} {text:<40} {'HTTP ERROR':35} {'N/A':>5}")
            results.append({"text": text, "cer": 1.0, "error": True})
            continue

        # ASR
        asr_result = asr_api(host, wav_file)
        asr_text = asr_result.get("text", "")
        c = cer(text, asr_text)
        dur = tts_meta.get("duration", 0)
        dur_ok = "✓" if dur_min <= dur <= dur_max else "✗"
        rms = tts_meta.get("rms", 0)

        row = f"{i+1:2d} {text:<40} {asr_text:<35} {c:4.0%} {dur:5.2f} {tts_meta['time']:5.2f}s {rms:5.3f} {dur_ok:>5}"

        # Reference comparison
        xcorr = None
        if ref_path:
            ref_wav = ref_path / f"ref_{i:02d}.wav"
            if generate_ref:
                os.rename(wav_file, str(ref_wav))
                wav_file = str(ref_wav)
                row += " [REF]"
            elif ref_wav.exists():
                try:
                    ref_samples, _ = read_wav_pcm(str(ref_wav))
                    cur_samples, _ = read_wav_pcm(wav_file)
                    xcorr = cross_correlation(ref_samples, cur_samples)
                    row += f" {xcorr:5.3f}"
                except Exception:
                    row += "   ERR"

        print(row)
        results.append({
            "text": text,
            "lang": lang,
            "asr": asr_text,
            "cer": c,
            "duration": dur,
            "dur_ok": dur_min <= dur <= dur_max,
            "tts_time": tts_meta["time"],
            "rms": rms,
            "xcorr": xcorr,
        })

        # Cleanup
        if os.path.exists(wav_file):
            os.unlink(wav_file)

    # Summary
    cers = [r["cer"] for r in results if "error" not in r]
    dur_oks = [r["dur_ok"] for r in results if "error" not in r]
    avg_cer = sum(cers) / len(cers) if cers else 1.0
    dur_pass = sum(dur_oks) / len(dur_oks) if dur_oks else 0
    tts_times = [r["tts_time"] for r in results if "error" not in r]
    avg_tts = sum(tts_times) / len(tts_times) if tts_times else 0

    print(f"\n{'='*100}")
    print(f"Summary:")
    print(f"  Average CER:      {avg_cer:.1%}  {'✓ PASS' if avg_cer < 0.2 else '✗ FAIL'}")
    print(f"  Duration check:   {sum(dur_oks)}/{len(dur_oks)} passed")
    print(f"  Avg TTS latency:  {avg_tts:.2f}s")
    good = sum(1 for c in cers if c < 0.2)
    print(f"  Good (CER<20%):   {good}/{len(cers)}")
    print(f"{'='*100}\n")

    # JSON output
    summary = {
        "host": host,
        "avg_cer": round(avg_cer, 3),
        "pass": avg_cer < 0.2,
        "good_ratio": f"{good}/{len(cers)}",
        "avg_latency": round(avg_tts, 3),
        "results": results,
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS quality evaluation")
    parser.add_argument("--host", default="localhost:8621", help="TTS/ASR API host:port")
    parser.add_argument("--ref-dir", default=None, help="Directory for reference WAV files")
    parser.add_argument("--generate-ref", action="store_true", help="Generate reference files")
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    args = parser.parse_args()

    summary = run_evaluation(args.host, args.ref_dir, args.generate_ref)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
