from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


def _read_codes(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    if data.size % 16 != 0:
        raise ValueError(f"{path} has {data.size} int32 values, not divisible by 16")
    return data.reshape((-1, 16))


def _audio_metrics(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as reader:
        rate = reader.getframerate()
        frames = reader.getnframes()
        raw = reader.readframes(frames)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if samples.size == 0:
        return {"sample_rate": rate, "duration": 0.0, "samples": 0}
    win = max(1, rate // 10)
    rms = []
    zcr = []
    for start in range(0, samples.size, win):
        chunk = samples[start : start + win]
        if chunk.size == 0:
            continue
        rms.append(float(math.sqrt(float(np.mean(chunk * chunk)))))
        zcr.append(float(np.mean(np.signbit(chunk[:-1]) != np.signbit(chunk[1:]))) if chunk.size > 1 else 0.0)
    return {
        "sample_rate": rate,
        "duration": round(frames / rate, 3) if rate else 0.0,
        "samples": int(samples.size),
        "peak": int(np.max(np.abs(samples))),
        "clip_pct": round(float(np.mean(np.abs(samples) >= 32000) * 100.0), 4),
        "rms_mean": round(float(np.mean(rms)), 3) if rms else 0.0,
        "rms_max": round(float(np.max(rms)), 3) if rms else 0.0,
        "zcr_mean": round(float(np.mean(zcr)), 5) if zcr else 0.0,
        "zcr_gt_0_2_windows": int(sum(v > 0.2 for v in zcr)),
    }


def _compare_codes(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    n = min(a.shape[0], b.shape[0])
    common_a = a[:n]
    common_b = b[:n]
    mismatch = common_a != common_b
    mismatch_pos = np.argwhere(mismatch)
    first = mismatch_pos[0].tolist() if mismatch_pos.size else None
    frame_mismatch = np.any(mismatch, axis=1)
    return {
        "frames_a": int(a.shape[0]),
        "frames_b": int(b.shape[0]),
        "common_frames": int(n),
        "all_code_mismatch_count": int(np.sum(mismatch)),
        "frame_mismatch_count": int(np.sum(frame_mismatch)),
        "first_mismatch_frame_group": first,
        "primary_mismatch_count": int(np.sum(common_a[:, 0] != common_b[:, 0])),
        "cp_mismatch_count": int(np.sum(common_a[:, 1:] != common_b[:, 1:])),
        "a_first_frames": common_a[: min(5, n)].tolist(),
        "b_first_frames": common_b[: min(5, n)].tolist(),
    }


def _summarize_dump(prefix: str, dump_dir: Path, wav_path: Path | None) -> dict[str, Any]:
    codes_path = dump_dir / f"{prefix}_all_codes_i32.bin"
    if not codes_path.exists():
        raise FileNotFoundError(f"missing dump file: {codes_path}")
    codes = _read_codes(codes_path)
    primary = codes[:, 0]
    return {
        "prefix": prefix,
        "dump_dir": str(dump_dir),
        "codes_path": str(codes_path),
        "frames": int(codes.shape[0]),
        "primary_first_20": primary[:20].tolist(),
        "primary_unique": int(np.unique(primary).size),
        "cp_unique_per_group": [int(np.unique(codes[:, idx]).size) for idx in range(1, 16)],
        "audio": _audio_metrics(wav_path) if wav_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and compare Qwen3-TTS layer dumps.")
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--prefix-a", required=True)
    parser.add_argument("--prefix-b")
    parser.add_argument("--wav-a")
    parser.add_argument("--wav-b")
    parser.add_argument("--output")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    result: dict[str, Any] = {
        "a": _summarize_dump(args.prefix_a, dump_dir, Path(args.wav_a) if args.wav_a else None)
    }
    if args.prefix_b:
        result["b"] = _summarize_dump(args.prefix_b, dump_dir, Path(args.wav_b) if args.wav_b else None)
        codes_a = _read_codes(dump_dir / f"{args.prefix_a}_all_codes_i32.bin")
        codes_b = _read_codes(dump_dir / f"{args.prefix_b}_all_codes_i32.bin")
        result["code_diff"] = _compare_codes(codes_a, codes_b)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

