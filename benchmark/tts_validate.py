#!/usr/bin/env python3
"""
TTS Pipeline Step-by-Step Validator
=====================================
Compares a target TTS implementation against official reference data,
step by step AND end-to-end. Produces a diagnostic report showing
exactly where the pipeline diverges.

Two modes:
  1. File mode: compare pre-dumped .npy files from target vs reference
  2. API mode:  call /tts endpoint, run ASR, compare audio e2e

Usage:
    # Generate reference first (on WSL2 with official model):
    python tts_dump_reference.py --output-dir ./tts_ref_dump --seed 42

    # Then dump target intermediates (your implementation saves .npy files):
    # ... (implementation-specific)

    # Compare:
    python tts_validate.py \
        --ref-dir ./tts_ref_dump \
        --target-dir ./tts_target_dump \
        [--api-host localhost:8000] \
        [--case 0]

Report output:
    ✓ token_ids         MATCH (9 tokens)
    ✓ text_embeds       MATCH (cos=1.0000, max_diff=2.3e-6)
    ✓ prefill_logits    MATCH (cos=0.9998, max_diff=0.03)
    ✗ prefill_hidden    DIVERGE (cos=0.8912, max_diff=1.2)
      → First divergence at position 5, dim 234
    ✗ primary_codes     DIVERGE (3/10 mismatch, first at frame 2)
    ...
    E2E: CER=35% FAIL (reference: "你好世界", target ASR: "你好失败")
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

import numpy as np


# ─── Test cases (same as tts_eval.py / tts_dump_reference.py) ─────────────────
TEST_CASES = [
    ("你好", "chinese"),
    ("今天天气真不错", "chinese"),
    ("今天天气真不错，我们一起去公园散步吧。", "chinese"),
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


def cosine_sim(a, b):
    """Cosine similarity between two flat arrays."""
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_npy(path):
    """Load .npy file, return None if missing."""
    if os.path.exists(path):
        return np.load(path)
    return None


def read_wav_pcm(path):
    """Read WAV file, return (samples_float32, sample_rate)."""
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if ch > 1:
        samples = samples.reshape(-1, ch)[:, 0]
    return samples, sr


def tts_api(host, text, language, wav_path):
    """Call /tts API, save WAV, return metadata."""
    cmd = [
        "curl", "-s", "-o", wav_path,
        "-w", '{"http_code":%{http_code},"time":%{time_total}}',
        "-X", "POST", f"http://{host}/tts",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({"text": text, "language": language}),
    ]
    out = subprocess.check_output(cmd).decode()
    return json.loads(out)


def asr_api(host, wav_path):
    """Call /asr API, return transcription text."""
    cmd = [
        "curl", "-s",
        "-X", "POST", f"http://{host}/asr",
        "-F", f"file=@{wav_path}",
    ]
    out = subprocess.check_output(cmd).decode()
    try:
        return json.loads(out).get("text", "")
    except json.JSONDecodeError:
        return out.strip()


# ─── Comparison functions ─────────────────────────────────────────────────────

class StepResult:
    def __init__(self, name, passed, detail="", extra=None):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.extra = extra or {}

    def __str__(self):
        icon = "  PASS" if self.passed else "  FAIL"
        return f"{'✓' if self.passed else '✗'} {self.name:<25s}{icon}  {self.detail}"


def compare_int_array(name, ref, tgt, label=""):
    """Compare integer arrays (token_ids, codes, etc.)."""
    if ref is None:
        return StepResult(name, False, "reference missing")
    if tgt is None:
        return StepResult(name, False, "target missing")

    ref, tgt = ref.flatten(), tgt.flatten()
    if ref.shape != tgt.shape:
        return StepResult(name, False,
            f"shape mismatch: ref={ref.shape} vs tgt={tgt.shape}")

    match = np.array_equal(ref, tgt)
    if match:
        return StepResult(name, True, f"MATCH ({len(ref)} {label})")
    else:
        diff_idx = np.where(ref != tgt)[0]
        n_diff = len(diff_idx)
        first = diff_idx[0]
        return StepResult(name, False,
            f"DIVERGE ({n_diff}/{len(ref)} mismatch, first at idx {first}: "
            f"ref={ref[first]} vs tgt={tgt[first]})")


def compare_float_array(name, ref, tgt, cos_thresh=0.99, max_diff_thresh=1.0):
    """Compare float arrays with cosine similarity and max absolute diff."""
    if ref is None:
        return StepResult(name, False, "reference missing")
    if tgt is None:
        return StepResult(name, False, "target missing")

    if ref.shape != tgt.shape:
        # Try to handle minor shape differences (e.g. trailing padding)
        min_len = min(ref.shape[0], tgt.shape[0]) if ref.ndim == 1 else None
        if min_len and ref.ndim == 1:
            ref, tgt = ref[:min_len], tgt[:min_len]
        elif ref.ndim >= 2 and ref.shape[1:] == tgt.shape[1:]:
            min_len = min(ref.shape[0], tgt.shape[0])
            ref, tgt = ref[:min_len], tgt[:min_len]
        else:
            return StepResult(name, False,
                f"shape mismatch: ref={ref.shape} vs tgt={tgt.shape}")

    cos = cosine_sim(ref, tgt)
    max_diff = float(np.max(np.abs(ref.astype(np.float64) - tgt.astype(np.float64))))
    mean_diff = float(np.mean(np.abs(ref.astype(np.float64) - tgt.astype(np.float64))))

    passed = cos >= cos_thresh and max_diff <= max_diff_thresh
    detail = f"cos={cos:.4f}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}"
    if not passed:
        # Find first divergence location
        diff = np.abs(ref.astype(np.float64) - tgt.astype(np.float64))
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        detail += f"  worst@{idx}"

    return StepResult(name, passed, detail,
                      extra={"cos": cos, "max_diff": max_diff, "mean_diff": mean_diff})


def compare_codes_2d(name, ref, tgt):
    """Compare 2D code arrays [n_frames, n_groups]."""
    if ref is None:
        return StepResult(name, False, "reference missing")
    if tgt is None:
        return StepResult(name, False, "target missing")

    min_frames = min(ref.shape[0], tgt.shape[0])
    if ref.shape[1:] != tgt.shape[1:]:
        return StepResult(name, False,
            f"shape mismatch: ref={ref.shape} vs tgt={tgt.shape}")

    ref_c, tgt_c = ref[:min_frames], tgt[:min_frames]
    match_mask = (ref_c == tgt_c)
    frame_match = match_mask.all(axis=1)
    n_frame_match = int(frame_match.sum())

    if np.array_equal(ref_c, tgt_c):
        return StepResult(name, True,
            f"MATCH ({min_frames} frames × {ref.shape[1]} groups)")

    # Detailed divergence info
    first_bad_frame = int(np.where(~frame_match)[0][0])
    first_bad_group = int(np.where(~match_mask[first_bad_frame])[0][0])
    per_group_match = [int(match_mask[:, g].sum()) for g in range(ref.shape[1])]
    worst_group = int(np.argmin(per_group_match))

    return StepResult(name, False,
        f"DIVERGE (frames: {n_frame_match}/{min_frames} match, "
        f"first bad: frame {first_bad_frame} group {first_bad_group}, "
        f"worst group: {worst_group} ({per_group_match[worst_group]}/{min_frames}))")


# ─── Per-frame comparison ────────────────────────────────────────────────────

def compare_frames(ref_dir, tgt_dir, n_frames=5):
    """Compare per-frame intermediates."""
    results = []
    ref_frames = os.path.join(ref_dir, "frames")
    tgt_frames = os.path.join(tgt_dir, "frames")

    if not os.path.isdir(ref_frames):
        return [StepResult("per-frame", False, "reference frames/ dir missing")]
    if not os.path.isdir(tgt_frames):
        return [StepResult("per-frame", False, "target frames/ dir missing")]

    for i in range(n_frames):
        prefix = f"frame_{i}"

        # Talker input embed
        ref_inp = load_npy(os.path.join(ref_frames, f"{prefix}_input_embed.npy"))
        tgt_inp = load_npy(os.path.join(tgt_frames, f"{prefix}_input_embed.npy"))
        if ref_inp is not None or tgt_inp is not None:
            results.append(compare_float_array(
                f"  frame[{i}] input_embed", ref_inp, tgt_inp, 0.99, 0.5))

        # Talker hidden
        ref_h = load_npy(os.path.join(ref_frames, f"{prefix}_talker_hidden.npy"))
        tgt_h = load_npy(os.path.join(tgt_frames, f"{prefix}_talker_hidden.npy"))
        if ref_h is not None or tgt_h is not None:
            results.append(compare_float_array(
                f"  frame[{i}] talker_hidden", ref_h, tgt_h, 0.98, 2.0))

        # Talker logits
        ref_l = load_npy(os.path.join(ref_frames, f"{prefix}_talker_logits.npy"))
        tgt_l = load_npy(os.path.join(tgt_frames, f"{prefix}_talker_logits.npy"))
        if ref_l is not None or tgt_l is not None:
            results.append(compare_float_array(
                f"  frame[{i}] talker_logits", ref_l, tgt_l, 0.98, 5.0))

        # CP step logits
        ref_cp = load_npy(os.path.join(ref_frames, f"{prefix}_cp_step_logits.npy"))
        tgt_cp = load_npy(os.path.join(tgt_frames, f"{prefix}_cp_step_logits.npy"))
        if ref_cp is not None or tgt_cp is not None:
            results.append(compare_float_array(
                f"  frame[{i}] cp_logits", ref_cp, tgt_cp, 0.95, 10.0))

    return results


# ─── E2E comparison via API ──────────────────────────────────────────────────

def compare_e2e_api(host, ref_dir, case_idx):
    """Call /tts + /asr on target, compare with reference audio."""
    meta_path = os.path.join(ref_dir, str(case_idx), "metadata.json")
    if not os.path.exists(meta_path):
        return [StepResult("e2e", False, "reference metadata.json missing")]

    with open(meta_path) as f:
        meta = json.load(f)
    text = meta["text"]
    lang = meta.get("language", "chinese")

    results = []
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    try:
        # TTS
        tts_meta = tts_api(host, text, lang, tmp_wav)
        if tts_meta.get("http_code") != 200:
            return [StepResult("e2e_tts", False,
                f"TTS API error: HTTP {tts_meta.get('http_code')}")]
        tts_time = tts_meta.get("time", 0)

        # Check audio
        tgt_samples, tgt_sr = read_wav_pcm(tmp_wav)
        tgt_dur = len(tgt_samples) / tgt_sr
        results.append(StepResult("e2e_audio", tgt_dur > 0.1,
            f"dur={tgt_dur:.2f}s  sr={tgt_sr}  tts_time={tts_time:.2f}s"))

        # Compare with reference audio (RMS, duration ratio)
        ref_wav = os.path.join(ref_dir, str(case_idx), "audio.wav")
        if os.path.exists(ref_wav):
            ref_samples, ref_sr = read_wav_pcm(ref_wav)
            ref_dur = len(ref_samples) / ref_sr
            dur_ratio = tgt_dur / ref_dur if ref_dur > 0 else 0
            results.append(StepResult("e2e_dur_ratio", 0.5 < dur_ratio < 2.0,
                f"ref={ref_dur:.2f}s  tgt={tgt_dur:.2f}s  ratio={dur_ratio:.2f}"))

        # ASR roundtrip
        try:
            asr_text = asr_api(host, tmp_wav)
            c = cer(text, asr_text)
            results.append(StepResult("e2e_cer", c < 0.2,
                f"CER={c:.0%}  ref=\"{text}\"  asr=\"{asr_text}\""))
        except Exception as e:
            results.append(StepResult("e2e_cer", False, f"ASR error: {e}"))

    finally:
        os.unlink(tmp_wav)

    return results


# ─── Main comparison ─────────────────────────────────────────────────────────

def validate_case(ref_dir, tgt_dir, case_idx, api_host=None):
    """Run all comparisons for one test case."""
    ref_case = os.path.join(ref_dir, str(case_idx))
    tgt_case = os.path.join(tgt_dir, str(case_idx)) if tgt_dir else None

    results = []

    # Load reference metadata
    meta_path = os.path.join(ref_case, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        text = meta.get("text", "???")
    else:
        text = TEST_CASES[case_idx][0] if case_idx < len(TEST_CASES) else "???"

    print(f"\n{'='*70}")
    print(f"Case {case_idx}: \"{text}\"")
    print(f"{'='*70}")

    if tgt_case and os.path.isdir(tgt_case):
        # ── Step-by-step comparison ──
        print(f"\n--- Step-by-step (ref={ref_case}  tgt={tgt_case}) ---\n")

        # 1. Token IDs
        ref_tok = load_npy(os.path.join(ref_case, "token_ids.npy"))
        tgt_tok = load_npy(os.path.join(tgt_case, "token_ids.npy"))
        r = compare_int_array("token_ids", ref_tok, tgt_tok, "tokens")
        results.append(r); print(r)

        # 2. Text embeddings
        ref_emb = load_npy(os.path.join(ref_case, "text_embeds.npy"))
        tgt_emb = load_npy(os.path.join(tgt_case, "text_embeds.npy"))
        r = compare_float_array("text_embeds", ref_emb, tgt_emb, 0.999, 0.01)
        results.append(r); print(r)

        # 3. Prefill logits
        ref_pl = load_npy(os.path.join(ref_case, "prefill_logits.npy"))
        tgt_pl = load_npy(os.path.join(tgt_case, "prefill_logits.npy"))
        r = compare_float_array("prefill_logits", ref_pl, tgt_pl, 0.99, 1.0)
        results.append(r); print(r)

        # 4. Prefill hidden
        ref_ph = load_npy(os.path.join(ref_case, "prefill_hidden.npy"))
        tgt_ph = load_npy(os.path.join(tgt_case, "prefill_hidden.npy"))
        r = compare_float_array("prefill_hidden", ref_ph, tgt_ph, 0.99, 1.0)
        results.append(r); print(r)

        # 5. Primary codes
        ref_pc = load_npy(os.path.join(ref_case, "primary_codes.npy"))
        tgt_pc = load_npy(os.path.join(tgt_case, "primary_codes.npy"))
        r = compare_int_array("primary_codes", ref_pc, tgt_pc, "frames")
        results.append(r); print(r)

        # 6. CP codes
        ref_cc = load_npy(os.path.join(ref_case, "cp_codes.npy"))
        tgt_cc = load_npy(os.path.join(tgt_case, "cp_codes.npy"))
        r = compare_codes_2d("cp_codes", ref_cc, tgt_cc)
        results.append(r); print(r)

        # 7. Codec sums
        ref_cs = load_npy(os.path.join(ref_case, "codec_sums.npy"))
        tgt_cs = load_npy(os.path.join(tgt_case, "codec_sums.npy"))
        r = compare_float_array("codec_sums", ref_cs, tgt_cs, 0.95, 5.0)
        results.append(r); print(r)

        # 8. Per-frame intermediates
        print(f"\n--- Per-frame (first 5 frames) ---\n")
        frame_results = compare_frames(ref_case, tgt_case)
        for r in frame_results:
            results.append(r); print(r)

    else:
        if tgt_dir:
            print(f"\n  (no target dir at {tgt_case}, skipping step-by-step)")

    # ── E2E via API ──
    if api_host:
        print(f"\n--- E2E via API ({api_host}) ---\n")
        e2e_results = compare_e2e_api(api_host, ref_dir, case_idx)
        for r in e2e_results:
            results.append(r); print(r)

    return results


def print_summary(all_results):
    """Print final summary."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)

    print(f"  Total checks: {total}")
    print(f"  Passed:       {passed}")
    print(f"  Failed:       {failed}")

    if failed > 0:
        print(f"\n  Failed checks:")
        for r in all_results:
            if not r.passed:
                print(f"    {r}")

    print()
    if failed == 0:
        print("  RESULT: ALL PASS")
    else:
        # Find first failure to identify root cause
        first_fail = next(r for r in all_results if not r.passed)
        print(f"  RESULT: FAIL — first divergence at: {first_fail.name}")
        print(f"          {first_fail.detail}")

    return failed == 0


def main():
    p = argparse.ArgumentParser(
        description="TTS pipeline step-by-step validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step-by-step comparison (both dirs have .npy files):
  python tts_validate.py --ref-dir ./tts_ref_dump --target-dir ./tts_target_dump

  # E2E only via API:
  python tts_validate.py --ref-dir ./tts_ref_dump --api-host localhost:8000

  # Both:
  python tts_validate.py --ref-dir ./tts_ref_dump --target-dir ./tts_target_dump --api-host localhost:8000

  # Single case:
  python tts_validate.py --ref-dir ./tts_ref_dump --target-dir ./tts_target_dump --case 0
""")
    p.add_argument("--ref-dir", required=True,
                   help="Reference data directory (from tts_dump_reference.py)")
    p.add_argument("--target-dir", default=None,
                   help="Target data directory (same structure as ref)")
    p.add_argument("--api-host", default=None,
                   help="Target API host:port for e2e comparison")
    p.add_argument("--case", type=int, default=None,
                   help="Run single case index (default: all)")
    p.add_argument("--cases", type=int, default=3,
                   help="Number of cases to validate (default: 3)")
    args = p.parse_args()

    if not args.target_dir and not args.api_host:
        p.error("At least one of --target-dir or --api-host is required")

    if args.case is not None:
        cases = [args.case]
    else:
        cases = list(range(args.cases))

    all_results = []
    for ci in cases:
        results = validate_case(args.ref_dir, args.target_dir, ci, args.api_host)
        all_results.extend(results)

    ok = print_summary(all_results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
