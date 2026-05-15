"""Four runner functions used by perf.py.

Each runner:
  - takes a corpus subset (or prompt list) and runtime knobs
  - executes warmup runs (discarded) + steady runs (recorded)
  - returns a flat list of per-iteration dicts ready for stats.summarize()
"""
from __future__ import annotations
import concurrent.futures as cf
import json
import time
from pathlib import Path
from typing import Callable

from client import ASRClient, TTSClient, run_v2v, wav_duration_s


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "",
    "，。！？、；：""''（）《》【】「」『』" +
    "·～—-…,.!?:;\"'()<>[]{}/")

# Run of Chinese number characters + multipliers + 两 (= alt form of 2)
_CN_NUM_RE = None  # lazy-compiled inside _normalize_numbers


def _normalize_numbers(text: str, lang: str) -> str:
    """Canonicalize number expressions so '15米' and '十五米' compare equal,
    '2011年' and '二零一一年' compare equal, etc.

    Implementation: find runs of Chinese number chars and convert each run to
    Arabic via cn2an. Both reference and hypothesis are passed through this,
    so the result becomes invariant to Chinese-vs-Arabic number rendering.

    No-op if cn2an is missing or lang != 'zh'.
    """
    if lang != "zh" or not text:
        return text
    try:
        import cn2an
    except ImportError:
        return text
    import re
    global _CN_NUM_RE
    if _CN_NUM_RE is None:
        _CN_NUM_RE = re.compile(r"[零一二三四五六七八九十百千万亿两]+")

    def _repl(m):
        s = m.group(0)
        try:
            return str(cn2an.cn2an(s, "smart"))
        except Exception:
            return s
    return _CN_NUM_RE.sub(_repl, text)


def _normalize_for_match(text: str, lang: str) -> str:
    """Normalize numbers + strip punctuation + lowercase + collapse spaces.
    Char-level for zh, word-level for en (handled in jiwer cer/wer)."""
    text = _normalize_numbers(text, lang)
    s = text.translate(_PUNCT_TABLE).lower().strip()
    return " ".join(s.split())


def compute_error_rate(reference: str, hypothesis: str, lang: str) -> float:
    """CER for zh (char-level), WER for en/other (word-level). Uses jiwer if available."""
    if not reference or not hypothesis:
        return 1.0
    ref = _normalize_for_match(reference, lang)
    hyp = _normalize_for_match(hypothesis, lang)
    try:
        import jiwer
    except ImportError:
        # Fallback: Levenshtein-based via difflib (close enough for sanity)
        import difflib
        ref_units = list(ref) if lang == "zh" else ref.split()
        hyp_units = list(hyp) if lang == "zh" else hyp.split()
        if not ref_units:
            return 1.0
        sm = difflib.SequenceMatcher(a=ref_units, b=hyp_units, autojunk=False)
        opcodes = sm.get_opcodes()
        edits = sum(max(a2 - a1, b2 - b1) for tag, a1, a2, b1, b2 in opcodes if tag != "equal")
        return edits / len(ref_units)
    if lang == "zh":
        return jiwer.cer(ref, hyp)
    return jiwer.wer(ref, hyp)


# ---------------------------------------------------------------------------
# Corpus / prompt loaders
# ---------------------------------------------------------------------------

CORPUS_ROOT = Path(__file__).resolve().parent / "corpus"


def _interleave_by_group(files: list[dict]) -> list[dict]:
    """Round-robin across (category, lang) groups so that small `--runs` still
    covers every category/lang combo. Manifest order is grouped, which made
    `--warmup 2 --runs 5` land entirely in the first 7 zh_short entries."""
    groups: dict[tuple, list[dict]] = {}
    for f in files:
        groups.setdefault((f.get("category", ""), f.get("lang", "")), []).append(f)
    out: list[dict] = []
    while any(groups.values()):
        for key in list(groups.keys()):
            if groups[key]:
                out.append(groups[key].pop(0))
    return out


def load_corpus(category: str | None = None, lang: str | None = None,
                manifest_name: str = "manifest.json") -> list[dict]:
    """Returns manifest entries augmented with `bytes` field (loaded WAV).
    Files are interleaved across (category, lang) groups."""
    manifest = json.loads((CORPUS_ROOT / manifest_name).read_text())
    files = manifest["files"]
    if category:
        files = [f for f in files if f["category"] == category]
    if lang:
        files = [f for f in files if f["lang"] == lang]
    files = _interleave_by_group(files)
    out = []
    for entry in files:
        path = CORPUS_ROOT / entry["filename"]
        if not path.exists():
            continue
        out.append({**entry, "bytes": path.read_bytes()})
    return out


def load_prompts(category: str | None = None, lang: str | None = None) -> list[dict]:
    data = json.loads((CORPUS_ROOT / "tts_prompts.json").read_text())
    prompts = data["prompts"]
    if category:
        prompts = [p for p in prompts if p["category"] == category]
    if lang:
        prompts = [p for p in prompts if p["lang"] == lang]
    return prompts


# ---------------------------------------------------------------------------
# Single-stream runners
# ---------------------------------------------------------------------------

def _iter_loop(items: list, warmup: int, runs: int):
    """Yield (label, item, idx). Loops `items` cyclically if runs > len(items)."""
    n = len(items)
    if n == 0:
        return
    for k in range(warmup + runs):
        label = "warmup" if k < warmup else "steady"
        yield label, items[k % n], k


def run_asr(asr: ASRClient, corpus: list[dict],
            warmup: int = 3, runs: int = 10,
            mode: str = "streaming", eos_mode: str = "vad") -> list[dict]:
    """mode: 'offline' (POST /asr) or 'streaming' (WS /asr/stream)."""
    assert mode in ("offline", "streaming")
    records: list[dict] = []
    for label, entry, k in _iter_loop(corpus, warmup, runs):
        lang_map = {"zh": "Chinese", "en": "English"}
        language = lang_map.get(entry["lang"], "Chinese")
        try:
            if mode == "streaming":
                r = asr.transcribe_streaming(entry["bytes"], language, eos_mode)
            else:
                r = asr.transcribe_offline(entry["bytes"], language)
        except Exception as e:
            print(f"  [{label:6s} {k+1:02d}] {entry['id']:14s}  ERROR: {type(e).__name__}: {str(e)[:120]}")
            records.append({"label": label, "id": entry["id"], "error": f"{type(e).__name__}: {e}"})
            continue
        err_rate = compute_error_rate(entry.get("transcript", ""), r.text, entry["lang"])
        err_label = "cer" if entry["lang"] == "zh" else "wer"
        records.append({
            "label": label, "id": entry["id"], "lang": entry["lang"],
            "category": entry["category"],
            **r.as_dict,
            err_label: err_rate,
            "error_rate": err_rate,  # uniform key for summarize()
        })
        frtf = f"  fRTF={r.finalize_rtf:.3f}" if r.finalize_rtf is not None else ""
        print(f"  [{label:6s} {k+1:02d}] {entry['id']:14s}  "
              f"RTF={r.rtf:.3f}{frtf}  TFD={r.tfd_ms or 0:.0f}ms  "
              f"{err_label.upper()}={err_rate*100:.1f}%  "
              f"text='{r.text[:30]}...'")
    return records


def run_tts(tts: TTSClient, prompts: list[dict],
            warmup: int = 3, runs: int = 10) -> list[dict]:
    records: list[dict] = []
    for label, p, k in _iter_loop(prompts, warmup, runs):
        try:
            r = tts.synthesize(p["text"], p["lang"])
        except Exception as e:
            records.append({"label": label, "id": p["id"], "error": str(e)})
            continue
        records.append({
            "label": label, "id": p["id"], "lang": p["lang"],
            "category": p["category"],
            "audio_dur_s": r.audio_dur_s,
            "tfd_ms": r.tfd_ms, "total_ms": r.total_ms, "rtf": r.rtf,
        })
        print(f"  [{label:6s} {k+1:02d}] {p['id']:14s}  "
              f"RTF={r.rtf:.3f}  TFD={r.tfd_ms:.0f}ms  total={r.total_ms:.0f}ms  "
              f"dur={r.audio_dur_s:.2f}s")
    return records


def run_v2v_bench(asr: ASRClient, tts: TTSClient, corpus: list[dict],
                  warmup: int = 3, runs: int = 10,
                  eos_mode: str = "vad", llm_delay_ms: float = 0.0) -> list[dict]:
    records: list[dict] = []
    for label, entry, k in _iter_loop(corpus, warmup, runs):
        lang_map = {"zh": "Chinese", "en": "English"}
        try:
            r = run_v2v(asr, tts, entry["bytes"],
                        language_asr=lang_map.get(entry["lang"], "Chinese"),
                        language_tts=entry["lang"],
                        eos_mode=eos_mode, llm_delay_ms=llm_delay_ms)
        except Exception as e:
            records.append({"label": label, "id": entry["id"], "error": str(e)})
            continue
        records.append({
            "label": label, "id": entry["id"], "lang": entry["lang"],
            "category": entry["category"], "eos_mode": eos_mode,
            **r.as_dict,
        })
        print(f"  [{label:6s} {k+1:02d}] {entry['id']:14s}  "
              f"EOS->Audio={r.eos_to_first_audio_ms:.0f}ms "
              f"(ASR={r.asr_finalize_ms:.0f} +LLM={r.llm_delay_ms:.0f} "
              f"+TTS_TFD={r.tts_tfd_ms:.0f})  text='{r.asr_text[:25]}...'")
    return records


# ---------------------------------------------------------------------------
# Concurrent runner
# ---------------------------------------------------------------------------

def run_asr_noisy(asr: ASRClient, corpus: list[dict],
                   snr_db: float, noise_type: str = "babble",
                   warmup: int = 2, runs: int = 5) -> list[dict]:
    """Variant of run_asr that mixes noise at target SNR before sending."""
    from noise import add_noise
    pool = [e["bytes"] for e in corpus]
    records: list[dict] = []
    for label, entry, k in _iter_loop(corpus, warmup, runs):
        noisy_bytes = add_noise(entry["bytes"], snr_db, noise_type,
                                babble_pool=pool, seed=42 + k)
        lang_map = {"zh": "Chinese", "en": "English"}
        language = lang_map.get(entry["lang"], "Chinese")
        try:
            r = asr.transcribe_streaming(noisy_bytes, language)
        except Exception as e:
            records.append({"label": label, "id": entry["id"], "error": str(e)})
            continue
        err_rate = compute_error_rate(entry.get("transcript", ""), r.text, entry["lang"])
        records.append({
            "label": label, "id": entry["id"], "lang": entry["lang"],
            "category": entry["category"],
            "snr_db": snr_db, "noise_type": noise_type,
            **r.as_dict,
            "error_rate": err_rate,
        })
        print(f"  [{label:6s} {k+1:02d}] {entry['id']:14s}  "
              f"SNR={snr_db}dB {noise_type:6s}  "
              f"RTF={r.rtf:.3f}  err={err_rate*100:.1f}%  "
              f"text='{r.text[:30]}...'")
    return records


def run_concurrent(asr: ASRClient, tts: TTSClient,
                   corpus: list[dict], prompts: list[dict],
                   parallel: int = 2, runs_per_worker: int = 5,
                   mode: str = "asr_tts_simul") -> list[dict]:
    """
    mode:
      - "asr_only":      N parallel ASR streams
      - "tts_only":      N parallel TTS streams
      - "asr_tts_simul": N/2 ASR + N/2 TTS (simultaneous-interpretation pattern)
    """
    assert mode in ("asr_only", "tts_only", "asr_tts_simul")
    records: list[dict] = []

    def asr_worker(wid: int) -> list[dict]:
        out = []
        for k in range(runs_per_worker):
            entry = corpus[(wid + k) % len(corpus)]
            t0 = time.monotonic()
            try:
                r = asr.transcribe_streaming(
                    entry["bytes"],
                    "Chinese" if entry["lang"] == "zh" else "English",
                )
                out.append({"worker": wid, "kind": "asr", "k": k, "id": entry["id"],
                            **r.as_dict, "wall_ms": (time.monotonic() - t0) * 1000})
            except Exception as e:
                out.append({"worker": wid, "kind": "asr", "k": k, "id": entry["id"], "error": str(e)})
        return out

    def tts_worker(wid: int) -> list[dict]:
        out = []
        for k in range(runs_per_worker):
            p = prompts[(wid + k) % len(prompts)]
            t0 = time.monotonic()
            try:
                r = tts.synthesize(p["text"], p["lang"])
                out.append({"worker": wid, "kind": "tts", "k": k, "id": p["id"],
                            "audio_dur_s": r.audio_dur_s, "tfd_ms": r.tfd_ms,
                            "total_ms": r.total_ms, "rtf": r.rtf,
                            "wall_ms": (time.monotonic() - t0) * 1000})
            except Exception as e:
                out.append({"worker": wid, "kind": "tts", "k": k, "id": p["id"], "error": str(e)})
        return out

    workers: list[Callable] = []
    if mode == "asr_only":
        workers = [asr_worker] * parallel
    elif mode == "tts_only":
        workers = [tts_worker] * parallel
    else:  # asr_tts_simul
        n_asr = max(1, parallel // 2)
        n_tts = max(1, parallel - n_asr)
        workers = [asr_worker] * n_asr + [tts_worker] * n_tts

    t_start = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=len(workers)) as ex:
        futures = [ex.submit(w, i) for i, w in enumerate(workers)]
        for f in cf.as_completed(futures):
            records.extend(f.result())
    wall_total_ms = (time.monotonic() - t_start) * 1000

    for r in records:
        r["parallel"] = parallel
        r["concurrency_mode"] = mode
        r["scenario_wall_ms"] = wall_total_ms
    return records
