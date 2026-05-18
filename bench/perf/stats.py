"""Statistics + markdown report rendering.

Discards warmup rows. Drops `error` rows. Computes mean/p50/p95 per metric,
grouped by (lang, category) where applicable.
"""
from __future__ import annotations
from datetime import datetime
import json
import statistics as st
import time
from pathlib import Path


METRIC_LABELS = {
    "rtf": "Wall RTF",
    "finalize_rtf": "Finalize RTF",
    "tfd_ms": "TFD (ms)",
    "processing_ms": "Proc (ms)",
    "total_ms": "Total (ms)",
    "eos_to_final_ms": "EOS→Final (ms)",
    "eos_to_first_audio_ms": "EOS→Audio (ms)",
    "asr_finalize_ms": "ASR finalize (ms)",
    "vad_silence_ms": "VAD silence (ms)",
    "asr_finalize_compute_ms": "ASR compute (ms)",
    "endpoint_latency_ms": "Endpoint lat (ms)",
    "total_latency_ms": "Total lat (ms)",
    "tts_tfd_ms": "TTS TFD (ms)",
    "tts_total_ms": "TTS total (ms)",
    "wall_ms": "Wall (ms)",
    "error_rate": "CER/WER",
    "wer": "WER",
    "cer": "CER",
    "similarity": "Speaker sim",
}


def filter_steady(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("label", "steady") == "steady" and "error" not in r]


def percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * q
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(records: list[dict], group_by: tuple[str, ...] = ("category",),
              metrics: tuple[str, ...] = ("rtf", "tfd_ms")) -> dict:
    """Returns {group_key: {metric: {mean, p50, p95, n}}}."""
    steady = filter_steady(records)
    groups: dict[tuple, list[dict]] = {}
    for r in steady:
        key = tuple(r.get(g, "all") for g in group_by)
        groups.setdefault(key, []).append(r)

    out: dict[str, dict] = {}
    for key, items in groups.items():
        label = "/".join(str(k) for k in key) if key else "all"
        out[label] = {"n": len(items)}
        for m in metrics:
            vals = [r[m] for r in items if isinstance(r.get(m), (int, float))]
            if not vals:
                continue
            out[label][m] = {
                "mean": round(st.mean(vals), 3),
                "p50":  round(percentile(vals, 0.5), 3),
                "p95":  round(percentile(vals, 0.95), 3),
                "min":  round(min(vals), 3),
                "max":  round(max(vals), 3),
                "n":    len(vals),
            }
    return out


def render_markdown(scenario: str, summary: dict, raw: list[dict],
                    memory: dict | None, meta: dict) -> str:
    lines = [f"# Perf — {scenario}", ""]
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for k, v in meta.items():
        lines.append(f"- {k}: `{v}`")
    if memory:
        lines.append(f"- Memory peak: **{memory.get('peak_mib', 'n/a')} MiB** (mean {memory.get('mean_mib', 'n/a')}, n={memory.get('samples', 0)})")
    lines.append("")

    for group, stats in summary.items():
        n = stats.get("n", 0)
        lines.append(f"## Group: `{group}` (n={n})")
        lines.append("")
        lines.append("| Metric | Mean | P50 | P95 | Min | Max | N |")
        lines.append("|--------|-----:|----:|----:|----:|----:|--:|")
        for metric_key in ["rtf", "finalize_rtf", "tfd_ms", "error_rate", "similarity",
                           "eos_to_final_ms", "eos_to_first_audio_ms",
                           "vad_silence_ms", "asr_finalize_compute_ms",
                           "endpoint_latency_ms", "asr_finalize_ms",
                           "tts_tfd_ms", "tts_total_ms",
                           "processing_ms", "total_ms", "total_latency_ms", "wall_ms"]:
            m = stats.get(metric_key)
            if not m:
                continue
            label = METRIC_LABELS.get(metric_key, metric_key)
            if metric_key in ("rtf", "finalize_rtf", "similarity"):
                fmt = lambda v: f"{v:.3f}"
            elif metric_key == "error_rate":
                fmt = lambda v: f"{v*100:.2f}%"
            else:
                fmt = lambda v: f"{v:.0f}"
            lines.append(f"| {label} | {fmt(m['mean'])} | {fmt(m['p50'])} | "
                         f"{fmt(m['p95'])} | {fmt(m['min'])} | {fmt(m['max'])} | {m['n']} |")
        lines.append("")

    # Compact raw table — first 20 rows for sanity
    lines.append("## Raw samples (first 20 steady rows)")
    lines.append("")
    steady = filter_steady(raw)[:20]
    if steady:
        keys = sorted({k for r in steady for k in r.keys()}
                      - {"audio_bytes", "label"})
        header_keys = [k for k in ["id", "lang", "category", "rtf", "tfd_ms",
                                   "eos_to_first_audio_ms", "asr_finalize_ms",
                                   "tts_tfd_ms", "text", "asr_text"] if k in keys]
        if not header_keys:
            header_keys = list(keys)[:6]
        lines.append("| " + " | ".join(header_keys) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_keys)) + " |")
        for r in steady:
            row = []
            for k in header_keys:
                v = r.get(k, "")
                if isinstance(v, float):
                    v = f"{v:.3f}" if k == "rtf" else f"{v:.0f}"
                if isinstance(v, str):
                    v = v.replace("|", "/")[:30]
                row.append(str(v))
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def save_results(out_dir: Path, scenario: str, raw: list[dict],
    summary: dict, memory: dict | None, meta: dict) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    base = out_dir / f"{scenario}_{stamp}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    payload = {
        "scenario": scenario,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": meta, "memory": memory,
        "summary": summary, "raw": raw,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    md_path.write_text(render_markdown(scenario, summary, raw, memory, meta))
    return json_path, md_path
