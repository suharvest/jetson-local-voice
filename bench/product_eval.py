#!/usr/bin/env python3
"""Product-level evaluation for deployed OpenVoiceStream services.

This is the release-gate harness: quick enough to run after deployment,
structured enough to feed README/device-selection docs.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import statistics
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "bench" / "perf" / "corpus"
RESULTS_DIR = ROOT / "bench" / "product_results"


@dataclass
class Target:
    name: str
    url: str


def normalize(text: str, *, mode: str) -> str:
    text = text.lower()
    drop = set(" \t\r\n。，、！？；：,.!?;:\"'()[]{}<>《》“”‘’")
    if mode == "zh":
        return "".join(ch for ch in text if ch not in drop)
    return " ".join("".join(ch if ch not in drop else " " for ch in text).split())


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = cur
    return prev[-1]


def error_rate(expected: str, actual: str, lang: str) -> float:
    if lang == "zh":
        ref = list(normalize(expected, mode="zh"))
        hyp = list(normalize(actual, mode="zh"))
    else:
        ref = normalize(expected, mode="en").split()
        hyp = normalize(actual, mode="en").split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def lcs_similarity(a: str, b: str) -> float:
    aa = normalize(a, mode="zh")
    bb = normalize(b, mode="zh")
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    prev = [0] * (len(bb) + 1)
    for ca in aa:
        cur = [0] * (len(bb) + 1)
        for j, cb in enumerate(bb, start=1):
            cur[j] = prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1] / max(len(aa), len(bb))


def wav_duration(data: bytes) -> float:
    with NamedTemporaryFile(suffix=".wav") as f:
        f.write(data)
        f.flush()
        with wave.open(f.name, "rb") as wf:
            return wf.getnframes() / wf.getframerate()


def request_json(url: str, timeout: float) -> dict:
    with OPENER.open(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict, timeout: float) -> tuple[bytes, float]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with OPENER.open(req, timeout=timeout) as resp:
        body = resp.read()
    return body, (time.monotonic() - start) * 1000


def post_tts_stream(url: str, text: str, timeout: float) -> dict:
    data = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/tts/stream",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with OPENER.open(req, timeout=timeout) as resp:
        first = resp.read(4)
        first_ms = (time.monotonic() - start) * 1000
        rest = resp.read()
    total_ms = (time.monotonic() - start) * 1000
    if len(first) != 4:
        raise RuntimeError("stream did not return sample-rate header")
    sample_rate = int.from_bytes(first, "little")
    samples = len(rest) // 2
    audio_dur = samples / sample_rate if sample_rate else 0.0
    return {
        "sample_rate": sample_rate,
        "audio_bytes": len(rest),
        "audio_duration_s": audio_dur,
        "ttfd_ms": first_ms,
        "total_ms": total_ms,
        "rtf": total_ms / (audio_dur * 1000) if audio_dur else None,
    }


def post_multipart(url: str, field: str, filename: str, data: bytes, timeout: float) -> dict:
    boundary = f"openvoicestream-{time.time_ns()}"
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    start = time.monotonic()
    with OPENER.open(req, timeout=timeout) as resp:
        parsed = json.loads(resp.read().decode("utf-8"))
    parsed["_processing_ms"] = (time.monotonic() - start) * 1000
    return parsed


def language_param(lang: str) -> str:
    if lang == "zh":
        return "zh"
    if lang == "en":
        return "en"
    return "auto"


def load_prompts(limit_per_group: int) -> list[dict]:
    data = json.loads((CORPUS_DIR / "tts_prompts.json").read_text())
    grouped: dict[tuple[str, str], int] = {}
    out = []
    for item in data["prompts"]:
        key = (item["category"], item["lang"])
        if grouped.get(key, 0) >= limit_per_group:
            continue
        grouped[key] = grouped.get(key, 0) + 1
        out.append(item)
    return out


def load_asr_corpus(limit_per_group: int) -> list[dict]:
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    grouped: dict[tuple[str, str], int] = {}
    out = []
    for item in manifest["files"]:
        key = (item["category"], item["lang"])
        if grouped.get(key, 0) >= limit_per_group:
            continue
        path = CORPUS_DIR / item["filename"]
        if not path.exists():
            continue
        grouped[key] = grouped.get(key, 0) + 1
        out.append({**item, "path": path})
    return out


def p50(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def summarize_rows(rows: list[dict], metric: str, group_keys: tuple[str, ...]) -> dict:
    groups: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        key = "/".join(str(row.get(k, "")) for k in group_keys)
        groups.setdefault(key, []).append(float(value))
    return {
        key: {"count": len(vals), "p50": p50(vals), "p95": p95(vals), "mean": statistics.mean(vals)}
        for key, vals in sorted(groups.items())
    }


def eval_target(target: Target, args: argparse.Namespace) -> dict:
    base = target.url.rstrip("/")
    result: dict = {
        "target": target.name,
        "base_url": base,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "errors": [],
        "warnings": [],
    }
    try:
        result["health"] = request_json(f"{base}/health", args.timeout_sec)
    except Exception as exc:
        result["errors"].append(f"health failed: {exc}")
        return result

    for endpoint in ("tts", "asr"):
        try:
            result[f"{endpoint}_capabilities"] = request_json(f"{base}/{endpoint}/capabilities", args.timeout_sec)
        except Exception as exc:
            # Some deployed images expose readiness through /health but do not
            # yet expose split capability endpoints. Keep measuring observable
            # behavior, and surface endpoint drift as a warning.
            result["warnings"].append(f"{endpoint} capabilities failed: {exc}")

    tts_rows = []
    if result["health"].get("tts"):
        for prompt in load_prompts(args.limit_per_group):
            for run in range(args.runs):
                row = {k: prompt[k] for k in ("id", "lang", "category", "text")}
                row["run"] = run
                try:
                    row.update(post_tts_stream(base, prompt["text"], args.timeout_sec))
                except Exception as exc:
                    row["error"] = str(exc)
                    result["errors"].append(f"tts {prompt['id']} run {run}: {exc}")
                tts_rows.append(row)
    result["tts_rows"] = tts_rows
    result["tts_summary"] = {
        "rtf": summarize_rows(tts_rows, "rtf", ("category", "lang")),
        "ttfd_ms": summarize_rows(tts_rows, "ttfd_ms", ("category", "lang")),
        "total_ms": summarize_rows(tts_rows, "total_ms", ("category", "lang")),
    }

    asr_caps_payload = result.get("asr_capabilities") or {}
    asr_caps = set(asr_caps_payload.get("capabilities") or [])
    offline_asr_known = "offline" in asr_caps
    offline_asr_unknown = bool(result["health"].get("asr")) and not asr_caps_payload
    should_try_offline_asr = offline_asr_known or offline_asr_unknown
    roundtrip_rows = []
    if result["health"].get("tts") and result["health"].get("asr") and should_try_offline_asr:
        # Run the core product gate before stress/corpus ASR cases. Some
        # backends intentionally fail oversized offline inputs; those failures
        # should be recorded without poisoning the TTS->ASR smoke loop.
        for text in args.roundtrip_text:
            row = {"text": text}
            try:
                wav, tts_ms = post_json(f"{base}/tts", {"text": text}, args.timeout_sec)
                parsed = post_multipart(f"{base}/asr?language=zh", "file", "roundtrip.wav", wav, args.timeout_sec)
                asr_text = str(parsed.get("text", "")).strip()
                row.update({
                    "tts_bytes": len(wav),
                    "tts_ms": tts_ms,
                    "asr_text": asr_text,
                    "asr_ms": parsed.get("_processing_ms"),
                    "similarity": lcs_similarity(text, asr_text),
                    "passed": bool(asr_text) and lcs_similarity(text, asr_text) >= args.min_roundtrip_sim,
                })
            except Exception as exc:
                row["error"] = str(exc)
                row["passed"] = False
                result["errors"].append(f"roundtrip {text!r}: {exc}")
            roundtrip_rows.append(row)
    elif result["health"].get("tts") and result["health"].get("asr"):
        result["roundtrip_note"] = "offline ASR capability not available; TTS -> /asr round-trip skipped"
    result["roundtrip_rows"] = roundtrip_rows
    result["roundtrip_pass"] = all(r.get("passed") for r in roundtrip_rows) if roundtrip_rows else None

    asr_rows = []
    if result["health"].get("asr") and should_try_offline_asr:
        for item in load_asr_corpus(args.limit_per_group):
            row = {k: item[k] for k in ("id", "lang", "category", "duration_s", "transcript")}
            try:
                parsed = post_multipart(
                    f"{base}/asr?language={language_param(item['lang'])}",
                    "file",
                    Path(item["filename"]).name,
                    Path(item["path"]).read_bytes(),
                    args.timeout_sec,
                )
                text = str(parsed.get("text", "")).strip()
                row["text"] = text
                row["processing_ms"] = parsed.get("_processing_ms")
                row["rtf"] = row["processing_ms"] / (float(item["duration_s"]) * 1000)
                row["error_rate"] = error_rate(item["transcript"], text, item["lang"])
            except Exception as exc:
                row["error"] = str(exc)
                result["errors"].append(f"asr {item['id']}: {exc}")
            asr_rows.append(row)
    elif result["health"].get("asr"):
        result["asr_note"] = "offline ASR capability not available; corpus /asr evaluation skipped"
    result["asr_rows"] = asr_rows
    result["asr_summary"] = {
        "rtf": summarize_rows(asr_rows, "rtf", ("category", "lang")),
        "error_rate": summarize_rows(asr_rows, "error_rate", ("category", "lang")),
    }
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    return result


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# OpenVoiceStream Product Evaluation",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Target | TTS backend | ASR backend | TTS short zh RTF p50 | ASR short zh error p50 | Roundtrip | Errors |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for r in results:
        health = r.get("health") or {}
        tts_rtf = (((r.get("tts_summary") or {}).get("rtf") or {}).get("short/zh") or {}).get("p50")
        asr_err = (((r.get("asr_summary") or {}).get("error_rate") or {}).get("short/zh") or {}).get("p50")
        lines.append(
            "| {target} | {tts} | {asr} | {tts_rtf} | {asr_err} | {rt} | {err} |".format(
                target=r.get("target"),
                tts=health.get("tts_backend"),
                asr=health.get("asr_backend"),
                tts_rtf=f"{tts_rtf:.3f}" if isinstance(tts_rtf, (int, float)) else "-",
                asr_err=f"{asr_err:.1%}" if isinstance(asr_err, (int, float)) else "-",
                rt="PASS" if r.get("roundtrip_pass") else "FAIL" if r.get("roundtrip_pass") is False else "-",
                err=len(r.get("errors") or []),
            )
        )
    lines.extend(["", "## Details", ""])
    for r in results:
        lines.extend([f"### {r.get('target')}", ""])
        lines.append("```json")
        compact = {
            "base_url": r.get("base_url"),
            "health": r.get("health"),
            "tts_summary": r.get("tts_summary"),
            "asr_summary": r.get("asr_summary"),
            "roundtrip_rows": r.get("roundtrip_rows"),
            "asr_note": r.get("asr_note"),
            "roundtrip_note": r.get("roundtrip_note"),
            "warnings": r.get("warnings"),
            "errors": r.get("errors"),
        }
        lines.append(json.dumps(compact, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def parse_targets(values: list[str]) -> list[Target]:
    targets = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--target must be name=url, got: {value}")
        name, url = value.split("=", 1)
        targets.append(Target(name=name.strip(), url=url.strip()))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", required=True, help="name=http://host:port")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--limit-per-group", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=180)
    parser.add_argument("--min-roundtrip-sim", type=float, default=0.2)
    parser.add_argument("--roundtrip-text", action="append")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    targets = parse_targets(args.target)
    if not args.roundtrip_text:
        args.roundtrip_text = ["你好，今天天气真不错。"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = [eval_target(target, args) for target in targets]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.out_dir / f"product_eval_{stamp}.json"
    md_path = args.out_dir / f"product_eval_{stamp}.md"
    json_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    md_path.write_text(render_markdown(results))
    print(f"Saved: {json_path}")
    print(f"       {md_path}")
    failed = any(r.get("errors") or r.get("roundtrip_pass") is False for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
