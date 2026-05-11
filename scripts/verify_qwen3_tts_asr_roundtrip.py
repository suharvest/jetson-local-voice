from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any


DEFAULT_CASES = [
    {
        "name": "short_cn",
        "text": "语音合成的稳定性。",
        "max_audio_length": 100,
    },
    {
        "name": "tail_cn",
        "text": "今天天气很好，我们一起测试语音合成。",
        "max_audio_length": 120,
    },
    {
        "name": "long_cn",
        "text": "语音合成的稳定性需要在长句、短句以及中英文标点之间保持自然清晰。今天我们继续验证官方后台的音质是否已经接近产品侧参考。",
        "max_audio_length": 180,
    },
    {
        "name": "mixed_zh_en",
        "text": "Qwen three TTS，现在听起来清晰吗？如果有 English words，也应该保持稳定自然。",
        "max_audio_length": 160,
    },
]


def _norm_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
    return text


def _similarity(expected: str, actual: str) -> float:
    exp = _norm_text(expected)
    act = _norm_text(actual)
    if not exp and not act:
        return 1.0
    return difflib.SequenceMatcher(None, exp, act).ratio()


def _wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as reader:
        frames = reader.getnframes()
        rate = reader.getframerate()
        channels = reader.getnchannels()
        width = reader.getsampwidth()
    return {
        "sample_rate": rate,
        "channels": channels,
        "sample_width": width,
        "frames": frames,
        "duration_s": round(frames / rate, 3) if rate else 0.0,
        "bytes": path.stat().st_size,
    }


def _run(cmd: list[str], *, input_text: str | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
        timeout=timeout,
    )


def _prepare_imports(args: argparse.Namespace) -> None:
    # The product app must win for the Python `backends` package. The overlay is
    # still needed for the native pybind module, but its legacy `backends/`
    # directory must not shadow the product wrapper.
    ordered = [args.app_dir, args.native_module_dir]
    sys.path[:] = [p for p in sys.path if p not in ordered]
    for path in reversed(ordered):
        if path and Path(path).is_dir():
            sys.path.insert(0, path)


def _synthesize(args: argparse.Namespace, text: str, max_audio_length: int, output: Path) -> dict[str, Any]:
    _prepare_imports(args)
    os.environ["JETSON_VOICE_TTS_BACKEND"] = args.backend
    os.environ["JETSON_VOICE_TTS_MODEL_BASE"] = args.model_base
    os.environ["JETSON_VOICE_TTS_NATIVE_MODULE_DIR"] = args.native_module_dir
    os.environ["JETSON_VOICE_TTS_SEED"] = str(args.seed)
    os.environ.setdefault("QWEN3_TTS_PRODUCT_SEGMENT_TEXT", "0")
    os.environ.setdefault("TTS_INT8_EOS_LOGIT_OFFSET", "0")
    os.environ.setdefault("QWEN3_TTS_EOS_BIAS_ONSET_MULT", "3")
    os.environ.setdefault("QWEN3_TTS_TEXT_MAX_FRAMES_MULT", "10")
    engines_dir = Path(args.model_base) / "engines"
    os.environ.setdefault("QWEN3_TALKER_ENGINE", str(engines_dir / "talker_decode_bf16.engine"))
    os.environ.setdefault("QWEN3_CP_ENGINE", str(engines_dir / "cp_unified_bf16.engine"))
    os.environ.setdefault("QWEN3_TTS_CP_KV_ENGINE", str(engines_dir / "cp_unified_bf16.engine"))

    from backends.trt_edge_llm_tts import TRTEdgeLLMTTSBackend

    t0 = time.time()
    backend = TRTEdgeLLMTTSBackend()
    backend.preload()
    wav_bytes, meta = backend.synthesize(
        text,
        max_audio_length=max_audio_length,
        seed=args.seed,
        segment_text=False,
        product_segment_text=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(wav_bytes)
    return {
        "meta": meta,
        "elapsed_s": round(time.time() - t0, 3),
        "wav": _wav_info(output),
    }


def _asr_with_product_qwen3(args: argparse.Namespace, wav_path: Path) -> dict[str, Any]:
    _prepare_imports(args)
    os.environ.setdefault("QWEN3_ASR_MODEL_BASE", args.qwen3_asr_model_base)

    from backends.qwen3_asr import Qwen3ASRBackend

    t0 = time.time()
    backend = Qwen3ASRBackend()
    backend.preload()
    result = backend.transcribe(wav_path.read_bytes(), language=args.language)
    return {
        "backend": "product_qwen3_asr",
        "text": result.text,
        "elapsed_s": round(time.time() - t0, 3),
    }


def _asr_with_paraformer_container(args: argparse.Namespace, wav_path: Path) -> dict[str, Any]:
    container_path = f"/tmp/{wav_path.name}"
    _run(["docker", "cp", str(wav_path), f"{args.paraformer_container}:{container_path}"], timeout=60)
    code = f"""
import json
import os
import sys
import time
from pathlib import Path

app_dir = Path('/opt/speech/app')
sys.path.insert(0, str(app_dir))
os.chdir(app_dir)
os.environ.setdefault('MODEL_DIR', '/opt/models')
os.environ.setdefault('LANGUAGE_MODE', 'zh_en')
os.environ.setdefault('ASR_BACKEND', 'paraformer_trt')

from asr_backend import create_asr_backend

backend = create_asr_backend('paraformer_trt')
t0 = time.time()
backend.preload()
t1 = time.time()
result = backend.transcribe(Path({container_path!r}).read_bytes(), language={args.language!r})
print(json.dumps({{
    'backend': getattr(backend, 'name', 'paraformer_trt'),
    'text': result.text,
    'preload_s': round(t1 - t0, 3),
    'elapsed_s': round(time.time() - t1, 3),
}}, ensure_ascii=False))
"""
    proc = _run(["docker", "exec", args.paraformer_container, "python3", "-c", code], timeout=args.asr_timeout)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _load_cases(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return DEFAULT_CASES
    with Path(path).open("r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise ValueError("cases file must contain a JSON list")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Qwen3-TTS by ASR round-trip on Jetson.")
    parser.add_argument("--app-dir", default="/tmp/jetson-voice-product-layer-0507/app")
    parser.add_argument("--model-base", default="/home/harvest/voice_test/models/qwen3-tts")
    parser.add_argument("--native-module-dir", default="/home/harvest/voice_test/app_overlay")
    parser.add_argument("--backend", default="product_explicit_kv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--out-dir", default="/tmp/qwen3_tts_asr_roundtrip")
    parser.add_argument("--cases-json", default="")
    parser.add_argument("--asr", choices=["paraformer_container", "product_qwen3"], default="paraformer_container")
    parser.add_argument("--paraformer-container", default="jetson_voice_paraformer_verify")
    parser.add_argument("--qwen3-asr-model-base", default="/home/harvest/voice_test/models/qwen3-asr-v2")
    parser.add_argument("--min-similarity", type=float, default=0.72)
    parser.add_argument("--asr-timeout", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cases = _load_cases(args.cases_json or None)
    report: dict[str, Any] = {
        "status": "PASS",
        "backend": args.backend,
        "seed": args.seed,
        "app_dir": args.app_dir,
        "model_base": args.model_base,
        "native_module_dir": args.native_module_dir,
        "asr": args.asr,
        "out_dir": str(out_dir),
        "cases": [],
    }

    for case in cases:
        name = case["name"]
        text = case["text"]
        max_audio_length = int(case.get("max_audio_length", 120))
        wav_path = out_dir / f"{name}.wav"
        case_report: dict[str, Any] = {
            "name": name,
            "expected": text,
            "max_audio_length": max_audio_length,
            "wav_path": str(wav_path),
        }
        try:
            case_report["tts"] = _synthesize(args, text, max_audio_length, wav_path)
            if args.asr == "paraformer_container":
                asr_report = _asr_with_paraformer_container(args, wav_path)
            else:
                asr_report = _asr_with_product_qwen3(args, wav_path)
            case_report["asr"] = asr_report
            sim = _similarity(text, asr_report.get("text", ""))
            case_report["similarity"] = round(sim, 4)
            case_report["pass"] = sim >= args.min_similarity
        except Exception as exc:
            case_report["pass"] = False
            case_report["error"] = str(exc)
        if not case_report["pass"]:
            report["status"] = "FAIL"
        report["cases"].append(case_report)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "roundtrip_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
