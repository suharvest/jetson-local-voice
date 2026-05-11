from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any


DEFAULT_TEXT = "语音合成的稳定性。"


class ContractError(RuntimeError):
    pass


def _norm(path: str | os.PathLike[str]) -> str:
    return str(Path(path).expanduser().resolve())


def _bool_env(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).lower() not in ("0", "false", "no")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as reader:
        frames = reader.getnframes()
        rate = reader.getframerate()
        channels = reader.getnchannels()
        sampwidth = reader.getsampwidth()
    return {
        "sample_rate": rate,
        "channels": channels,
        "sample_width": sampwidth,
        "frames": frames,
        "duration": round(frames / rate, 3) if rate else 0.0,
        "bytes": path.stat().st_size,
    }


def _inspect_engine(path: Path) -> dict[str, Any]:
    """Return TensorRT engine metadata when TensorRT Python is available.

    The contract script must still be importable on development machines that
    do not have TensorRT installed, so TensorRT stays behind this function.
    """
    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on Jetson env
        return {"path": str(path), "available": False, "reason": str(exc)}

    logger = trt.Logger(trt.Logger.WARNING)
    with path.open("rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise ContractError(f"TensorRT failed to deserialize engine: {path}")

    tensors: dict[str, dict[str, Any]] = {}
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        item: dict[str, Any] = {
            "mode": str(engine.get_tensor_mode(name)).split(".")[-1],
            "dtype": str(engine.get_tensor_dtype(name)).split(".")[-1],
            "shape": tuple(engine.get_tensor_shape(name)),
        }
        if item["mode"] == "INPUT":
            profiles = []
            for profile_idx in range(engine.num_optimization_profiles):
                profiles.append(
                    tuple(tuple(dim) for dim in engine.get_tensor_profile_shape(name, profile_idx))
                )
            item["profiles"] = profiles
        tensors[name] = item
    return {
        "path": str(path),
        "available": True,
        "profiles": engine.num_optimization_profiles,
        "tensors": tensors,
    }


def _engine_summary(meta: dict[str, Any], names: list[str]) -> dict[str, Any]:
    if not meta.get("available"):
        return meta
    tensors = meta.get("tensors", {})
    selected = {
        name: tensors[name]
        for name in names
        if name in tensors
    }
    return {
        "path": meta.get("path"),
        "available": True,
        "profiles": meta.get("profiles"),
        "tensors": selected,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _expect_path_under(path: str, root: str, label: str) -> None:
    path_n = _norm(path)
    root_n = _norm(root)
    _require(
        path_n == root_n or path_n.startswith(root_n + os.sep),
        f"{label} loaded from unexpected path: {path_n} not under {root_n}",
    )


def _expect_path_under_any(path: str, roots: list[str], label: str) -> None:
    path_n = _norm(path)
    root_ns = [_norm(root) for root in roots if root]
    if any(path_n == root or path_n.startswith(root + os.sep) for root in root_ns):
        return
    raise ContractError(f"{label} loaded from unexpected path: {path_n} not under any of {root_ns}")


def collect_contract(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["JETSON_VOICE_TTS_BACKEND"] = "product_explicit_kv"
    os.environ["JETSON_VOICE_TTS_SEED"] = str(args.seed)
    os.environ.setdefault("JETSON_VOICE_TTS_MODEL_BASE", args.model_base)
    os.environ.setdefault("JETSON_VOICE_TTS_NATIVE_MODULE_DIR", args.native_module_dir)

    app_dir = args.app_dir
    overlay = os.environ["JETSON_VOICE_TTS_NATIVE_MODULE_DIR"]
    overlay_backends = str(Path(overlay) / "backends")
    for path in (app_dir, overlay, overlay_backends):
        if path and Path(path).is_dir() and path not in sys.path:
            sys.path.insert(0, path)

    import backends.trt_edge_llm_tts as product_wrapper

    wrapper_file = getattr(product_wrapper, "__file__", "")
    backend = product_wrapper.TRTEdgeLLMTTSBackend()
    mode = backend._backend_mode()  # contract script intentionally checks the private switch.
    _require(mode == "product_explicit_kv", f"backend mode must be product_explicit_kv, got {mode!r}")

    backend.preload()
    product_backend = getattr(backend, "_product_backend", None)
    _require(product_backend is not None, "TRTEdgeLLMTTSBackend did not load product backend")

    qwen3_mod = importlib.import_module("backends.qwen3_trt")
    qwen3_file = getattr(qwen3_mod, "__file__", "")
    _expect_path_under_any(qwen3_file, [app_dir, overlay], "backends.qwen3_trt")

    import qwen3_speech_engine  # type: ignore

    so_file = getattr(qwen3_speech_engine, "__file__", "")
    expected_so = args.expected_so
    if expected_so:
        _require(_norm(so_file) == _norm(expected_so), f"native module mismatch: {so_file} != {expected_so}")
    else:
        _expect_path_under(so_file, overlay, "qwen3_speech_engine")

    model_base = Path(os.environ["JETSON_VOICE_TTS_MODEL_BASE"])
    engines_dir = model_base / "engines"
    onnx_dir = model_base / "onnx"
    config_path = onnx_dir / "config.json"
    talker_engine = Path(os.environ.get("QWEN3_TALKER_ENGINE", engines_dir / "talker_decode_bf16.engine"))
    cp_legacy_engine = Path(os.environ.get("QWEN3_CP_ENGINE", engines_dir / "cp_bf16.engine"))
    cp_kv_engine = engines_dir / "cp_unified_bf16.engine"
    vocoder_engine = engines_dir / "vocoder_fp16.engine"

    for label, path in (
        ("model_base", model_base),
        ("config", config_path),
        ("talker_engine", talker_engine),
        ("legacy_cp_engine", cp_legacy_engine),
        ("cp_kv_engine", cp_kv_engine),
        ("vocoder_engine", vocoder_engine),
    ):
        _require(path.exists(), f"missing {label}: {path}")

    config = _load_json(config_path)
    cp_active_groups = int(config.get("cp_active_groups", config.get("num_code_groups", 16) - 1))
    _require(
        cp_active_groups == args.expected_cp_active_groups,
        f"cp_active_groups must be {args.expected_cp_active_groups}, got {cp_active_groups}",
    )
    _require(args.seed != 0, "contract verification requires a non-zero request seed")

    talker_meta = _inspect_engine(talker_engine)
    cp_meta = _inspect_engine(cp_kv_engine)
    vocoder_meta = _inspect_engine(vocoder_engine)

    if talker_meta.get("available"):
        tensors = talker_meta["tensors"]
        for name in ("inputs_embeds", "logits", "last_hidden", "past_key_0", "new_past_key_0"):
            _require(name in tensors, f"talker engine missing tensor {name}")
        expected_fp32 = {
            "inputs_embeds": "FLOAT",
            "logits": "FLOAT",
            "last_hidden": "FLOAT",
            "past_key_0": "FLOAT",
            "new_past_key_0": "FLOAT",
        }
        for name, dtype in expected_fp32.items():
            _require(tensors[name]["dtype"] == dtype, f"talker {name} dtype must be {dtype}, got {tensors[name]['dtype']}")

    if cp_meta.get("available"):
        cp_tensors = cp_meta["tensors"]
        _require(cp_kv_engine.name == "cp_unified_bf16.engine", f"unexpected CP KV engine name: {cp_kv_engine.name}")
        _require("logits_all" in cp_tensors or "logits" in cp_tensors, "CP KV engine missing logits output")
        _require("new_past_key_0" in cp_tensors, "CP KV engine missing new_past_key_0")
        _require(
            cp_tensors["new_past_key_0"]["dtype"] in ("FLOAT", "BF16"),
            f"CP KV I/O dtype must be FLOAT or BF16, got {cp_tensors['new_past_key_0']['dtype']}",
        )

    vocoder_limit = int(os.environ.get("TTS_TRT_VOCODER_MAX_FRAMES", "100"))
    if _bool_env("TTS_VOCODER_TRT", "1"):
        _require(
            args.max_audio_length <= vocoder_limit or args.expect_frame_cap,
            f"TRT vocoder path must cap max_audio_length <= {vocoder_limit}; requested {args.max_audio_length}",
        )
    if vocoder_meta.get("available"):
        audio_values = vocoder_meta["tensors"].get("audio_values")
        if audio_values:
            _require(
                tuple(audio_values["shape"]) == (1, 192000),
                f"installed TRT vocoder output changed; update contract before relying on it: {audio_values['shape']}",
            )

    if args.verbose_engine_meta:
        talker_report = talker_meta
        cp_report = cp_meta
        vocoder_report = vocoder_meta
    else:
        talker_report = _engine_summary(
            talker_meta,
            ["inputs_embeds", "logits", "last_hidden", "past_key_0", "new_past_key_0"],
        )
        cp_report = _engine_summary(
            cp_meta,
            ["inputs_embeds", "logits_all", "logits", "past_length", "new_past_key_0"],
        )
        vocoder_report = _engine_summary(
            vocoder_meta,
            ["audio_codes", "audio_values", "lengths"],
        )

    report: dict[str, Any] = {
        "backend_mode": mode,
        "wrapper_module": wrapper_file,
        "product_backend_module": qwen3_file,
        "native_module": so_file,
        "model_base": str(model_base),
        "talker_engine": str(talker_engine),
        "cp_legacy_engine": str(cp_legacy_engine),
        "cp_kv_engine": str(cp_kv_engine),
        "vocoder_engine": str(vocoder_engine),
        "cp_active_groups": cp_active_groups,
        "seed": args.seed,
        "requested_max_audio_length": args.max_audio_length,
        "trt_vocoder_frame_cap": vocoder_limit,
        "talker_engine_meta": talker_report,
        "cp_engine_meta": cp_report,
        "vocoder_engine_meta": vocoder_report,
    }

    if args.run_sample:
        wav, meta = backend.synthesize(
            args.text,
            max_audio_length=args.max_audio_length,
            segment_text=True,
            seed=args.seed,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(wav)
        wav_meta = _wav_info(output)
        _require(wav_meta["bytes"] > 44, f"sample WAV is empty: {output}")
        _require(wav_meta["sample_rate"] == 24000, f"sample rate must be 24000, got {wav_meta['sample_rate']}")
        _require(not meta.get("segmented"), f"product_explicit_kv must not use generic segmentation: {meta}")
        _require(int(meta.get("seed", -1)) == args.seed, f"sample meta seed mismatch: {meta}")
        if _bool_env("TTS_VOCODER_TRT", "1"):
            _require(
                int(meta.get("n_frames", 0)) <= vocoder_limit,
                f"TRT vocoder frame cap not enforced: {meta}",
            )
        report["sample"] = {
            "output": str(output),
            "wav": wav_meta,
            "meta": meta,
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Qwen3-TTS product correctness contract.")
    parser.add_argument("--app-dir", default="/tmp/jetson-voice-product-layer-0507/app")
    parser.add_argument("--model-base", default="/home/harvest/voice_test/models/qwen3-tts")
    parser.add_argument("--native-module-dir", default="/home/harvest/voice_test/app_overlay")
    parser.add_argument("--expected-so", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-audio-length", type=int, default=100)
    parser.add_argument("--expected-cp-active-groups", type=int, default=15)
    parser.add_argument("--expect-frame-cap", action="store_true")
    parser.add_argument("--verbose-engine-meta", action="store_true")
    parser.add_argument("--run-sample", action="store_true")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output", default="/tmp/qwen3_tts_contract.wav")
    args = parser.parse_args()

    try:
        report = collect_contract(args)
    except ContractError as exc:
        print(json.dumps({"status": "CONTRACT_FAIL", "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    print(json.dumps({"status": "CONTRACT_OK", **report}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
