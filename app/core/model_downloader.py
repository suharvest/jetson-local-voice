"""On-demand model downloader.

Checks if required models exist for the current LANGUAGE_MODE.
Downloads missing models from CDN on first start; cached in /opt/models volume.

Models baked into the Docker image (zh_en) are always available.
English-only models (Kokoro TTS + Zipformer ASR) are downloaded on demand
when LANGUAGE_MODE=en, keeping the image small for default users.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

CDN_BASE = "https://sensecraft-statics.seeed.cc/solution-app/jetson-voice"

# Model registry: {dir_name: (cdn_filename, description)}
MODELS = {
    "zh_en": {
        "matcha-icefall-zh-en": ("models-matcha.tar.gz", "Matcha TTS (zh+en)"),
        "paraformer-streaming": ("models-paraformer.tar.gz", "Paraformer streaming ASR (zh+en)"),
    },
    "en": {
        "kokoro-multi-lang-v1_0": ("kokoro-multi-lang-v1_0.tar.bz2", "Kokoro TTS v1.0 (English, 53 speakers)"),
        "zipformer-en": ("models-zipformer-en.tar.gz", "Zipformer streaming ASR (English)"),
    },
    "shared": {
        "sensevoice": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
            "SenseVoice offline ASR (5 languages)",
        ),
    },
}

# Per-model files the freshness check insists on seeing.
# Without this, model dirs that engine_resolver populated with only
# auxiliary subdirs (engines/, onnx/ skeletons) pass the "non-empty"
# heuristic but still miss load-bearing resources such as tokens.txt.
_REQUIRED_FILES = {
    "matcha-icefall-zh-en": ("model-steps-3.onnx", "tokens.txt", "lexicon.txt"),
    "paraformer-streaming": ("encoder.onnx", "tokens.txt"),
    "zipformer-en": ("encoder.int8.onnx", "tokens.txt"),
    "kokoro-multi-lang-v1_0": ("model.onnx",),
    "sensevoice": ("model.int8.onnx",),
}


def _detect_tar_mode(filename: str) -> str:
    """Return tar open mode based on filename extension."""
    if filename.endswith(".tar.bz2"):
        return "bz2"
    return "gz"


def _download_and_extract(url: str, dest_dir: str) -> None:
    """Download a .tar.gz or .tar.bz2 from URL and extract to dest_dir.

    Uses curl (fast, with progress) if available, falls back to Python stdlib.
    """
    compress = _detect_tar_mode(url)

    if shutil.which("curl"):
        # curl + tar streaming: no temp file, shows progress
        tar_flag = "j" if compress == "bz2" else "z"
        cmd = f'curl -fSL --progress-bar "{url}" | tar x{tar_flag}f - -C "{dest_dir}"'
        subprocess.run(cmd, shell=True, check=True)
    else:
        # Pure Python fallback
        import tarfile
        import tempfile
        import urllib.request

        suffix = ".tar.bz2" if compress == "bz2" else ".tar.gz"
        logger.info("  Fetching %s ...", url)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            req = urllib.request.Request(url, headers={"User-Agent": "openvoicestream/1.0"})
            resp = urllib.request.urlopen(req, timeout=600)
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (10 * 1024 * 1024) < 1024 * 1024:
                    pct = downloaded * 100 // total
                    mb = downloaded // (1024 * 1024)
                    total_mb = total // (1024 * 1024)
                    logger.info("  Progress: %d/%d MB (%d%%)", mb, total_mb, pct)
        try:
            logger.info("  Extracting to %s ...", dest_dir)
            with tarfile.open(tmp_path, f"r:{compress}") as tar:
                tar.extractall(path=dest_dir)
        finally:
            os.unlink(tmp_path)


def ensure_models(language_mode: str = "zh_en", model_dir: str = "/opt/models") -> None:
    """Ensure all required models for the given language mode are present."""
    if language_mode == "rk":
        _ensure_rk_artifacts()
        if os.environ.get("RK_ENSURE_MATCHA_RESOURCES", "1").lower() in ("0", "false", "no"):
            return
        matcha = MODELS.get("zh_en", {}).get("matcha-icefall-zh-en")
        required = {"matcha-icefall-zh-en": matcha} if matcha else {}
        model_dir = os.environ.get("TTS_MODEL_DIR") or model_dir

    elif language_mode == "multilanguage":
        _ensure_qwen3_artifacts()
        # Some multilanguage profiles pair Qwen3 ASR with Matcha TTS. Only
        # those need the Matcha acoustic ONNX + lexicon; pure Qwen3 profiles
        # should not download or validate Matcha assets during startup.
        try:
            from app.core.profile_loader import current_profile
            tts_backend = (current_profile() or {}).get("tts_backend")
        except Exception:
            tts_backend = None
        matcha = MODELS.get("zh_en", {}).get("matcha-icefall-zh-en")
        if tts_backend == "jetson.matcha_trt" and matcha:
            required = {"matcha-icefall-zh-en": matcha}
        else:
            return
    else:
        required = {}
        required.update(MODELS.get(language_mode, {}))
        if os.environ.get("ENSURE_OFFLINE_ASR", "").lower() in ("1", "true", "yes"):
            required.update(MODELS.get("shared", {}))
    if not required:
        return

    missing = []
    for dir_name, (cdn_file, desc) in required.items():
        model_path = os.path.join(model_dir, dir_name)
        required_files = _REQUIRED_FILES.get(dir_name)
        # When required files are declared, look for the actual load-bearing
        # files recursively under the model dir (the tarball lays files
        # under subdirs in some upstream variants). Non-empty dir alone
        # is NOT a sufficient signal — engine_resolver may have written
        # the engines/ subdir before model_downloader runs.
        is_ready = False
        if os.path.isdir(model_path):
            if required_files:
                found = set()
                for root, _dirs, files in os.walk(model_path):
                    found.update(name for name in required_files if name in files)
                is_ready = found == set(required_files)
            elif os.listdir(model_path):
                is_ready = True
        if is_ready:
            logger.info("Model OK: %s (%s)", dir_name, desc)
        else:
            missing.append((dir_name, cdn_file, desc))

    if not missing:
        logger.info("All models for mode '%s' are ready.", language_mode)
        if language_mode == "en":
            _patch_kokoro_voices(model_dir)
        return

    logger.info(
        "Downloading %d missing model(s) for mode '%s'...",
        len(missing), language_mode,
    )

    os.makedirs(model_dir, exist_ok=True)

    for dir_name, cdn_file, desc in missing:
        # Use GitHub releases for models not hosted on CDN
        if cdn_file.startswith("http"):
            url = cdn_file
        elif cdn_file == "kokoro-multi-lang-v1_0.tar.bz2":
            url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{cdn_file}"
        else:
            url = f"{CDN_BASE}/{cdn_file}"
        logger.info("Downloading %s ...", desc)
        try:
            _download_and_extract(url, model_dir)
            logger.info("Downloaded %s OK.", desc)
        except Exception as e:
            logger.error("Failed to download %s: %s", desc, e)
            logger.error(
                "You can manually download from %s and extract to %s",
                url, model_dir,
            )
            sys.exit(1)

    if language_mode == "en":
        _patch_kokoro_voices(model_dir)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_qwen3_artifacts() -> None:
    """Verify or download Qwen3 artifacts for the active multilanguage profile.

    The deploy script + manifest live in the sibling `qwen3-edgellm-jetson`
    repo so they are not duplicated here. Set `QWEN3_EDGELLM_JETSON_ROOT` to
    override the default `~/project/qwen3-edgellm-jetson` lookup path.
    """
    if os.environ.get("QWEN3_ARTIFACT_AUTO_DOWNLOAD", "1").lower() in ("0", "false", "no"):
        logger.info("Qwen3 artifact auto-download disabled.")
        return

    qej_root = Path(
        os.environ.get(
            "QWEN3_EDGELLM_JETSON_ROOT",
            os.path.expanduser("~/project/qwen3-edgellm-jetson"),
        )
    )
    script = qej_root / "scripts" / "deploy_qwen3_artifacts.py"
    manifest = os.environ.get(
        "QWEN3_ARTIFACT_MANIFEST",
        str(qej_root / "deploy" / "artifacts" / "qwen3_manifest.json"),
    )
    artifact_set = os.environ.get("QWEN3_ARTIFACT_SET") or "orin-nano-highperf-2026-05-10"
    root = os.environ.get("QWEN3_ARTIFACT_ROOT")
    if not script.exists():
        logger.warning(
            "Qwen3 artifact deploy script missing at %s. Clone "
            "https://github.com/suharvest/qwen3-edgellm-jetson.git as a sibling "
            "of OpenVoiceStream or set QWEN3_EDGELLM_JETSON_ROOT to point at it.",
            script,
        )
        return

    cmd = [sys.executable, str(script), "--manifest", manifest, "--set", artifact_set]
    if root:
        cmd.extend(["--root", root])
    if os.environ.get("QWEN3_ARTIFACT_VERIFY_SHA256", "1").lower() not in ("0", "false", "no"):
        cmd.append("--verify-sha256")
    logger.info("Ensuring Qwen3 artifact set %s via %s", artifact_set, manifest)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Qwen3 artifact check/download failed with exit code %s", exc.returncode)
        sys.exit(exc.returncode)


def _ensure_rk_artifacts() -> None:
    """Verify or download RK model artifacts when an RK manifest is configured."""
    try:
        from app.core.rk_artifacts import ensure_rk_artifacts
        ensure_rk_artifacts()
    except Exception as exc:
        logger.error("RK artifact check/download failed: %s", exc)
        sys.exit(1)


# Custom voice patches: replace unused speakers in voices.bin with custom voices.
# Each voice embedding is (510, 1, 256) float32 = 522240 bytes.
# Patches are stored in /opt/speech/voices/ (baked into Docker image).
_VOICE_PATCHES = {
    52: "af_cute.bin",  # replaces zm_yunyang (sid=52) with cute voice
}
_VOICE_BYTES = 510 * 1 * 256 * 4  # 522240


def _patch_kokoro_voices(model_dir: str) -> None:
    """Patch voices.bin with custom voice embeddings if not already applied."""
    voices_bin = os.path.join(model_dir, "kokoro-multi-lang-v1_0", "voices.bin")
    if not os.path.isfile(voices_bin):
        return

    patch_dir = os.path.join(os.path.dirname(__file__), "..", "voices")
    marker = voices_bin + ".patched"

    if os.path.isfile(marker):
        return

    for sid, patch_file in _VOICE_PATCHES.items():
        patch_path = os.path.join(patch_dir, patch_file)
        if not os.path.isfile(patch_path):
            logger.warning("Voice patch %s not found, skipping", patch_path)
            continue
        with open(patch_path, "rb") as f:
            patch_data = f.read()
        if len(patch_data) != _VOICE_BYTES:
            logger.warning("Voice patch %s has wrong size %d, skipping", patch_file, len(patch_data))
            continue
        offset = sid * _VOICE_BYTES
        with open(voices_bin, "r+b") as f:
            f.seek(offset)
            f.write(patch_data)
        logger.info("Patched voices.bin sid=%d with %s", sid, patch_file)

    # Write marker so we don't re-patch on every startup
    with open(marker, "w") as f:
        f.write("patched\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    mode = os.environ.get("LANGUAGE_MODE", "zh_en")
    model_dir = os.environ.get("MODEL_DIR", "/opt/models")
    ensure_models(mode, model_dir)
