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
        "sensevoice": ("models-sensevoice.tar.gz", "SenseVoice offline ASR (5 languages)"),
    },
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
            req = urllib.request.Request(url, headers={"User-Agent": "jetson-voice/1.0"})
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
    required = {}
    required.update(MODELS.get(language_mode, MODELS["zh_en"]))
    # SenseVoice is optional — don't block startup for it (lazy-loaded)

    missing = []
    for dir_name, (cdn_file, desc) in required.items():
        model_path = os.path.join(model_dir, dir_name)
        if os.path.isdir(model_path) and os.listdir(model_path):
            logger.info("Model OK: %s (%s)", dir_name, desc)
        else:
            missing.append((dir_name, cdn_file, desc))

    if not missing:
        logger.info("All models for mode '%s' are ready.", language_mode)
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    mode = os.environ.get("LANGUAGE_MODE", "zh_en")
    model_dir = os.environ.get("MODEL_DIR", "/opt/models")
    ensure_models(mode, model_dir)
