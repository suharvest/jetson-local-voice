"""Map a user-facing ``preset`` (voice_clone / multilang / lite_zh_en / ...)
plus the auto-detected device tier to a concrete profile JSON name.

Operators set ``OVS_PRESET`` (or pass an explicit ``OVS_PROFILE`` to override).
Legacy ``SEEED_LOCAL_VOICE_*`` variables remain supported. This module owns:

  1. Detecting which device the container is running on (Jetson Nano/NX/AGX
     are the same image but different SKUs; RK and RPi each have their own
     image already, but the detector still works so a future "one image"
     redesign is possible).
  2. Looking up the (device, preset) → profile_name mapping table for the
     curated recommended-set per use case.

Presets (top-level user intent)
-------------------------------
- ``voice_clone``   : Qwen3 ASR + Qwen3 TTS. Multilingual, voice cloning,
                      best audio quality. Single-stream, Jetson only.
- ``multilang``     : Qwen3 ASR + Matcha TTS. Multilingual input, zh+en
                      output, supports multi-stream. Jetson + RK.
- ``lite_zh_en``    : Paraformer ASR + Matcha TTS. zh+en only, max
                      throughput.  All accelerated devices.
- ``lite_en``       : (planned) English-only ASR + TTS.
- ``asr_zh_en``     : (planned) ASR-only zh+en for very small devices.
- ``asr_en``        : (planned) English ASR-only.

The PRESET_TABLE below only enumerates the Jetson combinations we have
verified end-to-end so far; other (device, preset) combinations raise
``UnsupportedPreset`` until their profile is authored and validated.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class UnsupportedPreset(RuntimeError):
    """Raised when (device, preset) has no curated profile yet."""


# ---------------------------------------------------------------------------
# Device tier detection
# ---------------------------------------------------------------------------

# Canonical device tier strings. Keep tight; one tier per hardware SKU we
# explicitly support a curated profile for.
TIER_JETSON_ORIN_NANO = "jetson-orin-nano"
TIER_JETSON_ORIN_NX = "jetson-orin-nx"
TIER_JETSON_ORIN_AGX = "jetson-orin-agx"
TIER_RK3576 = "rk3576"
TIER_RK3588 = "rk3588"
TIER_RPI5 = "rpi5"
TIER_RPI4 = "rpi4"
TIER_UNKNOWN = "unknown"


def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="ignore")
    except OSError:
        return ""


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _detect_total_ram_gb() -> float:
    txt = _read("/proc/meminfo")
    m = re.search(r"^MemTotal:\s+(\d+)\s*kB", txt, re.MULTILINE)
    if not m:
        return 0.0
    return int(m.group(1)) / (1024 * 1024)


def _detect_jetson_tier() -> Optional[str]:
    """Return jetson tier string if we are on a Jetson Orin board, else None."""
    if not os.path.exists("/etc/nv_tegra_release"):
        return None
    # SoC family from device tree
    model = _read("/proc/device-tree/model").strip("\x00").strip()
    ram_gb = _detect_total_ram_gb()
    # Heuristics by RAM: Nano = 8GB, NX = 16GB, AGX = 32-64GB.
    # Cross-check with the model name when present.
    if "AGX" in model.upper() or ram_gb >= 24:
        return TIER_JETSON_ORIN_AGX
    if "NX" in model.upper() or ram_gb >= 12:
        return TIER_JETSON_ORIN_NX
    return TIER_JETSON_ORIN_NANO


def _detect_rk_tier() -> Optional[str]:
    # Some boards (e.g. Radxa ROCK 5T) put only a marketing name in
    # /proc/device-tree/model and expose the SoC family via the
    # `compatible` node instead. Check both. Inside Docker neither
    # /proc/device-tree nor /sys/firmware is exposed by default, so
    # also fall back to /proc/cpuinfo's "Hardware" field (kernel-
    # populated, visible inside containers).
    model = _read("/proc/device-tree/model").lower()
    compat = _read("/proc/device-tree/compatible").lower()
    cpuinfo_hw = ""
    if not model and not compat:
        for line in _read("/proc/cpuinfo").splitlines():
            if line.lower().startswith(("hardware", "model")):
                _, _, val = line.partition(":")
                cpuinfo_hw += " " + val.strip().lower()
    haystack = model + " " + compat + " " + cpuinfo_hw
    if "rk3588" in haystack:
        return TIER_RK3588
    if "rk3576" in haystack:
        return TIER_RK3576
    return None


def _detect_rpi_tier() -> Optional[str]:
    # /proc/device-tree is not exposed inside an unprivileged Docker
    # container; fall back to /proc/cpuinfo's "Model" field which IS
    # visible (kernel-populated, not bind-mounted).
    model = _read("/proc/device-tree/model").lower()
    if not model:
        for line in _read("/proc/cpuinfo").splitlines():
            if line.lower().startswith("model"):
                _, _, val = line.partition(":")
                model = val.strip().lower()
                break
    if "raspberry pi 5" in model or "raspberry pi compute module 5" in model:
        return TIER_RPI5
    if "raspberry pi 4" in model or "raspberry pi compute module 4" in model:
        return TIER_RPI4
    return None


def detect_device_tier() -> str:
    """Best-effort detection. Falls back to ``unknown`` so callers can decide
    whether to error or accept an explicit override.

    Operators can force a tier by setting ``OVS_DEVICE_TIER``.
    """
    override = os.environ.get("OVS_DEVICE_TIER") or os.environ.get("SEEED_LOCAL_VOICE_DEVICE_TIER")
    if override:
        logger.info("device tier from env: %s", override)
        return override
    for fn in (_detect_jetson_tier, _detect_rk_tier, _detect_rpi_tier):
        tier = fn()
        if tier:
            logger.info("device tier auto-detected: %s", tier)
            return tier
    logger.warning("device tier unknown — no Jetson/RK/RPi signature found")
    return TIER_UNKNOWN


# ---------------------------------------------------------------------------
# Preset → profile table
# ---------------------------------------------------------------------------

# (device_tier, preset) → profile filename stem (without .json).
# Only entries listed here are considered supported. Adding a new
# (tier, preset) combination requires authoring a profile JSON and
# validating it on real hardware.
PRESET_TABLE: dict[tuple[str, str], str] = {
    # Jetson — verified end-to-end on Orin Nano + NX.
    (TIER_JETSON_ORIN_NANO, "voice_clone"): "jetson-multilang-highperf",
    (TIER_JETSON_ORIN_NX,   "voice_clone"): "jetson-multilang-highperf-nx",
    (TIER_JETSON_ORIN_AGX,  "voice_clone"): "jetson-multilang-highperf-nx",

    (TIER_JETSON_ORIN_NANO, "multilang"):   "jetson-qwen3asr-matcha",
    (TIER_JETSON_ORIN_NX,   "multilang"):   "jetson-qwen3asr-matcha-nx",
    (TIER_JETSON_ORIN_AGX,  "multilang"):   "jetson-qwen3asr-matcha-nx",

    (TIER_JETSON_ORIN_NANO, "lite_zh_en"):  "jetson-zh-en",
    (TIER_JETSON_ORIN_NX,   "lite_zh_en"):  "jetson-zh-en",
    (TIER_JETSON_ORIN_AGX,  "lite_zh_en"):  "jetson-zh-en",

    # RPi — ASR-only preset (asr_zh_en) for CM4 / RPi4 with no spare CPU
    # budget for TTS. Reference: https://github.com/Allenkzl/Local-Automatic-Speech-Recognition
    (TIER_RPI4, "asr_zh_en"): "rpi4-asr-zh-en",
    # RPi5 inherits the same ASR-only profile by default; future
    # rpi5-lite-zh-en would add Matcha TTS on top when validated.
    (TIER_RPI5, "asr_zh_en"): "rpi4-asr-zh-en",

    # RK — multilang preset (Qwen3 ASR via RKNN/RKLLM + Matcha TTS via RKNN).
    # Backed by rkvoice-stream submodule; needed env to drive its backend
    # away from the rk3576/w8a8 defaults is set in the profile JSON.
    (TIER_RK3576, "multilang"): "rk3576-multilang",
    (TIER_RK3588, "multilang"): "rk3588-multilang",

    # Future entries (profiles not yet authored):
    #   (TIER_RK3576, "lite_zh_en"): "rk3576-lite-zh-en",
    #   (TIER_RK3588, "lite_zh_en"): "rk3588-lite-zh-en",
    #   (TIER_RPI5,   "lite_zh_en"): "rpi5-lite-zh-en",
}

KNOWN_PRESETS = sorted({preset for _, preset in PRESET_TABLE.keys()})


def resolve_profile_name(preset: str, device_tier: Optional[str] = None) -> str:
    """Return the profile-JSON name for a (preset, device) pair.

    Raises ``UnsupportedPreset`` if the combination is not in the table.
    """
    if device_tier is None:
        device_tier = detect_device_tier()
    key = (device_tier, preset)
    if key not in PRESET_TABLE:
        supported_for_device = [
            p for (d, p) in PRESET_TABLE if d == device_tier
        ]
        raise UnsupportedPreset(
            f"No curated profile for device={device_tier!r} preset={preset!r}. "
            f"Supported presets on this device: {supported_for_device or '[none]'}; "
            f"all known presets: {KNOWN_PRESETS}"
        )
    return PRESET_TABLE[key]
