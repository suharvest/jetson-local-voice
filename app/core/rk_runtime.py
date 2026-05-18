"""Rockchip runtime compatibility checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("/opt/rk-runtime/MANIFEST.json")
DEFAULT_RKNNRT = Path("/usr/lib/librknnrt.so")
DEFAULT_RKLLMRT = Path("/opt/asr/lib/librkllmrt.so")


class RKRuntimeError(RuntimeError):
    """Raised when the Rockchip runtime does not match the image manifest."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strict() -> bool:
    return os.environ.get("RK_RUNTIME_STRICT", "1").lower() not in ("0", "false", "no")


def _check_file(label: str, path: Path, expected: dict, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"{label} missing at {path}")
        return
    got_size = path.stat().st_size
    got_sha = _sha256(path)
    exp_size = int(expected.get("size") or 0)
    exp_sha = str(expected.get("sha256") or "")
    if exp_size and got_size != exp_size:
        errors.append(f"{label} size mismatch: got {got_size}, expected {exp_size}")
    if exp_sha and got_sha != exp_sha:
        errors.append(f"{label} sha256 mismatch: got {got_sha}, expected {exp_sha}")
    logger.info("%s OK: %s sha256=%s", label, path, got_sha[:12])


def check_rk_runtime(profile: dict | None = None) -> None:
    """Validate the RK userspace runtime before RK backends import native libs.

    The RK image vendors known-good userspace libraries. If an operator bind
    mounts a different library over them, fail early with an actionable message
    instead of letting RKNN/RKLLM fail later with opaque native errors.
    """
    profile = profile or {}
    env = profile.get("env") or {}
    if env.get("LANGUAGE_MODE") != "rk" and os.environ.get("LANGUAGE_MODE") != "rk":
        return

    manifest_path = Path(os.environ.get("RK_RUNTIME_MANIFEST", DEFAULT_MANIFEST))
    if not manifest_path.exists():
        msg = f"RK runtime manifest missing at {manifest_path}"
        if _strict():
            raise RKRuntimeError(msg)
        logger.warning("%s; continuing because RK_RUNTIME_STRICT=0", msg)
        return

    manifest = json.loads(manifest_path.read_text())
    runtime = manifest.get("runtime") or {}
    errors: list[str] = []

    expected_lite = str(runtime.get("rknn_toolkit_lite2") or "")
    try:
        got_lite = importlib.metadata.version("rknn-toolkit-lite2")
    except importlib.metadata.PackageNotFoundError:
        got_lite = ""
    if expected_lite and got_lite != expected_lite:
        errors.append(
            f"rknn-toolkit-lite2 mismatch: got {got_lite or 'missing'}, expected {expected_lite}"
        )
    else:
        logger.info("rknn-toolkit-lite2 OK: %s", got_lite)

    _check_file(
        "librknnrt",
        Path(os.environ.get("RKNNRT_LIB_PATH", DEFAULT_RKNNRT)),
        runtime.get("librknnrt") or {},
        errors,
    )
    _check_file(
        "librkllmrt",
        Path(os.environ.get("RKLLM_LIB_PATH", DEFAULT_RKLLMRT)),
        runtime.get("rkllm_runtime") or {},
        errors,
    )

    if not errors:
        return

    guidance = (
        "Rockchip runtime version mismatch. Use the runtime libraries baked into "
        "this image, remove overriding host mounts for librknnrt/librkllmrt, or "
        "update the device BSP/runtime to the version declared in "
        f"{manifest_path}. If you intentionally use a different runtime, publish "
        "a matching RK artifact set and manifest; set RK_RUNTIME_STRICT=0 only "
        "for debugging."
    )
    message = guidance + " Details: " + "; ".join(errors)
    if _strict():
        raise RKRuntimeError(message)
    logger.warning(message)
