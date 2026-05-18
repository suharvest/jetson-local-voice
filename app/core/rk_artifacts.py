"""Optional Rockchip model artifact downloader.

RK userspace runtime libraries are baked into the image. Model artifacts
(.rknn/.rkllm/tokenizer/config/lexicon) are larger and SoC/profile-specific,
so they are described by an external manifest when automatic download is used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://huggingface.co"
DEFAULT_REVISION = "main"
_UA = "openvoicestream-rk/1.0"


class RKArtifactError(RuntimeError):
    """Raised when RK artifacts cannot be downloaded or verified."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise RKArtifactError(f"download failed: {url}: {exc}") from exc
    os.replace(tmp, dest)


def _load_manifest() -> dict | None:
    manifest_path = os.environ.get("RK_ARTIFACT_MANIFEST", "").strip()
    repo_id = os.environ.get("RK_ARTIFACT_REPO_ID", "").strip()
    if manifest_path:
        path = Path(manifest_path)
        if not path.exists():
            raise RKArtifactError(f"RK_ARTIFACT_MANIFEST not found: {path}")
        return json.loads(path.read_text())
    if repo_id:
        endpoint = os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")
        revision = os.environ.get("RK_ARTIFACT_REVISION", DEFAULT_REVISION)
        url = f"{endpoint}/{repo_id}/resolve/{revision}/rk_manifest.json"
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise RKArtifactError(f"failed to fetch RK manifest: {url}: {exc}") from exc
    return None


def _contract_strict() -> bool:
    return os.environ.get("RK_ARTIFACT_CONTRACT_STRICT", "1").lower() not in (
        "0",
        "false",
        "no",
    )


def _validate_runtime_contract(spec: dict) -> None:
    """Validate runtime env against the selected RK artifact set contract.

    RKNN artifacts are static-shape model binaries. A wrong runtime shape env
    can still produce audio bytes, but the audio may fail closed-loop ASR.
    Keep the expected env next to the artifact set so profiles and compose
    files cannot silently drift away from the validated model contract.
    """
    contract = spec.get("runtime_contract") or {}
    expected_env = contract.get("env") or {}
    errors: list[str] = []
    for key, expected in expected_env.items():
        got = os.environ.get(key)
        expected_s = str(expected)
        if got != expected_s:
            errors.append(f"{key}: got {got!r}, expected {expected_s!r}")

    if not errors:
        return

    message = (
        "RK artifact runtime contract mismatch for selected artifact set. "
        "Use the profile/compose env that was validated with these artifacts, "
        "or publish a new artifact set with matching runtime_contract. "
        "Details: "
        + "; ".join(errors)
    )
    if _contract_strict():
        raise RKArtifactError(message)
    logger.warning("%s", message)


def ensure_rk_artifacts() -> None:
    """Download RK artifacts if an RK manifest/repo is configured.

    No-op by default so existing host-mounted deployments continue to work.
    Set RK_ARTIFACT_MANIFEST or RK_ARTIFACT_REPO_ID to enable.
    """
    if os.environ.get("RK_ARTIFACT_AUTO_DOWNLOAD", "1").lower() in ("0", "false", "no"):
        logger.info("RK artifact auto-download disabled.")
        return

    manifest = _load_manifest()
    if not manifest:
        logger.info("No RK artifact manifest configured; using mounted/model-volume artifacts.")
        return

    set_name = os.environ.get("RK_ARTIFACT_SET") or manifest.get("default_set")
    sets = manifest.get("artifact_sets") or {}
    spec = sets.get(set_name)
    if not set_name or not spec:
        raise RKArtifactError(
            f"RK artifact set {set_name!r} not found; available={sorted(sets)}"
        )

    root = Path(os.environ.get("RK_ARTIFACT_ROOT") or spec.get("root") or "/")
    repo_id = os.environ.get("RK_ARTIFACT_REPO_ID") or manifest.get("hf_repo_id")
    endpoint = os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")
    revision = os.environ.get("RK_ARTIFACT_REVISION") or manifest.get("revision") or DEFAULT_REVISION
    if not repo_id:
        raise RKArtifactError("RK artifact manifest must declare hf_repo_id or set RK_ARTIFACT_REPO_ID")

    logger.info("Ensuring RK artifact set %s under %s", set_name, root)
    for item in spec.get("files") or []:
        rel = item["path"].lstrip("/")
        source_rel = item.get("source_path", rel).lstrip("/")
        dest = root / rel
        expected_sha = item.get("sha256")
        if dest.exists() and (not expected_sha or _sha256(dest) == expected_sha):
            logger.info("RK artifact OK: %s", dest)
            continue
        url = f"{endpoint}/{repo_id}/resolve/{revision}/{source_rel}"
        logger.info("Downloading RK artifact %s -> %s", source_rel, rel)
        _download(url, dest)
        if expected_sha:
            got = _sha256(dest)
            if got != expected_sha:
                dest.unlink(missing_ok=True)
                raise RKArtifactError(
                    f"sha256 mismatch for {rel}: got {got}, expected {expected_sha}"
                )
    _validate_runtime_contract(spec)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        ensure_rk_artifacts()
    except RKArtifactError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
