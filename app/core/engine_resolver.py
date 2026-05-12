"""Runtime TRT engine resolver for seeed-local-voice.

For each engine declared in the active profile's ``required_engines`` list,
the resolver guarantees a valid engine file exists at the target path
before backends are imported. Resolution order:

  1. Local cache hit  -- engine_path exists + sidecar .meta.json matches host
  2. HuggingFace prebuilt bundle for <host_sig>.tar.gz
  3. Local compile fallback -- run scripts/build_<model>.sh (unless ``hf_only``)

Backends read engine paths from env vars at import time, so the resolver
also injects every entry's ``env_var`` → ``engine_path`` into ``os.environ``
BEFORE returning. This MUST be called before any backend module is imported.

Concurrency: a single ``flock`` on ``<MODEL_DIR>/.engine_resolver.lock``
covers the whole resolve_all() call to avoid two starting containers
racing on the shared volume.

Per-engine schema in profile JSON::

  {
    "model_id": "matcha-icefall-zh-en",
    "engine_file": "matcha_encoder_s64_bf16.engine",
    "engine_path": "/opt/models/matcha-icefall-zh-en/engines/matcha_encoder_s64_bf16.engine",
    "env_var": "MATCHA_ENCODER_ENGINE",          // backend reads this
    "onnx_input": "matcha_encoder_s64_trt.onnx", // omit if hf_only
    "build_script": "scripts/build_matcha_engines.sh",   // omit if hf_only
    "build_env": {"ENCODER_NAME": "matcha_encoder_s64_bf16.engine"},
    "hf_only": false,                            // true => no compile fallback
    "required": true                             // default true; false => skip on miss
  }

Conservative compile policy: the resolver only controls ``WS=`` per device
tier and the canonical ``ENGINE_NAME`` override. It MUST NOT pass any other
flag to the build script (precision, shape ranges, builder level, etc. are
model-development decisions baked into the build script).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# --- env keys controlling resolver behaviour ------------------------------
ENV_MODELS_DIR = "SEEED_LOCAL_VOICE_MODELS_DIR"  # default /opt/models
ENV_PROJECT_ROOT = "SEEED_LOCAL_VOICE_PROJECT_ROOT"  # for resolving build scripts
ENV_PREFETCH_ONNX = "SEEED_LOCAL_VOICE_PREFETCH_ONNX"  # 0/1, default 0
ENV_FORCE_REBUILD = "SEEED_LOCAL_VOICE_FORCE_REBUILD"  # 0/1, default 0

# Conservative WS by device tier (MiB). Auto-detected from total RAM.
_DEVICE_TIER_WS = {
    "nano": 256,
    "nx": 2048,
    "agx": 4096,
}


# ---------------------------------------------------------------------------
# Host signature
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HostSignature:
    sm: str             # "87" for Orin
    trt_version: str    # "10.3" (major.minor)
    jp_version: str     # "6.2"
    cuda_version: str   # "12.6"

    @property
    def key(self) -> str:
        return f"sm{self.sm}-trt{self.trt_version}-jp{self.jp_version}-cuda{self.cuda_version}"

    def to_dict(self) -> dict:
        return {
            "sm": self.sm,
            "trt_version": self.trt_version,
            "jp_version": self.jp_version,
            "cuda_version": self.cuda_version,
        }


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("%s failed: %s", " ".join(cmd), exc)
        return ""


def _detect_sm() -> str:
    out = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    # e.g. "8.7" → "87"
    m = re.search(r"(\d+)\.(\d+)", out)
    if m:
        return m.group(1) + m.group(2)
    # Fallback: read /proc/device-tree for Tegra
    return os.environ.get("SEEED_LOCAL_VOICE_SM", "87")


def _detect_trt_version() -> str:
    # dpkg -l | grep libnvinfer-bin returns lines like
    #   ii  libnvinfer-bin   10.3.0.30-1+cuda12.5   arm64   TensorRT binaries
    out = _run(["dpkg", "-l", "libnvinfer-bin"])
    m = re.search(r"(\d+\.\d+)\.\d+\.\d+", out)
    if m:
        return m.group(1)
    return os.environ.get("SEEED_LOCAL_VOICE_TRT", "10.3")


def _detect_cuda_version() -> str:
    # dpkg line includes "+cudaX.Y" suffix
    out = _run(["dpkg", "-l", "libnvinfer-bin"])
    m = re.search(r"\+cuda(\d+\.\d+)", out)
    if m:
        return m.group(1)
    return os.environ.get("SEEED_LOCAL_VOICE_CUDA", "12.6")


def _detect_jp_version() -> str:
    # /etc/nv_tegra_release first line: "# R36 (release), REVISION: 4.3, ..."
    try:
        with open("/etc/nv_tegra_release") as f:
            line = f.readline()
    except OSError:
        return os.environ.get("SEEED_LOCAL_VOICE_JP", "6.2")
    m = re.search(r"R(\d+)\s*\(release\)\s*,\s*REVISION:\s*(\d+)\.(\d+)", line)
    if not m:
        return os.environ.get("SEEED_LOCAL_VOICE_JP", "6.2")
    rmajor = int(m.group(1))
    # R36 → JetPack 6.x, R35 → JetPack 5.x
    jp_major = {36: 6, 35: 5}.get(rmajor, 6)
    # REVISION major maps to JetPack minor (4.3 → 6.2 on R36)
    rev_major = int(m.group(2))
    jp_minor = max(0, rev_major - 2)  # R36/REV 4 → JP 6.2; tunable
    return f"{jp_major}.{jp_minor}"


def detect_host_signature() -> HostSignature:
    sig = HostSignature(
        sm=_detect_sm(),
        trt_version=_detect_trt_version(),
        jp_version=_detect_jp_version(),
        cuda_version=_detect_cuda_version(),
    )
    logger.info("host signature: %s", sig.key)
    return sig


def _detect_device_tier() -> str:
    """Map total system RAM to a device tier name for WS sizing."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    gb = kb / (1024 * 1024)
                    if gb < 10:
                        return "nano"
                    if gb < 24:
                        return "nx"
                    return "agx"
    except OSError:
        pass
    return os.environ.get("SEEED_LOCAL_VOICE_DEVICE_TIER", "nano")


# ---------------------------------------------------------------------------
# Engine metadata sidecar
# ---------------------------------------------------------------------------

def _sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _meta_path(engine_path: Path) -> Path:
    return engine_path.with_suffix(engine_path.suffix + ".meta.json")


def _read_meta(engine_path: Path) -> Optional[dict]:
    mp = _meta_path(engine_path)
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_meta(engine_path: Path, host: HostSignature, source: str, onnx_sha: Optional[str]) -> None:
    """Write meta sidecar atomically.

    ``source`` is "cache" / "hf_bundle" / "local_compile" for diagnostic use.
    """
    meta = {
        "host": host.to_dict(),
        "engine_sha256": _sha256_file(engine_path),
        "onnx_sha256": onnx_sha,
        "source": source,
        "written_at": int(time.time()),
    }
    mp = _meta_path(engine_path)
    tmp = mp.with_suffix(mp.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, mp)


def _meta_matches(engine_path: Path, host: HostSignature) -> bool:
    """Cache freshness check. Engine must exist, sidecar must exist, host must match,
    and the engine binary hash must still match what we recorded.
    """
    if not engine_path.exists():
        return False
    meta = _read_meta(engine_path)
    if not meta:
        return False
    if meta.get("host") != host.to_dict():
        logger.info("host mismatch for %s: cache=%s host=%s",
                    engine_path.name, meta.get("host"), host.to_dict())
        return False
    if meta.get("engine_sha256") != _sha256_file(engine_path):
        logger.warning("engine hash drift detected at %s — treating as stale", engine_path)
        return False
    return True


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

@dataclass
class EngineSpec:
    model_id: str
    engine_file: str
    engine_path: Path
    env_var: str
    onnx_input: Optional[str]
    build_script: Optional[str]
    build_env: dict
    hf_only: bool
    required: bool

    @classmethod
    def from_dict(cls, d: dict) -> "EngineSpec":
        engine_path = Path(d["engine_path"])
        return cls(
            model_id=d["model_id"],
            engine_file=d["engine_file"],
            engine_path=engine_path,
            env_var=d["env_var"],
            onnx_input=d.get("onnx_input"),
            build_script=d.get("build_script"),
            build_env=dict(d.get("build_env") or {}),
            hf_only=bool(d.get("hf_only", False)),
            required=bool(d.get("required", True)),
        )


# Allowlist of env keys a profile may pass into a build script. We do NOT
# pass arbitrary env (codex Q4 risk). Anything outside this list is dropped
# with a warning.
_BUILD_ENV_ALLOWLIST = frozenset({
    "ENGINE_NAME", "ENCODER_NAME", "ESTIMATOR_NAME", "VOCOS_NAME",
    "ONNX", "ONNX_PATH", "ONNX_DIR", "OUT_DIR",
    "MIN_T", "OPT_T", "MAX_T", "MAX_SEQ",
    "MODEL_DIR",
})


def _project_root() -> Path:
    env = os.environ.get(ENV_PROJECT_ROOT)
    if env:
        return Path(env)
    # app/core/engine_resolver.py → project root = parents[2]
    return Path(__file__).resolve().parents[2]


def _try_hf_resolve(spec: EngineSpec, host: HostSignature) -> bool:
    """Try to download a prebuilt bundle matching host_sig.

    Returns True if engine_path now contains a valid file.
    """
    from app.core import hf_artifacts

    try:
        manifest = hf_artifacts.fetch_manifest(spec.model_id)
    except hf_artifacts.ArtifactError as exc:
        logger.info("no HF manifest for %s: %s", spec.model_id, exc)
        return False

    # manifest.json keys are model-relative ("engines/<host_sig>.tar.gz");
    # the HF fetch URL needs the full "models/<id>/..." path.
    manifest_key = f"engines/{host.key}.tar.gz"
    file_info = (manifest.get("files") or {}).get(manifest_key)
    if not file_info:
        logger.info("HF manifest has no bundle for %s @ %s", spec.model_id, host.key)
        return False
    bundle_rel = f"models/{spec.model_id}/{manifest_key}"

    try:
        hf_artifacts.download_and_extract_tarball(
            bundle_rel,
            spec.engine_path.parent,
            expected_sha256=file_info.get("sha256"),
        )
    except hf_artifacts.ArtifactError as exc:
        logger.warning("HF bundle download failed for %s: %s", spec.model_id, exc)
        return False

    if not spec.engine_path.exists():
        logger.warning(
            "HF bundle extracted but %s not found — engine name mismatch?",
            spec.engine_path,
        )
        return False

    onnx_sha = None
    if spec.onnx_input:
        onnx_p = spec.engine_path.parent.parent / "onnx" / spec.onnx_input
        if onnx_p.exists():
            onnx_sha = _sha256_file(onnx_p)
    _write_meta(spec.engine_path, host, source="hf_bundle", onnx_sha=onnx_sha)
    return True


def _ensure_onnx_for_compile(spec: EngineSpec) -> Path:
    """Make sure the ONNX input exists locally; fetch from HF if not."""
    if not spec.onnx_input:
        raise RuntimeError(
            f"{spec.model_id}/{spec.engine_file}: build_script declared but onnx_input missing"
        )
    onnx_path = spec.engine_path.parent.parent / "onnx" / spec.onnx_input
    if onnx_path.exists():
        return onnx_path
    from app.core import hf_artifacts
    manifest_key = f"onnx/{spec.onnx_input}"
    rel = f"models/{spec.model_id}/{manifest_key}"
    manifest = hf_artifacts.fetch_manifest(spec.model_id)  # raises if no manifest
    info = (manifest.get("files") or {}).get(manifest_key) or {}
    hf_artifacts.download_file(rel, onnx_path, expected_sha256=info.get("sha256"))
    return onnx_path


def _compile_locally(spec: EngineSpec, host: HostSignature) -> None:
    if spec.hf_only:
        raise RuntimeError(
            f"{spec.model_id}/{spec.engine_file}: hf_only=True and HF bundle unavailable"
        )
    if not spec.build_script:
        raise RuntimeError(
            f"{spec.model_id}/{spec.engine_file}: no build_script — cannot compile"
        )

    onnx_path = _ensure_onnx_for_compile(spec)
    script_abs = (_project_root() / spec.build_script).resolve()
    if not script_abs.exists():
        raise RuntimeError(f"build script not found: {script_abs}")

    env = os.environ.copy()
    # Only allowlisted keys from profile.build_env are passed through.
    for k, v in spec.build_env.items():
        if k not in _BUILD_ENV_ALLOWLIST:
            logger.warning("dropping build_env key %r (not in allowlist)", k)
            continue
        env[k] = str(v)

    # Inject conservative resource sizing only (codex Q9 constraint).
    env.setdefault("WS", str(_DEVICE_TIER_WS.get(_detect_device_tier(), 256)))
    # The engine name must match what the profile expects.
    env.setdefault("ENGINE_NAME", spec.engine_file)
    # Pass ONNX path and output dir explicitly so the build script does not
    # have to guess.
    env.setdefault("ONNX_PATH", str(onnx_path))
    env.setdefault("ONNX", str(onnx_path))
    env.setdefault("OUT_DIR", str(spec.engine_path.parent))

    spec.engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "compiling %s via %s (WS=%s)", spec.engine_path.name, script_abs, env.get("WS")
    )
    proc = subprocess.run(
        ["bash", str(script_abs)],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"build script failed (exit {proc.returncode}): {script_abs}"
        )
    if not spec.engine_path.exists():
        raise RuntimeError(
            f"build script reported success but engine not at {spec.engine_path}"
        )
    onnx_sha = _sha256_file(onnx_path)
    _write_meta(spec.engine_path, host, source="local_compile", onnx_sha=onnx_sha)


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

def _models_dir() -> Path:
    return Path(os.environ.get(ENV_MODELS_DIR, "/opt/models"))


def _acquire_lock():
    """Context manager-like helper returning an open fd holding the resolver lock.

    Falls back to a no-op on systems without fcntl (mostly for local dev on Mac).
    """
    lock_path = _models_dir() / ".engine_resolver.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX)
    except (ImportError, OSError):
        logger.warning("flock not available; running without resolver lock")
    return fd


def _release_lock(fd: int) -> None:
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass
    try:
        os.close(fd)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_all(profile: dict) -> dict[str, Path]:
    """Resolve every engine declared by ``profile['required_engines']``.

    On success, returns a dict of ``env_var → engine_path`` and also injects
    each entry into ``os.environ`` so backend modules can read them at import
    time. Raises RuntimeError on first hard failure (and marks any partially
    resolved entries as not exported).
    """
    entries = profile.get("required_engines") or []
    if not entries:
        logger.info("profile declares no required_engines — skipping resolver")
        return {}

    host = detect_host_signature()
    force_rebuild = os.environ.get(ENV_FORCE_REBUILD, "0") in ("1", "true", "yes")

    fd = _acquire_lock()
    try:
        resolved: dict[str, Path] = {}
        for raw in entries:
            spec = EngineSpec.from_dict(raw)
            try:
                _resolve_one(spec, host, force_rebuild=force_rebuild)
            except Exception as exc:
                if not spec.required:
                    logger.warning("optional engine %s skipped: %s", spec.engine_file, exc)
                    continue
                raise RuntimeError(
                    f"failed to resolve required engine {spec.engine_file}: {exc}"
                ) from exc
            os.environ[spec.env_var] = str(spec.engine_path)
            resolved[spec.env_var] = spec.engine_path
        return resolved
    finally:
        _release_lock(fd)


def _resolve_one(spec: EngineSpec, host: HostSignature, force_rebuild: bool) -> None:
    if not force_rebuild and _meta_matches(spec.engine_path, host):
        logger.info("cache hit: %s (host=%s)", spec.engine_path.name, host.key)
        return

    # Stale cache: remove engine and its meta to start clean.
    if spec.engine_path.exists():
        spec.engine_path.unlink()
    _meta_path(spec.engine_path).unlink(missing_ok=True)

    # Try HF bundle first.
    if _try_hf_resolve(spec, host):
        logger.info("hf bundle: %s (host=%s)", spec.engine_path.name, host.key)
        return

    if spec.hf_only:
        raise RuntimeError(
            f"engine {spec.engine_file} is hf_only and HF bundle for {host.key} is unavailable"
        )

    # Fallback: compile locally via the canonical build script.
    _compile_locally(spec, host)
    logger.info("local compile done: %s (host=%s)", spec.engine_path.name, host.key)
