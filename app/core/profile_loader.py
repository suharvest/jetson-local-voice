"""Runtime profile loader for deploy-time backend selection.

Profiles are intentionally thin: they set environment defaults before backend
modules are imported. Explicit environment variables still win.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level cache of the most recently loaded profile dict.
# apply_profile_from_env() populates this; current_profile() reads it.
# Empty dict means "no profile loaded" (current_profile() returns {}).
_CURRENT_PROFILE: dict = {}


def current_profile() -> dict:
    """Return the most recently loaded profile dict.

    If no profile has been applied (apply_profile_from_env returned early
    because no SEEED_LOCAL_VOICE_PROFILE env was set, or it raised), this
    returns an empty dict. Callers that require a key should raise on miss.
    """
    return _CURRENT_PROFILE


def _project_root() -> Path:
    # __file__ = <repo>/app/core/profile_loader.py → parents[2] = <repo>
    return Path(__file__).resolve().parents[2]


def _profile_path(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    if candidate.suffix != ".json":
        candidate = candidate.with_suffix(".json")
    return _project_root() / "configs" / "profiles" / candidate.name


def apply_profile_from_env() -> dict:
    """Apply SEEED_LOCAL_VOICE_PROFILE(_JSON) environment defaults.

    Returns the parsed profile dict, or an empty dict when no profile was
    requested. Environment variables already set by the operator are preserved.
    """
    profile_ref = os.environ.get("SEEED_LOCAL_VOICE_PROFILE_JSON") or os.environ.get("SEEED_LOCAL_VOICE_PROFILE")
    if not profile_ref:
        return {}

    path = _profile_path(profile_ref)
    with open(path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    global _CURRENT_PROFILE
    _CURRENT_PROFILE = profile

    env_defaults = profile.get("env", {})
    applied = []
    for key, value in env_defaults.items():
        if key not in os.environ or os.environ.get(key) == "":
            # Allow profiles to reference other env vars via ${VAR} or $VAR
            # so paths like "${QWEN3_ARTIFACT_ROOT}/engines/..." resolve at
            # apply time. Falls back to literal pass-through when the value
            # has no expansions.
            os.environ[key] = os.path.expandvars(str(value))
            applied.append(key)

    os.environ.setdefault("SEEED_LOCAL_VOICE_PROFILE_NAME", profile.get("name", path.stem))
    logger.info(
        "Applied profile %s from %s (%d env defaults; explicit env wins)",
        os.environ.get("SEEED_LOCAL_VOICE_PROFILE_NAME"),
        path,
        len(applied),
    )
    return profile
