#!/usr/bin/env python3
"""Verify or download Qwen3 artifact sets for Jetson Voice.

This script is intentionally independent of the Qwen runtime package. It keeps
Jetson Voice deployable from a JSON profile plus a Hugging Face artifact repo.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


def load_manifest(path: str) -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def required_paths(manifest: dict, set_name: str, root_override: str | None) -> tuple[Path, list[Path]]:
    sets = manifest.get("artifact_sets", {})
    if set_name not in sets:
        raise KeyError(f"artifact set {set_name!r} not found; available: {', '.join(sorted(sets))}")
    artifact_set = sets[set_name]
    root = Path(root_override or os.environ.get("QWEN3_ARTIFACT_ROOT") or artifact_set.get("root") or "/opt/models/qwen3-edgellm")
    paths = [root / rel for rel in artifact_set.get("required_files", [])]
    return root, paths


def verify(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def snapshot_download(manifest: dict, set_name: str, root: Path) -> None:
    repo_id = os.environ.get("QWEN3_HF_REPO_ID") or manifest.get("hf_repo_id")
    if not repo_id or repo_id.startswith("REPLACE_WITH_"):
        raise RuntimeError(
            "Qwen3 HF repo is not configured. Set QWEN3_HF_REPO_ID or fill deploy/artifacts/qwen3_manifest.json."
        )
    revision = os.environ.get("QWEN3_HF_REVISION") or manifest.get("revision", "main")
    repo_type = manifest.get("repo_type", "model")
    include = [f"{rel}*" for rel in manifest["artifact_sets"][set_name].get("required_files", [])]

    root.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        hf_snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=str(root),
            allow_patterns=include,
            local_dir_use_symlinks=False,
        )
        return
    except ImportError:
        pass

    hf_bin = shutil.which("hf")
    if hf_bin:
        cmd = [
            hf_bin,
            "download",
            repo_id,
            "--repo-type",
            repo_type,
            "--revision",
            revision,
            "--local-dir",
            str(root),
        ]
        for pattern in include:
            cmd.extend(["--include", pattern])
        subprocess.run(cmd, check=True)
        return

    raise RuntimeError("Install huggingface_hub or the hf CLI to download Qwen3 artifacts.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.environ.get("QWEN3_ARTIFACT_MANIFEST", "deploy/artifacts/qwen3_manifest.json"))
    parser.add_argument("--set", dest="set_name", default=os.environ.get("QWEN3_ARTIFACT_SET") or "orin-nano-highperf-2026-05-10")
    parser.add_argument("--root", default=os.environ.get("QWEN3_ARTIFACT_ROOT"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    root, paths = required_paths(manifest, args.set_name, args.root)
    missing = verify(paths)
    if not missing:
        print(f"Qwen3 artifact set {args.set_name} OK at {root}")
        return 0

    print(f"Qwen3 artifact set {args.set_name} missing {len(missing)} file(s):")
    for path in missing:
        print(f"  {path}")

    if args.check_only:
        return 2

    snapshot_download(manifest, args.set_name, root)
    missing = verify(paths)
    if missing:
        print("Qwen3 artifact download completed but required files are still missing:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 3
    print(f"Qwen3 artifact set {args.set_name} downloaded to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
