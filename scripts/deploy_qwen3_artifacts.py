#!/usr/bin/env python3
"""Verify or download Qwen3 artifact sets for Jetson Voice.

This script is intentionally independent of the Qwen runtime package. It keeps
Jetson Voice deployable from a JSON profile plus a Hugging Face artifact repo.
"""

from __future__ import annotations

import argparse
import hashlib
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


def warn_root_writable(root: Path) -> None:
    """If `root` cannot be written by the current user, print the canonical
    workaround so the operator does not have to re-run the script with sudo."""
    target = root if root.exists() else root.parent
    try:
        target.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        default_root = "/opt/models/qwen3-edgellm"
        print(
            f"WARN: cannot write to {root}. The default profiles expect "
            f"{default_root}; if you can't sudo, re-run with `--root "
            f"$HOME/qwen3-models` and start jetson-voice with "
            f"`QWEN3_ARTIFACT_ROOT=$HOME/qwen3-models`. "
            f"Profiles in configs/profiles/multilanguage-qwen-*.json now "
            f"resolve every engine path via that variable.",
            file=sys.stderr,
        )


def verify(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


_SHA256_BUF = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_SHA256_BUF)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def default_checksum_path(manifest_path: Path) -> Path:
    return manifest_path.parent / "qwen3_checksums.json"


def load_checksums(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "artifact_sets": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_checksums(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def relative_keys(manifest: dict, set_name: str) -> list[str]:
    return list(manifest["artifact_sets"][set_name].get("required_files", []))


def verify_checksums(
    checksums: dict, set_name: str, root: Path, rel_keys: list[str]
) -> tuple[int, list[tuple[str, str]]]:
    """Return (checked_count, mismatches). mismatches: list of (rel, reason)."""
    set_entry = checksums.get("artifact_sets", {}).get(set_name) or {}
    files = set_entry.get("files") or {}
    if not files:
        return 0, []
    mismatches: list[tuple[str, str]] = []
    checked = 0
    for rel in rel_keys:
        meta = files.get(rel)
        if not meta:
            continue
        path = root / rel
        if not path.exists():
            mismatches.append((rel, "missing"))
            continue
        expected_size = meta.get("size_bytes")
        if expected_size is not None and path.stat().st_size != expected_size:
            mismatches.append((rel, f"size mismatch: got {path.stat().st_size}, want {expected_size}"))
            continue
        expected_sha = meta.get("sha256")
        if expected_sha:
            actual = sha256_file(path)
            if actual != expected_sha:
                mismatches.append((rel, f"sha256 mismatch: got {actual}, want {expected_sha}"))
                continue
        checked += 1
    return checked, mismatches


def generate_sidecar(
    checksums: dict, set_name: str, root: Path, rel_keys: list[str]
) -> dict:
    files: dict[str, dict] = {}
    for rel in rel_keys:
        path = root / rel
        if not path.exists():
            print(f"  skip (missing): {rel}", file=sys.stderr)
            continue
        files[rel] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        print(f"  {rel} -> {files[rel]['sha256'][:12]}... ({files[rel]['size_bytes']} bytes)")
    checksums.setdefault("schema_version", 1)
    checksums.setdefault("artifact_sets", {})
    checksums["artifact_sets"][set_name] = {"files": files}
    return checksums


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
    parser.add_argument(
        "--checksums",
        default=os.environ.get("QWEN3_ARTIFACT_CHECKSUMS"),
        help="Path to checksum sidecar JSON (default: <manifest dir>/qwen3_checksums.json)",
    )
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Verify SHA-256 (and recorded size) of every required file present in the sidecar.",
    )
    parser.add_argument(
        "--generate-sidecar",
        action="store_true",
        help="Walk the artifact root, compute SHA-256 for each required file, and (over)write the sidecar entry for the selected set.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    manifest_path = resolve_path(args.manifest)
    checksum_path = Path(args.checksums) if args.checksums else default_checksum_path(manifest_path)
    root, paths = required_paths(manifest, args.set_name, args.root)
    rel_keys = relative_keys(manifest, args.set_name)
    warn_root_writable(root)

    if args.generate_sidecar:
        missing = verify(paths)
        if missing:
            print(
                f"--generate-sidecar requires all required files present; missing {len(missing)} file(s):",
                file=sys.stderr,
            )
            for path in missing:
                print(f"  {path}", file=sys.stderr)
            return 4
        print(f"Generating SHA-256 sidecar for {args.set_name} at {root}")
        checksums = load_checksums(checksum_path)
        generate_sidecar(checksums, args.set_name, root, rel_keys)
        write_checksums(checksum_path, checksums)
        print(f"Wrote {checksum_path}")
        return 0

    missing = verify(paths)
    if missing:
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
    else:
        print(f"Qwen3 artifact set {args.set_name} OK at {root}")

    if args.verify_sha256:
        checksums = load_checksums(checksum_path)
        checked, mismatches = verify_checksums(checksums, args.set_name, root, rel_keys)
        if mismatches:
            print(
                f"SHA-256 verification failed for {args.set_name}: {len(mismatches)} mismatch(es):",
                file=sys.stderr,
            )
            for rel, reason in mismatches:
                print(f"  {rel}: {reason}", file=sys.stderr)
            return 5
        if checked == 0:
            print(
                f"SHA-256 sidecar has no entries for {args.set_name}; nothing to verify. "
                f"Run --generate-sidecar on a trusted machine to populate {checksum_path}.",
                file=sys.stderr,
            )
        else:
            print(f"SHA-256 verified {checked} file(s) for {args.set_name}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
