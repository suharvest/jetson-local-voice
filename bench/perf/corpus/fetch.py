#!/usr/bin/env python3
"""Populate corpus/ from a source dir or CDN bundle; verify SHA256 against manifest.

Usage:
  # Copy from local dir (~/bench/wavs by default), then verify
  python fetch.py --from ~/bench/wavs

  # Fetch tarball from Seeed CDN (once upload exists)
  python fetch.py --from cdn

  # Only verify what is already there
  python fetch.py --verify

  # Update manifest with current sha256 (after first authoritative fetch)
  python fetch.py --recompute-hashes
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "manifest.json"
MANIFEST = DEFAULT_MANIFEST  # mutable; --manifest overrides in main()
CDN_URL = "https://sensecraft-statics.seeed.cc/solution-app/jetson-voice/models-perf-corpus.tar.gz"
HF_REPO = "harvestsu/seeed-local-voice-perf-corpus"
HF_ENDPOINTS = {
    "default": "https://huggingface.co",
    "mirror":  "https://hf-mirror.com",   # for CN
}


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_from_dir(src_root: Path, files: list[dict]) -> int:
    n = 0
    for entry in files:
        src = src_root / entry["filename"]
        if not src.exists():
            # accept flat layout too: src_root/zh_short_01.wav
            flat = src_root / Path(entry["filename"]).name
            if flat.exists():
                src = flat
            else:
                print(f"  miss: {entry['id']} (looked for {entry['filename']} and {flat.name})")
                continue
        dst = ROOT / entry["filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        n += 1
    return n


def fetch_cdn() -> int:
    import tarfile, tempfile, urllib.request
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        print(f"Downloading {CDN_URL} ...")
        req = urllib.request.Request(CDN_URL, headers={"User-Agent": "openvoicestream/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            shutil.copyfileobj(resp, tmp)
        tmp_path = tmp.name
    print("Extracting ...")
    with tarfile.open(tmp_path, "r:gz") as tar:
        tar.extractall(ROOT)
    os.unlink(tmp_path)
    return 1


def _try_one(url: str, dst: Path, timeout: int = 30) -> bool:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "openvoicestream/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, dst.open("wb") as f:
            shutil.copyfileobj(resp, f)
        return True
    except Exception as e:
        print(f"    fail ({url.split('/')[2]}): {type(e).__name__}: {str(e)[:80]}")
        return False


def fetch_hf(endpoint: str = "auto") -> int:
    """Download every file in manifest from HF dataset repo.

    Endpoint policy:
      - 'auto'    : try huggingface.co first, fall back to hf-mirror.com per file
      - 'default' : huggingface.co only
      - 'mirror'  : hf-mirror.com only
      - <url>     : explicit base (e.g., http://internal-mirror/...)

    Raw `resolve/main/<path>` — no `huggingface_hub` dep required.
    """
    if endpoint in HF_ENDPOINTS:
        order = [HF_ENDPOINTS[endpoint]]
    elif endpoint == "auto":
        order = [HF_ENDPOINTS["default"], HF_ENDPOINTS["mirror"]]
    else:
        order = [endpoint.rstrip("/")]

    m = load_manifest()
    n = 0
    for entry in m["files"]:
        dst = ROOT / entry["filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        ok = False
        for base in order:
            url = f"{base.rstrip('/')}/datasets/{HF_REPO}/resolve/main/{entry['filename']}"
            print(f"  {entry['id']:14s} <- {base.split('//')[1]}")
            if _try_one(url, dst):
                ok = True
                break
        if ok:
            n += 1
    return n


def verify(files: list[dict], strict: bool) -> tuple[int, int]:
    ok, missing, drift = 0, 0, 0
    for entry in files:
        path = ROOT / entry["filename"]
        if not path.exists():
            print(f"  MISS  {entry['id']:14s}  {entry['filename']}")
            missing += 1
            continue
        actual = sha256_of(path)
        expected = entry.get("sha256", "") or ""
        if not expected:
            print(f"  ?     {entry['id']:14s}  sha256={actual[:16]}...  (manifest empty)")
            ok += 1
        elif actual != expected:
            print(f"  DRIFT {entry['id']:14s}  actual={actual[:16]}...  expected={expected[:16]}...")
            drift += 1
        else:
            print(f"  ok    {entry['id']:14s}")
            ok += 1
    print(f"\n{ok} ok / {missing} missing / {drift} drift")
    if strict and (missing or drift):
        sys.exit(1)
    return ok, missing + drift


def recompute(files: list[dict]) -> None:
    m = load_manifest()
    changed = 0
    for entry, m_entry in zip(files, m["files"]):
        path = ROOT / entry["filename"]
        if not path.exists():
            continue
        digest = sha256_of(path)
        if m_entry.get("sha256") != digest:
            m_entry["sha256"] = digest
            changed += 1
    save_manifest(m)
    print(f"Updated {changed} sha256 entries in manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=None,
                    help="dir | 'hf' (auto-fallback) | 'hf-only' | 'hf-mirror' | 'cdn' | omit=verify only")
    ap.add_argument("--verify", action="store_true",
                    help="verify SHA256 against manifest")
    ap.add_argument("--recompute-hashes", action="store_true",
                    help="update manifest sha256 fields from current files")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on any miss/drift (use in CI)")
    ap.add_argument("--manifest", default=None,
                    help="path to manifest JSON (default manifest.json)")
    ap.add_argument("--multilingual", action="store_true",
                    help="shortcut for --manifest multilingual_manifest.json")
    args = ap.parse_args()
    if args.multilingual and not args.manifest:
        args.manifest = "multilingual_manifest.json"

    global MANIFEST
    if args.manifest:
        MANIFEST = Path(args.manifest)
        if not MANIFEST.is_absolute():
            MANIFEST = ROOT / args.manifest

    m = load_manifest()
    files = m["files"]

    if args.src == "cdn":
        fetch_cdn()
    elif args.src in ("hf", "hf-auto"):
        n = fetch_hf("auto")  # huggingface.co -> hf-mirror.com fallback
        print(f"Fetched {n}/{len(files)} files from HuggingFace (auto-fallback)")
    elif args.src == "hf-only":
        n = fetch_hf("default")
        print(f"Fetched {n}/{len(files)} files from huggingface.co (no fallback)")
    elif args.src == "hf-mirror":
        n = fetch_hf("mirror")
        print(f"Fetched {n}/{len(files)} files from hf-mirror.com")
    elif args.src:
        src_root = Path(os.path.expanduser(args.src))
        if not src_root.is_dir():
            sys.exit(f"source not a directory: {src_root}")
        n = copy_from_dir(src_root, files)
        print(f"Copied {n}/{len(files)} files from {src_root}")

    if args.recompute_hashes:
        recompute(files)
        return

    if args.src or args.verify or not any(vars(args).values()):
        print("\nVerifying:")
        verify(files, strict=args.strict)


if __name__ == "__main__":
    main()
