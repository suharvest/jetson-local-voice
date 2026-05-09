#!/usr/bin/env python3
"""Expand a Qwen3-TTS pruned text embedding by a bounded number of rows.

The runtime expects:
  - text_embedding.safetensors containing rows in reduced-vocab order.
  - token_map.bin as uint32 red_id -> original tokenizer id.

This script preserves the existing row order, appends selected new tokenizer
ids, and writes a complete talker directory with the expanded table and map.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


VOCAB = 151936
DIM = 2048

DEFAULT_CORPUS = [
    "你好",
    "你好，今天天气很好。",
    "请打开灯，然后把音量调低一点。",
    "请用中文回答这个问题。",
    "欢迎使用 Jetson voice assistant.",
    "The weather is nice today.",
    "Please turn on the light and lower the volume.",
    "This is a low latency streaming speech test.",
    "The price is twenty three dollars and fifty cents.",
    "今天是2026年5月8日，温度是23.5摄氏度。",
    "请播放音乐、暂停、继续、上一首、下一首。",
    "网络连接失败，请稍后再试。",
    "模型正在加载，请等待。",
    "北京、上海、深圳、广州、杭州、成都、南京、苏州。",
    "Hello, this is Qwen three text to speech.",
    "Mixed language: 请用 English 简单回答。",
    "日本語の短いテストです。",
    "한국어 짧은 테스트입니다.",
    "Français, Deutsch, Español, Italiano, Português.",
    "email support@example.com, phone 13800138000, URL https://example.com",
]


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def load_text_embedding_memmap(path: Path) -> np.memmap:
    header, data_start = read_safetensors_header(path)
    if set(header.keys()) != {"text_embedding"}:
        raise ValueError(f"unexpected safetensors tensors: {list(header.keys())}")
    meta = header["text_embedding"]
    if meta.get("dtype") != "F16" or meta.get("shape") != [VOCAB, DIM]:
        raise ValueError(f"unexpected text_embedding metadata: {meta}")
    begin, end = meta["data_offsets"]
    if begin != 0 or end != VOCAB * DIM * 2:
        raise ValueError(f"unexpected data_offsets: {meta['data_offsets']}")
    return np.memmap(path, dtype=np.float16, mode="r", offset=data_start, shape=(VOCAB, DIM))


def write_text_embedding_safetensors(path: Path, rows: np.ndarray) -> None:
    rows = np.ascontiguousarray(rows, dtype=np.float16)
    header = {
        "text_embedding": {
            "dtype": "F16",
            "shape": [int(rows.shape[0]), int(rows.shape[1])],
            "data_offsets": [0, int(rows.nbytes)],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(rows.tobytes(order="C"))


def corpus_token_ids(tokenizer_path: Path, corpus: list[str]) -> list[int]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    seen: list[int] = []
    added = set()
    for line in corpus:
        ids = tokenizer.encode(line).ids
        for token_id in ids:
            if 0 <= token_id < VOCAB and token_id not in added:
                added.add(token_id)
                seen.append(token_id)
    return seen


def build_expanded_map(
    base_map: np.ndarray,
    tokenizer_path: Path,
    add_rows: int,
    corpus_lines: list[str],
    force_ids: list[int],
) -> np.ndarray:
    keep = set(int(x) for x in base_map)
    additions: list[int] = []

    def add(token_id: int) -> None:
        if 0 <= token_id < VOCAB and token_id not in keep:
            keep.add(token_id)
            additions.append(token_id)

    for token_id in force_ids:
        add(token_id)
    for token_id in corpus_token_ids(tokenizer_path, corpus_lines):
        add(token_id)
    for token_id in range(VOCAB):
        if len(additions) >= add_rows:
            break
        add(token_id)

    if len(additions) != add_rows:
        raise RuntimeError(f"only added {len(additions)} rows, requested {add_rows}")
    return np.concatenate([base_map.astype(np.uint32), np.asarray(additions, dtype=np.uint32)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-talker-dir", required=True)
    parser.add_argument("--full-talker-dir", required=True)
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--add-rows", type=int, default=5000)
    parser.add_argument("--corpus", action="append", default=[])
    parser.add_argument("--corpus-file", action="append", type=Path, default=[])
    parser.add_argument("--force-id", action="append", type=int, default=[])
    args = parser.parse_args()

    base_dir = Path(args.base_talker_dir)
    full_dir = Path(args.full_talker_dir)
    out_dir = Path(args.out_dir)
    file_corpus: list[str] = []
    for corpus_file in args.corpus_file:
        file_corpus.extend(
            line.strip()
            for line in corpus_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    corpus_lines = DEFAULT_CORPUS + args.corpus + file_corpus
    force_ids = list(args.force_id)

    base_map = np.fromfile(base_dir / "token_map.bin", dtype=np.uint32)
    expanded_map = build_expanded_map(
        base_map,
        Path(args.tokenizer_json),
        args.add_rows,
        corpus_lines,
        force_ids,
    )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(base_dir, out_dir)

    full_embedding = load_text_embedding_memmap(full_dir / "text_embedding.safetensors")
    rows = full_embedding[expanded_map.astype(np.int64)]
    write_text_embedding_safetensors(out_dir / "text_embedding.safetensors", rows)
    expanded_map.astype(np.uint32).tofile(out_dir / "token_map.bin")

    added = expanded_map[len(base_map) :]
    print(
        json.dumps(
            {
                "base_rows": int(len(base_map)),
                "added_rows": int(len(added)),
                "total_rows": int(len(expanded_map)),
                "embedding_bytes": int(rows.nbytes),
                "embedding_mib": round(rows.nbytes / 1024 / 1024, 2),
                "forced_present": {str(t): bool(t in set(map(int, expanded_map))) for t in force_ids},
                "first_added": added[:20].astype(int).tolist(),
                "last_added": added[-20:].astype(int).tolist(),
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
