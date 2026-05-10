#!/usr/bin/env python3
"""Quantize an EdgeLLM embedding safetensors file to FP8 E4M3.

The EdgeLLM runtime expects:
  - embedding: FP8 E4M3, shape [vocab, hidden]
  - embedding_scale: FP32, shape [vocab, hidden / 128]

For Qwen3-TTS text embeddings, pass:
  --tensor-name text_embedding --scale-name text_embedding_scale
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


FP8_E4M3_MAX = 448.0
BLOCK_SIZE = 128


def quantize_embedding(embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if embedding.ndim != 2:
        raise ValueError(f"embedding must be 2D, got {embedding.ndim}D")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required")

    vocab, hidden = embedding.shape
    if hidden % BLOCK_SIZE != 0:
        raise ValueError(f"hidden size {hidden} is not divisible by {BLOCK_SIZE}")

    groups = hidden // BLOCK_SIZE
    fp32 = embedding.float().view(vocab, groups, BLOCK_SIZE)
    scales = fp32.abs().amax(dim=-1).clamp(min=1e-4) / FP8_E4M3_MAX
    quantized = (fp32 / scales.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return quantized.reshape(vocab, hidden).to(torch.float8_e4m3fn), scales.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--tensor-name",
        default="embedding",
        help="Input tensor name to quantize and output tensor name to write",
    )
    parser.add_argument(
        "--scale-name",
        default="embedding_scale",
        help="Output tensor name for per-row FP8 scales",
    )
    args = parser.parse_args()

    tensors = load_file(args.input)
    if args.tensor_name not in tensors:
        raise KeyError(f"{args.input} does not contain a '{args.tensor_name}' tensor")

    embedding_fp8, scales = quantize_embedding(tensors[args.tensor_name])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({args.tensor_name: embedding_fp8, args.scale_name: scales}, args.output)

    input_mb = args.input.stat().st_size / 1024 / 1024
    output_mb = args.output.stat().st_size / 1024 / 1024
    print(
        f"wrote {args.output} shape={tuple(embedding_fp8.shape)} "
        f"input_mb={input_mb:.1f} output_mb={output_mb:.1f}"
    )


if __name__ == "__main__":
    main()
