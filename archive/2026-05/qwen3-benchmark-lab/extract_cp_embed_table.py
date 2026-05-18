#!/usr/bin/env python3
"""Extract cp_embed and codec_embed weights from ONNX to binary files for fast C++ loading.

The cp_embed ONNX model is an embedding lookup:
  inputs: token_id [1,1] int64, layer_idx [1] int64
  output: embedding [1,1,D] float32

Instead of calling ORT 30,720 times (15 layers × 2048 tokens) at startup (~15.5s),
this script extracts the weight initializers directly from the ONNX model and saves
them as a flat fp32 binary that C++ can fread in ~0.1s.

Usage:
    python3 extract_cp_embed_table.py --model-dir /path/to/sherpa/model/dir

Output files (written to --model-dir):
    cp_embed_fp32.bin      — [n_layers * cp_vocab * hidden_dim] float32 LE
    codec_embed_fp32.bin   — [vocab_size * hidden_dim] float32 LE

The binary layout matches the C++ table indexing:
    cp_embed:    table[layer * vocab * D + token * D + d]
    codec_embed: table[token * D + d]
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper


def inspect_onnx(model_path: str) -> None:
    """Print all initializer shapes — useful for debugging the ONNX structure."""
    model = onnx.load(model_path)
    print(f"\n=== Initializers in {model_path} ===")
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        print(f"  {init.name}: shape={arr.shape} dtype={arr.dtype}")
    print(f"\n=== Inputs ===")
    for inp in model.graph.input:
        print(f"  {inp.name}: {inp.type}")
    print(f"\n=== Outputs ===")
    for out in model.graph.output:
        print(f"  {out.name}: {out.type}")


def extract_cp_embed(model_path: str, out_path: str,
                     n_layers: int, cp_vocab: int, hidden_dim: int) -> bool:
    """Extract cp_embed weights from ONNX initializers.

    The model internally has nn.Embedding tables. The ONNX graph layout may be:
      Option A: A single combined table [n_layers * cp_vocab, D] that looks up by
                (layer_idx * cp_vocab + token_id).
      Option B: Per-layer embedding tables [cp_vocab, D] (one per layer/group).

    We find the weight tensor whose total element count equals n_layers*cp_vocab*D
    (option A) or sum of per-layer tables. Falls back to ORT inference if not found.

    Returns True on success.
    """
    model = onnx.load(model_path)

    target_elements = n_layers * cp_vocab * hidden_dim
    print(f"  Looking for cp_embed weight: {n_layers}×{cp_vocab}×{hidden_dim} "
          f"= {target_elements:,} elements ({target_elements * 4 / 1024 / 1024:.1f} MB fp32)")

    # Collect all float initializers sorted by element count descending
    candidates = []
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        if arr.dtype in (np.float32, np.float16) or str(arr.dtype) == "bfloat16":
            candidates.append((init.name, arr))

    # --- Option A: single combined table [n_layers * cp_vocab, D] ---
    for name, arr in candidates:
        flat = arr.reshape(-1)
        if flat.size == target_elements:
            print(f"  Found combined embedding table: '{name}' shape={arr.shape}")
            table = flat.reshape(n_layers, cp_vocab, hidden_dim).astype(np.float32)
            table.tofile(out_path)
            size_mb = table.nbytes / 1024 / 1024
            print(f"  Saved {out_path} ({size_mb:.1f} MB)")
            return True

    # --- Option B: per-layer tables [cp_vocab, D] each ---
    per_layer = [(name, arr) for name, arr in candidates
                 if arr.reshape(-1).size == cp_vocab * hidden_dim]
    if len(per_layer) >= n_layers:
        print(f"  Found {len(per_layer)} per-layer tables of shape [{cp_vocab}, {hidden_dim}]")
        # Take the first n_layers in order of appearance
        table = np.stack([arr.reshape(cp_vocab, hidden_dim).astype(np.float32)
                          for _, arr in per_layer[:n_layers]], axis=0)
        assert table.shape == (n_layers, cp_vocab, hidden_dim)
        table.tofile(out_path)
        size_mb = table.nbytes / 1024 / 1024
        print(f"  Saved {out_path} ({size_mb:.1f} MB)")
        return True

    # --- Fallback: infer via ORT (slow but correct) ---
    print(f"  WARNING: No matching initializer found in ONNX. Falling back to ORT inference.")
    print(f"  Available initializers:")
    for name, arr in candidates:
        print(f"    {name}: {arr.shape} ({arr.reshape(-1).size:,} elements)")
    print(f"  Running ORT inference ({n_layers}×{cp_vocab} = {n_layers * cp_vocab:,} calls)...")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path,
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except Exception:
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    in_names = [inp.name for inp in sess.get_inputs()]
    out_name = sess.get_outputs()[0].name

    table = np.zeros((n_layers, cp_vocab, hidden_dim), dtype=np.float32)
    for layer in range(n_layers):
        lid = np.array([layer], dtype=np.int64)
        for tid in range(cp_vocab):
            token = np.array([[tid]], dtype=np.int64)
            feeds = {in_names[0]: token, in_names[1]: lid}
            result = sess.run([out_name], feeds)[0]
            table[layer, tid] = result.reshape(hidden_dim)
        print(f"    Layer {layer}/{n_layers} done")

    table.tofile(out_path)
    size_mb = table.nbytes / 1024 / 1024
    print(f"  Saved {out_path} ({size_mb:.1f} MB) [via ORT inference]")
    return True


def extract_codec_embed(model_path: str, out_path: str,
                        vocab_size: int, hidden_dim: int) -> bool:
    """Extract codec_embed weights from ONNX initializers.

    The model is a simple embedding: token_id [1,T] → [1,T,D].
    We look for a single weight table [vocab_size, D].

    Returns True on success.
    """
    model = onnx.load(model_path)

    target_elements = vocab_size * hidden_dim
    print(f"  Looking for codec_embed weight: {vocab_size}×{hidden_dim} "
          f"= {target_elements:,} elements ({target_elements * 4 / 1024 / 1024:.1f} MB fp32)")

    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        if arr.dtype not in (np.float32, np.float16) and str(arr.dtype) != "bfloat16":
            continue
        flat = arr.reshape(-1)
        if flat.size == target_elements:
            print(f"  Found embedding table: '{init.name}' shape={arr.shape}")
            table = flat.reshape(vocab_size, hidden_dim).astype(np.float32)
            table.tofile(out_path)
            size_mb = table.nbytes / 1024 / 1024
            print(f"  Saved {out_path} ({size_mb:.1f} MB)")
            return True

    # Fallback: single batch ORT inference (fast — one call for all vocab)
    print(f"  No matching initializer found. Falling back to ORT single-batch inference...")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path,
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except Exception:
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    all_ids = np.arange(vocab_size, dtype=np.int64).reshape(1, vocab_size)
    result = sess.run([out_name], {in_name: all_ids})[0]  # [1, vocab, D]
    table = result.reshape(vocab_size, hidden_dim).astype(np.float32)
    table.tofile(out_path)
    size_mb = table.nbytes / 1024 / 1024
    print(f"  Saved {out_path} ({size_mb:.1f} MB) [via ORT single-batch]")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract cp_embed/codec_embed weights from ONNX to fp32 binary files.")
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing ONNX models and config.json")
    parser.add_argument("--inspect", action="store_true",
                        help="Print ONNX initializer shapes and exit")
    parser.add_argument("--n-layers", type=int, default=15,
                        help="Number of cp_embed layers (num_code_groups - 1, default=15)")
    parser.add_argument("--cp-vocab", type=int, default=2048,
                        help="CP vocabulary size (default=2048)")
    parser.add_argument("--vocab-size", type=int, default=3072,
                        help="Codec vocab size (default=3072)")
    parser.add_argument("--hidden-dim", type=int, default=1024,
                        help="Hidden dimension D (default=1024)")
    parser.add_argument("--skip-cp", action="store_true",
                        help="Skip cp_embed extraction")
    parser.add_argument("--skip-codec", action="store_true",
                        help="Skip codec_embed extraction")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"ERROR: model-dir does not exist: {model_dir}", file=sys.stderr)
        sys.exit(1)

    cp_embed_onnx = model_dir / "code_predictor_embed.onnx"
    codec_embed_onnx = model_dir / "codec_embed.onnx"

    if args.inspect:
        if cp_embed_onnx.exists():
            inspect_onnx(str(cp_embed_onnx))
        if codec_embed_onnx.exists():
            inspect_onnx(str(codec_embed_onnx))
        return

    ok = True

    # --- Extract cp_embed ---
    if not args.skip_cp:
        if not cp_embed_onnx.exists():
            print(f"WARNING: {cp_embed_onnx} not found, skipping cp_embed extraction")
        else:
            out_path = str(model_dir / "cp_embed_fp32.bin")
            print(f"\n[1/2] Extracting cp_embed from {cp_embed_onnx}")
            ok &= extract_cp_embed(
                str(cp_embed_onnx), out_path,
                args.n_layers, args.cp_vocab, args.hidden_dim)

    # --- Extract codec_embed ---
    if not args.skip_codec:
        if not codec_embed_onnx.exists():
            print(f"WARNING: {codec_embed_onnx} not found, skipping codec_embed extraction")
        else:
            out_path = str(model_dir / "codec_embed_fp32.bin")
            print(f"\n[2/2] Extracting codec_embed from {codec_embed_onnx}")
            ok &= extract_codec_embed(
                str(codec_embed_onnx), out_path,
                args.vocab_size, args.hidden_dim)

    if ok:
        print("\nDone. Binary files are ready.")
        print(f"  cp_embed_fp32.bin:    {model_dir}/cp_embed_fp32.bin")
        print(f"  codec_embed_fp32.bin: {model_dir}/codec_embed_fp32.bin")
        print("\nStartup will now skip ORT inference and load binary files directly.")
        print(f"  cp_embed:    ~15.5s → <0.1s")
        print(f"  codec_embed: ~0.016s → <0.01s")
    else:
        print("\nERROR: One or more extractions failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
