#!/usr/bin/env python3
"""ONNX surgery: replace dynamic make_pad_mask subgraphs with external mask inputs.

The Paraformer decoder ONNX has two ``make_pad_mask`` subgraphs that generate padding
masks for ``acoustic_embeds`` and ``enc``. These use Range/ReduceMax/Less ops that
produce dynamic shape expressions TensorRT can't resolve, causing the FSMN Conv layer
to fail Cask consistency checks.

In our runtime usage, CIF produces exact token counts without padding, so these masks
are always all-ones. We externalize them as inputs for TRT compatibility while
preserving the original semantics for cases where padding might occur.

Surgery:
1. Add two float32 inputs: ``pad_mask`` [1, token_length] and ``enc_pad_mask`` [1, enc_length]
2. Remove all 22 nodes in ``/decoder/make_pad_mask/*`` and ``/decoder/make_pad_mask_1/*``
3. Rewire consumers (``/decoder/Unsqueeze`` and ``/decoder/Unsqueeze_1``) to use the new inputs

Usage:
    python3 scripts/surgery_paraformer_decoder.py \\
        --input /tmp/decoder.onnx \\
        --output /tmp/decoder-trt.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto


def find_mask_subgraphs(model: onnx.ModelProto):
    """Find the two make_pad_mask subgraphs and their final outputs."""
    mask_nodes = []
    mask1_nodes = []

    for n in model.graph.node:
        if n.name.startswith("/decoder/make_pad_mask/"):
            mask_nodes.append(n)
        elif n.name.startswith("/decoder/make_pad_mask_1/"):
            mask1_nodes.append(n)

    # Find the final output nodes (Cast_3) and their consumers
    cast3_node = None
    cast3_1_node = None
    for n in mask_nodes:
        if n.name == "/decoder/make_pad_mask/Cast_3":
            cast3_node = n
    for n in mask1_nodes:
        if n.name == "/decoder/make_pad_mask_1/Cast_3":
            cast3_1_node = n

    if cast3_node is None or cast3_1_node is None:
        raise RuntimeError("Could not find Cast_3 nodes in make_pad_mask subgraphs")

    return mask_nodes, mask1_nodes, cast3_node, cast3_1_node


def find_consumer(model: onnx.ModelProto, tensor_name: str):
    """Find first consumer node of a tensor."""
    for n in model.graph.node:
        for idx, inp in enumerate(n.input):
            if inp == tensor_name:
                return n, idx
    return None, -1


def replace_mask_subgraphs(model: onnx.ModelProto):
    """Replace both make_pad_mask subgraphs with external inputs."""

    mask_nodes, mask1_nodes, cast3, cast3_1 = find_mask_subgraphs(model)

    print(f"  make_pad_mask nodes: {len(mask_nodes)}")
    print(f"  make_pad_mask_1 nodes: {len(mask1_nodes)}")

    cast3_out = cast3.output[0]   # /decoder/make_pad_mask/Cast_3_output_0
    cast3_1_out = cast3_1.output[0]  # /decoder/make_pad_mask_1/Cast_3_output_0

    # Find consumers
    unsq_consumer, unsq_idx = find_consumer(model, cast3_out)
    unsq1_consumer, unsq1_idx = find_consumer(model, cast3_1_out)

    if unsq_consumer is None or unsq1_consumer is None:
        raise RuntimeError("Could not find Unsqueeze consumers of mask outputs")

    print(f"  make_pad_mask output -> {unsq_consumer.name} ({unsq_consumer.op_type}) idx={unsq_idx}")
    print(f"  make_pad_mask_1 output -> {unsq1_consumer.name} ({unsq1_consumer.op_type}) idx={unsq1_idx}")

    # The Unsqueeze nodes expand the 1D mask to [1, 1, L] or [1, L, 1] for broadcasting.
    # We can feed the mask directly and skip the Unsqueeze, or keep the Unsqueeze.
    # Let's keep the Unsqueeze nodes (they add singleton dims) and just replace
    # the mask input to them.

    # Add new inputs. Shapes must be 2D [1, L] because downstream Unsqueeze
    # (axes=[2]) requires rank >= 2 input (valid axis range [-3, 2) for rank 2).
    pad_mask_input = helper.make_tensor_value_info(
        "pad_mask", TensorProto.FLOAT, ["batch_size", "token_length"]
    )
    enc_pad_mask_input = helper.make_tensor_value_info(
        "enc_pad_mask", TensorProto.FLOAT, ["batch_size", "enc_length"]
    )
    model.graph.input.append(pad_mask_input)
    model.graph.input.append(enc_pad_mask_input)

    # Rewire Unsqueeze consumers to use new mask inputs
    unsq_consumer.input[unsq_idx] = "pad_mask"
    unsq1_consumer.input[unsq1_idx] = "enc_pad_mask"

    # Remove all mask nodes
    all_mask_nodes = mask_nodes + mask1_nodes
    for n in all_mask_nodes:
        model.graph.node.remove(n)

    print(f"  Added inputs: pad_mask [batch_size, token_length], enc_pad_mask [batch_size, enc_length]")
    print(f"  Removed {len(all_mask_nodes)} mask nodes")

    return model


def validate_parity(original_path: str, modified_path: str):
    """Verify modified ONNX produces identical output to original."""
    import onnxruntime as ort

    sess_orig = ort.InferenceSession(original_path, providers=["CPUExecutionProvider"])
    sess_mod = ort.InferenceSession(modified_path, providers=["CPUExecutionProvider"])

    # Test cases with various shapes
    test_cases = [
        {"enc_frames": 40, "tokens": 5},   # streaming chunk
        {"enc_frames": 80, "tokens": 10},  # larger chunk
        {"enc_frames": 200, "tokens": 20},  # offline
    ]

    for tc in test_cases:
        enc_frames = tc["enc_frames"]
        n_tokens = tc["tokens"]

        # Prepare inputs
        enc = np.random.randn(1, enc_frames, 512).astype(np.float32)
        enc_len = np.array([enc_frames], dtype=np.int32)
        ae = np.random.randn(1, n_tokens, 512).astype(np.float32)
        ae_len = np.array([n_tokens], dtype=np.int32)
        caches = [np.random.randn(1, 512, 10).astype(np.float32) for _ in range(16)]

        # Original inference
        orig_inputs = {
            "enc": enc, "enc_len": enc_len,
            "acoustic_embeds": ae, "acoustic_embeds_len": ae_len,
        }
        for i in range(16):
            orig_inputs[f"in_cache_{i}"] = caches[i]

        orig_out = sess_orig.run(None, orig_inputs)
        orig_logits = orig_out[0]  # logits
        orig_sample_ids = orig_out[1]  # sample_ids

        # Modified inference with externally computed masks
        # In our usage, masks are always all-ones (no padding)
        pad_mask = np.ones((1, n_tokens), dtype=np.float32)
        enc_pad_mask = np.ones((1, enc_frames), dtype=np.float32)

        mod_inputs = {
            "enc": enc, "enc_len": enc_len,
            "acoustic_embeds": ae, "acoustic_embeds_len": ae_len,
            "pad_mask": pad_mask, "enc_pad_mask": enc_pad_mask,
        }
        for i in range(16):
            mod_inputs[f"in_cache_{i}"] = caches[i]

        mod_out = sess_mod.run(None, mod_inputs)
        mod_logits = mod_out[0]
        mod_sample_ids = mod_out[1]

        # Compare
        logits_diff = np.abs(mod_logits - orig_logits).max()
        ids_match = (mod_sample_ids == orig_sample_ids).all()

        print(f"  enc={enc_frames:3d} tokens={n_tokens:2d} | "
              f"logits_max_diff={logits_diff:.2e} | sample_ids_match={ids_match}")

        if logits_diff > 1e-4 or not ids_match:
            print(f"    WARNING: mismatch detected!")
            print(f"    Logits shape: orig={orig_logits.shape} mod={mod_logits.shape}")
            print(f"    Sample IDs shape: orig={orig_sample_ids.shape} mod={mod_sample_ids.shape}")
            if not ids_match:
                mismatch_pct = (mod_sample_ids != orig_sample_ids).mean() * 100
                print(f"    ID mismatch: {mismatch_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Surgery: Paraformer decoder ONNX for TRT")
    parser.add_argument("--input", required=True, help="Path to decoder.onnx")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--validate", action="store_true", help="Run ORT parity validation")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path}")
    model = onnx.load(str(input_path))
    print(f"  Graph: {len(model.graph.node)} nodes, {len(model.graph.input)} inputs")

    model = replace_mask_subgraphs(model)

    print(f"  After surgery: {len(model.graph.node)} nodes, {len(model.graph.input)} inputs")

    # Validate
    try:
        onnx.checker.check_model(model)
        print("  ONNX checker: valid")
    except onnx.checker.ValidationError as e:
        print(f"  ONNX checker FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")

    # Show new I/O
    print("\nModel inputs:")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
    print("Model outputs:")
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")

    if args.validate:
        print("\n=== Parity validation ===")
        validate_parity(str(input_path), str(output_path))
        print("Done.")


if __name__ == "__main__":
    main()
