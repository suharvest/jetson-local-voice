#!/usr/bin/env python3
"""
Re-export Qwen3-TTS code_predictor without If nodes.
Uses torch.index_select instead of if/else for generation_steps handling.
Based on ElBruno's approach.
"""
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

MODEL_ID = "/tmp/qwen3-full"
OUTPUT_DIR = "/tmp/qwen3-tts-export"


class CodePredictorNoIf(nn.Module):
    """Wrapper that eliminates the If node by using index_select."""

    def __init__(self, code_predictor):
        super().__init__()
        self.model = code_predictor.model  # transformer layers
        self.projection = code_predictor.small_to_mtp_projection

        # Stack all lm_head weights into single tensor [num_heads, vocab, hidden]
        all_weights = torch.stack([head.weight for head in code_predictor.lm_head])
        self.register_buffer("lm_head_weights", all_weights)
        print(f"  lm_head_weights: {all_weights.shape}")  # [16, 2048, 1024]

    def forward(self, inputs_embeds, generation_steps):
        # Project input
        hidden = self.projection(inputs_embeds)

        # Transformer forward (no past KV for simplicity — single-step stateless)
        out = self.model(
            inputs_embeds=hidden,
            past_key_values=None,
            use_cache=False,
        )
        hidden_states = out.last_hidden_state

        # Select lm_head by generation_step (replaces If node)
        # Use [0] indexing instead of squeeze() to avoid ONNX If node
        weight = self.lm_head_weights[generation_steps[0]]  # [vocab, hidden]
        logits = torch.matmul(hidden_states, weight.t())

        return logits


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Export code_predictor without If nodes")
    print("=" * 60)

    # Load model
    print(f"\n[1/3] Loading {MODEL_ID}...")
    t0 = time.time()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from qwen_tts import Qwen3TTSModel
    # KEY: use attn_implementation="eager" to avoid SDPA If nodes in ONNX
    tts_model = Qwen3TTSModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    print(f"  Loaded in {time.time()-t0:.1f}s (attn_implementation=eager)")

    # Find code_predictor
    print("\n[2/3] Wrapping code_predictor...")
    code_pred = None
    for name, module in tts_model.model.named_modules():
        if "code_predictor" in name and hasattr(module, 'lm_head'):
            code_pred = module
            print(f"  Found: {name} ({type(module).__name__})")
            break

    if code_pred is None:
        # List all modules with lm_head
        print("  Searching all modules...")
        for name, module in tts_model.model.named_modules():
            if hasattr(module, 'lm_head'):
                print(f"  Has lm_head: {name} ({type(module).__name__})")
            if "code" in name.lower() or "predict" in name.lower():
                print(f"  Candidate: {name} ({type(module).__name__})")
        sys.exit(1)

    wrapper = CodePredictorNoIf(code_pred)
    wrapper.eval()

    # Export to ONNX
    print("\n[3/3] Exporting ONNX...")
    output_path = os.path.join(OUTPUT_DIR, "code_predictor_no_if.onnx")

    # Dummy inputs (stateless — no KV cache for simplicity)
    B, S, H = 1, 1, 1024
    dummy_embeds = torch.randn(B, S, H)
    dummy_steps = torch.tensor([0], dtype=torch.int64)

    # Dynamic axes
    dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "generation_steps": {0: "num_steps"},
        "logits": {1: "seq_len"},
    }

    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_steps),
            output_path,
            input_names=["inputs_embeds", "generation_steps"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,  # Use legacy TorchScript exporter to avoid Cache issues
        )
    print(f"  Exported in {time.time()-t0:.1f}s")

    sz = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Output: {output_path} ({sz:.1f}MB)")

    # Verify: check for If nodes
    import onnx
    model_onnx = onnx.load(output_path)
    if_nodes = [n for n in model_onnx.graph.node if n.op_type == "If"]
    print(f"  If nodes: {len(if_nodes)} {'✗ STILL HAS IF' if if_nodes else '✓ NO IF NODES'}")

    # Verify with ORT
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    inputs = {i.name: i.shape for i in sess.get_inputs()}
    outputs = {o.name: o.shape for o in sess.get_outputs()}
    print(f"  Inputs: {inputs}")
    print(f"  Outputs: {outputs}")

    # Quick inference test
    test_emb = np.random.randn(1, 1, 1024).astype(np.float32)
    test_steps = np.array([0], dtype=np.int64)
    out = sess.run(None, {
        "inputs_embeds": test_emb,
        "generation_steps": test_steps,
    })
    print(f"  Test output shapes: {[o.shape for o in out]}")
    print("\nDone! Transfer to Jetson:")
    print(f"  scp {output_path} recomputer@100.67.111.58:/tmp/qwen3-tts-bench/model-trt-opt/")


if __name__ == "__main__":
    main()
