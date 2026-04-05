#!/usr/bin/env python3
"""Export the full 5-layer code_predictor transformer as a single ONNX.
No lm_head, no codebook - just the transformer: [1, 2, 1024] -> [1, 2, 1024]
"""
import torch
import torch.nn as nn
import numpy as np
import os
import onnx


class FullTransformerWrapper(nn.Module):
    """Full 5-layer transformer with baked-in RoPE and final RMSNorm.

    Input:  hidden_states [1, 2, 1024]
    Output: output_hidden [1, 2, 1024]  (after all 5 layers + final norm)
    """

    def __init__(self, model):
        super().__init__()
        # model is cp.model (the Qwen3TTSModel with layers, norm, rotary_emb)
        self.layers = model.layers
        self.norm = model.norm

        # Pre-compute RoPE for positions [0, 1]
        pos_ids = torch.arange(2).unsqueeze(0)
        dummy = torch.randn(1, 2, 1024)
        cos, sin = model.rotary_emb(dummy, pos_ids)
        self.register_buffer("cos", cos)  # [1, 2, 128]
        self.register_buffer("sin", sin)  # [1, 2, 128]

    def forward(self, hidden_states):
        # Run all 5 layers
        h = hidden_states
        for layer in self.layers:
            out = layer(
                h,
                position_embeddings=(self.cos, self.sin),
                use_cache=False,
            )
            h = out[0]

        # Final RMSNorm
        h = self.norm(h)
        return h


def main():
    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration

    print("Loading model...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    cp = model.talker.code_predictor

    outdir = os.path.expanduser("~/qwen3-tts-export/cp_layers")
    os.makedirs(outdir, exist_ok=True)

    wrapper = FullTransformerWrapper(cp.model)
    wrapper.eval()

    dummy_hidden = torch.randn(1, 2, 1024)

    # Verify output
    with torch.no_grad():
        out = wrapper(dummy_hidden)
        print(f"Output shape: {out.shape}, range: [{out.min():.3f}, {out.max():.3f}]")

    onnx_path = os.path.join(outdir, "cp_transformer_full.onnx")
    print(f"Exporting to {onnx_path}...")

    torch.onnx.export(
        wrapper,
        (dummy_hidden,),
        onnx_path,
        input_names=["hidden_states"],
        output_names=["output_hidden"],
        opset_version=14,
        do_constant_folding=True,
    )

    # Check total file size (onnx + data)
    onnx_size = os.path.getsize(onnx_path) / 1e6
    data_path = onnx_path + ".data"
    data_size = os.path.getsize(data_path) / 1e6 if os.path.exists(data_path) else 0
    print(f"ONNX: {onnx_size:.1f} MB, data: {data_size:.1f} MB, total: {onnx_size + data_size:.1f} MB")

    # Verify ops
    onnx_model = onnx.load(onnx_path)
    ops = [n.op_type for n in onnx_model.graph.node]
    op_set = sorted(set(ops))
    print(f"Ops ({len(op_set)}): {op_set}")
    print(f"MatMul: {ops.count('MatMul')}, Softmax: {ops.count('Softmax')}")
    print(f"Total nodes: {len(onnx_model.graph.node)}")

    # Verify ONNX correctness
    print("\nVerifying ONNX correctness...")
    import onnxruntime as ort

    test_input = torch.randn(1, 2, 1024)
    with torch.no_grad():
        ref = wrapper(test_input).numpy()

    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {"hidden_states": test_input.numpy()})[0]

    max_diff = np.abs(ref - onnx_out).max()
    cos_sim = np.dot(ref.flatten(), onnx_out.flatten()) / (
        np.linalg.norm(ref.flatten()) * np.linalg.norm(onnx_out.flatten())
    )
    print(f"max_diff={max_diff:.2e}, cosine={cos_sim:.6f}")

    # Also verify against sequential reference (running through actual model)
    print("\nVerifying against actual code_predictor.model...")
    with torch.no_grad():
        ref2 = cp.model(inputs_embeds=test_input, return_dict=True).last_hidden_state.numpy()

    max_diff2 = np.abs(ref2 - onnx_out).max()
    cos_sim2 = np.dot(ref2.flatten(), onnx_out.flatten()) / (
        np.linalg.norm(ref2.flatten()) * np.linalg.norm(onnx_out.flatten())
    )
    print(f"vs actual model: max_diff={max_diff2:.2e}, cosine={cos_sim2:.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
