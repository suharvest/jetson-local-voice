import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import onnx


class SingleLayerWrapper(nn.Module):
    """Wraps one transformer layer with baked-in RoPE (fixed positions [0,1]).

    Input:  hidden_states [1, 2, 1024]
    Output: output_hidden [1, 2, 1024]

    No KV cache needed - each code_predictor step processes 2 fresh tokens.
    """

    def __init__(self, layer, cos, sin):
        super().__init__()
        self.layer = layer
        # Bake RoPE cos/sin as buffers (fixed positions [0,1])
        self.register_buffer("cos", cos)  # [1, 2, 128]
        self.register_buffer("sin", sin)  # [1, 2, 128]

    def forward(self, hidden_states):
        # hidden_states: [1, 2, 1024]
        out = self.layer(
            hidden_states,
            position_embeddings=(self.cos, self.sin),
            use_cache=False,
        )
        return out[0]  # [1, 2, 1024]


def main():
    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration

    print("Loading model...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    cp = model.talker.code_predictor

    # Pre-compute RoPE for positions [0, 1]
    pos_ids = torch.arange(2).unsqueeze(0)
    dummy = torch.randn(1, 2, 1024)
    cos, sin = cp.model.rotary_emb(dummy, pos_ids)
    print(f"RoPE cos: {cos.shape}, sin: {sin.shape}")

    outdir = os.path.expanduser("~/qwen3-tts-export/cp_layers")
    os.makedirs(outdir, exist_ok=True)

    dummy_hidden = torch.randn(1, 2, 1024)

    for i in range(5):
        print(f"\n=== Exporting layer {i} ===")
        layer = cp.model.layers[i]
        wrapper = SingleLayerWrapper(layer, cos, sin)
        wrapper.eval()

        # Verify output
        with torch.no_grad():
            out = wrapper(dummy_hidden)
            print(f"  Output shape: {out.shape}, range: [{out.min():.3f}, {out.max():.3f}]")

        onnx_path = os.path.join(outdir, f"cp_layer{i}.onnx")

        torch.onnx.export(
            wrapper,
            (dummy_hidden,),
            onnx_path,
            input_names=["hidden_states"],
            output_names=["output_hidden"],
            opset_version=14,
            do_constant_folding=True,
        )

        size_mb = os.path.getsize(onnx_path) / 1e6
        print(f"  Saved: {onnx_path} ({size_mb:.1f} MB)")

        # Verify ONNX ops
        onnx_model = onnx.load(onnx_path)
        ops = [n.op_type for n in onnx_model.graph.node]
        op_set = sorted(set(ops))
        print(f"  Ops ({len(op_set)}): {op_set}")
        print(f"  MatMul: {ops.count('MatMul')}, Softmax: {ops.count('Softmax')}")
        print(f"  Total nodes: {len(onnx_model.graph.node)}")

        # Check for problematic ops
        problem_ops = {"Gather", "ScatterND", "Where", "NonZero", "Loop", "If", "Scan"}
        found = set(op_set) & problem_ops
        if found:
            print(f"  WARNING: NPU-incompatible ops: {found}")
        else:
            print("  OK: No known problematic ops")

    # Verify ONNX outputs match PyTorch
    print("\n=== Verifying ONNX correctness ===")
    import onnxruntime as ort

    test_input = torch.randn(1, 2, 1024)
    for i in range(5):
        wrapper = SingleLayerWrapper(cp.model.layers[i], cos, sin)
        wrapper.eval()

        with torch.no_grad():
            ref = wrapper(test_input).numpy()

        sess = ort.InferenceSession(os.path.join(outdir, f"cp_layer{i}.onnx"))
        onnx_out = sess.run(None, {"hidden_states": test_input.numpy()})[0]

        max_diff = np.abs(ref - onnx_out).max()
        cos_sim = np.dot(ref.flatten(), onnx_out.flatten()) / (
            np.linalg.norm(ref.flatten()) * np.linalg.norm(onnx_out.flatten())
        )
        print(f"  Layer {i}: max_diff={max_diff:.2e}, cosine={cos_sim:.6f}")

    # Also export lm_heads and codebook embeddings for CPU inference
    print("\n=== Exporting lm_heads and codebooks ===")
    for j in range(15):
        lm_w = cp.lm_head[j].weight.data.numpy()  # [2048, 1024]
        cb_w = cp.model.codec_embedding[j].weight.data.numpy()  # [2048, 1024]
        np.save(os.path.join(outdir, f"lm_head_{j}.npy"), lm_w)
        np.save(os.path.join(outdir, f"codebook_{j}.npy"), cb_w)
    print(f"  Saved 15 lm_heads and 15 codebooks to {outdir}")

    print("\nDone! Files in:", outdir)


if __name__ == "__main__":
    main()
