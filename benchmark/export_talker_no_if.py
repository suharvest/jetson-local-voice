#!/usr/bin/env python3
"""
Export Qwen3-TTS talker (prefill + decode) without If nodes.
Uses attn_implementation="eager" and avoids squeeze.
"""
import os, sys, time, torch, torch.nn as nn, numpy as np

MODEL_ID = "/tmp/qwen3-full"
OUTPUT_DIR = "/tmp/qwen3-tts-export"


class TalkerDecodeWrapper(nn.Module):
    """Talker single-step decode without KV cache (stateless, for TRT benchmark)."""
    def __init__(self, talker):
        super().__init__()
        self.model = talker.model  # transformer layers
        self.codec_head = talker.codec_head  # lm_head for codec tokens

    def forward(self, inputs_embeds, attention_mask):
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
        )
        hidden = out.last_hidden_state
        logits = self.codec_head(hidden)
        return logits, hidden


class TalkerPrefillWrapper(nn.Module):
    """Talker prefill (same as decode but for multiple tokens)."""
    def __init__(self, talker):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head

    def forward(self, inputs_embeds, attention_mask):
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
        )
        hidden = out.last_hidden_state
        logits = self.codec_head(hidden)
        return logits, hidden


def export_model(wrapper, name, dummy_inputs, input_names, output_names, dynamic_axes):
    output_path = os.path.join(OUTPUT_DIR, f"{name}.onnx")
    print(f"\n  Exporting {name}...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    sz = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Exported in {time.time()-t0:.1f}s → {output_path} ({sz:.1f}MB)")

    # Verify no If nodes
    import onnx
    model = onnx.load(output_path)
    if_nodes = [n for n in model.graph.node if n.op_type == "If"]
    print(f"  If nodes: {len(if_nodes)} {'NO IF' if not if_nodes else 'STILL HAS IF!'}")

    # Verify with ORT
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    for i in sess.get_inputs():
        print(f"  IN: {i.name} {i.shape}")
    for o in sess.get_outputs():
        print(f"  OUT: {o.name} {o.shape}")

    return output_path, len(if_nodes)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    print("=" * 60)
    print("Export Talker (prefill + decode) without If nodes")
    print("=" * 60)

    print(f"\n[1/3] Loading {MODEL_ID} with eager attention...")
    t0 = time.time()

    # Monkey-patch masking to avoid vmap (not traceable by TorchScript)
    import transformers.masking_utils as mu
    _orig_create = mu.create_causal_mask

    def simple_causal_mask(*args, **kwargs):
        """Return None — let eager attention handle causal masking internally."""
        return None

    mu.create_causal_mask = simple_causal_mask
    print("  Patched create_causal_mask for TorchScript tracing")

    from qwen_tts import Qwen3TTSModel
    tts_model = Qwen3TTSModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Find talker
    talker = None
    for name, mod in tts_model.model.named_modules():
        if "talker" in name and hasattr(mod, "codec_head") and hasattr(mod, "model"):
            talker = mod
            print(f"  Found talker: {name} ({type(mod).__name__})")
            break

    if talker is None:
        print("  Searching...")
        for name, mod in tts_model.model.named_children():
            print(f"  {name}: {type(mod).__name__}")
            if hasattr(mod, "codec_head"):
                talker = mod
                print(f"  Found talker: {name}")
                break

    if talker is None:
        print("ERROR: talker not found")
        sys.exit(1)

    H = 1024  # hidden size

    # [2/3] Export decode (single step)
    print("\n[2/3] Talker decode (single step)")
    decode_wrapper = TalkerDecodeWrapper(talker)
    decode_wrapper.eval()

    dummy_emb = torch.randn(1, 1, H)
    dummy_mask = torch.ones(1, 1, dtype=torch.int64)

    export_model(
        decode_wrapper, "talker_decode_no_if",
        (dummy_emb, dummy_mask),
        ["inputs_embeds", "attention_mask"],
        ["logits", "hidden_states"],
        {
            "inputs_embeds": {1: "seq_len"},
            "attention_mask": {1: "total_len"},
            "logits": {1: "seq_len"},
            "hidden_states": {1: "seq_len"},
        },
    )

    # [3/3] Export prefill (multiple tokens)
    print("\n[3/3] Talker prefill")
    prefill_wrapper = TalkerPrefillWrapper(talker)
    prefill_wrapper.eval()

    dummy_emb_50 = torch.randn(1, 50, H)
    dummy_mask_50 = torch.ones(1, 50, dtype=torch.int64)

    export_model(
        prefill_wrapper, "talker_prefill_no_if",
        (dummy_emb_50, dummy_mask_50),
        ["inputs_embeds", "attention_mask"],
        ["logits", "hidden_states"],
        {
            "inputs_embeds": {1: "seq_len"},
            "attention_mask": {1: "seq_len"},
            "logits": {1: "seq_len"},
            "hidden_states": {1: "seq_len"},
        },
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
