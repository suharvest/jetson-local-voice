#!/usr/bin/env python3
"""Export talker_decode + code_predictor WITH KV cache and WITHOUT If nodes."""
import os, sys, time, torch, torch.nn as nn, numpy as np

MODEL_ID = "/tmp/qwen3-full"
OUTPUT_DIR = "/tmp/qwen3-tts-export"
H = 1024

# ---- Code Predictor with KV cache, no If ----
class CPNoIf(nn.Module):
    def __init__(self, cp):
        super().__init__()
        self.model = cp.model
        self.projection = cp.small_to_mtp_projection
        all_w = torch.stack([h.weight for h in cp.lm_head])
        self.register_buffer("lm_head_weights", all_w)

    def forward(self, inputs_embeds, generation_steps, past_keys, past_values):
        hidden = self.projection(inputs_embeds)
        # Skip internal model KV cache — run without cache, rely on full input
        # For TRT: stateless per-step, caller manages KV externally
        out = self.model(inputs_embeds=hidden, past_key_values=None, use_cache=False)
        hs = out.last_hidden_state
        weight = self.lm_head_weights[generation_steps[0]]
        logits = torch.matmul(hs, weight.t())
        return logits

# ---- Talker Decode with KV cache, no If ----
class TalkerDecNoIf(nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head

    def forward(self, inputs_embeds, attention_mask, position_ids):
        # Stateless — no KV cache for TRT compatibility
        # Caller accumulates input sequence instead
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        hs = out.last_hidden_state
        logits = self.codec_head(hs)
        return logits, hs

def export_onnx(model, name, args, input_names, output_names, dynamic_axes):
    path = os.path.join(OUTPUT_DIR, f"{name}.onnx")
    print(f"\n  Exporting {name}...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(model, args, path,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=17,
            do_constant_folding=True, dynamo=False)
    sz = os.path.getsize(path) / 1024 / 1024
    print(f"  Done in {time.time()-t0:.1f}s → {path} ({sz:.1f}MB)")
    # Check If nodes
    import onnx
    m = onnx.load(path)
    ifs = [n for n in m.graph.node if n.op_type == "If"]
    print(f"  If nodes: {len(ifs)} {'✓' if not ifs else '✗'}")
    return path

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Monkey-patch causal mask
    import transformers.masking_utils as mu
    mu.create_causal_mask = lambda *a, **kw: None

    print("Loading model with eager attention...")
    from qwen_tts import Qwen3TTSModel
    mdl = Qwen3TTSModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    talker = mdl.model.talker
    cp = talker.code_predictor

    # ---- Export code_predictor with KV ----
    print("\n=== Code Predictor (with KV cache) ===")
    cp_wrap = CPNoIf(cp).eval()
    B, S = 1, 1
    cp_args = (
        torch.randn(B, 2, H),           # inputs_embeds [1, 2, 1024] (first call)
        torch.tensor([0], dtype=torch.int64),
        torch.zeros(5, B, 8, 0, 128),   # past_keys (empty)
        torch.zeros(5, B, 8, 0, 128),   # past_values
    )
    export_onnx(cp_wrap, "cp_kv_no_if", cp_args,
        ["inputs_embeds", "generation_steps", "past_keys", "past_values"],
        ["logits", "present_keys", "present_values"],
        {"inputs_embeds": {1: "seq"}, "generation_steps": {0: "ns"},
         "past_keys": {3: "past_len"}, "past_values": {3: "past_len"},
         "logits": {1: "seq"},
         "present_keys": {3: "total_len"}, "present_values": {3: "total_len"}})

    # ---- Export talker_decode with KV ----
    print("\n=== Talker Decode (with KV cache) ===")
    td_wrap = TalkerDecNoIf(talker).eval()
    past_len = 13
    td_args = (
        torch.randn(B, 1, H),
        torch.ones(B, past_len + 1, dtype=torch.int64),
        torch.tensor([[[past_len]]] * 3, dtype=torch.int64),
        torch.randn(28, B, 8, past_len, 128),
        torch.randn(28, B, 8, past_len, 128),
    )
    export_onnx(td_wrap, "talker_dec_kv_no_if", td_args,
        ["inputs_embeds", "attention_mask", "position_ids", "past_keys", "past_values"],
        ["logits", "hidden_states", "present_keys", "present_values"],
        {"attention_mask": {1: "total_len"},
         "past_keys": {3: "past_len"}, "past_values": {3: "past_len"},
         "present_keys": {3: "total_len"}, "present_values": {3: "total_len"}})

    print("\nDone! Models at:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
