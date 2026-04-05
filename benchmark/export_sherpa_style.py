#!/usr/bin/env python3
"""Export Qwen3-TTS 0.6B to ONNX — sherpa-onnx style (10 sub-models).

Adapted from HeiSir2014/sherpa-onnx for our model version's attribute names.

Usage:
    python3 export_sherpa_style.py --model /tmp/qwen3-full --output-dir /tmp/qwen3-sherpa --device cuda:0
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoProcessor

# Monkey-patch create_causal_mask: replace vmap-based version with simple triangular mask
import transformers.masking_utils

def _simple_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, **kwargs):
    """Simple causal mask without vmap — TorchScript-traceable."""
    dtype = input_embeds.dtype
    device = input_embeds.device
    batch, seq_len = input_embeds.shape[:2]

    if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
        past_len = past_key_values.get_seq_length()
    else:
        past_len = 0

    total_len = past_len + seq_len
    # For decode (seq_len=1): no masking needed
    if seq_len == 1:
        return None
    # For prefill: lower-triangular causal mask
    mask = torch.triu(torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype), diagonal=past_len + 1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)

transformers.masking_utils.create_causal_mask = _simple_causal_mask

from qwen_tts.core.models import (
    Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor,
)

# ---------------------------------------------------------------------------
# 1. text_project: token IDs -> projected text embeddings
# ---------------------------------------------------------------------------
def export_text_project(model, output_dir, opset=14):
    print("Exporting text_project.onnx ...")

    class TextProject(nn.Module):
        def __init__(self, talker):
            super().__init__()
            self.text_embed = talker.model.text_embedding
            self.text_projection = talker.text_projection

        def forward(self, input_ids):
            return self.text_projection(self.text_embed(input_ids))

    w = TextProject(model.talker).eval()
    dummy = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=model.device)
    torch.onnx.export(w, (dummy,), f"{output_dir}/text_project.onnx",
                       input_names=["input_ids"], output_names=["text_embed"],
                       dynamic_axes={"input_ids": {1: "T"}, "text_embed": {1: "T"}},
                       opset_version=opset, dynamo=False)
    print("  Done")


# ---------------------------------------------------------------------------
# 2. codec_embed: codec token IDs -> embeddings
# ---------------------------------------------------------------------------
def export_codec_embed(model, output_dir, opset=14):
    print("Exporting codec_embed.onnx ...")

    class CodecEmbed(nn.Module):
        def __init__(self, talker):
            super().__init__()
            self.embed = talker.model.codec_embedding

        def forward(self, token_ids):
            return self.embed(token_ids)

    w = CodecEmbed(model.talker).eval()
    dummy = torch.tensor([[100]], dtype=torch.long, device=model.device)
    torch.onnx.export(w, (dummy,), f"{output_dir}/codec_embed.onnx",
                       input_names=["token_ids"], output_names=["embed"],
                       dynamic_axes={"token_ids": {1: "T"}, "embed": {1: "T"}},
                       opset_version=opset, dynamo=False)
    print("  Done")


# ---------------------------------------------------------------------------
# 3. code_predictor_embed: residual code ID + layer_idx -> embedding
# ---------------------------------------------------------------------------
def export_code_predictor_embed(model, output_dir, opset=14):
    print("Exporting code_predictor_embed.onnx ...")
    cp = model.talker.code_predictor
    embed_layers = list(cp.get_input_embeddings())
    num_groups = len(embed_layers)

    class CPEmbed(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = nn.ModuleList(layers)

        def forward(self, token_id, layer_idx):
            stacked = torch.stack([l(token_id) for l in self.layers], dim=0)
            return stacked[layer_idx]

    w = CPEmbed(embed_layers).eval()
    dummy_tok = torch.tensor([[100]], dtype=torch.long, device=model.device)
    dummy_idx = torch.tensor(0, dtype=torch.long, device=model.device)
    torch.onnx.export(w, (dummy_tok, dummy_idx), f"{output_dir}/code_predictor_embed.onnx",
                       input_names=["token_id", "layer_idx"], output_names=["embed"],
                       opset_version=opset, dynamo=False)
    print(f"  Done ({num_groups} layers)")


# ---------------------------------------------------------------------------
# 4. code_predictor: context + gen_step -> logits (stateless, no KV cache)
# ---------------------------------------------------------------------------
def export_code_predictor(model, output_dir, opset=14):
    print("Exporting code_predictor.onnx ...")
    cp = model.talker.code_predictor

    class CP(nn.Module):
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor
            # Stack all 15 lm_head weights into a single tensor for dynamic indexing
            # Each lm_head[i] is Linear(1024, 2048, bias=False)
            self.stacked_weights = nn.Parameter(
                torch.stack([h.weight for h in predictor.lm_head]),  # [15, 2048, 1024]
                requires_grad=False,
            )

        def forward(self, context, gen_step):
            # Run transformer backbone (no lm_head selection inside)
            out = self.predictor.model(
                inputs_embeds=context, use_cache=False, return_dict=True,
            )
            hidden = out.last_hidden_state[:, -1:, :]  # [1, 1, 1024]
            # Dynamic lm_head selection via stacked weights
            weight = self.stacked_weights[gen_step[0]]  # [2048, 1024]
            logits = torch.matmul(hidden, weight.t())  # [1, 1, 2048]
            return logits

    w = CP(cp).eval()
    D = model.talker.config.hidden_size
    dummy_ctx = torch.randn(1, 2, D, device=model.device)
    dummy_step = torch.tensor([0], dtype=torch.long, device=model.device)
    torch.onnx.export(w, (dummy_ctx, dummy_step), f"{output_dir}/code_predictor.onnx",
                       input_names=["context", "gen_step"], output_names=["logits"],
                       dynamic_axes={"context": {1: "ctx_len"}},
                       opset_version=opset, dynamo=False)
    print("  Done")


# ---------------------------------------------------------------------------
# 5. talker_prefill: embeddings -> logits + hidden + KV cache
# ---------------------------------------------------------------------------
def export_talker_prefill(model, output_dir, opset=14):
    print("Exporting talker_prefill.onnx ...")
    talker = model.talker
    N = talker.config.num_hidden_layers
    D = talker.config.hidden_size
    H = talker.config.num_key_value_heads
    dh = getattr(talker.config, 'head_dim', D // talker.config.num_attention_heads)

    class Prefill(nn.Module):
        def __init__(self, talker):
            super().__init__()
            self.talker = talker

        def forward(self, inputs_embeds, attention_mask):
            out = self.talker.model(inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    use_cache=True, return_dict=True)
            hidden = out.last_hidden_state
            logits = self.talker.codec_head(hidden[:, -1:, :])
            pkv = out.past_key_values.to_legacy_cache()
            return (logits, hidden) + tuple(t for kv in pkv for t in kv)

    w = Prefill(talker).eval()
    T = 10
    dummy_e = torch.randn(1, T, D, device=model.device)
    dummy_m = torch.ones(1, T, dtype=torch.long, device=model.device)

    kv_names = []
    for i in range(N):
        kv_names += [f"past_key_{i}", f"past_value_{i}"]

    torch.onnx.export(w, (dummy_e, dummy_m), f"{output_dir}/talker_prefill.onnx",
                       input_names=["inputs_embeds", "attention_mask"],
                       output_names=["logits", "last_hidden"] + kv_names,
                       dynamic_axes={
                           "inputs_embeds": {1: "T"}, "attention_mask": {1: "T"},
                           "last_hidden": {1: "T"},
                           **{n: {2: "T"} for n in kv_names},
                       },
                       opset_version=opset, dynamo=False)
    print(f"  Done ({N} layers, {len(kv_names)} KV tensors)")


# ---------------------------------------------------------------------------
# 6. talker_decode: single token + KV cache -> logits + hidden + updated KV
# ---------------------------------------------------------------------------
def export_talker_decode(model, output_dir, opset=14):
    print("Exporting talker_decode.onnx ...")
    talker = model.talker
    N = talker.config.num_hidden_layers
    D = talker.config.hidden_size
    H = talker.config.num_key_value_heads
    dh = getattr(talker.config, 'head_dim', D // talker.config.num_attention_heads)

    class Decode(nn.Module):
        def __init__(self, talker, num_layers):
            super().__init__()
            self.talker = talker
            self.num_layers = num_layers

        def forward(self, inputs_embeds, attention_mask, *past_kv_flat):
            from transformers.cache_utils import DynamicCache
            cache = DynamicCache()
            for i in range(self.num_layers):
                cache.update(past_kv_flat[2*i], past_kv_flat[2*i+1], i)
            out = self.talker.model(inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    past_key_values=cache,
                                    use_cache=True, return_dict=True)
            hidden = out.last_hidden_state
            logits = self.talker.codec_head(hidden)
            new_pkv = out.past_key_values.to_legacy_cache()
            return (logits, hidden) + tuple(t for kv in new_pkv for t in kv)

    w = Decode(talker, N).eval()
    T_past = 10
    dummy_e = torch.randn(1, 1, D, device=model.device)
    dummy_m = torch.ones(1, T_past + 1, dtype=torch.long, device=model.device)
    dummy_kv = [torch.randn(1, H, T_past, dh, device=model.device) for _ in range(N * 2)]

    in_kv = []; out_kv = []
    for i in range(N):
        in_kv += [f"past_key_{i}", f"past_value_{i}"]
        out_kv += [f"new_past_key_{i}", f"new_past_value_{i}"]

    torch.onnx.export(w, (dummy_e, dummy_m, *dummy_kv),
                       f"{output_dir}/talker_decode.onnx",
                       input_names=["inputs_embeds", "attention_mask"] + in_kv,
                       output_names=["logits", "last_hidden"] + out_kv,
                       dynamic_axes={
                           "inputs_embeds": {1: "seq_len"},
                           "attention_mask": {1: "full_len"},
                           "last_hidden": {1: "seq_len"},
                           **{n: {2: "past_len"} for n in in_kv},
                           **{n: {2: "new_len"} for n in out_kv},
                       },
                       opset_version=opset, dynamo=False)
    print(f"  Done ({N} layers)")


# ---------------------------------------------------------------------------
# 7. vocoder (tokenizer12hz_decode): codec codes -> audio
# ---------------------------------------------------------------------------
def export_vocoder(model, output_dir, opset=18):
    print("Exporting tokenizer12hz_decode.onnx ...")
    import io, onnx

    processor = AutoProcessor.from_pretrained(model.config._name_or_path, fix_mistral_regex=True)
    speech_model = processor.speech_tokenizer.model
    upsample_rate = speech_model.decode_upsample_rate

    # Register aten::diff symbolic
    def _diff_symbolic(g, x, n, dim, prepend, append):
        from torch.onnx.symbolic_helper import _get_const
        dim_val = _get_const(dim, "i", "dim")
        axes = g.op("Constant", value_t=torch.tensor([dim_val], dtype=torch.long))
        zero = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
        one = g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
        big = g.op("Constant", value_t=torch.tensor([9223372036854775807], dtype=torch.long))
        a = g.op("Slice", x, zero, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)), axes, one)
        b = g.op("Slice", x, one, big, axes, one)
        diff_result = g.op("Sub", b, a)
        first = g.op("Slice", x, zero, one, axes, one)
        zero_pad = g.op("Sub", first, first)
        return g.op("Concat", zero_pad, diff_result, axis_i=dim_val)
    torch.onnx.register_custom_op_symbolic("aten::diff", _diff_symbolic, 18)

    class DecoderFwd(nn.Module):
        def __init__(self, decoder, rate):
            super().__init__()
            self.decoder = decoder
            self.rate = rate

        def forward(self, audio_codes):
            wav = self.decoder(audio_codes.transpose(1, 2)).squeeze(1)
            lengths = (audio_codes[..., 0] >= 0).sum(dim=1) * self.rate
            return wav, lengths

    w = DecoderFwd(speech_model.decoder, upsample_rate).eval()
    dummy = torch.randint(0, 1024, (1, 100, 16), device=model.device)

    buf = io.BytesIO()
    torch.onnx.export(w, (dummy,), buf,
                       input_names=["audio_codes"], output_names=["audio_values", "lengths"],
                       dynamic_axes={"audio_codes": {1: "T"}, "audio_values": {1: "S"}},
                       opset_version=opset, do_constant_folding=False, dynamo=False)
    buf.seek(0)

    # Fix bool->CumSum type mismatch
    onnx_model = onnx.load_model_from_string(buf.getvalue())
    name_to_node = {o: node for node in onnx_model.graph.node for o in node.output}
    cast_added = 0
    for i, node in enumerate(list(onnx_model.graph.node)):
        if node.op_type == "CumSum":
            src = name_to_node.get(node.input[0])
            if src and src.op_type in ("Not", "Equal", "Less", "Greater", "And", "Or"):
                cast_name = node.input[0] + "_i64"
                cast_node = onnx.helper.make_node("Cast", inputs=[node.input[0]], outputs=[cast_name], to=7)
                node.input[0] = cast_name
                onnx_model.graph.node.insert(i, cast_node)
                cast_added += 1
    onnx.save(onnx_model, f"{output_dir}/tokenizer12hz_decode.onnx")
    print(f"  Done (inserted {cast_added} Cast nodes)")


# ---------------------------------------------------------------------------
# Save config
# ---------------------------------------------------------------------------
def save_config(model, output_dir):
    tc = model.talker.config
    cfg = {
        "hidden_size": tc.hidden_size,
        "num_hidden_layers": tc.num_hidden_layers,
        "num_attention_heads": tc.num_attention_heads,
        "num_key_value_heads": tc.num_key_value_heads,
        "num_code_groups": tc.num_code_groups,
        "vocab_size": tc.vocab_size,
        "tts_bos_token_id": model.config.tts_bos_token_id,
        "tts_eos_token_id": model.config.tts_eos_token_id,
        "tts_pad_token_id": model.config.tts_pad_token_id,
        "codec_bos_id": tc.codec_bos_id,
        "codec_eos_token_id": tc.codec_eos_token_id,
        "codec_pad_id": tc.codec_pad_id,
        "codec_think_id": tc.codec_think_id,
        "codec_think_bos_id": tc.codec_think_bos_id,
        "codec_think_eos_id": tc.codec_think_eos_id,
        "codec_nothink_id": tc.codec_nothink_id,
        "codec_language_id": dict(tc.codec_language_id),
    }
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"Saved config.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--output-dir", default="./qwen3-sherpa")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

    print(f"Loading model: {a.model}")
    model = AutoModel.from_pretrained(
        a.model, device_map=a.device, dtype=torch.float32,
        attn_implementation="eager",  # avoid SDPA+GQA that TorchScript can't trace
    )
    model.eval()
    print(f"Model loaded on {model.device}")

    export_text_project(model, a.output_dir)
    export_codec_embed(model, a.output_dir)
    export_code_predictor_embed(model, a.output_dir)
    export_code_predictor(model, a.output_dir)
    export_talker_prefill(model, a.output_dir)
    export_talker_decode(model, a.output_dir)
    try:
        export_vocoder(model, a.output_dir)
    except Exception as e:
        print(f"  Vocoder export failed: {e}")
        print("  (Use existing vocoder.onnx from elbruno instead)")
    save_config(model, a.output_dir)

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"Files in: {a.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
