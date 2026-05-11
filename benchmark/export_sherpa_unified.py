#!/usr/bin/env python3
"""Export Qwen3-TTS 0.6B to ONNX — unified talker (prefill + decode in one model).

This script exports a SINGLE talker ONNX that handles both:
  - Prefill mode: inputs_embeds [1, S, 1024]  + past_key_* [1, 8, 0, 128]  (S > 1)
  - Decode mode:  inputs_embeds [1, 1, 1024]  + past_key_* [1, 8, P, 128]  (P > 0)

The trick: trace with S=5, P=10 (general case) so the causal-mask branch is taken.
For decode (S=1) the triu mask becomes all-zeros (no masking), which is correct.

Other sub-models (text_project, codec_embed, code_predictor_embed, code_predictor,
vocoder) are unchanged from export_sherpa_style.py and are still exported here.

Usage:
    python3 export_sherpa_unified.py \\
        --model /tmp/qwen3-full \\
        --output-dir /tmp/qwen3-unified \\
        --device cuda:0
"""
import argparse, json, os, sys, types, importlib.machinery
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Mock torchaudio BEFORE importing qwen_tts — qwen_tts.__init__ imports
# torchaudio which may be broken/incompatible on the export machine.
# We only need torchaudio for the vocoder export (which we skip with --skip-aux).
# ---------------------------------------------------------------------------
if "torchaudio" not in sys.modules:
    for _name in ["torchaudio", "torchaudio.compliance", "torchaudio.compliance.kaldi",
                  "torchaudio._extension", "torchaudio.lib"]:
        _fake = types.ModuleType(_name)
        _fake.__path__ = ["/tmp/fake_torchaudio"]
        _fake.__file__ = "/tmp/fake_torchaudio/__init__.py"
        _fake.__spec__ = importlib.machinery.ModuleSpec(
            _name, None, origin="/tmp/fake_torchaudio/__init__.py"
        )
        if _name == "torchaudio.compliance.kaldi":
            _fake.fbank = lambda *a, **kw: None
        sys.modules[_name] = _fake
    # Also set top-level attrs so sub-imports don't fail
    sys.modules["torchaudio"].compliance = sys.modules["torchaudio.compliance"]
    sys.modules["torchaudio.compliance"].kaldi = sys.modules["torchaudio.compliance.kaldi"]

from transformers import AutoConfig, AutoModel, AutoProcessor

# ---------------------------------------------------------------------------
# Monkey-patch create_causal_mask: UNIFIED version that works for any (S, P)
#
# Key insight: we must NOT branch on seq_len == 1 in Python — that would bake
# the branch away during TorchScript tracing.  Instead always compute the
# triu mask from tensor shapes.  When S=1 the triu diagonal is P+1 > 1, so the
# [1,1,P+1] mask is all-zeros — identical to "no masking", correct for decode.
# ---------------------------------------------------------------------------
import transformers.masking_utils


def _unified_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, **kwargs):
    """Unified causal mask — always computed from tensor shapes, no Python branch on seq_len.

    Works for both prefill (seq>1) and decode (seq=1, past>0):
      - Prefill: triu(full(S, S+P, -inf), diagonal=P+1) — standard causal mask
      - Decode:  triu(full(1, P+1, -inf), diagonal=P+1) — all zeros (no masking)
    """
    dtype = input_embeds.dtype
    device = input_embeds.device
    batch, seq_len = input_embeds.shape[0], input_embeds.shape[1]

    if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
        past_len = past_key_values.get_seq_length()
    else:
        past_len = 0

    total_len = past_len + seq_len
    mask = torch.triu(
        torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype),
        diagonal=past_len + 1,
    )
    return mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)


transformers.masking_utils.create_causal_mask = _unified_causal_mask

from qwen_tts.core.models import (
    Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor,
)


# ---------------------------------------------------------------------------
# 1. text_project: token IDs -> projected text embeddings  (unchanged)
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
# 2. codec_embed: codec token IDs -> embeddings  (unchanged)
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
# 3. code_predictor_embed  (unchanged)
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
# 4. code_predictor  (unchanged)
# ---------------------------------------------------------------------------
def export_code_predictor(model, output_dir, opset=14):
    print("Exporting code_predictor.onnx ...")
    cp = model.talker.code_predictor

    class CP(nn.Module):
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor
            self.stacked_weights = nn.Parameter(
                torch.stack([h.weight for h in predictor.lm_head]),  # [15, 2048, 1024]
                requires_grad=False,
            )

        def forward(self, context, gen_step):
            out = self.predictor.model(
                inputs_embeds=context, use_cache=False, return_dict=True,
            )
            hidden = out.last_hidden_state[:, -1:, :]  # [1, 1, 1024]
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
# 5. talker_unified: SINGLE model for both prefill and decode
#
# Interface:
#   Inputs:
#     inputs_embeds   [1, S, D]       — S=1 for decode, S>1 for prefill
#     attention_mask  [1, S+P]        — optional but kept for compatibility
#     past_key_{i}    [1, H, P, dh]   — P=0 for prefill start
#     past_value_{i}  [1, H, P, dh]
#   Outputs:
#     logits          [1, S, vocab]
#     last_hidden     [1, S, D]
#     new_past_key_{i}   [1, H, S+P, dh]
#     new_past_value_{i} [1, H, S+P, dh]
#
# Tracing strategy: use S=5 (>1) and P=10 (>0) so the general causal-mask
# path is traced.  Dynamic axes ensure the graph works for any (S, P).
# ---------------------------------------------------------------------------
def export_talker_unified(model, output_dir, opset=14):
    print("Exporting talker_unified.onnx ...")
    talker = model.talker
    N = talker.config.num_hidden_layers
    D = talker.config.hidden_size
    H = talker.config.num_key_value_heads
    dh = getattr(talker.config, 'head_dim', D // talker.config.num_attention_heads)

    class TalkerUnified(nn.Module):
        def __init__(self, talker, num_layers):
            super().__init__()
            self.talker = talker
            self.num_layers = num_layers

        def forward(self, inputs_embeds, attention_mask, *past_kv_flat):
            from transformers.cache_utils import DynamicCache
            cache = DynamicCache()
            for i in range(self.num_layers):
                cache.update(past_kv_flat[2 * i], past_kv_flat[2 * i + 1], i)
            out = self.talker.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            hidden = out.last_hidden_state
            logits = self.talker.codec_head(hidden)
            new_pkv = out.past_key_values.to_legacy_cache()
            return (logits, hidden) + tuple(t for kv in new_pkv for t in kv)

    w = TalkerUnified(talker, N).eval()

    # Trace with S=5 (prefill) AND P=10 (non-empty past) — general case.
    # This ensures the causal-mask branch is taken, not the "no mask" path.
    S_trace = 5
    P_trace = 10
    dummy_e = torch.randn(1, S_trace, D, device=model.device)
    dummy_m = torch.ones(1, S_trace + P_trace, dtype=torch.long, device=model.device)
    dummy_kv = [torch.randn(1, H, P_trace, dh, device=model.device) for _ in range(N * 2)]

    in_kv = []
    out_kv = []
    for i in range(N):
        in_kv += [f"past_key_{i}", f"past_value_{i}"]
        out_kv += [f"new_past_key_{i}", f"new_past_value_{i}"]

    # Dynamic axes: seq_len and past_len are both variable
    dyn = {
        "inputs_embeds": {1: "seq_len"},
        "attention_mask": {1: "full_len"},
        "last_hidden": {1: "seq_len"},
        "logits": {1: "seq_len"},
    }
    for n in in_kv:
        dyn[n] = {2: "past_len"}
    for n in out_kv:
        dyn[n] = {2: "new_len"}

    with torch.no_grad():
        torch.onnx.export(
            w,
            (dummy_e, dummy_m, *dummy_kv),
            f"{output_dir}/talker_unified.onnx",
            input_names=["inputs_embeds", "attention_mask"] + in_kv,
            output_names=["logits", "last_hidden"] + out_kv,
            dynamic_axes=dyn,
            opset_version=opset,
            dynamo=False,
        )
    print(f"  Done ({N} layers, trace S={S_trace} P={P_trace})")


# ---------------------------------------------------------------------------
# 6. vocoder  (unchanged)
# ---------------------------------------------------------------------------
def export_vocoder(model, output_dir, opset=18):
    print("Exporting tokenizer12hz_decode.onnx ...")
    import io, onnx

    processor = AutoProcessor.from_pretrained(model.config._name_or_path, fix_mistral_regex=True)
    speech_model = processor.speech_tokenizer.model
    upsample_rate = speech_model.decode_upsample_rate

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
        "export_mode": "unified",  # distinguish from split export
    }
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("Saved config.json")


# ---------------------------------------------------------------------------
# Verify: check that exported ONNX has dynamic dims and runs in both modes
# ---------------------------------------------------------------------------
def verify_unified(output_dir, N, H, D, dh):
    import onnx, onnxruntime as ort, numpy as np

    path = f"{output_dir}/talker_unified.onnx"
    print(f"\nVerifying {path} ...")

    # Shape check
    m = onnx.load(path, load_external_data=False)
    for inp in m.graph.input:
        shape = inp.type.tensor_type.shape
        if shape is None:
            continue
        dims = [d.dim_param or str(d.dim_value) for d in shape.dim]
        print(f"  input {inp.name}: {dims}")

    # Check inputs_embeds dim[1] is symbolic
    ok = True
    for inp in m.graph.input:
        if inp.name == "inputs_embeds":
            d1 = inp.type.tensor_type.shape.dim[1]
            if d1.dim_param == "":
                print("  WARN: inputs_embeds dim[1] is NOT dynamic!")
                ok = False
            else:
                print(f"  OK: inputs_embeds dim[1] = '{d1.dim_param}' (dynamic)")

    # Runtime test
    # Note: attention_mask is folded away during tracing — the model absorbs it
    # via the monkey-patched _unified_causal_mask which uses input_embeds shape.
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in sess.get_inputs()}

    # Prefill mode: S=5, P=0
    prefill_inputs = {"inputs_embeds": np.random.randn(1, 5, D).astype(np.float32)}
    for i in range(N):
        prefill_inputs[f"past_key_{i}"] = np.zeros((1, H, 0, dh), dtype=np.float32)
        prefill_inputs[f"past_value_{i}"] = np.zeros((1, H, 0, dh), dtype=np.float32)
    try:
        res = sess.run(None, prefill_inputs)
        print(f"  Prefill (S=5,P=0): logits={res[0].shape} OK")
    except Exception as e:
        print(f"  Prefill FAILED: {e}")
        ok = False

    # Decode mode: S=1, P=5
    decode_inputs = {"inputs_embeds": np.random.randn(1, 1, D).astype(np.float32)}
    for i in range(N):
        decode_inputs[f"past_key_{i}"] = np.random.randn(1, H, 5, dh).astype(np.float32)
        decode_inputs[f"past_value_{i}"] = np.random.randn(1, H, 5, dh).astype(np.float32)
    try:
        res2 = sess.run(None, decode_inputs)
        print(f"  Decode  (S=1,P=5): logits={res2[0].shape} OK")
    except Exception as e:
        print(f"  Decode FAILED: {e}")
        ok = False

    # Decode mode: S=1, P=100 (stress test)
    decode_large = {"inputs_embeds": np.random.randn(1, 1, D).astype(np.float32)}
    for i in range(N):
        decode_large[f"past_key_{i}"] = np.random.randn(1, H, 100, dh).astype(np.float32)
        decode_large[f"past_value_{i}"] = np.random.randn(1, H, 100, dh).astype(np.float32)
    try:
        res3 = sess.run(None, decode_large)
        print(f"  Decode  (S=1,P=100): logits={res3[0].shape} OK")
    except Exception as e:
        print(f"  Decode large FAILED: {e}")
        ok = False

    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--output-dir", default="./qwen3-unified")
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-aux", action="store_true",
                   help="Skip text_project, codec_embed, code_predictor, vocoder")
    p.add_argument("--verify", action="store_true",
                   help="Verify exported ONNX in ORT after export")
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

    print(f"Loading model: {a.model}")
    model = AutoModel.from_pretrained(
        a.model, device_map=a.device, dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded on {model.device}")

    if not a.skip_aux:
        export_text_project(model, a.output_dir)
        export_codec_embed(model, a.output_dir)
        export_code_predictor_embed(model, a.output_dir)
        export_code_predictor(model, a.output_dir)

    export_talker_unified(model, a.output_dir)

    if not a.skip_aux:
        try:
            export_vocoder(model, a.output_dir)
        except Exception as e:
            print(f"  Vocoder export failed: {e}")
            print("  (Use existing tokenizer12hz_decode.onnx from prior export)")

    save_config(model, a.output_dir)

    if a.verify:
        tc = model.talker.config
        N = tc.num_hidden_layers
        H = tc.num_key_value_heads
        D = tc.hidden_size
        dh = getattr(tc, 'head_dim', D // tc.num_attention_heads)
        ok = verify_unified(a.output_dir, N, H, D, dh)
        if ok:
            print("\nVerification PASSED")
        else:
            print("\nVerification FAILED — check warnings above")

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"Files in: {a.output_dir}")
    for f in sorted(os.listdir(a.output_dir)):
        path = os.path.join(a.output_dir, f)
        size = os.path.getsize(path)
        print(f"  {f}: {size / 1e6:.1f} MB" if size > 1e6 else f"  {f}: {size / 1e3:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
