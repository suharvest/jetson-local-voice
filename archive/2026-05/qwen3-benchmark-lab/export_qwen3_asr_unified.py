#!/usr/bin/env python3
"""Export Qwen3-ASR 0.6B to ONNX — unified decoder (prefill + decode in one model).

This script exports:
  1. encoder.onnx        — unchanged from export_qwen3_asr.py
  2. decoder_unified.onnx — SINGLE model handling both prefill and autoregressive step:
       Prefill: input_embeds [1, S, D] + position_ids [1, S] + past_key_* [1, H, 0, dh]
       Decode:  input_embeds [1, 1, D] + position_ids [1, 1] + past_key_* [1, H, P, dh]
  3. embed_tokens.bin    — FP16 embedding table (unchanged)
  4. config.json / tokenizer — unchanged

The key design decisions:
  - Trace with S=5, P=10 (general case) so the causal-mask branch is always taken.
  - For decode (S=1): triu(full(1, P+1, -inf), diagonal=P+1) = all zeros = no masking.
  - The causal mask is computed from tensor shapes (no Python branch on seq_len).
  - dynamic_axes covers both seq_len (dim 1 of input_embeds/position_ids) and
    past_len (dim 2 of all KV tensors).

Usage:
    python3 export_qwen3_asr_unified.py \\
        --model Qwen/Qwen3-ASR-0.6B \\
        --output-dir /tmp/qwen3-asr-unified \\
        --device cuda:0 \\
        --verify
"""

import argparse
import json
import os
import sys
import types
import importlib.machinery

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Mock torchaudio
# ---------------------------------------------------------------------------
for _name in ["torchaudio", "torchaudio.compliance", "torchaudio.compliance.kaldi"]:
    _fake = types.ModuleType(_name)
    _fake.__path__ = ["/tmp/fake"]
    _fake.__file__ = "/tmp/fake/__init__.py"
    _fake.__spec__ = importlib.machinery.ModuleSpec(
        _name, None, origin="/tmp/fake/__init__.py"
    )
    if _name == "torchaudio.compliance.kaldi":
        _fake.fbank = lambda *a, **kw: None
    sys.modules[_name] = _fake

# ---------------------------------------------------------------------------
# Monkey-patch create_causal_mask — only needed if the ASR decoder uses it.
# The wrapper below computes its own causal mask, but patch anyway for safety.
# ---------------------------------------------------------------------------
import transformers.masking_utils


def _unified_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, **kwargs):
    """Unified causal mask: no Python branch on seq_len == 1."""
    dtype = input_embeds.dtype
    device = input_embeds.device
    batch, seq_len = input_embeds.shape[0], input_embeds.shape[1]
    if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
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

from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONV_WINDOW = 100
TOKENS_PER_WINDOW = 13
ATTN_WINDOW_SIZE = 104


def _conv_out_len(t):
    return (t + 1) // 2


def _get_feat_extract_output_lengths(input_lengths):
    leave = input_lengths % CONV_WINDOW
    t = _conv_out_len(leave)
    t = _conv_out_len(t)
    t = _conv_out_len(t)
    return t + (input_lengths // CONV_WINDOW) * TOKENS_PER_WINDOW


# ---------------------------------------------------------------------------
# Shared attention helpers (same as original)
# ---------------------------------------------------------------------------

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _attention(query, key, value, mask, scaling, num_kv_groups):
    key = _repeat_kv(key, num_kv_groups)
    value = _repeat_kv(value, num_kv_groups)
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if mask is not None:
        attn_weights = attn_weights + mask[:, :, :, : key.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous()


def _decoder_layer_forward(layer, hidden_states, cos, sin, mask, past_key, past_value, num_kv_groups):
    attn = layer.self_attn
    batch, seq, _ = hidden_states.shape

    residual = hidden_states
    normed = layer.input_layernorm(hidden_states)

    input_shape = normed.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    query_states = attn.q_norm(attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
    value_states = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

    query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key is not None:
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)

    attn_output = _attention(
        query_states, key_states, value_states, mask, attn.scaling, num_kv_groups
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn.o_proj(attn_output)

    hidden_states = residual + attn_output
    residual = hidden_states
    normed = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(normed)

    return hidden_states, key_states, value_states


# ---------------------------------------------------------------------------
# Encoder wrapper (unchanged from original)
# ---------------------------------------------------------------------------
def _encoder_attention(q, k, v, mask, scaling):
    attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
    attn_weights = attn_weights + mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn_weights, v)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def _encoder_layer_forward(layer, x, attn_mask, scaling, num_heads, head_dim):
    sa = layer.self_attn
    batch, seq, _ = x.shape
    residual = x
    normed = layer.self_attn_layer_norm(x)
    q = sa.q_proj(normed).view(batch, seq, num_heads, head_dim).transpose(1, 2)
    k = sa.k_proj(normed).view(batch, seq, num_heads, head_dim).transpose(1, 2)
    v = sa.v_proj(normed).view(batch, seq, num_heads, head_dim).transpose(1, 2)
    attn_out = _encoder_attention(q, k, v, attn_mask, scaling)
    attn_out = sa.out_proj(attn_out)
    x = residual + attn_out
    residual = x
    normed = layer.final_layer_norm(x)
    x = residual + layer.fc2(F.gelu(layer.fc1(normed)))
    return x


class EncoderWrapper(nn.Module):
    """Trace-friendly encoder: mel [1, 128, T] -> audio_features [1, T', output_dim]."""

    def __init__(self, audio_tower):
        super().__init__()
        self.conv2d1 = audio_tower.conv2d1
        self.conv2d2 = audio_tower.conv2d2
        self.conv2d3 = audio_tower.conv2d3
        self.conv_out = audio_tower.conv_out
        self.positional_embedding = audio_tower.positional_embedding
        self.layers = audio_tower.layers
        self.ln_post = audio_tower.ln_post
        self.proj1 = audio_tower.proj1
        self.proj2 = audio_tower.proj2
        self.act = audio_tower.act
        self.scaling = self.layers[0].self_attn.scaling
        self.d_model = audio_tower.config.d_model
        self.num_heads = audio_tower.config.encoder_attention_heads
        self.head_dim = self.d_model // self.num_heads
        self.output_dim = audio_tower.config.output_dim

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        T = mel.shape[2]
        pad_amount = (CONV_WINDOW - T % CONV_WINDOW) % CONV_WINDOW
        mel = F.pad(mel, (0, pad_amount))
        T_padded = mel.shape[2]
        num_conv_windows = T_padded // CONV_WINDOW

        x = mel.squeeze(0)
        x = x.reshape(128, num_conv_windows, CONV_WINDOW)
        x = x.permute(1, 0, 2)
        x = x.unsqueeze(1)

        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))

        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x = self.conv_out(x)

        pos_embed = self.positional_embedding(t)
        x = x + pos_embed.unsqueeze(0)

        valid_count = _get_feat_extract_output_lengths(T)
        flat = x.reshape(-1, self.d_model)
        flat = flat[:valid_count]

        attn_pad = (ATTN_WINDOW_SIZE - valid_count % ATTN_WINDOW_SIZE) % ATTN_WINDOW_SIZE
        flat = F.pad(flat, (0, 0, 0, attn_pad))
        total_padded = valid_count + attn_pad
        num_attn_windows = total_padded // ATTN_WINDOW_SIZE
        x = flat.reshape(num_attn_windows, ATTN_WINDOW_SIZE, self.d_model)

        positions = torch.arange(total_padded, device=mel.device)
        positions = positions.reshape(num_attn_windows, ATTN_WINDOW_SIZE)
        pad_mask = (positions >= valid_count).to(mel.dtype) * torch.finfo(mel.dtype).min
        attn_mask = pad_mask.unsqueeze(1).unsqueeze(1)

        for layer in self.layers:
            x = _encoder_layer_forward(
                layer, x, attn_mask, self.scaling, self.num_heads, self.head_dim
            )

        x = x.reshape(-1, self.d_model)[:valid_count]
        x = x.unsqueeze(0)
        x = self.ln_post(x)
        x = self.act(self.proj1(x))
        x = self.proj2(x)
        return x


# ---------------------------------------------------------------------------
# Unified decoder wrapper
#
# Interface:
#   Inputs:
#     input_embeds    [1, S, D]       — precomputed embeddings (S=1 for decode)
#     position_ids    [1, S]          — absolute positions
#     past_key_{i}    [1, H, P, dh]   — P=0 for first prefill
#     past_value_{i}  [1, H, P, dh]
#   Outputs:
#     logits          [1, S, vocab]
#     new_past_key_{i}   [1, H, S+P, dh]
#     new_past_value_{i} [1, H, S+P, dh]
#
# Note: No audio_features or audio_offset here — the caller is responsible for
# constructing input_embeds (scatter audio features before calling this model).
# embed_tokens.bin is saved separately so the caller can do token embedding.
#
# The causal mask is ALWAYS computed from tensor shapes (no Python seq_len==1 branch):
#   triu(full(S, S+P, -inf), diagonal=P+1)
# When S=1 this is a [1, P+1] row with triu diagonal P+1 >= 1, giving all zeros.
# ---------------------------------------------------------------------------
class DecoderUnifiedWrapper(nn.Module):
    """Unified prefill + decode: input_embeds + per-layer KV -> logits + updated KV."""

    def __init__(self, text_model, lm_head, text_config):
        super().__init__()
        self.layers = text_model.layers
        self.norm = text_model.norm
        self.rotary_emb = text_model.rotary_emb
        self.lm_head = lm_head
        self.num_kv_groups = text_config.num_attention_heads // text_config.num_key_value_heads
        self.num_layers = text_config.num_hidden_layers

    def forward(self, input_embeds, position_ids, *past_kv_flat):
        # MRoPE: tile 1D position_ids to 3D (all three sections get same value for ASR)
        pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)
        cos, sin = self.rotary_emb(input_embeds, pos_3d)

        # Compute causal mask from tensor shapes — NO Python branch on seq_len.
        # past_len comes from the first KV tensor's dim 2.
        # Shape of past_kv_flat[0]: [1, H, P, dh]
        past_len = past_kv_flat[0].shape[2]
        seq_len = input_embeds.shape[1]
        total_len = past_len + seq_len
        causal_mask = torch.triu(
            torch.full(
                (seq_len, total_len),
                torch.finfo(input_embeds.dtype).min,
                device=input_embeds.device,
                dtype=input_embeds.dtype,
            ),
            diagonal=past_len + 1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        hidden_states = input_embeds
        new_kv = []

        for i, layer in enumerate(self.layers):
            past_key = past_kv_flat[2 * i]
            past_value = past_kv_flat[2 * i + 1]
            hidden_states, key_states, value_states = _decoder_layer_forward(
                layer, hidden_states, cos, sin, causal_mask,
                past_key=past_key, past_value=past_value,
                num_kv_groups=self.num_kv_groups,
            )
            new_kv.append(key_states)
            new_kv.append(value_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return (logits,) + tuple(new_kv)


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_encoder(model, output_dir, opset=14):
    print("Exporting encoder.onnx ...")
    audio_tower = model.thinker.audio_tower
    wrapper = EncoderWrapper(audio_tower).eval()

    dummy_mel = torch.randn(1, 128, 997, device=next(wrapper.parameters()).device, dtype=torch.float32)

    with torch.no_grad():
        test_out = wrapper(dummy_mel)
        expected_tokens = _get_feat_extract_output_lengths(997)
        assert test_out.shape == (1, expected_tokens, audio_tower.config.output_dim), (
            f"Shape mismatch: {test_out.shape}"
        )

    output_path = os.path.join(output_dir, "encoder.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_mel,),
            output_path,
            input_names=["mel"],
            output_names=["audio_features"],
            dynamic_axes={"mel": {2: "time"}, "audio_features": {1: "enc_time"}},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    _fix_reshape_allowzero(output_path)

    data_path = output_path + ".data"
    if os.path.exists(data_path):
        size = os.path.getsize(data_path)
        if size < 1_800_000_000:
            print("  Embedding encoder weights into .onnx proto...")
            enc = onnx.load(output_path, load_external_data=True)
            onnx.save(enc, output_path)
            os.remove(data_path)

    size = os.path.getsize(output_path)
    print(f"  Done: encoder.onnx ({size / 1e6:.1f} MB)")


def export_decoder_unified(model, output_dir, opset=14):
    print("Exporting decoder_unified.onnx ...")
    text_config = model.config.thinker_config.text_config
    N = text_config.num_hidden_layers
    D = text_config.hidden_size
    H = text_config.num_key_value_heads
    dh = text_config.head_dim

    text_model = model.thinker.model
    lm_head = model.thinker.lm_head
    wrapper = DecoderUnifiedWrapper(text_model, lm_head, text_config).eval()
    device = next(wrapper.parameters()).device

    # Trace with S=5 (prefill-like), P=10 (non-empty past) — general case.
    # This ensures the triu causal mask path is taken during tracing.
    S_trace = 5
    P_trace = 10
    dummy_e = torch.randn(1, S_trace, D, device=device, dtype=torch.float32)
    dummy_pos = torch.arange(P_trace, P_trace + S_trace, device=device, dtype=torch.long).unsqueeze(0)
    dummy_kv = [torch.randn(1, H, P_trace, dh, device=device, dtype=torch.float32) for _ in range(N * 2)]

    in_kv = []
    out_kv = []
    for i in range(N):
        in_kv += [f"past_key_{i}", f"past_value_{i}"]
        out_kv += [f"new_past_key_{i}", f"new_past_value_{i}"]

    dyn = {
        "input_embeds": {1: "seq_len"},
        "position_ids": {1: "seq_len"},
        "logits": {1: "seq_len"},
    }
    for n in in_kv:
        dyn[n] = {2: "past_len"}
    for n in out_kv:
        dyn[n] = {2: "new_len"}

    output_path = os.path.join(output_dir, "decoder_unified.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_e, dummy_pos, *dummy_kv),
            output_path,
            input_names=["input_embeds", "position_ids"] + in_kv,
            output_names=["logits"] + out_kv,
            dynamic_axes=dyn,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    _fix_reshape_allowzero(output_path)
    _consolidate_external_data(output_path, output_dir)

    size = os.path.getsize(output_path)
    data_file = os.path.join(output_dir, "decoder_unified.onnx.data")
    data_size = os.path.getsize(data_file) if os.path.exists(data_file) else 0
    print(f"  Done: decoder_unified.onnx ({size / 1e6:.1f} MB graph" +
          (f" + {data_size / 1e6:.1f} MB weights)" if data_size else ")"))
    print(f"  Traced with S={S_trace}, P={P_trace}")


def save_embed_tokens(model, output_dir):
    print("Saving embed_tokens.bin ...")
    embed_weight = model.thinker.model.embed_tokens.weight.data
    embed_np = embed_weight.cpu().to(torch.float16).numpy()
    embed_path = os.path.join(output_dir, "embed_tokens.bin")
    embed_np.tofile(embed_path)
    print(f"  Done: {embed_np.shape} FP16 ({embed_np.nbytes / 1e6:.1f} MB)")


def save_config(model, output_dir):
    thinker_config = model.config.thinker_config
    audio_config = thinker_config.audio_config
    text_config = thinker_config.text_config

    config = {
        "model_type": "qwen3_asr",
        "export_mode": "unified",
        "encoder": {
            "num_layers": audio_config.encoder_layers,
            "hidden_size": audio_config.d_model,
            "num_heads": audio_config.encoder_attention_heads,
            "ffn_dim": audio_config.encoder_ffn_dim,
            "conv_channels": audio_config.downsample_hidden_size,
            "output_dim": audio_config.output_dim,
            "downsample_factor": 8,
            "num_mel_bins": audio_config.num_mel_bins,
        },
        "decoder": {
            "num_layers": text_config.num_hidden_layers,
            "hidden_size": text_config.hidden_size,
            "num_attention_heads": text_config.num_attention_heads,
            "num_key_value_heads": text_config.num_key_value_heads,
            "head_dim": text_config.head_dim,
            "intermediate_size": text_config.intermediate_size,
            "vocab_size": text_config.vocab_size,
            "rope_theta": text_config.rope_theta,
            "rms_norm_eps": text_config.rms_norm_eps,
            "tie_word_embeddings": text_config.tie_word_embeddings,
            "rope_scaling": {
                "mrope_section": text_config.rope_scaling.get("mrope_section", [24, 20, 20]),
                "interleaved": text_config.rope_scaling.get("mrope_interleaved", True),
            },
        },
        "mel": {
            "sample_rate": 16000,
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 128,
            "fmin": 0,
            "fmax": 8000,
        },
        "special_tokens": {
            "audio_start_token_id": thinker_config.audio_start_token_id,
            "audio_end_token_id": thinker_config.audio_end_token_id,
            "audio_pad_token_id": thinker_config.audio_token_id,
            "eos_token_ids": [151643, 151645],
            "pad_token_id": 151643,
            "im_start_token_id": 151644,
            "im_end_token_id": 151645,
        },
    }

    output_path = os.path.join(output_dir, "config.json")
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print("Saved config.json")


def save_tokenizer(model_id, output_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print("Saved tokenizer files")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_decoder_unified(output_dir, N, H, D, dh):
    import onnxruntime as ort

    path = os.path.join(output_dir, "decoder_unified.onnx")
    print(f"\nVerifying {path} ...")

    # Shape check
    m = onnx.load(path, load_external_data=False)
    for inp in m.graph.input:
        shape = inp.type.tensor_type.shape
        if shape is None:
            continue
        dims = [d.dim_param or str(d.dim_value) for d in shape.dim]
        print(f"  input {inp.name}: {dims}")

    ok = True
    for inp in m.graph.input:
        if inp.name == "input_embeds":
            d1 = inp.type.tensor_type.shape.dim[1]
            if d1.dim_param == "":
                print("  WARN: input_embeds dim[1] is NOT dynamic!")
                ok = False
            else:
                print(f"  OK: input_embeds dim[1] = '{d1.dim_param}' (dynamic)")

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    # Prefill mode: S=5, P=0
    prefill_inputs = {
        "input_embeds": np.random.randn(1, 5, D).astype(np.float32),
        "position_ids": np.arange(5, dtype=np.int64).reshape(1, 5),
    }
    for i in range(N):
        prefill_inputs[f"past_key_{i}"] = np.zeros((1, H, 0, dh), dtype=np.float32)
        prefill_inputs[f"past_value_{i}"] = np.zeros((1, H, 0, dh), dtype=np.float32)
    try:
        res = sess.run(None, prefill_inputs)
        print(f"  Prefill (S=5,P=0): logits={res[0].shape} ✓")
    except Exception as e:
        print(f"  Prefill FAILED: {e}")
        ok = False

    # Decode mode: S=1, P=5
    decode_inputs = {
        "input_embeds": np.random.randn(1, 1, D).astype(np.float32),
        "position_ids": np.array([[5]], dtype=np.int64),
    }
    for i in range(N):
        decode_inputs[f"past_key_{i}"] = np.random.randn(1, H, 5, dh).astype(np.float32)
        decode_inputs[f"past_value_{i}"] = np.random.randn(1, H, 5, dh).astype(np.float32)
    try:
        res2 = sess.run(None, decode_inputs)
        print(f"  Decode  (S=1,P=5): logits={res2[0].shape} ✓")
    except Exception as e:
        print(f"  Decode FAILED: {e}")
        ok = False

    # Also test decode with large past (P=100) to stress dynamic axes
    decode_large = {
        "input_embeds": np.random.randn(1, 1, D).astype(np.float32),
        "position_ids": np.array([[100]], dtype=np.int64),
    }
    for i in range(N):
        decode_large[f"past_key_{i}"] = np.random.randn(1, H, 100, dh).astype(np.float32)
        decode_large[f"past_value_{i}"] = np.random.randn(1, H, 100, dh).astype(np.float32)
    try:
        res3 = sess.run(None, decode_large)
        print(f"  Decode  (S=1,P=100): logits={res3[0].shape} ✓")
    except Exception as e:
        print(f"  Decode large FAILED: {e}")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Utilities (same as original)
# ---------------------------------------------------------------------------
def _consolidate_external_data(onnx_path, output_dir):
    basename = os.path.basename(onnx_path)
    model_obj = onnx.load(onnx_path, load_external_data=True)
    total_size = sum(
        t.raw_data.__len__() if t.raw_data else 0
        for t in model_obj.graph.initializer
    )
    if total_size < 1_800_000_000:
        print(f"  Inlining weights ({total_size / 1e6:.0f} MB) into {basename}...")
        onnx.save(model_obj, onnx_path)
    else:
        data_filename = basename + ".data"
        print(f"  Consolidating weights ({total_size / 1e6:.0f} MB) into {data_filename}...")
        onnx.save_model(
            model_obj, onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_filename,
        )
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath) and not os.path.splitext(f)[1] and f not in (
            "Makefile", "LICENSE", "README",
        ):
            if f.startswith("onnx__") or f == "embed_tokens.weight":
                os.remove(fpath)


def _fix_reshape_allowzero(onnx_path):
    model = onnx.load(onnx_path, load_external_data=False)
    count = 0
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        for attr in list(node.attribute):
            if attr.name == "allowzero" and attr.i == 1:
                node.attribute.remove(attr)
                count += 1
    if count > 0:
        onnx.save(model, onnx_path)
    if count:
        print(f"  Fixed {count} Reshape allowzero attrs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Export Qwen3-ASR 0.6B to ONNX (unified decoder)")
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--output-dir", default="/tmp/qwen3-asr-unified")
    p.add_argument("--device", default="cpu")
    p.add_argument("--opset", type=int, default=14)
    p.add_argument("--skip-encoder", action="store_true")
    p.add_argument("--skip-decoder", action="store_true")
    p.add_argument("--verify", action="store_true", help="Run ORT verification after export")
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)

    print(f"Loading model: {a.model}")
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        a.model,
        dtype=torch.float32,
        device_map=a.device,
    )
    model.eval()
    print(f"Model loaded on {a.device}")

    if not a.skip_encoder:
        export_encoder(model, a.output_dir, opset=a.opset)

    if not a.skip_decoder:
        export_decoder_unified(model, a.output_dir, opset=a.opset)
        save_embed_tokens(model, a.output_dir)

    save_config(model, a.output_dir)
    save_tokenizer(a.model, a.output_dir)

    if a.verify and not a.skip_decoder:
        text_config = model.config.thinker_config.text_config
        ok = verify_decoder_unified(
            a.output_dir,
            N=text_config.num_hidden_layers,
            H=text_config.num_key_value_heads,
            D=text_config.hidden_size,
            dh=text_config.head_dim,
        )
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
        if size > 1e6:
            print(f"  {f}: {size / 1e6:.1f} MB")
        else:
            print(f"  {f}: {size / 1e3:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
