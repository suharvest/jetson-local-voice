#!/usr/bin/env python3
"""Export Qwen3-ASR 0.6B to ONNX — encoder + decoder_prefill + decoder_step.

Per-layer KV cache (not stacked) for TRT compatibility.
Uses TorchScript trace (dynamo=False) to avoid shape baking.
Standard ops only, external data format for >2GB models.

Usage:
    python3 export_qwen3_asr.py --device cpu --output-dir /tmp/qwen3-asr-onnx
    python3 export_qwen3_asr.py --device cuda:0 --output-dir /tmp/qwen3-asr-onnx
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
# Mock torchaudio (not available / not needed for export)
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
# Monkey-patch create_causal_mask for trace-friendly export
# ---------------------------------------------------------------------------
import transformers.masking_utils


def _simple_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, **kwargs):
    """Simple causal mask without vmap — TorchScript-traceable."""
    dtype = input_embeds.dtype
    device = input_embeds.device
    batch, seq_len = input_embeds.shape[:2]

    if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
        past_len = past_key_values.get_seq_length()
    else:
        past_len = 0

    total_len = past_len + seq_len
    if seq_len == 1:
        return None
    mask = torch.triu(
        torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype),
        diagonal=past_len + 1,
    )
    return mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)


transformers.masking_utils.create_causal_mask = _simple_causal_mask

from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)

# ---------------------------------------------------------------------------
# Constants from the Qwen3-ASR prompt format
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
# Encoder helpers (reimplemented for trace-friendly export)
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
# Decoder helpers (reimplemented for trace-friendly export)
# ---------------------------------------------------------------------------

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
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
# 1. Export encoder
# ---------------------------------------------------------------------------
def export_encoder(model, output_dir, opset=14):
    print("Exporting encoder.onnx ...")
    audio_tower = model.thinker.audio_tower
    wrapper = EncoderWrapper(audio_tower).eval()

    # Use non-round frame count to exercise padding
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
            dynamic_axes={
                "mel": {2: "time"},
                "audio_features": {1: "enc_time"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    # Fix Reshape allowzero for DirectML/TRT compatibility
    _fix_reshape_allowzero(output_path)

    # Embed weights into proto if small enough
    data_path = output_path + ".data"
    if os.path.exists(data_path):
        size = os.path.getsize(data_path)
        if size < 1_800_000_000:
            print("  Embedding encoder weights into .onnx proto...")
            enc = onnx.load(output_path, load_external_data=True)
            onnx.save(enc, output_path)
            os.remove(data_path)
        else:
            print(f"  Encoder weights too large to inline ({size / 1e6:.0f} MB), keeping external data")

    size = os.path.getsize(output_path)
    print(f"  Done: encoder.onnx ({size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 2. Export decoder_prefill (per-layer KV cache)
# ---------------------------------------------------------------------------
class DecoderPrefillWrapper(nn.Module):
    """Prefill: input_ids + audio_features -> logits + last_hidden + per-layer KV cache."""

    def __init__(self, text_model, lm_head, text_config):
        super().__init__()
        self.embed_tokens = text_model.embed_tokens
        self.layers = text_model.layers
        self.norm = text_model.norm
        self.rotary_emb = text_model.rotary_emb
        self.lm_head = lm_head
        self.num_kv_groups = text_config.num_attention_heads // text_config.num_key_value_heads

    def forward(self, input_ids, position_ids, audio_features, audio_offset):
        input_embeds = self.embed_tokens(input_ids)

        # Scatter audio features over audio_pad positions
        audio_len = audio_features.shape[1]
        indices = torch.arange(audio_len, device=input_ids.device) + audio_offset[0]
        indices = indices.unsqueeze(0).unsqueeze(-1).expand_as(audio_features)
        input_embeds = input_embeds.scatter(1, indices, audio_features)

        batch, seq_len = input_embeds.shape[:2]

        # MRoPE: tile 1D position_ids to 3D (all three sections get same value for ASR)
        pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)
        cos, sin = self.rotary_emb(input_embeds, pos_3d)

        # Causal mask
        causal_mask = torch.full(
            (seq_len, seq_len),
            torch.finfo(input_embeds.dtype).min,
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        hidden_states = input_embeds
        kv_outputs = []

        for layer in self.layers:
            hidden_states, key_states, value_states = _decoder_layer_forward(
                layer, hidden_states, cos, sin, causal_mask,
                past_key=None, past_value=None,
                num_kv_groups=self.num_kv_groups,
            )
            kv_outputs.append(key_states)
            kv_outputs.append(value_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return (logits, hidden_states) + tuple(kv_outputs)


def export_decoder_prefill(model, output_dir, opset=14):
    print("Exporting decoder_prefill.onnx ...")
    text_config = model.config.thinker_config.text_config
    N = text_config.num_hidden_layers
    D = text_config.hidden_size

    text_model = model.thinker.model
    lm_head = model.thinker.lm_head
    wrapper = DecoderPrefillWrapper(text_model, lm_head, text_config).eval()
    device = next(wrapper.parameters()).device

    seq_len = 100
    audio_len = 80
    audio_offset_val = 9
    dummy_ids = torch.zeros(1, seq_len, device=device, dtype=torch.long)
    dummy_pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    dummy_audio = torch.randn(1, audio_len, D, device=device, dtype=torch.float32)
    dummy_offset = torch.tensor([audio_offset_val], device=device, dtype=torch.long)

    kv_names = []
    for i in range(N):
        kv_names += [f"past_key_{i}", f"past_value_{i}"]

    output_path = os.path.join(output_dir, "decoder_prefill.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_pos, dummy_audio, dummy_offset),
            output_path,
            input_names=["input_ids", "position_ids", "audio_features", "audio_offset"],
            output_names=["logits", "last_hidden"] + kv_names,
            dynamic_axes={
                "input_ids": {1: "S"},
                "position_ids": {1: "S"},
                "audio_features": {1: "A"},
                "logits": {1: "S"},
                "last_hidden": {1: "S"},
                **{n: {2: "S"} for n in kv_names},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    _fix_reshape_allowzero(output_path)
    _consolidate_external_data(output_path, output_dir)

    size = os.path.getsize(output_path)
    data_file = os.path.join(output_dir, "decoder_prefill.onnx.data")
    data_size = os.path.getsize(data_file) if os.path.exists(data_file) else 0
    print(f"  Done: decoder_prefill.onnx ({size / 1e6:.1f} MB graph + {data_size / 1e6:.1f} MB weights)")


# ---------------------------------------------------------------------------
# 3. Export decoder_step (per-layer KV cache, no embedding table)
# ---------------------------------------------------------------------------
class DecoderStepWrapper(nn.Module):
    """Autoregressive step: input_embeds + per-layer KV -> logits + updated per-layer KV."""

    def __init__(self, text_model, lm_head, text_config):
        super().__init__()
        self.layers = text_model.layers
        self.norm = text_model.norm
        self.rotary_emb = text_model.rotary_emb
        self.lm_head = lm_head
        self.num_kv_groups = text_config.num_attention_heads // text_config.num_key_value_heads
        self.num_layers = text_config.num_hidden_layers

    def forward(self, input_embeds, position_ids, *past_kv_flat):
        # MRoPE: tile 1D position_ids to 3D
        pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)
        cos, sin = self.rotary_emb(input_embeds, pos_3d)

        # No causal mask needed for single-token (attends to all past + self)
        mask = None

        hidden_states = input_embeds
        new_kv = []

        for i, layer in enumerate(self.layers):
            past_key = past_kv_flat[2 * i]
            past_value = past_kv_flat[2 * i + 1]
            hidden_states, key_states, value_states = _decoder_layer_forward(
                layer, hidden_states, cos, sin, mask,
                past_key=past_key, past_value=past_value,
                num_kv_groups=self.num_kv_groups,
            )
            new_kv.append(key_states)
            new_kv.append(value_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return (logits,) + tuple(new_kv)


def export_decoder_step(model, output_dir, opset=14):
    print("Exporting decoder_step.onnx ...")
    text_config = model.config.thinker_config.text_config
    N = text_config.num_hidden_layers
    D = text_config.hidden_size
    H = text_config.num_key_value_heads
    dh = text_config.head_dim

    text_model = model.thinker.model
    lm_head = model.thinker.lm_head
    wrapper = DecoderStepWrapper(text_model, lm_head, text_config).eval()
    device = next(wrapper.parameters()).device

    T_past = 100
    dummy_e = torch.randn(1, 1, D, device=device, dtype=torch.float32)
    dummy_pos = torch.tensor([[T_past]], device=device, dtype=torch.long)
    dummy_kv = [torch.randn(1, H, T_past, dh, device=device) for _ in range(N * 2)]

    in_kv = []
    out_kv = []
    for i in range(N):
        in_kv += [f"past_key_{i}", f"past_value_{i}"]
        out_kv += [f"new_past_key_{i}", f"new_past_value_{i}"]

    output_path = os.path.join(output_dir, "decoder_step.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_e, dummy_pos, *dummy_kv),
            output_path,
            input_names=["input_embeds", "position_ids"] + in_kv,
            output_names=["logits"] + out_kv,
            dynamic_axes={
                **{n: {2: "past_len"} for n in in_kv},
                **{n: {2: "new_len"} for n in out_kv},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    _fix_reshape_allowzero(output_path)
    _consolidate_external_data(output_path, output_dir)

    size = os.path.getsize(output_path)
    data_file = os.path.join(output_dir, "decoder_step.onnx.data")
    data_size = os.path.getsize(data_file) if os.path.exists(data_file) else 0
    print(f"  Done: decoder_step.onnx ({size / 1e6:.1f} MB" +
          (f" + {data_size / 1e6:.1f} MB weights)" if data_size else ")"))


# ---------------------------------------------------------------------------
# 4. Save embed_tokens.bin (FP16)
# ---------------------------------------------------------------------------
def save_embed_tokens(model, output_dir):
    print("Saving embed_tokens.bin ...")
    embed_weight = model.thinker.model.embed_tokens.weight.data
    embed_np = embed_weight.cpu().to(torch.float16).numpy()
    embed_path = os.path.join(output_dir, "embed_tokens.bin")
    embed_np.tofile(embed_path)
    print(f"  Done: {embed_np.shape} FP16 ({embed_np.nbytes / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 5. Save config + tokenizer
# ---------------------------------------------------------------------------
def save_config(model, output_dir):
    thinker_config = model.config.thinker_config
    audio_config = thinker_config.audio_config
    text_config = thinker_config.text_config

    config = {
        "model_type": "qwen3_asr",
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
    print(f"Saved config.json")


def save_tokenizer(model_id, output_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    # Clean up unwanted files
    keep = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
            "added_tokens.json", "special_tokens_map.json"}
    for f in os.listdir(output_dir):
        if f.endswith(".py") or f.endswith(".txt"):
            # Don't remove .txt if it's merges.txt
            if f == "merges.txt":
                continue
            os.remove(os.path.join(output_dir, f))
    print(f"Saved tokenizer files")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _consolidate_external_data(onnx_path, output_dir):
    """Reload ONNX model and re-save with consolidated external data or inlined.

    torch.onnx.export scatters external data into per-tensor files.
    This reloads and saves as a single .data file, or inlines if small enough.
    """
    # Check if there are scattered external data files
    basename = os.path.basename(onnx_path)
    model_obj = onnx.load(onnx_path, load_external_data=True)

    # Calculate total weight size
    total_size = sum(
        t.raw_data.__len__() if t.raw_data else 0
        for t in model_obj.graph.initializer
    )

    if total_size < 1_800_000_000:
        # Small enough to inline
        print(f"  Inlining weights ({total_size / 1e6:.0f} MB) into {basename}...")
        onnx.save(model_obj, onnx_path)
    else:
        # Save with single external data file
        data_filename = basename + ".data"
        print(f"  Consolidating weights ({total_size / 1e6:.0f} MB) into {data_filename}...")
        onnx.save_model(
            model_obj,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_filename,
        )

    # Clean up scattered tensor files
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath) and not os.path.splitext(f)[1] and f not in (
            "Makefile", "LICENSE", "README",
        ):
            # Heuristic: files with no extension that look like tensor names
            if f.startswith("onnx__") or f == "embed_tokens.weight":
                os.remove(fpath)


def _fix_reshape_allowzero(onnx_path):
    """Remove allowzero=1 from Reshape nodes for DirectML/TRT compatibility."""
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
    p = argparse.ArgumentParser(description="Export Qwen3-ASR 0.6B to ONNX")
    p.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    p.add_argument("--output-dir", default="/tmp/qwen3-asr-onnx")
    p.add_argument("--device", default="cpu")
    p.add_argument("--opset", type=int, default=14)
    p.add_argument("--skip-encoder", action="store_true")
    p.add_argument("--skip-decoder", action="store_true")
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
        export_decoder_prefill(model, a.output_dir, opset=a.opset)
        export_decoder_step(model, a.output_dir, opset=a.opset)
        save_embed_tokens(model, a.output_dir)

    save_config(model, a.output_dir)
    save_tokenizer(a.model, a.output_dir)

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
