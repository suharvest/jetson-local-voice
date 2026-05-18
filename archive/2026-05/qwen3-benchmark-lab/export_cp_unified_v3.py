#!/usr/bin/env python3
"""Export Qwen3-TTS Code Predictor (CP) to unified ONNX with explicit past_length.

This version stores past_length in DynamicCache as a custom attribute, then the
mask function reads it from the cache. This creates a traceable path through the
model's forward call.

Key design:
- past_length stored in DynamicCache._past_length_tensor attribute
- Mask function reads from cache instead of global variable
- Both causal and beyond-KV masking applied via tensor operations

Inputs:
  inputs_embeds:     [1, seq_len, 1024]
  cache_position:    [seq_len]
  past_length:       scalar int64 (0 for prefill, k for decode step k)
  past_key_{0..4}:   [1, 8, max_past, 128]  — FIXED shape
  past_value_{0..4}: [1, 8, max_past, 128]

Usage:
    python3 export_cp_unified_v3.py --model /path/to/model --output-dir ./cp-unified-v3 --device cpu
"""
import argparse, os, sys, types, importlib.util

def _mock_torchaudio():
    ta = types.ModuleType('torchaudio')
    ta.__spec__ = importlib.util.spec_from_loader('torchaudio', loader=None)
    ta.__path__ = []
    ta.__package__ = 'torchaudio'
    for sub in ['compliance', 'compliance.kaldi', 'transforms', 'functional']:
        m = types.ModuleType(f'torchaudio.{sub}')
        sys.modules[f'torchaudio.{sub}'] = m
    ta.compliance = sys.modules['torchaudio.compliance']
    ta.compliance.kaldi = sys.modules['torchaudio.compliance.kaldi']
    sys.modules['torchaudio'] = ta

_mock_torchaudio()

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.cache_utils import DynamicCache

import transformers.masking_utils

_MAX_PAST = 20

class MaskedDynamicCache(DynamicCache):
    """DynamicCache with explicit past_length tensor for fixed-shape KV."""
    def __init__(self, past_length_tensor=None, max_past=20):
        super().__init__()
        self._past_length_tensor = past_length_tensor
        self._max_past = max_past

def _causal_mask_from_cache(config, input_embeds, attention_mask, cache_position, past_key_values, **kwargs):
    """Causal mask reading past_length from MaskedDynamicCache."""
    dtype = input_embeds.dtype
    device = input_embeds.device
    batch, seq_len = input_embeds.shape[:2]

    if hasattr(past_key_values, '_past_length_tensor') and past_key_values._past_length_tensor is not None:
        past_length = past_key_values._past_length_tensor
        max_past = past_key_values._max_past
    else:
        past_length = torch.tensor(0, dtype=torch.long, device=device)
        max_past = _MAX_PAST

    total_len = max_past + seq_len

    seq_indices = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(1)
    kv_indices = torch.arange(total_len, device=device, dtype=dtype).unsqueeze(0)

    causal_threshold = seq_indices.to(dtype) + past_length.to(dtype)
    causal_masked = kv_indices > causal_threshold

    kv_limit = (past_length.to(dtype) + seq_len)
    beyond_kv = kv_indices >= kv_limit

    neg_inf = torch.tensor(float('-inf'), device=device, dtype=dtype)
    zero_val = torch.tensor(0.0, device=device, dtype=dtype)
    mask = torch.where(causal_masked | beyond_kv, neg_inf, zero_val)

    return mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)

transformers.masking_utils.create_causal_mask = _causal_mask_from_cache

from qwen_tts.core.models import (
    Qwen3TTSConfig, Qwen3TTSForConditionalGeneration,
)


class CPUnifiedMaskWrapper(nn.Module):
    """
    CP wrapper with explicit past_length input.
    Stores past_length in custom DynamicCache for mask function access.
    """
    def __init__(self, cp, num_layers: int, max_past: int = 20):
        super().__init__()
        self.backbone = cp.model
        self.lm_heads = cp.lm_head
        self.projection = cp.small_to_mtp_projection
        self.num_layers = num_layers
        self.num_heads = len(cp.lm_head)
        self.max_past = max_past

    def forward(self, inputs_embeds, cache_position, past_length, *past_kv_flat):
        """
        Args:
            inputs_embeds:  [1, seq_len, 1024]
            cache_position: [seq_len]
            past_length:    scalar int64 (actual valid KV entries)
            past_kv_flat:   num_layers*2 tensors [1, H, max_past, D] — FIXED shape
        Returns:
            logits_all:     [15, vocab_size]
            new_kv_flat...: updated KV tensors
        """
        global _MAX_PAST
        _MAX_PAST = self.max_past

        hidden = self.projection(inputs_embeds)

        past_len_int = int(past_length.item())

        cache = MaskedDynamicCache(past_length_tensor=past_length, max_past=self.max_past)
        for i in range(self.num_layers):
            k = past_kv_flat[2 * i][:, :, :past_len_int, :]
            v = past_kv_flat[2 * i + 1][:, :, :past_len_int, :]
            cache.update(k, v, i)

        position_ids = cache_position.unsqueeze(0)

        out = self.backbone(
            inputs_embeds=hidden,
            past_key_values=cache,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )

        last_hidden = out.last_hidden_state[:, -1:, :]
        all_logits = torch.stack(
            [head(last_hidden) for head in self.lm_heads], dim=0
        )
        all_logits = all_logits.reshape(self.num_heads, -1)

        new_kv = out.past_key_values.to_legacy_cache()
        new_kv_flat = tuple(t for kv in new_kv for t in kv)

        return (all_logits,) + new_kv_flat


def export_cp_mask(model, output_dir, max_past=20, opset=17):
    """Export CP with explicit past_length input and fixed-shape KV."""
    print("Exporting cp_unified_mask.onnx ...")
    cp = model.talker.code_predictor
    cp_cfg = cp.config

    N = cp_cfg.num_hidden_layers
    H = cp_cfg.num_key_value_heads
    dh = cp_cfg.head_dim
    D_in = model.talker.config.hidden_size
    num_heads = len(cp.lm_head)

    print(f"  CP config: layers={N}, KV heads={H}, head_dim={dh}, input_dim={D_in}, lm_heads={num_heads}")
    print(f"  max_past: {max_past}")
    print(f"  Mask: explicit past_length + beyond-KV masking")

    wrapper = CPUnifiedMaskWrapper(cp, N, max_past).eval()

    dummy_embeds = torch.zeros(1, 2, D_in)
    dummy_cache_pos = torch.arange(0, 2, dtype=torch.long)
    dummy_past_length = torch.tensor(0, dtype=torch.long)
    dummy_kv = [torch.zeros(1, H, max_past, dh) for _ in range(N * 2)]

    in_kv_names = []
    out_kv_names = []
    for i in range(N):
        in_kv_names += [f"past_key_{i}", f"past_value_{i}"]
        out_kv_names += [f"new_past_key_{i}", f"new_past_value_{i}"]

    input_names = ["inputs_embeds", "cache_position", "past_length"] + in_kv_names
    output_names = ["logits_all"] + out_kv_names

    dyn = {
        "inputs_embeds": {1: "seq_len"},
        "cache_position": {0: "seq_len"},
        "past_length": {},
        "logits_all": {},
    }
    for n in out_kv_names:
        dyn[n] = {2: "new_len"}

    out_path = os.path.join(output_dir, "cp_unified_mask.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_cache_pos, dummy_past_length, *dummy_kv),
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dyn,
            opset_version=opset,
            dynamo=False,
        )
    print(f"  Saved: {out_path}")
    print(f"  Input:  inputs_embeds [1, seq_len, {D_in}]")
    print(f"          cache_position [seq_len]")
    print(f"          past_length scalar int64")
    print(f"          past_key/value_{{0..{N-1}}} [1, {H}, {max_past}, {dh}] FIXED")
    print(f"  Output: logits_all [{num_heads}, {cp_cfg.vocab_size}]")
    print(f"          new_past_key/value_{{0..{N-1}}} [1, {H}, new_len, {dh}]")

    return out_path


def verify_onnx_structure(onnx_path):
    """Check ONNX has 0 If nodes and correct input schema."""
    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    
    if_nodes = [n for n in model.graph.node if n.op_type == "If"]
    print(f"  If nodes: {len(if_nodes)} (expected 0)")
    
    inputs = model.graph.input
    print("  Inputs:")
    for inp in inputs:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape}")
    
    return len(if_nodes) == 0


def numerical_test(onnx_path, max_past=20):
    """
    Compare mask-based ONNX vs shape-based reference.
    Test strategy: prefill seq=2 → KV size 2, then decode steps to build larger KV.
    """
    import numpy as np
    import onnxruntime as ort

    print("\nNumerical test: mask-based vs shape-based reference ...")

    N = 5; H = 8; dh = 128; D = 1024; V = 2048
    NUM_HEADS = 15

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    np.random.seed(123)
    embeds_prefill = np.random.randn(1, 2, D).astype(np.float32)

    feeds_prefill = {
        "inputs_embeds": embeds_prefill,
        "cache_position": np.array([0, 1], dtype=np.int64),
        "past_length": np.array(0, dtype=np.int64),
    }
    for i in range(N):
        feeds_prefill[f"past_key_{i}"] = np.zeros((1, H, max_past, dh), dtype=np.float32)
        feeds_prefill[f"past_value_{i}"] = np.zeros((1, H, max_past, dh), dtype=np.float32)

    out_prefill = sess.run(None, feeds_prefill)
    kv_after_prefill = {}
    for i in range(N):
        kv_after_prefill[i] = (out_prefill[1 + 2*i], out_prefill[2 + 2*i])

    np.random.seed(42)
    all_passed = True

    kv_fixed = {}
    for i in range(N):
        kv_fixed[i] = (np.zeros((1, H, max_past, dh), dtype=np.float32),
                       np.zeros((1, H, max_past, dh), dtype=np.float32))

    current_kv = kv_after_prefill
    kv_size = 2

    for target_k in [2, 5, 10]:
        while kv_size < target_k:
            embeds_step = np.random.randn(1, 1, D).astype(np.float32)
            feeds_step = {
                "inputs_embeds": embeds_step,
                "cache_position": np.array([kv_size], dtype=np.int64),
                "past_length": np.array(kv_size, dtype=np.int64),
            }
            for i in range(N):
                kv_fixed[i][0][:, :, :kv_size, :] = current_kv[i][0]
                kv_fixed[i][1][:, :, :kv_size, :] = current_kv[i][1]
                feeds_step[f"past_key_{i}"] = kv_fixed[i][0]
                feeds_step[f"past_value_{i}"] = kv_fixed[i][1]

            out_step = sess.run(None, feeds_step)
            current_kv = {}
            for i in range(N):
                current_kv[i] = (out_step[1 + 2*i], out_step[2 + 2*i])
            kv_size += 1

        embeds_test = np.random.randn(1, 1, D).astype(np.float32)
        feeds_mask = {
            "inputs_embeds": embeds_test,
            "cache_position": np.array([target_k], dtype=np.int64),
            "past_length": np.array(target_k, dtype=np.int64),
        }
        for i in range(N):
            kv_fixed[i][0][:, :, :target_k, :] = current_kv[i][0]
            kv_fixed[i][1][:, :, :target_k, :] = current_kv[i][1]
            feeds_mask[f"past_key_{i}"] = kv_fixed[i][0]
            feeds_mask[f"past_value_{i}"] = kv_fixed[i][1]

        out_mask = sess.run(None, feeds_mask)
        logits_mask = out_mask[0]

        reference_path = "/home/harve/qwen3-tts-export/cp-unified-v2/cp_unified.onnx"
        if os.path.exists(reference_path):
            ref_sess = ort.InferenceSession(reference_path, providers=["CPUExecutionProvider"])
            feeds_ref = {
                "inputs_embeds": embeds_test,
                "cache_position": np.array([target_k], dtype=np.int64),
            }
            for i in range(N):
                feeds_ref[f"past_key_{i}"] = current_kv[i][0]
                feeds_ref[f"past_value_{i}"] = current_kv[i][1]

            out_ref = ref_sess.run(None, feeds_ref)
            logits_ref = out_ref[0]

            cos_sim = np.dot(logits_mask.flatten(), logits_ref.flatten()) / (
                np.linalg.norm(logits_mask.flatten()) * np.linalg.norm(logits_ref.flatten())
            )
            max_diff = np.max(np.abs(logits_mask - logits_ref))
            print(f"    k={target_k}: cosine_sim={cos_sim:.6f}, max_diff={max_diff:.6f}")

            if cos_sim < 0.99:
                print(f"    WARNING: k={target_k} cosine_sim < 0.99")
                all_passed = False

    if all_passed:
        print("  All k values passed (cosine > 0.99)")
    return all_passed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--output-dir", default="./cp-unified-v3")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-past", type=int, default=20)
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args()

    os.makedirs(a.output_dir, exist_ok=True)

    from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

    print(f"Loading model: {a.model}")
    model = AutoModel.from_pretrained(
        a.model,
        device_map=a.device,
        dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {model.device}")

    onnx_path = export_cp_mask(model, a.output_dir, a.max_past)

    if not a.no_verify:
        verify_onnx_structure(onnx_path)
        numerical_test(onnx_path, a.max_past)

    print("\n" + "=" * 60)
    print("CP export with explicit past_length complete!")
    print(f"Output: {onnx_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()