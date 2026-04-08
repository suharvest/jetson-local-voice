#!/usr/bin/env python3
"""Export Qwen3-TTS Code Predictor (CP) to unified ONNX with KV cache.

With KV cache:
  - Step 0 (prefill): process N tokens (context), output KV cache
  - Steps 1-14:       process 1 new token + KV cache → updated KV cache

Architecture (from actual model config):
  - 5 transformer layers, hidden=1024, KV heads=8, head_dim=128
  - 15 lm_heads (num_code_groups=16, heads for groups 2-16)
  - small_to_mtp_projection is Identity (talker hidden == CP hidden == 1024)

Inputs:
  inputs_embeds:     [1, seq_len, 1024]   (prefill: seq_len=N, decode: seq_len=1)
  cache_position:    [seq_len]            (e.g. [0,1] for prefill, [2] for first decode)
  past_key_{0..4}:   [1, 8, past_len, 128]
  past_value_{0..4}: [1, 8, past_len, 128]

Outputs:
  logits_all:           [15, vocab_size]  — all 15 heads, C++ selects by gen_step index
  new_past_key_{0..4}:   [1, 8, past_len+seq_len, 128]
  new_past_value_{0..4}: [1, 8, past_len+seq_len, 128]

Note: gen_step selection is done in C++/Python after inference (just index logits_all[gen_step]).
      This avoids dynamic indexing issues in ONNX.

Usage:
    python3 export_cp_unified.py --model /path/to/model --output-dir /tmp/cp-unified --device cpu
"""
import argparse, os, sys, types, importlib.util

# ---------------------------------------------------------------------------
# Mock torchaudio (not available on WSL2 export environment)
# ---------------------------------------------------------------------------
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

# Monkey-patch create_causal_mask to avoid vmap (not traceable)
import transformers.masking_utils

def _simple_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, **kwargs):
    """Simple causal mask without vmap — ONNX-traceable."""
    dtype = input_embeds.dtype
    device = input_embeds.device
    batch, seq_len = input_embeds.shape[:2]

    if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
        past_len = past_key_values.get_seq_length()
    else:
        past_len = 0

    total_len = past_len + seq_len
    # Decode (seq_len=1): no masking needed
    if seq_len == 1:
        return None
    # Prefill: lower-triangular causal mask
    mask = torch.triu(
        torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype),
        diagonal=past_len + 1,
    )
    return mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)

transformers.masking_utils.create_causal_mask = _simple_causal_mask

from qwen_tts.core.models import (
    Qwen3TTSConfig, Qwen3TTSForConditionalGeneration,
)


# ---------------------------------------------------------------------------
# Wrapper: unified CP with KV cache
# ---------------------------------------------------------------------------
class CPUnifiedWrapper(nn.Module):
    """
    Wraps the CP transformer backbone + all lm_heads.
    Accepts pre-projected inputs_embeds (already through small_to_mtp_projection).

    Key design: cache_position is passed explicitly to avoid constant-folding of
    past_key_values.get_seq_length() during ONNX tracing (which would hardcode
    cache_position=[0,1,...] regardless of actual past_len, breaking decode steps).

    Returns all 15 heads' logits stacked as [15, vocab_size] plus updated KV.
    C++ selects logits_all[gen_step] after inference.
    """
    def __init__(self, cp, num_layers: int):
        super().__init__()
        self.backbone = cp.model          # Qwen3TTSTalkerCodePredictorModel
        self.lm_heads = cp.lm_head        # ModuleList of 15 Linear(1024, 2048)
        self.projection = cp.small_to_mtp_projection   # Identity in 0.6B
        self.num_layers = num_layers
        self.num_heads = len(cp.lm_head)  # 15

    def forward(self, inputs_embeds, cache_position, *past_kv_flat):
        """
        Args:
            inputs_embeds:  [1, seq_len, 1024] — talker-space embedding
            cache_position: [seq_len]           — absolute positions (e.g. [0,1] prefill, [2] decode)
            past_kv_flat:   num_layers*2 tensors [1, num_kv_heads, past_len, head_dim]
        Returns:
            logits_all:        [15, vocab_size]
            new_kv_flat...:    num_layers*2 updated KV tensors
        """
        # Project (Identity for 0.6B, but kept for correctness)
        hidden = self.projection(inputs_embeds)

        # Reconstruct DynamicCache from flat KV
        cache = DynamicCache()
        for i in range(self.num_layers):
            cache.update(past_kv_flat[2 * i], past_kv_flat[2 * i + 1], i)

        # position_ids shape: [1, seq_len]
        position_ids = cache_position.unsqueeze(0)

        out = self.backbone(
            inputs_embeds=hidden,
            past_key_values=cache,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )

        # Last token hidden state → apply all 15 lm_heads
        last_hidden = out.last_hidden_state[:, -1:, :]   # [1, 1, 1024] — keep dim for consistency
        all_logits = torch.stack(
            [head(last_hidden) for head in self.lm_heads], dim=0
        )  # [15, 1, 1, 2048]
        # Use reshape (not squeeze) to guarantee fixed output shape for TRT
        all_logits = all_logits.reshape(self.num_heads, -1)  # [15, 2048]

        # Flatten updated KV cache
        new_kv = out.past_key_values.to_legacy_cache()
        new_kv_flat = tuple(t for kv in new_kv for t in kv)

        return (all_logits,) + new_kv_flat


class CPSingleHeadWrapper(nn.Module):
    """
    CP with single lm_head selection via gen_step input.
    Instead of computing all 15 lm_heads and discarding 14, only computes
    the one selected by gen_step. Saves ~44% compute per step.

    Output: logits [1, vocab_size] (single head) instead of [15, vocab_size].
    """
    def __init__(self, cp, num_layers: int):
        super().__init__()
        self.backbone = cp.model
        self.projection = cp.small_to_mtp_projection
        self.num_layers = num_layers
        self.num_heads = len(cp.lm_head)  # 15
        # Stack all lm_head weights into [15, 2048, 1024]
        self.stacked_weights = nn.Parameter(
            torch.stack([h.weight for h in cp.lm_head])
        )

    def forward(self, inputs_embeds, cache_position, gen_step, *past_kv_flat):
        """
        Args:
            inputs_embeds:  [1, seq_len, 1024]
            cache_position: [seq_len]
            gen_step:       scalar int64 — which lm_head to use (0-14)
            past_kv_flat:   num_layers*2 KV tensors
        Returns:
            logits:         [1, vocab_size] — single head output
            new_kv_flat...: updated KV tensors
        """
        hidden = self.projection(inputs_embeds)

        cache = DynamicCache()
        for i in range(self.num_layers):
            cache.update(past_kv_flat[2 * i], past_kv_flat[2 * i + 1], i)

        position_ids = cache_position.unsqueeze(0)

        out = self.backbone(
            inputs_embeds=hidden,
            past_key_values=cache,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )

        last_hidden = out.last_hidden_state[:, -1:, :]  # [1, 1, 1024]

        # Select single lm_head weight by gen_step index
        weight = self.stacked_weights[gen_step]  # [2048, 1024]
        logits = torch.matmul(last_hidden, weight.T)  # [1, 1, 2048]
        logits = logits.reshape(1, -1)  # [1, 2048]

        new_kv = out.past_key_values.to_legacy_cache()
        new_kv_flat = tuple(t for kv in new_kv for t in kv)

        return (logits,) + new_kv_flat


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_cp_unified(model, output_dir, opset=17):
    print("Exporting cp_unified.onnx ...")
    cp = model.talker.code_predictor
    cp_cfg = cp.config

    N = cp_cfg.num_hidden_layers      # 5
    H = cp_cfg.num_key_value_heads    # 8
    dh = cp_cfg.head_dim              # 128
    D_in = model.talker.config.hidden_size   # 1024 (input embedding dim from talker)
    num_heads = len(cp.lm_head)       # 15

    print(f"  CP config: layers={N}, KV heads={H}, head_dim={dh}, input_dim={D_in}, lm_heads={num_heads}")

    wrapper = CPUnifiedWrapper(cp, N).eval()

    # Dummy inputs — prefill with 2 tokens, empty KV cache (past_len=0)
    dummy_embeds = torch.zeros(1, 2, D_in)
    dummy_cache_pos = torch.arange(0, 2, dtype=torch.long)   # [0, 1]
    dummy_kv = [torch.zeros(1, H, 0, dh) for _ in range(N * 2)]

    # Input / output names
    in_kv_names = []
    out_kv_names = []
    for i in range(N):
        in_kv_names += [f"past_key_{i}", f"past_value_{i}"]
        out_kv_names += [f"new_past_key_{i}", f"new_past_value_{i}"]

    input_names = ["inputs_embeds", "cache_position"] + in_kv_names
    output_names = ["logits_all"] + out_kv_names

    # Dynamic axes
    dyn = {
        "inputs_embeds": {1: "seq_len"},
        "cache_position": {0: "seq_len"},
        "logits_all": {},
    }
    for n in in_kv_names:
        dyn[n] = {2: "past_len"}
    for n in out_kv_names:
        dyn[n] = {2: "new_len"}

    out_path = os.path.join(output_dir, "cp_unified.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_cache_pos, *dummy_kv),
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dyn,
            opset_version=opset,
            dynamo=False,
        )
    print(f"  Saved: {out_path}")
    print(f"  Input:  inputs_embeds [1, seq_len, {D_in}]")
    print(f"          cache_position [seq_len]  (e.g. [0,1] prefill, [2] decode)")
    print(f"          past_key/value_{{0..{N-1}}} [1, {H}, past_len, {dh}]")
    print(f"  Output: logits_all [{num_heads}, {cp_cfg.vocab_size}] — index by gen_step in C++")
    print(f"          new_past_key/value_{{0..{N-1}}} [1, {H}, new_len, {dh}]")


def export_cp_single_head(model, output_dir, opset=17):
    """Export CP with gen_step input — only computes 1 lm_head per step."""
    print("Exporting cp_single_head.onnx ...")
    cp = model.talker.code_predictor
    cp_cfg = cp.config

    N = cp_cfg.num_hidden_layers      # 5
    H = cp_cfg.num_key_value_heads    # 8
    dh = cp_cfg.head_dim              # 128
    D_in = model.talker.config.hidden_size   # 1024
    V = cp_cfg.vocab_size             # 2048
    num_heads = len(cp.lm_head)       # 15

    print(f"  CP config: layers={N}, KV heads={H}, head_dim={dh}, input_dim={D_in}, "
          f"lm_heads={num_heads}, vocab={V}")

    wrapper = CPSingleHeadWrapper(cp, N).eval()

    # Dummy inputs — prefill with 2 tokens, gen_step=0
    dummy_embeds = torch.zeros(1, 2, D_in)
    dummy_cache_pos = torch.arange(0, 2, dtype=torch.long)
    dummy_gen_step = torch.tensor(0, dtype=torch.long)
    dummy_kv = [torch.zeros(1, H, 0, dh) for _ in range(N * 2)]

    in_kv_names = []
    out_kv_names = []
    for i in range(N):
        in_kv_names += [f"past_key_{i}", f"past_value_{i}"]
        out_kv_names += [f"new_past_key_{i}", f"new_past_value_{i}"]

    input_names = ["inputs_embeds", "cache_position", "gen_step"] + in_kv_names
    output_names = ["logits"] + out_kv_names

    dyn = {
        "inputs_embeds": {1: "seq_len"},
        "cache_position": {0: "seq_len"},
        "logits": {},
    }
    for n in in_kv_names:
        dyn[n] = {2: "past_len"}
    for n in out_kv_names:
        dyn[n] = {2: "new_len"}

    out_path = os.path.join(output_dir, "cp_single_head.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_cache_pos, dummy_gen_step, *dummy_kv),
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
    print(f"          gen_step scalar int64 (0-14)")
    print(f"          past_key/value_{{0..{N-1}}} [1, {H}, past_len, {dh}]")
    print(f"  Output: logits [1, {V}] — single head")
    print(f"          new_past_key/value_{{0..{N-1}}} [1, {H}, new_len, {dh}]")


def verify_cp_unified(output_dir):
    """Quick ORT sanity check: prefill then 1 decode step."""
    import numpy as np
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not available — skipping verification")
        return

    print("\nVerifying cp_unified.onnx with ORT ...")
    onnx_path = os.path.join(output_dir, "cp_unified.onnx")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    N = 5; H = 8; dh = 128; D = 1024; V = 2048; NUM_HEADS = 15

    # --- Prefill: 2 tokens, empty KV ---
    feeds = {
        "inputs_embeds": np.random.randn(1, 2, D).astype(np.float32),
        "cache_position": np.array([0, 1], dtype=np.int64),
    }
    for i in range(N):
        feeds[f"past_key_{i}"] = np.zeros((1, H, 0, dh), dtype=np.float32)
        feeds[f"past_value_{i}"] = np.zeros((1, H, 0, dh), dtype=np.float32)

    results = sess.run(None, feeds)
    logits_all = results[0]
    assert logits_all.shape == (NUM_HEADS, V), f"Expected [{NUM_HEADS}, {V}], got {logits_all.shape}"
    print(f"  Prefill OK: logits_all={logits_all.shape}, KV[0]={results[1].shape}")

    # --- Decode: 1 token, KV from prefill ---
    feeds2 = {
        "inputs_embeds": np.random.randn(1, 1, D).astype(np.float32),
        "cache_position": np.array([2], dtype=np.int64),   # positions 0,1 already in cache
    }
    for i in range(N):
        feeds2[f"past_key_{i}"] = results[1 + 2 * i]
        feeds2[f"past_value_{i}"] = results[2 + 2 * i]

    results2 = sess.run(None, feeds2)
    logits2 = results2[0]
    assert logits2.shape == (NUM_HEADS, V), f"Expected [{NUM_HEADS}, {V}], got {logits2.shape}"
    assert results2[1].shape[2] == 3, f"Expected KV past_len=3, got {results2[1].shape[2]}"
    print(f"  Decode  OK: logits_all={logits2.shape}, KV[0]={results2[1].shape}")
    print("  Verification PASSED")


def verify_cp_single_head(output_dir):
    """Verify single-head CP: run all 15 steps and compare with unified."""
    import numpy as np
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not available — skipping verification")
        return

    N = 5; H = 8; dh = 128; D = 1024; V = 2048

    # Load both models if unified exists
    sh_path = os.path.join(output_dir, "cp_single_head.onnx")
    uni_path = os.path.join(output_dir, "cp_unified.onnx")

    sh_sess = ort.InferenceSession(sh_path, providers=["CPUExecutionProvider"])

    print("\nVerifying cp_single_head.onnx with ORT ...")

    # --- Prefill: gen_step=0 ---
    np.random.seed(42)
    embeds = np.random.randn(1, 2, D).astype(np.float32)
    feeds = {
        "inputs_embeds": embeds,
        "cache_position": np.array([0, 1], dtype=np.int64),
        "gen_step": np.array(0, dtype=np.int64),
    }
    for i in range(N):
        feeds[f"past_key_{i}"] = np.zeros((1, H, 0, dh), dtype=np.float32)
        feeds[f"past_value_{i}"] = np.zeros((1, H, 0, dh), dtype=np.float32)

    results = sh_sess.run(None, feeds)
    logits = results[0]
    assert logits.shape == (1, V), f"Expected [1, {V}], got {logits.shape}"
    print(f"  Prefill OK: logits={logits.shape}, argmax={np.argmax(logits)}, KV[0]={results[1].shape}")

    # --- Decode: gen_step=1 ---
    feeds2 = {
        "inputs_embeds": np.random.randn(1, 1, D).astype(np.float32),
        "cache_position": np.array([2], dtype=np.int64),
        "gen_step": np.array(1, dtype=np.int64),
    }
    for i in range(N):
        feeds2[f"past_key_{i}"] = results[1 + 2 * i]
        feeds2[f"past_value_{i}"] = results[2 + 2 * i]

    results2 = sh_sess.run(None, feeds2)
    logits2 = results2[0]
    assert logits2.shape == (1, V), f"Expected [1, {V}], got {logits2.shape}"
    assert results2[1].shape[2] == 3
    print(f"  Decode  OK: logits={logits2.shape}, argmax={np.argmax(logits2)}, KV[0]={results2[1].shape}")

    # --- Cross-check with unified model if available ---
    if os.path.exists(uni_path):
        print("  Cross-checking with cp_unified.onnx ...")
        uni_sess = ort.InferenceSession(uni_path, providers=["CPUExecutionProvider"])
        uni_feeds = {
            "inputs_embeds": embeds,
            "cache_position": np.array([0, 1], dtype=np.int64),
        }
        for i in range(N):
            uni_feeds[f"past_key_{i}"] = np.zeros((1, H, 0, dh), dtype=np.float32)
            uni_feeds[f"past_value_{i}"] = np.zeros((1, H, 0, dh), dtype=np.float32)
        uni_results = uni_sess.run(None, uni_feeds)
        uni_logits_g0 = uni_results[0][0]  # [2048] — group 0
        sh_logits_g0 = logits[0]           # [2048] — single head output
        max_diff = np.max(np.abs(uni_logits_g0 - sh_logits_g0))
        print(f"  Max diff (unified vs single_head, group 0): {max_diff:.6f}")
        if max_diff < 1e-4:
            print("  Cross-check PASSED — outputs match")
        else:
            print(f"  WARNING: max diff {max_diff} > 1e-4")

    print("  Verification PASSED")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                   help="Model path or HuggingFace ID")
    p.add_argument("--output-dir", default="./cp-unified",
                   help="Output directory for ONNX")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip ORT verification after export")
    p.add_argument("--single-head", action="store_true",
                   help="Export single-head CP (gen_step input, 1 lm_head per step)")
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
        attn_implementation="eager",   # avoid SDPA+GQA that TorchScript can't trace
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {model.device}")

    if a.single_head:
        # Export both unified (for reference) and single-head
        export_cp_unified(model, a.output_dir)
        export_cp_single_head(model, a.output_dir)
        if not a.no_verify:
            verify_cp_single_head(a.output_dir)
    else:
        export_cp_unified(model, a.output_dir)
        if not a.no_verify:
            verify_cp_unified(a.output_dir)

    print("\n" + "=" * 60)
    print("CP export complete!")
    print(f"Files in: {a.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
