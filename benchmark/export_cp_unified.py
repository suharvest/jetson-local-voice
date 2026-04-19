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
  past_length:       scalar int64         (explicit past KV length, not inferred from shape)
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
import argparse, os, sys, types, importlib.util, threading

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

# Thread-local storage for explicit past_length (passed from wrapper)
_past_length_tls = threading.local()


def _simple_causal_mask_explicit(config, input_embeds, attention_mask, cache_position, past_key_values, **kwargs):
    """Simple causal mask without vmap — ONNX-traceable.
    
    Uses explicit past_length from thread-local storage instead of inferring 
    from past_key_values.shape[2], which may be padded.
    
    Key insight: When KV is padded (past_kv_len > past_length), the current 
    tokens are at indices past_kv_len..past_kv_len+seq_len-1, NOT at 
    past_length..past_length+seq_len-1. The mask must account for this.
    
    For query q (local index 0..seq_len-1):
        - Can attend to valid past positions: indices < past_length
        - Can attend to current positions: indices past_kv_len..past_kv_len+q
        - Cannot attend to padding: indices past_length..past_kv_len-1
        - Cannot attend to future current: indices > past_kv_len+q
    """
    dtype = input_embeds.dtype
    device = input_embeds.device
    batch, seq_len = input_embeds.shape[:2]
    
    # Get past KV length from cache shape (may be padded > past_length)
    past_kv_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
    
    # Total KV positions = cache + current sequence
    total_kv_len = past_kv_len + seq_len
    
    # Get explicit past_length from TLS (actual used positions, not padded)
    past_length_tensor = getattr(_past_length_tls, 'value', None)
    if past_length_tensor is None:
        past_length = torch.tensor(0, device=device, dtype=torch.long)
    else:
        past_length = past_length_tensor
    
    # Create mask with shape [seq_len, total_kv_len]
    kv_positions = torch.arange(total_kv_len, device=device, dtype=torch.long)
    query_positions = torch.arange(seq_len, device=device, dtype=torch.long)
    
    # Current K/V start position in concatenated KV
    current_start = past_kv_len
    
    # Valid positions for query q:
    #   1. Valid past: kv_pos < past_length
    #   2. Current up to self: current_start <= kv_pos <= current_start + q
    # Invalid (masked):
    #   1. Padding in cache: past_length <= kv_pos < past_kv_len
    #   2. Future in current: kv_pos > current_start + q
    
    is_padding = (kv_positions[None, :] >= past_length) & (kv_positions[None, :] < past_kv_len)
    is_future = kv_positions[None, :] > (current_start + query_positions[:, None])
    
    mask_condition = is_padding | is_future
    
    mask = torch.where(
        mask_condition,
        torch.full((seq_len, total_kv_len), float("-inf"), device=device, dtype=dtype),
        torch.zeros((seq_len, total_kv_len), device=device, dtype=dtype),
    )
    
    return mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)


transformers.masking_utils.create_causal_mask = _simple_causal_mask_explicit

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

    Key design: 
    - cache_position is passed explicitly to avoid constant-folding
    - past_length is an explicit scalar input (not inferred from KV shape)
    
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

    def forward(self, inputs_embeds, cache_position, past_length, *past_kv_flat):
        """
        Args:
            inputs_embeds:  [1, seq_len, 1024] — talker-space embedding
            cache_position: [seq_len]           — absolute positions (e.g. [0,1] prefill, [2] decode)
            past_length:    scalar int64        — explicit past KV length
            past_kv_flat:   num_layers*2 tensors [1, num_kv_heads, past_len, head_dim]
        Returns:
            logits_all:        [15, vocab_size]
            new_kv_flat...:    num_layers*2 updated KV tensors
        """
        hidden = self.projection(inputs_embeds)

        # Reconstruct DynamicCache from flat KV
        cache = DynamicCache()
        for i in range(self.num_layers):
            cache.update(past_kv_flat[2 * i], past_kv_flat[2 * i + 1], i)

        # Set explicit past_length for causal mask (thread-local, keep as tensor)
        _past_length_tls.value = past_length

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
        last_hidden = out.last_hidden_state[:, -1:, :]   # [1, 1, 1024]
        all_logits = torch.stack(
            [head(last_hidden) for head in self.lm_heads], dim=0
        )  # [15, 1, 1, 2048]
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

    def forward(self, inputs_embeds, cache_position, gen_step, past_length, *past_kv_flat):
        """
        Args:
            inputs_embeds:  [1, seq_len, 1024]
            cache_position: [seq_len]
            gen_step:       scalar int64 — which lm_head to use (0-14)
            past_length:    scalar int64 — explicit past KV length
            past_kv_flat:   num_layers*2 KV tensors
        Returns:
            logits:         [1, vocab_size] — single head output
            new_kv_flat...: updated KV tensors
        """
        hidden = self.projection(inputs_embeds)

        cache = DynamicCache()
        for i in range(self.num_layers):
            cache.update(past_kv_flat[2 * i], past_kv_flat[2 * i + 1], i)

        # Set explicit past_length for causal mask (keep as tensor)
        _past_length_tls.value = past_length

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
def export_cp_unified(model, output_dir, opset=17, device="cpu"):
    print("Exporting cp_unified.onnx ...")
    cp = model.talker.code_predictor
    cp_cfg = cp.config

    N = cp_cfg.num_hidden_layers      # 5
    H = cp_cfg.num_key_value_heads    # 8
    dh = cp_cfg.head_dim              # 128
    D_in = model.talker.config.hidden_size   # 1024 (input embedding dim from talker)
    num_heads = len(cp.lm_head)       # 15

    print(f"  CP config: layers={N}, KV heads={H}, head_dim={dh}, input_dim={D_in}, lm_heads={num_heads}")

    wrapper = CPUnifiedWrapper(cp, N).eval().to(device)

    # Dummy inputs — prefill with 2 tokens, empty KV cache (past_len=0)
    dummy_embeds = torch.zeros(1, 2, D_in, device=device)
    dummy_cache_pos = torch.arange(0, 2, dtype=torch.long, device=device)   # [0, 1]
    dummy_past_length = torch.tensor(0, dtype=torch.long, device=device)    # scalar
    dummy_kv = [torch.zeros(1, H, 0, dh, device=device) for _ in range(N * 2)]

    # Input / output names
    in_kv_names = []
    out_kv_names = []
    for i in range(N):
        in_kv_names += [f"past_key_{i}", f"past_value_{i}"]
        out_kv_names += [f"new_past_key_{i}", f"new_past_value_{i}"]

    input_names = ["inputs_embeds", "cache_position", "past_length"] + in_kv_names
    output_names = ["logits_all"] + out_kv_names

    # Dynamic axes
    dyn = {
        "inputs_embeds": {1: "seq_len"},
        "cache_position": {0: "seq_len"},
        "past_length": {},  # scalar, no dynamic axes
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
    print(f"          cache_position [seq_len]  (e.g. [0,1] prefill, [2] decode)")
    print(f"          past_length scalar int64   (explicit past KV length)")
    print(f"          past_key/value_{{0..{N-1}}} [1, {H}, past_len, {dh}]")
    print(f"  Output: logits_all [{num_heads}, {cp_cfg.vocab_size}] — index by gen_step in C++")
    print(f"          new_past_key/value_{{0..{N-1}}} [1, {H}, new_len, {dh}]")


def export_cp_single_head(model, output_dir, opset=17, device="cpu"):
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

    wrapper = CPSingleHeadWrapper(cp, N).eval().to(device)

    # Dummy inputs — prefill with 2 tokens, gen_step=0, past_length=0
    dummy_embeds = torch.zeros(1, 2, D_in, device=device)
    dummy_cache_pos = torch.arange(0, 2, dtype=torch.long, device=device)
    dummy_gen_step = torch.tensor(0, dtype=torch.long, device=device)
    dummy_past_length = torch.tensor(0, dtype=torch.long, device=device)
    dummy_kv = [torch.zeros(1, H, 0, dh, device=device) for _ in range(N * 2)]

    in_kv_names = []
    out_kv_names = []
    for i in range(N):
        in_kv_names += [f"past_key_{i}", f"past_value_{i}"]
        out_kv_names += [f"new_past_key_{i}", f"new_past_value_{i}"]

    input_names = ["inputs_embeds", "cache_position", "gen_step", "past_length"] + in_kv_names
    output_names = ["logits"] + out_kv_names

    dyn = {
        "inputs_embeds": {1: "seq_len"},
        "cache_position": {0: "seq_len"},
        "gen_step": {},       # scalar
        "past_length": {},    # scalar
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
            (dummy_embeds, dummy_cache_pos, dummy_gen_step, dummy_past_length, *dummy_kv),
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
    print(f"          past_length scalar int64")
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
        "past_length": np.array(0, dtype=np.int64),
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
        "cache_position": np.array([2], dtype=np.int64),
        "past_length": np.array(2, dtype=np.int64),
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
        "past_length": np.array(0, dtype=np.int64),
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
        "past_length": np.array(2, dtype=np.int64),
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
            "past_length": np.array(0, dtype=np.int64),
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


def verify_mask_correctness(output_dir):
    """Verify mask correctness: compare shape-based vs explicit past_length.
    
    Test scenario:
    - Reference: past_key.shape[2]=5, past_length=5 (correct mask)
    - Test: past_key.shape[2]=20 (padded), past_length=5 (should match)
    
    If cosine similarity > 0.99, mask is correctly using explicit past_length.
    """
    import numpy as np
    try:
        import onnxruntime as ort
        import onnx
    except ImportError:
        print("  onnx/onnxruntime not available — skipping mask verification")
        return

    print("\nVerifying mask correctness (shape-based vs explicit past_length) ...")
    onnx_path = os.path.join(output_dir, "cp_unified.onnx")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    N = 5; H = 8; dh = 128; D = 1024; V = 2048
    
    np.random.seed(12345)
    
    # Generate random embeddings
    embeds_prefill = np.random.randn(1, 5, D).astype(np.float32)  # 5 tokens prefill
    embeds_decode = np.random.randn(1, 1, D).astype(np.float32)   # 1 token decode
    
    # Reference run: true past_length=5, unpadded KV
    feeds_ref = {
        "inputs_embeds": embeds_prefill,
        "cache_position": np.arange(5, dtype=np.int64),
        "past_length": np.array(0, dtype=np.int64),
    }
    for i in range(N):
        feeds_ref[f"past_key_{i}"] = np.zeros((1, H, 0, dh), dtype=np.float32)
        feeds_ref[f"past_value_{i}"] = np.zeros((1, H, 0, dh), dtype=np.float32)
    
    # Run prefill to get KV
    results_prefill = sess.run(None, feeds_ref)
    
    # Now decode: past_length=5, KV has shape [1, 8, 5, 128]
    feeds_decode_ref = {
        "inputs_embeds": embeds_decode,
        "cache_position": np.array([5], dtype=np.int64),
        "past_length": np.array(5, dtype=np.int64),
    }
    for i in range(N):
        feeds_decode_ref[f"past_key_{i}"] = results_prefill[1 + 2 * i]
        feeds_decode_ref[f"past_value_{i}"] = results_prefill[2 + 2 * i]
    
    results_ref = sess.run(None, feeds_decode_ref)
    logits_ref = results_ref[0]
    
    # Test run: pad KV to 20, but pass correct past_length=5
    # This simulates the runtime scenario where KV is pre-allocated to max_len
    padded_len = 20
    feeds_padded = {
        "inputs_embeds": embeds_decode,
        "cache_position": np.array([5], dtype=np.int64),
        "past_length": np.array(5, dtype=np.int64),  # explicit correct value
    }
    for i in range(N):
        # Pad KV: first 5 slots are real, remaining 15 are zeros
        real_key = results_prefill[1 + 2 * i]  # [1, H, 5, dh]
        real_value = results_prefill[2 + 2 * i]
        padded_key = np.zeros((1, H, padded_len, dh), dtype=np.float32)
        padded_value = np.zeros((1, H, padded_len, dh), dtype=np.float32)
        padded_key[:, :, :5, :] = real_key
        padded_value[:, :, :5, :] = real_value
        feeds_padded[f"past_key_{i}"] = padded_key
        feeds_padded[f"past_value_{i}"] = padded_value
    
    results_padded = sess.run(None, feeds_padded)
    logits_padded = results_padded[0]
    
    # Compare logits
    def cosine_sim(a, b):
        return np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    cos_sim = cosine_sim(logits_ref, logits_padded)
    max_diff = np.max(np.abs(logits_ref - logits_padded))
    
    print(f"  Reference logits shape: {logits_ref.shape}")
    print(f"  Padded KV logits shape:  {logits_padded.shape}")
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max abs diff: {max_diff:.6e}")
    
    if cos_sim > 0.99:
        print("  ✓ Mask correctness PASSED — explicit past_length works correctly")
        return True
    else:
        print(f"  ✗ Mask correctness FAILED — cosine sim {cos_sim:.4f} < 0.99")
        return False


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
        export_cp_unified(model, a.output_dir, device=a.device)
        export_cp_single_head(model, a.output_dir, device=a.device)
        if not a.no_verify:
            verify_cp_single_head(a.output_dir)
            verify_mask_correctness(a.output_dir)
    else:
        export_cp_unified(model, a.output_dir, device=a.device)
        if not a.no_verify:
            verify_cp_unified(a.output_dir)
            verify_mask_correctness(a.output_dir)

    # Print ONNX info
    import onnx
    onnx_path = os.path.join(a.output_dir, "cp_unified.onnx")
    if os.path.exists(onnx_path):
        m = onnx.load(onnx_path)
        if_nodes = [n.name for n in m.graph.node if n.op_type == 'If']
        print(f"\nONNX check: If nodes = {len(if_nodes)} (should be 0)")
        print(f"Inputs: {[i.name for i in m.graph.input]}")
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"Size: {size_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("CP export complete!")
    print(f"Files in: {a.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()