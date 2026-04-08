#!/usr/bin/env python3
"""
Qwen3-TTS Reference Data Dumper
================================
Runs the official Qwen3-TTS pipeline using low-level PyTorch model APIs and
dumps ALL intermediate tensors for cross-implementation comparison (C++ TRT,
RK3576 RKNN, etc.).

For each test case, saves:
  {output_dir}/{case_idx}/
    token_ids.npy          — int64 text token IDs
    text_embeds.npy        — [1, N, 1024] float32 text embeddings
    prefill_logits.npy     — [1, N, 3072] float32 full talker prefill logits
    prefill_hidden.npy     — [1, N, 1024] float32 talker prefill hidden states
    primary_codes.npy      — [n_frames] int32 primary codec code per frame
    cp_codes.npy           — [n_frames, 15] int32 CP residual codes per frame
    codec_sums.npy         — [n_frames, 1024] float32 codec_sum embed per frame
    audio.wav              — synthesized audio
    metadata.json          — text, lang, n_frames, sample_rate, seed, model info

  {output_dir}/{case_idx}/frames/  (first 5 frames only)
    frame_{i}_talker_logits.npy    — [1, 1, 3072]
    frame_{i}_talker_hidden.npy    — [1, 1, 1024]
    frame_{i}_cp_step_logits.npy   — [15, 2048] all 15 CP step logits
    frame_{i}_input_embed.npy      — [1, 1, 1024] talker input for this frame

Usage:
    python tts_dump_reference.py \\
        --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        --output-dir ./tts_ref_dump \\
        --device cuda \\
        --seed 42 \\
        --cases 3

Run on WSL2 (harve@100.73.210.80) with:
    source /tmp/qwen3-tts-env/bin/activate
    cd /tmp/Qwen3-TTS
    python /path/to/tts_dump_reference.py
"""

import argparse
import importlib.util
import json
import os
import struct
import sys
import types
import wave
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Add Qwen3-TTS source to path BEFORE any other imports that might need it
# ---------------------------------------------------------------------------
sys.path.insert(0, "/tmp/Qwen3-TTS")

# ---------------------------------------------------------------------------
# Mock torchaudio — not available in export/benchmark environments
# ---------------------------------------------------------------------------
def _mock_torchaudio():
    ta = types.ModuleType("torchaudio")
    ta.__spec__ = importlib.util.spec_from_loader("torchaudio", loader=None)
    ta.__path__ = []
    ta.__package__ = "torchaudio"
    ta.__version__ = "0.0.0"
    for sub in ["compliance", "compliance.kaldi", "transforms", "functional",
                "_extension", "_extension.utils"]:
        m = types.ModuleType(f"torchaudio.{sub}")
        m.__spec__ = importlib.util.spec_from_loader(f"torchaudio.{sub}", loader=None)
        m.__path__ = []
        sys.modules[f"torchaudio.{sub}"] = m
    ta.compliance = sys.modules["torchaudio.compliance"]
    ta.compliance.kaldi = sys.modules["torchaudio.compliance.kaldi"]
    sys.modules["torchaudio"] = ta

_mock_torchaudio()

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache

# ---------------------------------------------------------------------------
# Patch create_causal_mask to avoid vmap (not needed at inference, and
# causes issues in some environments; exact same patch used in export scripts)
# ---------------------------------------------------------------------------
import transformers.masking_utils

def _simple_causal_mask(config, input_embeds, attention_mask, cache_position,
                         past_key_values, **kwargs):
    """Simple causal mask: lower-triangular, no vmap dependency."""
    dtype = input_embeds.dtype
    device = input_embeds.device
    batch, seq_len = input_embeds.shape[:2]

    if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
        past_len = past_key_values.get_seq_length()
    else:
        past_len = 0

    total_len = past_len + seq_len
    if seq_len == 1:
        return None  # decode step: no causal mask needed
    # Prefill: strict lower-triangular
    mask = torch.triu(
        torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype),
        diagonal=past_len + 1,
    )
    return mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)

transformers.masking_utils.create_causal_mask = _simple_causal_mask

# ---------------------------------------------------------------------------
# Model registration (must happen after path insert and patch)
# ---------------------------------------------------------------------------
from qwen_tts.core.models import (  # noqa: E402
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
)

AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

# ---------------------------------------------------------------------------
# Constants (from ONNX pipeline scripts — verified correct)
# ---------------------------------------------------------------------------
CODEC_EOS       = 2150
CODEC_THINK     = 2154
CODEC_THINK_BOS = 2156
CODEC_THINK_EOS = 2157
CODEC_PAD       = 2148
CODEC_BOS       = 2149
TTS_BOS         = 151672
TTS_EOS         = 151673
TTS_PAD         = 151671
IM_START        = 151644
ASSISTANT_ID    = 77091   # "assistant" token
NEWLINE_ID      = 198     # "\n" token

LANG_IDS = {
    "chinese":  2055,
    "english":  2050,
    "japanese": 2058,
    "korean":   2064,
}

SAMPLE_RATE     = 24000
CODEC_HZ        = 12.5     # frames per second
NUM_CP_STEPS    = 15       # residual code groups (CP autoregressive steps)

# Test cases matching tts_eval.py (text, language)
ALL_TEST_CASES = [
    ("你好",                                        "chinese"),
    ("今天天气真不错",                              "chinese"),
    ("今天天气真不错，我们一起去公园散步吧。",      "chinese"),
    ("欢迎使用语音合成系统",                        "chinese"),
    ("Hello, how are you today?",                   "english"),
]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_top_k(logits: torch.Tensor, top_k: int = 50, temperature: float = 0.9) -> int:
    """Sample one token from logits with top-k + temperature."""
    logits = logits.float().flatten() / temperature
    # Keep only top-k
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k)
    min_val = values[-1]
    logits = logits.masked_fill(logits < min_val, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ---------------------------------------------------------------------------
# WAV writer (no scipy/soundfile dependency)
# ---------------------------------------------------------------------------
def write_wav(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    """Write mono float32 audio [-1, 1] to a 16-bit PCM WAV file."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    n = len(pcm)
    with open(path, "wb") as f:
        data_size = n * 2
        f.write(struct.pack("<4sI4s", b"RIFF", 36 + data_size, b"WAVE"))
        f.write(struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(struct.pack("<4sI", b"data", data_size))
        f.write(pcm.tobytes())


# ---------------------------------------------------------------------------
# Input embedding construction
# Matches the official SDK non_streaming_mode generate() path.
#
# Layout (all element-wise additions, then concatenation):
#   role_prefix  = text_proj([IM_START, assistant_id, newline_id])  → [1, 3, 1024]
#   codec_prefix = [THINK, THINK_BOS, lang_id, THINK_EOS, PAD, BOS] → [1, 6, 1024]
#                  (codec_embed lookup, NOT text_proj)
#   tts_part     = [tts_pad×4, tts_bos]  (text_proj)                → [1, 5, 1024]
#   prefix       = tts_part ⊕ codec_prefix[:-1]                     → [1, 5, 1024]
#   embed        = cat([role_prefix, prefix])                        → [1, 8, 1024]
#   text_e       = text_proj(text_ids)                               → [1, N, 1024]
#   text_eos_e   = cat([text_e, text_proj([TTS_EOS])])               → [1, N+1, 1024]
#   codec_pad_n  = codec_embed(PAD) × (N+1)                          → [1, N+1, 1024]
#   text_block   = text_eos_e ⊕ codec_pad_n                          → [1, N+1, 1024]
#   final        = text_proj([TTS_PAD]) ⊕ codec_embed(BOS)           → [1, 1, 1024]
#   total        = cat([embed, text_block, final])                   → [1, 9+N, 1024]
# ---------------------------------------------------------------------------
def build_input_embeds(
    text_ids: List[int],
    lang: str,
    talker,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct the full prefill embedding tensor.

    Returns:
        input_embeds: [1, T, 1024]  (bfloat16, on device)
        tts_pad_embed: [1, 1, 1024] — reused as decode step base
    """
    lang_id = LANG_IDS.get(lang, LANG_IDS["english"])

    def text_proj(ids: List[int]) -> torch.Tensor:
        """IDs → text projection (embed_tokens + text_projection MLP)."""
        id_t = torch.tensor(ids, dtype=torch.long, device=device)
        # embed_tokens is on the talker model
        raw = talker.model.embed_tokens(id_t)          # [N, 2048] typically
        return talker.text_projection(raw).unsqueeze(0)  # [1, N, 1024]

    def codec_emb(token_id: int) -> torch.Tensor:
        """Codec token ID → codec embedding [1, 1, 1024]."""
        id_t = torch.tensor([token_id], dtype=torch.long, device=device)
        return talker.model.codec_embedding(id_t).unsqueeze(0)  # [1, 1, 1024]

    # 1. Role prefix
    role_e = text_proj([IM_START, ASSISTANT_ID, NEWLINE_ID])   # [1, 3, 1024]

    # 2. Codec prefix (6 tokens: think, think_bos, lang, think_eos, pad, bos)
    codec_prefix = torch.cat([
        codec_emb(CODEC_THINK),
        codec_emb(CODEC_THINK_BOS),
        codec_emb(lang_id),
        codec_emb(CODEC_THINK_EOS),
        codec_emb(CODEC_PAD),
        codec_emb(CODEC_BOS),
    ], dim=1)  # [1, 6, 1024]

    # 3. TTS tokens (text_proj space)
    tts_pad_e = text_proj([TTS_PAD])   # [1, 1, 1024]  — saved for decode reuse
    tts_bos_e = text_proj([TTS_BOS])   # [1, 1, 1024]
    tts_eos_e = text_proj([TTS_EOS])   # [1, 1, 1024]

    # 4. prefix block = [tts_pad×4, tts_bos] ADD codec_prefix[:-1] (skip BOS)
    tts_part = torch.cat([tts_pad_e.expand(1, 4, -1), tts_bos_e], dim=1)  # [1, 5, 1024]
    prefix = tts_part + codec_prefix[:, :-1, :]   # [1, 5, 1024]

    embed = torch.cat([role_e, prefix], dim=1)    # [1, 8, 1024]

    # 5. Text block: text_ids + TTS_EOS, ADD codec_pad × (N+1)
    text_e   = text_proj(text_ids)                                  # [1, N, 1024]
    text_eos = torch.cat([text_e, tts_eos_e], dim=1)                # [1, N+1, 1024]
    N1 = text_eos.shape[1]
    codec_pad_block = codec_emb(CODEC_PAD).expand(1, N1, -1)        # [1, N+1, 1024]
    text_block = text_eos + codec_pad_block                         # [1, N+1, 1024]

    # 6. Final token: tts_pad ADD codec_bos
    final = tts_pad_e + codec_emb(CODEC_BOS)                        # [1, 1, 1024]

    input_embeds = torch.cat([embed, text_block, final], dim=1)     # [1, T, 1024]
    return input_embeds, tts_pad_e


# ---------------------------------------------------------------------------
# Core dump logic for a single test case
# ---------------------------------------------------------------------------
@torch.no_grad()
def dump_test_case(
    case_idx: int,
    text: str,
    lang: str,
    model: Qwen3TTSForConditionalGeneration,
    tokenizer,
    output_dir: str,
    device: torch.device,
    seed: int,
    top_k: int,
    temperature: float,
    max_frames: int,
    dump_n_frames: int,
):
    """Run one test case and dump all intermediates."""
    print(f"\n{'='*70}")
    print(f"Case {case_idx}: '{text}' ({lang})")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    case_dir = os.path.join(output_dir, str(case_idx))
    frames_dir = os.path.join(case_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    talker = model.talker

    # ------------------------------------------------------------------
    # 1. Tokenize text
    # ------------------------------------------------------------------
    # Use the tokenizer to get text token IDs (no special tokens added here;
    # the special tokens are handled manually in build_input_embeds)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"  Token IDs: {token_ids} (len={len(token_ids)})")

    np.save(
        os.path.join(case_dir, "token_ids.npy"),
        np.array(token_ids, dtype=np.int64),
    )

    # ------------------------------------------------------------------
    # 2. Build input embeddings
    # ------------------------------------------------------------------
    input_embeds, tts_pad_embed = build_input_embeds(
        token_ids, lang, talker, device
    )
    # input_embeds: [1, T, 1024]  bfloat16
    print(f"  Input embeds: {input_embeds.shape}")

    # Extract text_embeds separately (the text portion only, before mixing)
    # This is the output of text_proj(text_token_ids) for the actual text tokens
    with torch.no_grad():
        text_id_t = torch.tensor(token_ids, dtype=torch.long, device=device)
        raw_text = talker.model.embed_tokens(text_id_t)
        text_embeds = talker.text_projection(raw_text).unsqueeze(0)  # [1, N, 1024]

    np.save(
        os.path.join(case_dir, "text_embeds.npy"),
        text_embeds.cpu().float().numpy(),
    )
    print(f"  text_embeds: {text_embeds.shape}")

    # ------------------------------------------------------------------
    # 3. Talker prefill
    # ------------------------------------------------------------------
    T = input_embeds.shape[1]
    print(f"  Running talker prefill (seq_len={T})...")

    prefill_out = talker.model(
        inputs_embeds=input_embeds,
        use_cache=True,
        return_dict=True,
    )
    # prefill_hidden: [1, T, 1024]  bfloat16
    prefill_hidden = prefill_out.last_hidden_state
    # prefill_logits via codec_head: [1, T, 3072]
    prefill_logits = talker.codec_head(prefill_hidden)

    np.save(
        os.path.join(case_dir, "prefill_logits.npy"),
        prefill_logits.cpu().float().numpy(),
    )
    np.save(
        os.path.join(case_dir, "prefill_hidden.npy"),
        prefill_hidden.cpu().float().numpy(),
    )
    print(f"  prefill_logits: {prefill_logits.shape}")
    print(f"  prefill_hidden: {prefill_hidden.shape}")

    # KV cache from prefill — used for decode steps
    talker_kv = prefill_out.past_key_values  # DynamicCache

    # Working logits = last position's logits
    current_logits = prefill_logits[:, -1:, :]   # [1, 1, 3072]
    # Working hidden = last position's hidden (for CP input)
    current_hidden = prefill_hidden[:, -1:, :]   # [1, 1, 1024]

    # ------------------------------------------------------------------
    # 4. Decode loop
    # ------------------------------------------------------------------
    all_primary_codes: List[int]     = []
    all_cp_codes: List[List[int]]   = []
    all_codec_sums: List[np.ndarray] = []

    for frame_idx in range(max_frames):
        # ---- 4a. Sample primary code from talker logits ----
        primary_code = sample_top_k(
            current_logits[0, -1, :], top_k=top_k, temperature=temperature
        )
        if primary_code == CODEC_EOS:
            print(f"  EOS at frame {frame_idx}")
            break

        all_primary_codes.append(primary_code)

        # ---- 4b. Codec embedding for primary code ----
        pc_id = torch.tensor([primary_code], dtype=torch.long, device=device)
        primary_codec_embed = talker.model.codec_embedding(pc_id).unsqueeze(0)  # [1,1,1024]

        # ---- 4c. Build CP input: [talker_hidden, primary_codec_embed] ----
        # CP input at step 0 is concatenation of last talker hidden + primary codec embed
        # shape: [1, 2, 1024]
        cp_input = torch.cat([current_hidden, primary_codec_embed], dim=1)

        # ---- 4d. CP autoregressive loop (15 steps) ----
        # Use low-level components directly so we can:
        #   a) select the correct lm_head per step (cp_step index)
        #   b) capture per-step logits before sampling
        cp_kv = DynamicCache()
        residual_codes: List[int] = []
        cp_step_logits_list: List[np.ndarray] = []

        # CP backbone is: talker.code_predictor.model (transformer)
        # CP lm_heads:    talker.code_predictor.lm_head (ModuleList of 15)
        # CP projection:  talker.code_predictor.small_to_mtp_projection (Identity for 0.6B)
        cp_backbone = talker.code_predictor.model
        cp_lm_heads = talker.code_predictor.lm_head      # ModuleList[15]
        cp_projection = talker.code_predictor.small_to_mtp_projection

        cp_embed_in = cp_projection(cp_input)  # [1, 2, 1024] (Identity for 0.6B)
        cp_cache_pos = torch.arange(cp_embed_in.shape[1], dtype=torch.long, device=device)

        for cp_step in range(NUM_CP_STEPS):
            cp_backbone_out = cp_backbone(
                inputs_embeds=cp_embed_in,
                past_key_values=cp_kv,
                cache_position=cp_cache_pos,
                position_ids=cp_cache_pos.unsqueeze(0),
                use_cache=True,
                return_dict=True,
            )
            cp_kv = cp_backbone_out.past_key_values

            # Select the correct lm_head for this step
            cp_last_hidden = cp_backbone_out.last_hidden_state[:, -1:, :]  # [1,1,1024]
            step_logits = cp_lm_heads[cp_step](cp_last_hidden)             # [1,1,2048]
            step_logits_2d = step_logits.squeeze(1)                        # [1,2048]

            cp_step_logits_list.append(step_logits_2d.cpu().float().numpy())

            # Sample residual code
            rc = sample_top_k(
                step_logits_2d[0], top_k=top_k, temperature=temperature
            )
            residual_codes.append(rc)

            # Next CP input: CP codec embedding for this residual code
            # Each CP step has its own embedding table: code_predictor.model.codec_embedding[cp_step]
            rc_id = torch.tensor([rc], dtype=torch.long, device=device)
            cp_codec_embed = talker.code_predictor.model.codec_embedding[cp_step](rc_id)
            # cp_codec_embed: [1, 1024]
            cp_embed_in = cp_projection(cp_codec_embed.unsqueeze(0).unsqueeze(0))  # [1,1,1024]
            cp_cache_pos = torch.tensor(
                [cp_cache_pos[-1].item() + 1], dtype=torch.long, device=device
            )

        all_cp_codes.append(residual_codes)

        # ---- 4e. codec_sum = primary_embed + sum(residual_embeds) ----
        codec_sum = primary_codec_embed.clone()  # [1, 1, 1024]
        for cp_step, rc in enumerate(residual_codes):
            rc_id = torch.tensor([rc], dtype=torch.long, device=device)
            rc_embed = talker.code_predictor.model.codec_embedding[cp_step](rc_id)
            codec_sum = codec_sum + rc_embed.unsqueeze(0).unsqueeze(0)

        all_codec_sums.append(codec_sum.squeeze(0).squeeze(0).cpu().float().numpy())

        # ---- 4f. Save per-frame data (first dump_n_frames only) ----
        if frame_idx < dump_n_frames:
            # Input embed to talker for this frame (what was fed to decode)
            # For frame 0: this is tts_pad + codec_sum (but that's the NEXT frame's input)
            # We save the actual input that produced current_logits for this frame.
            # For frame 0, it came from the prefill's last hidden; for frame>0, from decode.
            # Instead, save what WILL be fed as input for the NEXT talker step.
            next_talker_input = tts_pad_embed + codec_sum   # [1, 1, 1024]

            np.save(
                os.path.join(frames_dir, f"frame_{frame_idx}_talker_logits.npy"),
                current_logits.cpu().float().numpy(),
            )
            np.save(
                os.path.join(frames_dir, f"frame_{frame_idx}_talker_hidden.npy"),
                current_hidden.cpu().float().numpy(),
            )
            np.save(
                os.path.join(frames_dir, f"frame_{frame_idx}_cp_step_logits.npy"),
                np.array(cp_step_logits_list, dtype=np.float32),  # [15, 2048]
            )
            np.save(
                os.path.join(frames_dir, f"frame_{frame_idx}_input_embed.npy"),
                next_talker_input.cpu().float().numpy(),
            )

        # ---- 4g. Talker decode step ----
        # Next input: tts_pad_embed + codec_sum (sum of all 16 codec embeds)
        next_embed = tts_pad_embed + codec_sum   # [1, 1, 1024]

        decode_out = talker.model(
            inputs_embeds=next_embed,
            past_key_values=talker_kv,
            use_cache=True,
            return_dict=True,
        )
        talker_kv = decode_out.past_key_values

        current_hidden = decode_out.last_hidden_state  # [1, 1, 1024]
        current_logits = talker.codec_head(current_hidden)  # [1, 1, 3072]

        if (frame_idx + 1) % 10 == 0:
            print(f"  Frame {frame_idx + 1}, primary_code={primary_code}")

    # ------------------------------------------------------------------
    # 5. Save per-case arrays
    # ------------------------------------------------------------------
    n_frames = len(all_primary_codes)
    print(f"  Generated {n_frames} frames ({n_frames / CODEC_HZ:.2f}s audio)")

    if n_frames == 0:
        print("  WARNING: No frames generated!")
        return

    np.save(
        os.path.join(case_dir, "primary_codes.npy"),
        np.array(all_primary_codes, dtype=np.int32),
    )
    np.save(
        os.path.join(case_dir, "cp_codes.npy"),
        np.array(all_cp_codes, dtype=np.int32),
    )
    np.save(
        os.path.join(case_dir, "codec_sums.npy"),
        np.array(all_codec_sums, dtype=np.float32),
    )

    # ------------------------------------------------------------------
    # 6. Vocoder → audio
    # ------------------------------------------------------------------
    print("  Running vocoder...")
    # Build codec token matrix: [1, 16, n_frames]
    # Row 0: primary_codes, rows 1-15: cp_codes per frame
    codes_matrix = np.zeros((1, 16, n_frames), dtype=np.int64)
    codes_matrix[0, 0, :] = np.array(all_primary_codes, dtype=np.int64)
    for fi in range(n_frames):
        for ci in range(NUM_CP_STEPS):
            codes_matrix[0, ci + 1, fi] = all_cp_codes[fi][ci]

    codes_tensor = torch.from_numpy(codes_matrix).to(device)  # [1, 16, n_frames]

    # Use model's speech_tokenizer / vocoder
    # The official model has model.speech_tokenizer which wraps the vocoder.
    # We call it via model.speech_tokenizer.decode() or directly via the vocoder.
    try:
        # Preferred: use model.speech_tokenizer.decode for official audio
        codes_for_decode = [{"audio_codes": codes_tensor[0]}]  # list of dicts
        wavs, fs = model.speech_tokenizer.decode(codes_for_decode)
        audio = wavs[0] if isinstance(wavs, list) else wavs
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().float().numpy()
        audio = audio.flatten()
        actual_sr = fs
    except Exception as e:
        print(f"  speech_tokenizer.decode failed ({e}), trying vocoder directly...")
        try:
            # Fallback: call model.vocoder directly
            # vocoder input: [1, 16, n_frames] int64
            audio_out = model.vocoder(codes_tensor)
            if isinstance(audio_out, (list, tuple)):
                audio_out = audio_out[0]
            audio = audio_out.cpu().float().numpy().flatten()
            actual_sr = SAMPLE_RATE
        except Exception as e2:
            print(f"  Vocoder also failed ({e2}), skipping audio save")
            audio = None
            actual_sr = SAMPLE_RATE

    if audio is not None:
        wav_path = os.path.join(case_dir, "audio.wav")
        write_wav(wav_path, audio, sr=actual_sr)
        print(f"  Audio: {wav_path} ({len(audio)/actual_sr:.2f}s)")
    else:
        wav_path = None

    # ------------------------------------------------------------------
    # 7. Metadata
    # ------------------------------------------------------------------
    talker_cfg = talker.config
    cp_cfg = talker.code_predictor.config
    meta = {
        "case_idx": case_idx,
        "text": text,
        "language": lang,
        "n_frames": n_frames,
        "audio_duration_s": round(n_frames / CODEC_HZ, 3),
        "sample_rate": int(actual_sr),
        "seed": seed,
        "top_k": top_k,
        "temperature": temperature,
        "primary_codes_sample": all_primary_codes[:10],
        "model_config": {
            "talker_hidden_size": talker_cfg.hidden_size,
            "talker_num_layers": talker_cfg.num_hidden_layers,
            "talker_num_kv_heads": talker_cfg.num_key_value_heads,
            "cp_hidden_size": cp_cfg.hidden_size,
            "cp_num_layers": cp_cfg.num_hidden_layers,
            "cp_num_kv_heads": cp_cfg.num_key_value_heads,
            "cp_vocab_size": cp_cfg.vocab_size,
            "codec_hz": CODEC_HZ,
        },
        "files": {
            "token_ids": "token_ids.npy",
            "text_embeds": "text_embeds.npy (float32 [1,N,1024])",
            "prefill_logits": "prefill_logits.npy (float32 [1,T,3072])",
            "prefill_hidden": "prefill_hidden.npy (float32 [1,T,1024])",
            "primary_codes": "primary_codes.npy (int32 [n_frames])",
            "cp_codes": "cp_codes.npy (int32 [n_frames, 15])",
            "codec_sums": "codec_sums.npy (float32 [n_frames, 1024])",
            "audio": "audio.wav",
            "frames": {
                "frame_i_talker_logits": "[1,1,3072] float32 — logits at frame i",
                "frame_i_talker_hidden": "[1,1,1024] float32 — hidden at frame i",
                "frame_i_cp_step_logits": "[15,2048] float32 — all 15 CP logits",
                "frame_i_input_embed": "[1,1,1024] float32 — next talker input after frame i",
            },
        },
    }

    with open(os.path.join(case_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {case_dir}/")
    print(f"    token_ids.npy       [{len(token_ids)}]")
    print(f"    text_embeds.npy     {text_embeds.shape}")
    print(f"    prefill_logits.npy  {prefill_logits.shape}")
    print(f"    prefill_hidden.npy  {prefill_hidden.shape}")
    print(f"    primary_codes.npy   [{n_frames}]")
    print(f"    cp_codes.npy        [{n_frames}, 15]")
    print(f"    codec_sums.npy      [{n_frames}, 1024]")
    print(f"    frames/             ({min(n_frames, dump_n_frames)} frames × 4 files)")
    if wav_path:
        print(f"    audio.wav           ({n_frames / CODEC_HZ:.2f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS reference data dumper — runs official pipeline "
                    "and saves all intermediate tensors for cross-implementation comparison."
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Model path or HuggingFace ID",
    )
    parser.add_argument(
        "--output-dir", default="./tts_ref_dump",
        help="Root output directory (per-case subdirs created inside)",
    )
    parser.add_argument(
        "--device", default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (cuda recommended for speed)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--cases", type=int, default=3,
        help="Number of test cases to run (first N from built-in list)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=200,
        help="Maximum codec frames per utterance (safety limit)",
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k for sampling",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--dump-frames", type=int, default=5,
        help="Number of frames to dump detailed per-frame data for",
    )
    args = parser.parse_args()

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    print(f"  dtype=bfloat16, device={args.device}, attn=eager")
    model = AutoModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,          # BF16 required — FP16 overflows in Qwen3 QK^T
        device_map=args.device,
        attn_implementation="eager",   # SDPA+GQA cannot be traced/profiled cleanly
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded. Device: {next(model.parameters()).device}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # ------------------------------------------------------------------
    # Print model architecture summary
    # ------------------------------------------------------------------
    talker = model.talker
    cp = talker.code_predictor
    print(f"\nModel architecture:")
    print(f"  Talker: {talker.config.num_hidden_layers}L "
          f"h={talker.config.hidden_size} "
          f"kv_heads={talker.config.num_key_value_heads}")
    print(f"  CP:     {cp.config.num_hidden_layers}L "
          f"h={cp.config.hidden_size} "
          f"kv_heads={cp.config.num_key_value_heads} "
          f"vocab={cp.config.vocab_size} "
          f"n_heads={len(cp.lm_head)}")

    # Verify CP codec_embedding structure
    if hasattr(cp.model, "codec_embedding"):
        ce = cp.model.codec_embedding
        print(f"  CP codec_embedding: {type(ce).__name__} "
              f"(len={len(ce) if hasattr(ce, '__len__') else 'N/A'})")
    else:
        print("  WARNING: cp.model.codec_embedding not found — check model version")

    # ------------------------------------------------------------------
    # Run test cases
    # ------------------------------------------------------------------
    test_cases = ALL_TEST_CASES[: args.cases]
    print(f"\nRunning {len(test_cases)} test cases → {args.output_dir}/")

    for case_idx, (text, lang) in enumerate(test_cases):
        dump_test_case(
            case_idx=case_idx,
            text=text,
            lang=lang,
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            device=device,
            seed=args.seed,
            top_k=args.top_k,
            temperature=args.temperature,
            max_frames=args.max_frames,
            dump_n_frames=args.dump_frames,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Done. Reference data saved to: {args.output_dir}/")
    print(f"  Structure per case:")
    print(f"    {{case_idx}}/token_ids.npy        — int64 text token IDs")
    print(f"    {{case_idx}}/text_embeds.npy      — float32 [1,N,1024]")
    print(f"    {{case_idx}}/prefill_logits.npy   — float32 [1,T,3072]")
    print(f"    {{case_idx}}/prefill_hidden.npy   — float32 [1,T,1024]")
    print(f"    {{case_idx}}/primary_codes.npy    — int32 [n_frames]")
    print(f"    {{case_idx}}/cp_codes.npy         — int32 [n_frames,15]")
    print(f"    {{case_idx}}/codec_sums.npy       — float32 [n_frames,1024]")
    print(f"    {{case_idx}}/audio.wav")
    print(f"    {{case_idx}}/frames/frame_i_*.npy — first {args.dump_frames} frames")
    print(f"    {{case_idx}}/metadata.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
