#!/usr/bin/env python3
"""Generate real calibration data for INT8 talker engine build.

Uses the text_projection ORT model + tokenizer to produce realistic
inputs_embeds distributions that match actual TTS inference, replacing
the synthetic np.random.randn() data in build_talker_int8_engine.py.

Output: bench/calib/calib_batches.npz — N batches of (name, array) pairs
ready for use by CtypesCalibrator.get_batch().

Usage (on Jetson):
  python3 gen_calib_data.py
"""

import os, sys
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("QWEN3_MODEL_BASE", "/opt/models/qwen3-tts")
ONNX_DIR = os.path.join(MODEL_DIR, "onnx")
TOKENIZER_DIR = os.path.join(MODEL_DIR, "tokenizer")
OUT_DIR = os.environ.get("OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
N_BATCHES = int(os.environ.get("CALIB_BATCHES", "50"))

# Model constants
HIDDEN_DIM = 1024       # talker hidden dim
TEXT_EMBED_DIM = 2048   # text embedding dim (before projection)
TEXT_VOCAB = 151936
N_LAYERS = 28
N_HEADS = 8
HEAD_DIM = 128
MAX_PAST = 200

# Calibration texts covering English + Chinese, short + long
CALIB_TEXTS = [
    # English short
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Testing audio synthesis quality.",
    "A single sentence for calibration.",
    "One two three four five.",
    "Good morning, how are you today?",
    "This is a calibration data generator.",
    "Speech synthesis requires accurate data.",
    "Machine learning models need good data.",
    "The weather is nice today.",
    # English medium
    "This is a longer sentence that will produce more tokens for better calibration coverage.",
    "We need diverse text to capture the full range of embedding distributions.",
    "Different languages and sentence structures help improve the calibration quality.",
    "The transformer model uses multi-head attention with rotary position embeddings.",
    "Autoregressive decoding generates one token at a time using the previous context.",
    # Chinese
    "你好世界。",
    "今天天气很好，适合出门散步。",
    "这是一段用于校准的测试文本。",
    "语音合成需要准确的校准数据。",
    "不同长度的句子可以覆盖更多的嵌入分布。",
]


def load_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel
    tok = Tokenizer(models.BPE(
        os.path.join(TOKENIZER_DIR, "vocab.json"),
        os.path.join(TOKENIZER_DIR, "merges.txt"),
    ))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    return tok


def load_text_projection():
    """Load the text projection model (FP16 embed table + projection ONNX)."""
    import onnxruntime as ort

    # Load FP16 embedding table
    embed_path = os.path.join(ONNX_DIR, "text_embed_fp16.bin")
    fsize = os.path.getsize(embed_path)
    table = np.fromfile(embed_path, dtype=np.uint16)
    text_embed_vocab = fsize // 2 // TEXT_EMBED_DIM
    table = table.reshape(text_embed_vocab, TEXT_EMBED_DIM)

    # Load projection ONNX
    proj_path = os.path.join(ONNX_DIR, "text_projection_only.onnx")
    sess = ort.InferenceSession(proj_path, providers=['CPUExecutionProvider'])

    return table, sess


def text_to_embeds(text, tokenizer, embed_table, proj_sess):
    """Convert text to talker input embeddings [1, T, HIDDEN_DIM]."""
    token_ids = tokenizer.encode(text).ids
    T = len(token_ids)
    E = TEXT_EMBED_DIM

    # Gather FP16 embeddings: [T, E]
    embeds = np.zeros((T, E), dtype=np.float32)
    for t, tid in enumerate(token_ids):
        if 0 <= tid < embed_table.shape[0]:
            embeds[t] = embed_table[tid].view(np.float16).astype(np.float32)
        else:
            embeds[t] = embed_table[0].view(np.float16).astype(np.float32)  # fallback

    # Run projection ONNX: [1, T, E] → [1, T, HIDDEN_DIM]
    inp = embeds.reshape(1, T, E)
    out = proj_sess.run(None, {proj_sess.get_inputs()[0].name: inp})[0]
    return out  # [1, T, HIDDEN_DIM]


def generate_calib_data():
    print(f"Loading tokenizer from {TOKENIZER_DIR}...")
    tokenizer = load_tokenizer()

    print(f"Loading text projection from {ONNX_DIR}...")
    embed_table, proj_sess = load_text_projection()

    rng = np.random.RandomState(42)
    batches = []

    print(f"Generating {N_BATCHES} calibration batches from {len(CALIB_TEXTS)} texts...")

    for b in range(N_BATCHES):
        batch = {}

        # Pick a random text and compute its embeddings
        text = CALIB_TEXTS[b % len(CALIB_TEXTS)]
        text_embeds = text_to_embeds(text, tokenizer, embed_table, proj_sess)
        # text_embeds shape: [1, T, HIDDEN_DIM]
        T = text_embeds.shape[1]
        emb_scale = np.abs(text_embeds).mean()

        # inputs_embeds: use a single token embedding (decode step)
        # Pick a random position from the text
        pos = rng.randint(0, T)
        batch["inputs_embeds"] = text_embeds[:, pos:pos+1, :].copy().astype(np.float32)  # [1, 1, 1024]

        # attention_mask: causal mask for varying past lengths
        past_len = rng.randint(0, MAX_PAST + 1)  # 0..200
        batch["attention_mask"] = np.ones((1, past_len + 1), dtype=np.int64)
        batch["position_ids"] = np.array([[past_len]], dtype=np.int64)

        # KV cache: use realistic scale (match embedding statistics)
        for i in range(N_LAYERS):
            # Past KV: zeros for unused slots, small random for used slots
            # Scale based on observed embedding distribution
            kv_scale = emb_scale * 3.0  # KV activations ~same magnitude as residual stream
            pk = rng.randn(1, N_HEADS, past_len, HEAD_DIM).astype(np.float32) * kv_scale
            pv = rng.randn(1, N_HEADS, past_len, HEAD_DIM).astype(np.float32) * kv_scale
            batch[f"past_key_{i}"] = pk
            batch[f"past_value_{i}"] = pv

        batches.append(batch)

        if (b + 1) % 10 == 0:
            print(f"  Batch {b+1}/{N_BATCHES}: text='{text[:40]}...' T={T} "
                  f"past_len={past_len} emb_scale={emb_scale:.4f} kv_scale={kv_scale:.6f}")

    # Save
    out_path = os.path.join(OUT_DIR, "calib_batches.npz")
    save_dict = {}
    for b, batch in enumerate(batches):
        for name, arr in batch.items():
            save_dict[f"batch{b}_{name}"] = arr

    np.savez_compressed(out_path, **save_dict)
    print(f"\nSaved {len(batches)} batches to {out_path}")
    print(f"  File size: {os.path.getsize(out_path)/1024:.1f} KB")
    print(f"  Batch keys: {list(batches[0].keys())[:5]}...")

    # Print embedding statistics for verification
    all_embs = np.concatenate([b["inputs_embeds"].ravel() for b in batches])
    print(f"\nEmbedding statistics (for calibration verification):")
    print(f"  min={all_embs.min():.4f}  max={all_embs.max():.4f}")
    print(f"  mean={all_embs.mean():.6f}  std={all_embs.std():.4f}")
    print(f"  abs_mean={np.abs(all_embs).mean():.4f}")

    return out_path


if __name__ == "__main__":
    generate_calib_data()
