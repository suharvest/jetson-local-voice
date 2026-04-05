#!/usr/bin/env python3
"""Qwen3-TTS — Sherpa-style ONNX pipeline.

Uses fresh exports: same-source prefill/decode (compatible KV cache),
stateless code predictor, no If nodes, no attention_mask/position_ids.

Usage:
    python3 tts_sherpa.py --text "Hello world." --lang english
    python3 tts_sherpa.py --text "你好世界" --lang chinese
"""
import argparse, json, os, time, wave
import numpy as np
import onnxruntime as ort

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "/tmp/qwen3-tts-bench/model")
SHERPA_DIR = os.environ.get("TTS_SHERPA_DIR", "/tmp/qwen3-sherpa")

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
with open(f"{SHERPA_DIR}/config.json") as f:
    CFG = json.load(f)

D = CFG["hidden_size"]
N_LAYERS = CFG["num_hidden_layers"]
N_GROUPS = CFG["num_code_groups"]
TALKER_VOCAB = CFG["vocab_size"]
CP_VOCAB = 2048

TTS_BOS = CFG["tts_bos_token_id"]
TTS_EOS = CFG["tts_eos_token_id"]
TTS_PAD = CFG["tts_pad_token_id"]
CODEC_BOS = CFG["codec_bos_id"]
CODEC_EOS = CFG["codec_eos_token_id"]
CODEC_PAD = CFG["codec_pad_id"]
CODEC_NOTHINK = CFG["codec_nothink_id"]
CODEC_THINK_BOS = CFG["codec_think_bos_id"]
CODEC_THINK_EOS = CFG["codec_think_eos_id"]
LANG_IDS = CFG["codec_language_id"]

# ---------------------------------------------------------------------------
# ONNX sessions
# ---------------------------------------------------------------------------
print("Loading ONNX models...")
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

text_proj_s = ort.InferenceSession(f"{SHERPA_DIR}/text_project.onnx", so, providers=PROVIDERS)
codec_emb_s = ort.InferenceSession(f"{SHERPA_DIR}/codec_embed.onnx", so, providers=PROVIDERS)
cp_emb_s = ort.InferenceSession(f"{SHERPA_DIR}/code_predictor_embed.onnx", so, providers=PROVIDERS)
cp_s = ort.InferenceSession(f"{SHERPA_DIR}/code_predictor.onnx", so, providers=PROVIDERS)
prefill_s = ort.InferenceSession(f"{SHERPA_DIR}/talker_prefill.onnx", so, providers=PROVIDERS)
decode_s = ort.InferenceSession(f"{SHERPA_DIR}/talker_decode.onnx", so, providers=PROVIDERS)

# Vocoder: use existing elbruno model (already validated)
voc_path = f"{MODEL_DIR}/vocoder.onnx"
if os.path.exists(voc_path):
    voc_s = ort.InferenceSession(voc_path, so, providers=PROVIDERS)
    print(f"  Vocoder loaded from {voc_path}")
else:
    voc_s = None
    print(f"  Vocoder not found at {voc_path}")

# Output name maps
pf_outs = [o.name for o in prefill_s.get_outputs()]
dc_ins = [i.name for i in decode_s.get_inputs()]
dc_outs = [o.name for o in decode_s.get_outputs()]

# BPE tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
tok_dir = f"{MODEL_DIR}/tokenizer"
tok = Tokenizer(BPE(f"{tok_dir}/vocab.json", f"{tok_dir}/merges.txt"))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
def tokenize(text): return tok.encode(text).ids

print("Models loaded.")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def text_proj(ids):
    return text_proj_s.run(None, {"input_ids": np.array([ids], dtype=np.int64)})[0]

def codec_emb(ids):
    return codec_emb_s.run(None, {"token_ids": np.array([ids], dtype=np.int64)})[0]

def cp_embed(token_id, layer_idx):
    return cp_emb_s.run(None, {
        "token_id": np.array([[token_id]], dtype=np.int64),
        "layer_idx": np.array(layer_idx, dtype=np.int64),
    })[0]

def cp_predict(context, gen_step):
    return cp_s.run(None, {
        "context": context.astype(np.float32),
        "gen_step": np.array([gen_step], dtype=np.int64),
    })[0]

def sample(logits, vocab_size, k=50, t=0.9, suppress_eos=False, eos_id=None):
    l = logits.flatten()[:vocab_size].astype(np.float64)
    if suppress_eos and eos_id is not None:
        l[eos_id] = -1e9
    l = l / max(t, 1e-6)
    if 0 < k < len(l):
        threshold = np.partition(l, -k)[-k]
        l[l < threshold] = -1e9
    l = l - l.max()
    p = np.exp(l)
    p = p / p.sum()
    return int(np.random.choice(len(p), p=p))


# ---------------------------------------------------------------------------
# Build prefill embeddings (sherpa-onnx streaming pattern)
# ---------------------------------------------------------------------------
def build_prefill(text, lang="english"):
    """Build prefill embedding sequence.

    Layout:
      [role_prefix(3)] + [tts_pad+nothink, tts_pad+think_bos, tts_pad+think_eos,
       tts_bos+codec_pad] + [text_body[0]+codec_bos]
    Trailing text: [text_body[1:] + tts_eos] — fed one per decode step
    """
    lang_id = LANG_IDS.get(lang, 2050)

    # Role prefix: <|im_start|>assistant\n
    role_ids = [151644, 77091, 198]
    role_emb = text_proj(role_ids)  # [1, 3, D]

    # Special text embeddings
    special_emb = text_proj([TTS_BOS, TTS_EOS, TTS_PAD])
    tts_bos_e = special_emb[:, 0:1, :]
    tts_eos_e = special_emb[:, 1:2, :]
    tts_pad_e = special_emb[:, 2:3, :]

    # Codec embeddings for prefix tokens
    codec_prefix = codec_emb([CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS, CODEC_PAD, CODEC_BOS])
    # [1, 5, D]: [nothink, think_bos, think_eos, pad, bos]

    # Text body
    text_ids = tokenize(text)
    body_emb = text_proj(text_ids)  # [1, N, D]
    n_body = len(text_ids)

    codec_pad_e = codec_emb([CODEC_PAD])  # [1, 1, D]

    # Build prefill: role(3) + codec_prefix(3) + tts_bos+pad(1) + body[0]+bos(1) = 8
    prefill_len = 3 + 3 + 1 + 1
    prefill = np.zeros((1, prefill_len, D), dtype=np.float32)

    pos = 0
    # Role prefix
    prefill[0, pos:pos+3, :] = role_emb[0, :3, :]
    pos += 3

    # tts_pad + codec_{nothink, think_bos, think_eos}
    for i in range(3):
        prefill[0, pos, :] = tts_pad_e[0, 0, :] + codec_prefix[0, i, :]
        pos += 1

    # tts_bos + codec_pad
    prefill[0, pos, :] = tts_bos_e[0, 0, :] + codec_prefix[0, 3, :]
    pos += 1

    # body[0] + codec_bos
    prefill[0, pos, :] = body_emb[0, 0, :] + codec_prefix[0, 4, :]
    pos += 1

    # Trailing text: body[1:] + tts_eos (one per decode step)
    trailing = []
    for i in range(1, n_body):
        trailing.append(body_emb[0, i, :] + codec_pad_e[0, 0, :])
    trailing.append(tts_eos_e[0, 0, :] + codec_pad_e[0, 0, :])

    return prefill, trailing, tts_pad_e, codec_pad_e


# ---------------------------------------------------------------------------
# Synthesize
# ---------------------------------------------------------------------------
def synthesize(text, lang="english", output="/tmp/tts_sherpa.wav", max_frames=200, seed=42):
    np.random.seed(seed)
    print(f"\nSynth: \"{text}\" ({lang})")
    t_total = time.perf_counter()

    prefill_emb, trailing_text, tts_pad_e, codec_pad_e = build_prefill(text, lang)

    # --- Prefill ---
    t0 = time.perf_counter()
    pf_result = prefill_s.run(None, {"inputs_embeds": prefill_emb})
    pf_map = dict(zip(pf_outs, pf_result))
    logits = pf_map["logits"]
    last_hidden = pf_map["last_hidden"]
    # Build KV cache dict for decode
    kv_cache = {k: v for k, v in pf_map.items() if k.startswith("past_")}
    pf_ms = (time.perf_counter() - t0) * 1000
    print(f"  Prefill: {pf_ms:.0f}ms ({prefill_emb.shape[1]} tokens)")

    # --- Decode loop ---
    all_codes = []
    dt_times = []
    ct_times = []

    for step in range(max_frames):
        # Sample primary code from talker logits
        primary_code = sample(logits[0, -1, :], TALKER_VOCAB,
                              suppress_eos=(step < 2), eos_id=CODEC_EOS)
        if primary_code == CODEC_EOS:
            print(f"  EOS at step {step}")
            break

        # --- Code predictor: 15 residual codes ---
        t_cp = time.perf_counter()
        primary_e = codec_emb([primary_code])  # [1, 1, D]
        lh_last = last_hidden[:, -1:, :]  # [1, 1, D]
        cp_ctx = np.concatenate([lh_last, primary_e], axis=1)  # [1, 2, D]
        codec_sum = primary_e[0, 0, :].copy()

        frame_codes = [primary_code]
        for j in range(N_GROUPS - 1):
            cp_logits = cp_predict(cp_ctx, j)
            rc = sample(cp_logits, CP_VOCAB)
            frame_codes.append(rc)
            re = cp_embed(rc, j)  # [1, 1, D]
            cp_ctx = np.concatenate([cp_ctx, re], axis=1)
            codec_sum += re[0, 0, :]
        ct_times.append(time.perf_counter() - t_cp)
        all_codes.append(frame_codes)

        # --- Next talker input ---
        if step < len(trailing_text):
            text_e = trailing_text[step]
        else:
            text_e = tts_pad_e[0, 0, :] + codec_pad_e[0, 0, :]
        next_emb = (codec_sum + text_e).reshape(1, 1, D).astype(np.float32)

        # --- Talker decode ---
        t_d = time.perf_counter()
        feeds = {"inputs_embeds": next_emb}
        feeds.update(kv_cache)
        dc_result = decode_s.run(None, feeds)
        dc_map = dict(zip(dc_outs, dc_result))
        logits = dc_map["logits"]
        last_hidden = dc_map["last_hidden"]
        # Update KV cache: new_past_key_i -> past_key_i
        kv_cache = {}
        for k, v in dc_map.items():
            if k.startswith("new_past_"):
                kv_cache[k.replace("new_", "")] = v
        dt_times.append(time.perf_counter() - t_d)

        if (step + 1) % 10 == 0:
            print(f"  Frame {step + 1}")

    n = len(all_codes)
    if n == 0:
        print("  No frames generated!")
        return

    dur = n / 12.5  # 12.5 Hz codec rate
    da = np.mean(dt_times) * 1000
    ca = np.mean(ct_times) * 1000

    # --- Vocoder ---
    if voc_s:
        t0 = time.perf_counter()
        codes_arr = np.array(all_codes, dtype=np.int64)  # [T, 16]
        # vocoder expects [1, 16, T]
        vi = voc_s.get_inputs()[0].name
        wav = voc_s.run(None, {vi: codes_arr.T[np.newaxis, :, :]})[0].flatten()
        tv = (time.perf_counter() - t0) * 1000

        with wave.open(output, "w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
            wf.writeframes((wav * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
    else:
        tv = 0
        # Save codes for external vocoder
        np.save("/tmp/tts_codes.npy", np.array(all_codes, dtype=np.int64))
        print(f"  Codes saved to /tmp/tts_codes.npy ({n}x16)")

    total = (time.perf_counter() - t_total) * 1000
    print(f"\n  === TIMING ({n} frames, {dur:.1f}s audio) ===")
    print(f"  Prefill:     {pf_ms:>7.0f}ms")
    print(f"  Talker/step: {da:>7.1f}ms")
    print(f"  CP/step:     {ca:>7.1f}ms")
    print(f"  Per-step:    {da+ca:>7.1f}ms  RTF={(da+ca)/80:.2f}")
    if voc_s:
        print(f"  Vocoder:     {tv:>7.0f}ms")
    print(f"  Total:       {total:>7.0f}ms  RTF={total/1000/dur:.2f}")
    if voc_s:
        print(f"  Saved: {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="Hello, welcome to the voice synthesis system.")
    p.add_argument("--lang", default="english", choices=list(LANG_IDS.keys()))
    p.add_argument("--output", default="/tmp/tts_sherpa.wav")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    synthesize(a.text, a.lang, a.output, a.max_frames, a.seed)
