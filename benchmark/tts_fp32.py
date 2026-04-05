#!/usr/bin/env python3
"""Qwen3-TTS 0.6B end-to-end TTS — verified working pipeline.
All FP32 models (elbruno) + PyTorch-exported embeddings + sampling."""
import argparse, os, sys, time
import numpy as np, onnxruntime as ort, wave

FP32 = "/tmp/qwen3-tts-bench/model"
CEMB = "/tmp/correct-emb"
CPU = ["CPUExecutionProvider"]
CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Special token IDs
CODEC_EOS = 2150
CODEC_THINK = 2154
CODEC_THINK_BOS = 2156
CODEC_THINK_EOS = 2157
CODEC_PAD = 2148
CODEC_BOS = 2149
TTS_BOS = 151672
TTS_EOS = 151673
TTS_PAD = 151671
IM_START = 151644
IM_END = 151645
LANG_IDS = {"chinese": 2055, "english": 2050, "japanese": 2058, "korean": 2064}

# ---- Load models ----
print("Loading models...")
o = ort.SessionOptions()
o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
o.intra_op_num_threads = 6

prefill_s = ort.InferenceSession(f"{FP32}/talker_prefill.onnx", sess_options=o, providers=CPU)
decode_s = ort.InferenceSession(f"{FP32}/talker_decode.onnx", sess_options=o, providers=CPU)
cp_s = ort.InferenceSession(f"{FP32}/code_predictor.onnx", sess_options=o, providers=CPU)
voc_s = ort.InferenceSession(f"{FP32}/vocoder.onnx", providers=CUDA)

pf_on = [x.name for x in prefill_s.get_outputs()]
dc_on = [x.name for x in decode_s.get_outputs()]
cp_on = [x.name for x in cp_s.get_outputs()]
vi = voc_s.get_inputs()[0].name

# ---- Load embeddings ----
print("Loading embeddings...")
talker_ce = np.load(f"{CEMB}/talker_codec_embedding.npy")   # [3072, 1024]
cp_ces = [np.load(f"{CEMB}/cp_codec_embedding_{i}.npy") for i in range(15)]  # [2048, 1024] x 15
# Text embedding from elbruno (same model), projection from PyTorch export
text_emb_w = np.load(f"{FP32}/embeddings/text_embedding.npy")  # [151936, 2048]
fc1_w = np.load(f"{CEMB}/tp_fc1_weight.npy")   # [2048, 2048]
fc1_b = np.load(f"{CEMB}/tp_fc1_bias.npy")     # [2048]
fc2_w = np.load(f"{CEMB}/tp_fc2_weight.npy")   # [1024, 2048]
fc2_b = np.load(f"{CEMB}/tp_fc2_bias.npy")     # [1024]

# ---- Tokenizer ----
print("Loading tokenizer...")
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    tok = Tokenizer(BPE(f"{FP32}/tokenizer/vocab.json", f"{FP32}/tokenizer/merges.txt"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    def tokenize(text):
        return tok.encode(text).ids
    print("  BPE tokenizer loaded")
except Exception as e:
    print(f"  Tokenizer failed: {e}"); sys.exit(1)

# ---- Helpers ----
def text_proj(ids):
    """Text token IDs → 1024-dim talker space via text_projection."""
    embs = []
    for tid in ids:
        x = text_emb_w[tid].astype(np.float32)
        h = x @ fc1_w.T + fc1_b
        h = h * (1 / (1 + np.exp(-h)))  # SiLU
        embs.append(h @ fc2_w.T + fc2_b)
    return np.array(embs).reshape(1, len(ids), 1024)

def codec_emb(tid):
    return talker_ce[tid].reshape(1, 1, 1024).astype(np.float32)

def cp_emb(tid, group):
    return cp_ces[group][tid].reshape(1, 1, 1024).astype(np.float32)

def sample(logits, top_k=50, temp=0.9):
    l = logits.flatten().astype(np.float64) / temp
    ti = np.argpartition(l, -top_k)[-top_k:]
    m = np.full_like(l, -np.inf); m[ti] = l[ti]
    e = np.exp(m - np.max(m)); p = e / e.sum()
    return int(np.random.choice(len(p), p=p))

# ---- Build input embeddings (matching official SDK) ----
def build_input_embeds(text, lang="english"):
    """Construct talker input matching official SDK generate() non-streaming mode.

    Structure (element-wise ADD, not concat):
      [role(3)] + [tts_pad*4+tts_bos ⊕ codec_prefix[:-1]] + [text+eos ⊕ codec_pad*N] + [tts_pad ⊕ codec_bos]

    Total = 3 + 5 + (text_len+1) + 1 tokens
    """
    lang_id = LANG_IDS.get(lang, LANG_IDS["english"])

    # Build token IDs with special tokens as literal IDs
    # <|im_start|>=151644  assistant=77091  \n=198  <|im_end|>=151645
    text_token_ids = tokenize(text)
    role_ids = [IM_START, 77091, 198]        # <|im_start|>assistant\n
    text_ids = text_token_ids                 # actual text content

    # TTS special embeddings (text_projection space)
    tts_bos_e = text_proj([TTS_BOS])
    tts_eos_e = text_proj([TTS_EOS])
    tts_pad_e = text_proj([TTS_PAD])

    # 1. Role prefix: text_proj(role_ids)  → [1, 3, 1024]
    role_e = text_proj(role_ids)

    # 2. Codec prefix: [think, think_bos, lang, think_eos, pad, bos] → [1, 6, 1024]
    codec_in = np.concatenate([
        codec_emb(CODEC_THINK), codec_emb(CODEC_THINK_BOS),
        codec_emb(lang_id), codec_emb(CODEC_THINK_EOS),
        codec_emb(CODEC_PAD), codec_emb(CODEC_BOS),
    ], axis=1)  # [1, 6, 1024]

    # 3. TTS prefix + codec ADD (official: tts_pad*(N-2) + tts_bos added with codec[:-1])
    tts_part = np.concatenate([
        np.tile(tts_pad_e, (1, 4, 1)),  # 4 x tts_pad
        tts_bos_e,                       # 1 x tts_bos
    ], axis=1)  # [1, 5, 1024]
    prefix = tts_part + codec_in[:, :-1, :]  # ADD, [1, 5, 1024]

    embed = np.concatenate([role_e, prefix], axis=1)  # [1, 8, 1024]

    # 4. Non-streaming text: text+eos ADDED with codec_pad
    # Official: text_proj(text_ids) + tts_eos ⊕ codec_pad * (text_len+1)
    text_e = text_proj(text_ids)   # [1, N, 1024]
    text_eos = np.concatenate([text_e, tts_eos_e], axis=1)  # [1, N+1, 1024]
    codec_pad_n = np.tile(codec_emb(CODEC_PAD), (1, len(text_ids) + 1, 1))  # [1, N+1, 1024]
    text_combined = text_eos + codec_pad_n  # ADD

    # 5. Final: tts_pad + codec_bos
    final = tts_pad_e + codec_emb(CODEC_BOS)  # ADD, [1, 1, 1024]

    embed = np.concatenate([embed, text_combined, final], axis=1)
    print(f"  Text tokens: {len(text_ids)}, embed: {embed.shape}")
    return embed.astype(np.float32), tts_pad_e

# ---- Main synthesis ----
def synthesize(text, lang="english", output="/tmp/tts_out.wav", max_frames=60, seed=None):
    if seed is not None:
        np.random.seed(seed)

    print(f"\nSynthesizing: \"{text}\" ({lang})")
    t_total = time.perf_counter()

    # Build input
    ie, tts_pad_e = build_input_embeds(text, lang)
    print(f"  Input: {ie.shape}")

    # Prefill
    t0 = time.perf_counter()
    seq = ie.shape[1]
    pos = np.arange(seq, dtype=np.int64).reshape(1, 1, seq).repeat(3, axis=0)
    po = prefill_s.run(None, {"inputs_embeds": ie, "attention_mask": np.ones((1, seq), dtype=np.int64), "position_ids": pos})
    pm = dict(zip(pf_on, po))
    logits = pm["logits"]
    hidden = pm["hidden_states"][:, -1:, :]
    kv = {
        "present_keys": np.stack([pm[f"present_key_{i}"] for i in range(28)]),
        "present_values": np.stack([pm[f"present_value_{i}"] for i in range(28)]),
    }
    print(f"  Prefill: {(time.perf_counter()-t0)*1000:.0f}ms")

    # Decode loop
    all_codes = []
    ts = seq
    t0 = time.perf_counter()

    for frame in range(max_frames):
        fc = sample(logits[0, -1, :])
        if fc == CODEC_EOS:
            print(f"  EOS at frame {frame}")
            break

        # Code predictor (autoregressive with KV cache)
        fce = codec_emb(fc)
        ci = np.concatenate([hidden, fce], axis=1)
        cpk = np.zeros((5, 1, 8, 0, 128), dtype=np.float32)
        cpv = np.zeros((5, 1, 8, 0, 128), dtype=np.float32)
        subs = []
        for step in range(15):
            co = cp_s.run(None, {"inputs_embeds": ci, "generation_steps": np.array([step], dtype=np.int64),
                                 "past_keys": cpk, "past_values": cpv})
            cm = dict(zip(cp_on, co))
            sc = sample(cm["logits"][0, -1, :])
            subs.append(sc)
            cpk = cm["present_keys"]; cpv = cm["present_values"]
            ci = cp_emb(sc, step)

        all_codes.append([fc] + subs)

        # Next input: sum all 16 codec embeds + tts_pad
        se = fce.copy()
        for i, sc in enumerate(subs):
            se = se + cp_emb(sc, i)
        ne = se + tts_pad_e

        ts += 1
        feed = {
            "inputs_embeds": ne,
            "attention_mask": np.ones((1, ts), dtype=np.int64),
            "position_ids": np.array([[[ts - 1]]] * 3, dtype=np.int64),
            "past_keys": kv["present_keys"],
            "past_values": kv["present_values"],
        }
        do = decode_s.run(None, feed)
        dm = dict(zip(dc_on, do))
        logits = dm["logits"]; hidden = dm["hidden_states"][:, -1:, :]
        kv = {"present_keys": dm["present_keys"], "present_values": dm["present_values"]}

        if (frame + 1) % 10 == 0:
            print(f"  Frame {frame+1}/{max_frames}")

    decode_ms = (time.perf_counter() - t0) * 1000
    n = len(all_codes)
    audio_dur = n / 12.5
    print(f"  Decode: {n} frames ({audio_dur:.1f}s) in {decode_ms:.0f}ms")

    if n == 0:
        print("  No frames generated!"); return

    # Vocoder
    t0 = time.perf_counter()
    ca = np.array(all_codes, dtype=np.int64).T[np.newaxis, :, :]
    wav = voc_s.run(None, {vi: ca})[0].flatten()
    voc_ms = (time.perf_counter() - t0) * 1000
    print(f"  Vocoder: {voc_ms:.0f}ms, rms={np.sqrt(np.mean(wav**2)):.4f}")

    # Save
    with wave.open(output, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
        wf.writeframes((wav * 32767).clip(-32768, 32767).astype(np.int16).tobytes())

    total_ms = (time.perf_counter() - t_total) * 1000
    print(f"  Total: {total_ms:.0f}ms | RTF: {total_ms/1000/audio_dur:.2f}")
    print(f"  Saved: {output} ({audio_dur:.1f}s)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="Hello, welcome to the voice synthesis system.")
    p.add_argument("--lang", default="english", choices=list(LANG_IDS.keys()))
    p.add_argument("--output", default="/tmp/tts_out.wav")
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    synthesize(a.text, a.lang, a.output, a.max_frames, a.seed)
