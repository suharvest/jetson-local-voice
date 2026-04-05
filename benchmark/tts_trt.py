#!/usr/bin/env python3
"""Qwen3-TTS 0.6B — TRT-accelerated pipeline.
Uses ORT TRT EP for vocoder, FP32 CPU for talker/code_predictor (TRT has If node issues).
Measures per-component timing for optimization planning."""
import argparse, os, time
import numpy as np, onnxruntime as ort, wave

FP32 = "/tmp/qwen3-tts-bench/model"
CEMB = "/tmp/correct-emb"
CPU = ["CPUExecutionProvider"]
CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# TRT EP for vocoder (proven 3x speedup)
TRT_VOC = [
    ("TensorrtExecutionProvider", {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "/tmp/trt_cache_voc",
        "trt_max_workspace_size": str(1024*1024*1024),
    }),
    "CUDAExecutionProvider", "CPUExecutionProvider",
]

# Token IDs
CODEC_EOS = 2150; CODEC_THINK = 2154; CODEC_THINK_BOS = 2156
CODEC_THINK_EOS = 2157; CODEC_PAD = 2148; CODEC_BOS = 2149
TTS_BOS = 151672; TTS_EOS = 151673; TTS_PAD = 151671
IM_START = 151644
LANG_IDS = {"chinese": 2055, "english": 2050, "japanese": 2058, "korean": 2064}

# ---- Load ----
print("Loading models...")
o = ort.SessionOptions()
o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
o.intra_op_num_threads = 6

prefill_s = ort.InferenceSession(f"{FP32}/talker_prefill.onnx", sess_options=o, providers=CPU)
decode_s = ort.InferenceSession(f"{FP32}/talker_decode.onnx", sess_options=o, providers=CPU)
cp_s = ort.InferenceSession(f"{FP32}/code_predictor.onnx", sess_options=o, providers=CPU)

os.makedirs("/tmp/trt_cache_voc", exist_ok=True)
print("  Loading vocoder with TRT EP (first run builds engine)...")
voc_s = ort.InferenceSession(f"{FP32}/vocoder.onnx", providers=TRT_VOC)

pf_on = [x.name for x in prefill_s.get_outputs()]
dc_on = [x.name for x in decode_s.get_outputs()]
cp_on = [x.name for x in cp_s.get_outputs()]
vi = voc_s.get_inputs()[0].name

print("Loading embeddings...")
talker_ce = np.load(f"{CEMB}/talker_codec_embedding.npy")
cp_ces = [np.load(f"{CEMB}/cp_codec_embedding_{i}.npy") for i in range(15)]
text_emb_w = np.load(f"{FP32}/embeddings/text_embedding.npy")
fc1_w = np.load(f"{CEMB}/tp_fc1_weight.npy")
fc1_b = np.load(f"{CEMB}/tp_fc1_bias.npy")
fc2_w = np.load(f"{CEMB}/tp_fc2_weight.npy")
fc2_b = np.load(f"{CEMB}/tp_fc2_bias.npy")

print("Loading tokenizer...")
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
tok = Tokenizer(BPE(f"{FP32}/tokenizer/vocab.json", f"{FP32}/tokenizer/merges.txt"))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
def tokenize(t): return tok.encode(t).ids

# ---- Helpers ----
def text_proj(ids):
    embs = []
    for tid in ids:
        x = text_emb_w[tid].astype(np.float32)
        h = x @ fc1_w.T + fc1_b
        h = h * (1 / (1 + np.exp(-h)))
        embs.append(h @ fc2_w.T + fc2_b)
    return np.array(embs).reshape(1, len(ids), 1024)

def ce(tid): return talker_ce[tid].reshape(1,1,1024).astype(np.float32)
def cpe(tid, g): return cp_ces[g][tid].reshape(1,1,1024).astype(np.float32)

def sample(logits, top_k=50, temp=0.9):
    l = logits.flatten().astype(np.float64) / temp
    ti = np.argpartition(l, -top_k)[-top_k:]
    m = np.full_like(l, -np.inf); m[ti] = l[ti]
    e = np.exp(m - np.max(m)); p = e / e.sum()
    return int(np.random.choice(len(p), p=p))

def build_input_embeds(text, lang="english"):
    lang_id = LANG_IDS.get(lang, 2050)
    text_ids = tokenize(text)
    role_ids = [IM_START, 77091, 198]
    tts_bos_e = text_proj([TTS_BOS])
    tts_eos_e = text_proj([TTS_EOS])
    tts_pad_e = text_proj([TTS_PAD])
    role_e = text_proj(role_ids)
    codec_in = np.concatenate([ce(CODEC_THINK), ce(CODEC_THINK_BOS), ce(lang_id), ce(CODEC_THINK_EOS), ce(CODEC_PAD), ce(CODEC_BOS)], axis=1)
    tts_part = np.concatenate([np.tile(tts_pad_e, (1,4,1)), tts_bos_e], axis=1)
    prefix = tts_part + codec_in[:, :-1, :]
    embed = np.concatenate([role_e, prefix], axis=1)
    text_e = text_proj(text_ids)
    text_eos = np.concatenate([text_e, tts_eos_e], axis=1)
    codec_pad_n = np.tile(ce(CODEC_PAD), (1, len(text_ids)+1, 1))
    embed = np.concatenate([embed, text_eos + codec_pad_n, tts_pad_e + ce(CODEC_BOS)], axis=1)
    return embed.astype(np.float32), tts_pad_e

# ---- Synthesize ----
def synthesize(text, lang="english", output="/tmp/tts_out.wav", max_frames=60, seed=None):
    if seed is not None: np.random.seed(seed)
    print(f"\nSynthesizing: \"{text}\" ({lang})")
    timings = {}
    t_total = time.perf_counter()

    # Build embeddings
    t0 = time.perf_counter()
    ie, tts_pad_e = build_input_embeds(text, lang)
    timings["embed"] = (time.perf_counter() - t0) * 1000
    print(f"  Embed: {ie.shape} ({timings['embed']:.0f}ms)")

    # Prefill
    t0 = time.perf_counter()
    seq = ie.shape[1]
    pos = np.arange(seq, dtype=np.int64).reshape(1,1,seq).repeat(3, axis=0)
    po = prefill_s.run(None, {"inputs_embeds": ie, "attention_mask": np.ones((1,seq), dtype=np.int64), "position_ids": pos})
    pm = dict(zip(pf_on, po))
    logits = pm["logits"]; hidden = pm["hidden_states"][:, -1:, :]
    kv = {"present_keys": np.stack([pm[f"present_key_{i}"] for i in range(28)]),
          "present_values": np.stack([pm[f"present_value_{i}"] for i in range(28)])}
    timings["prefill"] = (time.perf_counter() - t0) * 1000
    print(f"  Prefill: {timings['prefill']:.0f}ms")

    # Decode loop
    all_codes = []; ts = seq
    decode_times = []; cp_times = []; emb_times = []

    for frame in range(max_frames):
        fc = sample(logits[0, -1, :])
        if fc == CODEC_EOS:
            print(f"  EOS at frame {frame}"); break

        # Code predictor
        t_cp = time.perf_counter()
        fce = ce(fc)
        ci = np.concatenate([hidden, fce], axis=1)
        cpk = np.zeros((5,1,8,0,128), dtype=np.float32)
        cpv = np.zeros((5,1,8,0,128), dtype=np.float32)
        subs = []
        for step in range(15):
            co = cp_s.run(None, {"inputs_embeds": ci, "generation_steps": np.array([step], dtype=np.int64),
                                 "past_keys": cpk, "past_values": cpv})
            cm = dict(zip(cp_on, co))
            sc = sample(cm["logits"][0, -1, :])
            subs.append(sc); cpk = cm["present_keys"]; cpv = cm["present_values"]
            ci = cpe(sc, step)
        cp_times.append(time.perf_counter() - t_cp)
        all_codes.append([fc] + subs)

        # Build next input
        t_emb = time.perf_counter()
        se = fce.copy()
        for i, sc in enumerate(subs): se = se + cpe(sc, i)
        ne = se + tts_pad_e
        emb_times.append(time.perf_counter() - t_emb)

        # Decode
        t_dec = time.perf_counter()
        ts += 1
        feed = {"inputs_embeds": ne, "attention_mask": np.ones((1,ts), dtype=np.int64),
                "position_ids": np.array([[[ts-1]]]*3, dtype=np.int64),
                "past_keys": kv["present_keys"], "past_values": kv["present_values"]}
        do = decode_s.run(None, feed)
        dm = dict(zip(dc_on, do))
        logits = dm["logits"]; hidden = dm["hidden_states"][:, -1:, :]
        kv = {"present_keys": dm["present_keys"], "present_values": dm["present_values"]}
        decode_times.append(time.perf_counter() - t_dec)

    n = len(all_codes); dur = n / 12.5
    timings["decode_total"] = sum(decode_times) * 1000
    timings["cp_total"] = sum(cp_times) * 1000
    timings["emb_total"] = sum(emb_times) * 1000
    if n > 0:
        timings["decode_avg"] = np.mean(decode_times) * 1000
        timings["cp_avg"] = np.mean(cp_times) * 1000
        timings["per_step"] = timings["decode_avg"] + timings["cp_avg"]
    print(f"  Decode: {n} frames ({dur:.1f}s)")
    if n > 0:
        print(f"    Per-step: talker={timings['decode_avg']:.1f}ms + cp={timings['cp_avg']:.1f}ms = {timings['per_step']:.1f}ms")
        print(f"    RTF decode: {timings['per_step']/80:.2f}")

    if n == 0: print("  No frames!"); return

    # Vocoder (TRT FP16)
    t0 = time.perf_counter()
    ca = np.array(all_codes, dtype=np.int64).T[np.newaxis,:,:]
    wav = voc_s.run(None, {vi: ca})[0].flatten()
    timings["vocoder"] = (time.perf_counter() - t0) * 1000
    print(f"  Vocoder: {timings['vocoder']:.0f}ms (TRT FP16)")

    with wave.open(output, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
        wf.writeframes((wav * 32767).clip(-32768, 32767).astype(np.int16).tobytes())

    total = (time.perf_counter() - t_total) * 1000
    print(f"\n  === TIMING SUMMARY ===")
    print(f"  Embed build:    {timings['embed']:>7.0f}ms")
    print(f"  Prefill:        {timings['prefill']:>7.0f}ms")
    print(f"  Decode total:   {timings['decode_total']:>7.0f}ms  (talker)")
    print(f"  Code pred total:{timings['cp_total']:>7.0f}ms")
    print(f"  Vocoder:        {timings['vocoder']:>7.0f}ms  (TRT FP16)")
    print(f"  Total:          {total:>7.0f}ms")
    print(f"  Audio:          {dur:.1f}s")
    print(f"  Overall RTF:    {total/1000/dur:.2f}")
    print(f"  Saved: {output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="Hello, welcome to the voice synthesis system.")
    p.add_argument("--lang", default="english", choices=list(LANG_IDS.keys()))
    p.add_argument("--output", default="/tmp/tts_out.wav")
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    synthesize(a.text, a.lang, a.output, a.max_frames, a.seed)
