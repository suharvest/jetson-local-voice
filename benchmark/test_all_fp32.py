#!/usr/bin/env python3
"""Test: ALL FP32 models from elbruno (same source, no mixing)."""
import numpy as np, onnxruntime as ort, wave, time

FP32 = "/tmp/qwen3-tts-bench/model"
INT8 = "/tmp/qwen3-tts-bench/model-int8"  # only for cp_embed (not in elbruno)
CPU = ["CPUExecutionProvider"]
CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CODEC_EOS = 2150

o = ort.SessionOptions()
o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
o.intra_op_num_threads = 6

print("Loading ALL FP32 models (elbruno)...")
prefill = ort.InferenceSession(f"{FP32}/talker_prefill.onnx", sess_options=o, providers=CPU)
decode = ort.InferenceSession(f"{FP32}/talker_decode.onnx", sess_options=o, providers=CPU)
cp = ort.InferenceSession(f"{FP32}/code_predictor.onnx", sess_options=o, providers=CPU)
# Use FP32 NPY embeddings (same source as talker/code_predictor)
CEMB = "/tmp/correct-emb"  # Exported directly from PyTorch model
talker_codec_emb = np.load(f"{CEMB}/talker_codec_embedding.npy")  # [3072, 1024]
cp_codec_embs = [np.load(f"{CEMB}/cp_codec_embedding_{i}.npy") for i in range(15)]  # 15 x [2048, 1024]
EMB = f"{FP32}/embeddings"  # text projection weights (elbruno)
text_emb_weights = np.load(f"{EMB}/text_embedding.npy")  # [151936, 2048]
fc1_w = np.load(f"{EMB}/text_projection_fc1_weight.npy")  # [2048, 2048]
fc1_b = np.load(f"{EMB}/text_projection_fc1_bias.npy")    # [2048]
fc2_w = np.load(f"{EMB}/text_projection_fc2_weight.npy")  # [1024, 2048]
fc2_b = np.load(f"{EMB}/text_projection_fc2_bias.npy")    # [1024]

def codec_embed_lookup(token_id):
    return talker_codec_emb[token_id:token_id+1].reshape(1, 1, 1024).astype(np.float32)

def cp_embed_lookup(token_id, group):
    return cp_codec_embs[group][token_id:token_id+1].reshape(1, 1, 1024).astype(np.float32)

def text_project(token_id):
    """text_embedding → text_projection (fc1 → silu → fc2)"""
    x = text_emb_weights[token_id].astype(np.float32)  # [2048]
    h = x @ fc1_w.T + fc1_b  # [2048]
    h = h * (1 / (1 + np.exp(-h)))  # SiLU
    out = h @ fc2_w.T + fc2_b  # [1024]
    return out.reshape(1, 1, 1024)

print("Using FP32 NPY embeddings (same source as models)")
voc = ort.InferenceSession(f"{FP32}/vocoder.onnx", providers=CUDA)

pf_on = [x.name for x in prefill.get_outputs()]
dc_in = {x.name: x.shape for x in decode.get_inputs()}
dc_on = [x.name for x in decode.get_outputs()]
print(f"Decode inputs: {list(dc_in.keys())}")
cp_on = [x.name for x in cp.get_outputs()]
vi = voc.get_inputs()[0].name

tts_pad = text_project(151671)  # tts_pad_token_id
ie = np.load("/tmp/ref_input_embeds.npy").astype(np.float32)
mask = np.ones((1, ie.shape[1]), dtype=np.int64)

print(f"Input: {ie.shape}")

# Prefill
t0 = time.perf_counter()
seq = ie.shape[1]
pos_ids = np.arange(seq, dtype=np.int64).reshape(1, 1, seq).repeat(3, axis=0)  # [3, 1, seq]
po = prefill.run(None, {"inputs_embeds": ie, "attention_mask": mask, "position_ids": pos_ids})
pm = dict(zip(pf_on, po))
logits = pm["logits"]
hidden = pm["hidden_states"][:, -1:, :]  # last token hidden
# Prefill outputs separate KV per layer, decode expects stacked
# Stack present_key_0..27 → [28, B, 8, T, 128]
pk_list = [pm[f"present_key_{i}"] for i in range(28)]
pv_list = [pm[f"present_value_{i}"] for i in range(28)]
kv = {
    "present_keys": np.stack(pk_list, axis=0),
    "present_values": np.stack(pv_list, axis=0),
}
print(f"Prefill: {(time.perf_counter()-t0)*1000:.0f}ms logits={logits.shape} kv={kv['present_keys'].shape}")

# Decode
all_codes = []
ts = ie.shape[1]

for frame in range(30):
    # Sample first code (talker) with temperature + top-k
    _fl = logits[0, -1, :].astype(np.float64) / 0.9
    _tk = 50; _ti = np.argpartition(_fl, -_tk)[-_tk:]
    _m = np.full_like(_fl, -np.inf); _m[_ti] = _fl[_ti]
    _e = np.exp(_m - np.max(_m)); _p = _e / _e.sum()
    fc = int(np.random.choice(len(_p), p=_p))
    if fc == CODEC_EOS:
        print(f"  EOS at frame {frame}")
        break

    # Code predictor
    fce = codec_embed_lookup(fc)
    ci = np.concatenate([hidden, fce], axis=1)
    cpk = np.zeros((5, 1, 8, 0, 128), dtype=np.float32)
    cpv = np.zeros((5, 1, 8, 0, 128), dtype=np.float32)
    subs = []
    for step in range(15):
        co = cp.run(None, {"inputs_embeds": ci, "generation_steps": np.array([step], dtype=np.int64),
                           "past_keys": cpk, "past_values": cpv})
        cm = dict(zip(cp_on, co))
        # MUST sample (not greedy) — greedy sub-codes degenerate to silence
        _sl = cm["logits"][0, -1, :].astype(np.float64) / 0.9
        _tk = 50; _ti = np.argpartition(_sl, -_tk)[-_tk:]
        _m = np.full_like(_sl, -np.inf); _m[_ti] = _sl[_ti]
        _e = np.exp(_m - np.max(_m)); _p = _e / _e.sum()
        sc = int(np.random.choice(len(_p), p=_p))
        subs.append(sc)
        cpk = cm["present_keys"]; cpv = cm["present_values"]
        ci = cp_embed_lookup(sc, step)

    codes = [fc] + subs
    all_codes.append(codes)

    # Next: sum ALL 16 codec embeddings + tts_pad (official SDK pattern)
    se = codec_embed_lookup(fc)
    for i, sc in enumerate(subs):
        se = se + cp_embed_lookup(sc, i)
    ne = se + tts_pad

    if frame == 0:
        print(f"  next_emb mean: {ne.mean():.6f} hidden mean: {hidden.mean():.6f}")

    # Decode step
    ts += 1
    attn = np.ones((1, ts), dtype=np.int64)
    pos = np.array([[[ts - 1]]] * 3, dtype=np.int64)  # [3, 1, 1]
    feed = {
        "inputs_embeds": ne,
        "attention_mask": attn,
        "position_ids": pos,
        "past_keys": kv["present_keys"],
        "past_values": kv["present_values"],
    }

    do = decode.run(None, feed)
    dm = dict(zip(dc_on, do))
    logits = dm["logits"]; hidden = dm["hidden_states"][:, -1:, :]
    kv = {"present_keys": dm["present_keys"], "present_values": dm["present_values"]}

    if frame < 5 or (frame + 1) % 10 == 0:
        print(f"  Frame {frame}: fc={fc} h_mean={hidden.mean():.6f} subs[:3]={subs[:3]}")

n = len(all_codes)
print(f"\n{n} frames ({n/12.5:.2f}s)")
if n > 0:
    ca = np.array(all_codes, dtype=np.int64)
    ct = ca.T[np.newaxis, :, :]
    vo = voc.run(None, {vi: ct})
    wav = vo[0].flatten()
    rms = np.sqrt(np.mean(wav**2))
    print(f"Vocoder: {len(wav)} samples, rms={rms:.4f}")
    with wave.open("/tmp/test_fp32_all.wav", "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
        wf.writeframes((wav * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
    print(f"Saved /tmp/test_fp32_all.wav")
