#!/usr/bin/env python3
"""Clean e2e test: verified FP32 code_predictor + correct embedding sum."""
import time, numpy as np, onnxruntime as ort, wave

INT8 = "/tmp/qwen3-tts-bench/model-int8"
FP32 = "/tmp/qwen3-tts-bench/model"
CPU, CUDA = ["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]
CODEC_EOS = 2150
MAX_FRAMES = 40

def sess(path, provs):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = 6
    return ort.InferenceSession(path, sess_options=o, providers=provs)

print("Loading models...")
prefill_s = sess(f"{INT8}/talker_prefill_q.onnx", CPU)
decode_s = sess(f"{INT8}/talker_decode_q.onnx", CPU)
cp_s = sess(f"{FP32}/code_predictor.onnx", CPU)  # FP32 with KV cache
cp_emb_s = sess(f"{INT8}/code_predictor_embed_q.onnx", CPU)
codec_emb_s = sess(f"{INT8}/codec_embed_q.onnx", CPU)
text_proj_s = sess(f"{INT8}/text_project_q.onnx", CPU)
vocoder_s = sess(f"{FP32}/vocoder.onnx", CUDA)

prefill_onames = [o.name for o in prefill_s.get_outputs()]
decode_inames = [i.name for i in decode_s.get_inputs()]
decode_onames = [o.name for o in decode_s.get_outputs()]
cp_onames = [o.name for o in cp_s.get_outputs()]
voc_iname = vocoder_s.get_inputs()[0].name

# tts_pad_embed for decode steps
tts_pad = text_proj_s.run(None, {"input_ids": np.array([[151671]], dtype=np.int64)})[0]

# Load reference input embeddings
input_emb = np.load("/tmp/ref_input_embeds.npy").astype(np.float32)
print(f"Input: {input_emb.shape}")

# === PREFILL ===
print("\nPrefill...")
mask = np.ones((1, input_emb.shape[1]), dtype=np.int64)
pout = prefill_s.run(None, {"inputs_embeds": input_emb, "attention_mask": mask})
pm = dict(zip(prefill_onames, pout))
logits = pm["logits"]
hidden = pm["last_hidden"]
kv = {n: v for n, v in pm.items() if "present_" in n}

# === DECODE LOOP ===
print("Decoding...")
all_codes = []
total_seq = input_emb.shape[1]

for frame in range(MAX_FRAMES):
    # 1. Sample first code from talker
    first_code = int(np.argmax(logits[0, -1, :]))
    if first_code == CODEC_EOS:
        print(f"  EOS at frame {frame}")
        break

    # 2. Code predictor (FP32 with KV cache) — autoregressive 15 steps
    fc_emb = codec_emb_s.run(None, {"input_ids": np.array([[first_code]], dtype=np.int64)})[0]
    cp_in = np.concatenate([hidden, fc_emb], axis=1)  # [1, 2, 1024]
    cp_pk = np.zeros((5, 1, 8, 0, 128), dtype=np.float32)
    cp_pv = np.zeros((5, 1, 8, 0, 128), dtype=np.float32)

    sub_codes = []
    for step in range(15):
        cp_out = cp_s.run(None, {
            "inputs_embeds": cp_in,
            "generation_steps": np.array([step], dtype=np.int64),
            "past_keys": cp_pk,
            "past_values": cp_pv,
        })
        cpm = dict(zip(cp_onames, cp_out))
        sc = int(np.argmax(cpm["logits"][0, -1, :]))
        sub_codes.append(sc)
        cp_pk = cpm["present_keys"]
        cp_pv = cpm["present_values"]
        cp_in = cp_emb_s.run(None, {
            "input_ids": np.array([[sc]], dtype=np.int64),
            "generation_step": np.array([step], dtype=np.int64),
        })[0]

    codes = [first_code] + sub_codes
    all_codes.append(codes)

    # 3. Build next talker input: sum(codec_embeddings) + tts_pad
    # codec_embed(group0) + sum(cp_embed(group_i) for i in 1..15)
    sum_emb = fc_emb[:, :1, :].copy()
    for i, sc in enumerate(sub_codes):
        gi = cp_emb_s.run(None, {
            "input_ids": np.array([[sc]], dtype=np.int64),
            "generation_step": np.array([i], dtype=np.int64),
        })[0]
        sum_emb = sum_emb + gi[:, :1, :]
    next_emb = sum_emb + tts_pad

    # 4. Talker decode step
    total_seq += 1
    attn = np.ones((1, total_seq), dtype=np.int64)
    feed = {"inputs_embeds": next_emb, "attention_mask": attn}
    for n in decode_inames:
        if n.startswith("past_key_"):
            feed[n] = kv[f"present_key_{n.split('_')[-1]}"]
        elif n.startswith("past_value_"):
            feed[n] = kv[f"present_value_{n.split('_')[-1]}"]

    dout = decode_s.run(None, feed)
    dm = dict(zip(decode_onames, dout))
    logits = dm["logits"]
    hidden = dm["last_hidden"]
    kv = {n: v for n, v in dm.items() if "present_" in n}

    if (frame + 1) % 5 == 0:
        print(f"  Frame {frame+1}: code0={first_code}")

n = len(all_codes)
print(f"\n{n} frames ({n/12.5:.2f}s)")

# === VOCODER ===
if n > 0:
    codes_arr = np.array(all_codes, dtype=np.int64).transpose(1, 0)  # [16, N]
    codes_trt = codes_arr[np.newaxis, :, :]  # [1, 16, N]
    vout = vocoder_s.run(None, {voc_iname: codes_trt})
    wav = vout[0].flatten()
    print(f"Vocoder: {wav.shape} ({len(wav)/24000:.2f}s)")

    out_path = "/tmp/test_clean.wav"
    with wave.open(out_path, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
        wf.writeframes((wav * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
    print(f"Saved: {out_path}")
