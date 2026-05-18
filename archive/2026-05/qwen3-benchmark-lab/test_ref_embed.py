#!/usr/bin/env python3
"""Test: feed official SDK's input_embeds into ONNX talker pipeline.
If this produces correct audio, the issue is in embedding construction only."""
import time, numpy as np, onnxruntime as ort, os, struct, wave

INT8 = "/tmp/qwen3-tts-bench/model-int8"
FP32 = "/tmp/qwen3-tts-bench/model"
REF_EMBEDS = "/tmp/ref_input_embeds.npy"
CODEC_EOS = 2150
MAX_FRAMES = 30

def create_sess(path, provs):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = 6
    return ort.InferenceSession(path, sess_options=o, providers=provs)

CPU = ["CPUExecutionProvider"]
CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]

print("Loading reference input_embeds...")
input_emb = np.load(REF_EMBEDS).astype(np.float32)
print(f"  Shape: {input_emb.shape}")  # [1, 13, 1024]

# Pre-compute trailing text hidden (text embeddings for streaming injection)
# The input_emb structure is: [role(3)] [codec_prefix(5)] [text+eos(4)] [final(1)]
# text tokens start at position 8 (after role+codec_prefix), text+eos = positions 8..11
# trailing_text_hidden = text_embed for each decode step
# For non-streaming: text is pre-fed in prefill, trailing = tts_pad_embed for all steps
# Actually, in non-streaming mode (line 182-205 of generate()):
#   All text is already in the prefill input
#   trailing_text_hidden is computed from text_embed + tts_eos
# But during decode, inputs_embeds += trailing_text_hidden[:, step] OR tts_pad_embed

# For simplicity: compute tts_pad_embed to add at each decode step
text_project = create_sess(f"{INT8}/text_project_q.onnx", CPU)
tts_pad_id = 151671
tts_pad_emb = text_project.run(None, {"input_ids": np.array([[tts_pad_id]], dtype=np.int64)})[0]  # [1, 1, 1024]
print(f"  tts_pad_embed: {tts_pad_emb.shape}")
del text_project

# Load models
print("Loading ONNX models...")
prefill = create_sess(f"{INT8}/talker_prefill_q.onnx", CPU)
decode = create_sess(f"{INT8}/talker_decode_q.onnx", CPU)
# Use FP32 code_predictor WITH KV cache (INT8 version is simplified, no KV cache)
code_pred = create_sess(f"{FP32}/code_predictor.onnx", CPU)
cp_embed = create_sess(f"{INT8}/code_predictor_embed_q.onnx", CPU)
codec_embed = create_sess(f"{INT8}/codec_embed_q.onnx", CPU)
vocoder = create_sess(f"{FP32}/vocoder.onnx", CUDA)

cp_out_names = [o.name for o in code_pred.get_outputs()]

prefill_out_names = [o.name for o in prefill.get_outputs()]
decode_in_names = [i.name for i in decode.get_inputs()]
decode_out_names = [o.name for o in decode.get_outputs()]
vocoder_in = vocoder.get_inputs()[0].name

# Prefill
print("\nPrefill...")
seq_len = input_emb.shape[1]
mask = np.ones((1, seq_len), dtype=np.int64)
t0 = time.perf_counter()
pout = prefill.run(None, {"inputs_embeds": input_emb, "attention_mask": mask})
print(f"  Done in {(time.perf_counter()-t0)*1000:.0f}ms")

pmap = {n: v for n, v in zip(prefill_out_names, pout)}
logits = pmap["logits"]       # [1, seq, 3072]
hidden = pmap["last_hidden"]  # [1, 1, 1024]
kv = {n: v for n, v in pmap.items() if "present_" in n}
print(f"  logits: {logits.shape}, kv tensors: {len(kv)}")

# Decode loop
print("\nDecode loop...")
all_codes = []
total_seq = seq_len

for frame in range(MAX_FRAMES):
    # Sample from logits (greedy for determinism)
    codec_logits = logits[0, -1, :]  # all 3072
    first_code = int(np.argmax(codec_logits))

    if first_code == CODEC_EOS:
        print(f"  EOS at frame {frame}")
        break

    # Code predictor: 15 sub-codes (autoregressive with KV cache)
    # FP32 code_predictor has: inputs_embeds, generation_steps, past_keys, past_values
    # Input = [past_hidden, codec_embed(first_code)] concatenated for first call
    first_code_emb = codec_embed.run(None, {
        "input_ids": np.array([[first_code]], dtype=np.int64)
    })[0]  # [1, 1, 1024]

    cp_input = np.concatenate([hidden, first_code_emb], axis=1)  # [1, 2, 1024]
    codes = [first_code]
    cp_past_k = np.zeros((5, 1, 8, 0, 128), dtype=np.float32)
    cp_past_v = np.zeros((5, 1, 8, 0, 128), dtype=np.float32)

    for step in range(15):
        cp_feed = {
            "inputs_embeds": cp_input,
            "generation_steps": np.array([step], dtype=np.int64),
            "past_keys": cp_past_k,
            "past_values": cp_past_v,
        }
        cp_out = code_pred.run(None, cp_feed)
        cp_map = dict(zip(cp_out_names, cp_out))
        cp_logits = cp_map["logits"]  # [1, seq, 2048]

        sub_logits = cp_logits[0, -1, :]
        sub_code = int(np.argmax(sub_logits))
        codes.append(sub_code)

        # Update KV cache
        cp_past_k = cp_map["present_keys"]
        cp_past_v = cp_map["present_values"]

        # Next input: cp_embed of sub_code for this group
        cp_input = cp_embed.run(None, {
            "input_ids": np.array([[sub_code]], dtype=np.int64),
            "generation_step": np.array([step], dtype=np.int64),
        })[0]  # [1, 1, 1024]

    all_codes.append(codes)

    # Next input: sum of all codec group embeddings (official SDK does this)
    # inputs_embeds = codec_hiddens.sum(1, keepdim=True)
    # For now use codec_embed of first code as approximation
    all_code_embs = [codec_embed.run(None, {"input_ids": np.array([[codes[0]]], dtype=np.int64)})[0]]
    for gi in range(15):
        emb_gi = cp_embed.run(None, {
            "input_ids": np.array([[codes[gi + 1]]], dtype=np.int64),
            "generation_step": np.array([gi], dtype=np.int64),
        })[0]
        all_code_embs.append(emb_gi)
    next_emb = np.sum(np.stack([e[:, 0:1, :] for e in all_code_embs], axis=0), axis=0)
    # Add text embedding for this step (tts_pad for non-streaming after text is consumed)
    next_emb = next_emb + tts_pad_emb

    # Talker decode
    total_seq += 1
    attn = np.ones((1, total_seq), dtype=np.int64)
    feed = {"inputs_embeds": next_emb, "attention_mask": attn}
    for n in decode_in_names:
        if n.startswith("past_key_"):
            idx = n.replace("past_key_", "")
            feed[n] = kv.get(f"present_key_{idx}", np.zeros((1,8,total_seq-1,128), dtype=np.float32))
        elif n.startswith("past_value_"):
            idx = n.replace("past_value_", "")
            feed[n] = kv.get(f"present_value_{idx}", np.zeros((1,8,total_seq-1,128), dtype=np.float32))

    dout = decode.run(None, feed)
    dmap = {n: v for n, v in zip(decode_out_names, dout)}
    logits = dmap["logits"]
    hidden = dmap["last_hidden"]
    kv = {n: v for n, v in dmap.items() if "present_" in n}

    if (frame + 1) % 5 == 0:
        print(f"  Frame {frame+1}: codes[0]={codes[0]}")

n_frames = len(all_codes)
print(f"\nGenerated {n_frames} frames ({n_frames/12.5:.2f}s audio)")

if n_frames == 0:
    print("No frames generated!")
    exit(1)

# Vocoder
print("Vocoder...")
codes_arr = np.array(all_codes, dtype=np.int64)  # [N, 16]
codes_trt = codes_arr.T[np.newaxis, :, :]  # [1, 16, N]
vout = vocoder.run(None, {vocoder_in: codes_trt})
wav = vout[0].flatten()
print(f"  Waveform: {wav.shape}, {len(wav)/24000:.2f}s")

# Save
out_path = "/tmp/test_ref_embed.wav"
audio_int16 = (wav * 32767).clip(-32768, 32767).astype(np.int16)
with wave.open(out_path, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(audio_int16.tobytes())
print(f"Saved: {out_path}")
