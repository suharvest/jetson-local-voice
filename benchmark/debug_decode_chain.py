#!/usr/bin/env python3
"""Debug: step through decode chain to find where codes diverge."""
import numpy as np, onnxruntime as ort

INT8 = "/tmp/qwen3-tts-bench/model-int8"
CPU = ["CPUExecutionProvider"]
o = ort.SessionOptions()
o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
o.intra_op_num_threads = 6

prefill_s = ort.InferenceSession(f"{INT8}/talker_prefill_q.onnx", sess_options=o, providers=CPU)
decode_s = ort.InferenceSession(f"{INT8}/talker_decode_q.onnx", sess_options=o, providers=CPU)
codec_emb_s = ort.InferenceSession(f"{INT8}/codec_embed_q.onnx", sess_options=o, providers=CPU)
text_proj_s = ort.InferenceSession(f"{INT8}/text_project_q.onnx", sess_options=o, providers=CPU)

pf_onames = [o.name for o in prefill_s.get_outputs()]
dc_inames = [i.name for i in decode_s.get_inputs()]
dc_onames = [o.name for o in decode_s.get_outputs()]

tts_pad = text_proj_s.run(None, {"input_ids": np.array([[151671]], dtype=np.int64)})[0]

input_emb = np.load("/tmp/ref_input_embeds.npy").astype(np.float32)
mask = np.ones((1, input_emb.shape[1]), dtype=np.int64)
pout = prefill_s.run(None, {"inputs_embeds": input_emb, "attention_mask": mask})
pm = dict(zip(pf_onames, pout))
logits = pm["logits"]
kv = {n: v for n, v in pm.items() if "present_" in n}

k0 = kv.get("present_key_0")
print(f"Prefill KV key_0: {k0.shape}")
print(f"KV count: {len(kv)}")

first_code = int(np.argmax(logits[0, -1, :]))
print(f"\nFrame 0: {first_code} (ref: 1995)")

# Use ref step1 embeds to verify decode works
ref_step1 = np.load("/tmp/ref_step1_embeds.npy").astype(np.float32)

# Also compute our own step1 (simplified: just codec_embed + tts_pad)
fc_emb = codec_emb_s.run(None, {"input_ids": np.array([[first_code]], dtype=np.int64)})[0]
our_step1 = fc_emb + tts_pad

print(f"\nRef step1 mean: {ref_step1.mean():.6f}")
print(f"Our step1 mean: {our_step1.mean():.6f}")
print(f"Diff: {np.abs(ref_step1 - our_step1).mean():.6f}")

ref_codes = [1995, 215, 212, 1181, 462]
total_seq = input_emb.shape[1]

for label, step1_emb in [("REF_EMBEDS", ref_step1), ("OUR_EMBEDS", our_step1)]:
    # Reset KV to prefill state
    kv_local = {n: v.copy() for n, v in pm.items() if "present_" in n}
    ts = total_seq

    print(f"\n=== {label} ===")
    ne = step1_emb
    for frame in range(1, 5):
        ts += 1
        attn = np.ones((1, ts), dtype=np.int64)
        feed = {"inputs_embeds": ne, "attention_mask": attn}
        for n in dc_inames:
            if n.startswith("past_key_"):
                idx = n.replace("past_key_", "")
                feed[n] = kv_local["present_key_" + idx]
            elif n.startswith("past_value_"):
                idx = n.replace("past_value_", "")
                feed[n] = kv_local["present_value_" + idx]

        dout = decode_s.run(None, feed)
        dm = dict(zip(dc_onames, dout))
        logits = dm["logits"]
        kv_local = {n: v for n, v in dm.items() if "present_" in n}

        code = int(np.argmax(logits[0, -1, :]))
        print(f"  Frame {frame}: {code} (ref: {ref_codes[frame]})")

        fc_emb = codec_emb_s.run(None, {"input_ids": np.array([[code]], dtype=np.int64)})[0]
        ne = fc_emb + tts_pad
