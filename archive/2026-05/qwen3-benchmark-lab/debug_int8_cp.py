#!/usr/bin/env python3
"""Debug INT8 code_predictor calling pattern."""
import numpy as np, onnxruntime as ort

INT8 = "/tmp/qwen3-tts-bench/model-int8"
CPU = ["CPUExecutionProvider"]
o = ort.SessionOptions()
o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
o.intra_op_num_threads = 6

cp = ort.InferenceSession(f"{INT8}/code_predictor_q.onnx", sess_options=o, providers=CPU)
cp_emb = ort.InferenceSession(f"{INT8}/code_predictor_embed_q.onnx", sess_options=o, providers=CPU)
codec_emb = ort.InferenceSession(f"{INT8}/codec_embed_q.onnx", sess_options=o, providers=CPU)
prefill = ort.InferenceSession(f"{INT8}/talker_prefill_q.onnx", sess_options=o, providers=CPU)

input_emb = np.load("/tmp/ref_input_embeds.npy").astype(np.float32)
mask = np.ones((1, input_emb.shape[1]), dtype=np.int64)
pout = prefill.run(None, {"inputs_embeds": input_emb, "attention_mask": mask})
pnames = [o.name for o in prefill.get_outputs()]
pmap = dict(zip(pnames, pout))
hidden = pmap["last_hidden"]
first_code = 1995

fc_emb = codec_emb.run(None, {"input_ids": np.array([[first_code]], dtype=np.int64)})[0]
ref_codes = [1995, 1159, 355, 22, 1174, 1093, 625, 1814, 1058, 905, 1846, 248, 1677, 889, 812, 901]

# Test: accumulate embeddings, force reference codes
print("=== Accumulating with reference codes ===")
accum = np.concatenate([hidden, fc_emb], axis=1)
for step in range(5):
    out = cp.run(None, {"inputs_embeds": accum, "generation_step": np.array([step], dtype=np.int64)})
    sc = int(np.argmax(out[0].flatten()))
    ref_sc = ref_codes[step + 1]
    status = "OK" if sc == ref_sc else "DIFF"
    print(f"  step={step}: INT8={sc:4d} ref={ref_sc:4d} {status}  accum_shape={accum.shape}")

    # Append reference code embedding for next step
    next_emb = cp_emb.run(None, {
        "input_ids": np.array([[ref_sc]], dtype=np.int64),
        "generation_step": np.array([step], dtype=np.int64),
    })[0]
    accum = np.concatenate([accum, next_emb], axis=1)

# Test: what if generation_step means something else?
print("\n=== Try different generation_step values ===")
accum = np.concatenate([hidden, fc_emb], axis=1)
for gs in [0, 1, 2, 14]:
    out = cp.run(None, {"inputs_embeds": accum, "generation_step": np.array([gs], dtype=np.int64)})
    sc = int(np.argmax(out[0].flatten()))
    print(f"  gen_step={gs:2d}: code={sc:4d} (ref step0: 1159)")

# Test: just fc_emb alone
print("\n=== Just first_code_embed ===")
for gs in [0, 1, 2]:
    out = cp.run(None, {"inputs_embeds": fc_emb, "generation_step": np.array([gs], dtype=np.int64)})
    sc = int(np.argmax(out[0].flatten()))
    print(f"  gen_step={gs}: code={sc:4d}")
