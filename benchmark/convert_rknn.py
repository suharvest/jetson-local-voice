#!/usr/bin/env python3
"""Convert Qwen3-TTS ONNX models to RKNN format for RK3576 (v4).

Key insight: RKNN dynamic_input fixed shapes must match the ONNX trace shapes
for any node that has shape-dependent constants baked in during tracing.

For talker_prefill (traced with T=8), we must use seq_len=8 in RKNN.
For talker_decode (traced with T_past=8), we must use past_len=8 in RKNN.

After conversion, we'll need to re-export with the desired fixed shapes.
"""

import os
import time
import onnx
from rknn.api import RKNN

ONNX_DIR = os.path.expanduser("~/qwen3-tts-export/qwen3-tts-0.6b-12hz")
RKNN_DIR = os.path.expanduser("~/qwen3-tts-export/qwen3-tts-0.6b-12hz-rknn")
TARGET = "rk3576"
os.makedirs(RKNN_DIR, exist_ok=True)


def convert_model(model_name, fixed_shapes=None, opt_level=3):
    onnx_path = os.path.join(ONNX_DIR, f"{model_name}.onnx")
    rknn_path = os.path.join(RKNN_DIR, f"{model_name}.rknn")
    
    if not os.path.exists(onnx_path):
        return ("SKIP", "not found")
    
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"Converting: {model_name} ({onnx_size_mb:.1f} MB)")
    
    # Show inputs
    model = onnx.load(onnx_path)
    for inp in model.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  Input: {inp.name} {dims}")
    del model
    
    t0 = time.time()
    rknn = RKNN(verbose=False)
    
    try:
        kwargs = {"target_platform": TARGET, "optimization_level": opt_level}
        if fixed_shapes:
            kwargs["dynamic_input"] = [fixed_shapes]
        
        ret = rknn.config(**kwargs)
        if ret != 0:
            return ("FAIL", f"config returned {ret}")
        
        print(f"  load_onnx...")
        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            return ("FAIL", f"load_onnx returned {ret}")
        
        print(f"  build...")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            return ("FAIL", f"build returned {ret}")
        
        print(f"  export_rknn...")
        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            return ("FAIL", f"export returned {ret}")
        
        elapsed = time.time() - t0
        rknn_size_mb = os.path.getsize(rknn_path) / (1024 * 1024)
        result = f"{rknn_size_mb:.1f} MB ({elapsed:.1f}s)"
        print(f"  OK: {onnx_size_mb:.1f} MB -> {result}")
        return ("OK", result)
    
    except Exception as e:
        err = str(e).split('\n')[0][:100]
        print(f"  FAIL: {err}")
        return ("FAIL", err)
    finally:
        rknn.release()


results = {}

# 1. codec_embed: 1 input, [1, seq_len] -> fix to [1, 1]
results["codec_embed"] = convert_model("codec_embed", fixed_shapes=[[1, 1]])

# 2. code_predictor_embed: static inputs [1,1] and scalar
results["code_predictor_embed"] = convert_model("code_predictor_embed")

# 3. speaker_encoder: [1, time, 128] -> traced with time=100
# Try opt_level=0 to avoid constant folding
results["speaker_encoder"] = convert_model(
    "speaker_encoder", fixed_shapes=[[1, 100, 128]], opt_level=0
)

# 4. code_predictor: 1 input [1, ctx_len, 1024] -> fix to [1, 2, 1024]
results["code_predictor"] = convert_model(
    "code_predictor", fixed_shapes=[[1, 2, 1024]]
)

# 5. text_project: [1, seq_len] -> fix to [1, 128]
results["text_project"] = convert_model("text_project", fixed_shapes=[[1, 128]])

# 6. talker_prefill: traced with T=8
# Inputs: [1, seq_len, 1024], [1, seq_len]
# Must match trace T=8 to avoid Reshape mismatches
results["talker_prefill"] = convert_model(
    "talker_prefill", fixed_shapes=[[1, 8, 1024], [1, 8]]
)

# 7. talker_decode: traced with T_past=8
# Inputs: [1, 1, 1024], [1, full_len], + 56 KV tensors [1, 8, past_len, 128]
# full_len = past_len + 1 = 9
T_PAST = 8
decode_shapes = [
    [1, 1, 1024],          # inputs_embeds (static)
    [1, T_PAST + 1],       # attention_mask
]
for _ in range(28):
    decode_shapes.append([1, 8, T_PAST, 128])  # past_key
    decode_shapes.append([1, 8, T_PAST, 128])  # past_value

results["talker_decode"] = convert_model("talker_decode", fixed_shapes=decode_shapes)


# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name in ["codec_embed", "code_predictor_embed", "speaker_encoder",
             "code_predictor", "text_project", "talker_prefill", "talker_decode"]:
    status, detail = results.get(name, ("?", "?"))
    print(f"  {name:30s}  {status:6s}  {detail}")

print(f"\nOutput files:")
if os.path.exists(RKNN_DIR):
    for f in sorted(os.listdir(RKNN_DIR)):
        if f.endswith(".rknn"):
            size_mb = os.path.getsize(os.path.join(RKNN_DIR, f)) / (1024*1024)
            print(f"  {f:45s}  {size_mb:8.1f} MB")

print(f"\nNOTE: talker_prefill and talker_decode are fixed at trace shapes (T=8).")
print(f"For production, re-export ONNX with the desired fixed seq_len/past_len,")
print(f"then re-convert to RKNN.")
