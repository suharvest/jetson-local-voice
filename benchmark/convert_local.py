#!/usr/bin/env python3
"""
Convert Qwen3-TTS 0.6B FP32 → FP16 ONNX.
No huggingface_hub needed — expects models already at /tmp/qwen3-tts-convert/fp32/
"""
import os, sys, time, shutil
import numpy as np
import onnx
from onnx import numpy_helper

WORK = "/tmp/qwen3-tts-convert"
SRC = f"{WORK}/fp32"
DST_FP16 = f"{WORK}/fp16"

MODELS = [
    "vocoder.onnx",
    "talker_prefill.onnx",
    "talker_decode.onnx",
    "code_predictor.onnx",
    "speaker_encoder.onnx",
]

def convert_fp16(src_path, dst_path):
    name = os.path.basename(src_path)
    print(f"\n  Loading {name}...")
    t0 = time.time()
    model = onnx.load(src_path, load_external_data=True)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print(f"  Converting to FP16...")
    t0 = time.time()
    converted = 0
    for tensor in model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT:
            arr = numpy_helper.to_array(tensor)
            arr_fp16 = arr.astype(np.float16)
            new_tensor = numpy_helper.from_array(arr_fp16, name=tensor.name)
            tensor.CopyFrom(new_tensor)
            converted += 1
    print(f"  Converted {converted} tensors in {time.time()-t0:.1f}s")

    print(f"  Saving...")
    t0 = time.time()
    # Check total size to decide external data
    total = sum(t.ByteSize() for t in model.graph.initializer)
    total_mb = total / 1024 / 1024
    if total_mb > 1500:
        onnx.save(model, dst_path,
                  save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=os.path.basename(dst_path) + ".data")
    else:
        try:
            onnx.save(model, dst_path)
        except ValueError:
            # protobuf >2GB limit
            onnx.save(model, dst_path,
                      save_as_external_data=True,
                      all_tensors_to_one_file=True,
                      location=os.path.basename(dst_path) + ".data")

    sz = os.path.getsize(dst_path) / 1024 / 1024
    ext = dst_path + ".data"
    ext_sz = os.path.getsize(ext) / 1024 / 1024 if os.path.exists(ext) else 0
    print(f"  Saved in {time.time()-t0:.1f}s → {sz:.1f}MB" + (f" + {ext_sz:.1f}MB data" if ext_sz else ""))

def main():
    os.makedirs(DST_FP16, exist_ok=True)

    print("=" * 60)
    print("FP32 → FP16 ONNX Conversion")
    print("=" * 60)

    if not os.path.exists(SRC):
        print(f"ERROR: Source dir {SRC} not found!")
        print(f"Copy FP32 models here first.")
        sys.exit(1)

    for name in MODELS:
        src = f"{SRC}/{name}"
        dst = f"{DST_FP16}/{name}"

        if not os.path.exists(src):
            print(f"\n[skip] {name} not found at {src}")
            continue

        if os.path.exists(dst):
            print(f"\n[skip] {name} already converted")
            continue

        sz = os.path.getsize(src) / 1024 / 1024
        ext = f"{src}.data"
        ext_sz = os.path.getsize(ext) / 1024 / 1024 if os.path.exists(ext) else 0
        print(f"\n{'='*60}")
        print(f"{name} ({sz:.1f}MB + {ext_sz:.1f}MB external)")

        try:
            convert_fp16(src, dst)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # Copy config and tokenizer
    for item in ["embeddings", "tokenizer"]:
        s = f"{SRC}/{item}"
        d = f"{DST_FP16}/{item}"
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copytree(s, d)
            print(f"\n[copy] {item}/")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total = 0
    for f in sorted(os.listdir(DST_FP16)):
        fp = os.path.join(DST_FP16, f)
        if os.path.isfile(fp):
            s = os.path.getsize(fp) / 1024 / 1024
            total += s
            if s > 1:
                print(f"  {f}: {s:.1f}MB")
    print(f"  TOTAL: {total:.0f}MB")

if __name__ == "__main__":
    main()
