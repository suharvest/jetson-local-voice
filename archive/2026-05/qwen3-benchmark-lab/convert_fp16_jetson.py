#!/usr/bin/env python3
"""
Convert FP32 ONNX → FP16 on Jetson (memory-constrained).
Processes one model at a time, smallest first.
"""
import os, sys, time, gc, shutil
import numpy as np

SRC = "/tmp/qwen3-tts-bench/model"
DST = "/tmp/qwen3-tts-bench/model-fp16"

# Order by size (smallest first to validate pipeline)
MODELS = [
    "speaker_encoder.onnx",    # ~34MB
    "code_predictor.onnx",     # ~420MB  ← key bottleneck
    "vocoder.onnx",            # ~436MB  ← key bottleneck
    # talker_prefill/decode are 1.7GB each — try last, may OOM
]

def get_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for l in f:
                if l.startswith("VmRSS:"):
                    return int(l.split()[1]) // 1024
    except:
        pass
    return 0

def convert_one(src_path, dst_path):
    import onnx
    from onnx import numpy_helper

    name = os.path.basename(src_path)
    print(f"  Loading {name}... (RSS: {get_rss_mb()}MB)")
    t0 = time.time()
    model = onnx.load(src_path, load_external_data=True)
    print(f"  Loaded in {time.time()-t0:.1f}s (RSS: {get_rss_mb()}MB)")

    t0 = time.time()
    n = 0
    for tensor in model.graph.initializer:
        if tensor.data_type == 1:  # FLOAT
            arr = numpy_helper.to_array(tensor)
            new = numpy_helper.from_array(arr.astype(np.float16), name=tensor.name)
            tensor.CopyFrom(new)
            del arr, new
            n += 1
    print(f"  Converted {n} tensors in {time.time()-t0:.1f}s (RSS: {get_rss_mb()}MB)")

    t0 = time.time()
    try:
        onnx.save(model, dst_path)
    except ValueError:
        onnx.save(model, dst_path,
                  save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=os.path.basename(dst_path) + ".data")
    del model
    gc.collect()

    sz = os.path.getsize(dst_path) / 1024 / 1024
    ext = dst_path + ".data"
    ext_sz = os.path.getsize(ext) / 1024 / 1024 if os.path.exists(ext) else 0
    print(f"  Saved in {time.time()-t0:.1f}s → {sz:.1f}MB" + (f" + {ext_sz:.1f}MB" if ext_sz else ""))
    print(f"  (RSS after cleanup: {get_rss_mb()}MB)")

def main():
    os.makedirs(DST, exist_ok=True)
    print("=" * 60)
    print("FP32 → FP16 Conversion (Jetson, memory-aware)")
    print("=" * 60)

    for name in MODELS:
        src = os.path.join(SRC, name)
        dst = os.path.join(DST, name)
        if not os.path.exists(src):
            print(f"\n[skip] {name} not found")
            continue
        if os.path.exists(dst):
            print(f"\n[skip] {name} already done")
            continue

        sz = os.path.getsize(src) / 1024 / 1024
        ext = src + ".data"
        ext_sz = os.path.getsize(ext) / 1024 / 1024 if os.path.exists(ext) else 0
        print(f"\n{'='*60}")
        print(f"{name} ({sz:.1f}MB + {ext_sz:.1f}MB)")
        try:
            convert_one(src, dst)
        except Exception as e:
            print(f"  FAILED: {e}")

    # Copy embeddings/tokenizer
    for d in ["embeddings", "tokenizer"]:
        s, dd = os.path.join(SRC, d), os.path.join(DST, d)
        if os.path.exists(s) and not os.path.exists(dd):
            shutil.copytree(s, dd)
            print(f"\n[copy] {d}/")

    print(f"\n{'='*60}")
    print("Done. FP16 models at:", DST)
    for f in sorted(os.listdir(DST)):
        fp = os.path.join(DST, f)
        if os.path.isfile(fp) and os.path.getsize(fp) > 1024*1024:
            print(f"  {f}: {os.path.getsize(fp)/1024/1024:.1f}MB")

if __name__ == "__main__":
    main()
