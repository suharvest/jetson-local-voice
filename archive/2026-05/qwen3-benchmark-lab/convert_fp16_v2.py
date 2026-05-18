#!/usr/bin/env python3
"""Convert FP32 ONNX → FP16 using onnxruntime graph transformer (proper full-graph conversion)."""
import os, sys, time, gc
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

SRC = "/tmp/qwen3-tts-bench/model"
DST = "/tmp/qwen3-tts-bench/model-fp16"

MODELS = [
    "speaker_encoder.onnx",
    "code_predictor.onnx",
    "vocoder.onnx",
]

def convert_fp16_full(src_path, dst_path):
    """Convert model to FP16 including graph operations."""
    name = os.path.basename(src_path)
    print(f"  Loading {name}...")
    model = onnx.load(src_path, load_external_data=True)

    print(f"  Converting full graph to FP16...")
    t0 = time.time()

    # Convert all float initializers to float16
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            new = numpy_helper.from_array(arr.astype(np.float16), name=init.name)
            init.CopyFrom(new)

    # Convert graph input/output types
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        t = vi.type.tensor_type
        if t.elem_type == TensorProto.FLOAT:
            t.elem_type = TensorProto.FLOAT16

    # Convert node attributes that specify float type
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.FLOAT:
                attr.f = np.float16(attr.f).item()
            if attr.type == onnx.AttributeProto.FLOATS:
                attr.floats[:] = [np.float16(f).item() for f in attr.floats]
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT:
                arr = numpy_helper.to_array(attr.t)
                new = numpy_helper.from_array(arr.astype(np.float16))
                attr.t.CopyFrom(new)

    print(f"  Converted in {time.time()-t0:.1f}s")

    # Save
    t0 = time.time()
    # Remove old fp16 if exists
    if os.path.exists(dst_path):
        os.remove(dst_path)
    ext = dst_path + ".data"
    if os.path.exists(ext):
        os.remove(ext)

    try:
        onnx.save(model, dst_path)
    except ValueError:
        onnx.save(model, dst_path,
                  save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=os.path.basename(dst_path) + ".data")

    sz = os.path.getsize(dst_path) / 1024 / 1024
    ext_sz = os.path.getsize(ext) / 1024 / 1024 if os.path.exists(ext) else 0
    print(f"  Saved in {time.time()-t0:.1f}s → {sz:.1f}MB" + (f" + {ext_sz:.1f}MB" if ext_sz else ""))

    del model; gc.collect()

def main():
    os.makedirs(DST, exist_ok=True)
    print("=" * 60)
    print("FP32 → FP16 Full Graph Conversion")
    print("=" * 60)

    for name in MODELS:
        src = os.path.join(SRC, name)
        dst = os.path.join(DST, name)
        if not os.path.exists(src):
            print(f"\n[skip] {name}")
            continue
        sz = os.path.getsize(src) / 1024 / 1024
        ext = src + ".data"
        ext_sz = os.path.getsize(ext) / 1024 / 1024 if os.path.exists(ext) else 0
        print(f"\n{name} ({sz:.1f}MB + {ext_sz:.1f}MB)")
        try:
            convert_fp16_full(src, dst)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    print("\nDone.")
    for f in sorted(os.listdir(DST)):
        fp = os.path.join(DST, f)
        if os.path.isfile(fp) and os.path.getsize(fp) > 1024*1024:
            print(f"  {f}: {os.path.getsize(fp)/1024/1024:.1f}MB")

if __name__ == "__main__":
    main()
