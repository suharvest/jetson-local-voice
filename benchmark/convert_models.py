#!/usr/bin/env python3
"""
Convert Qwen3-TTS 0.6B FP32 ONNX models to:
1. FP16 ONNX (for TRT EP on Jetson)
2. QDQ INT8 ONNX (proper CUDA-compatible INT8 format)

Run on x86 machine with: pip install onnx onnxruntime numpy
"""

import os
import sys
import time
import shutil
import numpy as np

WORK = "/tmp/qwen3-tts-convert"
HF_REPO = "elbruno/Qwen3-TTS-12Hz-0.6B-Base-ONNX"

# Models to convert (name, has_external_data, priority)
MODELS = [
    ("vocoder.onnx", True, "HIGH"),          # Biggest bottleneck
    ("talker_prefill.onnx", True, "HIGH"),
    ("talker_decode.onnx", True, "HIGH"),
    ("code_predictor.onnx", False, "MED"),
    ("speaker_encoder.onnx", True, "LOW"),    # Small, FP16 enough
]


def download_model():
    from huggingface_hub import snapshot_download
    model_dir = f"{WORK}/fp32"
    if os.path.exists(model_dir) and any(f.endswith('.onnx') for f in os.listdir(model_dir)):
        print(f"[skip] Model already at {model_dir}")
        return model_dir
    print(f"[download] {HF_REPO}...")
    snapshot_download(repo_id=HF_REPO, local_dir=model_dir)
    return model_dir


def convert_fp16(src_path, dst_path):
    """Convert ONNX model from FP32 to FP16."""
    import onnx
    from onnx import numpy_helper

    print(f"  Loading {os.path.basename(src_path)}...")
    model = onnx.load(src_path, load_external_data=True)

    print(f"  Converting to FP16...")
    # Convert initializers (weights) to FP16
    converted = 0
    for tensor in model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT:
            arr = numpy_helper.to_array(tensor)
            arr_fp16 = arr.astype(np.float16)
            new_tensor = numpy_helper.from_array(arr_fp16, name=tensor.name)
            tensor.CopyFrom(new_tensor)
            converted += 1

    print(f"  Converted {converted} tensors to FP16")

    # Save without external data (FP16 is small enough to inline)
    size_mb = sum(t.ByteSize() for t in model.graph.initializer) / 1024 / 1024
    if size_mb > 1500:
        # Too large, save with external data
        onnx.save(model, dst_path,
                  save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=os.path.basename(dst_path) + ".data")
    else:
        onnx.save(model, dst_path)

    print(f"  Saved: {dst_path} ({os.path.getsize(dst_path)/1024/1024:.1f}MB)")
    return True


def convert_qdq_int8(src_path, dst_path, model_name):
    """Convert to QDQ INT8 (CUDA EP compatible) using static quantization."""
    from onnxruntime.quantization import quantize_static, CalibrationDataReader
    from onnxruntime.quantization import QuantFormat, QuantType

    class DummyCalibReader(CalibrationDataReader):
        """Generate representative calibration data."""
        def __init__(self, model_name, n=50):
            self.n = n
            self.idx = 0
            self.model_name = model_name

        def get_next(self):
            if self.idx >= self.n:
                return None
            self.idx += 1

            if "vocoder" in self.model_name:
                return {"codes": np.random.randint(0, 2048, (1, 16, 25), dtype=np.int64)}
            elif "prefill" in self.model_name:
                seq = np.random.randint(20, 100)
                return {
                    "inputs_embeds": np.random.randn(1, seq, 1024).astype(np.float32),
                    "attention_mask": np.ones((1, seq), dtype=np.int64),
                }
            elif "decode" in self.model_name:
                return {
                    "inputs_embeds": np.random.randn(1, 1, 1024).astype(np.float32),
                    "attention_mask": np.ones((1, 101), dtype=np.int64),
                }
            elif "code_predictor" in self.model_name:
                return {
                    "inputs_embeds": np.random.randn(1, 1, 1024).astype(np.float32),
                    "generation_step": np.array([0], dtype=np.int64),
                }
            elif "speaker" in self.model_name:
                return {"mel_spectrogram": np.random.randn(1, 128, 128).astype(np.float32)}
            return None

    print(f"  QDQ INT8 quantization...")
    try:
        quantize_static(
            model_input=src_path,
            model_output=dst_path,
            calibration_data_reader=DummyCalibReader(model_name),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
            op_types_to_quantize=["MatMul", "Gemm"],  # Skip Conv for vocoder safety
        )
        print(f"  Saved: {dst_path} ({os.path.getsize(dst_path)/1024/1024:.1f}MB)")
        return True
    except Exception as e:
        print(f"  QDQ failed: {e}")
        return False


def main():
    os.makedirs(f"{WORK}/fp16", exist_ok=True)
    os.makedirs(f"{WORK}/qdq_int8", exist_ok=True)

    print("=" * 60)
    print("Qwen3-TTS 0.6B Model Conversion")
    print("=" * 60)

    # Step 1: Download
    model_dir = download_model()

    # Step 2: Convert each model
    for name, has_ext, priority in MODELS:
        src = f"{model_dir}/{name}"
        if not os.path.exists(src):
            # Check in subdirectories
            for sub in ["", "embeddings/"]:
                if os.path.exists(f"{model_dir}/{sub}{name}"):
                    src = f"{model_dir}/{sub}{name}"
                    break

        if not os.path.exists(src):
            print(f"\n[skip] {name} not found")
            continue

        size = os.path.getsize(src) / 1024 / 1024
        ext_data = f"{src}.data"
        ext_size = os.path.getsize(ext_data) / 1024 / 1024 if os.path.exists(ext_data) else 0
        print(f"\n{'='*60}")
        print(f"[{priority}] {name} ({size:.1f}MB + {ext_size:.1f}MB external)")
        print(f"{'='*60}")

        # FP16
        dst_fp16 = f"{WORK}/fp16/{name}"
        if not os.path.exists(dst_fp16):
            print(f"\n  --- FP16 conversion ---")
            t0 = time.time()
            try:
                convert_fp16(src, dst_fp16)
                print(f"  Time: {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  FP16 failed: {e}")

        # QDQ INT8 (skip vocoder — keep FP16 for quality)
        if "vocoder" not in name:
            dst_qdq = f"{WORK}/qdq_int8/{name}"
            if not os.path.exists(dst_qdq):
                print(f"\n  --- QDQ INT8 conversion ---")
                t0 = time.time()
                convert_qdq_int8(src, dst_qdq, name)
                print(f"  Time: {time.time()-t0:.1f}s")
        else:
            print(f"\n  [skip QDQ] Vocoder stays FP16 (quality sensitive)")

    # Copy tokenizer & embeddings
    print(f"\n[copy] Tokenizer and embeddings...")
    for d in ["tokenizer", "embeddings"]:
        src_d = f"{model_dir}/{d}"
        if os.path.exists(src_d):
            for dst_base in [f"{WORK}/fp16", f"{WORK}/qdq_int8"]:
                dst_d = f"{dst_base}/{d}"
                if not os.path.exists(dst_d):
                    shutil.copytree(src_d, dst_d)

    # Summary
    print(f"\n{'='*60}")
    print("OUTPUT SUMMARY")
    print(f"{'='*60}")
    for variant in ["fp16", "qdq_int8"]:
        d = f"{WORK}/{variant}"
        total = 0
        print(f"\n{variant}/:")
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                fp = os.path.join(d, f)
                if os.path.isfile(fp):
                    s = os.path.getsize(fp) / 1024 / 1024
                    total += s
                    if s > 1:
                        print(f"  {f}: {s:.1f}MB")
            print(f"  TOTAL: {total:.0f}MB")

    print(f"\nModels ready at: {WORK}/")
    print("Transfer to Jetson with:")
    print(f"  scp -r {WORK}/fp16 recomputer@100.67.111.58:/tmp/qwen3-tts-bench/")
    print(f"  scp -r {WORK}/qdq_int8 recomputer@100.67.111.58:/tmp/qwen3-tts-bench/")


if __name__ == "__main__":
    main()
