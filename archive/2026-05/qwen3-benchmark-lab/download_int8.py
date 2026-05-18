#!/usr/bin/env python3
"""Download INT8 ONNX model and list files."""
from huggingface_hub import snapshot_download
import os

print("Downloading INT8 ONNX model (sivasub987)...")
path = snapshot_download(
    repo_id="sivasub987/Qwen3-TTS-0.6B-ONNX-INT8",
    local_dir="/tmp/qwen3-tts-bench/model-int8",
)
print(f"Downloaded to: {path}")
for root, dirs, files in os.walk(path):
    for f in sorted(files):
        fp = os.path.join(root, f)
        size = os.path.getsize(fp) / 1024 / 1024
        if size > 0.5:
            print(f"  {os.path.relpath(fp, path)}: {size:.1f} MB")
