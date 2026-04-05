#!/usr/bin/env python3
"""Download GGUF Q8 model and list files."""
from huggingface_hub import snapshot_download
import os

print("Downloading GGUF Q8_0 model (cgisky)...")
path = snapshot_download(
    repo_id="cgisky/qwen3-tts-custom-gguf",
    local_dir="/tmp/models/gguf",
    allow_patterns=["gguf_q8_0/*", "tokenizer/*", "speakers/*", "onnx/*"],
)
print(f"Downloaded to: {path}")
for root, dirs, files in os.walk(path):
    for f in sorted(files):
        fp = os.path.join(root, f)
        size = os.path.getsize(fp) / 1024 / 1024
        if size > 0.5:
            print(f"  {os.path.relpath(fp, path)}: {size:.1f} MB")
