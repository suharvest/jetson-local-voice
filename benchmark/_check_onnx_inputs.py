#!/usr/bin/env python3
import sys

for path in [
    "/Users/harvest/project/jetson-voice/models/qwen3-tts/onnx/code_predictor.onnx",
    "/Users/harvest/project/jetson-voice/models/qwen3-tts/onnx/vocoder_fp16.onnx",
]:
    data = open(path, "rb").read()
    print(f"\n=== {path.split('/')[-1]} ===")
    for n in [b"context", b"gen_step", b"logits", b"inputs_embeds", b"cache_position",
              b"past_length", b"past_key", b"logits_all", b"audio_codes", b"audio_values"]:
        idx = data.find(n)
        status = f"FOUND at {idx}" if idx >= 0 else "NOT FOUND"
        print(f"  {n.decode()}: {status}")
