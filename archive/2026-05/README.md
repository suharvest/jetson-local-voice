# Archive: 2026-05

This directory keeps historical experiments out of the repository root while
preserving scripts and notes that may still be useful for debugging or
reproducing earlier Qwen3 engine work.

- `qwen3-benchmark-lab/` was the old top-level `benchmark/` directory. It
  contains prototype Qwen3 ASR/TTS export, TensorRT, RKNN, and C++ runtime
  experiments.
- `tts-model-benchmarks/` was the old top-level `benchmarks/` directory. It
  contains one-off TTFT comparison scripts for F5-TTS, CosyVoice3, and Matcha.

Current product and performance harnesses live in `bench/`.
