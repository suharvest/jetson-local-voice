# Archived Qwen3 Temporary Scripts

These files were one-off investigation scripts from the Qwen3 ASR/TTS Orin optimization session.

They are intentionally kept out of the active root directory so the maintained paths are easier to review:

- `official`: minimal-diff EdgeLLM semantics/build/example path.
- `highperf`: Jetson Voice product path for Orin dual-resident low-latency V2V.

Do not use these archived scripts as product regression entry points. Promote a script back into `scripts/` or `bench/` only after giving it a stable CLI, tests, and documentation.

Large generated model artifacts are not part of the archive commit. In particular, `tmp_code2wav_onnx_real/` is ignored because it contains a large ONNX external-data file.
