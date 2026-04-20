# P4 — Qwen3-ASR 真流式化 Spec

**Source**: codex 诊断 2026-04-20，基于 `app/backends/qwen3_asr.py` + `benchmark/cpp/asr_pipeline.cpp` + `benchmark/cpp/tts_binding.cpp`。

## 1. 当前流程

- `Qwen3ASRStream.accept_waveform`: 只 resample 到 16kHz + append 原始 samples。**无 partial encode。**
- `Qwen3ASRStream.finalize`: concat 所有 chunks → `_transcribe_python` / `_transcribe_segmented` → encoder 跑全音频 `_encoder.run(None, {"mel": mel})`。
- **已存在** `Qwen3StreamingASRStream` 类：`accept_waveform` buffer samples, `_process_chunk` 推测编码 1.2s chunks via `_run_encoder`, `finalize` flush 尾部。
- C++ 路径：`asr_pipeline.cpp:Transcribe → RunEncoder`，pybind 入口 `tts_binding.cpp:ASRPipeline.transcribe`。

⚠️ 线上现象（`dec=1288ms` 随长度线性）说明生效的是非流式路径 `Qwen3ASRStream`，而不是已有的 `Qwen3StreamingASRStream`。

## 2. 模型 check

- `encoder_fp16.onnx` 加载时未查 `get_inputs()` shapes。
- C++ `RunEncoder` 构造 `[1, 128, mel_len]` 为 dynamic T dim（运行时推断）——但代码未显式验证。
- 假设 T 动态 → 主要精度风险：**每 chunk 独立编码，encoder self-attention/conv 在 chunk 边界看不到 prior frames**。Qwen3-style encoder 很可能需要 left context。

## 3. 架构方案

**Per-chunk encoder + rolling left-context window**：

- 每 **500ms 输出一次**（50 帧 @ 10ms hop）
- 但 encoder 吃 **1.5s window** = 1.0s left context + 0.5s 新音频（~150 mel frames）
- Encoder 输出先 trim 左侧 context-derived frames（按下采样 stride），只留 50 新帧给 decoder

**延迟预算（per chunk）**：
- encoder ≤ 80ms
- prefill ≤ 60ms
- partial decode ≤ 40ms
- **总计 ≤ 200ms** 在 chunk close 后

## 4. 最小 diff

1. `qwen3_asr.py:Qwen3StreamingASRStream._run_encoder`
   - 加 context_audio 参数
   - 编码 full window，返回仅右侧帧（按 encoder 下采样 stride 跳过 context 帧）

2. `qwen3_asr.py:Qwen3StreamingASRStream.accept_waveform` / `_process_chunk`
   - 维护 raw-sample 左 context ring buffer
   - 调 `_run_encoder` 时前置 context

3. **无 pybind 改动**（streaming 留 Python 侧）。如要搬到 C++：`tts_binding.cpp` 暴露 `ASRPipeline.run_encoder(mel_array)` 返回 encoder features（当前只暴露 `transcribe`）。

4. `app/main.py` WebSocket handler：默认 backend 切到 `Qwen3StreamingASRStream`（若不是默认）。

## 5. 主要风险

**精度**：trim encoder 输出帧时 **必须知道确切下采样 factor**。搞错会默默让 token timing 错位 → CER 可能崩。

缓解：
- 代码先验证 encoder 输出 shape vs 输入 mel length 的关系
- 实现后跑 round-trip 对比：同一条 3+ 秒音频，全音频 encode vs chunked encode，CER diff ≤ 10%

## 6. 工期

- **Python-only prototype**（扩展 `Qwen3StreamingASRStream`）: **1-2 天**
- 完整版（alignment tests + LocalAgreement + latency profiling + 长音频精度对比）: **3-5 天**
- Encoder ONNX re-export: **不需要**（假设 T 已 dynamic）。仅在 T 固定或需 real KV-cache outputs 时才重导。

## 7. 实施顺序建议

1. 先验证 `encoder_fp16.onnx` 的 T 是不是 dynamic（用 onnx 检查 input shape）
2. 测当前 `Qwen3StreamingASRStream._process_chunk` 的延迟 + 精度（先不动 code）
3. 加 left context ring buffer + trim
4. 全音频 vs chunked 精度对比
5. 在 `app/main.py` WebSocket 切到 streaming backend
6. V2V 测试验证 EOS→首音频下降
