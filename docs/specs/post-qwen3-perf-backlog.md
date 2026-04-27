# Post-Qwen3 稳定性 & 性能 backlog

2026-04-20 定版。按优先级排。

## P0 — 稳定性（必须先于任何延迟优化）

### S1: ASR ORT + TTS TRT capture CUDA stream 冲突

**现象**：并发或连续 V2V 触发
```
CUDA failure 906: operation would make the legacy stream depend on a capturing blocking stream
Myelin error at tts_trt_engine.cpp:972
```
容器退出重启。10 分钟测试 RestartCount=2。

**诊断方向**（派 codex 深挖）：
- ORT CUDA EP 是否用默认/legacy stream？能否 `session_options.user_compute_stream` 注入独立 stream？
- TTS 的 ThreadLocal capture 能否在 ASR active 时自动降级为 no-capture（CUDA graph 不用）？
- 是否改用全局 pipeline mutex（牺牲并发，换稳定性）？
- Reproducer 最小例：ASR 并发 + TTS Talker DecodeStep 同时发。

**修法优先级**：
1. **试 ORT `user_compute_stream`**（最干净，需改 `benchmark/cpp/asr_pipeline.cpp` 或 Python ORT session 初始化）
2. **动态 disable TTS CUDA graph capture 当 ASR active**（简单，但 TTS 首音延迟会波动）
3. **全局 pipeline mutex**（兜底，牺牲并发）

**验收**：500 次 back-to-back V2V + 10 并发 client × 50 次，RestartCount=0，无 CUDA 906。

### S2: TTS 偶发 20s 英文 catastrophic 输出

现象：`test_v2v_real.py` 一次里 "Hello world how are you today" → 20s 垃圾 audio。未复现（单独 `/tts` 调 15 次正常）。
**诊断**：特定状态组合（可能 CP engine sampling + Talker KV 状态互锁）。下一次重现时先抓 `/tts` 的原始 output wav + server 侧 log。

### S3: 宿主 swap 在用

11/15GB + swap 1.2GB。加 VAD/LLM 可能 OOM。
**修**：去掉 LANGUAGE_MODE=multilanguage 下仍加载的 Matcha/paraformer 模型（`app/model_downloader.py` 无条件 ensure 全部）。预计释 1-2GB。

---

## P1 — V2V 延迟（目标 EOS→首音频 ≤ 600ms）

目前 warm 中位 ~760ms。

### T4: Emit chunk 后再做 next decode（-17ms TTFT）

TTS Talker 每 5 frame 出一个 chunk，当前 chunk yield 和 next decode 串行。改为：
- yield chunk（socket write） + decode step N+1 **并行**
- 省掉 socket write 时间的串行等待

改动：`benchmark/cpp/tts_pipeline.cpp:GenerateStreaming`。预期 TTFT -17ms。

### T2: Prefill KV cache（-20~30ms TTFT）

首次 `/tts` 的 Prefill 跑 ORT ≈ 40ms。若在 preload 阶段先 dummy prefill 一次，首音延迟降到 15ms。
改动：`backends/qwen3_trt.py:preload`。

### ASR 真流式化

当前 `Qwen3StreamingASRStream` 是 "accumulate-then-finalize partial"，每 chunk 只解 4 token + hard-cap 时做 full decode。改进：
- chunk 级 VAD + 每段独立完整 decode（不累积 archive）
- 或者切 Zipformer-based streaming ASR（丢 Qwen3 多语言能力）

预期：V2V EOS→首音频 -200~500ms。高复杂度，2-5 天。

---

## P2 — 精度（当前 LibriSpeech median WER 30%）

### E1: 段边界 dedupe

现象："and we. we are glad" — 段边界词重复。
**修**：hard cap flush 前，检查新段首几 token 是否是 archive 末尾 token 的重复，如是则跳过。

### E2: 去 `current.` filler

finalize 路径偶发吐 "current." 尾缀。不确定源头。加 `<asr_text>` anchor 后已大幅减少但偶发仍有。
**诊断**：log final decode 的原始 output_ids 看 "current" 是不是模型真吐的。

### E3: 精度实测基础设施

有 `tests/asr_real_wav_eval/` + LibriSpeech dev-clean 15 条。可加：
- AISHELL dev-clean 中文 15 条
- Common Voice zh/en 噪声样本 10 条
- CER/WER 自动 CI

---

## P3 — 长稳 & 边缘

### L1: 500 req 同 shape 压测
### L2: 长音频 (>30s) 分段回归测
### L3: 跨 session GPU 内存泄漏检查

---

## 操作性问题（非代码）

- `opencode:rescue` companion `handleWaitAndResult is not defined` JS bug — 跑长任务偶发 `fetch failed` 提前返回。需要修 companion plugin。
- opencode 倾向于自行扩张改动范围（比如从 ASR refactor 顺手 revive P4 stash）。**必须派任务时列明**"**不要动这 3 个文件之外的内容**"。
