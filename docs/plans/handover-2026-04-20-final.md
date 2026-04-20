# Jetson-voice Handover — 2026-04-20 Final

## TL;DR

今天从 V2V crash 调了一路到 ASR 真精度。最终状态：
- **V2V crash 已修**（4 commit 链：race mutex → GIL 反转 → ModeGlobal → ThreadLocal）
- **ASR WER 从 100% 垃圾到 30%（LibriSpeech 真人英文）**，通过 ASR_TEXT anchor 修复 + hard-cap 全 decode
- **但稳定性仍不够 production**：ASR+TTS 并发会触发新一类 CUDA stream 冲突（error 906），RestartCount=2 in 10 分钟测试
- **不建议直接接 reachy-mini** — 见 §稳定性差距

## 当前分支 HEAD

- `feature/t1-cp-graph-cache`
- HEAD: `4222ecd fix(asr): full-decode window on hard cap flush — WER 80%→30%`
- main: `b4a3f2b fix(tts): cudaStreamCaptureModeThreadLocal` — 含 V2V crash 全部修复

## 今天关键 commit

### TTS 侧（V2V crash）
| Commit | 内容 |
|---|---|
| `c92315b` | 粗锁串行化（过渡） |
| `a302fef` | 定位 race 的 commit |
| `033f165` | 窄锁 + `py::gil_scoped_release` — 解 GIL 反转死锁 |
| `854730a` | `cudaStreamCaptureModeGlobal` → `ThreadLocal` — 解跨 stream 污染 |

### ASR 侧（wrapper 精度）
| Commit | 内容 |
|---|---|
| `759bfa2` | VAD silence flush + hard cap — ≤3 chunk 防 OOD |
| `cf817c6` | `ASR_TEXT` anchor 无条件 append — 100% → 80% WER |
| `4222ecd` | Hard cap 时做完整 decode 再 archive — 80% → **30% WER** |

### 过渡 commit（revert 了）
- `6403894` 去 deque maxlen（方向错）
- `5b5bdef/e2990ce/d401612` opencode PR1+PR2+left-context（精度回归）
- `beff3b5` opencode 假真人 eval（TTS 假数据）

## 性能基线（LibriSpeech 15 条真人英文）

| 指标 | 数值 |
|---|---|
| Median WER | **30%** |
| Mean WER | 32% |
| P90 WER | 45% |
| Max WER | 55% |

## V2V 延迟（Jetson 本机 `recomputer-desktop`）

| Run | 短中文 | 长中文 (3-4s) | 英文 |
|---|---|---|---|
| Cold | 1108ms | 1077ms | 1063ms |
| Warm | 756ms | 742ms | 770ms |

## 资源占用

- 容器：4.0 GB / 8 GB (50%)
- 宿主 Jetson：11 GB used / 15 GB total (73%，swap 1.2 GB)
- 主进程 RSS: 3.8 GB
- 启动：~25s 到 TRT engine ready

---

## 稳定性差距（P0，必须修才能接 reachy-mini）

### Issue 1 — ASR ORT + TTS TRT capture CUDA stream 冲突

**现象**：单 serial V2V 第 3 次崩，并发 100% 崩。错误码：
```
CUDA failure 906: operation would make the legacy stream depend 
on a capturing blocking stream
[TRT-ERR] Myelin (No Myelin Error exists)
tts_trt_engine.cpp:972 — operation failed due to a previous error during capture
```

**根因**：
- ASR 使用 ONNX Runtime CUDA EP，ORT 默认在 **legacy/default CUDA stream** 上跑
- TTS Talker 在 ThreadLocal 模式下做 `cudaStreamBeginCapture`
- 两者虽在不同 thread，但**共享 GPU context**。legacy stream 和 capturing stream 有隐式依赖 → error 906
- Capture 进入 ERROR state → 下次 EndCapture 崩 → Myelin error → 容器退出 → RestartCount++

**候选修法**（需 codex 分析出首选）：
1. **ORT CUDA EP 指定非默认 stream** — session options `user_compute_stream`（需重建 ORT session 逻辑）
2. **全局 ASR+TTS 互斥锁** — 牺牲并发，但简单
3. **禁用 CUDA Graph capture 当 ASR active** — 动态开关，但延迟会波动
4. **ASR → TRT engine（彻底离开 ORT）** — 大重构

### Issue 2 — 5-second 后偶发 TTS 生成时长失控

`test_v2v_real.py` 里英文句曾一次产 20s 垃圾音频。15 次直接 `/tts` 复测全正常。说明**特定状态组合下 CP engine 采样走偏**。未复现难修。

### Issue 3 — 宿主 swap 在用

11/15GB used + 1.2GB swap 表示物理内存紧张。加 VAD/LLM 可能 OOM。

---

## 性能优化 backlog（稳定性修完后推进）

| # | 项 | 预期收益 | 依赖 | 说明 |
|---|---|---|---|---|
| **P1-T4** | emit chunk 后再 decode | -17ms TTFT | 稳定 | Talker decode 和 chunk yield 错开 |
| **P1-T2** | Prefill KV cache | -20~30ms TTFT | 稳定 | 首次 prefill 结果缓存 |
| **P1-ASR-stream** | ASR 真流式化 | -200~500ms V2V | 稳定 | 当前 accumulate-then-finalize |
| **P2-mem** | 去 Sherpa/paraformer 冗余加载 | 释放 ~1-2GB | — | LANGUAGE_MODE=multilanguage 却仍加载 Matcha/paraformer |
| **P2-INT8** | ASR encoder INT8 | -30% enc time | — | 可能精度损失 |
| **P3-500req** | 长稳 500 req 同 shape | — | Issue 1 修 | 压测 |

## 非流式 ASR 剩余 WER 来源

30% median WER 里分解：
- ~15% 段边界词重复（"and we. we are glad"）
- ~10% 模型音素混淆（"brienne" vs "brion"）
- ~5% 首尾 filler token ("current.")

继续降需要：
- 段边界 overlap decode + dedupe
- 或切到离线 full-encode path

---

## 悬案

### 悬案 1 — TTS→ASR round-trip 精度差
TTS 合成语音有 artifact，ASR 对合成音训练分布外。V2V 转写质量 << 真人音频 ASR 质量。**评 ASR 不要用 V2V round-trip**，用真人 WAV（LibriSpeech/AISHELL）。

### 悬案 2 — opencode companion JS bug (`handleWaitAndResult is not defined`)
Session 期间偶发 `fetch failed`。跑长任务不稳。operational 问题。
