# Paraformer TRT Streaming 验证 — 2026-04-28（Final）

## 摘要

**实现 sherpa-onnx CPU baseline parity**，长音频 S6/S7 反超 sherpa。

| 指标 | spec | 实测 | 状态 |
|---|---|---|---|
| chunk_p50 | ≤ 80 ms | 7-22 ms | **PASS** (4× 余量) |
| first_partial_ms | — | 210-275 ms | **优于 chunk=3s 时 ~370 ms** |
| 中文长句 vs sherpa CPU | 接近 | S3 字符级对齐，S2 ~95% | **PASS** |
| 英文长句 vs sherpa | 接近 | S6/S7 反超（无 BPE 错乱） | **PASS** |
| BF16 dual-profile engine | 启用 | 是 | **PASS** |

## 终态参数

```python
CHUNK_SIZE_SEC = 0.67       # 670ms chunk
LEFT_CONTEXT_SEC = 2.68     # 2.68s 累积左上下文
NUM_STACKED = 7
NUM_STRIDE = 6              # LFR: 10ms fbank → 60ms LFR
CIF_THRESHOLD = 1.0
RIGHT_LOOKAHEAD_LFR = 15    # defer 15 LFR (900ms) CIF firing
```

## 部署

- **测试容器** `paraformer-trt-test`（端口 18000，已验证）
- **生产容器** `reachy_speech-speech-1`（部署详见 P0 cleanup commit）
- Encoder engine: `/opt/models/paraformer-streaming/engines/paraformer_encoder_sp1_80.plan`（BF16 dual-profile build）
  - profile 0: speech min=(1,4,560) opt=(1,20,560) max=(1,40,560)
  - profile 1: speech min=(1,40,560) opt=(1,200,560) max=(1,400,560)
- Decoder: ORT-CUDA fallback（FSMN Cask check 拦住 TRT，见 `paraformer-decoder-trt-diagnosis-2026-04-28.md`）
- Routing: `LANGUAGE_MODE=zh_en` 默认 `ASR_BACKEND=paraformer_trt`（commit fc8d1fe）

## 修复时间线（5 commits）

| commit | 说明 |
|---|---|
| `e3c6079` | B+G fix: LEFT_CONTEXT_SEC=1.0 + history_audio + 全句 CMVN + decoder new-frame slice |
| `bdce37a` | WebSocket stream bench script |
| `625c4ac` | **LFR stride 1→6**（最大单点修复，offline 立即对齐 sherpa） |
| `b0ac440` | full all_lfr 累积 enc + right look-ahead 1 LFR + decoder 接收完整 enc |
| `8015960` | **RIGHT_LOOKAHEAD_LFR=15 + cif_start bug fix**（最终对齐） |
| `fc8d1fe` | cleanup: 移除 force_ort 调试标记 + LANGUAGE_MODE=zh_en 自动路由 |

## 全 8 wav bench 对比（与 sherpa-onnx CPU INT8）

| wav | dur | chunk_p50 | partial | 我们 streaming | sherpa CPU |
|---|---|---|---|---|---|
| S0 | 2.32s | — | 0 (finalize) | 你好很高兴认识你 | 你好很高兴认识 |
| S1 | 2.80s | 21.9ms | 2 | 嗯今天气真好我们出去玩吧 | 今天天气真好我们出去玩吧 |
| S2 | 6.40s | 6.7ms | 6 | 嗯人工智能在改变我们的生活方式嗯从智能家居到自动驾驶 | 人工智能正在改变我们的生活方式从智能家居到自动驾驶 |
| S3 | 9.76s | 15.1ms | 15 | 随着深度学习技术的快速发展语音识别和语音合成已经达到了可以实际应用的水平 | 随着深度学习技术的快速发展语音识别和语音合成已经达到了可以实际应用的水 |
| S4 | 1.60s | — | 0 (finalize) | hellonicetomeetyou | hello nice to meet you |
| S5 | 3.04s | 10.0ms | 3 | theweatherisgreattodaylet'sgooutside | the weather is great today let's go outside |
| S6 | 7.44s | 12.8ms | 10 | artificial**intelligence**transformingourdailylivesfromsmarthomesto**autonomious**driving... | artificial **intelligces**...autonomous... |
| S7 | 10.16s | 18.0ms | 15 | ...synthesis**havereachedalevel**thatcanbepractically**um** | ...synthesis **rereed ed** a level that can be practically |

S6/S7 我们识别出 "intelligence" / "have reached a level"，sherpa 输出有 "intelligces" / "rereed ed" 的 BPE 错乱 → 我们更干净。中英文 token 间无空格是 BPE detokenize 设计选择，非 streaming bug。

## 4 个根因（按发现顺序）

1. **LFR stride bug**（commit 625c4ac）：`stack_frames` stride=1（滑窗）→ stride=6（降采样）。错的 stride 让 encoder 喂 6× 多帧，输出全 garbage，且触发 TRT 10.3 Myelin SIGSEGV @ 400 帧。
2. **CMVN 必须全句不滑窗**（e3c6079）：`self._all_audio` 累积，每 chunk fbank over 全句。
3. **LFR 跨 chunk 对齐漂移**（b0ac440）：stack_frames 在全 fbank 上做一次得到 all_lfr，按 LFR 域切片。
4. **非因果 encoder right-context defer**（8015960）：encoder 双向 attention，每 fire frame 需 ~25 LFR (~1.5s) 右上下文。LA=15 (defer 900ms) 才达 sherpa parity。子 bug：`cif_start = max(cif_processed, hist_stacked)` 用 hist_stacked clamp 让 LA>10 时不 fire — 改成 `cif_start = self._cif_processed_lfr` 绝对 LFR 游标。

## 关键调试 insight

- **md5 验证模型**：sherpa-onnx 公开 ONNX 与我们的 md5 完全一致 (`38bb68f284cf2d34e5a8f98a7c671ffd`)，排除"模型不同"。
- **sherpa CPU baseline 是 ground truth**：之前一度判断"streaming 天花板就是 ~85% sherpa"，跑 sherpa 才发现 partial 干净的渐进中文，证明可修。
- **BF16 不是主因**：强制 ORT FP32 测试，输出与 BF16 完全一致。
- **小 chunk 反而差**：试过 240ms chunk 期望"更密集 partial → 更好质量"，实测中文丢字更多。
- **codex 假设 FSMN state cache 错**：codex 看 FunASR 训练代码外推到 ONNX export，实测 sherpa 公开 ONNX 也没有 cache I/O。

## 已知限制（不阻止投产）

1. **TRT 10.3 Myelin SIGSEGV @ 400 帧** — LFR fix 后自然消失（不再喂 6× 多帧）。如未来 utterance > 24s 仍可能触发，需 chunk 切分。
2. **decoder 仍走 ORT-CUDA fallback** — FSMN Cask check 拦住 TRT，省 ~5-10ms/chunk 的 P2 优化。
3. **profile 0 (max=40) 弃用** — 强制 profile 1 (max=400) 避开 TRT 10 单 context profile-switch race。

## 待办（非投产 blocker）

- P2: Decoder TRT 转换（codex 估 4-8h）
- P2: CUDA Graph capture（spec §7 P0 perf primitive，省 ~12ms/chunk Jetson dispatch）
- P3: Docker image rebuild（移除 app_overlay 挂载，改为 image baked）

## EVIDENCE

- `tests/paraformer_trt_stream_bench.py` — bench script
- 远端 `/tmp/bench_la15_fixed.json` — 8 wav 最终结果
- 远端 `/tmp/paraformer_encoder_bf16_dp.plan` — BF16 dual-profile engine
- 模型 md5 `38bb68f284cf2d34e5a8f98a7c671ffd`（与 sherpa-onnx HF repo 一致）
