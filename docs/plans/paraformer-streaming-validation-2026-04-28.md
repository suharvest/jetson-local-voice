# Paraformer TRT Streaming 验证 — 2026-04-28

## 摘要

| 指标 | spec | 实测 | 状态 |
|---|---|---|---|
| chunk_p50 | ≤ 80 ms | 11.5 ms (S2) / 14.6 ms (S3) | **PASS** |
| 流式无 TRT 错误 | 0 | 0 | **PASS** |
| 中文 streaming vs offline 字面 | 接近 | 关键词召回但仍 ~30-40% 字差 | **PARTIAL** |
| BF16 dual-profile engine | 启用 | 已部署但 profile 0 (max=40) 弃用 | **PARTIAL** |

Latency 已经远低于 spec gate，可以投产；中文长句质量仍弱于 offline，根因待调研。

## 部署组件

- 测试容器 `paraformer-trt-test`（orin-nx，端口 18000），与生产 `reachy_speech-speech-1` 共存
- Encoder engine: `/opt/models/paraformer-streaming/engines/paraformer_encoder_sp1_80.plan`（实际为 BF16 dual-profile build）
  - profile 0: speech min=(1,4,560) opt=(1,20,560) max=(1,40,560)
  - profile 1: speech min=(1,40,560) opt=(1,200,560) max=(1,400,560)
- Decoder: ORT-CUDA fallback（FSMN Cask check 拦住 TRT，见 `paraformer-decoder-trt-diagnosis-2026-04-28.md`）
- 旧 FP32 单 profile engine 归档为 `*.fp32_singleprofile_archive`（未删）

## 修复时间线

1. **B+G fix**（commit `e3c6079`）—— `LEFT_CONTEXT_SEC = 1.0` + `_history_audio` 跨 chunk 维护，`_process_one_chunk` 拼接 history+new 后做 fbank，CIF 只取 new 部分
2. **profile_idx force=1**（同 commit）—— 跳过 TRT 10 单 context profile-switch race，短输入 pad 到 40 帧
3. **Fix A: decoder 只接收 new-frame slice**（同 commit）—— `_run_decoder(enc[:, hist_stacked:, :], …)`，cache + CIF 状态跨非重叠 frame range 推进
4. **Fix B: full-utterance CMVN**（同 commit）—— `self._all_audio` 累积，`compute_fbank(all_audio)` 后从尾切 (history+new) 喂 encoder

Bench 脚本 `tests/paraformer_trt_stream_bench.py`（commit `bdce37a`）。

## Bench 结果对比

测试音频：`/tmp/S{0..7}_16k.wav`（orin-nx），chunk 400ms 实速发送。

### 修复前（profile 阈值不匹配，41-80 帧静默失败）

| wav | streaming final | partial / chunks |
|---|---|---|
| S2 6.4s | 嗯是嗯呃我安是嗯三七十哦十 | 14/16 |
| S3 9.8s | so是不学good快走拜下离别俄语吟鹅嗯打闹一时纪应用的水 | 24/25 |

字面糊化，partial 表面看着多但内容错乱（chunk 失败回填）。

### B+G 完成、Fix A 前

| wav | streaming final | partial / chunks |
|---|---|---|
| S2 | 嗯智能哎哦啊嗯家居四嗯十 | 4/16 |
| S3 | 在深学习技术的快发展于和语言和水成已经达到了可以十点际用是用是水 | 11/25 |

S3 关键词"深度学习/快速发展/可以"已能识别。

### Fix A 后

| wav | streaming final | partial / chunks |
|---|---|---|
| S2 | 人智能哎呀五六呃收啊嗯家居到四嗯加十 | 5/16 |
| S3 | (与前接近，partial=11) | 11/25 |

Fix A 是单步最大改善（来自 codex round 2 诊断）。

### Fix A + Fix B（最终）

| wav | dur(s) | chunk_p50 | chunk_p95 | partial | streaming final | offline reference |
|---|---|---|---|---|---|---|
| S2 | 6.40 | 11.53 ms | 17.90 ms | 5/16 | 人智能哎呀五呃十啊是啊嗯是家居到四嗯家嗯十 | 嗯智能智能呃愿我生活方式同事能家居大师嗯家室 |
| S3 | 9.76 | 14.58 ms | 29.91 ms | 12/25 | 在深习技术的快发展石时别语言水长经已经到达到了可以十际用的水 | 随着深度学习技术的快速发展与实别和语合成已经达到了可以实际应用了水 |

Fix B 边际改善有限（partial 11→12 / 4→5），说明 alpha 偏移不是当前主要瓶颈。

## 已知风险与遗留问题

1. **TRT 10.3 Myelin SIGSEGV @ n_frames=400** — 已知 bug，offline 长音频曾触发批量崩溃。Streaming 单 chunk 安全（≤140 帧 with history）。详见 `project_trt103_myelin_cp_pool` memory。
2. **Profile 0（max=40）形同虚设** — 因 TRT 10 单 context profile-switch race，代码强制 profile_idx=1。未来若做正确切换或拆 2 contexts，可释放 profile 0 的短 chunk 优化空间。
3. **Streaming 质量 gap 30-40%（中文长句）** — Fix A+B 已让关键词召回，但 S2/S3 仍非 offline 等价。怀疑：
   - **D 假设（FSMN 重叠帧 first-time vs 后续 encoder 输出不一致）**——需 monotonic offset 设计，改动较大
   - **训练分布 mismatch**——paraformer-streaming 原生 chunk_size / left_chunk_size 可能不是 400ms+1s。codex 调研 ticket 已派
4. **`reachy_speech-speech-1` 在 BF16 build 时被 OOM kill** — 已重启回来，restart policy 已恢复

## 下一步建议

- **可投产**：latency 满足 spec，关键词召回可用。先打包进 v3.5 docker image
- **codex 在调研** paraformer-streaming 原生 chunking scheme（chunk_size / left_chunk / states 维护方式），结果回来后决定是否做 D 修复 / 改 chunking 方案
- 若 D 方案改动太大，可考虑：固定切到 offline-only paraformer + 端点检测（VAD）→ 用户讲话停顿后整段送 offline，牺牲实时 partial 但不牺牲质量
- Decoder TRT 转换 (P2)、Matcha M2 推进 (P0) 仍按原 TODO

## 关键 commit

- `e3c6079` perf(asr): paraformer streaming context + decoder + CMVN fixes
- `bdce37a` test(asr): WebSocket stream bench script
- `3bf809d` docs(plans): Paraformer/Matcha TRT migration TODO snapshot

## EVIDENCE 文件

- `tests/paraformer_trt_stream_bench.py` — bench script
- 远端 `/tmp/bench_fixB.json` — 最终 bench 结果
- 远端 `/tmp/paraformer_encoder_bf16_dp.plan` — 新 BF16 dual-profile engine
- 归档 `/tmp/paraformer_encoder_max400.plan.fp32_singleprofile_archive` — 旧 engine
