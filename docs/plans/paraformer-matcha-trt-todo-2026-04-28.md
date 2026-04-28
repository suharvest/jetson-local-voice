# Paraformer + Matcha TRT 移植 TODO（2026-04-28 收档）

主进度：**Paraformer offline /asr 已工作**，RTF 0.047 达 spec ≤0.05。Matcha 0%。今日已 commit：`6421635 docs` / `a86c69b feat backend` / `64bf23d build scripts` / `9ef78b2 fix BPE+dedup`。

## Paraformer 待办

### P0 — 流式闭环（acceptance gate 关键）
- [ ] `/asr/stream` WebSocket 真测——chunk_p50 ≤ 80ms（spec §1 拒绝阈值 >120ms），目前**完全没量**
- [ ] `ParaformerTRTStream` 在长会话（5+ 句连续讲话）下跨 chunk 状态稳定性验证
- [ ] Streaming 路径选 dual-profile 第二 profile（`min=4 opt=20 max=80`），当前 backend 不区分 streaming/offline shape，可能 streaming 模式下激活的是大 profile 浪费 GPU
- [ ] Endpoint detection（trailing silence + utterance rules）行为对齐 sherpa（参 `app/backends/sherpa_asr.py:283-286`）
- [ ] Reset / end_utterance 控制消息测试

### P1 — 质量基线
- [ ] CER vs sherpa baseline 对比（spec gate 相对劣化 <2%，拒绝 >5%），建 golden set（aishell 子集 / librispeech 子集）
- [ ] CMVN 全局 stats（codex 怀疑但低优先）—— 找训练时 mean/var 文件加载

### P2 — 进一步性能（追 RTF < 0.03）
- [ ] Decoder TRT 转换（codex 估 4-8h，需绕 FSMN Cask check 的图改造，参 `paraformer-decoder-trt-diagnosis-2026-04-28.md`）
- [ ] 加 CUDA Graph cache（spec §7 关键 primitive，`_run_encoder_trt` 当前没 graph capture，Jetson 每 enqueue ~12ms 可省）

### P3 — 工程化
- [ ] paraformer_trt.py 集成进 Docker 镜像（当前是 app_overlay 挂载）
- [ ] `LANGUAGE_MODE=zh_en` 自动选 paraformer_trt 路径（spec §1，目前 env 手动）

## Matcha 待办（M2 / M3 / M6，预估 8-16h）

完整 spec 在 `matcha-paraformer-trt-2026-04-27.md` §2 + §6。M1 manifest 已确认 ONNX 结构。

### M2 准备
- [ ] **§0 ODE step ablation（N=1/3/10）**——P0 决策点。rkvoice 实测 1-step Euler dt=1.0 比 3-step 又快又准，必须先验
- [ ] RandomNormalLike 外化 + probe-first ONNX surgery（参 `rkvoice-stream/models/tts/matcha/fix_matcha_rknn.py`）
- [ ] ONNX surgery：Range/Slice 黑名单 op 替换（参 `rkvoice-stream/models/tts/matcha/analyze_matcha_onnx.py`）
- [ ] int64 → int32 cast (M1 manifest 实测)

### M2/M3 build + 实施
- [ ] Encoder TRT engine（FP16/BF16 attention/norm 强制 FP32）
- [ ] Estimator TRT engine（ODE step 数据 N 决定）
- [ ] Vocoder (Vocos) TRT engine（输出 mag/x/y STFT 三元组）
- [ ] `MatchaTRTBackend` 类 + `app/backends/matcha_trt.py`
- [ ] CPU ISTFT 后处理（torch.istft 或 scipy.signal.istft）
- [ ] Bucket 选择 + `mel_frames ≈ 11.9·num_tokens + 51` 公式（参 `rkvoice-stream/rkvoice_stream/backends/tts/matcha.py:426-428`）
- [ ] FP16 mel 能量异常平滑保险网（window=5 中值 / 0.5 / 1.8 阈值，参 rkvoice matcha.py:546-580）

### M2/M3 验收
- [ ] Golden round-trip test（mel L2 < 5% vs ORT baseline）
- [ ] TTFT ≤ 200ms（spec §1）
- [ ] RTF ≤ 0.15（spec §1）

### M6 集成
- [ ] Factory `TTS_BACKEND=matcha_trt` 接入
- [ ] `/tts` + `/tts/stream` API 不变性验证
- [ ] LANGUAGE_MODE 自动选择路径

## 不做（spec §5 D 明确）
- Kokoro TRT 移植
- INT8 量化（Jetson 小 batch 反慢，参 memory `feedback_jetson_int8_small_batch.md`）
- 抽 `TRTAutoregressiveEngine` base class 重构

## 关键 reference 文件
- Spec: `docs/plans/matcha-paraformer-trt-2026-04-27.md`
- M1 manifest: `docs/plans/matcha-paraformer-trt-m1-manifest-2026-04-28.md`
- 4 份诊断 doc：NaN / decoder FSMN / transcript bug / pipeline audit / CER optimization
- BF16 engine 在 orin-nx `/tmp/paraformer_encoder_bf16_max400.plan`（已部署）+ `/tmp/paraformer_encoder_bf16.plan`（旧 max=80）
- Test 容器 orin-nx `paraformer-trt-test`（端口 18000）
- Test 音频 orin-nx `/tmp/{S0..S7}_16k.wav`
