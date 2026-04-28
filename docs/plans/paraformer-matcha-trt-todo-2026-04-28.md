# Paraformer + Matcha TRT 移植 TODO（2026-04-28 收档）

主进度：**Paraformer offline + streaming 全部达 sherpa-onnx CPU baseline parity**（部分长音频反超 sherpa）。Matcha 0%，下一步主线。

## Paraformer 状态：✅ 完成

### 完成项
- ✅ Offline /asr 字符级对齐 sherpa CPU baseline
- ✅ Streaming /asr/stream sherpa parity（S6/S7 反超）
- ✅ chunk_p50 7-22ms（spec gate 80ms，4× 余量）
- ✅ first_partial 210-275ms
- ✅ BF16 dual-profile engine 部署生效
- ✅ TRT 10.3 Myelin SIGSEGV @ 400 帧（LFR fix 后消失）
- ✅ LANGUAGE_MODE=zh_en 自动路由 paraformer_trt
- ✅ force_ort 调试代码清理
- ✅ validation doc 落档

### 关键参数（终态）
```python
CHUNK_SIZE_SEC = 0.67       # 670ms
LEFT_CONTEXT_SEC = 2.68     # 2.68s
NUM_STACKED = 7
NUM_STRIDE = 6              # LFR
CIF_THRESHOLD = 1.0
RIGHT_LOOKAHEAD_LFR = 15    # defer 900ms
```

### 4 个根因（按发现顺序）
1. LFR stride 1→6（最大单点 fix，commit 625c4ac）
2. CMVN 必须全句不滑窗（e3c6079）
3. LFR 跨 chunk alignment 漂移（b0ac440）
4. 非因果 encoder right-context defer LA=15 + cif_start bug fix（8015960）

### Paraformer 后续优化（非 blocker）
- P2: CUDA Graph capture 省 ~12ms/chunk Jetson dispatch（4-8h，spec §7 P0 perf primitive，与 ASR/TTS 共用）
- P2: Decoder TRT 转换（4-8h，需绕 FSMN Cask check，省 ~5-10ms/chunk，**ROI 低不做**）
- P3: Docker image 集成（当前 app_overlay 挂载，rebuild 进 image）

## Matcha M2/M3/M6（下一步主线，预估 8-16h）

完整 spec 在 `matcha-paraformer-trt-2026-04-27.md` §2 + §6。M1 manifest 已确认 ONNX 结构（`matcha-paraformer-trt-m1-manifest-2026-04-28.md`）。

### Matcha 架构
| 组件 | 后端 | 输入 | 输出 |
|---|---|---|---|
| Text encoder | TRT | text ids | encoder hidden states |
| Estimator | TRT | hidden + noise | mel spectrogram |
| Vocoder (Vocos) | TRT | mel | STFT triple (mag/x/y) |
| ISTFT | CPU | STFT triple | waveform |

### M2 准备（先做）
- [ ] **§0 ODE step ablation（N=1/3/10）**——P0 决策点。rkvoice-stream 实测 1-step Euler dt=1.0 比 3-step 又快又准，必须先验
- [ ] RandomNormalLike 外化 + probe-first ONNX surgery（参 `rkvoice-stream/models/tts/matcha/fix_matcha_rknn.py`）
- [ ] ONNX surgery：Range/Slice 黑名单 op 替换（参 `rkvoice-stream/models/tts/matcha/analyze_matcha_onnx.py`）
- [ ] int64 → int32 cast（M1 manifest 实测）

### M2/M3 build + 实施
- [ ] Encoder TRT engine（FP16/BF16，attention/norm 强制 FP32 防溢出，同 Qwen3 经验）
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
- INT8 量化（Jetson 小 batch 反慢，memory `feedback_jetson_int8_small_batch.md`）
- 抽 `TRTAutoregressiveEngine` base class 重构

## 关键 reference 文件
- Spec: `docs/plans/matcha-paraformer-trt-2026-04-27.md`
- M1 manifest: `docs/plans/matcha-paraformer-trt-m1-manifest-2026-04-28.md`
- Validation 终态: `docs/plans/paraformer-streaming-validation-2026-04-28.md`
- Memory: `project_paraformer_streaming_sherpa_parity.md` + `feedback_streaming_baseline_first.md`
- BF16 engine 在 orin-nx `/tmp/paraformer_encoder_bf16_dp.plan`（部署到 `/opt/models/paraformer-streaming/engines/paraformer_encoder_sp1_80.plan`）
- Test 容器 orin-nx `paraformer-trt-test`（端口 18000）
- Test 音频 orin-nx `/tmp/{S0..S7}_16k.wav`
- rkvoice-stream Matcha 参考：`/Users/harvest/project/rkvoice-stream/models/tts/matcha/`（fix_matcha_rknn.py / analyze_matcha_onnx.py）+ `rkvoice-stream/rkvoice_stream/backends/tts/matcha.py`
