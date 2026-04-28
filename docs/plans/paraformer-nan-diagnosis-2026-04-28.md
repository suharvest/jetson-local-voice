# Paraformer Encoder TRT NaN 诊断

**日期**: 2026-04-28
**根因**: FP16 TRT engine 在 19 层 CFSMN 残差累加中溢出 → alphas 全 NaN
**状态**: Sonnet + Codex 独立诊断收敛到同一结论

## 1. 二分定位

TRT 是故障边界。`_run_encoder_ort()` 和 `_run_encoder_trt()` 接收相同 stacked features；sherpa-onnx ORT 同一个 encoder.onnx 工作正常；只有 TRT FP16 plan 出 NaN → **engine 精度问题，不是前处理问题**。

代码已预见此情况：`paraformer_trt.py:532, :545` 在 preload warmup 检测 alphas NaN 并提示用 `--bf16` / `--best` 重 build。当前 `scripts/build_paraformer_trt.sh:45` 用的是 `--fp16`。

## 2. 五个 Hypothesis 裁决

| # | Hypothesis | 裁决 | 引用 |
|---|---|---|---|
| A | CMVN 错（per-utterance vs 全局） | UNCERTAIN（影响精度但不致 NaN） | `paraformer_trt.py:163` |
| B | FBank 参数不匹配 sherpa | FAIL（25ms/10ms/512/80/7/0.97 全对） | `paraformer_trt.py:65` |
| C | speech_lengths 值错 | FAIL（stack 后 frame count 正确） | `paraformer_trt.py:172` |
| D | buffer 字节大小算错 | FAIL（`* 4` 显式乘 dtype size） | `paraformer_trt.py:876` |
| E | ORT path 也 NaN | UNCERTAIN（需 runtime 验，但 sherpa baseline 排除） | - |

## 3. 真凶

FP16 TRT encoder 在 19 层 CFSMN stack 残差累加产生非有限值。Paraformer 的 CFSMN 中间激活动态范围超过 FP16，short-chunk per-utterance CMVN 让分布偏离训练，在残差路径累计放大到溢出。**与 Qwen3 attention FP16 overflow 同类问题**（memory: `feedback_bf16_attention.md` + `feedback_qwen3_rmsnorm_fp16_overflow.md`）。

## 4. 修复方案

### Patch 1（首选）：`scripts/build_paraformer_trt.sh:45`
```diff
-  --fp16 \
+  --bf16 \
```
BF16 与 FP32 同 exponent 范围，在 Orin Ampere 架构上由 TRT 10.3 原生支持。

**已知问题（Sonnet 实测）**：Orin NX 8GB 上 BF16 trtexec build OOM crash。需要 workspace size 限制或在更大 RAM 设备上 cross-build。

### Patch 2：`paraformer_trt.py` ~L845（运行时 NaN 兜底）
```diff
         if orig_n_frames < n_frames:
             enc_out = enc_out[:, :orig_n_frames, :]
             alphas_out = alphas_out[:, :orig_n_frames]
+
+        if not np.isfinite(alphas_out).all():
+            logger.error("Encoder TRT produced non-finite alphas (n_frames=%d); falling back to ORT", n_frames)
+            return None, None
```
让 `_run_encoder()` 自动 fallback 到 ORT，避免 TRT 输出污染下游。

### Patch 3（Sonnet 已实施）：preload warmup NaN 探针 + `_enc_provider` 状态字段
当 warmup 检测 NaN 时把 `_enc_provider` 设为 `"ort_cuda"`，整个 session 走 ORT。代码已在 `paraformer_trt.py` 34.9KB 版本里。

## 5. Fallback Plan

如果 BF16 TRT 还是 NaN（极小概率）或 OOM 解不掉：
- **永久 ORT-CUDA encoder + TRT decoder**（`PARAFORMER_ENC_BACKEND=ort_cuda` env flag）
- Encoder 在 latency budget 中约占 ~15ms，ORT-CUDA on Orin NX 与 TRT 大致等量级
- 这条路的 chunk_p50 / RTF 必须实测才能定

## 6. 验证流程

测试容器活在 orin-nx port 18000（paraformer_trt backend，目前全 ORT fallback 模式）：

```bash
# 1. /health 确认 backend 状态
fleet exec orin-nx -- 'curl -s http://localhost:18000/health'

# 2. WebSocket 流式测 chunk_p50
fleet exec orin-nx -- 'python3 /tmp/stream_bench.py --url ws://localhost:18000/asr/stream --audio test.wav'

# 3. /asr 离线测 RTF
fleet exec orin-nx -- 'curl -X POST http://localhost:18000/asr -F audio=@test.wav'
```

## 7. 决策树

```
现状：full ORT fallback 容器在 18000 跑着
  │
  ├─ 测 chunk_p50 / RTF
  │   ├─ PASS spec gate → ship as-is，不需要 BF16
  │   └─ FAIL gate → 走 BF16 rebuild
  │       ├─ build OOM → cross-build 或 workspace 优化
  │       └─ build OK → 重测，应该 PASS
  └─ 同时给 paraformer_trt.py 加 Patch 2 NaN 兜底
```

最高 ROI：**先测当前 ORT fallback 数字**，再决定是否投资 BF16 rebuild。
