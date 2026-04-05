# Qwen3-TTS 0.6B — Jetson Orin Nano Benchmark & Lessons Learned

## 测试环境

- **设备**: recomputer-desktop (Jetson Orin Nano, 15GB shared RAM, JetPack 6 L4T R36.4.3)
- **容器**: jetson-voice:v3.0-slim
- **ORT**: 1.20.0 (CUDA + TensorRT v10.3 + CPU providers)
- **日期**: 2026-04-03

## 端到端验证结果

Pipeline 已跑通，中英文均可正确生成语音。

| 测试 | ASR 识别结果 | 帧数 | 音频时长 |
|------|-------------|------|---------|
| "Hello, welcome to the voice synthesis system." | "hello welcome to the voice synathis sts system" | 37 | 3.0s |
| "你好，欢迎使用语音合成系统。" | **"你好欢迎使用语音合成系统"** ✅ | 45 | 3.6s |

## 性能数据

### TRT FP16 引擎（trtexec 独立 benchmark）

| 组件 | FP32 CPU | TRT FP16 | 提速 |
|------|----------|----------|------|
| talker_decode (无 KV cache) | ~140ms | **12.0ms** | 11.7x |
| code_predictor (无 KV cache) | ~225ms/15步 | **36ms/15步** (2.4ms/步) | 6.3x |
| vocoder (2s 音频) | 1603ms | **65ms** | 24.7x |

### 端到端 pipeline（FP32 CPU + FP32 CUDA vocoder）

| 指标 | "Hello world." (30 frames) |
|------|---------------------------|
| Prefill | 239ms |
| Decode total | 8926ms (297ms/step) |
| Vocoder | 1603ms |
| Total | 10792ms |
| RTF | 4.50x |

### 理论最优（全 TRT FP16，trtexec 数据）

| 指标 | 预估值 |
|------|--------|
| Per-step | talker 12ms + cp 36ms = **48ms** |
| RTF | **0.60** ✅ 实时 |
| TTFA | ~70ms |
| 对比 Matcha TTFA | 60ms（接近） |

## 关键经验教训

### 1. INT8 量化：QLinearConv/MatMulInteger 在 Jetson 上不可用

`sivasub987/Qwen3-TTS-0.6B-ONNX-INT8` 使用 `onnxruntime.quantization.quantize_dynamic()` 产生的 INT8 算子（QLinearConv、MatMulInteger）在 Jetson 上：
- **CUDA EP**: 回退到 CPU，比纯 CPU 更慢（大量 CPU↔GPU memcpy）
- **TRT EP**: 无法解析 MatMulInteger 算子
- **vocoder INT8 CUDA vs FP32 CUDA**: 7023ms vs 204ms = **34.5x 慢**

**教训**: Jetson 上 INT8 必须通过 TensorRT 自己的 PTQ（post-training quantization）或 QDQ 格式。`quantize_dynamic()` 产出的 INT8 ONNX 只能在 CPU 上跑。

### 2. INT8 code predictor 精度不够

sivasub987 的 INT8 code_predictor_q.onnx 对同样输入产出完全不同的 sub-codes（和 FP32 参考对比 0/5 匹配）。

**教训**: code predictor 对量化敏感，INT8 需要校准数据，不能用 dynamic quantization。

### 3. ONNX If 节点是 TRT 转换的最大障碍

If 节点来源：
- **SDPA attention 的 `is_causal` 判断** → 解决：`attn_implementation="eager"`
- **`aten::squeeze` 操作** → 解决：用 `tensor[idx]` 替代 `squeeze()`
- **`create_causal_mask` 中的 `vmap`** → 解决：monkey-patch 返回 None

TRT 对 If 节点的限制：分支输出 shape 必须完全一致，且不支持动态 shape 在分支内。即使 `trtexec` 也无法处理。

**教训**: 导出 ONNX 前必须消除所有 If 节点。三板斧：eager attention + 避免 squeeze + patch causal mask。

### 4. DynamicCache 无法被 TorchScript trace 为 ONNX 输入

新版 transformers 的 KV cache 使用 `DynamicCache` 对象，TorchScript 无法将其 trace 为 ONNX 的显式 tensor 输入。

**解决方案**: 累积输入（stateless 模型 + 每步重新输入全部历史 embedding）。验证结果：5 步中 4 步精确匹配 KV cache 版本。对于 code predictor（5 层 transformer，seq max 16）性能影响可接受。

**教训**: 新版 transformers 与 ONNX TorchScript 导出的兼容性越来越差。考虑 `torch.export` (dynamo) 或 native TRT Python API。

### 5. 混用不同来源的 ONNX 模型会导致错误

sivasub987 的 INT8 模型（codec_embed_q, cp_embed_q）与 elbruno 的 FP32 模型（talker, code_predictor）来自不同的导出 pipeline，embedding 权重有微小差异，导致 sub-codes 退化为重复模式。

**教训**: 所有模型 + embedding 必须来自同一来源。最终方案：从 PyTorch 模型直接导出 embedding NPY 文件。

### 6. Code predictor 必须用 sampling，不能 greedy

Greedy decoding（argmax）下 sub-codes 退化为固定重复模式 [1722, 355, 310]，导致 vocoder 输出静音。PyTorch 官方 SDK 也是同样行为（greedy 模式）。

**教训**: code predictor 的 sub-code 生成必须用 top-k + temperature sampling（k=50, temp=0.9）。这是模型设计决定的，不是 bug。

### 7. Tokenizer 必须正确处理 special tokens

BPE tokenizer 从 vocab.json + merges.txt 加载时，不会识别 `<|im_start|>` 等 special tokens（拆分为多个普通 token）。

**教训**: 不用 chat template 字符串，直接用 special token IDs（如 151644）构造输入。text content 部分正常 tokenize。

### 8. FP32 talker prefill 输出格式与 INT8 不同

elbruno FP32 模型：
- prefill 输出分散 KV（`present_key_0..27`），decode 输入 stacked KV（`past_keys [28,...]`）
- 需要 `position_ids [3, 1, seq]`（M-RoPE 3D position embedding）
- hidden state 输出名为 `hidden_states`（不是 `last_hidden`）

sivasub987 INT8 模型：
- 统一的 `present_key_0..27` / `past_key_0..27` 格式
- 不需要 `position_ids`
- hidden state 输出名为 `last_hidden`

**教训**: 不同 ONNX 导出工具产出的模型接口差异很大，必须逐个检查 I/O 签名。

## 模型文件清单

### Jetson 容器内 (`reachy_speech-speech-1`)

| 路径 | 说明 | 大小 |
|------|------|------|
| `/tmp/qwen3-tts-bench/model/` | elbruno FP32 ONNX（talker, cp, vocoder） | ~5.6GB |
| `/tmp/qwen3-tts-bench/model-int8/` | sivasub987 INT8 ONNX | ~2GB |
| `/tmp/correct-emb/` | PyTorch 直出的 embedding NPY | ~200MB |
| `/tmp/trt_cache_voc/` | vocoder TRT FP16 engine cache | - |

### WSL2 (`harve@100.73.210.80`)

| 路径 | 说明 |
|------|------|
| `/tmp/qwen3-full/` | HuggingFace 原始 PyTorch 模型 |
| `/tmp/qwen3-tts-export/cp_kv_no_if.onnx` | 无 If + 内部 KV cache 的 code predictor (420MB) |
| `/tmp/qwen3-tts-export/talker_dec_kv_no_if.onnx` | 无 If + 内部 KV cache 的 talker decode (1694MB) |

### Jetson 宿主机 (`/tmp/`)

| 文件 | 说明 |
|------|------|
| `cp_kv_no_if.onnx` | 无 If code predictor (420MB) |
| `cp_fp16.engine` | 旧版 stateless CP TRT engine |
| `talker_decode_fp16.engine` | 旧版 stateless talker TRT engine |

## 优化路线图

### Phase 1: Code Predictor TRT ✅ 方案确定

用 `cp_kv_no_if.onnx` 的 stateless 累积输入模式：
- 每步把之前的 embedding 拼接后重新输入
- seq 从 2 增长到 16，TRT FP16 下 ~5-10ms 总计
- 验证：4/5 步精确匹配 KV cache 版本

TODO: 编译带动态 seq 的 TRT engine，集成到 pipeline

### Phase 2: Talker Decode GPU 加速 ✅ 完成

问题：FP32 talker with KV cache 有 If 节点 → TRT 无法转换

最终方案：新导出的 `talker_dec_kv_no_if.onnx` 成功：
- 无 If 节点 + 显式 KV cache（past_keys/values 作为 ONNX 输入/输出）
- trtexec 编译成功：**14.2ms/step** FP16
- DynamicCache 在 TorchScript trace 时被正确展开为显式 tensor I/O

中间验证：CUDA EP FP32 = 41ms/step（作为 fallback 方案）

### Phase 2.5: TRT Native API 集成 ✅ Talker 成功

**问题**: ORT TRT EP 在容器内编译 talker engine OOM（28 层模型需 ~8GB 编译峰值）

**解决**: 
1. 用 `trtexec` 在宿主机编译 TRT engine（停掉所有容器释放内存）
2. 从 vision-trt 容器拷贝 `tensorrt` + `pycuda` Python 包到 speech 容器
3. 用 TRT 原生 Python API 加载预编译的 `.engine` 文件

**结果**: `benchmark/tts_trt_native.py`
- Talker prefill: ORT CUDA EP（FP32，跑一次）
- Talker decode: **TRT FP16 native API = 28ms/step**（vs CUDA EP 44ms）
- Code predictor: ORT CUDA EP（FP32，137ms/step — 仍是瓶颈）
- Vocoder: ORT CUDA EP（855ms）

**注意事项**:
- Prefill 不能用 decode engine 模拟（逐 token 喂给 decode 模型会产生错误 KV cache，因为没有 causal mask）
- TRT engine 的 KV cache 是 FP16，ORT prefill 输出是 FP32 → 需要 `.astype(np.float16)` 转换
- FP32→FP16 精度截断导致输出轻微质量下降（"synthesis system" → "in the sysy"）
- 每次 `run()` 重新分配 GPU buffer（39ms），可通过预分配优化到接近 trtexec 的 14ms

**TRT engine 编译命令**（需在宿主机停掉所有容器后执行）:
```bash
# Talker decode (28层, 带KV cache, 无If节点)
trtexec --onnx=/tmp/talker_dec_kv_no_if.onnx \
  --saveEngine=/tmp/talker_dec_kv_fp16.engine --fp16 \
  --memPoolSize=workspace:2048MiB \
  --minShapes=inputs_embeds:1x1x1024,position_ids:3x1x1,past_keys:28x1x8x1x128,past_values:28x1x8x1x128 \
  --optShapes=inputs_embeds:1x1x1024,position_ids:3x1x1,past_keys:28x1x8x30x128,past_values:28x1x8x30x128 \
  --maxShapes=inputs_embeds:1x1x1024,position_ids:3x1x1,past_keys:28x1x8x80x128,past_values:28x1x8x80x128

# Code predictor (5层, 内部KV, 无If节点)  
trtexec --onnx=/tmp/cp_kv_no_if.onnx \
  --saveEngine=/tmp/cp_kv_fp16.engine --fp16 \
  --memPoolSize=workspace:1024MiB \
  --minShapes=inputs_embeds:1x1x1024,generation_steps:1 \
  --optShapes=inputs_embeds:1x2x1024,generation_steps:1 \
  --maxShapes=inputs_embeds:1x17x1024,generation_steps:1
```

**将 tensorrt/pycuda 安装到 slim 容器**:
```bash
# 从 vision-trt 容器拷贝
docker cp vision-trt:/usr/local/lib/python3.10/dist-packages/pycuda speech-container:/usr/local/lib/python3.10/dist-packages/
# 从宿主机拷贝 tensorrt（JetPack 自带）
docker cp /usr/lib/python3.10/dist-packages/tensorrt speech-container:/usr/local/lib/python3.10/dist-packages/
docker cp /usr/lib/python3.10/dist-packages/tensorrt_dispatch speech-container:/usr/local/lib/python3.10/dist-packages/
docker cp /usr/lib/python3.10/dist-packages/tensorrt_lean speech-container:/usr/local/lib/python3.10/dist-packages/
```

### 性能对比汇总（Round 1: elbruno 模型）

| Pipeline | Talker | CP x15 | Per-step | RTF | 正确性 |
|----------|--------|--------|----------|-----|--------|
| FP32 CPU | 140ms | 225ms | 365ms | 4.5x | ✅ |
| CUDA EP | 44ms | 122ms | 163ms | 2.0x | ✅ |
| TRT talker + CUDA CP | 28ms | 137ms | 165ms | 2.1x | ⚠️ |
| TRT 全链路 native | 14.8ms | 41.3ms | 56.1ms | 0.70x | ⚠️ FP16 精度 |

### 性能对比汇总（Round 2: sherpa-style 同源导出模型）

| Pipeline | Talker | CP x15 | Per-step | RTF | 正确性 |
|----------|--------|--------|----------|-----|--------|
| CUDA EP (sherpa) | 38ms | 105ms | 141ms | 1.76x | ✅ |
| TRT talker + CUDA CP | 29ms | 105ms | 139ms | 1.74x | ✅ |
| **TRT GPU-resident KV + IOBind** | **18ms** | **110ms** | **128ms** | **1.60x** | ✅ |
| trtexec benchmark | 13.5ms | 35ms(FP16)/68ms(FP32) | 49-82ms | 0.6-1.0x | — |

### Round 2 关键改进
- Prefill/decode 同源导出 → KV cache 兼容，无质量降级
- GPU-resident 双缓冲 KV cache → talker 从 40ms 降到 18ms
- CP FP16 TRT 有 layernorm 精度溢出 → 保持 ORT CUDA EP FP32
- ASR 验证: EN "hello welcome to the voice synthesis sy" ✅ / ZH "你好欢迎使用语音合成系统" ✅

脚本: `benchmark/tts_sherpa_trt.py`（最优）, `benchmark/tts_sherpa.py`（CUDA EP baseline）

### Round 3: BF16 突破 + sherpa-onnx C++ 编译

**关键发现: BF16 解决了 FP16 NaN 问题**

FP16 NaN 的根因不是 QK MatMul 溢出（FP32 累加器），而是：
- 模型权重的线性变换产生中间值 > 65504 (FP16 max)
- RMSNorm 的 sum(x²) 溢出 FP16
- TRT 的算子融合会跨越 Cast 节点，无法通过 ONNX 图修改解决

BF16 max = 3.4×10³⁸（和 FP32 同指数范围），彻底避免溢出。

| CP 精度 | Kernel/call | 15步 | NaN | ASR |
|---------|------------|------|-----|-----|
| FP32 | 4.5ms | 68ms | 0 | ✅ |
| FP16 | 2.3ms | 35ms | ❌ 全NaN | — |
| INT8 校准 | 2.95ms | 44ms | ❌ NaN | — |
| **BF16** | **3.23ms** | **48ms** | **0** | **✅ 完美** |

BF16 编译命令（一行）:
```python
config.set_flag(trt.BuilderFlag.BF16)  # 替代 FP16
```

性能对比（完整 pipeline）:

| Pipeline | Talker | CP×15 | Per-step | RTF | ASR |
|----------|--------|-------|----------|-----|-----|
| CUDA EP | 38ms | 105ms | 141ms | 1.76 | ✅ |
| TRT GPU-resident KV | 20ms | 110ms | 128ms | 1.60 | ✅ |
| **TRT + BF16 CP** | **20ms** | **88ms** | **90-114ms** | **1.1-1.4** | **✅** |
| 理论极限 (C++ native) | 13.5ms | 49ms | 63ms | 0.79 | — |

**sherpa-onnx C++ 编译**:
- HeiSir2014 fork (`develop` branch) 在 Jetson 原生编译成功
- 需修复 `Ort::Value` 默认构造（`{nullptr}`）
- 二进制: `/tmp/sherpa-onnx-build/build/bin/sherpa-onnx-offline-tts`
- 磁盘空间不足未能完成端到端测试（模型 ~8GB，可用 ~7GB）

下一步:
1. 清理磁盘或在容器内编译 sherpa-onnx
2. 用 C++ runtime 跑完整 pipeline（预期 ~73ms per-step, RTF 0.91）
3. 集成 BF16 TRT engine 到 C++ runtime

### 历史瓶颈: Code Predictor

CP 从 225ms (FP32 CPU) → 122ms (CUDA EP) → 41ms (TRT FP16) 逐步优化。

CP TRT 历程:
- `cp_kv_no_if.onnx` 编译成功（trtexec 2.3ms/step）
- 但 ORT TRT EP 在容器内加载后，累积输入模式产出 NaN
- 用 TRT native API 加载 `cp_kv_fp16.engine` 尚未测试

**下一步: 用 TRT native API 加载 CP engine，搭配 ORT CUDA EP 的 FP32 code_predictor 做 KV cache 管理**

### Phase 3: INT8 校准

- **目标**: talker_decode INT8 TRT（FP16 12ms → INT8 ~8-10ms）
- **校准数据**: 从 tts_fp32.py 的 decode 循环保存中间张量，50-100 句中英文
- **工具**: Python TRT API `IInt8EntropyCalibrator2`
- **注意**: code predictor 和 vocoder 不做 INT8
