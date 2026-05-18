# Qwen3-TTS 0.6B — Jetson Orin Nano 部署交接文档

## 项目概述

在 Jetson Orin Nano (15GB) 上部署 Qwen3-TTS 0.6B。目标：实时流式语音合成（RTF < 1.0）。

## 当前状态

### ✅ 已达成
- **Python BF16 pipeline: RTF=1.12, 90ms/step** — 中英文 ASR 验证完美
- GPU-resident KV cache 优化: talker 40ms→18ms
- BF16 解决 FP16 NaN: CP 3.23ms/call kernel
- sherpa-onnx C++ 编译跑通 (CUDA EP, 134ms/step)
- sherpa-onnx 兼容 ONNX 模型导出 (7 个子模型, 0 If 节点)

### ⚠️ 接近实时但未完成
- C++ TRT native API 方案设计完成，未实现
- 预计 C++ TRT native: ~67ms/step, RTF≈0.84

---

## 推荐方案: Python BF16 Pipeline

```bash
# 在容器内运行
docker exec reachy_speech-speech-1 env TRT_CP=/tmp/cp_bf16.engine \
  python3 /tmp/tts_sherpa_trt.py \
  --text "Hello, welcome to the voice synthesis system." \
  --lang english --seed 42
```

性能: 90ms/step, RTF=1.12, ASR 完美

---

## 到实时的最终方案: C++ TRT Native API

### 设计思路

```
热路径 (per-step, 需要最快):
  talker_decode → TRT native C++ API → 预编译 FP16 engine → 13.5ms
  CP × 15      → TRT native C++ API → 预编译 BF16 engine → 48.5ms

冷路径 (只跑一次或结尾):
  text_project, codec_embed, cp_embed → ORT CUDA EP (小模型)
  talker_prefill                      → ORT CUDA EP (跑一次)
  vocoder                             → ORT CUDA EP (utterance 结束时)
```

**预期: ~67ms/step, RTF≈0.84**

### 为什么不用 ORT TRT EP

| 问题 | 原因 |
|------|------|
| vocoder TRT 编译卡死 | sliding-window attention + ConvTranspose 组合, TRT 无法优化 |
| text_project TRT 编译 1h+ | 151K×2048 embedding 查找表, 不适合 TRT |
| FP16 NaN | 中间值溢出 65504, ORT TRT EP 不支持 BF16 flag |
| 容器内 OOM | 28 层 talker engine 编译需要 8GB+ 峰值内存 |

### TRT Native API 方案不存在这些问题
- **预编译 engine**: 用 trtexec/Python Builder 在宿主机编译, 不需要运行时编译
- **BF16 支持**: Python Builder API `config.set_flag(trt.BuilderFlag.BF16)`
- **混合 provider**: 热路径走 TRT native, 冷路径走 ORT CUDA EP
- **vocoder 不走 TRT**: 直接用 ORT CUDA EP

### 实现步骤

#### Step 1: 预编译 TRT Engine (已完成)

```bash
# Talker decode FP16 — 已有 /tmp/talker_decode_sherpa_fp16.engine
docker stop $(docker ps -q)  # 释放内存
trtexec --onnx=/tmp/talker_decode.onnx \
  --saveEngine=/tmp/talker_decode_fp16.engine --fp16 \
  --memPoolSize=workspace:2048MiB \
  --minShapes=inputs_embeds:1x1x1024,past_key_0:1x8x1x128,...  # 28层KV
  --optShapes=...past_key_0:1x8x30x128,...
  --maxShapes=...past_key_0:1x8x200x128,...

# CP BF16 — 已有 /tmp/cp_bf16.engine
# 用 Python Builder API (benchmark/build_cp_bf16.py):
python3 build_cp_bf16.py
# 关键一行: config.set_flag(trt.BuilderFlag.BF16)
```

#### Step 2: 在 sherpa-onnx C++ 中加 TRT Native Engine 类

在 `sherpa-onnx/csrc/offline-tts-qwen3-model.cc` 中加:

```cpp
#include <NvInfer.h>

class TRTNativeEngine {
 public:
  TRTNativeEngine(const std::string& engine_path) {
    // 加载预编译 engine
    std::ifstream file(engine_path, std::ios::binary);
    std::vector<char> data(std::istreambuf_iterator<char>(file), {});
    auto runtime = nvinfer1::createInferRuntime(logger_);
    engine_ = runtime->deserializeCudaEngine(data.data(), data.size());
    context_ = engine_->createExecutionContext();
    // 预分配 GPU buffers (双缓冲 KV cache)
    AllocateBuffers();
  }

  // GPU-resident KV cache: 只拷贝 inputs_embeds (4KB) 到 GPU,
  // logits+hidden (16KB) 回 CPU. KV cache 永远留在 GPU.
  std::pair<float*, float*> DecodeStep(const float* inputs_embeds) {
    // 1. memcpy inputs_embeds to GPU (4KB)
    // 2. set_tensor_address (pointer swap for KV, no memcpy)
    // 3. execute_async_v3
    // 4. memcpy logits+hidden back (16KB)
    // 5. swap parity (double buffer)
  }

 private:
  nvinfer1::ICudaEngine* engine_;
  nvinfer1::IExecutionContext* context_;
  // Double-buffered KV cache on GPU
  void* kv_A_[56];  // 28 keys + 28 values
  void* kv_B_[56];
  int parity_ = 0;
  int seq_len_ = 0;
};
```

#### Step 3: 修改 RunTalkerDecode 和 RunCodePredictor

```cpp
// 在 OfflineTtsQwen3Model::Impl 中:
TRTNativeEngine* talker_trt_ = nullptr;  // 可选, 有 engine 时用
TRTNativeEngine* cp_trt_ = nullptr;

TalkerDecodeResult RunTalkerDecode(...) {
  if (talker_trt_) {
    // 用 TRT native API, GPU-resident KV
    auto [logits, hidden] = talker_trt_->DecodeStep(embeds_data);
    // 包装成 Ort::Value 返回
  } else {
    // 回退到 ORT
    return RunTalkerDecodeORT(...);
  }
}
```

#### Step 4: 添加 CLI 参数

```
--qwen3-talker-decode-engine=/path/to/talker_fp16.engine
--qwen3-code-predictor-engine=/path/to/cp_bf16.engine
```

有 engine 参数时用 TRT native, 没有时回退 ORT.

### 关键注意事项

1. **TRT native API 链接**: 需要在 CMakeLists.txt 加 `-lnvinfer -lcudart`
2. **GPU-resident KV 双缓冲**: 参考 Python 的 `TRTTalkerEngine` 类 (`tts_sherpa_trt.py`)
3. **CP 的 gen_step**: BF16 engine 输入是 `[context, gen_step]`, context 每步增长
4. **数据类型匹配**: prefill 输出 FP32, TRT engine 可能期望 FP16 → 需要 cast
5. **Orin Nano SM87**: 支持 FP16 + BF16, `head_dim=128` (不是 64)

---

## 经验教训 (踩坑记录)

### 1. BF16 是唯一可行的低精度方案

| 精度 | CP kernel | NaN? | 原因 |
|------|-----------|------|------|
| FP32 | 4.5ms | ✅ | 安全 |
| FP16 | 2.3ms | ❌ | 中间值 > 65504 |
| INT8 校准 | 2.95ms | ❌ | 同 FP16 |
| FP16 混合精度 | 3.4ms | ❌ | TRT 融合掉 Cast 节点 |
| **BF16** | **3.23ms** | **✅** | **max=3.4×10³⁸, 和 FP32 同指数** |

**根因**: 不是 RMSNorm 的 sum(x²) 溢出, 是模型权重的线性变换本身产生 >65504 的中间值。
- 即使输入缩小到 [-3, 9] (pre-scale ×0.1), FP16 还是 NaN
- 即使只对 MatMul 用 FP16 其余全 FP32 (OBEY_PRECISION_CONSTRAINTS), 还是 NaN
- TRT 会融合 ONNX 图中的 Cast(FP16→FP32) 节点, 无法通过图修改解决
- **BF16 一行代码解决**: `config.set_flag(trt.BuilderFlag.BF16)`

### 2. ONNX 导出必须用 TorchScript (dynamo=False)

| 导出器 | 问题 |
|--------|------|
| **TorchScript** (`dynamo=False`) | ✅ shape 动态, 无固化 |
| Dynamo (默认) | ❌ shape 被固化, Reshape 报错 |

例外: **vocoder 必须用 dynamo** (TorchScript 不支持 `aten::__ior_`)

### 3. attention_mask 在 TorchScript trace 中被优化掉

PyTorch 的 `create_causal_mask` 用 vmap 实现, TorchScript 不能 trace。
我们用 `_simple_causal_mask` 替代, 但它从 `input_embeds.shape` 推导 mask, 
不依赖 `attention_mask` 参数 → trace 后 `attention_mask` 从 ONNX 输入中消失。

**解决**: 修改 sherpa-onnx C++ 代码使 attention_mask 可选:
```cpp
// offline-tts-qwen3-model.cc
if (talker_prefill_input_names_.size() > 1)
  inputs.push_back(std::move(attention_mask));
```

### 4. External data format 是必须的

ONNX protobuf 限制 2GB。talker (1.7GB) 和 text_project (1.2GB) 超限。
必须用 external data format:
```python
onnx.save_model(m, path, save_as_external_data=True,
    all_tensors_to_one_file=True, location=f"{name}.onnx.data", size_threshold=1024)
```

注意: `.onnx` 文件引用 `.data` 的路径是**相对的**, 两个文件必须在同一目录。
symlink 不行 (ORT 会拒绝), 必须硬拷贝。

### 5. Orin Nano 磁盘是严重瓶颈

116GB SSD, 实际可用:
- Docker images: ~20GB
- 系统 + packages: ~25GB
- 模型 (容器内): 10GB
- TRT engine cache: 3GB
- **剩余: ~10-15GB**

编译 TRT engine 需要 ~15GB 临时空间 → 必须停容器 + 清理后才能编译。
建议: 挂载 USB/NVMe 扩展存储, 或精简 Docker images。

### 6. ORT TRT EP 不适合这个模型

| 问题 | 原因 |
|------|------|
| vocoder 编译卡死 (1h+) | sliding-window attention + ConvTranspose |
| text_project 编译极慢 | 151K embedding 查找表 |
| 不支持 BF16 | ORT TRT EP 只有 `trt_fp16_enable` |
| 容器内 OOM | 28 层 talker 需要 8GB+ 编译峰值 |

**TRT native API 不存在这些问题** (预编译 engine, 不需要运行时编译)

### 7. GPU-resident KV cache 效果显著

talker decode: 40ms → 18ms (55% 加速)

原理: 56 个 KV tensor 不出 GPU, 每步只拷贝 inputs_embeds (4KB) 进 + logits (12KB) 出。
用双缓冲 (A/B set) 实现 pointer swap, 避免 input/output 地址冲突。

Python 实现: `tts_sherpa_trt.py` 的 `TRTTalkerEngine` 类

### 8. Code Predictor 的 gen_step 必须是动态的

导出 CP 时 `gen_step` 会被 TorchScript trace 折叠为常量 (trace 时值=0)。
解决: 用 stacked weights + tensor 索引替代 ModuleList 索引:
```python
self.stacked_weights = nn.Parameter(torch.stack([h.weight for h in predictor.lm_head]))
weight = self.stacked_weights[gen_step[0]]  # 动态索引, TorchScript 可 trace
```

### 9. sherpa-onnx TRT provider 默认不支持 Offline 模型

`session.cc` 中硬编码了 `provider_config == nullptr` 时 exit。
修复: 用默认 config 替代 exit:
```cpp
ProviderConfig default_trt_config;
const auto *cfg = provider_config ? provider_config : &default_trt_config;
```

### 10. torchaudio import 会导致 WSL2 上崩溃

qwen_tts 包 import chain 引入 torchaudio, WSL2 上 libtorchaudio.so 版本不兼容。
解决: 用 fake module mock:
```python
import importlib, types
spec = importlib.machinery.ModuleSpec("torchaudio", None)
ta = types.ModuleType("torchaudio"); ta.__spec__ = spec
sys.modules["torchaudio"] = ta
```

---

## 文件清单

### Pipeline 脚本 (按推荐度排序)

| 文件 | 说明 | 性能 | 质量 |
|------|------|------|------|
| `benchmark/tts_sherpa_trt.py` | **Python BF16 — 推荐** | 90ms/step RTF=1.12 | ✅ 完美 |
| `benchmark/tts_sherpa.py` | Python CUDA EP baseline | 141ms/step RTF=1.76 | ✅ |
| sherpa-onnx C++ `--provider=cuda` | C++ CUDA EP | 134ms/step | ⚠️ 质量待调 |

### 模型导出脚本

| 文件 | 说明 |
|------|------|
| `benchmark/export_sherpa_style.py` | 完整 7 模型导出 (TorchScript) |
| `benchmark/export_prefill_decode.py` | 单独导出 talker prefill/decode |
| `benchmark/build_cp_bf16.py` | BF16 CP engine 编译 |

### sherpa-onnx C++ 修改

| 文件 | 修改内容 |
|------|---------|
| `sherpa-onnx/csrc/offline-tts-qwen3-model.cc` | Ort::Value{nullptr} + attention_mask 可选 |
| `sherpa-onnx/csrc/offline-tts-qwen3-model.h` | Ort::Value{nullptr} |
| `sherpa-onnx/csrc/session.cc` | TRT offline 支持 |
| `sherpa-onnx/csrc/provider-config.h` | min_subgraph_size=20 |

### Jetson 容器文件

```
/tmp/qwen3-v2/                         # ONNX 模型 (10GB)
  text_project.onnx + .data            # 1.2GB, TorchScript
  talker_prefill.onnx + .data          # 1.7GB, TorchScript
  talker_decode.onnx + .data           # 1.7GB, TorchScript
  code_predictor.onnx                  # 421MB, TorchScript
  code_predictor_embed.onnx            # 121MB
  codec_embed.onnx                     # 13MB
  tokenizer12hz_decode.onnx            # 436MB, Dynamo
  config.json

/tmp/sherpa-onnx-offline-tts           # C++ 二进制
/tmp/libonnxruntime*.so*               # ORT 1.18.1
/tmp/qwen3-tts-bench/model/tokenizer/  # tokenizer (含 tokenizer.json)
/tmp/tts_sherpa_trt.py                 # Python BF16 pipeline
```

### Jetson 宿主机文件

```
/tmp/sherpa-onnx-build/                # sherpa-onnx 编译目录
/tmp/qwen3-models/                     # ONNX 模型副本 (宿主机)
/home/recomputer/TensorrtExecutionProvider*.engine  # 4 个 TRT cache
```

### WSL2 文件 (harve@100.73.210.80)

```
/tmp/qwen3-full/                       # PyTorch 原始模型
/tmp/qwen3-sherpa-v2/                  # 导出的完整 ONNX 集
/tmp/export_sherpa_v2.py               # 子 Agent 导出脚本
```

---

## 设备信息

| 设备 | 地址 | 用途 |
|------|------|------|
| Jetson Orin Nano | recomputer@100.67.111.58 (pw: 12345678) | 部署目标 |
| WSL2 (RTX 3060) | harve@100.73.210.80 | 模型导出 |
| 容器 | reachy_speech-speech-1 (jetson-voice:v3.0-slim) | 运行环境 |

---

## 性能演进

```
Round 1: CUDA EP (elbruno 模型)        → 163ms/step, RTF=2.0
Round 2: CUDA EP (sherpa 同源导出)      → 141ms, RTF=1.76
Round 3: TRT GPU-resident KV cache     → 128ms, RTF=1.60
Round 4: + BF16 CP engine              → 90ms,  RTF=1.12  ← 当前推荐
Round 5: C++ TRT native (预期)          → 67ms,  RTF=0.84  ← 下一步目标
```
