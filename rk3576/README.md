# RK3576 Speech Service

中英文 TTS + ASR 全栈语音服务，部署在 RK3576（6 TOPS NPU）上。

## 架构

```
TTS: 文本 → 文本前端(CPU) → Matcha ORT(CPU FP32) → mel → Vocos RKNN(NPU FP16) → ISTFT(CPU) → 音频
ASR: 音频 → VAD(CPU) → Encoder RKNN(NPU FP16) → RKLLM Decoder(NPU W4A16) → 文本
```

**为什么 Matcha 走 CPU 而非 NPU？**

Matcha TTS 是扩散模型（3-step ODE），RK3576 NPU 只有 FP16 精度。经过完整的精度分析（`accuracy_analysis()`、自定义算子实验、逐层 A/B 对比），确认 FP16 精度损失在 Conv/MatMul 硬件层面，无法通过自定义算子或 graph surgery 解决。ORT CPU (NEON FP32) 精度 9/10，延迟 443ms，是最优方案。

详细分析过程见 `rknn-optimization` skill 文档。

## 快速部署

### 前提条件

- RK3576 开发板（4GB+ 内存）
- Docker + docker-compose
- 设备需要以下模型文件（见下方获取方式）

### 1. 获取模型

```bash
# 在设备上创建目录
mkdir -p /home/cat/models/matcha-icefall-zh-en
mkdir -p /home/cat/matcha-data
mkdir -p /home/cat/qwen3-asr-models

# Matcha TTS 模型（sherpa-onnx 官方）
# 下载 matcha-icefall-zh-en 到 /home/cat/models/matcha-icefall-zh-en/
#   包含: lexicon.txt, tokens.txt, espeak-ng-data/
# 下载 model-steps-3.onnx 到 /home/cat/matcha-data/

# Vocos vocoder（需在 x86 上用 rknn-toolkit2 编译）
# 见下方"编译 Vocos RKNN"

# Qwen3-ASR 模型
# encoder: /home/cat/qwen3-asr-models/encoder/
# decoder: /home/cat/qwen3-asr-models/decoder/
# vad: /home/cat/qwen3-asr-models/vad/
```

### 2. 编译 Vocos RKNN（x86 上执行）

```bash
# 需要 rknn-toolkit2 >= 2.3.2
pip install rknn-toolkit2

# 编译
python rk3576/scripts/convert_vocos_16khz_rknn.py \
    --input vocos-16khz.onnx \
    --output vocos-16khz-600.rknn \
    --time-frames 600

# 传输到设备
scp vocos-16khz-600.rknn <device>:/home/cat/models/
```

### 3. 部署 Docker 服务

```bash
cd rk3576/

# 构建镜像
docker compose build

# 启动
docker compose up -d

# 检查健康
curl http://localhost:8621/health
# 应返回: {"tts":true,"tts_backend":"matcha_rknn","asr":true,...}
```

### 4. 测试

```bash
# TTS
curl -X POST http://localhost:8621/tts \
    -H "Content-Type: application/json" \
    -d '{"text":"你好世界","language":"zh"}' \
    --output test.wav

# ASR
curl -X POST http://localhost:8621/asr \
    -F "file=@test.wav"
```

## 配置说明

关键环境变量（`docker-compose.yml`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TTS_BACKEND` | `matcha_rknn` | TTS 后端选择 |
| `MATCHA_USE_ORT` | `1` | **推荐**: Matcha 走 ORT CPU FP32 (9/10 精度) |
| `MATCHA_MAX_PHONEMES` | `64` | 单段最大音素数 |
| `ASR_BACKEND` | `qwen3_asr_rk` | ASR 后端 |
| `ASR_DECODER_TYPE` | `rkllm` | ASR decoder 类型 |

### 切换到 RKNN NPU 模式（可选）

如果对延迟有极端要求（282ms vs 443ms），可以切换到 RKNN split 模式：

```bash
# 1. 生成 split 模型（x86 上执行）
python rk3576/scripts/fix_matcha_rknn.py \
    --input model-steps-3.onnx --output model-fixed.onnx
python rk3576/scripts/split_matcha_rknn.py \
    --all --onnx model-fixed.onnx --output-dir matcha-split/

# 2. 传输到设备
scp matcha-split/*.rknn matcha-split/*.npy <device>:/home/cat/models/matcha-split/

# 3. 去掉 MATCHA_USE_ORT 环境变量（docker-compose.yml）
# 服务会自动检测 matcha-split/ 目录并使用 split 模式

# 4. 设置 1-step ODE（推荐，减少 FP16 累积误差）
# 添加: MATCHA_ODE_STEPS=1
```

**注意**: RKNN 模式精度 7/10，部分音节会失真（FP16 Conv/MatMul 硬件限制）。

## 性能数据

### TTS (Matcha + Vocos)

| 模式 | 延迟 (7 tokens) | 精度 (round-trip) |
|------|-----------------|-------------------|
| **ORT CPU FP32 (推荐)** | **443ms** | **9/10** |
| RKNN split + 1-step | 282ms | 7/10 |
| RKNN 单模型 3-step | 535ms | 5/10 |

### ASR (Qwen3-ASR)

| 指标 | 值 |
|------|-----|
| RTF | 0.44 |
| 延迟 (2s 音频) | ~1.2s |

### 端到端 V2V

| 指标 | 值 |
|------|-----|
| TTS + ASR | ~1.6s |

## 文件结构

```
rk3576/
├── Dockerfile
├── docker-compose.yml
├── app/
│   ├── main.py                      # FastAPI 服务 (TTS/ASR/streaming/dialogue)
│   ├── tts_backend.py               # TTS backend 抽象
│   ├── tts_service.py               # Qwen3-TTS service (备选)
│   ├── asr_backend.py               # ASR backend 抽象
│   ├── backends/
│   │   ├── rknn_matcha_tts.py       # Matcha TTS (ORT/RKNN split/RKNN single)
│   │   ├── rknn_custom_ops.py       # RKNN 自定义算子 ctypes 注册
│   │   ├── cst_ops_neon.c           # ARM NEON 自定义算子 (Sin/Mul/Pow/Add/InstanceNorm)
│   │   ├── qwen3_asr_rk.py          # Qwen3-ASR backend
│   │   └── piper_rknn.py            # Piper VITS backend (备选)
│   └── qwen3asr/
│       ├── engine.py                # ASR engine
│       ├── stream.py                # 流式 ASR
│       └── mel.py                   # 纯 numpy STFT
├── scripts/
│   ├── fix_matcha_rknn.py           # Matcha ONNX surgery (probe-first)
│   ├── split_matcha_rknn.py         # Matcha 模型拆分 (encoder+estimator)
│   ├── convert_vocos_16khz_rknn.py  # Vocos RKNN 编译
│   ├── replace_all_ops.py           # 批量 ONNX op 替换为自定义 op
│   ├── fix_piper_rknn.py            # Piper VITS ONNX surgery
│   └── surgery_piper_custom_ops.py  # Piper 自定义算子 surgery
└── tests/                           # pytest 测试套件
    ├── conftest.py
    ├── metrics.py
    ├── test_roundtrip.py
    ├── test_tts.py
    ├── test_asr.py
    └── test_latency.py
```

## 设备上的模型目录

```
/home/cat/models/
├── matcha-icefall-zh-en/        # Matcha 文本前端 (lexicon, tokens)
├── vocos-16khz-600.rknn         # Vocos vocoder RKNN
├── matcha-s64.rknn              # Matcha 单体 RKNN (RKNN 模式备选)
└── matcha-split/                # Matcha split 模型 (RKNN split 模式用)
    ├── matcha-encoder-fp16.rknn
    ├── matcha-estimator-fp16.rknn
    └── time_emb_step{0,1,2}.npy

/home/cat/matcha-data/
└── model-steps-3.onnx           # Matcha ONNX (ORT 模式用)

/home/cat/qwen3-asr-models/
├── encoder/                     # ASR encoder RKNN
├── decoder/                     # ASR decoder RKLLM
└── vad/                         # VAD 模型
```

## 自定义算子框架

本项目实现了完整的 RKNN 自定义 CPU 算子框架，可复用于其他模型部署：

1. **ONNX 端**: 1:1 替换 `op_type`（如 `Sin` → `CstSin`），不改图拓扑
2. **Toolkit 端**: `rknn.reg_custom_op()` 注册 Python shape_infer + compute
3. **Runtime 端**: ctypes 调 `rknn_register_custom_ops()`，注册 C/NEON compute 函数
4. **C 实现**: `cst_ops_neon.c`，ARM NEON SIMD 加速

详见 `app/backends/rknn_custom_ops.py` 和 RKNN optimization skill。

## 已知限制

- RK3576 NPU 只有 FP16，扩散/ODE 类模型精度不够 → 用 ORT CPU
- RKLLM 和 RKNN 需要 domain 隔离 (`base_domain_id=1`)
- Vocos RKNN 固定 600 帧输入，超长句需要分段合成
- 英文 TTS 依赖 `espeak-ng`（Dockerfile 已安装）
