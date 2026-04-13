# RK3576 Speech Service

RK3576（6 TOPS NPU）上的可插拔语音服务框架。支持多种 TTS/ASR 后端，通过环境变量切换。

## 架构

```
┌─────────────────────────────────────────────────┐
│                HTTP API (:8621)                  │
│   POST /tts   POST /asr   WS /dialogue          │
├──────────────────────┬──────────────────────────┤
│     TTS Backend      │      ASR Backend          │
│  (可插拔, 环境变量切换)  │   (可插拔, 环境变量切换)    │
├──────────────────────┼──────────────────────────┤
│ • matcha_rknn (推荐)  │ • qwen3_asr_rk (推荐)    │
│ • piper_rknn         │                          │
│ • qwen3_rknn         │                          │
└──────────────────────┴──────────────────────────┘
```

每个后端内部自动选择最优执行路径（NPU / ORT CPU / 混合），调用方无感知。

## 快速开始

```bash
cd rk3576/
docker compose build
docker compose up -d
curl http://localhost:8621/health
```

## TTS 后端

通过 `TTS_BACKEND` 环境变量切换。

### matcha_rknn（推荐）

Matcha-icefall-zh-en，中英文，16kHz。

```yaml
TTS_BACKEND: matcha_rknn
MATCHA_USE_ORT: 1              # ORT CPU FP32 (推荐, 精度 9/10, ~443ms)
                                # 去掉此行则自动走 RKNN NPU (7/10, ~282ms)
MATCHA_ONNX_PATH: /path/to/model-steps-3.onnx
```

| 执行模式 | 延迟 | 精度 | 条件 |
|---------|------|------|------|
| ORT CPU FP32 | ~443ms | 9/10 | `MATCHA_USE_ORT=1` |
| RKNN split + 1-step | ~282ms | 7/10 | matcha-split/ 存在，无 ORT flag |
| RKNN 单模型 | ~535ms | 5/10 | 仅 matcha-s64.rknn |

**所需模型**:
- `model-steps-3.onnx` — Matcha 声学模型（[sherpa-onnx 官方](https://github.com/k2-fsa/sherpa-onnx)）
- `vocos-16khz-600.rknn` — Vocos vocoder（用 `scripts/convert_vocos_16khz_rknn.py` 编译）
- `matcha-icefall-zh-en/` — 文本前端（lexicon.txt, tokens.txt）

### piper_rknn

Piper VITS，多语言（en/zh/de/fr/ja），16kHz。

```yaml
TTS_BACKEND: piper_rknn
PIPER_MODEL_DIR: /opt/piper-models
PIPER_DEFAULT_LANG: en_US
```

混合部署：Encoder+DP 走 ORT CPU，Flow+Decoder 走 RKNN NPU。RTF ~0.07。

### qwen3_rknn

Qwen3-TTS 0.6B codec 模型（实验性）。

## ASR 后端

通过 `ASR_BACKEND` 环境变量切换。

### qwen3_asr_rk（推荐）

Qwen3-ASR，RKNN encoder + RKLLM decoder，52 语言，流式。

```yaml
ASR_BACKEND: qwen3_asr_rk
ASR_DECODER_TYPE: rkllm
ASR_MODEL_DIR: /opt/asr/models
```

| 指标 | 值 |
|------|-----|
| RTF | 0.44 |
| 延迟 (2s 音频) | ~1.2s |
| 流式 | 支持 (WebSocket) |

## 模型编译

NPU 模型需要在 x86 主机上用 rknn-toolkit2 编译。

```bash
# Vocos vocoder
python scripts/convert_vocos_16khz_rknn.py --input vocos.onnx --output vocos-16khz-600.rknn

# Matcha split (可选，RKNN 模式用)
python scripts/fix_matcha_rknn.py --input model-steps-3.onnx --output model-fixed.onnx
python scripts/split_matcha_rknn.py --all --onnx model-fixed.onnx

# Piper VITS
python scripts/fix_piper_rknn.py --input piper-model.onnx --output piper-fixed.onnx
```

## 自定义算子框架

本项目实现了 RKNN 自定义 CPU 算子的完整 pipeline，可用于任何需要在 NPU 模型中插入 CPU FP32 精度计算的场景：

1. ONNX 中 1:1 替换 `op_type`（不改图拓扑）
2. Toolkit 端注册 `shape_infer` + `compute`
3. Runtime 端 ctypes 调 `rknn_register_custom_ops()`
4. C 实现带 ARM NEON SIMD（`cst_ops_neon.c`）

已实现算子：Sin, Mul, Pow, Add, InstanceNorm, SplineCoupling。

详见 `app/backends/rknn_custom_ops.py`。

## 关键配置

| 变量 | 说明 |
|------|------|
| `TTS_BACKEND` | `matcha_rknn` / `piper_rknn` / `qwen3_rknn` |
| `ASR_BACKEND` | `qwen3_asr_rk` |
| `MATCHA_USE_ORT` | `1` = CPU FP32 (精度优先), 不设 = NPU (速度优先) |
| `MATCHA_ODE_STEPS` | RKNN 模式 ODE 步数, `1` 推荐 |
| `ASR_DECODER_TYPE` | `rkllm` (NPU) / `matmul` (NPU matmul API) |

## 文件结构

```
rk3576/
├── Dockerfile
├── docker-compose.yml
├── app/
│   ├── main.py                      # FastAPI 服务
│   ├── tts_backend.py               # TTS backend 接口
│   ├── asr_backend.py               # ASR backend 接口
│   ├── backends/                    # 各后端实现
│   │   ├── rknn_matcha_tts.py
│   │   ├── piper_rknn.py
│   │   ├── qwen3_asr_rk.py
│   │   ├── rknn_custom_ops.py       # 自定义算子注册
│   │   └── cst_ops_neon.c           # NEON 算子实现
│   └── qwen3asr/                    # ASR engine
├── scripts/                         # 模型编译/转换工具
│   ├── fix_matcha_rknn.py
│   ├── split_matcha_rknn.py
│   ├── convert_vocos_16khz_rknn.py
│   ├── fix_piper_rknn.py
│   └── replace_all_ops.py
└── tests/                           # pytest 测试
```

## 已知限制

- RK3576 NPU 只有 FP16，扩散/ODE 类模型精度不够 → 用 ORT CPU
- RKLLM 与 RKNN 需要 IOMMU domain 隔离 (`base_domain_id=1`)
- NPU 内存上限 ~180MB，超大模型需拆分或走 CPU
- 详细踩坑记录见 `rknn-optimization` skill
