# Qwen3 TTS + ASR — Jetson Orin NX 16GB 部署进展

## 最终状态 (2026-04-06)

### 性能指标

| 服务 | Backend | RTF | Per-token | 语言 | 特性 |
|------|---------|-----|-----------|------|------|
| TTS | C++ TRT FP16+BF16 | 0.84 | 67ms/step | 10 种 | 流式 + 声音克隆 |
| ASR | C++ TRT BF16 | 0.12-0.22 | 32-38ms/tok | 52 种 | 自动语言检测 |

### Voice-to-Voice 延迟

| 组合 | ASR | TTS | V2V (无LLM) |
|------|-----|-----|-------------|
| Sherpa TTS + Qwen3 ASR | 631ms | 741ms | **1.37s** |
| **双 Qwen3 (当前)** | ~600ms | ~1.7s | **~2.3s** |

### 内存占用

| 组合 | 占用 | 可用 | 状态 |
|------|------|------|------|
| Sherpa TTS + Qwen3 ASR | ~9.7GB | 6.3GB | ✅ 充裕 |
| **双 Qwen3** | **~14GB** | **~1GB** | ⚠️ 紧张但可运行 |

### 压测结果

| 场景 | 耗时 | GPU | RAM | 成功 |
|------|------|-----|-----|------|
| 10 并发 (5 TTS + 5 ASR) | 2.1s | 98% | 9.4GB | 10/10 |
| 5x ASR burst | 1.8s | 98% | 9.4GB | 5/5 |
| 5x TTS burst | 0.2s | 72% | 9.4GB | 5/5 |

GPU 监控: sysfs `/sys/devices/platform/bus@0/17000000.gpu/load`

---

## 架构

```
FastAPI (:8621)
├── TTS: qwen3_tts_engine.so::Pipeline
│     ├── Prefill: TRT FP16 unified engine (53ms, 无 ORT)
│     ├── Talker decode: TRT FP16 GPU-resident KV (18ms/step)
│     ├── CP ×15: TRT BF16 + GPU embed table (49ms)
│     └── Vocoder: ORT CUDA EP (chunked streaming)
│     Own Ort::Env "qwen3tts"
│
├── ASR: qwen3_tts_engine.so::ASRPipeline  
│     ├── Encoder: ORT CUDA EP (83ms)
│     ├── Prefill: ORT CUDA EP (234ms, TRT 暂不支持动态 seq)
│     ├── Decode: TRT BF16 GPU-resident KV (32ms/tok)
│     └── Embed tokens: C++ FP16→FP32 lookup
│     Own Ort::Env "qwen3asr"
│
└── Sherpa fallback: Matcha TTS + Paraformer ASR
```

### API 端点

| 端点 | 说明 |
|------|------|
| GET /health | backend + capabilities |
| GET /tts/capabilities | TTS 能力发现 |
| GET /asr/capabilities | ASR 能力发现 |
| POST /tts | 文本合成 |
| POST /tts/stream | 流式合成 (chunked vocoder) |
| POST /tts/clone | 声音克隆 (x-vector) |
| POST /tts/clone/embedding | 提取 speaker embedding |
| POST /asr | 语音识别 |
| WS /asr/stream | WebSocket 流式 ASR |

不支持的端点返回 501 + `{"required_capability": "..."}`

---

## 关键技术决策

### 1. 为什么用 C++ pybind11 而不是纯 Python

| 问题 | Python | C++ pybind11 |
|------|--------|-------------|
| ORT session 隔离 | 全局状态污染 | 独立 Ort::Env ✅ |
| TRT API | 需要 tensorrt+pycuda 包 | 直接链接 libnvinfer ✅ |
| 冷启动 | subprocess 每次 fork | 常驻内存 ✅ |
| KV cache | numpy memcpy | GPU-resident pointer swap ✅ |

### 2. 为什么热路径用 TRT，冷路径用 ORT

| 组件 | TRT | ORT CUDA EP |
|------|-----|-------------|
| Talker decode (每步) | ✅ 18ms | 38ms |
| CP ×15 (每步) | ✅ 49ms | 102ms |
| ASR decode (每 token) | ✅ 32ms | 90ms |
| Encoder (一次) | ❌ windowed attn 编不过 | ✅ 83ms |
| Vocoder (一次) | ❌ sliding-window attn | ✅ ~200ms |
| Prefill (一次) | ✅/⚠️ 动态 seq 问题 | ✅ 130-234ms |

### 3. 为什么用 BF16 而不是 FP16

Qwen3 attention 的 QK^T 中间值 >65504 (FP16 max)，导致 NaN。
BF16 指数范围和 FP32 一样 (max=3.4e38)，一行修复。
这个问题在 TTS CP 和 ASR decoder 上都出现过。

---

## 遇到的问题 + 解决方案

### P1: FP16 NaN (TTS CP + ASR decoder)
- **现象**: TRT FP16 engine 间歇性输出全零 logits
- **根因**: QK^T attention 溢出 >65504
- **解决**: `trtexec --bf16` 或 `config.set_flag(trt.BuilderFlag.BF16)`
- **影响**: 所有 Qwen3 系列 TRT engine 必须用 BF16

### P2: ORT CUDA EP 全局状态污染
- **现象**: sherpa TTS 先加载 → ASR ORT session 创建失败 (`/Squeeze_output_0`)
- **根因**: ORT CUDA EP 内部有全局 TRT cache，不同模型互相干扰
- **解决**: C++ pybind11 每个 pipeline 独立 `Ort::Env`
- **Workaround**: Python 层控制加载顺序（ASR 在 TTS 之前）

### P3: ONNX inputs_embeds seq_len 被固化
- **现象**: TRT engine 无法处理 seq_len >1 的 prefill
- **根因**: TorchScript trace 用 [1,1,1024] dummy → dim 被 TRT 推断为 static
- **解决**: 
  1. ONNX export 加 `dynamic_axes={"inputs_embeds": {1: "seq_len"}}`
  2. 或 protobuf patch 修改维度元数据
- **状态**: TTS unified engine 工作 ✅, ASR 仍用 ORT prefill

### P4: ONNX external data 文件名不匹配
- **现象**: `Failed to open file: talker_decode_ts.onnx.data`
- **根因**: ONNX 内部引用的 .data 文件名和实际文件名不同
- **解决**: symlink 或硬拷贝确保名称匹配

### P5: SimplifiedLayerNormalization (TRT 不支持)
- **现象**: 社区 ASR ONNX 的 TRT 编译失败
- **根因**: ORT opset 18 的 SimplifiedLayerNormalization，TRT 10.3 不认识
- **解决**: 手动替换为标准 op (Mul+ReduceMean+Sqrt+Reciprocal+Mul)

### P6: ScatterND reduction attribute
- **现象**: 社区 ASR ONNX TRT 编译失败
- **根因**: TRT 不支持 ScatterND 的 reduction attribute (即使值是 "none")
- **解决**: 移除 attribute

### P7: Docker 容器删除丢失模型
- **现象**: `docker rm` 后 /tmp 里的模型全没了
- **根因**: 模型放在容器 overlay 层，不在 persistent volume
- **解决**: 所有模型放 Docker volume (`speech-models:/opt/models`)

### P8: Encoder 降采样率算错 (ASR 中文)
- **现象**: 中文 ASR 只输出 "language" 一个 token
- **根因**: prompt 里 AUDIO_PAD 数量用 `mel_len//2`，实际是 `mel_len*13//100`
- **解决**: 修正公式

### P9: CP static context shape
- **现象**: code_predictor context 固定为 [1,2,1024]，步骤 >0 报错
- **根因**: ONNX 导出时 context dim 被固化
- **状态**: 调查中

### P10: 16GB 内存不够双 Qwen3
- **现象**: 两个 Qwen3 同时加载 OOM (swap, 2693ms/tok)
- **根因**: ORT prefill sessions 各占 1.7-3GB
- **解决**: TTS 用 unified TRT prefill (省 1.7GB)
- **状态**: 14/15GB 紧张运行，ASR prefill 待优化

---

## 文件结构

```
benchmark/cpp/                  C++ TRT 引擎 (12 文件)
├── tts_trt_engine.h/.cpp       TRTTalkerEngine (双缓冲 KV) + TRTCPEngine
├── tts_ort_models.h/.cpp       ORT 冷路径 (text_project, embed, vocoder 等)
├── tts_pipeline.h/.cpp         TTS 生成循环 (含 streaming)
├── asr_pipeline.h/.cpp         ASR 编码+解码循环
├── tts_binding.cpp             pybind11 (Pipeline + ASRPipeline + ASRDecoder)
├── CMakeLists.txt              构建系统
├── json_minimal.h              轻量 JSON 解析
└── main.cpp                    CLI 入口 (含 --profile)

app/                            FastAPI 服务
├── tts_backend.py              TTSBackend 抽象 + TTSCapability enum
├── asr_backend.py              ASRBackend 抽象 + ASRCapability + ASRStream
├── tts_service.py              薄代理
├── main.py                     路由 + 能力发现
└── backends/
    ├── sherpa.py               Matcha/Kokoro TTS
    ├── sherpa_asr.py           Paraformer/SenseVoice ASR
    ├── qwen3_trt.py            Qwen3-TTS C++ TRT (含 streaming)
    └── qwen3_asr.py            Qwen3-ASR C++ TRT

benchmark/                      工具脚本
├── export_sherpa_style.py      TTS ONNX 导出 (dynamic_axes 已修)
├── export_qwen3_asr.py         ASR ONNX 导出 (dynamic_axes 已修)
├── build_cp_bf16.py            TTS CP BF16 engine
├── build_asr_bf16_engine.sh    ASR decoder BF16 engine
├── extract_speaker_emb.py      声音克隆 mel→embedding
├── asr_qwen3_trt.py            Python ASR pipeline (参考)
├── tts_sherpa_trt.py           Python TTS pipeline (参考)
├── HANDOFF.md                  交接文档
├── QWEN3_ASR_PLAN.md           ASR 适配方案
└── RESULTS.md                  性能数据

deploy/
├── docker-compose.yml          基础配置
└── docker-compose.override.yml  qwen3 后端覆盖
```

## 模型位置 (Docker volume)

```
/opt/models/
├── qwen3-tts/          (8.1GB)
│   ├── onnx/           7 ONNX + .data
│   ├── engines/        talker_unified_fp16 + cp_bf16
│   └── tokenizer/      vocab.json + merges.txt
├── qwen3-asr-v2/       (7.2GB)
│   ├── encoder.onnx, decoder_prefill.onnx + .data
│   ├── decoder_step.onnx + .data
│   ├── asr_decoder_bf16.engine
│   ├── embed_tokens.bin (FP16)
│   └── tokenizer.json
├── matcha-icefall-zh-en/  (sherpa TTS)
└── paraformer-streaming/  (sherpa ASR)
```

## TRT Engine 编译命令

```bash
# TTS talker unified FP16 (支持 prefill+decode)
docker stop reachy_speech-speech-1
trtexec --onnx=talker_decode.onnx --saveEngine=talker_unified_fp16.engine --fp16 \
  --memPoolSize=workspace:2048MiB \
  --minShapes=inputs_embeds:1x1x1024,past_key_0:1x8x0x128,...(28层) \
  --maxShapes=inputs_embeds:1x200x1024,past_key_0:1x8x200x128,...

# TTS CP BF16
trtexec --onnx=code_predictor.onnx --saveEngine=cp_bf16.engine --bf16 \
  --minShapes=context:1x2x1024,gen_step:1 --maxShapes=context:1x17x1024,gen_step:1

# ASR decoder BF16
trtexec --onnx=decoder_step.onnx --saveEngine=asr_decoder_bf16.engine --bf16 \
  --minShapes=input_embeds:1x1x1024,position_ids:1x1,past_key_0:1x8x0x128,... \
  --maxShapes=input_embeds:1x1x1024,position_ids:1x1,past_key_0:1x8x500x128,...
```

## 部署流程

```bash
# 编译 C++
cd /tmp/qwen3-tts-cpp/build
cmake .. -DORT_ROOT=/tmp/ort-cpp -DCMAKE_BUILD_TYPE=Release
make -j4

# 部署
cp build/qwen3_tts_engine.*.so /home/recomputer/jetson-voice/app_overlay/

# 启动
cd /home/recomputer/jetson-voice/reachy_speech
TTS_BACKEND=qwen3_trt ASR_BACKEND=qwen3 docker compose up -d

# 测试
curl http://localhost:8621/health
curl -X POST http://localhost:8621/tts -H 'Content-Type: application/json' -d '{"text":"Hello","language":"english"}' -o test.wav
curl -X POST http://localhost:8621/asr -F 'file=@test.wav'
```

## Git Commits: 25+
