# RK3576 语音服务项目状态

> 更新: 2026-04-07

## 当前部署状态

**Docker 容器 `rk3576-speech` 运行中**，端口 8621，TTS + ASR 双服务。

```
GET /health → {
  "tts": true,  "tts_backend": "qwen3_rknn",
  "asr": true,  "asr_backend": "qwen3_asr_rk",
  "streaming_asr": true
}
```

### 硬件
- 设备: RK3576 LubanCat-3, 8GB RAM, 6 TOPS NPU
- Tailscale: cat-remote (100.89.94.11)
- 内存占用: ~5.8GB / 8GB（不能和 ai_lab_llm 共存）

---

## TTS 配置 (Qwen3-TTS 0.6B)

```
Talker:          RKLLM W4A16 Qwen3  → 43ms/步 (不可替代, SRAM+Flash Attention 闭源优化)
Code Predictor:  C engine W8A16      → 11.5ms/步 (rknn_matmul_api, per-column INT8)
Vocoder:         RKNN ctx25 INT8     → 620ms/chunk (2s 音频, RTF=0.31)
Speaker Encoder: ONNX CPU            → 24ms (声音克隆, 未集成到 Docker)
```

### TTS 性能

| 指标 | 值 |
|------|-----|
| AR 帧率 | 5.8 fps (172ms/帧) |
| 每帧构成 | talker 43ms + CP 11.5ms + overhead |
| RTF | ~4.86 (含 vocoder) |
| 首字延迟 (TTFT) | ~4s (25帧攒够后 vocoder) |

### TTS 码本质量

| 码本数 | 听感 | ASR 验证 | RTF (预估) |
|--------|------|---------|-----------|
| 16 | 最好 | 3/3 ✅ | 2.1 |
| 12 | 好 | 3/3 ✅ | ~1.6 |
| **10** | **可接受（最低）** | 3/3 ✅ | ~1.5 |
| 8 | 不够好 | 2/3 ⚠️ | 1.27 |
| 6 | 差 | 需听 | 1.06 |

> 用户确认：**至少 10 码本** 才有可接受听感。

---

## ASR 配置 (Qwen3-ASR 0.6B)

基于社区项目 [qzxyz/qwen3asr_rk](https://huggingface.co/qzxyz/qwen3asr_rk) 集成。

```
Encoder:  RKNN FP16 merged    → 431ms/4s-chunk (INT8 不可用, 质量崩)
Decoder:  RKLLM W4A16 Qwen3   → ~43ms/token (和 TTS talker 同架构)
VAD:      Silero ONNX CPU      → 可选
```

### ASR 性能

| 指标 | 值 |
|------|-----|
| RTF | 0.44 (enc 431ms + llm 455ms per 4s chunk) |
| 流式 chunk | 4s (推荐) |
| 说完→出文字 | ~0.5-0.6s (流式模式) |
| 语言 | 52 种 (中英日韩等) |
| language=auto | ✅ 已修复 (decoder 自动检测) |

### ASR 流式机制

```
音频分 4s chunk → RKNN encoder → RKLLM decoder → 文字
  - 滑动窗口 memory_num=2 (保留最近 2 个 chunk embedding)
  - token rollback=2 (修复 chunk 边界错误)
  - VAD 门控 (可选, 静音不触发 ASR)
  - 投机编码 (speculative encoding, 预编码 50% buffer)
```

---

## Docker 部署

### 镜像
```bash
docker build --network=host -t rk3576-speech:latest -f Dockerfile .
```

### 运行
```bash
docker run -d --name rk3576-speech \
  --privileged --network=host \
  -v /dev:/dev \
  -v /home/cat/qwen3-tts-rknn:/opt/tts/models:ro \
  -v /home/cat/models/talker_fullvocab_fixed_w4a16_rk3576.rkllm:/opt/tts/models/talker_fullvocab_fixed_w4a16_rk3576.rkllm:ro \
  -v /home/cat/qwen3-asr-models:/opt/asr/models:ro \
  -e MODEL_DIR=/opt/tts/models \
  -e TTS_BACKEND=qwen3_rknn \
  -e ASR_BACKEND=qwen3_asr_rk \
  -e ASR_MODEL_DIR=/opt/asr/models \
  --restart unless-stopped \
  rk3576-speech:latest
```

### 锁频（提升 ~27% 性能，需要 root）
```bash
# CPU 大核 2.2GHz
echo userspace > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo 2208000 > /sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed
echo 2208000 > /sys/devices/system/cpu/cpufreq/policy6/scaling_setspeed

# NPU 1GHz
echo userspace > /sys/class/devfreq/fdab0000.npu/governor
echo 1000000000 > /sys/class/devfreq/fdab0000.npu/userspace/set_freq

# DDR 2.7GHz
echo userspace > /sys/class/devfreq/dmc/governor
echo 2736000000 > /sys/class/devfreq/dmc/userspace/set_freq
```

### 模型文件位置 (cat-remote)

**TTS** (`/home/cat/qwen3-tts-rknn/`, 挂载到 `/opt/tts/models/`):
```
talker_fullvocab_fixed_w4a16_rk3576.rkllm  (683MB, 另外挂载)
text_project.rknn                           (606MB)
decoder_ctx25_int8.rknn                     (133MB)
code_predictor.rknn                         (174MB, RKNN fallback)
codec_embed.rknn                            (6MB)
code_predictor_embed.rknn                   (4MB)
codec_head_weight.npy                       (13MB)
codebook_embeds/                            (~32MB)
cp_weights/                                 (含 FP16 .bin + W8A16 .int8.bin/.scales.bin)
tokenizer/
```

**ASR** (`/home/cat/qwen3-asr-models/`, 挂载到 `/opt/asr/models/`):
```
encoder/rk3576/
  qwen3_asr_encoder_merged.fp16.4s.rk3576.rknn  (370MB)
rkllm/
  decoder_qwen3.w4a16_g128.rk3576.rkllm         (758MB)
embd/
  decoder_hf.embed_tokens.npy                    (594MB, mmap 加载)
tokenizer/
  tokenizer.json                                 (11MB)
mel_filters.npy                                  (201KB)
vad/
  silero_vad.onnx                                (2.2MB)
```

---

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | `{tts, tts_backend, asr, asr_backend, streaming_asr}` |
| `/tts` | POST | `{"text": "...", "sid": 0, "speed": 1.0}` → WAV |
| `/tts/stream` | POST | 流式 TTS (PCM chunks) |
| `/asr` | POST | multipart `file=@audio.wav` + `language=auto` → JSON |
| `/asr/stream` | WS | int16 PCM → JSON `{"text", "is_final"}` |

---

## 已探索但失败的方案

| 方案 | 结果 | 原因 |
|------|------|------|
| RKNN 整图 code_predictor W4A16 | cosine=0.01 | 编译器 bug |
| RKNN 整图 code_predictor FP16 | cosine=0.01 | disable_rules 破坏正确性 |
| RKNN per-layer code_predictor | 205ms (75 次调用) | 调用开销太大 |
| MTE matmul 替代 RKLLM talker | 250ms/步 | KV-cache DDR 带宽瓶颈, RKLLM 6x 快 |
| 解耦 CP 反馈 (零 residual) | 2.8% match | 完全失败 |
| ASR encoder INT8 | 质量崩 | 必须 FP16 |
| SRAM 环境变量 | 无效果 | RK3576 SRAM 被视频解码器占用 |
| vocoder optimization_level=3 | 无提升 | 硬件天花板 ~620ms |
| vocoder mmse 量化 | 无提升 | 和 normal 一样 |

---

## 已知限制

1. **RKLLM 独占 NPU**: TTS 和 ASR 不能并行, 用 threading.Lock 串行化
2. **内存**: TTS+ASR=5.8GB, 不能和其他大模型共存
3. **RKLLM 闭源**: talker 43ms/步是公开 API 天花板, 无法用 MTE 替代
4. **vocoder 慢**: 620ms/2s-chunk, 硬件下限
5. **TTS 不实时**: RTF=4.86, 10cb 约 1.5, 只有 5cb 以下才实时但听感差

---

## 优化方向（接手人参考）

### 短期可做（1-3 天）

| 方向 | 预期效果 | 说明 |
|------|---------|------|
| **锁频** | RTF 降 ~27% | NPU+DDR+CPU 锁频需 root, 当前未完全生效 |
| **减码本到 10** | RTF ~1.5 | 修改 tts_service.py 的 NUM_CODE_GROUPS, 听感可接受 |
| **ASR 加载 2s encoder** | TTFT 降到 2.3s | `encoder_sizes=[2,4]`, 多占 370MB 内存 |
| **声音克隆集成** | 支持 x-vector | speaker_encoder ONNX 已有, 需接入 prefill 注入 |

### 中期可做（1-2 周）

| 方向 | 预期效果 | 说明 |
|------|---------|------|
| **1 帧延迟 residual** | TTS 帧时间减半 | talker 和 CP 重叠, 需验证质量 |
| **pipeline 并行** | vocoder 和 AR 重叠 | vocoder 处理当前 chunk 时 AR 生成下一 chunk |
| **text_project 缩小** | 63ms→~10ms | shape [1,128]→[1,16] 重新导出 |

### 长期方向

| 方向 | 说明 |
|------|------|
| **Rocket 开源 NPU 驱动** | Linux 主线 + Mesa, 可替代闭源 RKLLM, 但 transformer 支持待实现 |
| **通用 MTE engine** | 从 ONNX 提取权重, matmul API 逐层执行, 已有 Zipformer 实现 |
| **RK3588 迁移** | 同架构, 多一个 NPU 核, SRAM 可用, 但核心瓶颈不变 |

---

## 关键技术决策记录

### 为什么 RKLLM 不可替代 (AR decoder)
RKLLM 内部做了 SRAM 零拷贝 + Flash Attention + 算子融合, 公开 matmul API 每步 250ms vs RKLLM 43ms, 差 6x。原因是 matmul API 每步都要从 DDR 读写 KV-cache, 带宽成为瓶颈。

### 为什么 ASR encoder 用 RKNN 而不是 MTE
Encoder 不是 AR (一次处理全部 token), RKNN 整图可以做算子融合 + exSDP attention, 比 MTE 逐层调用更快。且 FP16 精度无问题。

### 为什么 ASR 用 Qwen3-ASR 而不是 Zipformer/Whisper
- Zipformer: 已有 MTE engine (RTF=0.2), 但只支持中英, 无多语言
- Whisper: RKNN 无成熟方案, INT8 精度崩, CPU 太慢
- SenseVoice: 非流式, 无 NPU 加速
- **Qwen3-ASR**: 52 语言 + 流式 + RTF=0.44, 社区已验证

### 为什么 C engine W8A16 而不是 W4A16
W4A16 per-column 对称量化太粗暴, code_predictor 5 层误差累积严重 (精度 4%)。W8A16 per-column 精度 97% match。

---

## 文件结构

```
rk3576/
├── app/                          # FastAPI 服务
│   ├── main.py                   # 路由 (/tts, /asr, /health, /asr/stream)
│   ├── tts_backend.py            # TTS backend 抽象 + 工厂
│   ├── asr_backend.py            # ASR backend 抽象 + 工厂
│   ├── tts_service.py            # TTS pipeline (RKLLM+RKNN+C engine)
│   ├── rkllm_wrapper.py          # RKLLM ctypes (TTS talker)
│   ├── backends/
│   │   ├── qwen3_rknn.py         # TTS backend 实现
│   │   └── qwen3_asr_rk.py       # ASR backend 实现 (含 NPU lock)
│   └── qwen3asr/                 # Qwen3-ASR 核心库 (from qzxyz)
│       ├── engine.py             # ASR 主引擎
│       ├── stream.py             # 流式会话 (StreamSession)
│       ├── encoder.py            # RKNN encoder wrapper
│       ├── decoder.py            # RKLLM decoder wrapper
│       ├── mel.py                # Mel 频谱提取
│       ├── vad.py                # Silero VAD
│       ├── config.py             # 常量
│       └── utils.py              # 工具函数
├── engine/                       # C 推理引擎 (rknn_matmul_api)
│   ├── cp_engine.c               # code_predictor engine (W8A16)
│   ├── cp_engine.h
│   ├── cp_engine_wrapper.py      # Python ctypes wrapper
│   └── Makefile
├── mte/                          # MTE 通用 transformer engine
│   ├── engine/                   # Zipformer encoder C engine
│   └── scripts/                  # ONNX 分析 + 权重提取
├── scripts/                      # 转换/量化/benchmark 脚本
│   ├── quantize_w8a16.py         # CP 权重 FP16→INT8 量化
│   ├── convert_vocoder_rknn.py   # vocoder RKNN 转换
│   └── ...
├── Dockerfile
├── docker-compose.yml
├── STATUS.md                     # 本文件
├── LESSONS.md                    # 踩坑记录 (20+ 条)
└── README.md                     # 技术文档
```

## 设备信息

| 设备 | Fleet name | 用途 |
|------|-----------|------|
| RK3576 LubanCat-3 | cat-remote | 部署目标 |
| WSL2 RTX 3060 | wsl2-local | RKNN/RKLLM 模型转换 |
| Jetson Orin NX 16G | seeed-desktop | 对比基线 (TRT 方案) |

## 参考项目

- [qzxyz/qwen3asr_rk](https://huggingface.co/qzxyz/qwen3asr_rk) — ASR 方案来源
- [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — 官方 ASR 仓库
- [antirez/qwen-asr](https://github.com/antirez/qwen-asr) — 纯 C ASR 实现
- [andrewleech/qwen3-asr-onnx](https://github.com/andrewleech/qwen3-asr-onnx) — ONNX 导出
- [Rocket NPU driver](https://blog.tomeuvizoso.net/2025/07/rockchip-npu-update-6-we-are-in-mainline.html) — 开源 NPU 驱动 (Linux 主线)
