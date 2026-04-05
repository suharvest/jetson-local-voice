# RK3576 Qwen3-TTS 项目状态

> 更新: 2026-04-05

## 已验证的最优配置

```
Talker:          RKLLM W4A16 Qwen3  → 43ms/步 (不可替代)
Code Predictor:  C engine W8A16      → 126ms/15步 (matmul API + NEON)
                 或 RKLLM v1.2.3     → 120ms/15步 (但精度只有20%)
Vocoder:         RKNN ctx25 INT8     → 682ms/chunk (2s音频, RTF=0.34)
Speaker Encoder: ONNX CPU            → 24ms (声音克隆)
```

## 性能汇总

| 码本数 | CP 方案 | 每帧时间 | RTF | 实时？ |
|--------|---------|---------|-----|--------|
| 16 | C engine W8A16 | 43+126=169ms | 2.1 | ❌ |
| 16 | RKLLM (精度差) | 43+100=143ms | 1.8 | ❌ |
| 8 | C engine W8A16 | 43+59=102ms | 1.27 | ❌ (接近) |
| 8 | RKLLM (精度差) | 43+47=90ms | 1.12 | ❌ (接近) |
| 6 | C engine W8A16 | 43+42=85ms | 1.06 | ❌ (几乎) |
| 5 | C engine W8A16 | 43+34=77ms | 0.96 | ✅ |
| 4 | C engine W8A16 | 43+25=68ms | 0.85 | ✅ |

## 音频质量 (ASR 验证)

| 码本数 | 3 段文字 ASR 通过率 | 听感 |
|--------|-------------------|------|
| 16 | 3/3 ✅ | 最好 |
| 12 | 3/3 ✅ | 好 |
| 10 | 3/3 ✅ | 好 |
| 8 | 2/3 ⚠️ (1字识别错) | 可接受 |
| 6 | 3/3 ✅ | 需要听 |
| 4 | 不稳定 | 加赘字 |

音频样本: `rk3576/audio_samples/`

## 已探索但失败的方案

| 方案 | 结果 | 原因 |
|------|------|------|
| RKLLM 逐步调用 (v1.2.2) | 2250ms | 143ms/次调用开销 |
| RKLLM 逐步调用 (v1.2.3) | 120ms ✅ | cache 复用修复 + A55 核绑定 |
| RKLLM CP Qwen2 导出 | 精度 7% | q_norm/k_norm 丢失 |
| RKLLM CP Qwen3 导出 | 精度 20% | W4A16 量化误差累积 |
| RKNN 展开单图 | 403ms | 权重被 RKNN 编译器展开 |
| RKNN per-layer ONNX | 205ms | 75 次调用开销 |
| RKNN 全模型 ONNX | 150ms | 编译器 bug → 精度崩溃 |
| RKNN W4A16 talker | 250ms | 和 FP16 一样慢 (KV-cache 瓶颈) |
| RKNN 双核并行 RKLLM+RKNN | 无效 | RKLLM 独占 NPU 调度器 |
| 解耦 CP 反馈 (零 residual) | 完全失败 | 2.8% token match, 无 EOS |
| C engine W4A16 (naive quant) | 88ms 但精度 4% | per-column 对称量化太粗暴 |
| NPU Flash Attention (per-layer) | exSDP 融合成功 | 但调用开销抵消了计算优化 |
| NPU 自定义 kernel | 不可能 | 固定功能硬件, 无 ISA |
| NPU SRAM 控制 | 不可能 | 只有开关, 无精细控制 |
| NPU 任务链 | 不可能 | 闭源 librknnrt.so 独占 |

## 未验证的潜在优化

| 方案 | 预估效果 | 工作量 | 风险 |
|------|---------|--------|------|
| 1帧延迟 residual 近似 | 16cb 43ms RTF 0.54 | 1天验证 | 质量未知 |
| C engine W4A16 (RKNN toolkit量化) | 88ms, 16cb+talker=131ms | 1周 | 量化格式逆向 |
| 通用 transformer matmul engine | 复用到其他模型 | 2周 | 工程量 |
| Talker matmul engine (省内存) | 30ms(短句), 省620MB | 2-3周 | KV-cache 管理复杂 |

## 硬件限制 (RK3576 6 TOPS NPU)

- NPU 是固定功能单元 (CNA+DPU+PPU), 非可编程
- SRAM ~512KB, 不可编程分配
- W8A16 matmul: 0.109ms per [1024×3072] — 硬件下限
- W4A16 matmul: 0.106ms per [1024×2048] — 精度问题
- FP16 matmul: 0.357ms per [1024×3072] — 最慢但最准
- RKLLM 快 6x 因为: SRAM 复用 + 算子融合 + Flash Attention + 零拷贝 KV-cache
- 公开 API 无法复制这些优化

## 设备信息

| 设备 | Fleet name | 用途 |
|------|-----------|------|
| RK3576 LubanCat-3 | cat-remote | 部署目标 |
| WSL2 RTX 3060 | wsl2-local | 模型转换 |
| Jetson (参考) | seeed-desktop | 对比基线 |

## 文件结构

```
rk3576/
├── app/                  # FastAPI 服务 (对齐 jetson-voice API)
│   ├── main.py
│   ├── tts_service.py
│   └── rkllm_wrapper.py
├── engine/               # C 推理引擎 (rknn_matmul_api)
│   ├── cp_engine.c       # code_predictor engine (v2, 含融合优化)
│   ├── cp_engine.h
│   ├── cp_engine_wrapper.py
│   ├── Makefile
│   └── test_engine.py
├── scripts/              # 转换/benchmark 脚本
│   ├── export_*.py       # ONNX 导出
│   ├── convert_*.py      # RKNN 转换
│   ├── bench_*.py        # Benchmark
│   ├── quantize_*.py     # 量化
│   ├── replace_sin_polynomial.py  # vocoder Sin→Taylor
│   └── sweep_vocab.py    # RKLLM vocab 扫描
├── audio_samples/        # 码本质量对比 (6/8/10/12/16cb × 3 texts)
├── Dockerfile
├── docker-compose.yml
├── LESSONS.md            # 踩坑记录
├── PLAN-code-predictor-unroll.md
├── README.md             # 技术文档
└── STATUS.md             # 本文件
```
