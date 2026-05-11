# Qwen3 ASR + TTS 并发瓶颈测试 — orin-nx (Jetson Orin NX, JetPack 6.2)

**日期**: 2026-04-26
**设备**: orin-nx (Jetson Orin NX, JetPack 6.2 / TRT 10.3 / CUDA 12.5, 15GB RAM)
**Container**: `jetson-voice-speech:v3.1-with-vad` (image `3b73c1de81c9`)
**Endpoint**: `http://192.168.3.225:8621`
**N**: 5 runs/item × 5 items = 25 datapoints per phase

## 工况

- **A — solo ASR**: 顺序跑 5 wavs (S0~S4, 3-12s) × 5 reps = 25 calls
- **B — solo TTS**: 顺序跑 5 文本 (T0~T4, 短到长) × 5 reps = 25 calls
- **C — 并发**: 两个 worker thread 同时启动，A worker 跑 25 ASR + B worker 跑 25 TTS

## Wall-time (client-measured) 对比

### ASR (单位 ms)

| 句长 | A solo | C concurrent | C/A 倍率 |
|---|---:|---:|---:|
| S0 (~3s)  |  222 | 1768 | **8.0×** |
| S1 (~5s)  |  224 | 2016 | **9.0×** |
| S2 (~8s)  |  356 | 2852 | **8.0×** |
| S3 (~10s) |  445 | 3629 | **8.2×** |
| S4 (~12s) |  503 | 4609 | **9.2×** |
| **平均** | **350** | **2975** | **8.5×** |

### TTS (单位 ms)

| 文本 | B solo | C concurrent | C/B 倍率 |
|---|---:|---:|---:|
| T0 (~12 字)  | 1396 | 1705 | 1.22× |
| T1 (~14 字)  | 1811 | 2034 | 1.12× |
| T2 (~24 字)  | 2261 | 2805 | 1.24× |
| T3 (~30 字)  | 3045 | 3615 | 1.19× |
| T4 (~40 字)  | 4048 | 4608 | 1.14× |
| **平均** | **2512** | **2953** | **1.18×** |

## 服务端推理时间（不含 HTTP / queue）

| 指标 | A solo | C concurrent | 退化 |
|---|---:|---:|---:|
| ASR per_token_ms (mean) | 25.2 | 30.7 | **1.22×** |
| ASR per_token_ms (p95)  | 32.1 | 36.6 | 1.14× |
| ASR inference_ms (mean) | 333 | 389 | **1.17×** |
| ASR inference_ms (p95)  | 482 | 543 | 1.13× |

## Phase C 总耗时

74.4s 完成 25 ASR + 25 TTS。
对比 Phase A + Phase B 顺序跑 ≈ 25 × 0.35 + 25 × 2.51 = 71.5s。

**结论：Phase C ≈ Phase A + B（顺序）**，并发无加速。

## 瓶颈诊断

**核心结论：speech container 实际上是单线程串行处理请求**，并发不带来吞吐提升，纯排队。

证据链：
1. 服务端 `inference_time` 在 C 工况只比 A 慢 17%（333→389ms），`per_token_ms` 慢 22%（25.2→30.7ms）→ **GPU 推理本身基本不受并发影响**（说明 TRT engine 在 stream 上 serialize 得很彻底，没有真正打架）。
2. Wall-time ASR 工况 C 比 A 慢 **8.5×**，但 inference 只慢 1.17× → 多出来的 ~7× 全是**队列等待**。
3. TTS wall 只慢 1.18× → 因为 TTS 自身 wall 大（2.5s），即使队列里塞了几个 ASR (~0.4s 各)，相对时间影响小。
4. ASR wall 暴涨 → 因为 ASR 自身 wall 小（0.35s），但每次要等 1-3 个 TTS（每个 ~2.5s）才能轮到，等待时间远大于自身处理时间。

形象地说：

```
Solo ASR pipeline:  [ASR-0.35s]  [ASR-0.35s]  [ASR-0.35s] ...
Solo TTS pipeline:  [---TTS-2.5s---] [---TTS-2.5s---] ...
Concurrent C:       [ASR][TTS....][ASR][TTS....][ASR][TTS....]   ← 单 GPU stream 串行
                    ↑ ASR wall = 自己 0.35s + 等了 2.5s 的 TTS
```

## 推测 root cause（待 instrument 验证）

最可能的串行点（按概率排序）：

1. **TRT engine context 互锁** — ASR decoder + TTS CP / vocoder 共用同一个 CUDA stream/default context，Python GIL 或 C++ 粗锁让 enqueueV3 强制串行（参考 memory `project_cuda_graph_capture_traps.md`：CUDA Graph capture 用 `cudaStreamCaptureModeGlobal` 会全 context 污染，加 mutex 防并发）
2. **Python uvicorn worker 单进程** — `--port 8621` 只起一个 worker，async 函数里如果 ASR/TTS 调 C++ 时不释放 GIL，就退化成串行
3. **CP engine pool N=2 已满** — TRT 10.3 myelin bug 强制 N=2（memory `project_trt103_myelin_cp_pool.md`），TTS 并发突发会抢 slot，但本测试是单 ASR 单 TTS 流，应该不撞 pool

## 下一步建议

1. 主线程 / 容器内打 `py-spy dump` 看并发时 thread state，确认是 GIL / mutex 哪个
2. 把 ASR / TTS 拆到 **两个 uvicorn worker** 进程跑（用 `--workers 2` 或起两个容器各跑一个 backend），看吞吐能不能并行
3. 如果是 TRT engine context 互锁 → 给 ASR、TTS 各分配独立 cudaStream + 独立 ExecutionContext，不共享
4. 如果服务端 GIL 是关键 → C++ 推理那一段加 `pybind11::gil_scoped_release`

## 数据文件

- 原始 JSON: `orin-nx-concurrent-2026-04-26.json`
- GPU 采样: `gpu_{A,B,C}.csv`（Jetson iGPU `nvidia-smi` 返回 `[N/A]`，只有 tegrastats 能采，本次未跑）
