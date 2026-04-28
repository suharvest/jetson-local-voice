# Handover: Nano 8GB Qwen3 multilanguage 内存抢救 (2026-04-28)

## 终点状态

**结论**：实施全套优化后 Nano peak **7477 MB / 7620 MB**，仍 OOM at TTS CP slot 创建。**差 700-1000 MB 才能稳定生产**。

最新提交 commits（主线 main，HEAD 64bf23d 之后还有 1aefe26）：
- `1aefe26` docs(bench): P1 final measurement on Orin Nano 8GB
- `a715669` docs: P0b/P1 memory optimization specs + Orin Nano feasibility
- `02e6e3f` feat(scripts,tests): ASR decoder engine builder + P1 parity test
- `d1378ef` feat(tts,main): TTS load instrumentation + SKIP_ASR_WARMUP env gate
- `5d34cab` perf(asr): P0a + max_seq=200 + P1 trt_native + warmup gate + instrumentation
- `2f547b6` perf(cpp): TRT IStreamReader + TRTASREncoder for P1
- `64bf23d` build(scripts): standardize TRT engine build scripts with per-device profile

## 已完成的优化（按贡献排序）

| # | 优化 | 节省 | 在哪 |
|---|---|---:|---|
| 1 | **P1 ASR encoder ORT→TRT 原生** | -880 MB | `app/backends/qwen3_asr.py` `ASR_ENCODER_BACKEND=trt_native`; C++ class `TRTASREncoder` in `benchmark/cpp/tts_trt_engine.h/.cpp`; binding in `tts_binding.cpp` |
| 2 | **ASR decoder max_seq 500→200 重 build** | -560 MB | `scripts/build_asr_decoder_engine.sh`，新 engine `asr_decoder_bf16_max200.engine` |
| 3 | **SKIP_ASR_WARMUP env gate**（executor + backend warmup 都跳） | -370 MB | `app/main.py` + `app/backends/qwen3_asr.py` 顶部 |
| 4 | **P0a embed_tokens 保留 FP16**（不 .astype(np.float32)）| -300 MB | `app/backends/qwen3_asr.py:835` |
| 5 | **P0b TRT IStreamReader 流式加载** | startup peak -1GB | `benchmark/cpp/tts_trt_engine.cpp` 6 处 deserialize 全转，新 helper `LoadEngineStreaming` |
| 6 | **CP unified rebuild (v1 + ws=256)** | plan -209 MB | `scripts/build_tts_cp_engine.sh`，新 engine `cp_unified_bf16.engine` 215MB |
| 7 | **LAZY_TTS env gate** | TTS 推迟到首请求 | `app/main.py` + `app/tts_service.py` |
| 8 | **CP_POOL_SIZE=1**（部署 env） | -300 MB | env 变量 |
| 9 | **Per-step memory instrumentation** | 永久 debug 利器 | `[MEM:tag]` log 在 ASR/TTS 关键点 |

NX 上累计省了 1.5+ GB，**对所有设备净 win**。但 Nano 还差 700-1000 MB。

## 实测峰值演变

| 配置 | Peak (MB) | 死在哪 |
|---|---:|---|
| 默认 | 7503 | TTS Talker 都没载完 |
| P0a + P0b | 7503 | 同 |
| + max_seq=200 | 7516 | 同 |
| + SKIP_ASR_WARMUP | 7503 | Talker 都没载完 |
| **+ P1 + LAZY_TTS** | **7503** | **Talker 加载完 ✅，CP slot 创建死** |
| **+ 新 CP engine 215MB** | **7477** | 同上（plan 省 209MB 但 peak 只省 26MB）|

## 已 Verify 的不可走路径（社区研究 + 实测）

- ❌ **INT8 整体量化 Talker**：Jetson 小 batch autoregressive 实测 21% slower（reformat overhead）
- ❌ **TRT 10 weight streaming**：iGPU host==device 同物理 RAM，"streaming" 仅切 mapping 不省物理页
- ❌ **Hot-swap 进程间换 ASR/TTS**：TRT engine deserialize 25-60s 太慢，对话不可用
- ❌ **多容器拆分**：iGPU 共享 carveout 仍是 8GB
- ❌ **关 GUI 单独**：实测 idle 只占 50-100MB
- ❌ **ZRAM swappiness 调整**：GPU cudaMalloc 是 NVMAP pinned 不可 swap，只能救 host 端 100-200MB（不进 GPU 池）
- ❌ **TTS Talker workspace cap 砍**：workspace 只是 build-time tactic 选择上限，实际运行时只用很少；engine plan = 模型权重，不可压缩
- ❌ **TTS 单 profile + ORT prefill**：双栈反而吃更多内存
- ❌ **Layer dropping 28→24**：质量退化大，weeks 蒸馏

## 下一步：Vocab Pruning（spec 在 codex review 中）

**社区研究最大未挖掘的一招**：

参考 [AtomGradient swift-qwen3-tts](https://atomgradient.github.io/swift-qwen3-tts/)：
- Qwen3-TTS-1.7B 从 2.35GB → 808MB（embedding 622MB → 194MB），lossless
- 砍 vocab 152K → 47K active tokens（覆盖实际 workload corpus）
- 数学等价，无需 retrain
- 我们 ASR + TTS 都共享 152K Qwen3 vocab，理论 -350-450MB

Spec 在 codex review 中，会写到 `docs/plans/vocab-pruning-2026-04-28.md`。

## 备选 fallback（如果 vocab pruning 不够）

1. **TensorRT-LLM AWQ INT4 Talker**：activations 留 BF16（QK^T 不溢出），weights INT4 → -400-500MB Talker。需 custom plugin，2-3 周
2. **GGUF Q5_K_M Talker** ([khimaros/qwen3-tts.cpp](https://github.com/khimaros/qwen3-tts.cpp))：~380MB Talker 一定够。换 runtime，1 周 port
3. **Speech tokenizer encoder strip**：如不需 voice cloning 可删 codec_encoder 分支，-225MB
4. **Carveout reflash + ZRAM + 服务 triage**：组合 -150-300MB host 端

## 部署 + 测试 footguns（下次别再踩）

### 1. trtexec build 时 prod 自动 restart 抢 RAM
- 现象：build 跑到一半被 SIGKILL
- 原因：compose `restart: unless-stopped` 让 prod 自动重启
- 修：build 前 `docker compose stop speech` 而非 `docker stop`

### 2. ONNX schema 跨版本不一致
- 部署 `cp_unified_bf16.engine` 有 `gen_step` + `past_length` 双 scalar 输入（mid-state export）
- v1 `export_cp_unified.py` 只有 `past_length`（无 gen_step）
- v3 `export_cp_unified_v3.py` 同 v1（额外 fixed past）
- C++ TRTCPKVEngine 自动检测两种 variant: `is_single_head_`(gen_step), `has_past_length_input_`
- 我们用 v1 ONNX → all-at-once `logits_all` path → 数学等价部署版 single-head（单 forward 出全 15 CB logits vs 15 forward 各出 1）

### 3. Nano 上 ONNX 文件腐败
- `code_predictor.onnx` 285MB on Nano - protobuf 级别错误
- Mac 本地版本 420MB 完整可解析
- 需要从 source 重 sync

### 4. fleet transfer 一开始很慢（2-hop）
- `fleet transfer` 默认 source→Mac→dest，420MB 走 30+ min
- 用 `--direct` flag（fleet 新增）device→device 直连
- 需要 source 上有 sshpass 或 SSH key authority
- Mac 没 Nano 直接 SSH key，要 NX 跳板

### 5. WSL ssh 设置
- WSL Ubuntu 24.04 ed25519 key 加到 Nano `~/.ssh/authorized_keys`
- WSL 通过 home-win Tailscale 100.73.210.80 reach 100.92.125.65 直接走 Tailscale，速度好
- 直接 scp WSL→Nano 421MB ~3min

### 6. trtexec workspace 太小让 build 失败
- 试过 `--memPoolSize=workspace:256MiB` 让 vocoder build 跑 40min 还卡
- 256MB 让 TRT 跳过太多 tactic，可能选不到任何
- 实用最小 workspace: 512MB（vocoder/CP 都没问题）

### 7. recursive-dispatch 反模式（已记 memory）
- claude-rescue / opencode-rescue / codex-rescue Sonnet wrapper 看到大 prompt 会想 dispatch 子任务
- 必须在 prompt 加 ANTI-RECURSION 头
- agent 还可能 dispatch 内部 Task，看似活其实空跑

## 关键工件位置

### 本地 Mac
- `/Users/harvest/project/jetson-voice/scripts/build_*.sh` - 4 个 engine builders + surgery script + per-device profile
- `/Users/harvest/project/jetson-voice/tests/test_p1_encoder_parity.py` - ORT vs TRT 数值对比
- `/Users/harvest/project/jetson-voice/docs/plans/p0b-p1-memory-opt-2026-04-28.md` - codex 评估
- `/Users/harvest/project/jetson-voice/docs/plans/p1-asr-encoder-trt-impl-2026-04-28.md` - P1 spec
- `/Users/harvest/project/jetson-voice/docs/benchmarks/orin-nano-v34-slim-2026-04-28.md`
- `/Users/harvest/project/jetson-voice/docs/benchmarks/p1-nano-final-2026-04-28.md`

### Nano (orin-nano via Tailscale 100.92.125.65)
- `/home/harvest/voice_test/models/qwen3-tts/engines/cp_unified_bf16.engine` (215MB, 新 v1 build)
- `/home/harvest/voice_test/models/qwen3-tts/engines/*.bak.before_ws256*` (旧 engine 备份)
- `/home/harvest/voice_test/models/qwen3-tts/onnx/cp_unified.onnx` (421MB, v1 from WSL)
- `/home/harvest/voice_test/models/qwen3-asr-v2/asr_encoder_fp16.engine` (372MB, P1)
- `/home/harvest/voice_test/models/qwen3-asr-v2/asr_decoder_bf16.engine` (max_seq=200 版)
- `/home/harvest/voice_test/app_overlay/` - 全部 Python + .so overlay

### NX (orin-nx via Tailscale 100.82.225.102)
- `/home/harvest/jetson-voice/` - 主仓 + cpp source + scripts
- `/home/harvest/qwen3-asr-v2/` - 原始 ONNX + 新 build engines
- prod 容器 `reachy_speech-speech-1` 跑 v3.4-slim 镜像

### WSL (home-win 100.73.210.80)
- `/home/harve/qwen3-tts-export/out_v1/cp_unified.onnx` 421MB
- `~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base` 全模型
- torch 2.4 + transformers 4.57.3 + onnx 1.16 + qwen-tts 0.1.1

## 复现命令（下次接手快速上手）

```bash
# 测 Nano 全栈（当前最佳配置 + LAZY_TTS）
ssh harvest@100.82.225.102 "ssh harvest@100.92.125.65 'rm -f /tmp/tegrastats.log; nohup tegrastats --interval 500 --logfile /tmp/tegrastats.log > /dev/null 2>&1 &'"

fleet exec orin-nano -- 'docker rm -f voice_nano_test 2>/dev/null; docker run -d --name voice_nano_test --runtime nvidia --network host \
  -v /home/harvest/voice_test/models:/opt/models:rw \
  -v /home/harvest/voice_test/app_overlay/main.py:/opt/speech/app/main.py:ro \
  -v /home/harvest/voice_test/app_overlay/tts_service.py:/opt/speech/app/tts_service.py:ro \
  -v /home/harvest/voice_test/app_overlay/tts_backend.py:/opt/speech/app/tts_backend.py:ro \
  -v /home/harvest/voice_test/app_overlay/asr_backend.py:/opt/speech/app/asr_backend.py:ro \
  -v /home/harvest/voice_test/app_overlay/backends:/opt/speech/app/backends:ro \
  -v /home/harvest/voice_test/app_overlay/qwen3_speech_engine.cpython-310-aarch64-linux-gnu.so:/opt/speech/app/qwen3_speech_engine.cpython-310-aarch64-linux-gnu.so:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /lib/aarch64-linux-gnu:/host-libs:ro \
  -e LANGUAGE_MODE=multilanguage -e CP_POOL_SIZE=1 -e SKIP_ASR_WARMUP=1 -e ASR_ENCODER_BACKEND=trt_native -e LAZY_TTS=1 \
  jetson-voice-speech:v3.4-slim'

# 等 ~50s ASR ready，发 TTS 触发 lazy load
sleep 60
fleet exec orin-nano -- 'curl -sf --max-time 90 -X POST http://localhost:8621/tts -H "Content-Type: application/json" -d "{\"text\":\"你好\"}" -o /tmp/test.wav -w "http=%{http_code}\n"'

# 收数据
fleet exec orin-nano -- 'docker logs voice_nano_test 2>&1 | grep "\[MEM:"'
ssh harvest@100.82.225.102 "ssh harvest@100.92.125.65 'awk \"...\" /tmp/tegrastats.log'"
```

## Compact 后接手指引

1. 看 `docs/plans/vocab-pruning-2026-04-28.md`（codex 设计）
2. 派 deepseek-flash 实施：先在 WSL 跑 vocab discovery + ONNX surgery (Phase A)
3. WSL 验证 lossless 后 → push 到 Nano build TRT engine
4. Nano 测 RAM peak → 看是否 fit（差 700-1000 MB → 期望 vocab pruning 给 350-450MB）
5. 若不够补 speech encoder strip + carveout reflash + 服务 triage
6. 仍不够走 GGUF Q5_K_M (qwen3-tts.cpp) 兜底

## NX 16GB / AGX 32GB 建议

无论 Nano 最终能不能跑，下面优化对 NX/AGX 都是 net win，建议保留并部署：
- P0a + P0b + P1 + max_seq=200 + LAZY_TTS gate + SKIP_ASR_WARMUP gate
- 这些都是 opt-in env-driven，不影响默认行为

需要 doc 化的 deployment guide：
- Nano 8GB → `LANGUAGE_MODE=zh_en`（sherpa paraformer + matcha，~1GB）
- NX 16GB → `LANGUAGE_MODE=multilanguage`（qwen3 全栈 + 所有今日优化 enabled）
- AGX 32GB → 同 NX，可不开 LAZY_TTS（启动快）
