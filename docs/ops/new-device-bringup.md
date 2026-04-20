# 新 Jetson 设备 bring-up 指南

把当前（2026-04-20）`recomputer-desktop` 上的 jetson-voice 完整状态还原到新 Jetson 设备（AGX Orin 同型号）。

**目标**：新设备 30 分钟内跑通 `/health` + 3 句 V2V 测试。

## 0. 前提

- 新设备是 Jetson AGX Orin（或至少 Orin 家族，SM=8.7）
- JetPack 6.x，CUDA 12.x，TensorRT 10.3
- Docker 可用，NVIDIA Container Runtime 已装
- 网络通：能到 github.com 和 sensecraft-missionpack.seeed.cn（镜像 registry）
- 至少 **120 GB** 可用磁盘（模型 + docker image + swap）

## 1. 装 fleet 条目 + 测试连通

在 Mac 上：
```bash
# 编辑 ~/project/_hub/devices.json 添加新设备：
{
  "jetson-new": {
    "host": "<tailscale-ip>",
    "user": "recomputer",
    "password": "***",
    "tags": ["jetson","arm64","gpu","agx-orin"],
    "owner": "personal"
  }
}

fleet status jetson-new --json
```

## 2. Clone repo + branch

```bash
fleet exec jetson-new -- "cd /home/recomputer && git clone https://github.com/suharvest/jetson-local-voice.git jetson-voice"
fleet exec jetson-new -- "cd /home/recomputer/jetson-voice && git checkout feature/t1-cp-graph-cache"
# 或 main, 看需要哪版
```

## 3. 准备模型（最重 — ~16 GB）

模型不在 git 里。两种拿法：

### 方法 A — 从 recomputer-desktop 直接 rsync（最快，如果两台在同网）
```bash
# 宿主到宿主（绕过 docker volume）：
fleet transfer recomputer-desktop:/var/lib/docker/volumes/reachy_speech_speech-models/_data/ jetson-new:/home/recomputer/qwen3-models/
# 然后在 jetson-new 上建 docker volume 并塞进去
fleet exec jetson-new -- "docker volume create reachy_speech_speech-models && sudo cp -r /home/recomputer/qwen3-models/* /var/lib/docker/volumes/reachy_speech_speech-models/_data/"
```

需要模型：
- `/opt/models/qwen3-tts/engines/*.engine`（TRT 二进制 ≈ 3.4 GB）——**Orin 二进制可跨同型号 Orin 用**，不同 GPU 需重建
- `/opt/models/qwen3-tts/onnx/*`（ORT 模型）
- `/opt/models/qwen3-asr-v2/*.engine` + `*.onnx`（≈ 8 GB）
- `/opt/models/matcha-icefall-zh-en/*`（≈ 1 GB，sherpa 回退用）
- `/opt/models/paraformer-streaming/*`（≈ 400 MB）

### 方法 B — 从 sensecraft-missionpack registry 拉镜像 + 装模型
不确定模型是否在镜像里，以 A 为主。

### 重建 TRT engine（若 GPU 型号不同）

在 jetson-new 上：
```bash
cd /home/recomputer/jetson-voice/benchmark
# 需要 ONNX 源模型先齐：qwen3-tts/onnx/*, qwen3-asr-v2/decoder_unified.onnx
bash build_asr_bf16_engine.sh        # 构建 ASR decoder
# CP + Talker + Vocoder 的 build script 在 benchmark/cpp/ 下
# TODO: 整合一个 one-click build script
```

预期：engine 重建 30-60 分钟（Orin）。

## 4. Build 原生 .so（必须在目标设备）

pybind11 + CUDA + TRT 的 `.so` **必须在目标 Jetson 上 build**，不能跨机：
```bash
fleet exec jetson-new -- "bash -c 'cd /home/recomputer/jetson-voice/benchmark/cpp && bash build.sh'"
# 产物：benchmark/cpp/build_cmake/qwen3_speech_engine.cpython-310-aarch64-linux-gnu.so
```

**前置条件**：
- `/home/recomputer/ort-from-container/` 已解包出来（从镜像里拷出来的 onnxruntime headers + libs）
- `/usr/local/cuda/bin/nvcc` 可用
- pybind11 Python 包已装在 `/home/recomputer/.local/`

如果 `ort-from-container/` 没有，从 recomputer-desktop 拉：
```bash
fleet transfer recomputer-desktop:/home/recomputer/ort-from-container/ jetson-new:/home/recomputer/ort-from-container/
```

## 5. 准备 app_overlay（生产部署层，非 git）

**关键**：`app_overlay/` 不在 git 里但是 production mount 源（见 `memory/project_deploy_footguns.md`）。

两种做法：

### A — 从镜像 extract 一份，然后用 repo 的 app/ 覆盖
```bash
fleet exec jetson-new -- "bash -c '
  docker create --name tmp jetson-voice-speech:v3.1-with-vad
  docker cp tmp:/opt/speech/app/ /tmp/img_app
  docker rm tmp
  mkdir -p /home/recomputer/jetson-voice/app_overlay
  cp /tmp/img_app/*.py /home/recomputer/jetson-voice/app_overlay/
'"

# 用 repo 的 Qwen3 版本覆盖（关键）
fleet push jetson-new app/main.py /home/recomputer/jetson-voice/app_overlay/main.py
fleet push jetson-new app/tts_backend.py /home/recomputer/jetson-voice/app_overlay/tts_backend.py
fleet push jetson-new app/tts_service.py /home/recomputer/jetson-voice/app_overlay/tts_service.py
fleet push jetson-new app/asr_backend.py /home/recomputer/jetson-voice/app_overlay/asr_backend.py
fleet push jetson-new app/backends /home/recomputer/jetson-voice/app_overlay/backends

# 把刚 build 的 .so 放进去
fleet exec jetson-new -- "cp /home/recomputer/jetson-voice/benchmark/cpp/build_cmake/qwen3_speech_engine.cpython-310-aarch64-linux-gnu.so /home/recomputer/jetson-voice/app_overlay/"

chown -R recomputer:recomputer /home/recomputer/jetson-voice/app_overlay
```

### B — 从 recomputer-desktop 直接 rsync
```bash
fleet transfer recomputer-desktop:/home/recomputer/jetson-voice/app_overlay/ jetson-new:/home/recomputer/jetson-voice/app_overlay/
# 注意 .so 可能仍需在 jetson-new 上重 build，因 Orin 跨机 ABI 若 CUDA/TRT 版本一致理论可用
```

## 6. 起容器

`docker-compose.override.yml` 已在 repo 里（或从 recomputer-desktop 拉）。

```bash
fleet exec jetson-new -- "cd /home/recomputer/jetson-voice/reachy_speech && docker compose up -d"
# 等 ~30s load engine
sleep 30
fleet exec jetson-new -- "curl -sf http://localhost:8621/health"
# 期望: {"tts":true,"tts_backend":"qwen3_trt",...}
```

## 7. 验证 V2V 和真人 ASR 精度

从 Mac 跑：
```bash
# V2V 功能 + 延迟
NO_PROXY=<jetson-new-ip>,localhost HTTP_PROXY= uv run --with websocket-client --with numpy --with requests \
  python3 /tmp/test_v2v_real.py --host <jetson-new-ip>:8621

# 真人 ASR 精度（LibriSpeech dev-clean 需先下）
NO_PROXY=<jetson-new-ip>,localhost uv run --with jiwer --with websocket-client --with numpy --with soundfile \
  python3 tests/asr_real_wav_eval/run_eval.py --host <jetson-new-ip>:8621
```

期望：
- V2V 3/3 句不崩，EOS→首音频 1-2s 范围
- LibriSpeech median WER ~30%

## 8. 如果 V2V 崩（CUDA 906 或 Myelin error）

参考 handover §稳定性差距 Issue 1。回退到 Matcha+Paraformer：
```bash
# 编辑 docker-compose.override.yml 改 LANGUAGE_MODE=zh_en
fleet exec jetson-new -- "docker restart reachy_speech-speech-1"
```

见 `docs/ops/rollback-to-matcha.md`。

## 禁区（重申 — 新设备同适用）

- ❌ `rm -rf app_overlay/*`（python 文件不在 git，删了回不来）
- ❌ `docker compose down` / `docker compose up`（bind mount 路径缺失会自动创建空目录 → 毁）
- ❌ 裸 `cmake ..`（不设 `ORT_ROOT=/home/recomputer/ort-from-container` ABI 错）
- ✅ 只 `docker restart reachy_speech-speech-1`
- ✅ 只 `bash benchmark/cpp/build.sh`

## 时长估计（最佳情况）

| 步骤 | 预计时间 |
|---|---|
| 2. Clone | 1 min |
| 3. 模型 rsync (16GB from recomputer-desktop) | 10-15 min (千兆局域网) |
| 4. Build .so | 3-5 min |
| 5. Overlay 准备 | 2 min |
| 6. 起容器 + warmup | 1 min |
| 7. V2V 验证 | 2 min |
| **总计** | **20-30 min** |

## 必备文件清单（别漏）

新设备必须有的（不在 git）：
- `/home/recomputer/ort-from-container/` (headers + libs，从镜像 extract)
- `/opt/models/` (16 GB，挂 docker volume)
- `/home/recomputer/jetson-voice/app_overlay/*.py` + `backends/` + `.so`
- `/home/recomputer/jetson-voice/reachy_speech/docker-compose.override.yml`
- Docker image `jetson-voice-speech:v3.1-with-vad`（从 registry 拉或 export/import）
