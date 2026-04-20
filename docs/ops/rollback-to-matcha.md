# 回滚到 Matcha + Paraformer（生产配置）

2026-04-20 起草。Qwen3 TTS/ASR 稳定性不足以跑 production，需回滚到之前的 Matcha TTS + Paraformer ASR。

## 背景

现 production 部署用 Qwen3 TTS + Qwen3 ASR（多语言+52 种），但：
- ASR+TTS 并发触发 CUDA stream 冲突（见 handover §稳定性差距 Issue 1）
- 10 分钟测试容器重启 2 次
- 不适合 7×24 production

Matcha zh+en TTS + Paraformer streaming ASR 是之前稳定配置，够中英文生产用。

## 回滚步骤（在 `recomputer-desktop`）

### 1. 编辑 `docker-compose.override.yml` 切 env

路径：`/home/recomputer/jetson-voice/reachy_speech/docker-compose.override.yml`

把：
```yaml
environment:
  - LANGUAGE_MODE=multilanguage   # ← 这行触发 Qwen3
  - ASR_DIAG_LOG=1
```

改为：
```yaml
environment:
  - LANGUAGE_MODE=zh_en           # ← Matcha + Paraformer
  - TTS_BACKEND=sherpa            # 显式指定非 qwen3
```

### 2. Overlay 可保留也可移除

**保留 overlay（推荐）**：让新修的 `app/backends/*.py` 继续生效，`LANGUAGE_MODE=zh_en` 会自动让 `tts_backend.py` 走 sherpa 路径。Overlay 的 Qwen3 代码不会被调用。

**移除 overlay**（纯镜像）：如果不信 overlay 代码，把 override.yml 的整个 `volumes:` 段注释掉。容器会跑镜像原装 v3.1-with-vad 的 sherpa/paraformer 路径。

### 3. 重启容器

**禁止** `docker compose down/up`（参考 `.claude/projects/.../memory/project_deploy_footguns.md`）。只：
```bash
docker restart reachy_speech-speech-1
```

### 4. 验证

```bash
sleep 30
curl -sf http://localhost:8621/health
# 期望: "tts_backend":"sherpa", "asr_backend":"paraformer"
```

## 回滚后性能基线（历史数据）

- TTS: Matcha RTF ≈ 0.3-0.5，首音频 ~400ms
- ASR: Paraformer streaming latency ~200ms finalize
- 内存：~2.5 GB container
- 稳定：老数据 500+ req 无 crash

## 恢复到 Qwen3

改 env 反向，`docker restart`。Qwen3 engine 文件仍在 `/opt/models/qwen3-*`，不需要重新下载。

## 两套并存？

目前 backend 选择是全局 env 二择一。若要并存（如 `/asr_qwen3` 单独路由）需改 `app/main.py` 路由逻辑。不推荐，复杂度换不回多少好处。
