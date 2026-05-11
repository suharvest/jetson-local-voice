# Seeed Local Voice

**A fully local bilingual voice server with sub-200ms latency and zero cloud dependency — built for edge AI, robots, and real-time voice interaction. Runs on Jetson Orin, RK3576, RK3588, and Raspberry Pi 4/5.**

[![GitHub stars](https://img.shields.io/github/stars/suharvest/seeed-local-voice?style=social)](https://github.com/suharvest/seeed-local-voice)
[![sherpa-onnx](https://img.shields.io/badge/engine-sherpa--onnx-green.svg)](https://github.com/k2-fsa/sherpa-onnx)
[![Kokoro TTS](https://img.shields.io/badge/TTS-Kokoro%20v1.0-orange.svg)](https://huggingface.co/hexgrad/Kokoro-82M)
[![Docker](https://img.shields.io/badge/deploy-Docker-blue.svg)](https://www.docker.com/)
[![Devices](https://img.shields.io/badge/devices-Jetson%20%7C%20RK3576%20%7C%20RK3588%20%7C%20RPi4%2F5-76b900.svg)](#supported-devices)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

**Supported devices**: Jetson Orin (Nano / NX / AGX, CUDA) · RK3576 · RK3588 (Rockchip NPU) · Raspberry Pi 4 / 5 (CPU)

<p align="center">
  <img src="media/hero.png" alt="Seeed Local Voice — sub-200ms ASR + TTS on edge" width="640" />
</p>

<!-- TODO: Add demo GIF showing voice-in → text → voice-out round-trip -->

Run ASR and TTS locally on Jetson or any CUDA device with one Docker image. No cloud, no API keys, no internet — just fast, deployable voice infrastructure for edge AI systems.

## Key Features

- **Sub-200ms latency** — ASR-to-first-audio in ~110ms (zh+en) / ~180ms (en), beating cloud APIs by 3-5x
- **Under $500 total cost** — runs on a Jetson Orin Nano (~$250) or NX (~$400). No cloud bills, no API fees, ever
- **Bilingual** — Chinese+English (Paraformer + Matcha-TTS) or English-only (Zipformer + Kokoro v1.0), switch with one env var
- **One-command deploy** — `docker run` and you're done. Models auto-download on first start
- **Lightweight** — only ~650 MB RAM, ~32% CPU idle. Leaves room for LLM, vision, and other workloads on the same device
- **Streaming** — real-time WebSocket ASR with partial results; streaming TTS with sentence-level chunking
- **53 TTS voices** — Kokoro v1.0 with 53 speakers across 8 languages, plus custom voice support via pitch shift
- **Zero cloud dependency** — fully offline, runs on-device with CUDA acceleration

## Quick Start

Pull and run. Models auto-download on first start (~1 min) and are cached in a Docker volume:

```bash
# Chinese + English (default)
docker run -d --name seeed-local-voice \
  --runtime nvidia --ipc host \
  -p 8621:8000 \
  -v seeed-local-voice-models:/opt/models \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
  -v /lib/aarch64-linux-gnu:/host-libs:ro \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/host-nvidia-libs:/host-libs:/host-cuda \
  --restart unless-stopped \
  sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v3.0-slim

# Verify (wait ~40s for warmup)
curl http://localhost:8621/health
# {"asr":false,"tts":true,"streaming_asr":true}
```

**English-only mode** (Kokoro TTS + Zipformer ASR):

```bash
docker run -d --name seeed-local-voice \
  --runtime nvidia --ipc host \
  -p 8621:8000 \
  -e LANGUAGE_MODE=en \
  -v seeed-local-voice-models:/opt/models \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
  -v /lib/aarch64-linux-gnu:/host-libs:ro \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/host-nvidia-libs:/host-libs:/host-cuda \
  --restart unless-stopped \
  sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:v3.0-slim
```

**Deploy with compose** (recommended for production):

```bash
git clone https://github.com/suharvest/seeed-local-voice.git
cd seeed-local-voice

# Chinese + English (default)
docker compose -f deploy/docker-compose.yml up -d

# English only
LANGUAGE_MODE=en docker compose -f deploy/docker-compose.yml up -d

# Qwen3 multilingual ASR/TTS
SEEED_LOCAL_VOICE_PROFILE=jetson-multilang-highperf \
QWEN3_HF_REPO_ID=<your-org/qwen3-edgellm-jetson-artifacts> \
docker compose -f deploy/docker-compose.yml up -d
```

**Build from source** (for development):

```bash
cd seeed-local-voice
docker compose build && docker compose up -d
```

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Qwen3 Multilanguage Paths](#qwen3-multilanguage-paths)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Configuration](#configuration)
- [Models](#models)
- [Patched sherpa-onnx](#patched-sherpa-onnx)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

## Architecture

```text
┌───────────────────────────────────────────────────────────┐
│  Jetson Orin NX / Nano (CUDA 12.6)                        │
│                                                           │
│  FastAPI service (:8000)                                  │
│  ├── WS /asr/stream    Streaming ASR                      │
│  │     └─ zh_en: Paraformer  │  en: Zipformer             │
│  ├── POST /asr          SenseVoice offline ASR (both)     │
│  ├── POST /tts          Batch TTS                         │
│  └── POST /tts/stream   Streaming TTS                     │
│        └─ zh_en: Matcha-TTS  │  en: Kokoro v1.0           │
│                                                           │
│  sherpa-onnx + ONNX Runtime 1.20 (CUDA)                   │
└───────────────────────────────────────────────────────────┘
         ▲ HTTP / WebSocket
         │
   Any client (SBC, laptop, robot, ...)
```

Models are selected automatically based on `LANGUAGE_MODE`:

| Service | Endpoint | zh_en (default) | en | multilanguage | Protocol |
|---------|----------|-----------------|-----|---------------|----------|
| **Streaming ASR** | `WS /asr/stream` | Paraformer bilingual | Zipformer English | Qwen3-ASR (52 langs) | WebSocket: int16 PCM in, JSON out |
| **Streaming TTS** | `POST /tts/stream` | Matcha-TTS + Vocos | Kokoro v1.0 | Qwen3-TTS (voice clone) | HTTP: JSON in, raw PCM stream |
| **Batch TTS** | `POST /tts` | Matcha-TTS + Vocos | Kokoro v1.0 | Qwen3-TTS (voice clone) | HTTP: JSON in, WAV out |
| Offline ASR | `POST /asr` | SenseVoice (zh+en+ja+ko+yue) | SenseVoice (same) | Qwen3-ASR (52 langs) | HTTP: WAV upload, JSON out |

**Backend capabilities differ:**

| Backend | Speed control | Pitch shift | Voice clone | Languages | Streaming |
|---------|--------------|-------------|-------------|-----------|-----------|
| Sherpa (zh_en/en) | ✅ | ✅ | ❌ | 2 (zh+en) | ✅ |
| Qwen3 (multilanguage) | ❌ | ❌ | ✅ (x-vector) | 52 | ✅ |

The service is model-agnostic at the API level — clients send audio/text, get audio/text back. Swap models without changing client code. Unsupported parameters return `501` with `{"required_capability": "..."}`.

## Qwen3 Multilanguage Paths

Qwen3-ASR and Qwen3-TTS are exposed to Seeed Local Voice as the `multilanguage`
mode. The product service stays in this repository; Qwen-specific export,
engine build, worker/runtime glue, and performance scripts live in the
standalone companion repo
[`suharvest/qwen3-edgellm-jetson`](https://github.com/suharvest/qwen3-edgellm-jetson).
Large model artifacts live in the Hugging Face artifact repository
[`harvestsu/qwen3-edgellm-jetson-artifacts`](https://huggingface.co/harvestsu/qwen3-edgellm-jetson-artifacts)
described by the manifest in `qwen3-edgellm-jetson/deploy/artifacts/qwen3_manifest.json`.

**Quickest path on a fresh Orin NX (one shot, end-to-end):**

```bash
git clone https://github.com/suharvest/qwen3-edgellm-jetson.git
bash qwen3-edgellm-jetson/scripts/reproduce_qwen3_highperf.sh \
  --reference /path/to/24kHz_mono.wav   # optional: also gates voice clone
```

The orchestrator clones this repo + the EdgeLLM fork, builds the runtime,
downloads + SHA-256-verifies the HF artifacts, builds the slim docker
image (`deploy/docker/Dockerfile.jetson` with `--build-arg LANGUAGE_MODE=multilanguage`), starts the service, and runs
`scripts/verify_reproduction.sh` which fail-fasts on plugin-symbol set,
artifact integrity, TTS→ASR loopback, and voice clone. Exit 0 means the
slim container on port 18092 is healthy and serving the validated stack.

After that, you can sanity-check a running service against an existing
deployment with the same verifier alone:

```bash
bash qwen3-edgellm-jetson/scripts/verify_reproduction.sh \
    --plugin /opt/edgellm-bin/libNvInfer_edgellm_plugin.so \
    --artifact-root /opt/models/qwen3-edgellm \
    --service-url http://localhost:18092 \
    [--embedding /tmp/precomputed_speaker_emb.b64]
```

For manual deployments, profile selection follows the existing pattern:

```bash
SEEED_LOCAL_VOICE_PROFILE=jetson-multilang-highperf uvicorn app.main:app --host 0.0.0.0 --port 8621
```

Use `jetson-multilang-highperf-nx` on Orin NX when consuming the NX-native
engine set; the default `jetson-multilang-highperf` profile targets the Nano
artifact set.

Profiles live in [`configs/profiles`](configs/profiles). They set environment
defaults only; explicit env vars still override them. Qwen3-ASR and Qwen3-TTS
have two maintained runtime profiles under the same API surface:

| Profile | Goal | Default behavior |
|---------|------|------------------|
| `official` | Minimal-diff EdgeLLM example path. Keep it close enough to upstream that it can be reviewed or upstreamed as an official Qwen3 ASR/TTS example. | Only semantic/correctness fixes belong here: tokenizer/control-token layout, sampling semantics, runtime contract fixes, and stream callback correctness. It uses regular exported Talker/CodePredictor/Code2Wav directories and does not enable our Orin memory/latency strategies by default. |
| `highperf` | Product low-latency dual-resident path for our Orin deployment. | Full ASR/TTS vocab, ASR FP8 embedding directory when present, TTS W8A16 Talker support, CP BF16 I/O + `lm_head` pretranspose, stateful Code2Wav, CP decode CUDA graph, `ACTIVE_CP_GROUPS=13`. |

The repo default Qwen profile is `highperf` because it is the measured product
path. `LANGUAGE_MODE=multilanguage` now defaults to the EdgeLLM worker backend
(`ASR_BACKEND=trt_edgellm`, `TTS_BACKEND=trt_edgellm`). Older local Qwen
backends can still be selected explicitly for legacy reproduction, but they are
not product runtime choices.

Use `official` when preparing or validating an EdgeLLM-upstream-compatible example:

Branch ownership:

- EdgeLLM official/minimal branch: `official-qwen3-tts-upstream-runtime` in the EdgeLLM fork.
- Split upstream PR branches: `pr-jetson-build-compat`, `pr-export-builder-robustness`, `pr-qwen3-tts-runtime-correctness`.
- Seeed Local Voice integration branch for that minimal official backend: `product-qwen3-tts-official-backend`.
- Seeed Local Voice high-performance/product branch: `qwen3tts-accurate-20260507`.

```bash
EDGE_LLM_QWEN3_PROFILE=official \
ASR_BACKEND=trt_edgellm \
TTS_BACKEND=trt_edgellm \
EDGE_LLM_TTS_BACKEND=edgellm_worker \
EDGE_LLM_ASR_VOCAB_PRUNED=0 \
EDGE_LLM_TTS_VOCAB_PRUNED=0 \
uvicorn app.main:app --host 0.0.0.0 --port 8621
```

Use the high-performance path for Orin low-latency V2V:

```bash
EDGE_LLM_QWEN3_PROFILE=highperf \
ASR_BACKEND=trt_edgellm \
TTS_BACKEND=trt_edgellm \
EDGE_LLM_TTS_BACKEND=edgellm_worker \
EDGE_LLM_TTS_STATEFUL_CODE2WAV=1 \
EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR=/tmp/qwen3_code2wav_stateful_engine \
EDGE_LLM_TTS_CP_BF16_IO_DIR=/tmp/qwen3_tts_cp_lmhead_pretranspose_0510/cp_dir \
EDGE_LLM_ASR_VOCAB_PRUNED=0 \
EDGE_LLM_TTS_VOCAB_PRUNED=0 \
uvicorn app.main:app --host 0.0.0.0 --port 8621
```

Current Orin 8GB frozen baseline, MAXN_SUPER:

| Path | Metric | Result |
|------|--------|--------|
| Qwen3 highperf dual-resident V2V | warm `EOS -> first audio` | `~620ms` |
| Qwen3 highperf TTS-only | warm first chunk | `~540ms` |
| Qwen3 highperf TTS-only | RTF | `0.64-0.65` |
| Qwen3 highperf memory | dual-resident minimum `MemAvailable` | `~759-850MB` |
| Qwen3 highperf NX-native 40W locked | warm `EOS -> first audio` | `~595-596ms` |

Detailed artifacts, engine md5s, and quality results are in [`docs/plans/qwen3-current-frozen-baseline-2026-05-10.md`](docs/plans/qwen3-current-frozen-baseline-2026-05-10.md). Cross-device performance and replication notes are tracked in [`docs/performance/qwen3-orin-profiles-2026-05-10.md`](docs/performance/qwen3-orin-profiles-2026-05-10.md). The ASR and TTS plugin paths are intentionally separate in the frozen baseline: the TTS highperf plugin is not a drop-in replacement for the ASR plugin until its ASR Attention/FMHA path is separately validated.

Regression scope:

- `official`: semantic correctness and upstream-compatibility checks only. It should stay small and reviewable; do not add W8A16, FP8 embedding, pruned/full-vocab policy changes, stateful Code2Wav, CP graph, or engine-specific Orin tuning here.
- `highperf`: full product regression, including Qwen3 ASR + Qwen3 TTS dual residency, V2V latency, memory floor, and ASR round-trip quality. This is where our Orin-specific memory and latency work belongs.

Artifact handling:

```bash
# scripts and manifest live in the sibling qwen3-edgellm-jetson repo
cd ~/project/qwen3-edgellm-jetson
python3 scripts/deploy_qwen3_artifacts.py \
  --set orin-nano-highperf-2026-05-10 \
  --root /opt/models/qwen3-edgellm \
  --check-only
```

Without `--check-only`, the script downloads the missing files from the HF repo
identified by `QWEN3_HF_REPO_ID` or the manifest. `app/model_downloader.py`
auto-invokes the same script at service startup; set `QWEN3_EDGELLM_JETSON_ROOT`
if your checkout is not at `~/project/qwen3-edgellm-jetson`. Use
`--verify-sha256` to enforce the SHA-256 sidecar (catches partial downloads).

## API Reference

### Streaming ASR (WebSocket)

```
WS /asr/stream?sample_rate=16000&language=auto
```

- Client sends: raw **int16 PCM bytes** (audio chunks, e.g. 100ms each)
- Client sends: **empty bytes** `b""` to signal end of audio
- Server sends: JSON `{"text": "...", "is_final": bool, "is_stable": bool}`

```python
import asyncio, websockets

async def transcribe():
    async with websockets.connect("ws://jetson:8000/asr/stream?sample_rate=16000") as ws:
        for chunk in audio_chunks:  # np.int16 arrays
            await ws.send(chunk.tobytes())
            result = await ws.recv()  # partial results
        await ws.send(b"")  # signal end
        final = await ws.recv()  # {"text": "...", "is_final": true}
```

### Offline ASR (HTTP)

```bash
curl -X POST http://jetson:8000/asr \
  -F "file=@recording.wav" -F "language=auto"
# {"text": "transcribed text"}
```

### TTS (HTTP)

```bash
curl -X POST http://jetson:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "sid": 52, "speed": 1.0}' \
  --output output.wav
```

Parameters: `text` (required), `sid` (speaker ID, default 52), `speed` (rate, default 1.0)

**Note:** `speed` and `pitch` parameters only work in Sherpa backend (`zh_en`/`en` mode). Qwen3-TTS (`multilanguage` mode) does not support speed/pitch adjustment — these are Sherpa-specific capabilities.

### TTS Streaming (HTTP)

Returns raw PCM: first 4 bytes = sample rate (uint32 LE), then int16 samples.

```
POST /tts/stream
Content-Type: application/json
{"text": "Hello world", "sid": 52}
```

### Health Check

```
GET /health  →  {"asr": bool, "tts": bool, "streaming_asr": bool}
```

## Performance

### Latency (Jetson Orin NX 16GB, CUDA 12.6, MAXN)

| | ASR TTFT | TTS TTFT | ASR + TTS | vs Cloud |
|---|---------|---------|-----------|----------|
| **zh_en** (Chinese+English) | ~50ms | ~60ms | **~110ms** | 3-5x faster |
| **en** (English only) | ~50ms | ~130ms | **~180ms** | 2-4x faster |

Full voice-to-voice latency depends on LLM inference time (not included above).

### Resource Usage (Jetson Orin NX 16GB)

| State | CPU (8-core) | RAM | GPU |
|-------|-------------|-----|-----|
| Idle (models loaded) | ~32% | ~650 MB | 0% |
| During TTS inference | ~63% | ~658 MB | burst |
| During ASR streaming | ~32% | ~650 MB | minimal |

The service uses only ~650 MB RAM, leaving plenty of headroom for LLM inference (Ollama), computer vision, or robot control on the same device.

### Cost Comparison

| Setup | Hardware Cost | Per-request Cost | Latency |
|-------|-------------|-----------------|---------|
| **Seeed Local Voice (Orin Nano)** | **~$250 one-time** | **$0** | **~110-180ms** |
| **Seeed Local Voice (Orin NX)** | **~$400 one-time** | **$0** | **~110-180ms** |
| Google Cloud Speech + TTS | $0 | ~$0.01/request | 300-800ms |
| Azure Speech Services | $0 | ~$0.01/request | 200-500ms |
| OpenAI TTS + Whisper | $0 | ~$0.02/request | 500ms-2s |

At ~1000 requests/day, a Jetson pays for itself in under 2 months vs cloud APIs.

### Benchmarks

**zh_en mode:**

| Metric | Value |
|--------|-------|
| Paraformer TTFT | ~50ms |
| Paraformer finalize | ~45ms |
| Paraformer accuracy | 80.8% (26 synthetic sentences) |
| Matcha TTS TTFT | ~60ms (short text) |
| Matcha TTS latency | ~150ms (typical Chinese sentence) |

**en mode:**

| Metric | Value |
|--------|-------|
| Zipformer TTFT | ~50ms |
| Kokoro TTS TTFT | ~130ms (short text) |
| Kokoro TTS latency | ~300ms (typical sentence) |

### TTS Model Comparison

We evaluated 4 TTS models for TTFT (time-to-first-audio-chunk). Matcha-TTS was selected for zh_en mode (best Chinese quality), Kokoro for en mode (best English quality):

| Model | TTFT (short) | TTFT (long) | Chinese Quality | English Quality | Used in |
|-------|-------------|-------------|-----------------|-----------------|---------|
| **Matcha-TTS + Vocos** | ~60ms | ~150ms | Good | Fair | **zh_en** |
| **Kokoro v1.0** | ~130ms | ~300ms | — | Excellent | **en** |
| CosyVoice3 | ~800ms | ~2s | Excellent | — | — |
| F5-TTS | ~2.5s | ~5s | Excellent | — | — |

Benchmark scripts are in `benchmarks/`. See `benchmarks/archive/` for detailed F5-TTS optimization experiments (CUDA, TensorRT, NFE sweep).

### Performance Tuning

Run once after boot to lock clocks to max:

```bash
sudo ./setup-performance.sh
```

This sets MAXN power mode, locks CPU/GPU clocks, and disables dynamic frequency scaling. Critical for consistent inference latency.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGUAGE_MODE` | `zh_en` | `zh_en` (Chinese+English), `en` (English only), or `multilanguage` (Qwen3, 52 langs) |
| `TTS_PROVIDER` | `cuda` | ONNX execution provider |
| `TTS_DEFAULT_SID` | `52` | Default TTS speaker ID (52=af_cute, 3=af_heart) — Sherpa only |
| `TTS_DEFAULT_SPEED` | `1.0` | TTS playback speed — **Sherpa only** |
| `TTS_NUM_THREADS` | `4` | TTS inference threads |
| `TTS_PITCH_SHIFT` | `0` | Pitch shift in semitones — **Sherpa only** |
| `SENSEVOICE_LANGUAGE` | `auto` | SenseVoice language hint |
| `STREAMING_ASR_PROVIDER` | `cuda` | Streaming ASR execution provider |
| `MODEL_DIR` | `/opt/models` | Model storage directory |

Copy `.env.example` to `.env` to customize.

## Models

Auto-downloaded on first start and cached in a Docker volume:

| Model | Size | Mode | Purpose |
|-------|------|------|---------|
| Paraformer streaming zh-en | ~230 MB | `zh_en` | Streaming ASR (bilingual) |
| Matcha-TTS + Vocos zh-en | ~125 MB | `zh_en` | TTS synthesis |
| Zipformer streaming en | ~65 MB | `en` | Streaming ASR (English only) |
| Kokoro TTS v1.0 | ~719 MB | `en` | TTS synthesis (English, 53 speakers) |
| SenseVoice zh-en-ja-ko-yue | ~500 MB | both | Offline ASR (5 languages) |
| Qwen3-TTS 0.6B + TRT engines | ~2.5 GB | `multilanguage` | TTS + voice clone (52 languages) |
| Qwen3-ASR encoder + decoder | ~1.5 GB | `multilanguage` | ASR (52 languages, streaming) |

## Patched sherpa-onnx

Includes a patched sherpa-onnx that fixes Paraformer streaming tail truncation (stock version drops the last 1-3 characters). The patch:

1. **IsReady()** — forces decode of remaining frames after `InputFinished()`
2. **DecodeStream()** — zero-pads partial final chunks
3. **CIF force-fire** — emits residual tokens at end-of-stream

Pre-built `.so` files in `patches/sherpa-onnx-lib/` (aarch64, Python 3.10, CUDA 12.6).
See `patches/README.md` for rebuild instructions.

## Requirements

- **Jetson Orin Nano** (~$250) or **Orin NX** (~$400) with JetPack 6.2, CUDA 12.6 — or any CUDA-capable device with Docker
- Docker with `nvidia` runtime
- ~650 MB RAM (leaves room for LLM + other workloads)
- ~5 GB disk for models

## Project Structure

```text
seeed-local-voice/
├── app/                     # FastAPI service
│   ├── main.py              # Endpoints and startup
│   ├── asr_service.py       # SenseVoice offline ASR
│   ├── streaming_asr_service.py  # Paraformer/Zipformer streaming ASR
│   ├── tts_service.py       # TTS (Matcha / Kokoro, batch + streaming)
│   └── model_downloader.py  # On-demand model download + voice patching
├── voices/                  # Custom voice embeddings (auto-patched into model)
├── benchmarks/              # TTS model TTFT comparisons
├── patches/                 # Paraformer EOF truncation fix
├── scripts/                 # Model download, ORT patching
├── deploy/
│   ├── docker-compose.yml   # Production deploy (pre-built image)
│   └── docker/
│       ├── Dockerfile.jetson  # Jetson Orin Nano/NX/AGX (zh_en or multilanguage)
│       ├── Dockerfile.rk      # Rockchip RK3576/RK3588 NPU
│       └── Dockerfile.rpi     # Raspberry Pi 4/5 (CPU)
└── setup-performance.sh     # Jetson clock/power tuning
```

## Changelog

### v3.0-slim

- **95% smaller image** — 898 MB vs 17.7 GB. Multi-stage build extracts only the runtime Python packages (onnxruntime + sherpa-onnx) from v2.2 into an `ubuntu:22.04` base
- **Host GPU library mounts** — CUDA/TensorRT/cuDNN libraries are bind-mounted from the host JetPack installation instead of baked into the image, improving cross-JetPack version compatibility
- **Same performance** — identical TTS/ASR latency and CUDA provider support (TRT + CUDA + CPU)

> **Note:** v3.0-slim is derived from v2.2 (the full image). There is no standalone v3.0 fat image — `v3.0-slim` is the current production release. This version requires host GPU lib mounts at runtime (see Quick Start). This is standard practice for Jetson containers and matches the pattern used by vision-trt and other optimized images.

### v2.2

- **Enable endpoint detection** — server now proactively sends `is_final` when the speaker pauses (0.6s trailing silence), reducing response latency
- **Fix WebSocket lifecycle** — properly close connections after finalize to prevent stale connection reuse
- **Production deploy compose** — `deploy/docker-compose.yml` with pre-built image (no build step needed)

### v2.1

- Streaming TTS with sentence-level callback
- Custom voice embedding support via pitch shift

### v2.0

- Initial release: Paraformer + Matcha (zh_en), Zipformer + Kokoro (en)
- Patched sherpa-onnx for Paraformer streaming EOF fix

## Acknowledgements

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — speech inference engine powering all ASR and TTS models
- [next-gen Kaldi](https://github.com/k2-fsa) — the research foundation behind sherpa-onnx
- [Paraformer](https://github.com/modelscope/FunASR) — streaming bilingual ASR model
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) — fast flow-matching TTS (zh+en mode)
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) — high-quality English TTS with 53 speakers (en mode)
- [Zipformer](https://github.com/k2-fsa/icefall) — efficient transducer ASR (en mode)
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — multilingual offline ASR
