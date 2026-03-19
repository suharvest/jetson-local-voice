# Jetson Voice

**A fully local bilingual voice server with sub-200ms latency and zero cloud dependency — built for edge AI, robots, and real-time voice interaction.**

[![GitHub stars](https://img.shields.io/github/stars/Seeed-Projects/jetson-voice?style=social)](https://github.com/Seeed-Projects/jetson-voice)
[![sherpa-onnx](https://img.shields.io/badge/engine-sherpa--onnx-green.svg)](https://github.com/k2-fsa/sherpa-onnx)
[![Kokoro TTS](https://img.shields.io/badge/TTS-Kokoro%20v1.0-orange.svg)](https://huggingface.co/hexgrad/Kokoro-82M)
[![Docker](https://img.shields.io/badge/deploy-Docker-blue.svg)](https://www.docker.com/)
[![Jetson](https://img.shields.io/badge/platform-Jetson%20Orin-76b900.svg)](https://developer.nvidia.com/embedded-computing)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="media/hero.png" alt="Jetson Voice — sub-200ms ASR + TTS on edge" width="640" />
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
docker run -d --name jetson-voice \
  --runtime nvidia --ipc host \
  -p 8621:8000 \
  -v jetson-voice-models:/opt/models \
  --restart unless-stopped \
  sensecraft-missionpack.seeed.cn/solution/jetson-voice:v2.1

# Verify (wait ~40s for warmup)
curl http://localhost:8621/health
# {"asr":false,"tts":true,"streaming_asr":true}
```

**English-only mode** (Kokoro TTS + Zipformer ASR):

```bash
docker run -d --name jetson-voice \
  --runtime nvidia --ipc host \
  -p 8621:8000 \
  -e LANGUAGE_MODE=en \
  -v jetson-voice-models:/opt/models \
  --restart unless-stopped \
  sensecraft-missionpack.seeed.cn/solution/jetson-voice:v2.1
```

**Build from source:**

```bash
git clone https://github.com/Seeed-Projects/jetson-voice.git
cd jetson-voice
docker compose build && docker compose up -d
```

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
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

| Service | Endpoint | zh_en (default) | en | Protocol |
|---------|----------|-----------------|-----|----------|
| **Streaming ASR** | `WS /asr/stream` | Paraformer bilingual | Zipformer English | WebSocket: int16 PCM in, JSON out |
| **Streaming TTS** | `POST /tts/stream` | Matcha-TTS + Vocos | Kokoro v1.0 (53 speakers) | HTTP: JSON in, raw PCM stream |
| **Batch TTS** | `POST /tts` | Matcha-TTS + Vocos | Kokoro v1.0 | HTTP: JSON in, WAV out |
| Offline ASR | `POST /asr` | SenseVoice (zh+en+ja+ko+yue) | SenseVoice (same) | HTTP: WAV upload, JSON out |

The service is model-agnostic at the API level — clients send audio/text, get audio/text back. Swap models without changing client code.

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
| **Jetson Voice (Orin Nano)** | **~$250 one-time** | **$0** | **~110-180ms** |
| **Jetson Voice (Orin NX)** | **~$400 one-time** | **$0** | **~110-180ms** |
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
| `LANGUAGE_MODE` | `zh_en` | `zh_en` (Chinese+English) or `en` (English only) |
| `TTS_PROVIDER` | `cuda` | ONNX execution provider |
| `TTS_DEFAULT_SID` | `52` | Default TTS speaker ID (52=af_cute, 3=af_heart) |
| `TTS_DEFAULT_SPEED` | `1.0` | TTS playback speed |
| `TTS_NUM_THREADS` | `4` | TTS inference threads |
| `TTS_PITCH_SHIFT` | `0` | Pitch shift in semitones (e.g. `2` = higher, `-2` = lower) |
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
jetson-voice/
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
├── Dockerfile               # Multi-stage build for JetPack 6.2
├── docker-compose.yml       # nvidia runtime, GPU, model volume
└── setup-performance.sh     # Jetson clock/power tuning
```

## Acknowledgements

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — speech inference engine powering all ASR and TTS models
- [next-gen Kaldi](https://github.com/k2-fsa) — the research foundation behind sherpa-onnx
- [Paraformer](https://github.com/modelscope/FunASR) — streaming bilingual ASR model
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) — fast flow-matching TTS (zh+en mode)
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) — high-quality English TTS with 53 speakers (en mode)
- [Zipformer](https://github.com/k2-fsa/icefall) — efficient transducer ASR (en mode)
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — multilingual offline ASR
