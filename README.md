# Jetson Voice

**~110ms ASR + TTS on edge devices — GPU-accelerated voice stack powered by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), zero cloud dependency.**

[![sherpa-onnx](https://img.shields.io/badge/speech-sherpa--onnx-green.svg)](https://github.com/k2-fsa/sherpa-onnx)
[![Docker](https://img.shields.io/badge/deploy-Docker-blue.svg)](https://www.docker.com/)
[![Jetson](https://img.shields.io/badge/platform-Jetson%20Orin-76b900.svg)](https://developer.nvidia.com/embedded-computing)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="media/hero.png" alt="Jetson Voice — 50ms ASR + 60ms TTS on edge" width="640" />
</p>

Turn any CUDA device into a local voice server. Speak into it, get text back in 50ms. Send text, get speech in 60ms. No cloud, no API keys, no internet needed.

Jetson Voice wraps [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) models (Paraformer ASR, Matcha TTS, SenseVoice) in a FastAPI service with HTTP and WebSocket endpoints — deploy with one `docker compose up`.

### Latency (Jetson Orin NX 16GB, CUDA 12.6, MAXN)

| Stage | Model | Latency | Note |
|-------|-------|---------|------|
| **ASR** | Paraformer zh+en (streaming) | ~50ms TTFT | Primary. Real-time partial results via WebSocket |
| **TTS** | Matcha-TTS + Vocos (streaming) | ~60ms TTFT | Primary. Best Chinese quality, streaming PCM |
| ASR (fallback) | SenseVoice 5-lang | ~200ms | Batch mode, offline, lazy-loaded |

ASR + TTS combined: **~110ms** (ASR finalize + TTS first chunk). Full voice-to-voice latency depends on LLM inference time (not included here).

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Services](#services)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Configuration](#configuration)
- [Patched sherpa-onnx](#patched-sherpa-onnx)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

## Quick Start

### Option 1: Pre-built Image (Recommended)

Pull and run the pre-built image — all dependencies and models are baked in, no build or download needed:

```bash
docker run -d --name jetson-voice \
  --runtime nvidia --ipc host \
  -p 8621:8000 \
  -e TTS_DEFAULT_SID=0 \
  --restart unless-stopped \
  sensecraft-missionpack.seeed.cn/solution/jetson-voice:v1.0

# Wait ~40s for model warmup, then verify
curl http://localhost:8621/health
# {"asr":false,"tts":true,"streaming_asr":true}
```

### Option 2: Build from Source

```bash
git clone https://github.com/Seeed-Projects/jetson-voice.git
cd jetson-voice

# Build & run
docker compose build
docker compose up -d

# Verify
curl http://localhost:8000/health
# {"asr":true,"tts":true,"streaming_asr":true}
```

Models (~1.5 GB total) are auto-downloaded on first start.

## Architecture

```text
┌──────────────────────────────────────────────────┐
│  Jetson Orin NX (CUDA 12.6)                      │
│                                                  │
│  FastAPI service (:8000)                         │
│  ├── WS /asr/stream    Paraformer streaming ASR  │
│  ├── POST /asr          SenseVoice offline ASR   │
│  ├── POST /tts          Matcha batch TTS         │
│  └── POST /tts/stream   Matcha streaming TTS     │
│                                                  │
│  sherpa-onnx + ONNX Runtime 1.20 (CUDA)          │
└──────────────────────────────────────────────────┘
         ▲ HTTP/WebSocket
         │
   Any client (SBC, laptop, robot, ...)
```

The service is model-agnostic at the API level — clients send audio/text, get audio/text back. Swap models without changing client code.

## Services

| Service | Model | Endpoint | Protocol | Role |
|---------|-------|----------|----------|------|
| **Streaming ASR** | Paraformer bilingual zh+en | `WS /asr/stream` | WebSocket: int16 PCM in, JSON out | Primary ASR |
| **Streaming TTS** | Matcha-TTS + Vocos zh+en | `POST /tts/stream` | HTTP: JSON in, raw PCM stream | Primary TTS |
| Batch TTS | Matcha-TTS + Vocos zh+en | `POST /tts` | HTTP: JSON in, WAV out | |
| Offline ASR | SenseVoice zh+en+ja+ko+yue | `POST /asr` | HTTP: WAV upload, JSON out | Fallback |

## API Reference

### Streaming ASR (WebSocket)

```text
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
  -d '{"text": "Hello world", "sid": 3, "speed": 1.2}' \
  --output output.wav
```

Parameters: `text` (required), `sid` (speaker ID, default 3), `speed` (rate, default 1.0)

### TTS Streaming (HTTP)

Returns raw PCM: first 4 bytes = sample rate (uint32 LE), then int16 samples.

```text
POST /tts/stream
Content-Type: application/json
```

### Health Check

```text
GET /health  →  {"asr": bool, "tts": bool, "streaming_asr": bool}
```

## Performance

### Benchmarks (Jetson Orin NX 16GB, CUDA 12.6, MAXN mode)

| Metric | Value |
|--------|-------|
| Paraformer TTFT | ~50ms |
| Paraformer finalize | ~45ms |
| Paraformer accuracy | 80.8% (26 synthetic sentences) |
| Matcha TTS TTFT | ~60ms (short text) |
| Matcha TTS latency | ~150ms (typical Chinese sentence) |

### TTS Model Comparison

We evaluated 4 TTS models for TTFT (time-to-first-audio-chunk). Matcha-TTS was selected for best Chinese quality and lowest latency:

| Model | TTFT (short) | TTFT (long) | Chinese Quality | Selected |
|-------|-------------|-------------|-----------------|----------|
| **Matcha-TTS + Vocos** | ~60ms | ~150ms | Good | **Yes** |
| Kokoro v1.1 | ~130ms | ~300ms | Fair | No (Chinese quality) |
| CosyVoice3 | ~800ms | ~2s | Excellent | No (latency) |
| F5-TTS | ~2.5s | ~5s | Excellent | No (latency) |

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
| `TTS_PROVIDER` | `cuda` | ONNX execution provider |
| `TTS_DEFAULT_SID` | `3` | Default TTS speaker ID |
| `TTS_NUM_THREADS` | `4` | TTS inference threads |
| `SENSEVOICE_LANGUAGE` | `auto` | SenseVoice language hint |
| `STREAMING_ASR_PROVIDER` | `cuda` | Paraformer execution provider |
| `MODEL_DIR` | `/opt/models` | Model storage directory |

Copy `.env.example` to `.env` to customize.

### Models

Auto-downloaded on first start via `scripts/download_models.sh`:

| Model | Size | Purpose |
|-------|------|---------|
| Paraformer streaming zh-en | ~230 MB | Streaming ASR (bilingual) |
| Matcha-TTS + Vocos zh-en | ~125 MB | TTS synthesis |
| SenseVoice zh-en-ja-ko-yue | ~500 MB | Offline ASR (5 languages) |

## Patched sherpa-onnx

Includes a patched sherpa-onnx that fixes Paraformer streaming tail truncation (stock version drops the last 1-3 characters). The patch:

1. **IsReady()** — forces decode of remaining frames after `InputFinished()`
2. **DecodeStream()** — zero-pads partial final chunks
3. **CIF force-fire** — emits residual tokens at end-of-stream

Pre-built `.so` files in `patches/sherpa-onnx-lib/` (aarch64, Python 3.10, CUDA 12.6).
See `patches/README.md` for rebuild instructions.

## Requirements

- Jetson Orin NX 16GB (JetPack 6.2, CUDA 12.6) — or any CUDA-capable device with Docker
- Docker with `nvidia` runtime
- ~5 GB disk for models

## Project Structure

```text
jetson-local-voice/
├── app/                     # FastAPI service
│   ├── main.py              # Endpoints and startup
│   ├── asr_service.py       # SenseVoice offline ASR
│   ├── streaming_asr_service.py  # Paraformer streaming ASR
│   ├── tts_service.py       # Matcha TTS (batch + streaming)
│   └── vc_service.py        # Voice conversion (experimental)
├── benchmarks/              # TTS model TTFT comparisons
│   ├── test_f5tts_ttft.py   # F5-TTS vs Kokoro TTFT
│   ├── test_matcha_ttft.py  # Matcha vs Kokoro TTFT
│   ├── test_cosyvoice3_ttft.py  # CosyVoice3 per-stage timing
│   └── archive/             # Detailed F5-TTS optimization experiments
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
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) — fast flow-matching TTS
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — multilingual offline ASR
