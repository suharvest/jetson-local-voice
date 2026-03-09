# Jetson Voice

Turn a Jetson into a local voice device — GPU-accelerated ASR + TTS, zero cloud dependency.

## Why Jetson?

Edge voice pipelines need sub-second latency. Cloud TTS/ASR adds 200-500ms network round-trip before the first audio byte. Jetson Orin NX with CUDA runs the full voice stack locally:

| Stage | Model | Latency | Note |
|-------|-------|---------|------|
| **ASR (streaming)** | Paraformer zh+en | ~50ms TTFT | Real-time, partial results via WebSocket |
| **ASR (offline)** | SenseVoice 5-lang | ~200ms | Batch mode, lazy-loaded |
| **TTS** | Kokoro v1.1 | ~130ms TTFT | 7.5x realtime, streaming PCM |

ASR + TTS combined: **~180ms** (ASR finalize + TTS first chunk). Full voice-to-voice latency depends on LLM inference time (not included here).

## Quick Start

```bash
# On Jetson (JetPack 6.2)
git clone <this-repo>
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
│  ├── POST /tts          Kokoro batch TTS         │
│  └── POST /tts/stream   Kokoro streaming TTS     │
│                                                  │
│  sherpa-onnx + ONNX Runtime 1.20 (CUDA)          │
└──────────────────────────────────────────────────┘
         ▲ HTTP/WebSocket
         │
   Any client (SBC, laptop, robot, ...)
```

The service is model-agnostic at the API level — clients send audio/text, get audio/text back. Swap models without changing client code.

## Services

| Service | Model | Endpoint | Protocol |
|---------|-------|----------|----------|
| Streaming ASR | Paraformer bilingual zh+en | `WS /asr/stream` | WebSocket: int16 PCM in, JSON out |
| Offline ASR | SenseVoice zh+en+ja+ko+yue | `POST /asr` | HTTP: WAV upload, JSON out |
| TTS | Kokoro v1.1 multilingual | `POST /tts` | HTTP: JSON in, WAV out |
| TTS Streaming | Kokoro v1.1 multilingual | `POST /tts/stream` | HTTP: JSON in, raw PCM stream |

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
  -d '{"text": "Hello world", "sid": 3, "speed": 1.2}' \
  --output output.wav
```

Parameters: `text` (required), `sid` (speaker ID, default 3), `speed` (rate, default 1.0)

### TTS Streaming (HTTP)

Returns raw PCM: first 4 bytes = sample rate (uint32 LE), then int16 samples.

```
POST /tts/stream
Content-Type: application/json
```

### Health Check

```
GET /health  →  {"asr": bool, "tts": bool, "streaming_asr": bool}
```

## Performance

### Benchmarks (Jetson Orin NX 16GB, CUDA 12.6, MAXN mode)

| Metric | Value |
|--------|-------|
| Paraformer TTFT | ~50ms |
| Paraformer finalize | ~45ms |
| Paraformer accuracy | 80.8% (26 synthetic sentences) |
| Kokoro TTS RTF | 0.133 (7.5x realtime) |
| Kokoro TTS TTFT | ~130ms (short text) |
| Kokoro TTS latency | ~1s (typical Chinese sentence) |

### TTS Model Comparison

We evaluated 4 TTS models for TTFT (time-to-first-audio-chunk). Kokoro was selected for best latency-quality tradeoff:

| Model | TTFT (short) | TTFT (long) | Quality | Selected |
|-------|-------------|-------------|---------|----------|
| **Kokoro v1.1** | ~130ms | ~300ms | Good | Yes |
| Matcha-TTS | ~60ms | ~150ms | Fair | No (quality) |
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
| SenseVoice zh-en-ja-ko-yue | ~500 MB | Offline ASR (5 languages) |
| Paraformer streaming zh-en | ~230 MB | Streaming ASR (bilingual) |
| Kokoro v1.1 multilingual | ~800 MB | TTS synthesis |

## Patched sherpa-onnx

Includes a patched sherpa-onnx that fixes Paraformer streaming tail truncation (stock version drops the last 1-3 characters). The patch:

1. **IsReady()** — forces decode of remaining frames after `InputFinished()`
2. **DecodeStream()** — zero-pads partial final chunks
3. **CIF force-fire** — emits residual tokens at end-of-stream

Pre-built `.so` files in `patches/sherpa-onnx-lib/` (aarch64, Python 3.10, CUDA 12.6).
See `patches/README.md` for rebuild instructions.

## Requirements

- Jetson Orin NX 16GB (JetPack 6.2, CUDA 12.6)
- Docker with `nvidia` runtime
- ~5 GB disk for models

## Project Structure

```
jetson-voice/
├── app/                     # FastAPI service
│   ├── main.py              # Endpoints and startup
│   ├── asr_service.py       # SenseVoice offline ASR
│   ├── streaming_asr_service.py  # Paraformer streaming ASR
│   ├── tts_service.py       # Kokoro TTS (batch + streaming)
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
