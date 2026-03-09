# Jetson Speech Service

GPU-accelerated ASR + TTS service for Jetson Orin NX, using sherpa-onnx with CUDA.

## Services

| Service | Model | Endpoint | Description |
|---------|-------|----------|-------------|
| **Streaming ASR** | Paraformer bilingual zh+en | `WS /asr/stream` | Real-time streaming recognition |
| **Offline ASR** | SenseVoice zh+en+ja+ko+yue | `POST /asr` | Batch transcription (lazy-loaded) |
| **TTS** | Kokoro v1.1 multilingual | `POST /tts` | Text-to-speech synthesis |
| **TTS Stream** | Kokoro v1.1 multilingual | `POST /tts/stream` | Streaming PCM output |

## Requirements

- Jetson Orin NX 16GB (JetPack 6.2, CUDA 12.6)
- Docker with `nvidia` runtime
- ~5 GB disk for models (auto-downloaded on first run)

## Quick Start

```bash
# Build
docker compose build

# Run
docker compose up -d

# Check health
curl http://localhost:8000/health
# {"asr":true,"tts":true,"streaming_asr":true}
```

## API Reference

### Streaming ASR (WebSocket)

Primary ASR backend with ~50ms first-token latency.

```
WS /asr/stream?sample_rate=16000&language=auto
```

**Protocol:**
- Client sends: raw **int16 PCM bytes** (audio chunks, e.g. 100ms each)
- Client sends: **empty bytes** `b""` to signal end of audio
- Server sends: JSON `{"text": "...", "is_final": bool, "is_stable": bool}`

```python
import asyncio, websockets, numpy as np

async def transcribe():
    async with websockets.connect(
        "ws://localhost:8000/asr/stream?sample_rate=16000"
    ) as ws:
        # Send audio chunks as they arrive
        for chunk in audio_chunks:  # np.int16 arrays
            await ws.send(chunk.tobytes())
            msg = await ws.recv()
            print(msg)  # partial results

        # Signal end
        await ws.send(b"")
        final = await ws.recv()  # {"text": "...", "is_final": true}
```

### Offline ASR (HTTP)

Batch transcription. Model is lazy-loaded on first request.

```
POST /asr
Content-Type: multipart/form-data
```

```bash
curl -X POST http://localhost:8000/asr \
  -F "file=@recording.wav" \
  -F "language=auto"
# {"text": "transcribed text"}
```

### TTS (HTTP)

```
POST /tts
Content-Type: application/json
```

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "sid": 3, "speed": 1.2}' \
  --output output.wav
```

**Parameters:**
- `text` (required): Text to synthesize
- `sid`: Speaker ID (default: 3 = zf_001 Chinese female)
- `speed`: Speech rate (default: 1.0)

### TTS Streaming (HTTP)

Returns raw PCM: first 4 bytes = sample rate (uint32 LE), then int16 PCM.

```
POST /tts/stream
Content-Type: application/json
```

### Health Check

```
GET /health
```

Returns `{"asr": bool, "tts": bool, "streaming_asr": bool}`.
`asr` reflects whether SenseVoice has been loaded (lazy — false until first `/asr` call).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_PROVIDER` | `cuda` | ONNX execution provider |
| `TTS_DEFAULT_SID` | `3` | Default TTS speaker ID |
| `TTS_NUM_THREADS` | `4` | TTS inference threads |
| `SENSEVOICE_LANGUAGE` | `auto` | SenseVoice language hint |
| `STREAMING_ASR_PROVIDER` | `cuda` | Paraformer execution provider |
| `MODEL_DIR` | `/opt/models` | Model storage directory |

## Models

Models are auto-downloaded on first start via `scripts/download_models.sh`:

| Model | Size | Path |
|-------|------|------|
| SenseVoice zh-en-ja-ko-yue | ~500 MB | `/opt/models/sensevoice/` |
| Paraformer streaming zh-en | ~230 MB | `/opt/models/paraformer-streaming/` |
| Kokoro v1.1 multilingual | ~800 MB | `/opt/models/kokoro-multi-lang-v1_1/` |

## Patched sherpa-onnx

This deployment includes a patched sherpa-onnx that fixes Paraformer streaming
tail truncation (last 1-3 characters dropped). The patch modifies three areas
of `online-recognizer-paraformer-impl.h`:

1. `IsReady()` — forces decode of remaining frames after `InputFinished()`
2. `DecodeStream()` — zero-pads partial final chunks
3. CIF force-fire — emits residual tokens at end-of-stream

Pre-built `.so` files are in `patches/sherpa-onnx-lib/` (not in git).
See `patches/README.md` for rebuild instructions.

## Performance

Measured on Jetson Orin NX 16GB with CUDA 12.6:

| Metric | Value |
|--------|-------|
| ASR first-token latency | ~50ms |
| ASR finalize latency | ~45ms |
| ASR accuracy (patched) | 80.8% (26 TTS-synthesized sentences) |
| TTS RTF | 0.133 (7.5x realtime) |
| TTS latency | ~1s for a Chinese sentence |

## Client Integration

In `clawd.yaml`:

```yaml
stt:
  backend: paraformer-streaming
  speech_service_url: http://<jetson-ip>:8000

tts:
  backend: kokoro
  speech_service_url: http://<jetson-ip>:8000
```

Or via CLI:

```bash
clawd-reachy --stt paraformer-streaming --tts kokoro \
  --speech-url http://<jetson-ip>:8000
```
