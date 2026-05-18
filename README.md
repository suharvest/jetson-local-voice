# OpenVoiceStream

**Native-engine streaming ASR + TTS for edge dialogue.** One container, stable HTTP/WebSocket APIs, and validated paths across Jetson, Rockchip, and Raspberry Pi ecosystems.

<p align="center">
  <a href="https://github.com/suharvest/openvoicestream"><img src="https://img.shields.io/github/stars/suharvest/openvoicestream?style=social" alt="GitHub stars" /></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/ASR-Paraformer%20%7C%20Qwen3--ASR%20%7C%20SenseVoice-2f80ed.svg" alt="ASR: Paraformer, Qwen3-ASR, SenseVoice" /></a>
  <a href="#tts-model-comparison"><img src="https://img.shields.io/badge/TTS-Matcha--TTS%20%7C%20Qwen3--TTS%20%7C%20Kokoro-f97316.svg" alt="TTS: Matcha-TTS, Qwen3-TTS, Kokoro" /></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/engines-TensorRT--EdgeLLM%20%7C%20RKNN%20%7C%20sherpa--onnx-16a34a.svg" alt="Engines: TensorRT-EdgeLLM, RKNN, sherpa-onnx" /></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/deploy-Docker-2563eb.svg" alt="Deploy with Docker" /></a>
  <a href="#supported-devices"><img src="https://img.shields.io/badge/ecosystems-Jetson%20%7C%20Rockchip%20%7C%20Raspberry%20Pi-65a30d.svg" alt="Supported ecosystems: Jetson, Rockchip, Raspberry Pi" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-facc15.svg" alt="MIT license" /></a>
</p>

<p align="center">
  <img src="docs/media/hero.png" alt="OpenVoiceStream - streaming ASR and TTS for edge dialogue" width="760" />
</p>

OpenVoiceStream is a local voice stack for products that need real-time ASR and TTS on edge hardware. It runs fully on-device, avoids heavyweight ML frameworks in the hot path, and keeps the client API stable while you switch between sherpa-onnx, TensorRT-EdgeLLM, RKNN, and CPU ONNX backends.

## Why This Matters

OpenVoiceStream is meant to make local voice practical at product scale: start
with low-cost real-time voice I/O, then move up to human-like speech or a fully
local voice + LLM loop without changing the client API.

<p align="center">
  <img src="docs/media/solution-lineup.png" alt="OpenVoiceStream solution lineup: recommended hardware paths for real-time voice I/O, production edge voice, human-like local speech, and voice plus local LLM" width="900" />
</p>

Board prices vary by region and kit contents. The point is the order of
magnitude: simple Raspberry Pi-class boards can handle real-time voice input and
output, while Jetson-class edge AI boards can run expressive speech and local LLM
dialogue without a per-call speech API bill.

## Quick Start

Clone once on the target device. The installer validates the host, selects the
right compose file, pulls the image, starts the service, and can run health,
capability, TTS smoke, and TTS-to-ASR round-trip checks.

```bash
git clone --recurse-submodules https://github.com/suharvest/openvoicestream.git
cd openvoicestream

# Auto-detect Jetson, Rockchip, or Raspberry Pi.
deploy/install.sh --pull --verify
```

Choose explicitly when auto-detect is not enough:

```bash
deploy/install.sh --target jetson --pull --verify
deploy/install.sh --target rk3588 --pull --verify
deploy/install.sh --target rk3576 --pull --verify
deploy/install.sh --target rpi --pull --verify
```

After startup, the service listens on `http://device:8621`:

| Target | URL | Compose file | Image |
|---|---|---|---|
| Jetson | `http://device:8621` | `deploy/docker-compose.yml` | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.12-highperf` |
| RK3576 | `http://device:8621` | `deploy/docker-compose.rk.yml` | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-v1.4-closedloop` |
| RK3588 | `http://device:8621` | `deploy/docker-compose.radxa.yml` | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-v1.4-closedloop` |
| Raspberry Pi | `http://device:8621` | `deploy/docker-compose.rpi.yml` | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rpi-v1.0-onnx` |

The published Docker images currently keep the previous registry namespace so
existing deployments can pull the same artifacts during the rename.

Manual verification:

```bash
# Same default URL on Jetson, RK3576, RK3588, and Raspberry Pi.
deploy/verify.sh --url http://device:8621 --tts-smoke --roundtrip
curl http://device:8621/health
```

Client examples live in [`examples/`](examples/):

```bash
python3 examples/stream_tts_to_wav.py \
  --url http://device:8621 \
  --text "你好，欢迎使用 OpenVoiceStream。" \
  --out /tmp/ovs-tts.wav
```

**Deploy with compose** when you want to manage profiles yourself:

```bash
# Chinese + English on Jetson, using the lightweight Paraformer + Matcha path.
docker compose -f deploy/docker-compose.yml up -d

# English only on Jetson.
LANGUAGE_MODE=en docker compose -f deploy/docker-compose.yml up -d

# Qwen3 multilingual ASR/TTS on Jetson Orin NX.
OVS_PROFILE=jetson-multilang-highperf-nx \
docker compose -f deploy/docker-compose.yml up -d
```

`deploy/install.sh --pull --verify` auto-detects Jetson/RK/RPi when run on the
target device. The Jetson default stays on the lightweight `zh_en` path because
it is the fastest path to reproduce. Set a `jetson-multilang-*` profile when
you want the Qwen3 TensorRT-EdgeLLM route.

## Table of Contents

- [Why This Matters](#why-this-matters)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Qwen3 Multilingual Path](#qwen3-multilingual-path)
- [Performance](#performance)
- [Configuration](#configuration)
- [Models](#models)
- [Supported Devices](#supported-devices)
- [Patched sherpa-onnx](#patched-sherpa-onnx)
- [Project Structure](#project-structure)
- [Changelog](#changelog)
- [Acknowledgements](#acknowledgements)

## Key Features

- **Streaming-first API** — WebSocket ASR with partial/final results and HTTP streaming TTS with sentence-level audio chunks.
- **Native engine runtime** — TensorRT-EdgeLLM on Jetson, RKNN/RKLLM on Rockchip, sherpa-onnx and ONNX Runtime on CPU/CUDA paths.
- **Stable backend contract** — clients keep the same `/asr/stream`, `/tts`, `/tts/stream`, and `/health` calls when profiles change.
- **Measured low latency** — 58 ms EOS-to-first-audio on Jetson Orin NX with Paraformer + Matcha; 157 ms with Qwen3 ASR/TTS voice clone.
- **Multilingual options** — Chinese+English, English-only, and 52-language Qwen3 paths are exposed through the same service.
- **Container-first deploy** — prebuilt images, target-specific compose files, host checks, model downloads, and verification scripts are included.
- **LLM-ready agent layer** — `agent/` streams ASR results into an OpenAI-compatible or EdgeLLM backend, then streams LLM tokens directly back to TTS.
- **Fully local economics** — no speech API key, no per-call ASR/TTS bill, no runtime internet dependency after artifacts are cached, and no PyTorch/Transformers in the voice hot path.

## Architecture

```text
┌───────────────────────────────────────────────────────────┐
│  Edge device (Jetson Orin / RK3576 / RK3588 / RPi 4–5)    │
│                                                           │
│  FastAPI service (container :8000; host default :8621)     │
│  ├── WS /asr/stream    Streaming ASR                      │
│  │     └─ zh_en: Paraformer  │  en: Zipformer  │  multi: Qwen3-ASR │
│  ├── POST /asr          SenseVoice offline ASR (zh+en)    │
│  ├── POST /tts          Batch TTS                         │
│  └── POST /tts/stream   Streaming TTS                     │
│        └─ zh_en: Matcha-TTS  │  en: Kokoro v1.0  │  multi: Qwen3-TTS │
│                                                           │
│  Inference: sherpa-onnx · TRT-EdgeLLM · RKNN              │
└───────────────────────────────────────────────────────────┘
         ▲ HTTP / WebSocket
         │
   Any client (SBC, laptop, robot, kiosk, ...)
```

Models are selected automatically based on `LANGUAGE_MODE`:

| Service | Endpoint | zh_en (default) | en | multilingual | Protocol |
|---------|----------|-----------------|-----|---------------|----------|
| **Streaming ASR** | `WS /asr/stream` | Paraformer bilingual | Zipformer English | Qwen3-ASR (52 langs) | WebSocket: int16 PCM in, JSON out |
| **Streaming TTS** | `POST /tts/stream` | Matcha-TTS + Vocos | Kokoro v1.0 | Qwen3-TTS (voice clone) | HTTP: JSON in, raw PCM stream |
| **Batch TTS** | `POST /tts` | Matcha-TTS + Vocos | Kokoro v1.0 | Qwen3-TTS (voice clone) | HTTP: JSON in, WAV out |
| Offline ASR | `POST /asr` | SenseVoice (zh+en+ja+ko+yue) | SenseVoice (same) | Qwen3-ASR (52 langs) | HTTP: WAV upload, JSON out |

**Backend capabilities differ:**

| Backend | Speed control | Pitch shift | Voice clone | Languages | Streaming |
|---------|--------------|-------------|-------------|-----------|-----------|
| Sherpa (zh_en/en) | ✅ | ✅ | ❌ | 2 (zh+en) | ✅ |
| Qwen3 (multilingual) | ❌ | ❌ | ✅ (x-vector) | 52 | ✅ |
| RKNN (Rockchip) | ✅ | ✅ | ❌ | 2 (zh+en) | ✅ |

The service is model-agnostic at the API level — clients send audio/text, get audio/text back. Swap engines without changing client code. Unsupported parameters return `501` with `{"required_capability": "..."}`.

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
    async with websockets.connect("ws://device:8621/asr/stream?sample_rate=16000") as ws:
        for chunk in audio_chunks:  # np.int16 arrays
            await ws.send(chunk.tobytes())
            result = await ws.recv()  # partial results
        await ws.send(b"")  # signal end
        final = await ws.recv()  # {"text": "...", "is_final": true}
```

### Offline ASR (HTTP)

```bash
curl -X POST http://device:8621/asr \
  -F "file=@recording.wav" -F "language=auto"
# {"text": "transcribed text"}
```

### TTS (HTTP)

```bash
curl -X POST http://device:8621/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "sid": 52, "speed": 1.0}' \
  --output output.wav
```

Parameters: `text` (required), `sid` (speaker ID, default 52), `speed` (rate, default 1.0)

**Note:** `speed` works only on backends that advertise speed control
(Sherpa/Matcha/RKNN). Qwen3-TTS (`multilanguage` profiles) does not currently
support reliable speed or pitch adjustment, so clients should treat those
parameters as unsupported on Qwen3.

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

## Qwen3 Multilingual Path

`OVS_PROFILE=jetson-multilang-highperf*` enables Qwen3-ASR + Qwen3-TTS — 52 languages plus voice cloning. The integration code lives in this repo; Qwen-specific export, engine builds, and worker glue are maintained in the standalone companion repo [`suharvest/qwen3-edgellm-jetson`](https://github.com/suharvest/qwen3-edgellm-jetson) (pinned here as a submodule at `third_party/qwen3-edgellm-jetson/`). Large model artifacts live in [`harvestsu/qwen3-edgellm-jetson-artifacts`](https://huggingface.co/harvestsu/qwen3-edgellm-jetson-artifacts) on Hugging Face.

**Quickest path on a fresh Orin NX:**

```bash
git clone https://github.com/suharvest/qwen3-edgellm-jetson.git
bash qwen3-edgellm-jetson/scripts/reproduce_qwen3_highperf.sh \
  --reference /path/to/24kHz_mono.wav   # optional: gates the voice-clone path
```

The orchestrator builds the runtime, downloads + SHA-256-verifies the HF artifacts, builds the slim docker image, starts the service, and runs the verifier (`scripts/verify_reproduction.sh`). Exit 0 means the slim container on port 18092 is healthy and serving the validated stack.

**Two runtime profiles** under the same API surface:

| Profile | Goal | Default behavior |
|---------|------|------------------|
| `official` | Minimal-diff EdgeLLM example. Close enough to upstream that it can be reviewed or upstreamed as a Qwen3 ASR/TTS example. | Semantic/correctness fixes only — tokenizer layout, sampling, runtime contract, stream callback. Regular exported Talker/CodePredictor/Code2Wav directories. |
| `highperf` (default) | Product low-latency dual-resident path for Orin. | Full vocab, ASR FP8 embedding, TTS W8A16 Talker, CP BF16 I/O + `lm_head` pretranspose, stateful Code2Wav, CP decode CUDA graph, `ACTIVE_CP_GROUPS=13`. |

Use `jetson-multilang-highperf-nx` on Orin NX when consuming the NX-native engine set; the default `jetson-multilang-highperf` profile targets the Nano artifact set. Profiles in [`configs/profiles`](configs/profiles) set env defaults only; explicit env vars still override them.

For detailed branch ownership, engine env vars, frozen-baseline numbers, and artifact handling, see [`docs/plans/qwen3-current-frozen-baseline-2026-05-10.md`](docs/plans/qwen3-current-frozen-baseline-2026-05-10.md).

Current release status, image digests, artifact repositories, and known gaps are
tracked in [`docs/productization-status.md`](docs/productization-status.md).

## Performance

### Cross-Device Benchmarks (measured 2026-05-18)

Jetson/RPi rows are from the original local forced-EOS gate against
`http://127.0.0.1:8621`. RK rows were rerun after the true-streaming fix with
`QWEN3_ASR_CHUNK_CONFIRM=0`, `--eos vad`, and `--vad-silence-ms 800`; their V2V
column is split `/asr/stream` plus `/tts/stream`.

| Target / profile | Image | TTS backend | ASR backend | TTS RTF p50 | ASR fRTF p50 | ASR CER p50 | V2V EOS→audio p50 |
|---|---|---|---|---:|---:|---:|---:|
| Orin Nano `jetson-multilang-highperf` | `jetson-v1.12-highperf` | `trt_edgellm` | `trt_edgellm` | 0.470 | 0.076 | 5.3% | 251 ms |
| Orin NX `jetson-multilang-highperf-nx` | `jetson-v1.12-highperf` | `trt_edgellm` | `trt_edgellm` | 0.417 | 0.042 | 5.3% | 157 ms |
| Orin Nano `jetson-qwen3asr-matcha` | `jetson-v1.12-highperf` | `matcha_trt` | `trt_edgellm` | 0.024 | 0.075 | 5.3% | 286 ms |
| Orin NX `jetson-qwen3asr-matcha-nx` | `jetson-v1.12-highperf` | `matcha_trt` | `trt_edgellm` | 0.018 | 0.042 | 5.3% | 162 ms |
| Orin Nano `jetson-zh-en` | `jetson-v1.12-highperf` | `matcha_trt` | `paraformer_trt` | 0.023 | 0.077 | 13.3% | 327 ms |
| Orin NX `jetson-zh-en` | `jetson-v1.12-highperf` | `matcha_trt` | `paraformer_trt` | 0.018 | 0.015 | 10.5% | 58 ms |
| RK3588 `rk3588-default` | `rk-v1.4-closedloop` | `rk:matcha_rknn` | `rk:qwen3_asr_rk` | 0.124 | 0.318 | 60.7% | 394 ms |
| RK3576 `rk3576-default` | `rk-v1.4-closedloop` | `rk:matcha_rknn` | `rk:qwen3_asr_rk` | 0.290 | 0.265 | 63.2% | 1099 ms |
| Raspberry Pi 5 `rpi5-default` | `rpi-v1.0-onnx` | `sherpa` | `sherpa_asr` | 0.078 | 0.000 | 20.0% | 3 ms |

The previous RK rows were stale forced-EOS/chunk-confirm results. In the fixed
rerun, standalone Matcha TTS first audio is 51 ms p50 on RK3588 and 65 ms p50
on RK3576; split V2V is now ASR-finalize-bound at 394 ms / 1099 ms p50. The
real `/v2v/stream` path remains about 1.27 s p50 with the current 800 ms VAD
hangover.

Deployment footprint from the same run:

| Target | Image size | Model / engine volume | Resident memory | Startup to ready |
|---|---:|---:|---:|---:|
| Orin Nano | 2.02 GB | 5.14 GB | 2.14 GiB | 14 s |
| Orin NX | 2.02 GB | 5.45 GB | 1.02 GiB | 13 s |
| RK3588 | 767 MB | 3.31 GB ASR + 301 MB TTS | 4.09 GiB | 9 s |
| RK3576 | 767 MB | 2.21 GB ASR + 351 MB TTS | 2.71 GiB | 15 s |
| Raspberry Pi 5 | 568 MB | 2.19 GB | n/a from Docker stats | 9 s |

Concurrency smoke (`parallel=2`, `asr_tts_simul`) passed on Jetson Nano/NX
Paraformer+Matcha, RK3588, RK3576, and Raspberry Pi 5. Jetson p=2 is
functional but TTS becomes throughput-bound (RTF ~1.3-1.4), so use Orin NX or a
Qwen3 ASR + Matcha split when low-latency concurrent dialogue matters. Full raw
JSON paths and methodology are in
[`docs/benchmarks/streaming-release-gate-2026-05-18.md`](docs/benchmarks/streaming-release-gate-2026-05-18.md).

### TTS Model Comparison

The current release uses Matcha/Vocos for the bilingual path, Kokoro for
English-only deployments, and Qwen3-TTS when voice cloning or 52-language TTS is
required. The RTF numbers below are from the 2026-05-18 benchmark run where
available; the unused research models are kept as historical context.

| Model | Current role | Streaming RTF p50 | First audio p50 | Notes |
|-------|--------------|------------------:|----------------:|-------|
| **Matcha-TTS + Vocos** | Default bilingual TTS | 0.018 on Orin NX, 0.075 on RK3588, 0.078 on RPi5 | 2.6-7.5 ms | Fastest practical TTS path; no voice clone. |
| **Qwen3-TTS** | Multilingual voice clone | 0.417 on Orin NX, 0.470 on Orin Nano | 4.4-7.3 ms | Higher quality/features, much heavier than Matcha. |
| **Kokoro v1.0** | English-only TTS | Not in this benchmark run | Historical ~130 ms TTFT | Kept for English-only deployments. |
| CosyVoice3 | Research only | Not shipped | Historical ~800 ms TTFT | Higher quality, too heavy for this release. |
| F5-TTS | Research only | Not shipped | Historical ~2.5 s TTFT | Not suitable for low-latency edge dialogue. |

Current streaming benchmark scripts live in `bench/perf/`.

### Performance Tuning

Run once after boot on Jetson to lock clocks to max:

```bash
sudo ./scripts/setup-performance.sh
```

This sets MAXN power mode, locks CPU/GPU clocks, and disables dynamic frequency scaling. Critical for consistent inference latency.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OVS_PROFILE` | unset | Preferred OpenVoiceStream profile selector, e.g. `jetson-zh-en`, `jetson-multilang-highperf-nx`, `rk3588-default`, `rpi5-default` |
| `LANGUAGE_MODE` | `zh_en` | `zh_en` (Chinese+English), `en` (English only), or `multilanguage` (Qwen3, 52 langs; profiles usually set this for you) |
| `TTS_PROVIDER` | `cuda` | ONNX execution provider |
| `TTS_DEFAULT_SID` | `52` | Default TTS speaker ID (52=af_cute, 3=af_heart) — Sherpa only |
| `TTS_DEFAULT_SPEED` | `1.0` | TTS playback speed for backends that support it; Qwen3-TTS does not |
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

Measured Docker volume sizes in the current release are larger than individual
model tarballs because they include compiled engines and profile-specific
artifacts: 5.14-5.45 GB on Jetson, 2.56-3.61 GB on RK, and 2.19 GB on
Raspberry Pi 5.

## Supported Devices

OpenVoiceStream is validated on the following hardware. Any device in the same class should work; these are the ones we measure against.

| Device class | Validated on | Notes |
|---|---|---|
| **NVIDIA Jetson Orin** | Jetson Orin Nano 8GB, Orin NX 16GB, AGX Orin | CUDA 12.6 / JetPack 6.2. Full feature set including Qwen3 multilingual + voice clone. |
| **Rockchip NPU** | Radxa ROCK 5T (RK3588), Banana Pi BPI-M5 Pro (RK3576) | RKNN runtime. Qwen3-ASR works; release TTS uses the validated hybrid Matcha path. |
| **Raspberry Pi (CPU)** | Raspberry Pi 5 8GB, Raspberry Pi 4 4GB | CPU inference. Lowest BOM (~$80). Real-time zh+en commands. |

Requirements: Docker plus enough disk for the image and model volume. Current
measured footprints are about 7.5 GB total for Jetson, 3.2-4.4 GB for RK, and
2.8 GB for Raspberry Pi 5. Runtime memory depends on the profile: about 1.0-2.1
GiB on Jetson, 2.7-4.1 GiB on RK, and CPU-only on Raspberry Pi. On Jetson,
NVIDIA Container Runtime is required; on Rockchip, the host NPU driver
(`rknpu`) must be loaded.

## Patched sherpa-onnx

OpenVoiceStream ships a patched sherpa-onnx that fixes Paraformer streaming tail truncation (the stock version drops the last 1–3 characters). The patch:

1. **IsReady()** — forces decode of remaining frames after `InputFinished()`
2. **DecodeStream()** — zero-pads partial final chunks
3. **CIF force-fire** — emits residual tokens at end-of-stream

Pre-built `.so` files live in `patches/sherpa-onnx-lib/` (aarch64, Python 3.10, CUDA 12.6). See `patches/README.md` for rebuild instructions.

## Project Structure

```text
openvoicestream/
├── app/                     # FastAPI service
│   ├── main.py              # Endpoints and startup
│   ├── backends/            # Per-engine backends (sherpa / jetson / rk)
│   ├── core/                # VAD, ASR/TTS contracts, streaming primitives
│   └── model_downloader.py  # On-demand model download + voice patching
├── agent/                   # LLM voice agent (session, plugin system, audio I/O)
├── voices/                  # Custom voice embeddings (auto-patched into model)
├── bench/                   # Streaming + V2V latency benchmarks (perf harness)
├── patches/                 # Paraformer EOF truncation fix
├── scripts/                 # Engine build, model download, diagnostics
├── examples/                # API usage examples (TTS streaming, V2V client)
├── tests/                   # Integration and E2E tests
├── deploy/
│   ├── docker-compose.yml   # Production deploy (pre-built image)
│   ├── artifacts/           # Deployment manifests
│   └── docker/
│       ├── Dockerfile.jetson  # Jetson Orin Nano/NX/AGX (zh_en or multilingual)
│       ├── Dockerfile.rk      # Rockchip RK3576/RK3588 NPU
│       └── Dockerfile.rpi     # Raspberry Pi 4/5 (CPU)
├── configs/                 # Device profiles (Jetson, RK, RPi)
├── third_party/             # Submodules (independently maintained)
│   ├── qwen3-edgellm-jetson # Qwen3 export + engine build for Jetson
│   └── rkvoice-stream       # Rockchip NPU streaming voice runtime
└── docs/                    # Guides, runbooks, comparison reports
```

Clone with `--recurse-submodules` to pull `third_party/*`, or run `git submodule update --init --recursive` after cloning.

## Changelog

### Current Container Release

- **Jetson highperf image** — `jetson-v1.12-highperf`, 2.02 GB, with host CUDA/TensorRT libraries mounted from JetPack and models/engines cached in `speech-models`.
- **RK release image** — `rk-v1.4-closedloop`, 767 MB, with runtime-pinned RKNN dependencies and validated hybrid Matcha TTS.
- **Raspberry Pi image** — `rpi-v1.0-onnx`, 568 MB, CPU-only ONNX path.

See the 2026-05-18 benchmark report for image size, model volume,
resident memory, startup time, and concurrency results.

### v2.2

- **Endpoint detection** — server proactively sends `is_final` when the speaker pauses (0.6s trailing silence), reducing response latency
- **Fix WebSocket lifecycle** — properly close connections after finalize to prevent stale connection reuse
- **Production deploy compose** — `deploy/docker-compose.yml` with pre-built image (no build step needed)

### v2.1

- Streaming TTS with sentence-level callback
- Custom voice embedding support via pitch shift

### v2.0

- Initial release: Paraformer + Matcha (zh_en), Zipformer + Kokoro (en)
- Patched sherpa-onnx for Paraformer streaming EOF fix

## Contributing

Issues and PRs are welcome. The most useful contributions:

- New backend integrations (other NPUs, other inference engines)
- Streaming benchmarks on additional hardware
- Bug reports with reproducible audio samples and `LANGUAGE_MODE` / profile info
- Documentation improvements, especially deployment recipes for new devices

If you're working on a larger change, open an issue first to align on the approach. Sub-project changes (Qwen3 export, Rockchip runtime) belong in their own repos: [`qwen3-edgellm-jetson`](https://github.com/suharvest/qwen3-edgellm-jetson), [`rkvoice-stream`](https://github.com/suharvest/rkvoice-stream).

## Acknowledgements

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — speech inference engine powering the bilingual ASR and TTS paths
- [next-gen Kaldi](https://github.com/k2-fsa) — research foundation behind sherpa-onnx
- [Paraformer](https://github.com/modelscope/FunASR) — streaming bilingual ASR model
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) — fast flow-matching TTS (zh+en mode)
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) — high-quality English TTS with 53 speakers (en mode)
- [Zipformer](https://github.com/k2-fsa/icefall) — efficient transducer ASR (en mode)
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — multilingual offline ASR
- [Qwen3](https://huggingface.co/Qwen) — multilingual ASR + TTS foundation model (52-language path)
- [TensorRT-EdgeLLM](https://github.com/NVIDIA/TensorRT-LLM) — Jetson inference runtime for the Qwen3 path
- [RKNN Toolkit](https://github.com/rockchip-linux/rknn-toolkit2) — Rockchip NPU runtime for the RK3576/RK3588 path
