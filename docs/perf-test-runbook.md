# Seeed Local Voice — Per-Device Perf Test Runbook

A single source of truth for **how to launch the service on each device**
and **what to measure**, so you can fill in the performance table without
re-tracing setup.

Branch: `qwen3tts-accurate-20260507` (latest commit ≥ `f3ab241`).

## Image registry

| Device class | Image | Size |
|---|---|---|
| Jetson Orin (Nano/NX/AGX) | `sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.11` | 3.14 GB |
| RK3576 / RK3588 | `seeed-local-voice:rk-v1.1` (local on radxa; not yet pushed to registry) | 1.38 GB |
| RPi 4 / 5 (CM4 / CM5) | `seeed-local-voice:rpi-v1.1` (local on harvest-pi) | 560 MB |

History — Jetson image patches (each fix forced a rebake):
- v1.0 → first slim bake (CMD `sleep 9999` bug, no TRT libs)
- v1.1 → tensorrt + cuda-python + EdgeLLM workers added
- v1.2 → CMD fixed + NVIDIA_DRIVER_CAPABILITIES=all
- v1.3 → TRT runtime libs added but 1.14 GB layer rejected by registry (413)
- v1.4 → layered TRT/CUDA libs (all < 500 MB), passes registry
- v1.5 → EdgeLLM plugin replaced with W8A16-capable build
- v1.6 → engine_resolver + profile fixes added (missed full app/ refresh)
- v1.7 → full app/ + configs/ refresh, baked SEEED_LOCAL_VOICE_PROFILE blanked
- v1.8 → hf_artifacts User-Agent fix (hf-mirror 403)
- v1.9 → engine_resolver manifest key alignment (model-relative vs full path)
- v1.10 → multilang preset also pulls matcha-icefall-zh-en bundle from Seeed CDN
- v1.11 → model_downloader marker-file freshness check (catches stale dir created by engine_resolver)
Use v1.11 going forward.

All three are self-contained for their device family. Pull jetson-v1.7 from
the registry; rk-v1.1 / rpi-v1.1 are on-device builds — `docker save` or
HF upload if you need to distribute.

## Preset matrix (verified end-to-end)

| Preset | Jetson Nano | Jetson NX/AGX | RK3576 | RK3588 | RPi5/CM5 | RPi4/CM4 |
|---|---|---|---|---|---|---|
| **voice_clone**  (Qwen3 ASR + Qwen3 TTS) | ✅ | ☐ | — | — | — | — |
| **multilang**    (Qwen3 ASR + Matcha TTS) | ✅ | ☐ | ☐ | ✅ | — | — |
| **lite_zh_en**   (Paraformer + Matcha) | ☐ DL slow | ☐ DL slow | ☐ planned | ☐ planned | ✅ via rpi5-default | — |
| **asr_zh_en**    (Paraformer only) | — | — | — | — | ✅ | ✅ |

Legend: ✅ verified `/health` 200 on real hardware. ⚠️ runs but needs an
artifact fetch (HF) on first start. ☐ profile exists, smoke pending.
— means unsupported (will raise `UnsupportedPreset`).

## Containers left running for perf tests

After today's sweep:

| Device | Container | Image | Port | Preset | /health |
|---|---|---|---|---|---|
| orin-nano | `seeed-nano-v111` | jetson-v1.11 | 8000 (host net) | voice_clone | ✅ both ready |
| radxa | `seeed-rk-v11` | rk-v1.1 | 8000 (host net) | multilang | ✅ both ready |
| harvest-pi | `seeed-rpi-litezhen` | rpi-v1.1 | 8765 | rpi5-default (lite_zh_en) | ✅ both ready |
| harvest-pi | `seeed-rpi-asronly` | rpi-v1.1 | 8766 | asr_zh_en | ✅ asr ready, tts:null |

`docker ps` on each device confirms current state.

## Launch commands (per device)

### Jetson Orin Nano / NX / AGX

Requires:
- `--runtime nvidia` + `NVIDIA_DRIVER_CAPABILITIES=all` (bundled in image env)
- TRT engines + Qwen3 artifacts on host (for voice_clone / multilang)
- `HF_ENDPOINT=https://hf-mirror.com` if device has no public WAN

**voice_clone (default — best quality, voice cloning)**
```bash
docker run -d --name seeed-jetson --runtime nvidia --network host \
  -v /tmp/nano-audit:/opt/models/qwen3-edgellm:ro \
  -e SEEED_LOCAL_VOICE_PRESET=voice_clone \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e QWEN3_ARTIFACT_ROOT=/opt/models/qwen3-edgellm \
  sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.11
```

**multilang (Qwen3 ASR + Matcha TTS — supports multi-stream)**
```bash
docker run -d --name seeed-jetson-ml --runtime nvidia --network host \
  -v /tmp/nano-audit:/opt/models/qwen3-edgellm:ro \
  -v seeed-models:/opt/models \
  -e SEEED_LOCAL_VOICE_PRESET=multilang \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e QWEN3_ARTIFACT_ROOT=/opt/models/qwen3-edgellm \
  sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.11
```

Engine resolver downloads Matcha vocos engine bundle from
`harvestsu/seeed-local-voice-artifacts` on first start (~30s).

**lite_zh_en (Paraformer + Matcha — high throughput zh+en)**
```bash
docker run -d --name seeed-jetson-lite --runtime nvidia --network host \
  -v seeed-models:/opt/models \
  -e SEEED_LOCAL_VOICE_PRESET=lite_zh_en \
  -e HF_ENDPOINT=https://hf-mirror.com \
  sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.11
```

First start downloads sherpa Matcha (~120 MB) + Paraformer (~900 MB) ONNX
to volume; engine_resolver compiles or pulls TRT engines.

### RK3576 / RK3588 (radxa)

Requires:
- Host has `librknnrt.so` + `librkllmrt.so` + rkvoice-stream model bundle
- `--privileged` for NPU device access

**multilang on RK3588 (Qwen3 ASR via NPU + Matcha RKNN TTS)**
```bash
docker run -d --name seeed-rk --privileged --network host \
  -v /dev:/dev \
  -v /usr/lib/librknnrt.so:/usr/lib/librknnrt.so \
  -v /home/radxa/lib/librkllmrt.so:/opt/asr/lib/librkllmrt.so \
  -v /home/radxa/models:/opt/asr/models \
  -v /home/radxa/models/tts/matcha-icefall-zh-en:/opt/tts/models/matcha-icefall-zh-en \
  -v /home/radxa/models/tts/matcha-s64.rknn:/opt/tts/models/matcha-s64.rknn \
  -v /home/radxa/models/tts/vocos-16khz-600.rknn:/opt/tts/models/vocos-16khz-600.rknn \
  -v /home/radxa/matcha-data:/home/radxa/matcha-data:ro \
  -e SEEED_LOCAL_VOICE_PRESET=multilang \
  seeed-local-voice:rk-v1.1
```

profile_selector auto-detects rk3588 (or rk3576) via `/proc/cpuinfo` and picks `rk{soc}-multilang.json`.

### Raspberry Pi 5 / 4 / CM4 / CM5 (harvest-pi)

CPU-only (sherpa-onnx). No `--runtime nvidia`, no `--privileged`.

**asr_zh_en (RPi4/CM4 minimum: Paraformer streaming, no TTS)**
```bash
docker run -d --name seeed-rpi-asr -p 8000:8000 \
  -e SEEED_LOCAL_VOICE_PRESET=asr_zh_en \
  -v seeed-models:/opt/models \
  seeed-local-voice:rpi-v1.1
```

**lite_zh_en on RPi5/CM5 (Paraformer + Matcha CPU)**

Until `lite_zh_en × rpi5` is added to PRESET_TABLE, use explicit profile:
```bash
docker run -d --name seeed-rpi-lite -p 8000:8000 \
  -e SEEED_LOCAL_VOICE_PROFILE=rpi5-default \
  -v seeed-models:/opt/models \
  seeed-local-voice:rpi-v1.1
```

## Endpoints to measure

| Endpoint | Method | Purpose | Headers exposed |
|---|---|---|---|
| `/health` | GET | Backend readiness | — |
| `/asr/capabilities` | GET | ASR backend info | — |
| `/tts/capabilities` | GET | TTS backend info | — |
| `/asr` | POST multipart `file=@x.wav` | Offline ASR | `X-Inference-Time`, `X-Audio-Duration`, `X-RTF` |
| `/asr/stream` | WebSocket binary PCM int16 in | Streaming ASR | — |
| `/tts` | POST JSON `{"text":"...","speaker_id":0}` | Offline TTS → WAV | `X-Audio-Duration`, `X-Inference-Time`, `X-RTF` |
| `/tts/stream` | POST JSON → PCM stream | Streaming TTS | `Content-Type: audio/pcm`, `Sample-Rate` |
| `/tts/clone/embedding` | POST multipart WAV → JSON embedding | Voice clone extraction | — |
| `/tts/clone` | POST JSON `{"text":"...","speaker_embedding_b64":"..."}` | Voice-cloned TTS | same as /tts |

## Quick perf snapshot per device

For each device, run these 5 calls in order and record the headers:

```bash
# 0. Health & capabilities
curl -s http://HOST:PORT/health | jq
curl -s http://HOST:PORT/asr/capabilities | jq
curl -s http://HOST:PORT/tts/capabilities | jq

# 1. TTS short (1.5 sec)
curl -s -X POST http://HOST:PORT/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"你好世界","speaker_id":0}' \
  --output /tmp/tts_short.wav \
  -D /tmp/tts_short.hdr
grep -i "^x-" /tmp/tts_short.hdr

# 2. TTS medium (~5 sec)
curl -s -X POST http://HOST:PORT/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"今天天气真好，我想出去走走。","speaker_id":0}' \
  --output /tmp/tts_med.wav -D /tmp/tts_med.hdr
grep -i "^x-" /tmp/tts_med.hdr

# 3. ASR short — feed back the TTS output
curl -s -F file=@/tmp/tts_short.wav http://HOST:PORT/asr -D /tmp/asr.hdr
grep -i "^x-" /tmp/asr.hdr

# 4. Streaming ASR (websocat — feed 100ms PCM chunks)
# Manual; latency = time from first chunk to first partial JSON.

# 5. TTS streaming TTFB
time curl -s -X POST http://HOST:PORT/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"你好世界，我是 Seeed Local Voice"}' \
  --output /dev/null
```

Numbers to record into the perf table:
- **TTFA** (time to first audio byte) — measure with `time curl --output - .../tts/stream | head -c 1024 > /dev/null`
- **/tts RTF** (X-RTF header) — inference time / audio duration
- **/tts preload time** — from container `docker logs <name>` look for `"Speech service ready"` minus container start
- **/asr e2e latency** (X-Inference-Time) for 1.5s and 5s clips
- **/asr/stream first-partial delay** (manual)
- **Memory** — `docker stats <name> --no-stream`
- **Image size** — `docker images --format "{{.Size}}" seeed-local-voice:...`

## Measured ASR perf (local mode, 2026-05-13)

All numbers from **local-mode smoke** (`bench/perf/run_on_device.sh <node> -- asr --warmup 5 --runs 10`):
client runs on the device, talks to `127.0.0.1:8000`, eliminating Mac↔device network from measurements.
Corpus: 20 FLEURS files (10 zh + 10 en, 3-15s, sha256-locked). CER is character-level for zh, WER is word-level for en; both computed with Chinese-number normalization (cn2an) so "15米" and "十五米" compare equal.

### Latency: Finalize RTF p50 (compute-bound, lower=better)

`fRTF = eos_to_final_ms / audio_dur_ms`. Independent of client-side realtime pacing.

| Group | **Jetson Orin Nano** (Qwen3 TRT, voice_clone) | **RK3588** (Qwen3 RKNN w8a8, multilang) | **RK3576** (Qwen3 RKNN w4a16, multilang) | **RPi5** (sherpa-onnx, lite_zh_en) |
|---|---:|---:|---:|---:|
| short/zh (~3s) | **0.084** | 0.220 | 0.397 | **0.000** |
| short/en (~3s) | **0.072** | 0.254 | 0.386 | **0.000** |
| long/zh (~13s)  | 0.063 | **0.030** | 0.538 | **0.000** |
| long/en (~12s)  | 0.064 | 0.063 | 0.666 | **0.000** |

- **RPi5 ≈ 0**: sherpa is fully streaming, encoder + decoder both emit during chunk send; finalize is just a flush
- **Nano/RK3588 long-audio ~0.03-0.06**: both run Qwen3 ASR, encoder is fast for long content because mel batches stack efficiently
- **RK3576 ~0.4-0.7**: smaller NPU + w4a16 (4-bit) quant trade speed for fit; still well under realtime (RTF<1.0)
- **Nano short-after-long was a 4× cold-tactic regression**, fixed by pre-warming TRT shapes 1..6 at boot (see `EDGE_LLM_ASR_PREWARM_MAX`)

### Quality: CER/WER p50 (lower=better)

| Group | Nano | RK3588 | RK3576 | RPi5 |
|---|---:|---:|---:|---:|
| short/zh CER | 5.3% | **2.6%** | 5.3% | 10.5% |
| short/en WER | **0.0%** | 10.0% | 13.1% | 35.7% |
| long/zh CER (normalized) | 8.4% | 10.8% | **7.8%** | 14.5% |
| long/en WER | **3.0%** | 5.5% | 5.5% | 23.6% |

- **English short**: Nano wins (0%) — Qwen3 + TRT-EdgeLLM clearly better than RKNN port + sherpa
- **Chinese**: Nano/RK3588/RK3576 (all Qwen3 family) within ~5pp of each other; RPi5 (sherpa) trails by ~10pp
- **All Qwen3 variants share the same proper-noun limitation** ("Oravec" → "Work" on zh_long_03); fixing requires model-side tuning, not a device problem

### Known shared issues (today, not blocking)

1. **Mixed-script proper nouns**: "Oravec" / "Smith" in Chinese context → all Qwen3 variants mis-hear identically (~5 chars/error)
2. **FLEURS reference uses Arabic numerals** ("15米") while Chinese ASR speaks Chinese ("十五米"); perf tool's `cn2an` normalizer corrects for this — raw CER may show 30-40pp inflation if you bypass normalization

## Suggested perf table (template)

| Device | SoC | Preset | TTS RTF | ASR fRTF p50 (short) | ASR fRTF p50 (long) | CER zh-short | WER en-short | Image |
|---|---|---|---|---|---|---|---|---|
| Jetson Orin Nano | sm87 8GB | voice_clone | 0.69 | **0.084** | **0.063** | 5.3% | **0.0%** | 3.14 GB (v1.11/v1.12) |
| Jetson Orin Nano | sm87 8GB | multilang | TBD | TBD | TBD | TBD | TBD | 3.14 GB |
| Jetson Orin NX | sm87 16GB | voice_clone | TBD | TBD | TBD | TBD | TBD | 3.14 GB |
| RK3588 (Radxa ROCK 5T) | rk3588 16GB | multilang | 0.19 | 0.220 | **0.030** | **2.6%** | 10.0% | 1.38 GB (rk-v1.2) |
| RK3576 (cat-remote) | rk3576 8GB | multilang | TBD | 0.397 | 0.538 | 5.3% | 13.1% | 1.38 GB (rk-v1.2) |
| RPi5 | BCM2712 8GB | lite_zh_en | TBD | **0.000** | **0.000** | 10.5% | 35.7% | 560 MB (rpi-v1.1) |
| RPi5 | BCM2712 8GB | asr_zh_en | — | TBD | TBD | TBD | TBD | 560 MB |
| RPi4 / CM4 | BCM2711 4GB | asr_zh_en | — | TBD | TBD | TBD | TBD | 560 MB |

Measured 2026-05-13 (local mode, 5 warmup + 10 runs); see "Measured ASR perf" section above for full breakdown.

## Common gotchas

- **`SEEED_LOCAL_VOICE_PROFILE=...` overrides PRESET.** Don't pass both. New v1.7 / rk-v1.1 / rpi-v1.1 images do NOT bake a default PROFILE.
- **HF_ENDPOINT for China users**: set `https://hf-mirror.com` so engine_resolver can pull bundles. Defaults to `https://huggingface.co` (works internationally).
- **Jetson multilang requires the `qwen3-edgellm-jetson` artifact set** at the path the profile expects. The `-v /tmp/nano-audit:/opt/models/qwen3-edgellm:ro` mount is what supplies it on Nano; for AGX/NX you must `hf download harvestsu/qwen3-edgellm-jetson-artifacts --local-dir <path>` first.
- **RK image needs librknnrt.so + librkllmrt.so from host** via `-v` bind. The image does NOT bundle the runtime — different RK SoCs need different runtime versions.
- **Port 8000 conflicts**: if you already have a service on 8000 (e.g. existing jvrpi on harvest-pi), use `-p 8765:8000` or `--network host` carefully.
- **`docker logs <name>` can be noisy** during first start (model download progress bars). Filter with `| grep -E "INFO|WARN|ERROR|backend|Application|ready"`.

## Reload / restart commands

To re-run a container after stopping:
```bash
docker start seeed-nano-v17     # warm restart, all caches preserved
docker stop seeed-nano-v17       # graceful shutdown
docker logs -f seeed-nano-v17    # tail logs
docker exec -it seeed-nano-v17 bash  # shell inside
```

Volumes (`seeed-nano-models`, `seeed-models`, `/tmp/nano-audit`) are
preserved across restarts. Engines cached in the volume reload in
seconds rather than minutes.

## Rollback

Previous images preserved on each host. To rollback:
```bash
# Jetson
docker run -d --runtime nvidia --network host \
  ... \
  sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.5  # was last known-good

# RK
docker run ... seeed-local-voice:rk-v1.0   # works but uses old (rk3576-default) baked profile

# RPi
docker run ... seeed-local-voice:rpi-v1.0  # original Allenkzl-inspired CLI was the reference
```
