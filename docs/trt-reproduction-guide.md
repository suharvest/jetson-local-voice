# Paraformer + Matcha Pure TRT — Reproduction Guide

Converts the `zh_en` deployment profile from ONNX Runtime CUDA to pure TensorRT,
eliminating the heavy `dustynv/onnxruntime` base image dependency.

**Final state**: encoder=trt, decoder=trt, vocos=trt, acoustic=ort-cpu (lightweight).

## Prerequisites

- Jetson Orin (Nano/NX/AGX) with JetPack 6.x (R36.4.0+), TRT 10.3, CUDA 12.6
- Docker with `--runtime=nvidia`
- Model files downloaded by `app/core/model_downloader.py` on first start
- Python 3.10+ with `onnx`, `onnxruntime`, `numpy` (for ONNX surgery; can run on any machine)

## Quick Start (Pre-built Engines)

Pre-built engines for Orin Nano (SM 8.7, TRT 10.3, JP 6.2, CUDA 12.6) are
resolved automatically by `app.core.engine_resolver` from the Hugging Face
artifact repo. For manual placement, copy the encoder FP32 engine and decoder
TRT engine to the model directory:

```bash
# On Jetson device
mkdir -p /opt/models/paraformer-streaming/engines
cp deploy/artifacts/engines/orin-nano/paraformer_encoder_dp4_400.plan \
   /opt/models/paraformer-streaming/engines/paraformer_encoder_dp4_400.plan
cp deploy/artifacts/engines/orin-nano/paraformer_decoder_fp16.plan \
   /opt/models/paraformer-streaming/engines/paraformer_decoder_fp16.plan
```

Build and run:
```bash
docker build -f deploy/docker/Dockerfile.jetson \
  --build-arg LANGUAGE_MODE=zh_en \
  -t seeed-local-voice:jetson-zh_en .

docker run -d --name seeed-voice --network=host --runtime=nvidia \
  -e SEEED_LOCAL_VOICE_PROFILE=jetson-zh-en \
  -e MATCHA_ACOUSTIC_EP=SPLIT_TRT \
  -v seeed-models:/opt/models \
  seeed-local-voice:jetson-zh_en
```

## Full Reproduction (Build Engines from Scratch)

### Step 1: ONNX Surgery

Run on any machine with `onnx` and `onnxruntime` installed. The surgery scripts
modify the original ONNX models to be TRT-compatible.

**Paraformer Decoder** — externalize dynamic `make_pad_mask` subgraphs:
```bash
python3 scripts/surgery_paraformer_decoder.py \
  --input /path/to/decoder.onnx \
  --output /path/to/decoder-trt.onnx \
  --validate
```
Removes 22 nodes (Range/ReduceMax/Less chain that causes TRT Cask convolution
failure). Adds `pad_mask` [1, L] and `enc_pad_mask` [1, L] float32 inputs.
Parity validated: logits match original exactly.

**Matcha Acoustic** — externalize `RandomNormalLike` + fix `length_scale` type:
```bash
python3 scripts/surgery_matcha_acoustic.py \
  --input /path/to/model-steps-3.onnx \
  --output /path/to/model-steps-3-trt.onnx \
  --max-mel 800 \
  --validate
```
- Replaces `RandomNormalLike` with static noise input [1, 80, 800] + internal Slice
- Converts `length_scale` to INT64 fixed-point (×1000) for TRT shape-tensor compat
- NOTE: TRT build still blocked by duration predictor shape analysis; use ORT-CPU

### Step 2: Build TRT Engines

Run on the Jetson device with `trtexec` (included in TensorRT installation).

**Paraformer Encoder** (607MB ONNX → ~637MB FP32 engine):
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/path/to/encoder.onnx \
  --saveEngine=paraformer_encoder_dp4_400.plan \
  --minShapes=speech:1x40x560,speech_lengths:1 \
  --optShapes=speech:1x80x560,speech_lengths:1 \
  --maxShapes=speech:1x400x560,speech_lengths:1 \
  --memPoolSize=workspace:2048 --skipInference
```
Build time: ~1 min on Orin Nano with TensorRT 10.3. Use `--skipInference` to
avoid auto-inference shape mismatch. Do not build the encoder with `--bf16`:
the previous BF16 engine produced NaN outputs on real devices.

**Paraformer Decoder** (218MB ONNX → 118MB BF16 engine):
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/path/to/decoder-trt.onnx \
  --saveEngine=paraformer_decoder_bf16.plan \
  --bf16 \
  --minShapes=enc:1x1x512,acoustic_embeds:1x1x512,pad_mask:1x1,enc_pad_mask:1x1 \
  --optShapes=enc:1x40x512,acoustic_embeds:1x10x512,pad_mask:1x10,enc_pad_mask:1x40 \
  --maxShapes=enc:1x400x512,acoustic_embeds:1x40x512,pad_mask:1x40,enc_pad_mask:1x400 \
  --memPoolSize=workspace:1024
```
**CRITICAL**: Must use BF16. FP16 causes FSMN Conv precision loss → all tokens
identical (always outputs "的").

**Matcha Acoustic** (split TRT):
Full acoustic TensorRT remains blocked by the duration predictor's dynamic shape
chain, so the release path is split: duration/encoder stays on the lightweight
ORT-CPU graph and the high-compute ODE estimator step runs as a TensorRT BF16
engine (`MATCHA_ACOUSTIC_EP=SPLIT_TRT`). Vocos remains TensorRT.

### Step 3: Verify

```bash
# Check health
curl http://localhost:8621/health
# Expected: asr_providers={"encoder":"trt","decoder":"trt"}

# TTS test
curl -X POST http://localhost:8621/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"你好","speed":1.0}' -o /tmp/test.wav

# ASR test (closed loop)
curl -X POST http://localhost:8621/asr \
  -F "file=@/tmp/test.wav" -F "language=zh"

# Precision diagnosis (2×2 matrix):
docker exec <container> python3 /tmp/diag_2x2.py /tmp/test.wav
```

### Step 4: Precision Benchmarks (Orin Nano, "你好" TTS audio)

| Component | TRT vs ORT | Notes |
|-----------|-----------|-------|
| Encoder | no NaN on warmup and ASR round-trip | FP32 TRT |
| Decoder | bit-exact token match | BF16, identical to ORT |
| CIF | same token count (2) | TRT and ORT agree |
| Full pipeline | "你好" → "你好" | Correct |

## Architecture

```
zh_en profile (post-conversion):
┌─────────────────────────────────────────────┐
│ ASR Pipeline                                 │
│  audio → numpy fbank → LFR stack             │
│  → TRT Encoder (FP32, ~637MB)                │
│  → CIF (Python)                              │
│  → TRT Decoder (BF16, 118MB)                 │
│  → text                                      │
├─────────────────────────────────────────────┤
│ TTS Pipeline                                 │
│  text → tokenizer (Python)                   │
│  → ORT-CPU Acoustic (73MB ONNX)              │
│  → TRT Vocos (FP16, 52MB)                   │
│  → ISTFT (Python) → audio                    │
└─────────────────────────────────────────────┘

Removed dependencies:
  ✗ sherpa-onnx (~500MB)
  ✗ librosa (~200MB)
  ✗ ORT-CUDA EP (libonnxruntime_providers_cuda.so)
  ✗ dustynv/onnxruntime base image

Kept (lightweight):
  ✓ onnxruntime CPU-only (pip, ~15MB)
  ✓ cuda-python (for TRT)
  ✓ piper-phonemize (Matcha English G2P)
```

## File Manifest

| File | Purpose |
|------|---------|
| `scripts/surgery_paraformer_decoder.py` | Decoder ONNX → TRT-compatible |
| `scripts/surgery_matcha_acoustic.py` | Matcha acoustic ONNX surgery |
| `scripts/build_paraformer_trt.sh` | Build Paraformer TRT engines |
| `scripts/diag_2x2.py` | 2×2 precision diagnosis |
| `app/backends/jetson/paraformer_trt.py` | Paraformer TRT backend |
| `app/backends/jetson/matcha_trt.py` | Matcha TRT + ORT-CPU backend |
| `deploy/docker/Dockerfile.jetson` | Slim Docker build |
| `configs/profiles/jetson-zh-en.json` | zh_en deployment profile |
| `deploy/artifacts/engines/orin-nano/` | Pre-built TRT engines |
| `deploy/artifacts/onnx/` | Surgically-modified ONNX files |
