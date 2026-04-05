# RK3576 Speech Service Adaptation Plan

> Target: LubanCat-3 (RK3576, 8GB RAM, 6 TOPS NPU, Debian 12)
> Source: jetson-voice (Jetson Orin NX 16GB, CUDA EP)
> Date: 2026-04-04

## 1. Current Architecture (Jetson)

All components run via **sherpa-onnx** with onnxruntime **CUDAExecutionProvider**:

| Component | Model | Provider | Notes |
|-----------|-------|----------|-------|
| Streaming ASR (zh+en) | Paraformer streaming 220M | CUDA EP | `encoder.onnx` + `decoder.onnx` |
| Offline ASR | SenseVoice | CUDA EP | Lazy-loaded |
| TTS (zh+en) | Matcha + Vocos | CUDA EP | 16kHz, multi-speaker |
| TTS (en) | Kokoro v1.0 | CUDA EP | 53 speakers |

Docker image: `dustynv/onnxruntime:1.20-r36.4.0` (Jetson-specific)

## 2. Target Platform: RK3576

- **SoC**: Rockchip RK3576 (4x A72 + 4x A53)
- **NPU**: 6 TOPS, 2 cores
- **RAM**: 8 GB (shared with NPU)
- **OS**: Debian 12 (Bookworm), aarch64
- **Docker**: 29.3.1, Python 3.11.2
- **Fleet name**: `cat-remote` (Tailscale: 100.89.94.11)

## 3. sherpa-onnx RKNN Support Matrix

### Already supported (C++ layer)

| Model | RKNN | RK3576 | Type |
|-------|------|--------|------|
| Offline Paraformer-large | Yes | Commented out (only rk3588 tested) | Offline ASR |
| Offline SenseVoice | Yes | Yes | Offline ASR |
| Online Zipformer (transducer + CTC) | Yes | Not listed | Streaming ASR |
| Silero VAD | Yes | Yes | VAD |
| Whisper | Yes | Yes | Offline ASR |

### Not supported

| Model | RKNN | Notes |
|-------|------|-------|
| **Online Paraformer (streaming)** | No | No RKNN impl exists |
| **Matcha TTS** | No | No RKNN impl exists |
| **Kokoro TTS** | No | No RKNN impl exists |
| **Qwen3-TTS** | No | Needs RKNN + RKLLM hybrid |

Key source files:
- RKNN models: `sherpa-onnx/csrc/rknn/`
- Online Paraformer (ORT): `sherpa-onnx/csrc/online-paraformer-model.cc`
- RKNN dispatch: `sherpa-onnx/csrc/online-recognizer-impl.cc:41-59`

## 4. Chinese Solution Adaptation Paths

### 4.1 Streaming Paraformer ASR

**Current**: `OnlineRecognizer.from_paraformer(provider="cuda")` with `encoder.onnx` + `decoder.onnx`

**Option A: CPU onnxruntime (zero effort)**
- Change `provider="cpu"`, use standard aarch64 onnxruntime from PyPI
- Expected RTF: 0.3-0.8 on A72 cores (needs benchmarking)
- Pros: works immediately, no code changes
- Cons: no NPU acceleration

**Option B: RKNN NPU adaptation (~9-13 days)**
- Write `online-paraformer-model-rknn.h/.cc` (reference: `online-zipformer-transducer-model-rknn.cc`)
- Write `online-recognizer-paraformer-rknn-impl.h`
- Extend `OnlineStreamRknn` for Paraformer state management
- Export fixed-shape ONNX models (chunk_size=[0,10,5] = 600ms)
- Convert to RKNN format via `rknn.api`
- Modify `online-recognizer-impl.cc` RKNN dispatch to support Paraformer

Work breakdown:
| Task | Effort |
|------|--------|
| ONNX -> RKNN export script (fixed shape) | 1-2 days |
| C++ model wrapper (encoder + decoder + states) | 3-5 days |
| Recognizer integration + stream state | 2 days |
| Cross-compile + on-device debugging | 2-3 days |
| Python binding verification | 1 day |

Risks:
- Decoder has complex state (conv caches per block), RKNN tensor I/O all raw buffers
- CIF predictor may not map well to NPU ops, may need CPU fallback
- Possible NCHW/NHWC layout issues (seen in Zipformer RKNN impl)

### 4.2 Matcha TTS

**Current**: sherpa-onnx `OfflineTts` with Matcha acoustic model + Vocos vocoder, CUDA EP

**Only path: CPU onnxruntime**
- No RKNN adaptation exists for any TTS model in sherpa-onnx
- Matcha is relatively lightweight, CPU may be acceptable
- Needs benchmarking on RK3576

### 4.3 Qwen3-TTS (future, from fork)

**Architecture**: 10 sub-models, AR generation via 0.6B talker

| Sub-model | Deployment | Notes |
|-----------|-----------|-------|
| text_project, codec_embed, speaker_encoder | RKNN NPU | Small, fixed shape, trivial |
| code_predictor, code_predictor_embed | RKNN NPU or CPU | Max 17 steps, small |
| tokenizer12hz_decode_stream | RKNN NPU | Already fixed shape (T=75) |
| tokenizer12hz_decode (batch) | RKNN NPU | Needs fixed shape export |
| **talker_prefill + talker_decode** | **RKLLM** | 0.6B LLM with KV-cache |

## 5. RKLLM for Qwen3-TTS Talker

### Overview

- **SDK**: `airockchip/rknn-llm` v1.2.3 (separate from RKNN)
- **Purpose**: LLM autoregressive inference on Rockchip NPU
- **API**: C (`librkllmrt.so`), async callback pattern
- **Conversion**: HuggingFace -> RKLLM-Toolkit (needs CUDA PC) -> `.rkllm` file

### RK3576 Benchmarks (128 input, 64 output tokens)

| Model | Quant | First Token | Decode Speed | Memory |
|-------|-------|-------------|-------------|--------|
| Qwen2 0.5B | w4a16 | 328 ms | 34.2 tok/s | 426 MB |
| Qwen3 0.6B | w4a16 | 483 ms | 25.2 tok/s | 496 MB |
| Qwen3 0.6B | w8a8 | 449 ms | 17.1 tok/s | 780 MB |

### Feasibility for TTS

- Codec rate: 12.5 Hz (12.5 tokens per second of audio)
- Qwen3 0.6B w4a16 on RK3576: 25.2 tok/s
- **RTF = 12.5 / 25.2 = 0.50** (real-time feasible)
- Memory: ~496 MB for talker + ~200 MB for other models = ~700 MB total

### Conversion Pipeline

```python
from rkllm.api import RKLLM

llm = RKLLM()
llm.load_huggingface(model='path/to/qwen3-tts-talker', device='cuda')
llm.build(
    do_quantization=True,
    quantized_dtype="W4A16",
    target_platform="RK3576",
    num_npu_core=2,
    max_context=4096,
)
llm.export_rkllm("./talker_w4a16_rk3576.rkllm")
```

Requires CUDA GPU for quantization (use WSL2 machine: `wsl2-local`).

### Integration: Pure RKNN (RKLLM Abandoned)

**RKLLM v1.2.3 cannot convert the talker** — it has asymmetric attention dimensions
(head_dim=128, but hidden_size/num_heads=64). RKLLM hardcodes head_dim from hidden_size/num_heads.

**New approach**: All models run on RKNN, including talker. The ONNX export already uses
**stateless KV-cache** (explicit tensor I/O), same pattern as sherpa-onnx's Zipformer RKNN:

```
talker_prefill.onnx:
  Input:  inputs_embeds [1,T,1024] + attention_mask [1,T]
  Output: logits [1,1,3072] + last_hidden [1,T,1024]
          + 56 KV tensors [1, 8, T, 128] (28 layers × key+value)

talker_decode.onnx:
  Input:  inputs_embeds [1,1,1024] + attention_mask [1,T+1]
          + 56 past_KV [1, 8, T, 128]
  Output: logits [1,1,3072] + last_hidden [1,1,1024]
          + 56 new_KV [1, 8, T+1, 128]
```

KV-cache managed externally in C++. For RKNN, seq_len dimension must be fixed at export time.

## 6. Qwen3-TTS Voice Cloning

### Mechanism

Qwen3-TTS (Base model only) supports **zero-shot voice cloning** via a speaker encoder (ECAPA-TDNN, 1024-dim output). Two modes:

| Mode | Input | Prefill overhead | Quality |
|------|-------|-----------------|---------|
| **x-vector only** | Reference audio only | +1 position | Identity only, no prosody |
| **ICL (in-context learning)** | Reference audio + ref_text | +80+ positions (ref codec tokens) | Best (preserves prosody) |

### Speaker Embedding Injection

The speaker embedding is **element-wise added** to `tts_pad` embedding at one prefill position:

```
Standard prefill (8 positions):
  [role(3)] + [codec_prefix(4)] + [text[0]+codec_bos(1)]

Voice clone prefill (9 positions, x-vector only):
  [role(3)] + [codec_prefix(4)] + [tts_pad + speaker_embedding(1)] + [text[0]+codec_bos(1)]
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                   inserted here, element-wise ADD

ICL mode prefill (variable length):
  [role(3)] + [codec_prefix(4)] + [tts_pad + speaker_embedding(1)] + [ref_audio codec tokens...] + [text+codec_bos...]
```

### Data Flow

```
reference_audio [S] → resample to 24kHz
    → mel spectrogram [1, T, 128]
    → speaker_encoder.onnx → speaker_embedding [D=1024]
    → ADD with tts_pad at prefill position 7
    → talker prefill + AR generation (unchanged from here)
```

### Current Implementation Status (sherpa-onnx fork)

- `speaker_encoder.onnx`: exported and loaded in C++ (`OfflineTtsQwen3Model::RunSpeakerEncoder`)
- `tokenizer12hz_encode.onnx`: exported (for ICL mode ref audio → codec tokens)
- `GenerationConfig` already has `reference_audio`, `reference_sample_rate`, `reference_text` fields
- **Not wired into Generate()** — prefill construction in `offline-tts-qwen3-impl.cc:184-276` only has standard mode

### Implementation Effort

| Scope | Changes | Effort |
|-------|---------|--------|
| x-vector only mode | ~50-80 lines C++ in impl.cc + mel extraction utility | 1-2 days |
| Full ICL mode | +200 lines: tokenize ref audio, build extended prefill | 2-3 days |

### Model Compatibility

| Model | Voice Cloning | Style Control |
|-------|--------------|---------------|
| **Qwen3-TTS-12Hz-0.6B-Base** | Yes | No |
| **Qwen3-TTS-12Hz-0.6B-CustomVoice** | No (9 built-in speakers) | Yes (natural language instructions) |

### RKLLM Integration for Voice Cloning

Key RKLLM API features enabling voice cloning on RK3576:

- `RKLLM_INPUT_EMBED`: accepts pre-computed float embedding arrays (the mixed text_embed + speaker_embed + codec_embed input)
- `RKLLM_INFER_GET_LOGITS`: returns raw logits for codec token sampling
- Speaker encoder runs on RKNN NPU (small model, fast)

The speaker embedding is computed once per reference audio and injected into the prefill embedding sequence before sending to RKLLM — no changes needed in the RKLLM talker model itself.

## 7. Jetson TensorRT Optimization (Quick Win)

The current Jetson deployment can be optimized by switching from CUDA EP to TensorRT EP:

```bash
# Just change env vars
TTS_PROVIDER=trt
STREAMING_ASR_PROVIDER=trt
```

Benefits:
- 20-50% inference speedup (graph optimization + FP16)
- Eliminates warmup (current: 15 texts x 5 rounds = 75 warmup calls)
- Engine cached to disk after first compilation
- Zero code changes — sherpa-onnx already supports `provider="trt"`

Limitation: TensorRT currently only accelerates encoder; decoder/joiner fallback to CUDA EP.

## 8. Recommended Execution Order

### Phase 1: Validate (1-2 days)
1. Benchmark CPU onnxruntime on RK3576 for Paraformer streaming + Matcha TTS
2. Test TensorRT EP on Jetson (just env var change)

### Phase 2: Chinese Solution on RK3576 (1-2 weeks)
1. Build CPU-only Docker image for RK3576 (Debian 12 base, no CUDA deps)
2. Deploy Paraformer streaming + Matcha TTS with `provider="cpu"`
3. Benchmark end-to-end latency, decide if CPU is acceptable

### Phase 3: NPU Acceleration (2-4 weeks)
1. Convert SenseVoice to RKNN for RK3576 (existing script, low risk)
2. Attempt streaming Paraformer RKNN adaptation (high effort, see 4.1 Option B)
3. Or: switch to SenseVoice RKNN for ASR (offline, but NPU-accelerated)

### Phase 4: Qwen3-TTS on RK3576 (3-5 weeks)
1. Extract talker from Qwen3-TTS, convert via RKLLM-Toolkit on WSL2
2. Convert audio decoder + small models to RKNN
3. Build hybrid RKLLM + RKNN inference pipeline in C++
4. Integrate into sherpa-onnx fork

## 9. Device & Environment Reference

| Device | Role | Fleet Name | Specs |
|--------|------|-----------|-------|
| RK3576 board | Target deployment | `cat-remote` | 8GB RAM, Debian 12, Tailscale |
| WSL2 desktop | Model conversion (CUDA) | `wsl2-local` | 31GB RAM, GPU, Ubuntu 24.04 |
| Jetson Orin Nano | Reference/comparison | `seeed-desktop` | 15GB RAM, JetPack 6.x |
| Jetson AGX Orin | Heavy testing | `base-j50` | 61GB RAM, JetPack 6.x |
