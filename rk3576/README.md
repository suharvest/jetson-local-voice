# Qwen3-TTS on RK3576 — RKNN Deployment

## Overview

Deploy Qwen3-TTS (0.6B, 12Hz codec) on RK3576 (LubanCat-3, 8GB RAM, 6 TOPS NPU) with:
- Full NPU acceleration via RKNN (all 9 models)
- Voice cloning (x-vector mode) via speaker_encoder
- Stateless KV-cache for talker AR generation

## Architecture

```
Text → text_project (RKNN) → embeddings
                                  ↓
                    prefill_embeds + [optional: speaker_encoder (RKNN)]
                                  ↓
                    talker_prefill (RKNN) → logits + KV-cache
                                  ↓
                    AR loop: talker_decode (RKNN) → codec tokens
                        ↓
                    code_predictor + code_predictor_embed (RKNN) → residual codes
                        ↓
                    tokenizer12hz_decode_stream (RKNN) → audio waveform
```

## RKNN Models (rk3576, FP16, all 9/9 converted)

| Model | RKNN Size | Fixed Shape | Notes |
|-------|----------|-------------|-------|
| talker_prefill | 1,322 MB | embeds[1,32,1024] mask[1,32] | Max prefill 32 tokens |
| talker_decode | 894 MB | embeds[1,1,1024] mask[1,513] 56×KV[1,8,512,128] | Max 512 AR steps (~40s audio) |
| text_project | 606 MB | input_ids[1,128] | Text → embedding projection |
| tokenizer12hz_decode_stream | 237 MB | audio_codes[1,75,16] | Streaming decode (ctx=50+chunk=25) |
| tokenizer12hz_encode | 241 MB | audio[1,72000] | Audio → codec (voice clone ICL) |
| code_predictor | 174 MB | context[1,2,1024] | Residual codebook prediction |
| speaker_encoder | 18 MB | mel[1,300,128] | Voice cloning (onnxsim required) |
| codec_embed | 6 MB | token_ids[1,1] | Codec token embedding |
| code_predictor_embed | 4 MB | token_id[1,1] scalar | Residual embedding |
| **Total** | **3,502 MB** | | |

## Conversion Environment

- Machine: wsl2-local (RTX 3060, Ubuntu 24.04, Python 3.12)
- rknn-toolkit2 2.3.2 (requires onnx<=1.16.2)
- torch 2.4 (CPU, NOT 2.11 — TorchScript regression)
- qwen-tts 0.1.1

## Key Technical Decisions

### RKLLM Abandoned
RKLLM v1.2.3 cannot convert the talker — asymmetric attention:
- head_dim=128 (from q_proj [2048,1024])
- RKLLM expects head_dim = hidden_size/num_heads = 1024/16 = 64

### Pure RKNN with Stateless KV-cache
ONNX export uses explicit KV-cache tensor I/O (56 tensors: 28 layers × key+value).
C++ manages KV-cache buffers externally, same pattern as sherpa-onnx Zipformer RKNN.

### Fixed Shapes Required
RKNN needs all dimensions fixed at compile time. Reshape nodes bake constants at trace time.
Production shapes: prefill T=32, decode past_len=512.

### Conversion Workarounds
- speaker_encoder: run `onnxsim.simplify()` before RKNN to resolve If-node constant folding
- tokenizer decoder: force `attn_implementation="eager"` (SDPA breaks tracing)
- torch `_is_torch_greater_or_equal_than_2_5 = True` to avoid `aten::__ior_` export failure

## Files on wsl2-local

```
~/qwen3-tts-export/
├── qwen3-tts-0.6b-12hz-fixed/      # Fixed-shape ONNX models (5.7 GB)
├── qwen3-tts-0.6b-12hz-rknn-fixed/ # RKNN models for rk3576 (3.5 GB)
├── export_fixed_shapes.py           # Main 7-model ONNX export
├── export_remaining_v2.py           # tokenizer12hz_encode export
├── export_decode_stream.py          # tokenizer12hz_decode_stream export
└── convert_rknn_fixed.py            # Batch RKNN conversion
```

## Voice Cloning (C++ integration)

Implemented in sherpa-onnx fork (`offline-tts-qwen3-impl.cc`):
- Speaker embedding extracted via FeatureExtractor (128-dim mel, 24kHz) + speaker_encoder
- Injected at prefill position 7 via element-wise ADD with tts_pad embedding
- GenerationConfig.reference_audio/reference_sample_rate fields (already existed)
- Graceful fallback when speaker_encoder not loaded

## E2E Quality Verification (2026-04-04)

TTS → ASR loop on RTX 3060 (bfloat16, PyTorch eager attention):

| Input | ASR Result | RTF | Match |
|-------|-----------|-----|-------|
| 你好 | 你好 | 2.62 | MATCH |
| 今天天气真不错 | 今天天气真不错 | 2.31 | MATCH |
| Hello nice to meet you | hello nice to meet you | 1.97 | MATCH |
| 我是你的语音助手 | 我是你的语音助手 | 2.01 | MATCH |

ASR endpoint: `ws://100.67.111.58:8621/asr/stream` (Paraformer streaming)

## Resolved Issues

- tokenizer12hz_decode_stream: exported with eager attention + RKNN converted (237 MB)
- speaker_encoder RKNN: fixed by running onnxsim.simplify() before conversion (18 MB)
- export-onnx.py bugs: text_embed_tokens→text_embedding, lm_head→codec_head, head_dim, DynamicCache
- System torch broken by agent: fixed with uv venv isolation

## RK3576 Hardware Benchmark (2026-04-04)

Runtime: librknnrt 2.3.2, NPU driver 0.9.8, dual-core enabled

| Model | past_len | Quant | ms/step | RTF | Speedup |
|-------|----------|-------|---------|-----|---------|
| talker_decode | 512 | FP16 | 1306 | 16.3 | baseline |
| talker_decode | 128 | FP16 | 472 | 5.9 | 2.8x |
| talker_decode | 64 | FP16 | 342 | 4.3 | 3.8x |
| talker_decode | 32 | INT8 | 256 | 3.2 | 5.1x |
| talker_prefill | — | FP16 | 343 | — | one-shot |

Other models: codec_embed 1ms, code_predictor 20ms, speaker_encoder 24ms, text_project 63ms.

**Root cause**: NPU profiler shows 121ms pure compute for p32 across 422 ops + 140ms dispatch overhead. RK3576 NPU (6 TOPS / ~1.5 TFLOPS FP16) insufficient for 0.6B 28-layer transformer.

**Conclusion**: RK3576 cannot run Qwen3-TTS 0.6B in real-time. Best RTF=3.2x (need <1.0).

## RKLLM Breakthrough (2026-04-04)

RKLLM conversion initially failed (vocab_size=3072 too small). Fixed by padding to 151936.

| Metric | Value |
|--------|-------|
| Talker decode (RKLLM W4A16) | **55ms/step** (with code_predictor feedback) |
| Talker prefill | **56ms** |
| Vocab=151936 model | 683 MB, fastest decode |
| KV-cache persistence | `keep_history=1` required |
| Step-by-step API | mode=1 (GET_LAST_HIDDEN_LAYER) + CPU codec_head matmul |

### RKLLM Step-by-Step AR Loop
```
Per step:
  1. rkllm_run(embed, mode=1, keep_history=1) → hidden [1024]     45ms
  2. logits = hidden @ codec_head_weight (CPU numpy)                1ms
  3. sample primary token from logits[:3072]                        <1ms
  4. code_predictor(hidden) → 15 residual codes                    5-10ms
  5. next_embed = sum(16 codebook embeddings) + text_embed          <1ms
  Total: ~55ms/step
```

## Vocoder Optimization (2026-04-04)

Original: 10,547ms (29 Sin ops fallback to CPU).
After Sin→polynomial replacement: **1,942ms** (5.4x speedup).

| Fix | Before | After |
|-----|--------|-------|
| Remove Gather (embedding lookup) | 10,547ms | 10,495ms (no effect) |
| Replace 29 Sin → 7th-order Taylor + Clip | 10,495ms | **1,942ms** |

Vocoder RTF: 1942ms / 6s audio = **0.32** (faster than real-time).

Script: `rk3576/scripts/replace_sin_polynomial.py`

## Current E2E Pipeline Performance

| Stage | Time | Notes |
|-------|------|-------|
| text_project (RKNN) | 63ms | one-shot |
| talker prefill (RKLLM) | 56ms | one-shot |
| talker decode × 25 (RKLLM) | 1375ms | 25 frames = 2s audio |
| code_predictor × 25 (RKNN) | 250ms | 15 residual codes per frame |
| vocoder (RKNN, 75 frames) | 1942ms | 6s audio, RTF=0.32 |
| **First-chunk latency** | **~210ms** | 1 frame: prefill + decode + vocoder |
| **E2E for 2s audio** | **~2.4s (RTF≈1.2)** | |

## Remaining Optimizations

| Optimization | Expected | Actual | Status |
|-------------|----------|--------|--------|
| Vocoder ctx50→ctx25 FP16 | ~1300ms | 1183ms | Done |
| Vocoder ctx25 INT8 | ~650ms | **682ms** | Done ✅ Production |
| Vocoder ctx0 INT8 | ~350ms | **327ms** | Done (quality=0.83) |
| text_project shape [1,128]→[1,16] | ~10ms | — | Not done |
| Pipeline parallelism (talker∥vocoder) | — | — | Architecture |

### Optimized Pipeline (ctx25 INT8 vocoder)

| Stage | Time | Notes |
|-------|------|-------|
| text_project (RKNN) | 63ms | one-shot |
| talker prefill (RKLLM) | 56ms | one-shot |
| talker decode × 25 (RKLLM) | 1375ms | 2s audio |
| code_predictor × 25 | 250ms | |
| vocoder ctx25 INT8 (RKNN) | 682ms | 2s audio, RTF=0.34 |
| **First-chunk latency** | **~210ms** | |
| **E2E 2s audio** | **~2.4s (RTF≈1.2)** | |

### Model files for production

| Model | File | Size |
|-------|------|------|
| talker (RKLLM W4A16) | talker_fullvocab_fixed_w4a16_rk3576.rkllm | 683 MB |
| vocoder (RKNN INT8) | decoder_ctx25_int8.rknn | 133 MB |
| text_project (RKNN FP16) | text_project.rknn | 606 MB |
| code_predictor (RKNN FP16) | code_predictor.rknn | 174 MB |
| codec_embed (RKNN FP16) | codec_embed.rknn | 6 MB |
| code_predictor_embed (RKNN FP16) | code_predictor_embed.rknn | 4 MB |
| speaker_encoder (ONNX CPU) | speaker_encoder.onnx | 34 MB |
| codec_head weight (numpy) | codec_head.npy | ~12 MB |
| codebook embeddings (numpy) | codebook_embeds/ | ~32 MB |
| **Total** | | **~1.7 GB** |
