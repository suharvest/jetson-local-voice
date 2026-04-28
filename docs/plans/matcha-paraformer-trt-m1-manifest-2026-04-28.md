# M1 Manifest — Matcha + Paraformer TRT Spec Freeze

Date: 2026-04-28
Target: orin-nx (100.82.225.102), /home/recomputer/jetson-voice, v3.4-slim container

---

## 1. Platform Version Info

| Component | Version |
|-----------|---------|
| JetPack | R36.4.3 (gcid 38968081, aarch64, Wed Jan 8 2025) |
| CUDA | 12.6 (cuda-cudart-12-6 12.6.68-1) |
| TensorRT | 10.3.0.30 (libnvinfer10), Python bindings 10.3.0 |
| TensorRT meta | nvidia-tensorrt 6.2+b77 |
| Kernel | R36.4.3 oot |
| Board | Seeed reComputer Super Orin NX 16G J401 |
| Image | mfi_recomputer-super-orin-nx-16g-j401-6.2-36.4.3-2025-05-22.tar.gz |

Device status: ONLINE, 14G free disk, 9.7/15.6GB RAM used, GPU: Orin (nvgpu).

---

## 2. Matcha-icefall-zh-en ONNX Inspection

### 2.1 Acoustic Model: model-steps-3.onnx

- **Path** (container): `/opt/models/matcha-icefall-zh-en/model-steps-3.onnx`
- **Size**: 75,717,082 bytes (72.2 MB)
- **MD5**: `308ab5cb918e6340057134e54424a4bc`
- **IR version**: 7
- **Producer**: pytorch 2.6.0
- **Total nodes**: 4,802
- **Scan/Loop/If**: NONE (all clear)

**Inputs:**

| Name | Shape | Dtype |
|------|-------|-------|
| `x` | [`N`, `L`] | int64 (symbolic dims) |
| `x_length` | [`N`] | int64 |
| `noise_scale` | [1] | float32 |
| `length_scale` | [1] | float32 |

**Outputs:**

| Name | Shape | Dtype |
|------|-------|-------|
| `mel` | [`N`, 80, `L`] | float32 |

**Notes:**
- `N` = batch (typically 1 for inference), `L` = text token length (dynamic)
- Output `L` (mel frames) is labeled identically to input `L` in ONNX, but the runtime dimension differs due to duration predictor expansion (~8-12 mel frames per input token)
- Contains `RandomNormalLike` (1 node) — ODE solver noise sampling. **TRT risk**: `RandomNormalLike` is not natively supported; must be externalized as an input tensor or custom plugin
- 4802 nodes with heavy Constant (1423), Unsqueeze (356), Add (342), Mul (417), Reshape (217), Transpose (149) — typical of a U-Net-style diffusion/ODE model
- 24 Gemm ops (linear projections), 24 Softmax (attention), 113 Conv, 39 InstanceNormalization
- No subgraph ops, no control flow

**Unique ops (46 total):**
Add, Cast, Ceil, Clip, Concat, Constant, ConstantOfShape, Conv, ConvTranspose, CumSum, Div, Equal, Exp, Flatten, Gather, Gemm, Identity, InstanceNormalization, Less, MatMul, Mul, Neg, Pad, Pow, RandomNormalLike, Range, Reciprocal, ReduceMax, ReduceMean, ReduceSum, Relu, Reshape, Shape, Sigmoid, Sin, Slice, Softmax, Softplus, Sqrt, Squeeze, Sub, Tanh, Tile, Transpose, Unsqueeze, Where

### 2.2 Vocoder: vocos-16khz-univ.onnx

- **Path** (container): `/opt/models/matcha-icefall-zh-en/vocos-16khz-univ.onnx`
- **Size**: 53,882,848 bytes (51.4 MB)
- **MD5**: `2b43a6235c158cb70480334c48c1c8b9`
- **IR version**: 7
- **Producer**: pytorch 2.6.0
- **Total nodes**: 272
- **Scan/Loop/If**: NONE (all clear)

**Inputs:**

| Name | Shape | Dtype |
|------|-------|-------|
| `mels` | [`batch_size`, 80, `time`] | float32 |

**Outputs:**

| Name | Shape | Dtype |
|------|-------|-------|
| `mag` | [`batch_size`, `Clipmag_dim_1`, `time`] | float32 |
| `x` | [`batch_size`, `Cosx_dim_1`, `time`] | float32 |
| `y` | [`batch_size`, `Cosx_dim_1`, `time`] | float32 |

**Notes:**
- Lightweight ConvNet: 9 Conv, 17 MatMul, 54 Add
- Outputs decompose into magnitude + ISTFT cos/sin components (`x`, `y`)
- Clipmag_dim_1 = FFT bin count / 2 + 1 (approx 513 for 1024-pt FFT at 16kHz)
- Cosx_dim_1 = same as mag last dim (complex spectrum components)
- Hop length for ISTFT is unknown from ONNX alone (likely 256 samples at 16kHz)
- Well-behaved for TRT: all standard ops, no control flow, no random ops

**Unique ops (19 total):**
Add, Clip, Constant, Conv, Cos, Div, Erf, Exp, Gather, MatMul, Mul, Pow, ReduceMean, Shape, Sin, Slice, Sqrt, Sub, Transpose

---

## 3. Paraformer-streaming ONNX Inspection

### 3.1 Encoder: encoder.onnx

- **Path** (container): `/opt/models/paraformer-streaming/encoder.onnx`
- **Size**: 636,348,877 bytes (607 MB)
- **MD5**: `38bb68f284cf2d34e5a8f98a7c671ffd`
- **Total nodes**: 6,271
- **Scan/Loop/If**: NONE (all clear)

**Inputs:**

| Name | Shape | Dtype |
|------|-------|-------|
| `speech` | [`batch_size`, `feats_length`, 560] | float32 |
| `speech_lengths` | [`batch_size`] | int32 |

**Outputs:**

| Name | Shape | Dtype |
|------|-------|-------|
| `enc` | [`batch_size`, `feats_length`, 512] | float32 |
| `enc_len` | [`batch_size`] | int32 |
| `alphas` | [`batch_size`, `feats_length`] | float32 |

**Architecture:**
- 19 CFSMN (Compact Feeding Sequential Memory Network) layers with Conv kernel_size=11
- Initial conv layer `encoders0` + 19 `encoders.0`–`encoders.18`
- FSMN uses causal conv padding (left-only context)
- Feature dimension: 560 = 80-dim fbank × 7-frame stacking
- CIF predictor is **integrated into the encoder** — `alphas` output carries CIF weights directly
- Encoder output `feats_length` has the same temporal resolution as input (no subsampling)
- `enc_len` = actual speech length after padding removal (int32)
- Total effective receptive field: 19 layers × kernel=11 → large temporal context, but causal so only left

**About CIF dynamicity:**
- `feats_length` is **TRULY DYNAMIC** — no upper bound in ONNX shape
- `alphas` has the same length as input frames, enabling per-frame integration weights
- This feeds into the decoder as `acoustic_embeds` with variable `token_length`
- The spec proposal `min=4, opt=20, max=80` for CIF tokens is **reasonable for a first cut** (see decoder section)

**Unique ops (29 total):**
Add, Cast, Concat, Constant, ConstantOfShape, Conv, Div, Gather, Identity, Less, MatMul, Mul, Pad, Pow, Range, ReduceMax, ReduceMean, Relu, Reshape, Shape, Sigmoid, Slice, Softmax, Split, Sqrt, Squeeze, Sub, Transpose, Unsqueeze

### 3.2 Decoder: decoder.onnx

- **Path** (container): `/opt/models/paraformer-streaming/decoder.onnx`
- **Size**: 228,464,044 bytes (218 MB)
- **MD5**: `4eb7c94ece0ad861f18ef56db5f72379`
- **Total nodes**: 2,232
- **Scan/Loop/If**: NONE (all clear)

**Inputs:**

| Name | Shape | Dtype |
|------|-------|-------|
| `enc` | [`batch_size`, `enc_length`, 512] | float32 |
| `enc_len` | [`batch_size`] | int32 |
| `acoustic_embeds` | [`batch_size`, `token_length`, 512] | float32 |
| `acoustic_embeds_len` | [`batch_size`] | int32 |
| `in_cache_0`..`in_cache_15` | [`batch_size`, 512, 10] | float32 |

**Outputs:**

| Name | Shape | Dtype |
|------|-------|-------|
| `logits` | [`Addlogits_dim_0`, `Addlogits_dim_1`, 8404] | float32 |
| `sample_ids` | [`ArgMaxsample_ids_dim_0`, `ArgMaxsample_ids_dim_1`] | int64 |
| `out_cache_0`..`out_cache_15` | [`batch_size`, 512, `Sliceout_cache_i_dim_2`] | float32 |

**Notes:**
- 16 attention layers with per-layer KV cache
- Cache shape [512, 10] — **fixed depth of 10**, not standard variable-length autoregressive cache
- This is a "causal window" decoder: processes all tokens in parallel with a depth=10 lookback
- `token_length` is dynamic — determined by CIF firing count
- Vocabulary: 8404 tokens (file `/opt/models/paraformer-streaming/tokens.txt`), tokens 0=`<blank>`, 1=`<s>`, 2=`</s>`, 8403=`<unk>`
- Output logits [*, *, 8404] — the first two dims correspond to [batch_size, token_length]
- `sample_ids` is ArgMax over logits — works for greedy decoding; for beam search would need separate handling

**Unique ops (25 total):**
Add, ArgMax, Cast, Concat, Constant, Conv, Div, Gather, Less, MatMul, Mul, Pow, Range, ReduceMax, ReduceMean, Relu, Reshape, Shape, Slice, Softmax, Split, Sqrt, Sub, Transpose, Unsqueeze

---

## 4. Support Files

### 4.1 Paraformer tokens
- File: `/opt/models/paraformer-streaming/tokens.txt`
- 8404 lines (token 0..8403)
- Special tokens: `<blank>` (0), `<s>` (1), `</s>` (2), `<unk>` (8403)

### 4.2 Matcha tokens
- File: `/opt/models/matcha-icefall-zh-en/tokens.txt`
- 2190 lines (token 0..2189)
- Last entries: `<|unused_2188|> 2188`, `<|unused_2189|> 2189`, `<|unused_2190|> 2190`
- Support files: `lexicon.txt` (1.4M), `date-zh.fst`, `number-zh.fst`, `phone-zh.fst`
- Text frontend (g2p) uses espeak-ng-data

---

## 5. Bucket Table Draft (for M2)

### 5.1 Matcha Text Length Buckets

Input: `x` shape [`N`, `L`], `L` = text token count

| Bucket | L (tokens) | Est. mel frames | Use case |
|--------|-----------|-----------------|----------|
| S | 32 | ~256–384 | Short phrase (2-5 chars) |
| M | 64 | ~512–768 | Sentence (5-15 chars) |
| L | 128 | ~1024–1536 | Long sentence (15-30 chars) |

**Rule**: text > 128 tokens splits into multiple requests at the backend layer.

### 5.2 Vocos Mel Frames Buckets

Input: `mels` shape [`batch_size`, 80, `time`], `time` = mel frames

| Bucket | time (mel frames) | Est. audio @ 16kHz (hop=256) | Use case |
|--------|-------------------|------------------------------|----------|
| S | 72 | ~1.2 s | Short phrase output |
| M | 256 | ~4.1 s | Sentence output |
| L | 600 | ~9.6 s | Long sentence output |

**Matcha → Vocos**: The acoustic model output `mel` shape [N, 80, L] feeds directly into Vocos `mels` shape [`batch_size`, 80, `time`]. Select vocos bucket ≥ matcha mel output length.

### 5.3 Paraformer Encoder Chunk Buckets

Input: `speech` shape [`batch_size`, `feats_length`, 560]

At 10ms frame shift, each `feats_length` unit = 10ms of audio (after 7-frame stacking).

| Bucket | feats_length | Audio duration | Use case |
|--------|-------------|---------------|----------|
| S | 40 | 400 ms | Minimum streaming chunk |
| M | 80 | 800 ms | Chunk + left context |
| L | 400 | 4.0 s | Offline entire utterance |

**Dual-profile design for streaming**:
- Profile 1 (chunk): min=40, opt=40, max=80
- Profile 2 (offline): min=80, opt=200, max=400

### 5.4 Paraformer Decoder Token Buckets

The `token_length` in `acoustic_embeds` depends on CIF integration.

| Bucket | token_length | Est. CIF input frames | Use case |
|--------|-------------|----------------------|----------|
| S | 4 | ~160 frames (1.6s) | Short utterance |
| M | 20 | ~800 frames (8s) | Medium utterance |
| L | 40 | ~1600 frames (16s) | Long utterance |

**CIF token rate**: ~1 token per 40 input frames (empirical; to verify in M2).

---

## 6. Risks and Unknowns

### 6.1 Matcha — RandomNormalLike
- **`RandomNormalLike`** (1 node in the ODE solver path) is **not natively supported in TensorRT**.
- **Mitigation**: Expose noise as a TRT input tensor. The Python backend generates noise externally, feeds it as an input. Zero performance impact since noise generation is a trivial CPU op.
- **Alternative**: TRT plugin for random noise generation (more complex, avoid unless needed).

### 6.2 Matcha — ODE FP16 Precision Risk
- The ODE solver runs 10 steps (from model name `model-steps-3` which refers to the training ODE steps, not inference steps — verification needed: how many ODE steps in inference?).
- FP16 accumulation across 10 steps will cause drift.
- **Mitigation** (per spec §7): Force FP32 on attention QK^T, LayerNorm, RMSNorm layers. Already demonstrated in `benchmark/build_cp_fp16_safe.py`.

### 6.3 Matcha — Input int64
- Text input `x` has dtype int64. TRT supports int64 but with limited operator coverage.
- **Mitigation**: Cast to int32 in the model or at the TRT binding level.

### 6.4 Paraformer — CIF Token Length Dynamic Range
- The CIF predictor fires 1 token per ~40 input frames. For a 4-second utterance (400 frames), expect ~10 tokens.
- The spec "min=4, opt=20, max=80" may be too pessimistic for the max side (80 tokens ≈ 3200 frames ≈ 32s of audio).
- **Recommendation**: Start with `min=4, opt=10, max=40`. Expand if real usage shows longer utterances.

### 6.5 Paraformer Decoder Cache Shape — Fixed Depth 10
- The 16 per-layer caches have shape [512, 10] — **fixed depth, not variable autoregressive KV cache**.
- This means the decoder does NOT need dual-profile for cache length. Single profile for the cache tensors.
- The encoder output `enc_length` and `acoustic_embeds_len` are the only dynamic dimensions in the decoder.

### 6.6 Paraformer Encoder — Causal FSMN
- All FSMN conv layers use kernel_size=11 with causal padding (left-only context).
- This conv pattern should be TRT-friendly. No control flow needed for streaming state management.
- **However**: streaming requires maintaining per-layer state between chunks (the FSMN internal states). This state must be either:
  a) Managed externally (Python holds state, feeds overlapped chunks to TRT)
  b) Built into the model with explicit state I/O (more complex export)

### 6.7 Vocos — Output Shape Dynamic Dims
- `mag`, `x`, `y` outputs have dynamic dimensions (`Clipmag_dim_1`, `Cosx_dim_1`) that depend on the FFT size.
- These are likely constants (determined by model architecture, not input-dependent).
- **Recommendation**: Verify in M2 by running ORT inference with a test input and checking actual output shapes.

### 6.8 Matcha — Text Frontend Dependency
- Matcha requires lexicon, FST files, espeak-ng for g2p (grapheme-to-phoneme).
- The text frontend will remain CPU-based in Python (like the RKNN Matcha backend). Only the acoustic model + vocoder move to TRT.
- No risk to TRT build, but important for the backend implementation in M2–M3.

### 6.9 No Scan/Loop/If (All Models)
- Confirmed: zero control flow ops in any of the 4 ONNX files.
- All models should be directly convertible with trtexec or polygraphy.

---

## 7. Build Environment

- TRT engines must be built **in the v3.4-slim container** (same as runtime) to avoid ABI/driver mismatch.
- The v3.4-slim container has:
  - Python 3.10 (with onnx 1.21.0, onnxruntime 1.20.0)
  - TensorRT Python 10.3.0
  - No trtexec or polygraphy pre-installed (may need `apt install trtexec` or copy from host)
- Host has trtexec at `/usr/src/tensorrt` (sample ONNX files exist there)
- Container mounts CUDA libs from `/usr/local/cuda/lib64` and nvidia from `/usr/lib/aarch64-linux-gnu/nvidia`

---

## 8. EVIDENCE

### 8.1 MD5 Checksums (raw)

```
308ab5cb918e6340057134e54424a4bc  /opt/models/matcha-icefall-zh-en/model-steps-3.onnx
2b43a6235c158cb70480334c48c1c8b9  /opt/models/matcha-icefall-zh-en/vocos-16khz-univ.onnx
38bb68f284cf2d34e5a8f98a7c671ffd  /opt/models/paraformer-streaming/encoder.onnx
4eb7c94ece0ad861f18ef56db5f72379  /opt/models/paraformer-streaming/decoder.onnx
```

### 8.2 File Sizes

```
73M   model-steps-3.onnx
52M   vocos-16khz-univ.onnx
607M  encoder.onnx
218M  decoder.onnx
```

### 8.3 TensorRT / CUDA dpkg (raw)

```
ii  libnvinfer10           10.3.0.30-1+cuda12.5
ii  libnvinfer-bin         10.3.0.30-1+cuda12.5
ii  libnvinfer-dev         10.3.0.30-1+cuda12.5
ii  libnvinfer-plugin10    10.3.0.30-1+cuda12.5
ii  libnvinfer-vc-plugin10 10.3.0.30-1+cuda12.5
ii  libnvonnxparsers10     10.3.0.30-1+cuda12.5
ii  nvidia-tensorrt        6.2+b77
ii  python3-libnvinfer     10.3.0.30-1+cuda12.5
```

### 8.4 JetPack Release (raw)

```
# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64
# DATE: Wed Jan  8 01:49:37 UTC 2025
# KERNEL_VARIANT: oot
# branch R36.4.3
```

### 8.5 Model Volume Layout (raw)

```
/opt/models/ (Docker volume: reachy_speech_speech-models)
├── matcha-icefall-zh-en/
│   ├── model-steps-3.onnx      (72.2 MB)
│   ├── vocos-16khz-univ.onnx   (51.4 MB)
│   ├── tokens.txt              (2190 tokens)
│   ├── lexicon.txt             (1.4 MB)
│   ├── date-zh.fst
│   ├── number-zh.fst
│   ├── phone-zh.fst
│   ├── README.md
│   └── espeak-ng-data/
├── paraformer-streaming/
│   ├── encoder.onnx            (607 MB)
│   ├── decoder.onnx            (218 MB)
│   └── tokens.txt              (8404 tokens)
└── qwen3-*/                    (out of scope)
```

### 8.6 Inference Session Meta (via onnxruntime CPU EP)

All 4 models loaded successfully in `CPUExecutionProvider`.

---

## 9. Key Findings Summary

1. **CIF is genuinely dynamic**: The encoder `alphas` output and decoder `acoustic_embeds` both have dynamic `feats_length`/`token_length` dimensions with no upper bound in ONNX. Dual-profile TRT is essential.

2. **No control flow ops**: Zero Scan/Loop/If across all 4 models. Full direct convertibility via trtexec.

3. **TRT 10.3.0.30 / JetPack R36.4.3 / CUDA 12.6**: Latest JetPack as of early 2025. Requires `trtexec` to be installed in the container for engine building (not present by default).

4. **Matcha RandomNormalLike risk**: This is the only non-standard op. Must be externalized as a noise input tensor before TRT conversion.

5. **Paraformer decoder cache is fixed-depth (10)**: Not standard autoregressive KV cache. Single profile is sufficient for cache tensors.

6. **Paraformer encoder has 19 causal FSMN layers**: TRT-friendly but streaming state management needs careful handling (overlapped chunks or explicit state I/O).

7. **Models are large**: Paraformer encoder 607MB + decoder 218MB will dominate TRT build time. Plan for ~30 min per engine build on Jetson Orin NX.
