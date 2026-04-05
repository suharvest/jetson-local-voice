# Qwen3-ASR 0.6B — Jetson Orin NX 16GB Deployment Plan

## 1. Model Selection

### Target: Qwen3-ASR-0.6B

| Property | Value |
|----------|-------|
| Model | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) |
| Parameters | ~600M total (180M encoder + projector + Qwen3-0.6B decoder) |
| Weights (BF16 safetensors) | ~1.9 GB |
| ONNX FP16 (total) | ~1.0 GB |
| ONNX INT4 (total) | ~0.5 GB |
| Languages | 30 languages + 22 Chinese dialects (EN/ZH primary) |
| License | Apache 2.0 |
| Audio token rate | 12.5 Hz (8x downsampling from 128-dim Fbank) |
| Streaming | Yes (windowed encoder + sliding KV cache) |
| Reference TTFT | 92ms (vLLM on datacenter GPU) |

### Why 0.6B over 1.7B

- 1.7B is SOTA-competitive but too heavy for Orin NX alongside TTS (would consume ~4GB ONNX + TRT engines).
- 0.6B fits comfortably: ~1 GB ONNX FP16, ~1.5 GB TRT engines, leaving 10+ GB for TTS + OS.
- 0.6B accuracy is competitive among sub-1B ASR models. Sufficient for voice command and conversation transcription use cases.

### Why not keep SenseVoice (current ASR)

SenseVoice (current `asr_service.py`) is offline-only with no streaming support. Qwen3-ASR offers:
- Native streaming inference with windowed encoder
- Better multilingual accuracy (52 languages vs SenseVoice's limited set)
- Timestamp prediction support
- Same ONNX/TRT deployment pattern as our TTS — unified optimization strategy

---

## 2. Architecture Analysis

### Pipeline: Audio Encoder (AuT) + Projector + LLM Decoder (Qwen3)

```
Audio WAV
  │
  ▼
Fbank (128-dim, 100Hz)
  │
  ▼
┌─────────────────────────────────┐
│ AuT Encoder (180M params)       │
│  Conv2D stem: 3 layers          │
│    Conv2d(3x3, stride=2, pad=1) │
│    x3 → 128→64→32→16 freq bins  │
│    time: /8 → 12.5 Hz tokens    │
│  Sinusoidal pos embeddings      │
│  Transformer: 18 layers         │
│    d=896, bidirectional windowed │
│    attention (104-token windows) │
│  Final LayerNorm                │
│  Projection: proj1→GELU→proj2   │
│    896 → 1024 (decoder dim)     │
└─────────────────┬───────────────┘
                  │
                  ▼
┌─────────────────────────────────┐
│ Projector (linear)              │
│  Maps encoder output to LLM    │
│  embedding space                │
└─────────────────┬───────────────┘
                  │ Audio tokens replace <audio> placeholders
                  ▼
┌─────────────────────────────────┐
│ Qwen3 Decoder (0.6B)           │
│  28 layers, d=1024              │
│  16 Q-heads / 8 KV-heads (GQA) │
│  SwiGLU FFN, MRoPE             │
│  Self-attention only            │
│  Autoregressive text generation │
└─────────────────┬───────────────┘
                  │
                  ▼
              Text tokens → Tokenizer decode → Transcription
```

### Key architectural properties

1. **Encoder is chunked**: Mel frames split into 100-frame windows (8s audio each). Conv2D and attention run per-chunk. This is critical for streaming — chunks can be processed incrementally.
2. **Decoder is autoregressive**: Standard Qwen3 LLM generating text tokens. KV cache grows with output length.
3. **No cross-attention**: Encoder output replaces audio token embeddings in the decoder input. Decoder uses self-attention only — simpler ONNX graph.
4. **Streaming capability**: Encoder window eviction (keep last 4 windows ~32s), decoder prefix capping (~150 tokens). This bounds memory for indefinite streaming.

---

## 3. ONNX Export Strategy

### Proven approach: andrewleech/qwen3-asr-onnx

An existing community export ([andrewleech/qwen3-asr-onnx](https://github.com/andrewleech/qwen3-asr-onnx)) already splits the model into the exact components we need. We can either use these directly or reproduce the export with our own script.

### Component breakdown

| Component | File | Size (FP16) | Purpose |
|-----------|------|-------------|---------|
| **encoder** | `encoder.onnx` | ~360 MB | Conv2D stem + 18-layer transformer, processes mel chunks |
| **decoder_init** | `decoder_init.onnx` + `decoder_weights.data` | ~2 MB graph + 598 MB weights | Prefill: takes input_ids, has embedding table built-in |
| **decoder_step** | `decoder_step.onnx` (shares weights) | ~2 MB graph (shared weights) | Autoregressive: takes pre-looked-up input_embeds |
| **embed_tokens** | `embed_tokens.bin` | ~FP16 embed table | Token embedding lookup for decoder_step |
| **config.json** | `config.json` | tiny | Model config |
| **tokenizer.json** | `tokenizer.json` | tiny | Qwen3 tokenizer |

### Export lessons from TTS (directly applicable)

| TTS Lesson | ASR Application |
|------------|-----------------|
| TorchScript (`dynamo=False`) for attention models | Same — Qwen3 decoder uses identical attention |
| External data format for >2GB models | Encoder (360MB) fits inline; decoder weights need external data |
| `attention_mask` gets optimized away in trace | Same risk — make mask input optional in inference code |
| Stacked weights for dynamic indexing | Not needed — ASR decoder has single lm_head |
| Vocoder needs dynamo | Not applicable — ASR has no vocoder |

### Export script plan

Write `benchmark/export_qwen3_asr.py`:

```
1. Load Qwen3-ASR-0.6B from HuggingFace
2. Extract encoder (AuT + projector) → trace with dummy mel input
   - Input: mel_spectrogram [1, 128, T] (T dynamic)
   - Output: encoder_features [1, T/8, 1024]
3. Extract decoder_init (prefill) → trace with input_ids + encoder features
   - Inputs: input_ids [1, S], encoder_output [1, T/8, 1024]
   - Outputs: logits [1, S, V], past_key_values [28 * 2 tensors]
4. Extract decoder_step (autoregressive) → trace with single-token input
   - Inputs: input_embeds [1, 1, 1024], past_key_values
   - Outputs: logits [1, 1, V], new_past_key_values
5. Save embedding table as embed_tokens.bin (FP16)
```

Alternatively, use andrewleech's pre-exported ONNX files directly from HuggingFace to skip this step entirely.

---

## 4. TensorRT Compilation Plan

### Hot path vs cold path (same pattern as TTS)

```
Hot path (autoregressive decode, runs per token):
  decoder_step  → TRT native C++ API → FP16 engine
                   GPU-resident KV cache (double-buffered)

Cold path (runs once per utterance):
  encoder       → ORT CUDA EP (chunked, runs once per audio chunk)
  decoder_init  → ORT CUDA EP (prefill, runs once)
```

### Why this split

| Component | TRT feasibility | Rationale |
|-----------|----------------|-----------|
| **decoder_step** | Excellent | Same Qwen3 architecture as TTS talker_decode. We already have TRT engine patterns for 28-layer Qwen3 with GQA. FP16 safe (ASR output logits over vocab, not continuous values — no BF16 needed). |
| **encoder** | Risky | Conv2D + windowed bidirectional attention. Windowed attention with masking may cause TRT compilation issues (like TTS vocoder). Start with ORT CUDA EP. |
| **decoder_init** | Not worth it | Runs once per utterance. ORT CUDA EP is fine. |

### TRT engine build commands (on Jetson or WSL2)

```bash
# decoder_step FP16 — the hot path
trtexec --onnx=decoder_step.onnx \
  --saveEngine=decoder_step_fp16.engine --fp16 \
  --memPoolSize=workspace:2048MiB \
  --minShapes=input_embeds:1x1x1024,past_key_0:1x8x1x128,... \
  --optShapes=input_embeds:1x1x1024,past_key_0:1x8x50x128,... \
  --maxShapes=input_embeds:1x1x1024,past_key_0:1x8x500x128,...
```

### BF16 considerations

Unlike TTS (where CP had intermediate values >65504 causing FP16 NaN), ASR decoder logits are over a vocabulary — typically well-bounded. FP16 should be safe. However, if NaN appears during validation, fall back to BF16 (same `BuilderFlag.BF16` pattern as TTS CP engine).

### GPU-resident KV cache

Directly reuse the TTS double-buffer pattern:
- 28 layers x 2 (key+value) = 56 KV tensors on GPU
- Per-token: copy only input_embeds (4KB) to GPU, read logits (vocab_size * 4 bytes) back
- Pointer swap between A/B buffer sets each step
- For ASR, KV cache is smaller than TTS (output is text tokens, typically <200 tokens for a 30s utterance)

### Memory budget on Jetson Orin NX 16GB

| Component | GPU Memory |
|-----------|------------|
| Encoder ONNX (ORT CUDA EP) | ~400 MB |
| Decoder init ONNX (ORT CUDA EP) | ~600 MB |
| Decoder step TRT engine (FP16) | ~400 MB |
| KV cache (500 tokens max) | ~50 MB |
| **ASR total** | **~1.5 GB** |
| TTS (existing Qwen3-TTS) | ~2.5 GB |
| OS + CUDA runtime | ~2 GB |
| **Total** | **~6 GB** |
| **Headroom** | **~10 GB** |

Plenty of room. Could even run 1.7B if accuracy demands it (adds ~2 GB).

---

## 5. Integration Plan

### New ASR backend abstraction (mirrors TTS pattern)

Create `app/asr_backend.py` — abstract base class:

```python
class ASRCapability(str, Enum):
    OFFLINE = "offline"           # Transcribe complete audio files
    STREAMING = "streaming"       # Process audio chunks incrementally
    TIMESTAMPS = "timestamps"     # Word/character-level timestamps
    LANGUAGE_ID = "language_id"   # Auto-detect language
    MULTI_LANGUAGE = "multi_language"

class ASRBackend(ABC):
    name: str
    capabilities: set[ASRCapability]
    sample_rate: int  # Expected input sample rate (16000)

    def preload(self) -> None: ...
    def is_ready(self) -> bool: ...
    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult: ...
    def create_stream(self) -> ASRStream: ...  # For STREAMING capability
```

### Backend implementations

| Backend | File | Engine | Capabilities |
|---------|------|--------|-------------|
| `sensevoice` | `backends/sensevoice.py` | sherpa-onnx (existing) | OFFLINE, LANGUAGE_ID |
| `qwen3_asr_ort` | `backends/qwen3_asr_ort.py` | ORT CUDA EP | OFFLINE, STREAMING, TIMESTAMPS, LANGUAGE_ID, MULTI_LANGUAGE |
| `qwen3_asr_trt` | `backends/qwen3_asr_trt.py` | ORT + TRT native (hybrid) | OFFLINE, STREAMING, TIMESTAMPS, LANGUAGE_ID, MULTI_LANGUAGE |

### Streaming inference pipeline

```
WebSocket /ws/asr/stream
  │
  Client sends 16kHz PCM chunks (e.g. 100ms = 1600 samples)
  │
  ▼
┌──────────────────────────────┐
│ Audio buffer (accumulate to  │
│ 8s = 100 mel frames = 1      │
│ encoder chunk)               │
├──────────────────────────────┤
│ When buffer full:            │
│  1. Compute Fbank features   │
│  2. Run encoder on chunk     │
│  3. Append to decoder context│
│  4. Run decoder (continue    │
│     from previous KV cache)  │
│  5. Stream partial text back │
├──────────────────────────────┤
│ Window eviction:             │
│  Keep last 4 encoder windows │
│  (~32s context)              │
│  Cap decoder prefix at ~150  │
│  tokens                      │
└──────────────────────────────┘
```

### FastAPI endpoints (additions to main.py)

```
POST /asr/transcribe          — Offline: upload WAV, get full transcription
POST /asr/transcribe/stream   — SSE: upload WAV, stream partial results
WS   /ws/asr/stream           — WebSocket streaming (existing pattern)
GET  /asr/capabilities        — Backend capabilities discovery
```

### Configuration (environment variables)

```bash
ASR_BACKEND=qwen3_asr_trt          # or sensevoice, qwen3_asr_ort
QWEN3_ASR_MODEL_BASE=/opt/models/qwen3-asr
QWEN3_ASR_DECODER_ENGINE=/opt/models/qwen3-asr/engines/decoder_step_fp16.engine
QWEN3_ASR_PROVIDER=cuda            # cuda or cpu
```

---

## 6. Implementation Phases

### Phase 1: Validate ONNX (1-2 days)

1. Download [andrewleech/qwen3-asr-0.6b-onnx](https://huggingface.co/andrewleech/qwen3-asr-0.6b-onnx) FP16 variant
2. Write `benchmark/test_asr_onnx.py` — run encoder + decoder_init + decoder_step loop with ORT CPU
3. Verify transcription accuracy on test audio (CN + EN)
4. If community ONNX has issues, write our own `benchmark/export_qwen3_asr.py`

### Phase 2: ORT CUDA EP on Jetson (1-2 days)

1. Transfer ONNX files to Jetson Orin NX
2. Run full pipeline with ORT CUDA EP (all components)
3. Benchmark: measure encoder latency, prefill latency, per-token decode latency
4. Measure RTF for 10s and 30s audio clips
5. Expected: RTF < 0.5 (ASR is much lighter than TTS per-token)

### Phase 3: TRT native decoder (2-3 days)

1. Build decoder_step FP16 engine with trtexec on Jetson
2. Implement GPU-resident KV cache (reuse TTS pattern)
3. Hybrid pipeline: encoder (ORT) + decoder_init (ORT) + decoder_step (TRT)
4. Benchmark: expect 30-50% decode speedup over pure ORT
5. Validate no NaN in FP16 (test on long utterances)

### Phase 4: Streaming integration (2-3 days)

1. Create `app/asr_backend.py` base class
2. Implement `backends/qwen3_asr_trt.py`
3. Refactor existing `asr_service.py` into `backends/sensevoice.py`
4. Add WebSocket streaming endpoint
5. Add capability discovery endpoint

### Phase 5: Production hardening (1-2 days)

1. Add to `model_downloader.py` — auto-download Qwen3-ASR ONNX + TRT engines
2. Update `Dockerfile.slim` — include Qwen3-ASR in multi-stage build
3. Update `docker-compose.yml` — model volume mounts
4. Warmup and health check integration
5. Add to `/health` endpoint capability reporting

---

## 7. Expected Performance on Jetson Orin NX 16GB

### Offline transcription (10s audio)

| Component | ORT CUDA EP | TRT Hybrid |
|-----------|-------------|------------|
| Fbank extraction | ~5ms | ~5ms |
| Encoder (2 chunks) | ~50ms | ~50ms (ORT) |
| Decoder prefill | ~30ms | ~30ms (ORT) |
| Decoder autoregressive (~40 tokens) | ~200ms | ~100ms |
| **Total** | **~285ms** | **~185ms** |
| **RTF** | **~0.03** | **~0.02** |

### Streaming (continuous)

| Metric | Expected |
|--------|----------|
| Time-to-first-token | ~150ms (encoder chunk + first decode) |
| Per-token latency | ~5ms (TRT) / ~8ms (ORT) |
| Max sustained audio | Indefinite (window eviction) |
| GPU memory (steady state) | ~1.5 GB |

### Comparison with current SenseVoice ASR

| Metric | SenseVoice | Qwen3-ASR-0.6B |
|--------|-----------|-----------------|
| Streaming | No | Yes |
| Languages | Limited | 52 languages |
| Timestamps | No | Yes |
| Model size (ONNX) | ~200 MB | ~1.0 GB |
| Latency (10s audio) | ~100ms | ~185ms |
| Accuracy (CN+EN) | Good | Better (trained on 20M hours) |

SenseVoice remains lighter and faster for simple offline transcription. Qwen3-ASR adds streaming and multilingual capabilities. Both can coexist as selectable backends.

---

## 8. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Encoder windowed attention breaks TRT | High | Keep encoder on ORT CUDA EP (cold path). Only TRT the decoder. |
| FP16 NaN in decoder | Low | ASR logits are over vocab (bounded). If happens, use BF16 (proven pattern). |
| Community ONNX export has bugs | Medium | Validate first. Fall back to our own export script. |
| Streaming KV cache memory growth | Low | Window eviction caps at ~32s encoder + ~150 decoder tokens. |
| ORT version mismatch on Jetson | Medium | Pin ORT 1.20.0 (same as TTS). Test ONNX opset compatibility. |
| Disk space on Jetson | Low (Orin NX has more than Orin Nano) | ONNX FP16 ~1GB + TRT engine ~400MB = ~1.5GB total. |

---

## 9. References

- [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) — Official model card
- [Qwen3-ASR Technical Report (arxiv 2601.21337)](https://arxiv.org/abs/2601.21337) — Architecture details
- [andrewleech/qwen3-asr-onnx](https://github.com/andrewleech/qwen3-asr-onnx) — Community ONNX export (encoder/decoder split, INT4/FP16)
- [andrewleech/qwen3-asr-0.6b-onnx](https://huggingface.co/andrewleech/qwen3-asr-0.6b-onnx) — Pre-exported ONNX files
- [antirez/qwen-asr](https://github.com/antirez/qwen-asr) — C inference reference (architecture documentation)
- [Qwen3-ASR vLLM Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html) — Official vLLM integration
- [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — Official repo with streaming examples
- [Jetson Orin NX on NVIDIA Forums](https://forums.developer.nvidia.com/t/how-to-use-qwen3-asr-0-6b-on-jetson-orin-nano/361835) — Community discussion on Jetson deployment
- [Daumee/Qwen3-ASR-0.6B-ONNX-CPU](https://huggingface.co/Daumee/Qwen3-ASR-0.6B-ONNX-CPU) — Alternative ONNX export
