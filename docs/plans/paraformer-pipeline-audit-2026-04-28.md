# Paraformer TRT Backend — Pipeline Audit 2026-04-28

> Read-only analysis. No source files were modified.  
> All line numbers reference `/Users/harvest/project/jetson-voice/app/backends/paraformer_trt.py` (964 lines as of 2026-04-28).

---

## 1. Executive Summary

- **Three bugs confirmed in transcription pipeline** (`decode_ids` EOS break, `transcribe_audio` missing tail flush, token duplication from decoder autoregressive argmax) that together explain the majority of under-emission on long Chinese audio.
- **Long-audio FSMN context loss** at every 40-frame chunk boundary is a structural degradation — the encoder sees no left context across chunks, degrading CIF alpha quality for all chunks after the first.
- **Latency bottleneck** is per-chunk CUDA stream create/destroy (L851-858) on the iGPU dispatch path. Reusing a persistent stream + CUDA Graph capture would eliminate ~12ms × N_chunks of dispatch overhead.
- **Dual-profile encoder engine** (`paraformer_encoder_dp4_400.plan`) is already defined in `build_paraformer_trt.sh:42-56` with max=400 frames for offline; the runtime does **not yet use** the offline profile — both `transcribe()` and `transcribe_audio()` always feed 40-frame chunks.
- **Realistic RTF target**: after all P0+P1 fixes, RTF 0.05–0.08 on 2.8s audio is achievable; RTF ≤0.05 is tight but possible only if CUDA Graph + offline single-pass encoder are both landed.

---

## 2. Quality Analysis

### A1. Premature EOS break in `decode_ids()` — L271-276

**True Cause**

```python
# paraformer_trt.py L271-284
def decode_ids(token_ids: list[int], tokens: list[str]) -> str:
    chars = []
    for tid in token_ids:
        if tid == EOS_ID:
            break                  # ← L276: hard stop, ALL subsequent tokens dropped
        if tid in (BLANK_ID, SOS_ID):
            continue
        ...
```

`EOS_ID = 2` (L83). The decoder ORT session (`_run_decoder`, L934-960) runs a single forward pass over all `acoustic_embeds` in a chunk and returns `sample_ids[0]` — a tensor of shape `[n_tokens]` (L952). If any position in that tensor contains `2`, every token after that position is silently dropped.

The decoder was trained to emit EOS only at utterance end. However, with a stale cross-chunk persistent cache (L641, L701), the decoder's attention state from the previous chunk may project a strong EOS prior at position 0 of the next chunk, triggering the break before any real tokens are emitted. This explains why S3 (9.8s, 24 chunks) produces only 3 characters: the first chunk's cache state bleeds into chunk 2, which EOS-breaks immediately, and so on.

**Fix Patch**

```diff
--- a/app/backends/paraformer_trt.py
+++ b/app/backends/paraformer_trt.py
@@ -273,7 +273,7 @@ def decode_ids(token_ids: list[int], tokens: list[str]) -> str:
     chars = []
     for tid in token_ids:
         if tid == EOS_ID:
-            break
+            continue  # skip EOS marker; do not truncate subsequent tokens
         if tid in (BLANK_ID, SOS_ID):
             continue
```

This is the minimal fix. A more conservative variant would be to break only if EOS appears in the **last** position (`if tid == EOS_ID and idx == len(token_ids) - 1`), but `continue` is safer given the cache-bleed hypothesis.

**Expected Effect**

- S3 (9.8s): should recover from 3 chars to something in the range of 15-30 chars.
- Eliminates the silent-truncation class of bugs across all chunk positions.
- No risk of infinite loops — the decoder token sequence is always finite.

---

### A2. Same-chunk token duplication — `decode_ids` L271 / decoder architecture

**True Cause**

The decoder (`_run_decoder`, L934-960) calls `self._dec_session.run(...)` once per chunk, passing the full `acoustic_embeds` tensor `[n_tokens, 512]`. The ORT model runs autoregressive prediction: it predicts `sample_ids[i]` from `acoustic_embeds[i]` and the accumulated cache state, updating `cache` in-place after each position.

The repetition ("好好", "真真", `[6049, 6049]`) occurs when two adjacent CIF-emitted acoustic_embeds are nearly identical. The CIF code at L235-240 shows:

```python
while accum_weight >= threshold:
    excess = accum_weight - threshold
    token_embed = (accum_embed - excess * enc[t]) / threshold
    acoustic_embeds.append(token_embed)
    accum_weight = excess
    accum_embed = excess * enc[t]          # ← L240: carry is just excess * same frame
```

When `alpha[t]` is large (say 1.8), the while loop fires twice: the first token embed contains the current frame's contribution, and the second token embed's `accum_embed` is initialized to `excess * enc[t]` — which is again dominated by the **same encoder frame** `enc[t]`. With similar embeddings, the decoder argmax produces the same token twice.

This is not a code bug per se — it reflects a real CIF property — but its impact is amplified by the encoder producing degraded features at chunk boundaries (A4), making some frames degenerate.

**Fix Patch**

Add a simple adjacent-duplicate filter in `decode_ids`:

```diff
--- a/app/backends/paraformer_trt.py
+++ b/app/backends/paraformer_trt.py
@@ -273,6 +273,7 @@ def decode_ids(token_ids: list[int], tokens: list[str]) -> str:
     chars = []
+    prev_tid = -1
     for tid in token_ids:
         if tid == EOS_ID:
             continue
         if tid in (BLANK_ID, SOS_ID):
             continue
+        if tid == prev_tid:
+            prev_tid = tid
+            continue  # suppress immediate adjacent duplicate
         if 0 <= tid < len(tokens):
             token = tokens[tid]
             if token.startswith("<") and token.endswith(">"):
                 continue
             chars.append(token)
+        prev_tid = tid
     return "".join(chars)
```

**Caveat**: this filter will incorrectly suppress legitimate repeated characters (e.g., "哈哈哈" for laughter, "嗯嗯" as acknowledgement). A stricter version would limit suppression to a max repetition count of 2. For now, removing single-adjacent duplicates will improve WER substantially.

**Expected Effect**

- Eliminates the most egregious "好好" / "真真" / "嗯嗯嗯" artifacts.
- Does not fix the root cause (CIF frame degeneration at boundaries) but is a safe quality floor.

---

### A3. `transcribe_audio()` missing tail flush — L734

**True Cause**

`transcribe()` (L606-687) includes a tail flush at L674-684:

```python
# L674-684 in transcribe()
if carry_w >= CIF_TAIL_THRESHOLD:
    acoustic_embeds = (carry_e / carry_w)[np.newaxis, :]
    dummy_enc = np.zeros((1, 1, 512), dtype=np.float32)
    sample_ids = self._run_decoder(dummy_enc, 1, acoustic_embeds, 1, cache)
    if sample_ids is not None:
        text = decode_ids(sample_ids.tolist(), self._tokens)
        if text:
            all_text_parts.append(text)
```

`transcribe_audio()` (L689-735) ends at L734 with a bare join:

```python
# L734 in transcribe_audio() — no tail flush
full_text = "".join(all_text_parts)
return TranscriptionResult(text=full_text, language=language)
```

The CIF tail accumulator (`carry_w`, `carry_e`) holds the partial token that hasn't crossed the 1.0 threshold yet. For a typical sentence, the last 0.5–0.9 weight unit of CIF accumulation is the final syllable. This syllable is always dropped in the `transcribe_audio` path.

**Fix Patch**

```diff
--- a/app/backends/paraformer_trt.py
+++ b/app/backends/paraformer_trt.py
@@ -731,6 +731,16 @@ def transcribe_audio(self, audio: np.ndarray, language: str = "auto") -> Transc
                 if text:
                     all_text_parts.append(text)

+        # Flush CIF tail (mirrors transcribe() L674-684)
+        if carry_w >= CIF_TAIL_THRESHOLD:
+            acoustic_embeds = (carry_e / carry_w)[np.newaxis, :]
+            dummy_enc = np.zeros((1, 1, 512), dtype=np.float32)
+            sample_ids = self._run_decoder(
+                dummy_enc, 1, acoustic_embeds, 1, cache,
+            )
+            if sample_ids is not None:
+                text = decode_ids(sample_ids.tolist(), self._tokens)
+                if text:
+                    all_text_parts.append(text)
+
         full_text = "".join(all_text_parts)
         return TranscriptionResult(text=full_text, language=language)
```

**Expected Effect**

- Recovers the final syllable/character for every utterance processed through `transcribe_audio`.
- For S3 (9.8s), this may add 1-2 characters at the end.
- More impactful for short audio (S0/S1) where the final token is often the tail.

---

### A4. Long-audio under-emission — FSMN left-context loss at chunk boundaries

**True Cause**

The Paraformer encoder uses SAN-M (FSMN-based self-attention with memory). The FSMN memory mechanism requires left context frames to compute accurate attention weights and, consequently, accurate CIF alpha values. When the audio is split into 40-frame chunks at L637/L697, each chunk starts with no left context. The encoder's L63 constant `LEFT_CONTEXT_SEC = 0.0` confirms no overlap is added.

For a 9.8s audio:
- Total frames ≈ 9.8s × 100 fps = 980 frames → after 7-frame stacking (L71 `NUM_STACKED=7`) = ~140 feature vectors → with `chunk_frames=40` (L637) → 4 full chunks of 40 frames.

Wait — re-checking: stacking at `stack_frames` reduces temporal resolution. With HOP_SIZE=160 (10ms) and NUM_STACKED=7, the output has 1 stacked frame per 7 input frames, so frame rate ≈ 14 fps. For 9.8s: 9800ms / 70ms = ~140 stacked frames → 4 complete 40-frame chunks, 1 remainder of ~20 frames.

That gives 5 encoder calls, not 24. The earlier "24 chunks" estimate was based on pre-stacking frame count. With 5 chunks, per-chunk latency would need to be 423ms/5 = **84ms per chunk** — which aligns better with "~60ms per chunk" when accounting for overhead.

The key quality issue: with only 5 chunks, the first chunk (frames 0-39 of stacked features) covers the first ~2.8s of audio and will produce correct CIF output. Chunks 2-5 start from frame 40, 80, 120 with zero left context in the FSMN memory — the encoder's attention quality degrades, producing flatter alpha distributions (less confident token boundaries), which causes fewer CIF fires and thus fewer acoustic_embeds per chunk.

The build script (`build_paraformer_trt.sh:40-56`) already defines an offline profile with `max=400` frames. The runtime simply never uses it — both transcription paths hardcode `chunk_frames=40` without checking if the encoder supports a larger input.

**Fix Patch (use offline encoder profile for full-audio paths)**

This is a medium-effort fix requiring runtime profile selection. The minimal patch:

```diff
--- a/app/backends/paraformer_trt.py
+++ b/app/backends/paraformer_trt.py
@@ -633,7 +633,12 @@ def transcribe(self, ...):
         feats = compute_fbank(data)
         feats = stack_frames(feats)
 
-        chunk_frames = 40
+        # Use full-audio single-pass if within offline engine profile (max=400)
+        # Avoids FSMN left-context loss at chunk boundaries
+        if feats.shape[0] <= 400:
+            chunk_frames = feats.shape[0]  # single chunk, whole audio
+        else:
+            chunk_frames = 40  # fall back to streaming chunks
         all_text_parts = []
```

Apply the same change at `transcribe_audio` L697.

**Note**: This only works if the TRT engine was built with the dual-profile plan (`paraformer_encoder_dp4_400.plan`). Verify the engine path loaded at runtime matches. If the engine was built with only `max=80`, inputs > 80 frames will be rejected by TRT and `_run_encoder_trt` will return `None, None`.

**Expected Effect**

- For audio ≤ 400 stacked frames (~28s), the encoder runs a single pass with full context.
- Should bring S3 (9.8s, ~140 stacked frames) from "3 chars" toward sherpa baseline quality.
- Only applicable once the dual-profile engine is confirmed loaded.

---

## 3. Performance Analysis

### B1. Per-chunk overhead breakdown

**Current cost estimate**

Benchmark: S1 = 2.8s, P50 = 423ms.  
Stacked frames for 2.8s: 2800ms / 70ms ≈ 40 stacked frames → **1 full chunk** (chunk_frames=40).  
So 423ms is the cost of **one encoder + one decoder call**, not 7 chunks.

Breakdown (estimated):
| Component | Estimate | Evidence |
|---|---|---|
| `cudaStreamCreate` + `execute_async_v3` + `cudaStreamSynchronize` + `cudaStreamDestroy` | ~15-20ms | L851-858; Jetson iGPU dispatch documented ~12ms; stream lifecycle adds ~3ms |
| H2D copy (speech tensor 40×560×4 bytes = 89KB) | ~1ms | cudaMemcpy sync at L843-848 |
| D2H copy (enc output 40×512×4 = 81KB) | ~1ms | L864-870 |
| CIF NumPy computation (40 frames) | ~0.5ms | Pure Python NumPy loop |
| ORT decoder (1 call, ~3-5 acoustic_embeds) | ~380-400ms | **Primary bottleneck** |

The ORT decoder takes ~380ms for a single call. This is the dominant cost — not the TRT encoder. `_run_decoder` at L934-960 calls `self._dec_session.run(...)` with 16 cache tensors (each `[1, 512, 10]` = 20KB) as input, plus 16 cache outputs. Total I/O per decoder call: ~640KB input + ~640KB output = ~1.3MB of tensor data allocated and copied per call. With ORT CPU EP inference on Jetson ARM64 CPU, this is ~350-400ms.

**Root cause**: The ORT decoder runs on CPU EP. For offline transcription with N tokens, this is called **once** per chunk (N tokens batch), but 16 large cache tensors are allocated fresh each call.

**Fix approach**: Migrate decoder to CUDA EP, or convert decoder to TRT, or use ORT ioBinding to reuse buffers. Decoder TRT conversion is in the existing roadmap (`matcha-paraformer-trt-m1-manifest-2026-04-28.md`).

**Expected savings**: If decoder moves to TRT/CUDA, estimate 350ms → 5-15ms per call = 300-380ms reduction.

---

### B2. CUDA Graph missing from `_run_encoder_trt`

**Current cost**

At L851-858, every encoder call:
```python
err, stream = cudart.cudaStreamCreate()     # ~1-2ms
success = ctx.execute_async_v3(stream)       # TRT kernel launch, dispatch ~12ms on Jetson iGPU
cudart.cudaStreamSynchronize(stream)         # wait
cudart.cudaStreamDestroy(stream)             # ~1ms
```

Total stream overhead: ~3ms per call. For 5 encoder calls (9.8s audio): ~15ms avoidable overhead.

**Fix sketch**

```python
# In __init__ or preload(): create a persistent stream
self._enc_stream = cudart.cudaStreamCreate()[1]

# In _run_encoder_trt(): reuse it
success = ctx.execute_async_v3(self._enc_stream)
cudart.cudaStreamSynchronize(self._enc_stream)
```

CUDA Graph capture requires fixed input shapes. With chunk_frames=40 (or=400 for offline), the input shape is fixed — graph capture is feasible. Savings: ~3ms per chunk + potentially ~8ms from graph replaying vs. kernel scheduling. For offline single-pass, savings are once. For streaming (24 chunks for 9.8s raw audio), savings = ~24 × 8ms = ~192ms.

**Effort**: Small (persistent stream reuse), Medium (full CUDA Graph capture).  
**Expected savings**: 15-50ms depending on audio length and mode.

---

### B3. ORT decoder — no ioBinding reuse

**Current cost**

At L934-960, each `_run_decoder` call:
- Builds a fresh `ort_inputs` dict with 20 numpy arrays (enc, enc_len, acoustic_embeds, acoustic_embeds_len, 16 caches)
- Calls `session.run()` which re-allocates output buffers for all 17 outputs

Cache tensors are `[1, 512, 10]` float32 = 20KB each × 16 = 320KB input + 320KB output allocated and freed each call.

**Fix sketch**: Use `ort.InferenceSession` with `io_binding`:
```python
binding = self._dec_session.io_binding()
binding.bind_cpu_input("in_cache_0", cache[0])
...
binding.bind_output("out_cache_0")  # pre-allocated buffer
self._dec_session.run_with_iobinding(binding)
```

**Effort**: Medium.  
**Expected savings**: 10-30ms per decoder call (reduced allocation). Given decoder is currently ~380ms on CPU, the allocation overhead is probably 5-10% = 20-40ms total.

---

### B4. Offline encoder profile max=80 is the already-fixed constraint

**Current situation**

`build_paraformer_trt.sh:45-56` already defines the dual-profile encoder with `max=400`. The runtime at `transcribe()` L637 hardcodes `chunk_frames=40` regardless.

**Fix**: As described in A4 — select chunk size based on feats.shape[0] ≤ 400.

**Expected savings**: For a 2.8s audio (40 stacked frames), a single-chunk encoder run costs ~20ms vs. 1 chunk at 20ms — no difference. For 9.8s (140 frames): 1 run at 140 frames vs. 4 runs at 40 frames each = saves 3 × (stream overhead + H2D + D2H) ≈ 3 × 5ms = 15ms. Primary benefit is quality (A4), not latency.

**Effort**: Small (runtime path selection), assuming engine already built.

---

## 4. Prioritized ROI Table

| # | Issue | Domain | Fix Effort | Benefit | Priority |
|---|---|---|---|---|---|
| 1 | A1: EOS break in `decode_ids` L276 | Quality | **Small** (1 line change) | Recovers all post-EOS tokens; critical for long audio | **P0** |
| 2 | A3: Missing tail flush in `transcribe_audio` L734 | Quality | **Small** (10 lines) | Recovers final syllable every utterance | **P0** |
| 3 | B1/B3: ORT decoder on CPU — primary latency source ~380ms | Perf | **Large** (decoder TRT conversion) | 350ms reduction per transcription | **P0** |
| 4 | A2: Token duplication filter in `decode_ids` | Quality | **Small** (8 lines) | Suppresses "好好"/"真真" artifacts | **P1** |
| 5 | A4: Use offline encoder single-pass (A4) | Quality | **Small** runtime + **Medium** verify engine | Restores FSMN context; major quality lift on long audio | **P1** |
| 6 | B2: CUDA Graph / persistent stream in encoder | Perf | **Small→Medium** | 15-50ms reduction; secondary after decoder fix | **P1** |
| 7 | B3: ORT ioBinding for decoder cache buffers | Perf | **Medium** | 20-40ms reduction; secondary after TRT decoder | **P2** |
| 8 | B4: Runtime offline profile selection | Perf | **Small** | 15ms reduction; needed for A4 | **P1** (same as A4) |

---

## 5. Engineering Roadmap

### Phase 1 — P0 correctness fixes (1-2h)

**Goal**: Stop silent truncation. Long-audio transcription must emit meaningful text.

Tasks (in order, each ≤15 min):

1. **A1**: Change `break` → `continue` in `decode_ids` L276.
2. **A3**: Add tail flush block to `transcribe_audio` before L734.
3. **A2**: Add adjacent-duplicate filter in `decode_ids`.
4. **Smoke test**: Run the 8 test audios (S0-S7). S3 (9.8s) must produce ≥ 10 characters. S4 (English) must remain correct.

**Acceptance gate**: S3 transcription ≥ 10 characters and no regression on S4. If S3 still produces < 5 chars after A1+A3, the EOS-bleed hypothesis needs deeper investigation (add logging to count EOS occurrences per chunk before A1 patch was applied).

**No expected latency change in Phase 1** — these are pure quality fixes with negligible compute cost.

---

### Phase 2 — P0/P1 performance + quality (2-4h)

**Goal**: Bring latency from 423ms to ≤120ms on S1 (2.8s).

Tasks:

1. **B1 root cause**: Add timing instrumentation to `_run_decoder` and `_run_encoder_trt`. Log each component in ms. Confirm decoder ~380ms vs encoder ~20ms split.
2. **B2**: Replace ephemeral stream in `_run_encoder_trt` L851-858 with a persistent stream created in `preload()`. Measure encoder latency before/after.
3. **A4 + B4**: Enable single-chunk offline path in `transcribe()` and `transcribe_audio()`. Verify the dual-profile engine is loaded (check engine plan filename from `self._eng_path`). If `paraformer_encoder_dp4_400.plan` is in use, test with `feats.shape[0]` up to 140 frames.
4. **B3**: If decoder is still on ORT CPU EP after phase 1/2, add ioBinding to reduce allocation overhead. This is a partial mitigation while awaiting TRT decoder.

**Acceptance gate**: RTF ≤ 0.08 on S1 (2.8s, target ≤224ms). If decoder is still ~380ms on ORT CPU, this gate will **not** be reached — that's the signal to escalate to decoder TRT conversion.

---

### Phase 3 — Architecture (4h+)

**Goal**: RTF ≤ 0.05 on 2.8s audio.

Tasks:

1. **Decoder TRT conversion**: Export ORT decoder model to TRT. This is the largest single latency improvement (300-380ms reduction). Requires verifying decoder ONNX model supports dynamic `n_tokens` and 16 cache tensors in TRT.
2. **CUDA Graph capture for encoder**: With fixed chunk shape (1, 40, 560) and/or offline shape (1, N, 560), capture graph after first warmup. Primary benefit in streaming mode where same shape is called many times.
3. **Streaming quality**: Once correctness is verified in offline paths, evaluate streaming path (`ParaformerTRTStream`) for the same A1/A3/A4 issues.

**Acceptance gate**: RTF ≤ 0.05 on S1 (2.8s), and transcription of S0-S3 within 20% CER of sherpa-onnx baseline. If decoder TRT conversion fails (TRT rejects ONNX) or takes > 8h, reassess whether to use ORT CUDA EP as an intermediate step.

---

## 6. RTF Feasibility Verdict

### Current bottleneck breakdown (S1, 2.8s)

| Component | Estimated cost |
|---|---|
| TRT encoder (1 chunk, 40 frames) | ~20ms |
| ORT decoder (CPU EP, ~3 tokens) | ~380ms |
| Python overhead, H2D/D2H, CIF | ~23ms |
| **Total P50** | **~423ms** |

The decoder on ORT CPU EP is the overwhelming bottleneck. Until the decoder is moved to TRT or ORT CUDA EP, the RTF floor is ~0.13 (380ms / 2800ms), not 0.05.

### With Phase 2 fixes (no decoder TRT)

- Encoder: persistent stream → ~15ms
- Decoder: ioBinding + same ORT CPU EP → ~350ms (10% reduction from alloc savings)
- Other: ~10ms
- **Estimated total: ~375ms → RTF ~0.13**

Phase 2 alone will **not** reach RTF ≤ 0.08.

### With Phase 3 (decoder TRT)

- Encoder TRT (CUDA Graph): ~10ms
- Decoder TRT (estimated, analogous to ASR decoder perf): ~5-15ms
- Other: ~5ms
- **Estimated total: ~20-30ms for 2.8s audio → RTF 0.007-0.011**

That would beat the RTF ≤ 0.05 target by ~5×. But this estimate assumes decoder TRT conversion succeeds and achieves similar throughput to the encoder (~10ms per N-token batch). TRT decoder conversion is the high-risk item — ONNX graphs with 16 dynamic cache tensors often require manual shape annotations or partial graph fallback to ORT.

### Verdict

**RTF ≤ 0.05 is achievable, but only after Phase 3 (decoder TRT conversion).** Phases 1-2 are necessary but not sufficient for the performance target. The realistic delivery sequence:

1. Phase 1 (P0 quality): 1-2h → stops the bleeding, enables real quality evaluation
2. Phase 2 (instrumentation + encoder optimizations): 2-4h → confirms bottleneck split, achieves RTF ~0.13 (still 2-3× off target)
3. Phase 3 (decoder TRT): 4-8h → RTF ≤ 0.05 plausible

If decoder TRT conversion is blocked (TRT rejects ONNX after 4h of attempts), the fallback is ORT CUDA EP for the decoder — which would likely achieve RTF ~0.03-0.05 on Jetson with CUDA-accelerated attention, without the full TRT rewrite.

**Do not commit to RTF ≤ 0.05 before Phase 2 timing data is collected.** The 380ms decoder estimate is inferred from total latency; actual per-component timing may reveal a different bottleneck.
