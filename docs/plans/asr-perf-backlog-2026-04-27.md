# ASR Performance Backlog — 2026-04-27

Investigation into V2V steady-state ASR latency and remaining optimization
levers, modelled on `tts-perf-backlog-2026-04-19.md`.

## Current state

### V2V steady measurement (today, this session — `bench/v2v_simple.py`, S1.wav 2.8s zh)

```
Run 1/8 (cold):  EOS->Audio= 576ms  ASR=569ms  TTS=7ms
Run 2/8 (warm):  EOS->Audio= 517ms  ASR=509ms  TTS=6ms
Run 3/8 (warm):  EOS->Audio= 560ms  ASR=553ms  TTS=6ms
Run 4/8 (warm):  EOS->Audio= 634ms  ASR=626ms  TTS=6ms
Run 5/8 (warm):  EOS->Audio= 645ms  ASR=638ms  TTS=6ms
Run 6/8 (warm):  EOS->Audio= 497ms  ASR=490ms  TTS=6ms
Run 7/8 (warm):  EOS->Audio= 528ms  ASR=521ms  TTS=7ms
Run 8/8 (warm):  EOS->Audio= 448ms  ASR=441ms  TTS=6ms
warm_all median 528ms  steady (post-warmup) median 488ms
```

Today's steady ASR latency is in the **440–650ms range, median ~520ms, not
~310ms**. The 310ms figure in the brief was a snapshot from a previous run; the
relative split (ASR ≫ TTS) is unchanged, but ASR's absolute share is bigger
than the brief assumed. Same caveat: TTS first-chunk is essentially free
(~6 ms), so **all of the EOS→audio latency is ASR**.

### Architecture (HEAD `15330f7` + branch `feature/asr1-true-streaming` HEAD `be0ef76`)

Models: `/opt/models/qwen3-asr-v2/`
- `encoder_fp16.onnx` 378MB → ORT CUDA EP, `user_compute_stream=…`
- `asr_decoder_bf16.engine` 1.2GB → TRT (C++ pybind via `qwen3_speech_engine`),
  CUDA Graph enabled, `seq_len_max=500`, dual-profile bf16, prefill +
  decode_step API.
- `decoder_unified.onnx` / `decoder_step.onnx` 1.1–2.4GB ORT — fallback only.
- `embed_tokens.bin` 311MB — float16, kept in RAM as float32.
- `tokenizer.json` 11MB.

Pipeline at EOS (`/asr/stream`, empty binary frame):

```
ws_handler:                                                  app/main.py:506-516
  await stream.prepare_finalize()       # no-op for streaming
  final_text = await stream.finalize()  # ── this is the gating call ──
  ws.send_json({type:"final", text:...})
```

`Qwen3StreamingASRStream.finalize()` (qwen3_asr.py:344-386):
1. Drain unprocessed audio into `_encoder_frames` (re-encode with
   left-context for any sub-chunk tail).
2. **If `_episode_final` was not already set** (the typical case — VAD
   endpoint usually has not fired by the time the client sends EOS), call
   `_offline_final_text()` which calls `transcribe_audio()` →
   `_transcribe_python()`. **This is a full re-encode of the full 2.8 s of
   audio plus a fresh prefill + autoregressive decode.**

In other words, the streaming partials done during the live audio are
**discarded at EOS**. Everything done in `_process_streaming_chunk` (encoder
+ partial decode) is thrown away and the offline path runs from scratch.
The streaming logger line confirms cumulative pre-EOS work
(`enc=916ms dec=1704ms over 7 chunks of 2.8s audio`) that is not reused.

### What the 488 ms is made of (instrumented vs not)

`_transcribe_python` (qwen3_asr.py:934) measures `enc_ms` and `decode_ms`
locally but **never logs them**. We have no per-call timing for the offline
final path. Best inferred breakdown for 2.8 s of audio + ~10-token output:

| Stage | Source | Cost (estimated) |
|---|---|---|
| Mel filterbank (CPU, `WhisperFeatureExtractor`) | qwen3_asr.py:1066, runs in numpy | ~30–60 ms (pad to 4 s) |
| Encoder ORT CUDA (encoder_fp16) | qwen3_asr.py:938, 2.8s audio | ~80–150 ms (encoder is the heavy ONNX, no TRT engine) |
| Build prompt + input_embeds (Python loop, 30+ tokens) | qwen3_asr.py:949-953 | ~5–15 ms (per-token gather over 311MB embed_tokens via Python `for`) |
| TRT prefill (`_decoder.prefill`, ~30 token seq) | qwen3_asr.py:961 | ~30–50 ms (similar magnitude to TTS unified prefill at seq_len ~30) |
| TRT decode loop (10 tokens × CUDA-Graph step) | qwen3_asr.py:964-970 | ~150–250 ms (15–25 ms/token × 10) |
| Tokenizer decode + WS send | qwen3_asr.py:1031, async hop | ~5–10 ms |
| **Total** | | **~300–530 ms** ✓ matches measurement |

> EVIDENCE — in container log we already see TRT-side prefill numbers from the
> TTS side: `Prefill: ~16 ms (TRT unified)` for `seq_len=9`. ASR prefill at
> seq_len ≈ 30 will be 2-3× that. Encoder is the most-suspected single
> component but is **not currently instrumented** in this code path.

**A 5-min instrumentation patch (logger.info on `enc_ms`/`decode_ms` inside
`_transcribe_python`) is the prerequisite for ranking the candidates
below.** Without it we are guessing relative shares.

## Optimization candidates (ranked by expected ms saved)

### A0 — Instrument the offline final path (P0, prerequisite)
- Change one line: `logger.info("offline transcribe: enc=%.0fms prefill+dec=%.0fms tokens=%d", enc_ms, decode_ms, len(output_ids))` at qwen3_asr.py:1027.
- Expected saving: **0 ms (this is measurement)**.
- Effort: 1 line. Risk: zero.
- **Do this first**, then the rest of this list can be re-prioritised against real numbers.

### A1 — Reuse streaming encoder output instead of re-encoding at EOS (HIGH ROI)
- Today's path: `finalize()` calls `_offline_final_text()` → re-runs encoder over the **full** 2.8 s audio, even though the streaming session already encoded all of it (`_encoder_frames` holds the result, and `_total_encoder_frames` ≈ 36 frames).
- Fix: when `_encoder_frames` covers the entire `_utterance_audio_buffer` (the common case after `accept_waveform` drains all chunks), skip the re-encode and use `np.concatenate(_encoder_frames, axis=1)` as `enc_out`, then run the same prompt + prefill + decode path.
- Expected saving: **80–150 ms** (full encoder pass — likely the single biggest item).
- Risk: medium. The streaming encoder runs with overlapping left-context windows; per-chunk trim may leave 1–2 frame gaps at chunk boundaries vs a single-shot 2.8 s encode. Need a Jaccard ≥ 0.95 ASR-verify against the offline path on the standard test set before shipping. If the streaming encoder is "close enough" the saving is gold; if accuracy regresses, fall back to a hybrid (re-encode only the last chunk and reuse earlier frames).
- Complexity: ~30 lines in `_offline_final_text` / a new `_final_from_streaming_frames` helper.
- Depends on: A0 to confirm encoder is actually 80+ ms.

### A2 — Encoder TRT engine (replace ORT CUDA)
- Today: encoder is `encoder_fp16.onnx` (378 MB) on **ORT CUDA EP**. Decoder is already a TRT engine. The TTS side runs everything on TRT and has a CP pool because raw ORT was unacceptable. The ASR encoder stayed ORT.
- Fix: build a TRT engine for `encoder_fp16.onnx` (similar to TTS T1 lineage). `trtexec --onnx=encoder_fp16.onnx --fp16 --saveEngine=encoder_fp16.engine --shapes=mel:1x128x3000`. Wrap with the existing C++ pybind harness or extend `qwen3_speech_engine` with an encoder class.
- Expected saving: **30–60 ms** (TRT typically 1.3–1.8× over ORT CUDA on Orin NX for FP16 encoder workloads). Also removes the ORT-vs-TRT-stream conflict class entirely (P0 #906 was a symptom of this co-existence).
- Risk: medium. Need bf16 vs fp16 study (encoder may stay fp16 fine; decoder needed bf16 because of QK²overflow on Qwen3, encoder doesn't have that issue). Build infra is in place; engine size will be ~400 MB so swap/disk is fine.
- Complexity: ~80 lines C++ (mirror `TRTDecoder`) + build/deploy script + warmup. ~1 day of work.
- Risk-bonus: also enables `cudaStreamSynchronize` simplifications — currently we have to keep an isolated ORT stream because of the legacy-stream issue.

### A3 — Cache prompt-prefix input_embeds (skip Python token loop)
- Today: qwen3_asr.py:949-951 builds `input_embeds` by Python-looping
  `for i, tid in enumerate(prompt_ids): input_embeds[0, i] = self._embed_tokens[tid]`. The prompt prefix `[IM_START, 9125, 198, IM_END, 198, IM_START, 882, 198, AUDIO_START]` is **identical for every request** (10 tokens), and the suffix `[AUDIO_END, IM_END, 198, IM_START, 77091, 198, ASR_TEXT]` is identical per language (7-9 tokens). Only the AUDIO_PAD count varies.
- Fix: precompute prefix/suffix embeds once at init (per language), then per request: vectorised `np.copyto(input_embeds[:, audio_offset:], enc_out)` instead of any Python loop.
- Expected saving: **5–15 ms** (Python loop over 30+ tokens with 311 MB array indexing is surprisingly slow).
- Risk: very low.
- Complexity: ~20 lines.

### A4 — Mel filterbank GPU / drop padding waste
- Today: mel runs CPU via `transformers.WhisperFeatureExtractor`. For 2.8 s audio it pads to 3 s × 128 mel bins → ~30 ms in numpy.
- Fix: either (a) cuBLAS/torch GPU mel, or (b) reduce padding — current code uses `chunk_length=int(audio_secs)+1` so 2.8s → 3s, that's already optimal. Easier win: cache the FE per chunk_length (already done at qwen3_asr.py:1072) — no further mel optimization available without porting to GPU.
- Expected saving: **20–40 ms** if moved to GPU; very low if not.
- Risk: low.
- Complexity: medium (~100 lines + dependency on torch.audio mel kernel).
- Note: only worth doing AFTER A1+A2, otherwise mel is a small fraction.

### A5 — TRT prefill warm-up at multiple seq lengths
- Today: warm-up at qwen3_asr.py:838-862 only warms encoder + ORT decoder fallback. The TRT decoder is loaded but never warmed. First request gets the cold path (~30+ ms slower per TTS T7 lesson).
- Fix: at startup, run `_decoder.prefill` at a couple representative seq_lens (e.g., audio_len=20, 40, 80) and a `decode_step` round trip.
- Expected saving: **10–15 ms (first-request only)**, none on steady-state.
- Risk: very low. Mirrors TTS T7 pattern.
- Complexity: 15 lines.

### A6 — Decode loop micro-opts (numpy → CUDA argmax + embed gather)
- Today: per-token decode does `next_token = int(np.argmax(logits[0, -1, :]))` (CPU argmax of 152K-vocab logits, ~1 ms) + Python `embed_tokens[next_token].astype(...)` (CPU gather + dtype copy + H2D, ~1 ms).
- Fix: keep logits on GPU, argmax on GPU, embed gather on GPU (already in `embed_tokens` table that could live in GPU memory).
- Expected saving: **10–25 ms** for 10-token decode (1–2.5 ms/token).
- Risk: medium. Crosses the C++/Python boundary — would benefit from shoving the whole `prefill→decode_until_eos` loop into the C++ TRTDecoder.
- Complexity: substantial (~150 lines C++) — keep on backlog until A1/A2/A3 are done.

### A7 — Skip "build prompt" entirely on common short paths via cached prompt_ids
- Today: `_build_prompt(audio_len, lang)` builds the list anew per request, including `[AUDIO_PAD] * audio_len`. That's a Python list of size ~36; cheap but allocates.
- Fix: cache the prompt template arrays per (audio_len, lang) for common bucketed sizes.
- Expected saving: **<5 ms**. Skip — micro.

### A8 — "True streaming" final emission (use last partial as final)
- Today: streaming partials are computed during the audio stream but **discarded** at finalize (qwen3_asr.py:541-546 — `_offline_final_text` re-encodes from the audio buffer, not from `_partial_token_ids`).
- Fix: short-circuit finalize to return `_partial_text` if partials are non-empty AND have been stable for ≥1 chunk.
- Expected saving: **~440 ms** in the best case (entire offline pass eliminated). EOS→final reduces to "send the WS message".
- Risk: **HIGH**. Today's commit `3cb6f3f` ("final emission via offline full-audio decode") explicitly walked back from this strategy because partial text quality was lower than offline text. Reviving it would re-introduce that quality regression unless partials are aggressively improved (longer max_tokens, ring-buffer with boundary dedup that the new code already has).
- Complexity: ~30 lines in `finalize()` + an exhaustive A/B test on `tests/asr_real_wav_eval/`.
- Recommendation: **NOT recommended right now** — accuracy regression risk too high. Listed for completeness because it's the largest theoretical win.

### A9 — VAD endpoint silence shortened
- Today: `VAD_ENDPOINT_SILENCE_MS=1000` (env-overridable). Affects pre-EOS endpoint detection, NOT the EOS→final number. Lowering it would cut ~500 ms off the **interactive** turn-taking latency (because VAD-triggered finalize starts decoding before the client sends EOS), but does nothing for the V2V benchmark we measured.
- Worth a separate test for human-facing UX, but not in the same backlog.

### A10 — Streaming chunk size beyond 400 ms
- Today already moved 250 → 400 ms. Going further (e.g., 600 ms) cuts encoder-call frequency by another 33% but at the cost of partial latency (worse interactive feel). Marginal V2V gain because partials are not on the critical path. Skip.

## Won't do / already tried

- **silero-vad** — crashes ORT+TRT same-process (memory: `feedback_silero_vad_trt_conflict`). Replaced with webrtcvad which works fine.
- **A8 (use partial as final)** — explicitly reverted today (`3cb6f3f`); accuracy regression too large.
- **ASR encoder INT8** — in TTS context Jetson INT8 was a net loss for small-batch autoregressive (memory: `feedback_jetson_int8_small_batch`). Encoder is non-AR and large (40+ frames per call) so could differ, but expected gain is <30 ms with calibration risk; defer until A1/A2/A3 are done.

## Summary table

| ID | Item | ms saved (est.) | Effort | Risk | Status |
|---|---|---|---|---|---|
| **A0** | Instrument offline path | 0 (prereq) | 1 line | none | **DO FIRST** |
| **A1** | Reuse streaming encoder frames at EOS | **80–150** | 30 lines Py | medium | **HIGH ROI** |
| **A2** | Encoder TRT engine (replace ORT) | **30–60** | 80 lines C++ + build | medium | **HIGH ROI** |
| **A3** | Cache prompt-prefix embeds | 5–15 | 20 lines Py | low | low risk fill-in |
| A4 | Mel GPU | 20–40 (after A1/A2) | 100 lines | low | defer |
| A5 | TRT decoder warmup | 10–15 (cold only) | 15 lines | low | quick win |
| A6 | Decode loop GPU-side | 10–25 | 150 lines C++ | medium | defer |
| A7 | Cache prompt template | <5 | trivial | none | skip |
| A8 | Use partial as final | ~440 | 30 lines | **HIGH** (accuracy) | **don't** |
| A9 | Shorter VAD silence | UX only, not V2V | 1 env var | low | separate axis |
| A10 | Larger chunk size | <10 | trivial | low | skip |

**Stacking potential** (rough, not strictly additive):
A1 + A2 + A3 + A5 ≈ 130–250 ms saved → V2V ASR drops from ~488 ms median to **~250–360 ms median**, putting V2V steady around **260–370 ms** total.

## Next step recommendation

1. **Do A0 first** (one-line log) and re-run `v2v_simple.py` to nail the real
   enc-vs-decode split. ~5 minutes.
2. With numbers in hand, **A1 (reuse streaming frames) is the biggest single
   risk-controlled win** — the streaming encoder has already done the work
   and the existing code is throwing it away. If accuracy holds (Jaccard
   ≥ 0.95 vs offline), this is one PR for ~100 ms.
3. **A2 (encoder TRT engine) is the structural win**: it removes the ORT
   dependency from the hot path, eliminates the ORT-vs-TRT-stream conflict
   class for free, and matches the TTS architecture. Needs ~1 day of build
   + verify, but pays back forever.
4. A3 + A5 are cheap last-mile after A1/A2.

Real ASR bottleneck: **the encoder pass is run twice per utterance — once
during streaming (logged as `enc=900-1300ms cumulative`) and once again at
EOS (~80-150 ms inside the 488 ms final). The second encoder pass is pure
waste.** A1 closes that gap; A2 then halves what's left.
