# ASR Prefill/Decode Performance Plan - 2026-04-27

## Current state

Runtime ASR for the Python backend is not using the full C++ `ASRPipeline::Transcribe` path. The active offline path is `Qwen3ASRBackend._transcribe_python`, which runs ORT encoder, builds prompt embeddings in Python, then calls the pybind `TRTDecoder.prefill()` and `TRTDecoder.decode_step()` methods directly when TRT is available (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:934`, `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:960`, `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:970`).

Python currently has one combined timer around both prefill and autoregressive decode. The timer starts immediately before `self._decoder.prefill(input_embeds)` and stops after the decode loop, then logs `prefill+dec` rather than separate `prefill_ms` and `decode_ms` (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:955`, `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:1027`, `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:1030`). Therefore the observed `10 tokens / 218 ms total / ~22 ms per token` cannot currently be attributed to prefill, decode kernel, H2D, D2H, Python argmax, or graph misses from production Python logs.

The C++ benchmark pipeline already has the desired timing split: encoder timing is recorded around `RunEncoder`, prefill timing is recorded around either ORT or TRT prefill, and decode timing is recorded around the greedy loop (`/Users/harvest/project/jetson-voice/benchmark/cpp/asr_pipeline.cpp:285`, `/Users/harvest/project/jetson-voice/benchmark/cpp/asr_pipeline.cpp:303`, `/Users/harvest/project/jetson-voice/benchmark/cpp/asr_pipeline.cpp:334`, `/Users/harvest/project/jetson-voice/benchmark/cpp/asr_pipeline.cpp:388`). It also exposes decoder profiling printouts for H2D, kernel, D2H, total, and overhead when profiling is enabled (`/Users/harvest/project/jetson-voice/benchmark/cpp/asr_pipeline.cpp:403`).

Prompt construction is mostly fixed prefix plus variable encoder tokens. `_build_prompt` inserts role markers, the audio span, optional language tag, and `ASR_TEXT`; the variable portion is the number of `AUDIO_PAD` tokens equal to encoder output length (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:1055`). Runtime observations put this at about 20-30 encoder output tokens for the relevant utterances, so prefill sequence length is fixed prefix plus that small audio span.

The active TRT decoder is constructed as a 28-layer, 1024-hidden, 8-head, 128-head-dim, 151936-vocab engine with `max_seq=500`; CUDA Graph is enabled at load time (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:799`, `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:806`). The engine header indicates decode-step CUDA Graph caches by `(kv_len, parity)`, keeps cache across reset because GPU buffer addresses are fixed, and notes expected replay behavior after cache warmup (`/Users/harvest/project/jetson-voice/benchmark/cpp/tts_trt_engine.h:154`). The same header documents dual optimization profiles for batch prefill and autoregressive decode (`/Users/harvest/project/jetson-voice/benchmark/cpp/tts_trt_engine.h:166`).

The current pybind prefill wrapper returns full `[1, S, vocab_size]` logits to Python, copying the whole result into a numpy array (`/Users/harvest/project/jetson-voice/benchmark/cpp/tts_binding.cpp:110`). Python then does greedy selection with `np.argmax(logits[0, -1, :])` after prefill and after every decode step (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:962`, `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:965`, `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:970`). `decode_step` returns `[1, vocab_size]` or equivalent per step to Python, so each generated token pays a device-to-host logits transfer plus numpy argmax.

## Optimization candidates (rank by expected ms benefit descending)

### B6-variant: move greedy argmax/sampling into C++/GPU and return token IDs

Current state + problem: Python receives logits after prefill and after every decode step, then computes `np.argmax` on the host (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:965`, `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:970`). The pybind prefill wrapper copies full `[1, S, vocab_size]` logits into numpy (`/Users/harvest/project/jetson-voice/benchmark/cpp/tts_binding.cpp:110`). With vocab size 151936, even a single-step logits transfer is large enough to dominate small-token utterances if kernel replay is already fast.

Approach summary: add a C++/pybind method that runs greedy decode internally and returns token IDs plus timing metadata, or add narrower methods such as `prefill_argmax(input_embeds) -> token_id` and `decode_step_argmax(input_embed, eos_ids) -> token_id`. Prefer a GPU reduction kernel reading `d_logits_` directly before D2H; a CPU fallback can copy only a scalar token ID. Keep the Python loop structure initially, replacing only `logits = ...` and `np.argmax(...)` with token-returning calls, then optionally move the whole 200-token loop into C++ after validation. Relevant current files: `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:960`, `/Users/harvest/project/jetson-voice/benchmark/cpp/tts_trt_engine.h:228`, `/Users/harvest/project/jetson-voice/benchmark/cpp/tts_binding.cpp:110`.

Expected ms benefit: hypothesis 20-120 ms per 10-token utterance. Lower bound assumes D2H is already optimized; upper bound assumes `[1,V]` D2H and host argmax are a major share of the observed 218 ms.

Complexity: medium if adding argmax-only pybind calls; medium-high if moving the full EOS loop into C++ with metadata and tokenizer-independent output handling.

Risk level: medium. Pybind ABI and TensorRT engine buffer ownership must stay stable.

Accuracy risk: low for greedy argmax if tie behavior is documented and matches numpy for normal finite logits. Medium if this becomes sampling/top-k/top-p later.

Dependencies: B0 instrumentation should land first or alongside this so the measured gain can be attributed. Needs access to decoder device logits buffer or a C++ method that can run reduction before logits copy.

### B0: split Python instrumentation for prefill and decode

Current state + problem: Python logs `prefill+dec` as one bucket (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:1030`), while the benchmark C++ pipeline has separate `prefill_ms` and `decode_ms` (`/Users/harvest/project/jetson-voice/benchmark/cpp/asr_pipeline.cpp:303`, `/Users/harvest/project/jetson-voice/benchmark/cpp/asr_pipeline.cpp:388`). Without this split, every optimization decision is partly speculative.

Approach summary: time prefill separately around `self._decoder.prefill(input_embeds)` and time the autoregressive loop separately. Log `seq_len`, `audio_len`, `token_count`, `backend`, `max_seq`, and whether CUDA Graph appears captured/hit if a pybind property is available (`cuda_graph_captured` exists for the underlying engine binding at `/Users/harvest/project/jetson-voice/benchmark/cpp/tts_binding.cpp:127`). If profiling stats are exposed through pybind, add H2D/kernel/D2H/overhead fields matching the existing C++ profiling output (`/Users/harvest/project/jetson-voice/benchmark/cpp/asr_pipeline.cpp:407`).

Expected ms benefit: 0 ms direct runtime benefit; high diagnostic benefit. It should reduce optimization uncertainty enough to avoid implementing the wrong path.

Complexity: low for Python wall-clock split; medium if pybind profiling accessors need to be exposed.

Risk level: low if logging is passive and guarded.

Accuracy risk: none.

Dependencies: none for wall-clock fields; pybind exposure required for H2D/kernel/D2H and reliable graph hit/miss counters.

### B1: return only last-position prefill logits or prefill argmax

Current state + problem: prefill returns full `[1, S, vocab_size]` logits to Python, but Python only reads the final position (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:962`, `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:965`). The C++ benchmark also only needs the final prefill logits before entering the greedy loop (`/Users/harvest/project/jetson-voice/benchmark/cpp/asr_pipeline.cpp:342`).

Approach summary: change or add a pybind prefill variant that returns only last logits `[vocab_size]`, or better, returns the first greedy token ID. Keep the existing `prefill()` API for compatibility. If returning last logits, this saves roughly `(S-1) * V` host copy; if returning token ID, it avoids the first full logits D2H entirely.

Expected ms benefit: hypothesis 5-80 ms per utterance, depending on `seq_len` and actual D2H bandwidth/synchronization cost.

Complexity: medium.

Risk level: medium because prefill output shape is part of current Python contract.

Accuracy risk: low for greedy-only token ID; none for last-logits shape-preserving semantics if callers are updated deliberately.

Dependencies: B0 should record prefill output copy time if possible; B6-variant can subsume this if it includes prefill argmax.

### B2: warm and verify CUDA Graph cache for ASR decode

Current state + problem: Python enables CUDA Graph at decoder load (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:806`), but warm-up only runs the ORT fallback decoder path and does not explicitly run TRT prefill/decode steps to populate ASR graph cache (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:838`). The engine header says first use of a `(kv_len, parity)` graph pays capture/instantiate overhead and subsequent use replays quickly (`/Users/harvest/project/jetson-voice/benchmark/cpp/tts_trt_engine.h:255`).

Approach summary: add startup warm-up for representative TRT ASR prefill sequence lengths and 10-20 decode steps, then log graph captured/cache size. The goal is to remove cold graph capture from first utterance and confirm production traffic sees graph hits.

Expected ms benefit: hypothesis 0 ms steady-state if cache is already hot; 20-200 ms on first utterance or on unseen sequence lengths.

Complexity: low-medium.

Risk level: medium. Warm-up can increase startup latency and GPU memory pressure.

Accuracy risk: none if decoder state is reset after warm-up.

Dependencies: pybind graph-state visibility beyond `cuda_graph_captured` would make this testable.

### B3: move the full greedy decode loop into C++

Current state + problem: Python owns the loop, embedding lookup, EOS break, and repeated pybind calls (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:964`). This adds interpreter overhead and forces Python to orchestrate every token.

Approach summary: expose a method like `decode_greedy(input_embeds, embed_table, eos_ids, max_tokens) -> {token_ids, prefill_ms, decode_ms, per_step_ms, graph_hits}`. C++ already implements the equivalent greedy loop in benchmark code (`/Users/harvest/project/jetson-voice/benchmark/cpp/asr_pipeline.cpp:346`). Production Python can still build prompt embeddings initially, then call the C++ loop once.

Expected ms benefit: hypothesis 5-50 ms per 10-token utterance beyond B6, mostly from reduced pybind/Python overhead and better event timing.

Complexity: medium-high.

Risk level: medium-high because it moves output-control logic and metadata into C++.

Accuracy risk: low for greedy if token IDs are identical; medium if embedding table dtype/conversion diverges from Python.

Dependencies: B6-variant token selection or device argmax primitive.

### B4: reduce prompt/prefill length where semantically safe

Current state + problem: prefill length is fixed prompt plus `audio_len` `AUDIO_PAD` tokens, language tag, and `ASR_TEXT` anchor (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:1055`). Runtime observations show only about 20-30 encoder tokens for target utterances, so this is not the top bottleneck, but every extra token increases prefill compute and full-logits copy size.

Approach summary: audit whether any fixed role-marker tokens or optional language text can be cached or shortened without changing model behavior. Do not change the prompt format until there is an accuracy comparison suite because Qwen chat/audio markers are likely part of model alignment.

Expected ms benefit: hypothesis 0-20 ms per utterance.

Complexity: medium.

Risk level: high relative to benefit.

Accuracy risk: medium-high.

Dependencies: accuracy regression corpus with multilingual and silence cases.

### B5: reuse fixed-prefix KV cache across utterances

Current state + problem: every utterance re-prefills the fixed system/user/assistant prefix even though only audio tokens and language may vary. This is theoretically reusable up to the audio insertion boundary.

Approach summary: precompute KV for the invariant prefix before `AUDIO_PAD`, then continue prefill with variable audio embeddings. This requires engine support for non-empty past in batch prefill and careful position handling. It is not a first cut because the current batch prefill path appears designed for past length zero and full prompt embeddings (`/Users/harvest/project/jetson-voice/benchmark/cpp/tts_trt_engine.h:186`).

Expected ms benefit: hypothesis 5-40 ms per utterance if implemented correctly.

Complexity: high.

Risk level: high.

Accuracy risk: medium-high due to cache boundary, audio offset, position IDs, and prompt-marker semantics.

Dependencies: decoder export/profile support for prefill with past, robust state reset tests, and B0 timing to prove prefill is large enough to justify it.

## Recommended first cut

Implement B0 and the narrow B6-variant together. The first objective is to make the 218 ms / 10-token observation decomposable. The second objective is to remove the most suspicious per-token overhead without redesigning the ASR pipeline.

### B0 implementer-ready spec

Scope: edit only the Python instrumentation path and, if needed, pybind metadata accessors. Do not change decoding behavior.

In `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py`, split the single timer at `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:955` into:

- `prefill_t0` around `self._decoder.prefill(input_embeds)` at `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:961`.
- `decode_t0` around the loop at `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:964`.
- Keep `enc_ms` as-is from `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:937`.
- Replace the existing `decode_ms` meaning at `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:1027` with `prefill_ms`, `decode_ms`, and `prefill_decode_ms = prefill_ms + decode_ms`.

Log fields:

- `enc_ms`
- `prefill_ms`
- `decode_ms`
- `prefill_decode_ms`
- `total_ms`
- `seq_len`
- `audio_len`
- `audio_offset`
- `token_count`
- `per_token_decode_ms = decode_ms / max(token_count, 1)`
- `backend`
- `trt_max_seq`
- `cuda_graph_enabled` if accessible
- `cuda_graph_captured` from the existing pybind property shape shown at `/Users/harvest/project/jetson-voice/benchmark/cpp/tts_binding.cpp:127`
- `h2d_ms`, `kernel_ms`, `d2h_ms`, `overhead_ms` if profiling stats can be exposed from the existing C++ stats source at `/Users/harvest/project/jetson-voice/benchmark/cpp/asr_pipeline.cpp:407`

Use clear labels in the log line, for example:

`offline transcribe: enc=... prefill=... decode=... total=... seq_len=... audio_len=... tokens=... graph_captured=... h2d=... kernel=... d2h=...`

Acceptance criteria:

- For a short utterance, logs show separate `prefill_ms` and `decode_ms`; no production log line reports only `prefill+dec`.
- `n_tokens` and `per_token_ms` in `TranscriptionResult` remain populated and semantically clear. Prefer `per_token_ms` based on decode-only time; if compatibility requires total prefill+decode per token, log both and document the distinction.
- Text output token IDs are unchanged for a fixed test wav before and after instrumentation.
- When TRT is enabled, log includes `seq_len`, `audio_len`, `token_count`, and at least one CUDA Graph state field. If H2D/kernel/D2H are not accessible, the log says `decoder_profile=unavailable` rather than silently omitting them.
- Instrumentation is passive and can remain enabled in normal logs without excessive volume.

### B6-variant implementer-ready spec

Scope: avoid full `[1,V]` device-to-host logits transfer for each decode step and avoid Python `np.argmax` on decode logits. Preserve greedy decoding exactly.

Phase 1 API target:

- Add a pybind method on the decoder such as `decode_step_argmax(input_embeds, eos_ids=None) -> dict`.
- The method runs the existing TRT decode step, performs argmax in C++/CUDA, and returns at minimum `{ "token_id": int, "is_eos": bool }`.
- Add a prefill companion such as `prefill_argmax(input_embeds) -> dict` or extend `prefill()` with an opt-in `return_last_token=True`. It should seed KV exactly as current `prefill()` does, then return the greedy token from the last prefill position without returning full `[1,S,V]` logits.
- Keep current `prefill()` and `decode_step()` APIs intact for fallback/debug.

Implementation preference:

- Preferred: GPU reduction kernel reads the decoder logits device buffer directly and copies back only one token ID and optional max value.
- Acceptable first implementation: C++ performs host argmax after a D2H logits copy, but Python no longer receives logits. This is lower benefit and must be labeled as transitional in logs.
- Avoid changing tokenizer decode or prompt construction in this phase.

Python integration:

- In `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py`, replace the TRT branch at `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:960`.
- First token comes from `prefill_argmax(input_embeds)` instead of `result["logits"]` plus `np.argmax`.
- Loop appends `token_id`, builds the next embedding from `self._embed_tokens[token_id]`, then calls `decode_step_argmax`.
- EOS handling must remain equivalent to `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:966`.
- ORT fallback remains unchanged at `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:972`.

Acceptance criteria:

- For a fixed wav corpus, token ID sequences from old TRT greedy and new argmax path are identical. If ties occur, document tie handling and add a deterministic tie test.
- Python no longer indexes `logits[0, -1, :]` in the TRT path.
- Per-step D2H bytes for decode are reduced from roughly `vocab_size * sizeof(logit)` to one scalar token ID plus optional debug scalar. This must be verified by profiling fields or an explicit debug counter.
- B0 logs show lower `decode_ms` or lower `d2h_ms` on the same warmed engine. Expected win is hypothesis 20-120 ms for the observed 10-token utterance; accept the change only if measured improvement is at least 10 ms or D2H is proven negligible and the API simplification is retained for C++ loop follow-up.
- CUDA Graph remains enabled and graph cache behavior is not regressed.
- State reset between utterances still produces identical first-token behavior on repeated calls.

## Anti-patterns to avoid

Do not start with KV cache reuse across utterances. The fixed-prefix idea is tempting, but the boundary before audio injection is semantically delicate: `audio_offset` is found by locating `AUDIO_PAD` in the prompt (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:946`), and prompt layout includes audio markers, optional language, and `ASR_TEXT` (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:1055`). Reusing KV across the wrong boundary can silently change attention context.

Do not weaken state reset correctness. Streaming reset clears partial tokens, committed tokens, encoder frames, VAD counters, utterance audio, and finalization state (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:455`, `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:552`). Any decoder-side cache, graph, or token-selection state must preserve this behavior. CUDA Graph cache may survive reset only if it is shape/address cache, not semantic KV state.

Do not accept vague performance wins. Acceptance criteria must report `prefill_ms`, `decode_ms`, `seq_len`, `token_count`, and graph state on the same audio with warm/cold distinction. A single total latency number is not enough.

Do not optimize by changing streaming receptive field casually. Streaming uses a 400 ms chunk, 1.0 s left context, encoder hop of 1280 samples, and 5.0 s rolling buffer (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:66`). Shortening these can reduce compute but changes recognition quality and partial stability.

Do not treat rolling buffer changes as decode optimization. The rolling encoder buffer is trimmed by frame count in streaming (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:510`). Changing it affects context and boundary dedup, not just speed.

Do not bypass the >6 s segmentation guard. The backend deliberately splits long audio because Qwen3-ASR has observed deterministic truncation after about 6.5 s; `LONG_AUDIO_THRESHOLD_SEC` is 6.0 and `VAD_MAX_SEG_SEC` is 4.5 (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:84`). Performance changes must preserve this behavior unless the model issue is separately fixed and validated.

Do not move mel or VAD boundaries without an ASR-quality test. Offline path computes mel before encoder (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:895`), and segmented path computes mel per segment (`/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:916`). Segment-level mel boundaries can affect encoder output and transcript joins.

## Open questions

- How much of the observed 218 ms is prefill vs decode after warm CUDA Graph cache? B0 should answer this first.
- Does the current ASR pybind module expose `cuda_graph_captured`, `cuda_graph_enabled`, or profiling stats for `TRTDecoder`, or are those only present in adjacent binding code?
- Are decode-step logits FP32, FP16, or BF16 on the actual ASR engine, and how many bytes are copied D2H per step?
- Does `TRTDecoder.decode_step()` currently copy `last_hidden` to host even though Python ignores it? The C++ decode interface includes both logits and hidden outputs (`/Users/harvest/project/jetson-voice/benchmark/cpp/tts_trt_engine.h:140`).
- Are CUDA Graph keys for ASR decode mostly reused across utterances with variable `seq_len`, or does each prompt length create cold graph captures?
- What exact warmed-engine baseline should be used for acceptance: the observed 10-token / 218 ms sample, a small corpus, or both?
- Should `TranscriptionResult.per_token_ms` represent decode-only per token or combined prefill+decode per token for backwards compatibility?
- Is there an existing CI/test wav corpus that can assert exact token ID equality across Python argmax and C++/GPU argmax?
