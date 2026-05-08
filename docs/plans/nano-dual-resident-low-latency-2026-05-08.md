# Qwen3 ASR + Qwen3 TTS Nano Dual-Resident Plan - 2026-05-08

## Goal

After freezing the current work into commits, make Qwen3-ASR and Qwen3-TTS run as resident streaming services on the Nano target with low latency, while preserving recognition and generation quality.

Scope correction: the zh/en resident path is already verified and is not part of this plan. It can remain as fallback evidence, but it is not a phase gate and should not be used to claim this goal is complete. The target path is streaming Qwen3 ASR plus streaming Qwen3 TTS.

Primary target: `orin-nano`, the 8GB-class Nano device. `orin-nx` can be used for comparison, profiling, and full-stack sanity checks, but an NX pass does not prove Nano success.

## Current Evidence

- The repo has substantial uncommitted Qwen3-related work across ASR/TTS backends, EdgeLLM workers, native C++ pipeline, export tools, benchmarks, tests, and generated evidence. Freeze this before more optimization.
- Existing Nano evidence says the first full Qwen3-ASR + Qwen3-TTS dual residency attempts did not fit on Orin Nano 8GB:
  - `docs/benchmarks/orin-nano-v34-slim-2026-04-28.md`
  - `docs/benchmarks/p1-nano-final-2026-04-28.md`
- Best measured Nano run still OOM-killed while loading TTS after ASR was resident. At `asr_ready`, RSS was about 2.3 GB and system `MemAvailable` was about 2.3 GB; TTS engine loading then pushed the system to OOM.
- Later vocab-pruning and OOM mitigation work materially changed the gap:
  - `docs/plans/handover-2026-04-29-tts-oom-int8.md`
  - ASR vocab pruning saved about 389 MB.
  - TTS vocab pruning saved about 454 MB.
  - Total cumulative memory reclaimed versus the original path was about 1.6 GB.
  - The latest all-fixes run reached Qwen3 pipeline ready, then OOMed on the first synthesize call. The remaining measured gap was about 250-400 MB.
- New streaming-focused Nano reruns on 2026-05-08 changed the working gap:
  - Current `v3.5-clean` + reference TTS worker fails before TTS worker `ready`, not on first stream.
  - Baseline with ASR warmup reaches about `7465 / 7620 MB` and OOMs before TTS `ready`.
  - `EDGE_LLM_TTS_CUDA_GRAPH=0` still reaches about `7468 / 7620 MB`; graph capture is not the main pre-ready gap.
  - `EDGE_LLM_TTS_LAZY_CODE2WAV=1` is ignored by the deployed `/tmp/qwen3tts_ref_0507_from_nano/.../qwen3_tts_worker`; this binary does not contain that env hook.
  - `SKIP_ASR_WARMUP=1` drops the pre-TTS floor by roughly `600-700 MB`, but TTS still OOMs before `ready`.
  - The working memory target is now `900-1200 MB` reclaimed before TTS `ready` if the final configuration must also keep practical headroom.
- Later 2026-05-08 strategy work changed the status again:
  - A runner-compatible `vocoder100` Code2Wav engine reduced Code2Wav resident load by about `600 MB` versus the reference engine and improved first/final chunk Code2Wav time from about `679 ms` to about `435 ms` in TTS-only smoke.
  - TTS text embedding pruning from `151936` rows to `35669` rows adds roughly `430-500 MB` available memory on top of vocoder100 in TTS-only tests.
  - `vocoder100 + TTS text vocab pruning` is the first current-stack Nano run that kept ASR warmup enabled, reached TTS streaming warmup, and produced real `/tts/stream` PCM while both Qwen3 workers remained resident.
  - It is not yet a production-safe endpoint: minimum observed `MemAvailable` was about `27 MB` during startup streaming warmup and about `89-121 MB` during a real short `/tts/stream` request.
- Jetson gotchas record the same class of issue: Qwen3-ASR thinker/audio encoder plus Qwen3-TTS talker/CP/Code2Wav engines are too large for straightforward 8GB dual residency. The known directions are smaller ASR KV, disabling graph capture where it costs memory, lazy Code2Wav, releasing or splitting ASR before TTS, or using 16GB hardware.

## Non-Negotiables

Dual residency is a latency strategy, not an end in itself. A candidate only counts if it supports streaming low latency.

- ASR must stay resident and reusable across streaming chunks.
- TTS Talker, CP/KV, Code2Wav/vocoder, CUDA contexts, and the streaming executor path must be either resident or explicitly warmed before the measured request.
- Unload/reload between ASR and TTS is rejected for the primary path because engine deserialize is 25-60 seconds in prior testing.
- Lazy loading that happens before the first audio chunk is not acceptable as a final low-latency strategy. It can be used as a diagnostic step only.
- Disabling CUDA graph or graph warmup is acceptable only if the measured streaming TTFT/V2V budget still passes after memory fit.
- Offline `/tts` RTF is secondary. The user-facing gate is streaming V2V: `/asr/stream` to `/tts/stream`, EOS to first audio chunk, plus real-time chunk behavior.

## Precision And Quality Boundaries

These are known constraints from prior sessions. Do not rediscover them by trial and error.

| Component / stage | Minimum acceptable path | Known bad or risky path | Why it matters |
|---|---|---|---|
| Qwen3-TTS Talker baseline | explicit-KV BF16 TensorRT Talker, selected through the product path | hidden official/alternate Talker path without matching quality evidence | The accurate baseline was recovered around explicit-KV BF16 semantics; hidden path changes caused audible divergence. |
| Talker quantization candidate | W8A16 only: INT8 weights with BF16 activations/attention | full INT8, FP16 attention, INT4 as primary path | Attention QK^T must keep BF16-class exponent range. Prior notes expect INT8 small-batch latency cost, so it is conditional. |
| CodePredictor / CP | `cp_unified_bf16.engine`; BF16-capable compute/tactics | CP FP16, CP INT8, mixed precision that lets TRT fuse casts away | CP FP16/INT8 overflowed or produced NaN. BF16 was the stable low-precision floor. |
| CP active groups | `cp_active_groups == 15` | `cp_active_groups == 13` or zero-filled residual groups | Reducing residual codebooks changed the later primary-code trajectory and listening quality. |
| Text projection | host FP32 text projection for the accurate path | silently switching projection precision/runtime | The fixed accurate baseline depends on host FP32 text projection. Any lower precision needs A/B proof. |
| TTS embedding / vocab | vocab pruning with exact token-map indirection | dropping tokens outside the proven active set, changing token order, sorted-top-k sampling side effects | Vocab pruning is accepted because it is mathematically equivalent for covered tokens. Token remapping mistakes change logits/sampling. |
| Prefill layout | validated `prefill8` layout with role tokens, codec markers, and `codec_pad` | 9-token language-codec layout, dropping trailing `codec_pad` | Previous changes shifted autoregressive trajectory, caused swallowed tails and voice drift. |
| Sampling | request seed reaches Talker codec sampling and CP residual sampling; vocab-order top-k semantics | sorted-top-k random stream, seed only in Python or only in C++ | Seed and sampling order changed CP residual codes even with matching logits. |
| Vocoder / Code2Wav | validated Code2Wav engine or quality-approved streaming/chunked vocoder path | arbitrary TRT/ORT swap or chunk restart per text segment | Streaming must keep Talker/CP state continuous; vocoder changes can hide quality regressions. |
| ASR decoder / thinker | current pruned/padded BF16 path with quality gate | lower precision or smaller KV/max sequence without CER/WER proof | ASR memory cuts are allowed only if final streaming transcript quality holds. |
| Layer count / model structure | full layer count unless distilled and accepted | Talker layer dropping, codebook dropping | Prior notes mark layer dropping as large quality regression, not a quick memory fix. |

Precision-change gate:

- Treat BF16 as the floor for attention-heavy transformer compute unless a targeted calibration report proves otherwise.
- Any INT8 attempt must specify exactly which weights are quantized, which layers stay BF16, calibration corpus, memory saved, per-step latency delta, and quality delta.
- Any INT4/GGUF path is a separate runtime candidate, not a drop-in optimization. It must pass streaming TTFT and the same quality gate before it can replace the BF16 path.
- Never accept a precision or pruning change based only on "it generates audio" or "it fits memory".

## Success Criteria

Nano success means all of these pass on `orin-nano` with Qwen3-ASR and Qwen3-TTS, not Sherpa/Paraformer/Matcha:

- Qwen3-ASR and Qwen3-TTS are both available as long-running resident streaming services or workers under a documented production supervisor.
- `/health`, `/asr/info`, and `/tts/info` show Qwen3 backends ready after restart.
- The system can complete streaming V2V requests without OOM, worker crash, CUDA context failure, lazy first-chunk engine load, or forced manual reload.
- Warm V2V EOS-to-first-audio is measured through `/asr/stream` and `/tts/stream`. Initial memory-fit acceptance can record a temporary high latency, but the production gate must use the streaming-ready configuration with reusable contexts.
- First cold run after service start is recorded separately from warm runs.
- Recognition quality does not regress versus the Qwen3 ASR baseline: fixed audio corpus CER/WER and exact-match cases stay within thresholds.
- Generation quality does not regress versus the Qwen3 TTS baseline: fixed text corpus audio, ASR round-trip similarity/CER, duration/silence checks, and manual listen review stay within thresholds.
- Resident memory remains stable for at least 30 minutes and 100 sequential V2V runs, with no OOM kill, no restart, and no growing RSS trend.

## Phase 0 - Freeze Current Qwen3 Work

Objective: create a rollback point before changing memory-critical code.

Commit groups:

1. Qwen3 product/runtime code:
   - `app/backends/qwen3_asr.py`
   - `app/backends/qwen3_trt.py`
   - `app/backends/trt_edge_llm_tts.py`
   - `app/main.py` / `app/tts_service.py` if touched
2. Native worker and C++ runtime:
   - `native/edgellm_voice_worker/*`
   - `benchmark/cpp/*`
3. Export, profiling, and verification tools:
   - `scripts/verify_qwen3_tts_contract.py`
   - `scripts/verify_qwen3_tts_asr_roundtrip.py`
   - `scripts/verify_product_explicit_kv_tts.py`
   - `benchmark/export_*qwen3*`
   - Qwen3 profiling/sweep scripts that are intended to keep
4. Quality contracts and tests:
   - Qwen3 unit tests
   - Qwen3 hardware tests
   - `docs/contracts/*`
   - golden/reference assets that are actually used by gates
   - precision boundary notes from `docs/contracts/qwen3-tts-correctness-contract.md`, `docs/plans/qwen3-tts-accurate-continuation-2026-05-07.md`, and the Qwen3 handoff docs
5. Benchmark evidence and handoff docs:
   - `docs/benchmarks/*qwen*`
   - `docs/plans/*qwen*`
   - Nano OOM and NX comparison reports

Do not commit throwaway remote scripts, logs, or exploratory audio unless a plan or benchmark report references them as evidence.

Exit gate:

- Baseline commit hash is recorded.
- Local non-hardware Qwen3 tests/import checks pass where dependencies exist.
- The known Nano OOM baseline is preserved as a benchmark artifact.

## Phase 1 - Streaming Gap Accounting

Objective: quantify exactly how much memory is missing for the streaming-ready state before trying more strategies.

Current gap ledger:

| State | Evidence | Interpretation |
|---|---|---|
| Pre-pruning best run | `p1-nano-final-2026-04-28.md` | ASR resident, TTS load OOM; rough stable-production gap was 700-1000 MB. |
| After vocab pruning + mitigations | `handover-2026-04-29-tts-oom-int8.md` | Pipeline reaches ready, then first synth OOM; measured remaining gap is 250-400 MB. |
| Current `v3.5-clean` reference worker | `qwen3-streaming-gap-2026-05-08.md` | TTS worker OOMs before `ready`; graph-off does not help; current lazy Code2Wav env is unsupported by deployed worker; skipping ASR warmup saves ~600-700 MB and still fails. |
| Vocoder100 only | `qwen3-streaming-gap-2026-05-08.md` | Saves about 600 MB in Code2Wav but still fails first streaming warmup with ASR warmed; remaining practical gap about 800 MB. |
| Vocoder100 + TTS text pruning | `qwen3-streaming-gap-2026-05-08.md` | First current-stack dual-resident streaming success on Nano for short covered text; still only 27-121 MB headroom, so it is a fit proof, not production gate. |
| Streaming-ready target | Next soak/quality phase | Add several hundred MB more headroom, widen pruned vocab coverage, then measure warm-stream stability and V2V TTFT. |

Measure these four numbers on `orin-nano` before new optimization:

1. **Resident floor**: memory after Qwen3-ASR ready + Qwen3-TTS pipeline ready, before any synth call.
2. **First-stream peak**: memory floor during the first `/tts/stream` request until first audio chunk.
3. **Warm-stream peak**: memory floor during requests 2-10 after all reusable contexts are warm.
4. **Reusable-context delta**: first-stream peak minus resident floor, split by tag if instrumentation can isolate vocoder context, Talker first run, CP/KV, and Python/executor CUDA context.

Required report table:

| Config | Resident floor MemAvailable | First-stream min MemAvailable | Warm-stream min MemAvailable | TTFT p50/p95 | OOM point | Quality result |
|---|---:|---:|---:|---:|---|---|

Exit gate:

- The plan names the real streaming gap, not just load-time gap.
- Every candidate optimization is scored against that gap and its TTFT effect.

## Phase 2 - Establish Qwen3 Streaming Baseline Matrix

Objective: build a clear memory/latency matrix for the streaming path.

Run the same Qwen3 build on `orin-nano` and `orin-nx`, but only Nano results can close the Nano gate.

Baseline configurations:

1. ASR streaming-only resident:
   - Qwen3-ASR loads, creates `/asr/stream`, handles fixed audio corpus in real-time chunks.
   - Record RSS/HWM, `MemAvailable`, per-chunk latency, final latency, CER/WER.
2. TTS streaming-only resident:
   - Qwen3-TTS loads, runs `/tts/stream`, emits first audio chunk.
   - Record resident floor, first-stream peak, warm-stream peak, TTFT, total generation, quality gates.
3. Sequential same-process streaming load:
   - ASR first, then TTS.
   - TTS first, then ASR.
   - Record exact OOM point and component tag using streaming requests, not offline `/tts`.
4. Dual-worker streaming load:
   - ASR worker and TTS worker as separate processes.
   - Measure whether process separation changes peak, fragmentation, and TTFT.
5. Memory knobs with latency impact recorded:
   - `CP_POOL_SIZE=1`
   - `SKIP_ASR_WARMUP=1`
   - ASR TRT-native encoder where available
   - smaller ASR max sequence / KV engine
   - disable ASR and TTS CUDA graph
   - lazy TTS sub-engine load

Exit gate:

- A single Markdown table records for each config: ready/fail, OOM point, peak RAM, peak swap, ASR streaming latency/quality, TTS streaming TTFT/quality.
- The next optimization target is chosen from measured peak memory, not intuition.

## Phase 3 - Memory-Fit Workstream

Objective: get both Qwen3 streaming services resident without giving up the reuse needed for low latency.

Priority order:

1. Preserve and extend vocab pruning:
   - treat pruning as the best validated strategy so far because it is near-lossless and does not inherently add TTFT
   - confirm ASR pruned embed/lm_head and TTS pruned text_embed are active in the deployed binary
   - search for remaining unpruned vocab-sized allocations, especially CP/codebook logits, tokenizer side tables, and duplicate mappings
   - quantify any further pruning before changing quality-sensitive logic
2. Remove duplicated residency:
   - verify tokenizer/embed tables are not duplicated unnecessarily between ASR and TTS
   - share or mmap immutable CPU-side maps where possible
   - avoid Python copies of large token maps in both workers
3. Shrink ASR residency without hurting streaming:
   - ASR thinker max sequence/KV reduction
   - batch=1-only engine profiles
   - TRT-native encoder path instead of ORT if it lowers RSS
   - disable CUDA graph only if ASR streaming latency remains inside budget
4. Shrink TTS residency without moving work before first chunk:
   - `CP_POOL_SIZE=1`
   - lazy Code2Wav only as a diagnostic; production must prepay it before measured streaming requests or prove TTFT is unchanged
   - Code2Wav streaming-specific export should target the ONNX dummy/window length first. The current Qwen3-TTS tokenizer decoder ONNX freezes the 300-frame dummy into `waveform=1x1x576000`, so changing only TensorRT `max_code_len` is not enough.
   - For the current stream defaults, prioritize a `dummy_code_len=50`, `opt_code_len=25`, `max_code_len=50` Code2Wav spike, because emitted windows are normally `25` new frames plus `25` left-context frames.
   - 2026-05-08 spike result: a runner-compatible vocoder100 engine (`waveform=1x1x192000`, max `100` codec frames) saves about `600 MB` versus the reference Code2Wav and reduces first chunk Code2Wav time from about `679 ms` to `435 ms`, but still fails dual-resident first-stream warmup with ASR warmed (`143 MB` available before lazy Code2Wav, about `536 MB` minimum remaining gap).
   - 2026-05-08 combined result: vocoder100 plus TTS text embedding pruning reaches Qwen3 ASR+TTS dual-resident streaming and returns real `/tts/stream` PCM (`49,924` bytes for `你好`) with ASR warmup enabled. Minimum observed headroom is still too low (`27-121 MB` available), so this should be treated as the new baseline to harden, not the final production config.
   - Do not repeat the failed direct ONNX constant rewrite (`300 -> 50`, `576000 -> 96000`) without fixing the dependent shape/crop graph. ONNX Runtime exited during `sess.run` for that artifact.
   - The current pruned TTS vocab covers the warmup prompt and some common tokens but misses arbitrary Chinese text such as token id `104307` from `你好，今天天气很好。`; production pruning needs a wider coverage set before quality gates.
   - single-session CP/KV policy
   - split talker/CP/Code2Wav lifetimes only if it does not add engine reload or first-chunk delay
5. Reduce runtime overhead:
   - avoid PyTorch in resident paths
   - reuse fixed buffers in native runtime
   - inspect TRT binding dtype/shape and prevent hidden BF16/FP16 conversions from allocating extra buffers
6. Conditional compression:
   - INT8 Talker is allowed only if the measured latency hit still passes the streaming gate; prior notes expect +10-20 ms per step on Jetson small batch
   - INT4/GGUF is a fallback only if it can prove streaming TTFT and quality, not just memory fit
7. Reject latency-sacrificing fallbacks for the main path:
   - no hot swap between ASR and TTS
   - no unload ASR before TTS
   - no first-request engine deserialize
   - no "near-resident" label unless its added first-chunk latency is explicitly accepted as a separate product mode

Exit gate:

- Qwen3-ASR and Qwen3-TTS can both reach streaming-ready state on Nano without OOM.
- Peak physical RAM leaves practical headroom through first and warm streaming requests, not just a one-run survival with under 100 MB free.
- Quality gates still pass against the pre-memory-cut baseline.

## Phase 4 - Resident Supervisor and IPC

Objective: make the memory-fit configuration operable.

Actions:

1. Pick the supervisor shape:
   - one container with two worker processes, or
   - two containers with explicit memory limits and local IPC, or
   - one API service that owns two native workers
2. Define readiness:
   - ASR worker ready
   - TTS worker ready
   - model manifest and engine variant reported in `/asr/info` and `/tts/info`
3. Define restart policy:
   - worker-level restart for recoverable crash
   - container restart for unrecoverable CUDA state
   - log OOMKilled and engine variant in the report
4. Add smoke and soak scripts:
   - one Qwen3 ASR fixed audio request through `/asr/stream`
   - one Qwen3 TTS fixed text request through `/tts/stream`
   - one streaming V2V request from `/asr/stream` final text to `/tts/stream` first audio chunk
   - 100-run soak with memory sampling and restart count

Exit gate:

- Reboot or container restart returns both Qwen3 services to ready without manual intervention.
- 100-run soak has no OOM, no restart, and no progressive RSS growth.

## Phase 5 - Streaming Latency Optimization

Objective: reduce Qwen3 V2V latency after memory fit is stable.

Optimize one variable per benchmark:

1. ASR:
   - chunk size and endpointing
   - finalization delay
   - smaller KV/max sequence only if CER/WER holds
   - graph on/off only if it improves latency enough to justify memory
2. TTS:
   - first audio chunk policy
   - Code2Wav resident/warmed versus lazy tradeoff, with lazy rejected if it moves work before first chunk
   - CP cache/session policy
   - sampling settings only with quality review
   - CP per-shape CUDA graph cache if it can be enabled within the memory budget
   - system-prompt prefill KV cache
   - streaming prefill executor/CUDA-context fix
   - emit first chunk before the next Talker decode step
3. Runtime:
   - executor/thread pinning
   - CUDA stream ownership
   - buffer reuse
   - avoiding repeated engine/context creation

Exit gate:

- Each latency change includes before/after streaming p50, p95, cold, warm, peak memory, and quality report.
- No latency win is accepted if it worsens ASR CER/WER or TTS quality gates.

## Phase 6 - Quality Non-Regression Gates

Objective: make every Qwen3 memory or latency optimization prove quality.

Recognition gates:

- Fixed audio corpus with expected transcripts.
- Exact-match cases for short clear samples.
- CER/WER threshold for broader samples, recorded from the baseline commit.
- Streaming final transcript quality tracked separately from partial stability.

Generation gates:

- Fixed text corpus covering short Chinese, short English, mixed language, punctuation, and longer product text.
- Seeded/deterministic golden audio where possible.
- For non-bit-exact output:
  - non-empty audio
  - sample rate and PCM format correct
  - duration within expected band
  - no long leading/trailing silence
  - ASR round-trip similarity/CER within baseline threshold
  - manual listen review for precision, tokenizer, sampling, or vocoder changes

Exit gate:

- Every memory-fit or latency patch includes the benchmark path and quality report path.
- Precision changes such as INT8 are blocked until they pass both objective gates and manual listen review.

## Decision Points

Use these decisions to avoid drifting:

- If dual Qwen3 streaming-ready mode fits with at least several hundred MB headroom, continue to latency tuning on Nano.
- If it only fits with almost no headroom, keep optimizing memory before measuring latency seriously because first-stream allocations will be unstable.
- If true dual residency cannot fit after lossless cuts, decide between:
  - an explicitly non-primary near-resident mode with measured streaming latency penalty
  - split ASR and TTS across devices
  - requiring 16GB NX/AGX for Qwen3 full stack
- Do not re-open the zh/en path in this plan unless it is explicitly framed as fallback, not success.

## Recommended Immediate Next Steps

1. Treat the current practical gap as about `1.5 GB`, not the earlier `900-1200 MB` estimate. The instrumented run showed `158 MB` available at lazy TTS ready, `134 MB` before lazy Code2Wav on startup streaming warmup, and a TTS-only Code2Wav first load cost of about `1.29 GB`.
2. Use `/tts/stream` first PCM chunk as the readiness gate. `/health` can report ready after the startup TTS streaming warmup fails, so it is not sufficient for this goal.
3. Stop spending cycles on the existing `code2wav_stream100_nx_ws2048` engine for memory fit. It used about `1.36 GB` on lazy load and failed with illegal memory access.
4. Prioritize Code2Wav/vocoder memory work before smaller ASR-only cuts:
   - new streaming-specific Code2Wav export that does not materialize the fixed large output path (`trtexec` shows `waveform=1x1x576000` even for `codes=1x16x1`);
   - tactic/profile investigation only if it proves lower resident/context memory, not just smaller `max_code_len`; the existing `max_code_len=100` engine kept `max_time_steps=6000`, did not save memory, and failed at runtime;
   - runner changes that keep Code2Wav resident and warmed without extra first-user latency.
5. Keep ASR vocab pruning in the strategy list because it was historically the best safe cut, but do not expect a modest 35k-to-smaller sweep alone to close a `~1.5 GB` practical gap.
6. Commit the current Qwen3 work in the Phase 0 groups.
7. Add or tighten Qwen3 streaming quality gates before accepting any precision, profile, or vocoder-shape change.

## Updated Decision From 2026-05-08 Instrumented Runs

The latest measurement changes the priority order:

- `EDGE_LLM_TTS_LAZY_CODE2WAV=1` is a diagnostic tool, not a candidate production setting. It moves the failure from worker startup to first streaming chunk and therefore sacrifices the exact latency path this project is trying to protect.
- Code2Wav currently costs about `1.29 GB` to become resident on first use. Existing small-profile alternatives did not reduce this cost.
- The near-term technical target is a smaller or differently shaped Code2Wav/vocoder runtime, or a large no-quality-loss cut in TTS runtime memory. If that cannot reclaim about `1.5 GB`, the 8GB Nano target should be escalated to an explicit product decision rather than hidden behind lazy loading.

## TTS Talker Vocab Pruning Evaluation

Current deployed Qwen3-TTS EdgeLLM reference path:

- Manifest uses `talker_dir=/tmp/qwen3tts_ref_0507_from_nano/talker`.
- That directory still contains full text embedding: `text_embedding.safetensors`, tensor `text_embedding F16 [151936, 2048]`, file size `622,329,952` bytes.
- Historical pruned artifact exists separately at `/home/harvest/voice_test/models/qwen3-tts/onnx/text_embed_fp16_pruned.bin`, with `35,669` rows and file size `146,100,224` bytes.
- `token_map.bin` exists with `35,669` entries.
- Talker codec embedding is only `embedding F16 [3072, 1024]`; Talker output vocab is codec-space (`vocab_size=3072`), not the 151936 text vocab.

Conclusion:

- This EdgeLLM reference worker has not yet reused the older TTS text-vocab pruning path.
- Safe pruning target is only the Talker text embedding input table, not CodePredictor/codebook/codec embeddings.
- Expected memory recovery is about `622 MB - 146 MB = 476 MB` before allocator overhead. Historical notes measured this as about `454 MB`.
- This is worthwhile and low-latency-compatible, but it does not close the measured streaming gap alone: `134 MB + ~476 MB = ~610 MB` before lazy Code2Wav, still below the current `~1.29 GB` Code2Wav first-load cost.

Implementation shape:

- Convert or write the pruned table in the format the EdgeLLM runtime currently loads (`text_embedding.safetensors`), or teach it to load the existing `.bin`.
- Add `token_map.bin` loading in `Qwen3OmniTTSRuntime`.
- Keep tokenizer/API token IDs in original Qwen ID space.
- Map original IDs to reduced IDs immediately before `embeddingLookup()` for both normal text tokens and TTS special IDs (`tts_pad/bos/eos`).
- Fail loudly if a token is missing from the map; do not silently clamp.

Decision:

- Keep TTS Talker text-vocab pruning as strategy #2. It is materially useful and should be implemented after or alongside Code2Wav export work, but it is not a replacement for fixing Code2Wav/vocoder resident memory.

## Updated Decision From Vocoder50 + Pruned Text Run

Measured on `orin-nano` with ASR warmup enabled, TTS text embedding pruned to `35669` rows, and Code2Wav vocoder50 (`waveform=1x1x96000`):

- Startup warmup passed and real `/tts/stream` produced PCM while ASR stayed resident.
- Lowest observed startup `MemAvailable` improved from `27 MB` with vocoder100 to `86 MB` with vocoder50.
- Real repeated TTS requests stayed around `96-105 MB` available after final Code2Wav chunk.
- Swap remained active, and this is still only a fit proof, not a production headroom target.

Updated ordering:

1. Keep vocoder50 as the current best Code2Wav artifact for Nano 8GB.
2. Expand TTS pruned vocab only in small stages, then re-run the same dual-resident streaming gate.
3. Do not restore the full TTS text embedding on Nano 8GB; it would add roughly `454-476 MB`.
4. Continue looking for non-quality-loss Talker/CP memory reductions because Talker/CP runtime init alone drove available memory from about `4.0 GB` to `420 MB` before Code2Wav.

TTS pruned vocab expansion budget:

- Cost per token row is `2048 * 2 = 4096` bytes.
- `+5k` rows costs about `19.5 MB`.
- `+10k` rows costs about `39 MB`.
- `+20k` rows costs about `78 MB`.
- Given measured dual-resident headroom of only `~86-126 MB`, the next expansion should start at `+5k`, or `+10k` only if the token-coverage gain is clearly worth the risk.

Validation gate for every expanded vocab:

- ASR worker remains resident and warmed.
- TTS worker reaches ready.
- Startup TTS streaming warmup produces PCM.
- At least two real `/tts/stream` calls produce PCM.
- Record `worker_after_tts_runtime`, `worker_after_lazy_code2wav`, `worker_after_code2wav_final_chunk`, HTTP status, bytes, `time_starttransfer`, and `time_total`.

Operational lesson:

- Both ASR and TTS subprocess workers must drain stderr while waiting for JSON `ready`.
- ASR backend patches must preserve `EDGE_LLM_ASR_MANIFEST` parsing; otherwise the app falls back to default `/root/...` paths and silently disables the real Qwen3-ASR residency test.

## W8A16 Talker Trial Decision

W8A16 was tested after the vocoder50 baseline using existing Talker INT8 engines under `/home/harvest/voice_test/models/qwen3-tts/engines`.

Memory result:

- `talker_decode_int8.engine` is `445 MB` versus `876 MB` for `talker_decode_bf16.engine`.
- TTS-only `worker_after_tts_runtime` improved from about `2503 MB` available to about `3248 MB`.
- Dual-resident startup minimum improved from `86 MB` available with BF16/vocoder50 to `485 MB` available with W8A16/vocoder50.
- Real dual-resident `/tts/stream` stayed around `475 MB` available after final Code2Wav.

Quality/runtime result:

- The existing 445 MB W8A16 engines are single-profile, `inputs_embeds` max seqLen `1`.
- A temporary runtime patch made them runnable by using iterative prefill and by separating input max seqLen from past-KV max length.
- All tested W8A16 engines still produced only one codec frame for `你好` (`~80 ms`, `3840` PCM bytes).
- Forcing `QWEN3_TTS_MIN_EOS_FRAMES=10` and disabling auto EOS bias did not fix the one-frame output.

Decision:

- W8A16 is now a confirmed memory-recovery path, not a confirmed product path.
- Do not use it for production low-latency measurements until quality is repaired.
- Next W8 work should compare BF16 vs W8 numerical traces at:
  - prefill logits before/after logit adjustment;
  - sampled primary codec token sequence;
  - Talker hidden state feeding CodePredictor;
  - frame count and duration.
- If the single-profile iterative prefill is the quality source, rebuild W8A16 with a true prefill profile instead of accepting iterative prefill.
