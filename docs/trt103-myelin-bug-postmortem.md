# TRT 10.3 Myelin "already loaded binary graph" Bug — Postmortem

**Date**: 2026-04-19 → 2026-04-20
**Platform**: Jetson Orin NX 16GB, JetPack 6.2, TensorRT 10.3.0
**Affected**: Qwen3-TTS C++ TRT pipeline (`benchmark/cpp/tts_trt_engine.cpp`), specifically CP (code predictor) engine
**Status**: Mitigated via CP Engine Pool N=2 (commit `3777ce7` on `feature/cp-engine-pool`)

---

## Symptom

On Jetson, two consecutive `/tts/stream` requests with **same shape** (same text length):

- Request 1: normal
- Request 2: crashes with:
  ```
  [TRT] IExecutionContext::enqueueV3: Error Code 1: Myelin
    ([executor.cpp:myelinGraphLoad:836] Called with an already loaded binary graph.)
  [CPKV-AR] decode enqueueV3 FAILED
  ```

Different-shape requests back-to-back work fine. Real V2V in production was likely masking this — most consecutive requests differ in text length.

---

## Background: what is Myelin?

**Myelin is TensorRT 10's internal graph JIT compiler + executor.** TRT compiles optimized kernel sequences into "binary graphs" and Myelin loads + executes them at runtime. Entirely internal to TRT — no user-facing API, no knobs.

The error `myelinGraphLoad:836 Called with an already loaded binary graph` means: Myelin's loader tried to cache a shape-keyed binary graph, but saw its own cache already contains that key and hit an assertion. Should have been a cache hit; it misbehaves as a conflict.

---

## Root cause (confirmed)

**Myelin maintains a shape-keyed binary-graph cache scoped to the `ICudaEngine` object** (not the CUDA context, not the CUDA module loader).

When request 1's CP `enqueueV3` runs, Myelin caches the binary graphs keyed by the per-step KV shapes. Request 2 same-shape should cache-hit and replay; instead Myelin's state machine treats the cache entry as a duplicate load and bails.

This is a **TensorRT 10.3.0 regression bug**, not an application error. NVIDIA may have fixed it in TRT 10.4+, but:

- JetPack 6.x all ships TRT 10.3 (no upgrade path on current Orin)
- JetPack 7.x does not support Orin until at least Q2 2026 (likely slipping)

---

## Disproven hypotheses (11 total)

These have all been empirically tested and FAILED. Do not re-try.

| # | Hypothesis | Change | Result |
|---|---|---|---|
| 1 | Talker CUDA Graph capture mode pollutes CP | `cudaStreamCaptureModeGlobal → ThreadLocal` | Still crashes |
| 2 | Talker graph activity corrupts CP state | Python `enable_cuda_graph(False)` — disable Talker graph | Still crashes |
| 3 | IExecutionContext state leaks | `ResetExecutionContexts()` per request | Still crashes |
| 4 | Input shape bindings stale | `ResetInputShapes()` re-call `setInputShape` | Shape validation OK, Myelin still crashes |
| 5 | Profile state not re-initialized | `setOptimizationProfileAsync(profile, stream)` per request | Still crashes + new stream-capture race |
| 6 | Captured events leak in stream | `ResetCapturedEvents()` per request | Fixed line 798 event crash; Myelin unchanged |
| 7 | `cudaEventSynchronize` on captured event | Replace with `cudaStreamSynchronize(stream_)` | Event crash gone; Myelin unchanged |
| 8 | Dual-profile engine Myelin bug (TRT issue #2977) | Rebuild CP as single-profile (222MB from 444MB) | Still crashes identically |
| 9 | Dual-context on single engine triggers Myelin state machine | `has_dual_ctx_ = false`, single `context_` | Still crashes identically |
| 10 | CUDA module lazy-loading pollution | `CUDA_MODULE_LOADING=EAGER` env var | Still crashes |
| 11 | Shape padding to force variance per request | F4: append `1`/`12`/`123`/`1234` to text tokens | WORKS but digits get **pronounced** in audio (quality regression, unshippable) |

Lessons:
- All context-level / profile-level / CUDA-driver-level fixes fail → pollution source is **engine-scoped Myelin internal state**
- Text shape-padding works only because it changes Talker's output → CP's past sequence → CP shape varies. Cannot achieve this silently (no attention_mask binding on TRT engines; padding tokens are real embeds that propagate through attention and get decoded).

---

## V1 isolation test (key evidence)

Wrote `/tmp/v1_cp_isolation.py` (~60 lines) using `tensorrt` Python bindings on Jetson host. Loads **only** the CP engine, fires two same-shape `enqueueV3` calls in sequence.

Result:
```
=== enqueueV3 call #1 ===
=== enqueueV3 call #2 ===
=== Both calls OK — bug NOT reproduced in CP isolation ===
```

**CP alone never triggers the bug.** It requires Talker's `enqueueV3` to run first with the matching shape, which seeds Myelin's cache. Then CP with that same shape collides.

This proves the bug is **cross-engine Myelin-state contamination**, not CP-internal.

---

## Fix (deployed)

### CP Engine Pool N=2

- Maintain N independently-deserialized `ICudaEngine` instances in a `TRTCPKVEnginePool` wrapper
- Round-robin dispatch per request via atomic counter
- Per-slot `std::mutex` via RAII `SlotLease` serializes ops within a slot
- Two consecutive same-shape requests land on **different engine instances**, each with its own Myelin cache → no collision

**Files**:
- `benchmark/cpp/tts_trt_engine.h` — `TRTCPKVEnginePool` class
- `benchmark/cpp/tts_trt_engine.cpp` — pool impl
- `benchmark/cpp/tts_pipeline.cpp` — replaces `cp_kv_` with `cp_kv_pool_`, lease acquired at `GenerateInternal:442` / `GenerateStreaming:1018`

**Env vars**:
- `CP_POOL_SIZE` (default 2, clamped [1, 8]) — set to 1 for rollback
- `CP_GRAPH` (0/1, default 1) — enables T1 graph cache (stacked optimization)
- `CP_GRAPH_WARMUP=full|lazy` (default full) — startup-time graph capture

### Cost / benefit

| Metric | Before | After pool | After pool + T1 |
|---|---|---|---|
| Myelin errors on repeat requests | crash | **0** | 0 |
| Memory (container) | 3.7 GB | 3.85 GB | 4.3 GB |
| Streaming TTFT median | 528 ms | ~528 ms | **410 ms** |
| Streaming TTFT req1 | 528 ms | 528 ms | 438 ms |
| ASR round-trip quality | exact | exact | exact |

Stacked perf win (T1 per-`(actual_past, parity)` graph cache) became viable only after pool mitigation: T1 was blocked by the same Myelin bug until engine isolation.

### Why other approaches failed

| Approach | Why rejected |
|---|---|
| Context-only rotation (multiple `IExecutionContext` on one engine) | Myelin state is engine-scoped, not context-scoped — disproven hypothesis #3, #9 |
| `CUDA_MODULE_LOADING=EAGER` | Not a CUDA driver-level issue, Myelin is above that — disproven #10 |
| Cross-CUDA-primary-context isolation | If Myelin is engine-scoped (confirmed), separate CUDA contexts don't help. Also 8-16h implementation vs 3 day pool. |
| Shape padding (F4) | Works, but audio quality regression (pronounced digits) — unshippable |
| Wait for TRT 10.4+ via JetPack upgrade | No upgrade path for Orin in 2026 |
| Audio tail trim (pad + trim) | Token-to-frame ratio is heuristic (3-4x variable), prosody corruption on trailing tokens, adds 1.0-3.2s streaming holdback (breaks latency budget) |

Engine pool was the only option that preserves audio fidelity, is retractable (`CP_POOL_SIZE=1`), and fits memory budget.

---

## Debug methodology lessons

1. **Multiple agents misdiagnose Myelin error from logs** — sonnet claimed a string didn't appear when it did, opencode misattributed to dual-profile, codex misattributed to dual-context. Always CITE exact log lines, never paraphrase.
2. **nsys profile on Jetson requires `target-linux-tegra-armv8/` exact directory structure** into the container. Symlinking alone doesn't work.
3. **TRT verbose logger is useless for Myelin runtime events** — only fires during deserialize. For runtime-internal state, nsys CUDA API trace is the tool.
4. **Deploy+iterate cycle is ~10-12 min per attempt**. After 3 failed iterations, stop and do pure analysis. 9 hypotheses were burned partly because of insufficient analysis between attempts.
5. **CP isolation test cost was 30 min to write and saved the investigation.** Any cross-engine suspicion should have been A/B-tested earlier. Generalize: when a bug looks cross-component, write an isolation harness first.
6. **Error code 1: Myelin** is a nearly-opaque signal from NVIDIA. The only way to understand it was through empirical elimination + code-level reasoning about TRT's architecture.

---

## Related memory files

- `~/.claude/projects/-Users-harvest-project-jetson-voice/memory/project_trt103_myelin_cp_pool.md` — concise actionable version
- `docs/plans/handover-2026-04-19-v2.md` — morning-of-fix handover with 9-hypothesis table
- `docs/plans/tts-perf-backlog-2026-04-19.md` — perf context showing T1 was blocked behind this bug

## References

- NVIDIA TRT issue trackers: #2977, #4715 (dual-profile Myelin failures — partial context)
- JetPack 6.2 = TRT 10.3.0 (`/host-libs/libnvinfer.so.10.3.0`)
- JetPack 7.x Orin support: Q2 2026+ roadmap (slipping per internal NVIDIA comms)
