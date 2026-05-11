## Description
On Jetson Orin NX with JetPack 6.2 / TensorRT 10.3.0, a multi-profile TensorRT engine with dynamic KV-cache dimensions can fail on the second `enqueueV3` decode step when the input shape matches a prior call. Different-shape requests pass; rotating independently deserialized engines avoids the failure.

## Environment
- Platform: Jetson Orin NX 16GB (nvidia-smi: `Orin (nvgpu)`)
- JetPack: 6.2 / L4T R36.4.3
- TensorRT: 10.3.0 (`libnvinfer.so.10.3.0`, package `libnvinfer-bin 10.3.0.30-1+cuda12.5`)
- CUDA Toolkit: 12.6 (12.6.11-1 arm64)
- Driver: 540.4.0 (CUDA driver API reports CUDA 12.6)
- Python: 3.10
- `tensorrt` / `cuda-python` packages: 10.3.0 / 12.x

## Minimal reproduction

**Important caveat**: isolated-engine repro does NOT reproduce the bug. On our affected platform, the included `repro.py` (single-engine, two same-shape `enqueueV3` calls) exits 0 cleanly. The bug requires a specific **cross-engine call pattern**: Engine A's `enqueueV3` runs first, then Engine B's `enqueueV3` runs with a correlated shape, and on the SECOND such sequence (i.e. second request), Engine B's decode fails.

In our production pipeline:
- Engine A = Talker (autoregressive, multi-layer with its own KV cache)
- Engine B = CP ("code predictor", separate engine, dynamic `past_length` on input)
- Pattern: T.prefill → T.decode × N → B.decode × M per request; two requests back-to-back with identical text length crash on request 2's first B.decode.

This suggests Myelin's internal graph-key cache has state that is **engine-scoped** (isolated by rotating independent `ICudaEngine` deserializations, confirmed empirically) but gets **corrupted by cross-engine interaction** in a single process.

We are happy to provide both engine plan files privately for root-cause analysis. Included `repro.py` is a scaffold that can be extended to cover the two-engine pattern once we understand which shape tuple is the actual collision key.

### Single-engine scaffold

```bash
python3 repro.py /path/to/engine.plan
```

The script deserializes the plan, auto-selects the decode-like optimization profile (scored by max>0 dynamic dims), binds inputs at opt shape, and runs `enqueueV3` twice. On our platform this exits 0 — we include it to give a starting point for NVIDIA's engineers to extend. On a hypothetical platform where the bug triggers even in single-engine isolation, the second call would fail with:

```text
[TRT] IExecutionContext::enqueueV3: Error Code 1: Myelin
  ([executor.cpp:myelinGraphLoad:836] Called with an already loaded binary graph.)
```

## Expected behavior
`enqueueV3` with the same input shape as a prior call should succeed via graph replay / cache hit, not raise a Myelin state error.

## Actual behavior
The second same-shape decode call fails with:

```text
[TRT] IExecutionContext::enqueueV3: Error Code 1: Myelin
  ([executor.cpp:myelinGraphLoad:836] Called with an already loaded binary graph.)
```

## What we tested (disproven hypotheses)
- CUDA graph capture mode: changing global capture mode to thread-local did not help.
- CUDA graph activity: disabling graph execution around decode steps did not help.
- Execution context leakage: destroying/recreating `IExecutionContext` per request did not help.
- Stale input shapes: rebinding all input shapes per request validated shapes but still failed.
- Profile state: re-calling `setOptimizationProfileAsync` per request did not help.
- Captured event leakage: recreating events removed a secondary event error only.
- Event synchronization: replacing event sync with stream sync removed the secondary event error only.
- Dual-profile engine: rebuilding as a single-profile engine still failed identically.
- Dual-context use: forcing a single execution context still failed identically.
- CUDA module loading: `CUDA_MODULE_LOADING=EAGER` did not help.
- Shape variance padding: forcing shapes to differ avoided the error but changed decoded output, so it is not a valid fix.

## Current workaround
N independently-deserialized `ICudaEngine` pool with per-request rotation. Cost is approximately +220MB per extra engine. This proves the collision is engine-scoped, not process/context-scoped.

## Ask
1. Confirm whether this is a known issue in TRT 10.3.0
2. Provide a fix path (expected TRT release? existing patch?)
3. Clarify the scope of Myelin's shape-keyed graph cache across engines within a single process
4. We can share both engine plan files (~440MB + ~720MB) privately for root-cause analysis; please let us know the appropriate channel.
