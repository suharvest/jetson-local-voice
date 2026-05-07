# Jetson Voice EdgeLLM Product Integration Plan

Date: 2026-05-07

Terminology note: `HLM` in earlier ASR/transcript notes refers to
TensorRT-Edge-LLM. It is not a separate product layer. In this document,
`EdgeLLM baseline` means our fork of NVIDIA TensorRT-Edge-LLM pinned to a known
commit, while `Jetson Voice product layer` means this repository.

This document describes how Jetson Voice should use NVIDIA TensorRT-Edge-LLM as the lower-level inference framework, while keeping product behavior and unavoidable Qwen3-TTS compatibility work in our own product layer.

The intended shape is not "Jetson Voice becomes an EdgeLLM fork". The intended shape is:

```text
Jetson Voice product layer
  - worker protocol
  - streaming policy
  - product config and model manifests
  - quality-compatible Qwen3-TTS backend selection
  - validation and reproducible examples for NVIDIA

Our EdgeLLM baseline repository
  - NVIDIA upstream EdgeLLM
  - minimal local patches needed to make Qwen3-TTS runnable/correct on Jetson
  - clearly documented commits that can become upstream PRs or issue references

NVIDIA TensorRT-Edge-LLM upstream
  - inference framework
  - builders, plugins, runners, examples
```

## Goal

Use TensorRT-Edge-LLM as the inference framework dependency, but keep Jetson Voice responsible for product behavior and model-specific compatibility that is not ready for upstream.

The immediate target is:

- Qwen3-TTS with acceptable audio quality on Jetson Orin Nano/NX.
- Resident worker execution for low startup overhead.
- Streaming-friendly frame emission.
- Clear separation between upstreamable framework fixes, local EdgeLLM baseline patches, and product-only policy.

## Repository Roles

### NVIDIA EdgeLLM upstream

This stays the canonical inference framework. We should not force product protocol, worker lifecycle, HTTP/WebSocket behavior, or our release policy into this repository.

### Our EdgeLLM baseline

This is the lower-level repository Jetson Voice references. It should stay close to NVIDIA upstream and contain only changes that are required to build, export, run, or diagnose Qwen3-TTS on Jetson.

Every change here must be classified as one of:

- `upstream-pr`: general framework fix that should become a small PR.
- `upstream-issue-reference`: larger or uncertain change that demonstrates an official limitation.
- `local-baseline`: needed by our product now, but too model-specific or too large to push before NVIDIA gives direction.

### Jetson Voice product layer

This repository owns the product-facing composition. It can reference our EdgeLLM baseline by source path, git submodule, pinned commit, or packaged build artifacts, but it should not hide product behavior inside the EdgeLLM fork.

Jetson Voice owns:

- native worker binaries and IPC protocol
- model manifest and engine selection
- streaming adapter
- text segmentation and sampling policy
- stateful explicit-KV Talker session lifecycle
- service/API compatibility
- validation scripts, reference audio, and issue repro assets

## Current Finding

The official EdgeLLM Qwen3-TTS path can build and run on Orin with CuTe DSL GEMM enabled for SM87, but the generated audio is audibly degraded in our validation.

The quality issue is not primarily caused by tokenizer, Code2Wav, sampling defaults, or CodePredictor after the fixes already investigated. The remaining divergence starts when residual codec feedback is fed back into the Talker loop.

The known-good product Talker engine has this TensorRT I/O shape:

- `inputs_embeds`: FP32
- `past_key_i` / `past_value_i`: FP32
- `new_past_key_i` / `new_past_value_i`: FP32
- `logits`: FP32
- `last_hidden`: FP32

The official EdgeLLM `LLMEngineRunner + AttentionPlugin` path uses FP16 Q/K/V and FP16 KV cache for this Talker path. Enabling Orin CuTe DSL improves the GEMM/kernel implementation, but it does not change the autoregressive loop precision boundary.

This means the official path is runnable, but not currently quality-equivalent to the known-good product path for Qwen3-TTS. That difference is the main thing we need to communicate to NVIDIA.

## What We Reuse From EdgeLLM

We should continue to reuse:

- Qwen3-TTS runtime orchestration where practical.
- Tokenizer and chat-template handling.
- Code2Wav runner and TensorRT engine execution.
- CodePredictor integration where it matches Qwen3-TTS semantics.
- Build system, plugin loading, logging, and profiling conventions.
- Jetson CuTe DSL SM87 artifacts for the official paths that can use them correctly.

## Product-Side Extensions

Jetson Voice should own these pieces:

- Worker protocol: JSONL/IPC, resident processes, lifecycle management.
- Streaming adapter: frame callback to product chunk/audio events.
- Product sampling defaults and text segmentation policy.
- Engine path selection and model manifest validation.
- A Qwen3-TTS Talker backend adapter that can use the FP32-boundary explicit-KV TensorRT engine when the official Talker path does not meet quality requirements.
- For the explicit-KV product backend, long text must remain a single
  Talker/CP session unless the product has an explicit state-transfer strategy.
  Generic multi-request segmentation changes the sampled codec trajectory and
  can sound like repeated speaker/timbre switches.

The Talker backend should be an internal adapter, not scattered environment-variable conditionals:

```text
TalkerBackend
  OfficialLlmRunnerTalkerBackend
  ExplicitKvTalkerBackend
```

The higher-level TTS flow should depend on a small interface:

```text
prefill(inputs) -> logits, hidden, kv_state
decode(residual_embedding) -> logits, hidden, kv_state
```

This keeps the workaround replaceable if NVIDIA later fixes the official precision path.

## Official-Aligned Compatibility Layer

The product layer should expose an EdgeLLM-compatible backend boundary, but the concrete backend can be selected from a manifest.

Recommended internal shape:

```text
Qwen3TtsService
  TtsStreamAdapter
    - maps frame callbacks into product chunk/done/error events
    - owns JSONL/WebSocket/HTTP response compatibility

  Qwen3TtsRuntimeAdapter
    - tokenizer / prompt formatting
    - sampling policy
    - CodePredictor runner
    - Code2Wav runner
    - TalkerBackend selection

  TalkerBackend
    OfficialEdgeLlmTalkerBackend
      - uses official LLMEngineRunner path
      - kept for benchmarking and future NVIDIA fixes

    ProductExplicitKvTalkerBackend
      - uses the known-good explicit-KV TensorRT Talker engine
      - preserves the FP32 loop boundary required by current validation
      - may keep KV cache state inside a product-owned session
      - lives behind the same backend interface
```

This keeps Jetson Voice close to official EdgeLLM operationally, but avoids blocking the product on an upstream precision decision.

## What Is Unavoidable For Correct Qwen3-TTS Today

These are the items we should tell NVIDIA clearly. They are not just product preferences.

| Area | Why it matters | Current handling |
|---|---|---|
| Talker loop precision boundary | Official FP16 Q/K/V + FP16 KV cache path produces degraded audio in our Orin validation. Known-good path uses FP32 I/O/KV boundary around the autoregressive Talker loop. | Product backend adapter uses explicit-KV Talker engine until official path supports equivalent precision semantics. |
| Qwen3-TTS frame callback | Product streaming needs codec-frame emission before final wav assembly. This is a minimal runtime callback, not a product protocol. | Add/keep a small callback in our baseline; propose upstream only as generic `onCodecFrame(frameCodes, totalFrames)`. |
| Export/checkpoint robustness | Reproducible model export should support realistic HF checkpoint layouts, including sharded safetensors. | Keep local fix and propose as normal upstream robustness PR if still missing. |
| Builder robustness | Failed parser fallback must not reuse a polluted TensorRT network; env parsing should be strict. | Keep local fix and propose as separate upstream robustness PR if still missing. |
| Jetson SM87 build support | Orin needs deterministic SM87/CuTe build behavior. Official CuTe GEMM can work on Orin when configured correctly. | Keep build recipe in Jetson Voice docs; upstream only if source still has a generic CMake defect. |

## What Stays Product-Side

These should not be pushed to NVIDIA as framework changes unless NVIDIA asks for them:

- Jetson Voice worker protocol.
- Stateful resident worker lifecycle.
- Product streaming chunk schema.
- Text splitting and punctuation policy.
- Sampling defaults tuned for our service quality.
- Model manifest format and deployment layout.
- Explicit-KV cache session policy and validation assets.
- Explicit backend selection policy.

The product layer can include stateful worker support and explicit-KV cache handling because those are Jetson Voice concerns. The official issue can link to them as context, but should not require NVIDIA to adopt the product session protocol.

Stateful KV cache rules for the product backend:

- One KV cache state belongs to exactly one product request/session.
- Reset KV on new utterance, language/speaker change, error, or max-frame/max-token limit.
- Never reuse KV across users or unrelated text segments.
- Expose statefulness through a small product adapter, not through scattered environment variables.
- Seed all stochastic samplers in the request, including primary Talker sampling
  and CodePredictor residual-code sampling. A fixed request seed that only
  covers primary codes is insufficient; residual codes feed back into the
  Talker embedding for the next frame.
- Preserve the old Python reference sampler semantics for fixed-seed product
  output: NumPy `RandomState(seed)` uniform stream, top-k masking, and
  vocab-order categorical sampling. A C++ sampler that samples over a sorted
  top-k list can choose different CP residual codes from the same logits and
  seed.
- Generate all 15 CodePredictor residual codebooks by default. The historical
  `cp_active_groups=13` performance setting zero-fills the last two residual
  groups; a fixed-seed comparison showed that this changes every later frame
  and 75/100 primary codes for the long validation prompt, so it is not a
  correctness candidate unless separately re-qualified.
- The current TRT vocoder engine accepts up to 200 frames but its output tensor
  is fixed at 192000 samples (8 seconds). Product offline requests using this
  engine cap generation at 100 frames to avoid empty WAV output. Full-length
  longer utterances need either a re-exported long-output vocoder engine or a
  quality-approved ORT/streaming vocoder path.

## What We Should Tell NVIDIA

Open an upstream issue, not a large PR, for the precision/audio-quality finding.

The issue should include:

- Model: Qwen3-TTS 0.6B.
- Device: Jetson Orin, SM87.
- Official path: `LLMEngineRunner + AttentionPlugin`, CuTe DSL GEMM enabled.
- Symptom: generated audio is understandable only for very short spans or sounds sandy/degraded compared with the FP32-boundary explicit-KV TensorRT engine.
- Evidence: TensorRT engine I/O dtype comparison and fixed-text audio/code divergence.
- Minimal repro: fixed text, fixed sampling params, official engine command, generated wav, reference wav/RVQ codes.
- Request: guidance or support for precision-sensitive autoregressive audio models where KV/input loop state must remain FP32 or equivalent.

Do not frame this as "CuTe DSL is missing on Orin"; current evidence shows Orin SM87 CuTe GEMM can load and run. The issue is precision semantics, not CuTe availability.

The issue should link to our Jetson Voice repo/branch and point to:

- exact EdgeLLM baseline commit
- exact Jetson Voice commit
- model/export/build commands
- official-path output wav
- product-path output wav
- dtype dump for both Talker engines
- ASR/listening validation notes

## Upstream Candidates We Can Still Submit Separately

These are independent of the Talker precision workaround:

- Jetson build compatibility where confirmed against current upstream.
- Export/checkpoint robustness, especially sharded safetensors handling.
- Builder robustness, including fresh network creation after parse fallback and strict workspace env parsing.
- Minimal Qwen3-TTS frame callback for streaming:

```text
onCodecFrame(frameCodes, totalFrames)
```

These should be separate PRs and should not include the explicit-KV Talker backend unless NVIDIA asks for that direction.

## Patch Classification

Use this table when organizing commits:

| Patch group | Repo | Upstream action |
|---|---|---|
| Minimal codec-frame callback | EdgeLLM baseline | Candidate PR, small and generic. |
| Sharded safetensors / export robustness | EdgeLLM baseline | Candidate PR if missing upstream. |
| Builder parser fallback / workspace env validation | EdgeLLM baseline | Candidate PR if missing upstream. |
| SM87/CuTe build recipe or CMake defect | EdgeLLM baseline or Jetson Voice docs | PR only for source defect; otherwise product docs. |
| Explicit-KV Qwen3-TTS Talker backend | Jetson Voice product layer, possibly EdgeLLM baseline as local reference | Issue reference first, not immediate PR. |
| Stateful worker / KV cache compatibility / product protocol | Jetson Voice product layer | No upstream PR. |
| Streaming server protocol | Jetson Voice product layer | No upstream PR; only generic callback may go upstream. |

## Verified Product Path

Validated on `orin-nano` on 2026-05-07:

- EdgeLLM baseline source: `/tmp/edgellm-pr-qwen3-clean-0507`
- EdgeLLM baseline build: `/tmp/edgellm-pr-qwen3-clean-0507/build_clean_sm87_cutedsl_allprs`
- Product worker build: `/tmp/jetson-voice-product-layer-0507/build/edgellm_voice_worker`
- Required baseline patch for worker streaming: minimal `CodecFrameCallback` in `Qwen3OmniTTSRuntime`
- Baseline callback patch reference: `docs/patches/edgellm-codec-frame-callback.patch`
- Product backend mode: `JETSON_VOICE_TTS_BACKEND=product_explicit_kv`
- Product model base: `/home/harvest/voice_test/models/qwen3-tts`

Commands/results:

- Python unit tests: `python -m pytest app/tests/test_trt_edge_llm_tts.py app/tests/test_trt_edge_llm_ipc_paths.py` -> 11 passed.
- EdgeLLM baseline rebuild after callback patch: `edgellmCore` and `NvInfer_edgellm_plugin` built successfully.
- Jetson Voice worker rebuild: `qwen3_tts_worker` built successfully.
- Product pybind rebuilt on `orin-nano`: `/home/harvest/qwen3_speech_engine.cpython-310-aarch64-linux-gnu.so` and `/home/harvest/voice_test/app_overlay/qwen3_speech_engine.cpython-310-aarch64-linux-gnu.so`.
- Short product explicit-KV output after CP sampler seed fix: `qwen3tts-listen-0506/product-layer-0507/cp_seedfix_short_0507.wav`
- Default product explicit-KV long output after disabling generic segmentation and capping TRT vocoder frames: `qwen3tts-listen-0506/product-layer-0507/product_default_stateful_cap100_0507.wav`
- Exact long-punctuation reference text with TRT vocoder: `qwen3tts-listen-0506/product-layer-0507/cp_seedfix_long_punct_trtvoc_0507.wav`
- Stress comparison using streaming/chunked vocoder for >100 frames: `qwen3tts-listen-0506/product-layer-0507/cp_seedfix_long_streaming_0507.wav`

The old segmented samples are kept only as failure comparisons. They are not
the product-correct path for explicit-KV because each segment starts a new
request/session and can sound like a different speaker. The current product
path uses one explicit-KV session for a request; for the installed TRT vocoder
engine it caps offline output at 100 frames until we re-export a longer-output
vocoder engine or approve an alternate long-form vocoder path.

## Local Validation Matrix

Before treating the product path as releasable, validate:

- Correctness contract:
  `python3 scripts/verify_qwen3_tts_contract.py --run-sample --expect-frame-cap`
- Contract document:
  `docs/contracts/qwen3-tts-correctness-contract.md`
- Short Chinese text.
- Long Chinese punctuation-heavy text.
- Mixed Chinese/English text.
- Multi-question text.
- Generated audio pulled locally for listening.
- ASR round-trip sanity check.
- Runtime profile:
  - TTFT
  - RTF
  - peak memory
  - per-stage timing for Talker, CodePredictor, Code2Wav

Compare at least three paths:

- Official EdgeLLM Talker path.
- Product explicit-KV Talker path.
- Previous known-good product engine path.

## Near-Term Implementation Steps

1. Pin the Jetson Voice product repo to our EdgeLLM baseline commit.
2. Move direct-Talker environment variables behind config/manifest-based backend selection.
3. Formalize `TalkerBackend` and `TtsStreamAdapter`.
4. Add a small reproducible comparison script and store outputs under a dedicated validation directory.
5. Add a Jetson Voice README section that explains how to build the EdgeLLM baseline and then build product workers against it.
6. Draft the NVIDIA issue with links to this repo and the repro assets.
7. Revisit upstream PRs only after the product path is stable and the issue discussion clarifies NVIDIA's preferred direction.
