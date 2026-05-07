# Qwen3-TTS Accurate Continuation Plan

Date: 2026-05-07

This document defines how to continue from the known-good Qwen3-TTS baseline
without mixing experimental paths back into the product or the EdgeLLM fork.

Terminology:

- `EdgeLLM` means NVIDIA TensorRT-Edge-LLM.
- `EdgeLLM fork` means our local fork of NVIDIA EdgeLLM with Qwen3-TTS changes.
- `Jetson Voice` means the product repository and product runtime.
- Earlier transcript mentions of `HLM` should be read as `EdgeLLM`.

## Fixed Accurate Baseline

The current accurate baseline is:

```text
EdgeLLM fork:
  repo:   /Users/harvest/project/tensorrt-edge-llm
  branch: stable/qwen3tts-direct-bf16-nx-20260507
  commit: b9c57c8 Freeze Qwen3-TTS direct BF16 reference runtime

Jetson Voice:
  repo:   /Users/harvest/project/jetson-voice
  branch: qwen3tts-accurate-20260507
  commit: 794d3db Freeze Qwen3-TTS accurate reference path
```

The EdgeLLM fork branch was recovered from the exact runtime source that
produced correct output:

```text
orin-nano:/tmp/tensorrt-edge-llm-upstream-runtime-0505
```

The Orin NX validation runtime is:

```text
/tmp/qwen3tts_ref_0507_from_nano
```

The known-good runtime requires:

```text
QWEN3_TTS_SEED=42
QWEN3_TTS_DIRECT_TALKER_ENGINE=/home/harvest/voice_test/models/qwen3-tts/engines/talker_decode_bf16.engine
QWEN3_TTS_HOST_TEXT_PROJECTION=1
EDGELLM_PLUGIN_PATH=/tmp/qwen3tts_ref_0507_from_nano/build/libNvInfer_edgellm_plugin.so
```

Correctness-critical properties:

- explicit-KV Qwen3-TTS Talker path
- BF16 Talker decode engine
- host FP32 text projection
- BF16 CodePredictor engine and weights
- complete tokenizer export files
- Code2Wav engine from the validated Nano runtime
- request-wide seed propagation, including CodePredictor sampling

## Repository Responsibilities

### EdgeLLM fork

The EdgeLLM fork should contain only lower-level runtime changes that are needed
to make Qwen3-TTS accurate on Jetson, or small generic fixes that may later be
sent to NVIDIA.

Allowed here:

- Qwen3-TTS explicit-KV Talker runtime support.
- Precision handling required by the accurate path.
- Minimal codec-frame callback if needed by streaming.
- Build/export/runtime robustness fixes that are general EdgeLLM issues.
- Diagnostic logs that prove which backend, engine, dtype, and plugin were used.

Not allowed here:

- Jetson Voice IPC protocol.
- Product WebSocket/JSONL/HTTP response schema.
- Product text segmentation policy.
- Product session routing, user lifecycle, or API behavior.
- Product-specific fallback ordering unless hidden behind a generic runtime
  adapter boundary.

### Jetson Voice

Jetson Voice owns the product composition around the EdgeLLM fork.

Allowed here:

- Worker protocol and resident process lifecycle.
- Streaming adapter and product chunk/done/error events.
- Model manifests and engine path selection.
- Product request validation and correctness contract checks.
- Stateful request/session policy.
- ASR round-trip validation scripts and reference audio collection.
- Clear selection between `OfficialEdgeLlmTalkerBackend` and
  `ProductExplicitKvTalkerBackend`.

Not allowed here:

- Silent monkey-patching of EdgeLLM internals.
- Multiple hidden backend entry points that can run different code for the same
  request without printing the selected backend.
- Quality-affecting environment variables that are not reflected in the model
  manifest or validation logs.

## Development Branch Strategy

Use three separate lines of work.

### 1. Accurate baseline

Purpose: freeze the known-good reference.

Branches:

```text
EdgeLLM fork:  stable/qwen3tts-direct-bf16-nx-20260507
Jetson Voice:  qwen3tts-accurate-20260507
```

Rules:

- Do not add experimental code to these commits.
- Use them as comparison targets for every future change.
- If we need to regenerate reference audio, record the command, runtime path,
  and output files in docs.

### 2. EdgeLLM fork integration

Purpose: turn the recovered accurate runtime into a maintainable fork.

Recommended branch:

```text
qwen3tts-edgellm-fork-clean
```

Work to keep:

- direct BF16 / explicit-KV Talker support
- host FP32 text projection path
- CP BF16 compatibility
- minimal frame callback if streaming needs it
- required build/runtime diagnostics

Work to remove:

- one-off test environment switches
- abandoned official-path experiments
- quality experiments that are not part of the accurate contract
- product protocol code

Expected output:

- a small set of commits that can be read as a product-maintained EdgeLLM fork
- a README section explaining build, engine placement, and validation
- a minimal reproducible example that NVIDIA can run or inspect

### 3. Jetson Voice product integration

Purpose: make the product use the EdgeLLM fork through one explicit backend
adapter.

Recommended branch:

```text
qwen3tts-product-edgellm-fork-integration
```

Required product shape:

```text
Qwen3TtsService
  TtsStreamAdapter
  Qwen3TtsRuntimeAdapter
    OfficialEdgeLlmTalkerBackend
    ProductExplicitKvTalkerBackend
```

The product default should be `ProductExplicitKvTalkerBackend` until the
official EdgeLLM Talker path is quality-equivalent.

Every run must print or record:

- selected Talker backend
- EdgeLLM fork commit or build ID
- plugin `.so` path
- Talker engine path and dtype summary
- CodePredictor engine path and dtype summary
- Code2Wav engine path
- tokenizer directory
- seed
- frame count and stop reason
- streaming/offline mode

## NVIDIA Issue Strategy

Create an issue first, not a large PR.

The issue should say:

- We can produce correct Qwen3-TTS output on Jetson Orin using an EdgeLLM-based
  fork.
- The official path can run, but the audio quality diverges from the accurate
  explicit-KV/BF16 path.
- The strongest current evidence points to Talker precision/runtime semantics:
  the accurate path preserves the precision-sensitive autoregressive loop
  differently from the official LLM runner path.
- We are not asking NVIDIA to adopt Jetson Voice's product protocol.
- We are providing a minimal reproducible example and reference audio so NVIDIA
  can decide which fixes belong upstream.

Issue assets to provide:

- EdgeLLM fork branch and commit.
- Jetson Voice branch and commit.
- Build commands.
- Runtime command.
- Correct WAV outputs.
- Official-path comparison WAV if available.
- Dtype/engine dump for both paths.
- Notes about device, JetPack, TensorRT, CUDA, and model version.

Potential separate PRs after the issue:

- minimal codec-frame callback
- export/checkpoint robustness
- builder robustness
- Jetson SM87 build fixes only if confirmed missing from current upstream

Do not submit the explicit-KV product backend as an upstream PR until NVIDIA
comments on the desired runtime direction.

## Quality Gate

No new path should be called correct unless it passes this gate.

Required sample set:

- short Chinese
- long Chinese punctuation-heavy text
- mixed Chinese/English text
- English Seeed Studio text
- Japanese Seeed Studio text

Required checks:

- generated WAV is saved and pulled locally
- ASR round-trip is recorded when an ASR service is available
- runtime log proves the selected backend and engine paths
- comparison against the fixed accurate baseline is possible
- if long text is used, frame cap or long-vocoder behavior is explicitly logged

Reference output directory currently used for the three-language sample:

```text
/private/tmp/sherpa-tts/out/
```

Files:

```text
zh_qwen3tts_seeed_conversational_ai.wav
en_qwen3tts_seeed_conversational_ai.wav
ja_qwen3tts_seeed_conversational_ai.wav
```

## Performance Plan

Performance work is allowed only after the accurate path remains reproducible.

Priority order:

1. Resident worker startup amortization.
2. Stateful explicit-KV session management inside one request/session.
3. Streaming codec-frame callback and chunked Code2Wav path.
4. KV cache memory reuse with strict request isolation.
5. Long-output vocoder re-export or approved streaming vocoder path.

Stateful KV rules:

- one KV state per request/session
- reset on new utterance, language/speaker change, error, or stop condition
- never reuse KV across users
- print state reset/reuse decisions in debug logs
- keep product session policy in Jetson Voice, not in EdgeLLM fork

## Immediate Next Steps

1. Create a clean EdgeLLM fork branch from
   `stable/qwen3tts-direct-bf16-nx-20260507`.
2. Remove experimental code that is not needed for the accurate runtime.
3. Add or keep only the minimal runtime diagnostics needed to prove the path.
4. Create a clean Jetson Voice integration branch that selects the EdgeLLM fork
   through one backend adapter.
5. Run the five-sample quality gate on Orin NX.
6. Update the NVIDIA issue draft with branch links, commands, and outputs.
