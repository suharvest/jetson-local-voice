# EdgeLLM upstream cleanup plan for Qwen3-TTS

Date: 2026-05-07

EdgeLLM worktree:

```text
/Users/harvest/project/jetson-voice/.worktrees/tensorrt-edge-llm-official-qwen3-tts
```

Current branch:

```text
official-qwen3-tts-upstream-runtime
```

Base:

```text
NVIDIA/TensorRT-Edge-LLM origin/main at bbbab9a
```

Current state:

```text
HEAD is ahead of origin/main by 7 commits.
The worktree also has a large uncommitted diff from the final quality investigation.
Do not open upstream PRs from the dirty tree as-is.
```

Clean branches created from `origin/main`:

```text
pr-jetson-build-compat
pr-export-builder-robustness
pr-qwen3-tts-runtime-correctness
```

Current clean branch heads:

```text
pr-jetson-build-compat              395b1e7 Fix CuTe DSL CUDA 12.6 static linking
pr-export-builder-robustness        5d8b9cf Harden ONNX parsing and safetensors shard loading
pr-qwen3-tts-runtime-correctness    ae57c29 Expose Qwen3-TTS generation controls
```

Product integration branch:

```text
product-qwen3-tts-official-backend
```

## Current committed stack

```text
8bee99e Fix Qwen3-TTS LLM export checkpoint loading
891e84b Harden ONNX parsing and safetensors shard loading
fca4359 Respect explicit CUDA architectures
3f880f8 Fix CuTe DSL CUDA 12.6 static linking
73afadd Align Qwen3-TTS runtime semantics
8a0df08 Add native Qwen3-TTS CodePredictor engine path
631e7f8 Address Qwen3-TTS upstream review findings
```

## What is necessary for upstream

### PR 1: Jetson build compatibility

Keep:

```text
fca4359 Respect explicit CUDA architectures
3f880f8 Fix CuTe DSL CUDA 12.6 static linking
```

Why upstream:

- Generic Jetson/Orin build compatibility.
- Not product-specific.
- Does not depend on Qwen3-TTS runtime behavior.

Scope:

- `CMakeLists.txt`
- `cmake/CuteDsl.cmake`
- `cpp/kernels/talkerMLPKernels/cuteDslGemmRunner.h`

Action:

- Keep as a small standalone PR.
- If desired, fold in only a short comment explaining why explicit CUDA architectures must be respected.

Status:

- Ready in principle.
- Verified on Orin Nano as part of the combined clean validation below.

### PR 2: Export and builder robustness

Keep:

```text
8bee99e Fix Qwen3-TTS LLM export checkpoint loading
891e84b Harden ONNX parsing and safetensors shard loading
```

Why upstream:

- Correct checkpoint loading for Qwen3-TTS export.
- Sharded `model.safetensors.index.json` support is generic HF checkpoint handling.
- ONNX fallback parsing with a fresh network is generic builder correctness.
- Strict workspace env parsing is generic builder robustness.

Scope:

- `tensorrt_edgellm/llm_models/model_utils.py`
- `tensorrt_edgellm/llm_models/models/qwen3_omni_talker.py`
- `tensorrt_edgellm/llm_models/layers/...` only if needed by export correctness
- `cpp/builder/builderUtils.{h,cpp}`
- builder call sites updated for the new parser API
- `tests/python-unittests/test_qwen3_tts_export.py`

Action:

- Keep, but re-review the final squashed diff so it only contains export/builder robustness.
- Do not include runtime quality heuristics or product path logic.

Required validation before PR:

- WSL/x86 export test with `nvidia-modelopt` installed.
- Python unit tests for single-file and sharded safetensors.
- Orin-side runtime validation consumes the exported/runtime artifacts successfully in the combined clean validation below.

### PR 3: Qwen3-TTS runtime correctness

Keep in upstream form:

```text
73afadd Align Qwen3-TTS runtime semantics
8a0df08 Add native Qwen3-TTS CodePredictor engine path
631e7f8 Address Qwen3-TTS upstream review findings
```

Why upstream:

- Official backend could run but was not semantically correct for Qwen3-TTS in our validation.
- Native Qwen3-TTS CodePredictor engine path is a framework-level model support gap, not a product feature.
- Runtime fixes for active residual groups, language/control token layout, top-p sampling, RAII CUDA buffers, and generic CP fallback preservation are correctness fixes.

Scope:

- `cpp/runtime/qwen3OmniTTSRuntime.{h,cpp}`
- `cpp/kernels/talkerMLPKernels/talkerMLPKernels.{h,cu}`
- `cpp/builder/llmBuilder.{h,cpp}`
- Qwen3-TTS export wrapper changes needed to emit the native CP ONNX path
- Minimal example changes needed to pass `--codePredictorEngineDir`

Action:

- Keep as one Qwen3-TTS-specific PR, but clean the current dirty tree first.
- Fold `631e7f8` into the runtime/CP commits before upstream review.
- Keep generic CP fallback behavior unchanged when `qwen3_tts_cp.engine` is absent.

Required validation before PR:

- Orin native build:
  - `qwen3_tts_inference`
  - `llm_build`
  - `NvInfer_edgellm_plugin`
- End-to-end TTS on Orin using official backend artifacts.
- Short and long Qwen3-TTS samples:
  - Chinese short
  - Chinese long punctuation
  - mixed Chinese/English
  - multi-question text
- ASR semantic check for generated samples.

Status:

- Orin Nano native build passed for `NvInfer_edgellm_plugin` and `qwen3_tts_inference` when combined with `pr-jetson-build-compat`.
- Direct runtime validation generated short Chinese, short mixed Chinese/English, long Chinese punctuation, and mixed question samples through the official backend binary.
- Audio samples pulled to:

```text
/Users/harvest/project/jetson-voice/qwen3tts-listen-0506/clean-allprs-0507/
```

Build command used for the passing Orin validation:

```bash
cmake -S . -B build_clean_sm87_cutedsl_allprs \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCUDA_DIR=/usr/local/cuda-12.6 \
  -DCUDA_CTK_VERSION=12.6 \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DENABLE_CUTE_DSL=gemm \
  -DCUTE_DSL_ARTIFACT_TAG=sm_87
cmake --build build_clean_sm87_cutedsl_allprs \
  --target NvInfer_edgellm_plugin qwen3_tts_inference -j1
```

Important finding:

- `pr-qwen3-tts-runtime-correctness` by itself can build without CuTe DSL, but Qwen3-TTS Talker MLP then fails at runtime because the official path needs CuTe DSL GEMM.
- Enabling CuTe DSL GEMM on Jetson with CUDA 12.6 requires the generic build compatibility changes from `pr-jetson-build-compat`, especially `-DEMBEDDED_TARGET=jetson-orin`.
- Therefore final product validation should combine PR 1 + PR 2 + PR 3, even if upstream review receives them as separate PRs.

### PR 4: Formal generation controls

Added to the clean runtime branch as formal request fields:

- `codec_eos_logit_offset`
- independent CodePredictor sampling params:
  - `predictor_temperature`
  - `predictor_top_k`
  - `predictor_top_p`
- `min_audio_length`

Why separate:

- These are API/behavior changes, not pure correctness fixes.
- Reviewers may accept them if they are generic optional controls with default-off behavior.
- They should not be mixed into PR 3 unless required for correctness.

Action:

- Do not upstream the current env-var form.
- The clean branch uses request/config fields and documents defaults.
- Defaults should preserve official Qwen3-TTS behavior as much as possible.
- Product side sets `min_audio_length=30` by default to avoid short-sentence tail swallowing.

## What should not go upstream

### Diagnostic leftovers

Drop from upstream PRs:

- `QWEN3_TTS_DUMP_DIR`
- `QWEN3_TTS_DUMP_PREFIX`
- binary tensor dumps
- `QWEN3_TTS_GREEDY`
- one-off frame/code/logit dump logic

Reason:

- Useful for investigation, but not production framework behavior.

### Product-side policy

Keep in `jetson-voice`:

- resident worker process selection
- JSONL/IPC worker protocol
- streaming profiles
- product V2V orchestration
- product-side segmentation
- WAV concatenation
- path preference for product-built worker binaries
- fallback path selection for local model directories

Reason:

- These are product/service behavior, not EdgeLLM inference framework fixes.

## Latest validation result

Date: 2026-05-07

Validated combination:

```text
EdgeLLM clean checkout: /tmp/edgellm-pr-qwen3-clean-0507
Merged branches: pr-qwen3-tts-runtime-correctness + pr-jetson-build-compat
Build dir: /tmp/edgellm-pr-qwen3-clean-0507/build_clean_sm87_cutedsl_allprs
Talker: /home/harvest/qwen3-tts-edgellm-runtime/engines/talker
CodePredictor: /tmp/cp_product_unified_0506
Code2Wav: /home/harvest/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder/code2wav
Tokenizer: /home/harvest/qwen3-tts-trt-edge-llm-export
```

Generated samples:

```text
short_cn.wav          3.52s
short_mix.wav         7.12s
long_cn.wav           6.24s
mixed_question.wav    7.12s
product-worker-short.wav 2.16s
```

Product-side validation:

```text
uv run pytest app/tests/test_trt_edge_llm_tts.py app/tests/test_trt_edge_llm_ipc_paths.py
7 passed

cmake --build /tmp/build_edgellm_voice_worker_0507 --target qwen3_tts_worker qwen3_asr_worker -j1
qwen3_tts_worker and qwen3_asr_worker built on Orin Nano

/tmp/build_edgellm_voice_worker_0507/workers/qwen3_tts_worker
Generated /tmp/qwen3tts_product_worker_0507.wav through the official EdgeLLM backend
```

Product-side compatibility fix:

- The product worker no longer depends on investigation-only upstream streaming callback APIs.
- With clean official EdgeLLM headers, `stream=true` now falls back to full RVQ generation followed by a final Code2Wav chunk.
- The worker CMake links the CuTe DSL artifact and CUDA runtime shim from the configured EdgeLLM build when `ENABLE_CUTE_DSL` is enabled.

Remaining manual check:

- Listen to the four pulled WAVs and confirm the final subjective audio quality.
- If long-form content still needs all text in one request, rebuild/export with larger Talker KV capacity or keep product-side segmentation. The current official runtime correctly clamps max audio length to the engine KV capacity.

### Fixed global EOS workaround

Do not upstream as-is:

```text
QWEN3_TTS_DISABLE_AUTO_EOS_BIAS=1
QWEN3_TTS_MIN_EOS_FRAMES=30
```

What we learned:

- Disabling the extra progressive EOS bias and enforcing a short minimum frame count fixed audible short-sentence tail swallowing in validation.
- Fixed `30` is not a clean global framework default.
- Official Qwen3-TTS Python generation uses `min_new_tokens=2` and does not contain our progressive EOS bias.

Recommended final shape:

- Remove or disable the non-official progressive EOS bias by default if it is not justified by upstream code.
- Put minimum-duration policy in product side first.
- If EdgeLLM needs it, expose it as a request parameter instead of an env var.

### Precision and direct-engine experiments

Do not upstream as-is:

- `QWEN3_TTS_DIRECT_TALKER_ENGINE`
- `QWEN3_TTS_HOST_TEXT_PROJECTION`
- `QWEN3_TTS_ACTIVE_CP_GROUPS`
- direct explicit-KV Talker engine path
- host FP32 text projection fallback
- broad BF16/FP32 AttentionPlugin experiments unless separately proven and scoped

Reason:

- These were used to isolate the audio-quality issue.
- The final good samples did not require turning this whole experiment into upstream API.
- True FP32 boundary support may still be valuable, but it is a larger separate design discussion, not part of the immediate PR stack.

## Product-side changes to keep

Repository:

```text
/Users/harvest/project/jetson-voice
```

Keep product side:

- `app/backends/trt_edge_llm_ipc.py`
  - define `TTS_CODE_PREDICTOR_DIR`
  - pass official CP directory instead of old special CP envs
  - prefer product worker binary when present
- `app/backends/trt_edge_llm_tts.py`
  - pass `--codePredictorEngineDir`
  - remove `SpecialCodePredictorEngine` env path
  - segment long text before calling official backend
  - concatenate per-segment WAV output
  - keep product streaming behavior in product code
- `app/tests/test_trt_edge_llm_tts.py`
  - CP dir argument coverage
  - segmentation punctuation coverage
  - WAV concatenation coverage
- `app/tests/test_trt_edge_llm_ipc_paths.py`
  - product worker path preference
  - CP directory override coverage

These should not be copied to EdgeLLM upstream.

## Current dirty EdgeLLM worktree classification

Necessary or likely necessary, but must be cleaned:

- official request fields for independent CodePredictor sampling, if we keep them
- `codec_eos_logit_offset`, if converted to a documented request field
- dtype-aware `LLMEngineRunner` input validation, if retained as generic precision support
- BF16 KV cache support, only if backed by a focused precision PR

Drop before upstream PRs:

- all dump helpers and dump calls
- `QWEN3_TTS_GREEDY`
- env-based direct Talker engine path
- env-based host text projection path
- env-based active CP group override
- env-based min EOS frames
- env-based auto EOS disable
- broad AttentionPlugin precision changes unless split into a dedicated, reviewed PR

## Recommended branch/commit split

Create clean branches from `origin/main`:

```text
pr-jetson-build-compat
pr-export-builder-robustness
pr-qwen3-tts-runtime-correctness
```

Commit relationship:

- PR 1 branch contains only `fca4359` + `3f880f8`.
- PR 2 branch contains only `8bee99e` + `891e84b`, rebased onto PR 1 only if the builder API requires it.
- PR 3 branch currently contains PR 2 plus `73afadd` + `8a0df08` + `631e7f8`, because the runtime/export test changes depend on PR 2.
- Formal request/API generation controls are separate commits on `pr-qwen3-tts-runtime-correctness` and can be split into a fourth PR if review scope needs it.

Do not submit the current `official-qwen3-tts-upstream-runtime` branch directly to NVIDIA until the dirty tree is cleaned and the commits are re-split.

## Validation status

Already validated during investigation:

- Orin build of `qwen3_tts_inference` after current runtime changes.
- Short TTS samples with official backend are now audibly good.
- Longer samples were generated and pulled locally:

```text
/Users/harvest/project/jetson-voice/qwen3tts-listen-0506/official-mineos30-more-0507/
```

Still required for upstream-ready branches:

- Rebuild and test the cleaned branch, not the dirty investigation tree.
- Run WSL/x86 HF export from code.
- Build engines on Orin from exported ONNX.
- Run official backend end-to-end with the cleaned artifacts.
- Re-run product side with official backend enabled.
- Keep audio artifacts for reviewer/internal listening.
