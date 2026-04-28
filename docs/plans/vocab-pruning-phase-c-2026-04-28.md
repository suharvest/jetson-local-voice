Wire ASR vocab-pruned decoder by keeping tokenizer IDs original at API boundaries and using reduced IDs only for pruned model tensor indexing.

## 1. Architecture Decision

Recommended approach: implement Phase C first in Python in `app/backends/qwen3_asr.py`. Production Qwen3 ASR constructs prompt IDs in Python, builds `input_embeds` in Python, calls the C++ `TRTDecoder` only for `prefill()` / `decode_step()`, then runs greedy `np.argmax` in Python (`app/backends/qwen3_asr.py:619-627`, `app/backends/qwen3_asr.py:1080-1089`). Therefore insert `orig->red` before embedding lookup and `red->orig` immediately after argmax. Keep `output_ids` as original tokenizer IDs before `Tokenizer.decode()` (`app/backends/qwen3_asr.py:1158-1161`) and before streaming partial decode (`app/backends/qwen3_asr.py:531-536`).

Alternative approach: patch the benchmark/native `ASRPipeline` C++ path too. Relevant C++ files found by `rg -l "embed_tokens|token_id|argmax|decode" benchmark/cpp --glob '*.{cc,cpp,cxx,h,hpp}'`: `benchmark/cpp/main.cpp`, `benchmark/cpp/asr_pipeline.cpp`, `benchmark/cpp/tts_pipeline.h`, `benchmark/cpp/tts_trt_engine.h`, `benchmark/cpp/tts_trt_engine.cpp`, `benchmark/cpp/tts_binding.cpp`, `benchmark/cpp/asr_pipeline.h`, `benchmark/cpp/tts_pipeline.cpp`, `benchmark/cpp/tts_ort_models.h`, `benchmark/cpp/tts_ort_models.cpp`. ASR-relevant C++ is `asr_pipeline.cpp/.h`, `tts_binding.cpp`, and shared `tts_trt_engine.cpp/.h`. C++ `ASRPipeline::Transcribe()` does greedy argmax in C++ (`benchmark/cpp/asr_pipeline.cpp:346-385`) and compares EOS there (`benchmark/cpp/asr_pipeline.cpp:363-367`), but production Python does not instantiate `ASRPipeline`; it instantiates `TRTDecoder` (`app/backends/qwen3_asr.py:903-911`).

| Approach | Pros | Cons |
|---|---|---|
| Python indirection first | Matches production flow; no C++ build; smaller blast radius; can gate by env var and compare original/pruned quickly. | Per-token Python map lookup; must touch both streaming and offline loops; ORT fallback with full decoder should stay unpruned or be separately handled. |
| C++ ASRPipeline indirection | Needed if production later moves decode loop into `ASRPipeline`; can hide maps inside native pipeline. | Current production argmax is not there; requires C++ rebuild and pybind/API changes; higher risk of changing unused benchmark code first. |

## 2. Change Inventory

- `app/backends/qwen3_asr.py:55-64`: add env/config constants. Existing model root is `QWEN3_ASR_MODEL_BASE` (`line 55`); special IDs are hard-coded as original tokenizer IDs (`lines 58-64`). Add `ASR_VOCAB_PRUNED=0|1`, `ASR_TOKEN_MAP_PATH`, `ASR_PRUNED_ENGINE_NAME`, and `ASR_PRUNED_EMBED_NAME`. Keep defaults off.

- `app/backends/qwen3_asr.py:771-779`: add fields `_asr_vocab_pruned`, `_red2orig`, `_orig2red`, `_reduced_vocab_size`, `_eos_red_ids`. Pseudocode: load no maps when disabled; when enabled, `red2orig = np.fromfile(token_map.bin, uint32)`, `orig2red = np.full(151936, -1, int32); orig2red[red2orig] = arange(N)`.

- `app/backends/qwen3_asr.py:812-818`: switch engine selection. Before: first existing `asr_decoder_bf16.engine`, then `asr_decoder_fp16.engine`. After: if `ASR_VOCAB_PRUNED=1`, prefer `asr_decoder_pruned_bf16.engine`; fail closed if missing unless an explicit fallback env is added. Construct `TRTDecoder(engine_path, ..., vocab_size, 200)` with `vocab_size = _reduced_vocab_size`, not hard-coded `151936` (`app/backends/qwen3_asr.py:907-908`).

- `app/backends/qwen3_asr.py:897-900`: switch embed path. Before: `np.fromfile(embed_tokens.bin, float16).reshape(-1, 1024)`. After: if pruned, load `embed_tokens_pruned.bin`, assert row count equals `len(red2orig)`, and keep full `embed_tokens.bin` for disabled/original mode.

- `app/backends/qwen3_asr.py:581-627` and `app/backends/qwen3_asr.py:1045-1090`: TRT decode path. Token flow is: HTTP/WebSocket audio enters Python (`app/main.py:442-457`, `app/main.py:465-573`) -> Python mel/encoder -> Python `_build_prompt()` returns original IDs (`app/backends/qwen3_asr.py:1177-1192`) -> Python embeds lookup -> C++ `TRTDecoder.prefill()` -> Python logits -> Python argmax -> Python embed lookup for next step -> C++ `decode_step()` -> Python tokenizer decode. Change `embed_tokens[tid]` to `embed_tokens[orig2red[tid]]` for prompt IDs. Change `next_token = argmax(...)` to `next_red = argmax(...)`; compare `next_red in _eos_red_ids`; append `red2orig[next_red]` to `output_ids`; use `embed_tokens[next_red]` for the next step; call `decode_step(..., _reduced_vocab_size)` instead of `151936`.

- `app/backends/qwen3_asr.py:630-675` and `app/backends/qwen3_asr.py:1092-1144`: ORT fallback. Do not run pruned mode against `decoder_unified.onnx` / `decoder_step.onnx` unless pruned ONNX fallback files are also selected. Minimum viable behavior: when `ASR_VOCAB_PRUNED=1`, require TRT decoder and skip ORT fallback with a clear error.

- `app/backends/qwen3_asr.py:1177-1192`: `_build_prompt()` uses original sentinel IDs `IM_START`, `IM_END`, `AUDIO_START`, `AUDIO_END`, `AUDIO_PAD`, `ASR_TEXT` and tokenizer language IDs. Leave returned IDs original; only map at embedding lookup. `audio_offset = prompt_ids.index(AUDIO_PAD)` stays original-space and remains correct because it finds the sentinel position, not an embedding row.

- `benchmark/cpp/asr_pipeline.cpp:71-140`, `benchmark/cpp/asr_pipeline.cpp:316-385`, `benchmark/cpp/asr_pipeline.h:76-89`: Phase B only. C++ currently loads `embed_tokens.bin`, indexes by original `token_id`, has `vocab_size_=151936`, and compares original EOS constants. If optimized later, add token-map loading, reduced `EmbedLookup(red_id)`, C++ `red->orig` after argmax, and red-space EOS constants.

## 3. Risk Points and Mitigation

Silent wrong decode scenarios: using original IDs to index `embed_tokens_pruned.bin`; passing `151936` to `decode_step()` with reduced logits; appending reduced IDs to `Tokenizer.decode()`; comparing reduced argmax IDs to original EOS `{151643,151645}`; missing a prompt special token from `token_map.bin`; allowing ORT fallback to full-vocab decoder while pruned maps are active.

Unit tests should assert: map round trip for all special constants; every `_build_prompt()` token maps to a non-negative reduced ID for `language=None`, `english`, and `chinese`; `embed_tokens_pruned[orig2red[id]]` equals full `embed_tokens[id]` for a small known ID set; a fake logits vector with max at `orig2red[151645]` terminates without appending EOS; a fake non-EOS red ID appends the corresponding original ID and decodes through the unchanged tokenizer.

## 4. Smoke Test Protocol

Use five short WAVs or synthesize fixed reference audio from these sentences: `你好，今天天气不错。`; `请把灯打开，然后调低音量。`; `Hello, this is a Jetson voice test.`; `The temperature is twenty three degrees.`; `请用 English 回答这个问题。`

Run original with `ASR_VOCAB_PRUNED=0`, then pruned with `ASR_VOCAB_PRUNED=1`, same `LANGUAGE_MODE=multilanguage`, same `ASR_ENCODER_BACKEND`. Save JSON text and metadata from `/asr`. Compute CER as Levenshtein distance over Unicode characters after normalizing whitespace and lowercasing ASCII. Compare pruned against original output first; expected delta CER should be 0 for covered vocab. Also compare both against manual transcript to catch original/pruned shared errors.

## 5. Implementation Phases

Phase A: Python-only minimum viable. Edit only `app/backends/qwen3_asr.py`; add env gate, token-map loader, engine/embed selection, red-space decode loop, and disabled ORT fallback in pruned mode. No source changes in `app/main.py` or `app/asr_backend.py`; they only route requests and backend selection (`app/asr_backend.py:90-120`, `app/main.py:102-109`, `app/main.py:449-457`, `app/main.py:485-494`).

Phase B: C++ optimization if Python indirection is too slow or if production adopts `ASRPipeline`. Patch `benchmark/cpp/asr_pipeline.*` and pybind constructor/config in `benchmark/cpp/tts_binding.cpp:430-484`; make `vocab_size()` reduced and return original `text_ids`.

## 6. Config / Feature Flag

Use `ASR_VOCAB_PRUNED=1` as the single feature flag, default `0`. Gate map loading, pruned embed path, pruned engine path, reduced vocab constructor, reduced `decode_step()` size, and ORT fallback behavior in `app/backends/qwen3_asr.py`.

Default production remains original: `QWEN3_ASR_MODEL_BASE=/opt/models/qwen3-asr-v2`, `embed_tokens.bin`, and `asr_decoder_bf16.engine` / `asr_decoder_fp16.engine`. Pruned mode should require `token_map.bin`, `embed_tokens_pruned.bin`, and `asr_decoder_pruned_bf16.engine` under the same base unless overridden by explicit path env vars. Rollback is one env change: unset `ASR_VOCAB_PRUNED` and restart.
