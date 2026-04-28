# TTS Talker Vocab Pruning Spec 2026-04-28

REUSE DIRECTLY: TTS Talker uses the same original Qwen text-token ID space as ASR, so `/home/harve/qwen3-vocab-pruning/keep_ids.json` can drive TTS text-embedding pruning; do not prune codec vocab or redo corpus/tokenization.

## 1. Tokenizer Verification

Production TTS tokenization is Python-side: `app/backends/qwen3_trt.py` sets `QWEN3_TOKENIZER_DIR` to `/opt/models/qwen3-tts/tokenizer` by default, loads `vocab.json` and `merges.txt`, then calls `Tokenizer(BPE(...))` with `ByteLevel(add_prefix_space=False)` (`app/backends/qwen3_trt.py:34-42`, `app/backends/qwen3_trt.py:139-156`). ASR loads `/opt/models/qwen3-asr-v2/tokenizer.json` (`app/backends/qwen3_asr.py:55`, `app/backends/qwen3_asr.py:1007-1012`). The runtime paths differ, but both are Qwen text-token ID spaces: ASR pruning constructs a 151936-entry `orig2red` map (`app/backends/qwen3_asr.py:930-940`), and TTS C++ hardcodes the split text embedding vocab to 151936 (`benchmark/cpp/tts_ort_models.cpp:56-68`). Docs also record “ASR + TTS both share 152K Qwen3 vocab” (`docs/plans/handover-2026-04-28-nano-multilang.md:55-64`). Verdict: same vocabulary ID space for pruning, even if deployed tokenizer file format differs. `keep_ids.json` is reusable directly after sorting into `red2orig`.

## 2. Prunable Weights Inventory

Prunable target is TTS text embedding before Talker input projection. Legacy exports show `text_project.onnx` wraps `talker.model.text_embedding` then `talker.text_projection` (`benchmark/export_sherpa_unified.py:91-110`, `benchmark/export_sherpa_style.py:44-64`). Current C++ prefers a split external file: `sherpa_dir + "/text_embed_fp16.bin"` plus `text_projection_only.onnx`; if present, it loads the FP16 table into `text_embed_table_` and runs projection ONNX afterward (`benchmark/cpp/tts_ort_models.cpp:56-74`). The Mac repo path is therefore the exported ONNX directory’s `text_embed_fp16.bin`; benchmark default roots put TTS model files under `/tmp/qwen3-tts-bench/model` / `/tmp/qwen3-sherpa` (`benchmark/cpp/main.cpp:75-79`). Nano production path is `/opt/models/qwen3-tts/onnx/text_embed_fp16.bin`, because `QWEN3_SHERPA_DIR` defaults to `/opt/models/qwen3-tts/onnx` (`app/backends/qwen3_trt.py:34-41`); handover also lists deployed Qwen3-TTS ONNX under `/home/harvest/voice_test/models/qwen3-tts/onnx` (`docs/plans/handover-2026-04-28-nano-multilang.md:124-127`).

Hidden dimensions: raw text embedding is 2048-wide, then projected into 1024-wide Talker space. Evidence: benchmark comments load `text_embedding.npy` as `[151936, 2048]` and projection `fc2` as `[1024, 2048]` (`benchmark/test_all_fp32.py:24-28`); C++ derives `text_embed_dim_` from file size then feeds `[1,T,D]` to `text_projection_only.onnx` (`benchmark/cpp/tts_ort_models.cpp:198-223`). The transformer/codec hidden dimension is 1024 (`benchmark/cpp/tts_pipeline.cpp:144-149`, `benchmark/export_talker_no_if.py:133-141`). If WSL’s actual file is `[152064,2048]`, derive vocab from file size instead of keeping the current 151936 constant.

No other text-vocab 152K Talker weights were found in current runtime. `talker_prefill` / `talker_decode` take `inputs_embeds`, not `input_ids` (`benchmark/export_talker_no_if.py:19-28`, `benchmark/export_talker_no_if.py:38-47`). `codec_head` is codec logits only (`benchmark/export_talker_no_if.py:16-18`). Do NOT prune `lm_head`: CP `lm_head` is `[15,2048,1024]` codec vocab (`benchmark/export_cp_unified_v3.py:100-145`, `benchmark/export_cp_unified_v3.py:211-212`), and Talker sampling uses `cfg_.vocab_size` from codec config, not text vocab (`benchmark/cpp/tts_pipeline.cpp:140-148`, `benchmark/cpp/tts_pipeline.cpp:535-540`).

## 3. Token ID Flow

HTTP calls enter `tts_service.synthesize()` / `clone_voice()` and dispatch to the selected backend (`app/tts_service.py:47-72`). `Qwen3TRTBackend.synthesize()` tokenizes user text to original IDs, then calls `self._engine.synthesize(text, lang, token_ids)` (`app/backends/qwen3_trt.py:167-177`); voice clone and streaming do the same (`app/backends/qwen3_trt.py:200-210`, `app/backends/qwen3_trt.py:248-277`). Pybind receives those `token_ids` and calls `TTSPipeline::SynthesizeWithTokenIds` (`benchmark/cpp/tts_binding.cpp:154-183`, `benchmark/cpp/tts_binding.cpp:185-212`, `benchmark/cpp/tts_binding.cpp:282-356`). C++ then enters `GenerateInternal`, calls `BuildPrefill`, copies `token_ids_ptr` into `text_ids`, and calls `ort_->TextProject(text_ids)` (`benchmark/cpp/tts_pipeline.cpp:311-376`, `benchmark/cpp/tts_pipeline.cpp:933-941`). The actual embed lookup is C++ CPU code in `ORTModels::TextProject`: for each `input_ids[t]`, index `text_embed_table_`, convert FP16 to FP32, and run `text_projection_only.onnx` (`benchmark/cpp/tts_ort_models.cpp:198-227`). Insert `orig->red` immediately before this lookup, or earlier in Python only if C++ is also switched to the pruned table.

## 4. Architecture Decision

Minimum blast radius is C++-localized indirection in `ORTModels::TextProject`: load `text_embed_fp16_pruned.bin`, load `token_map.bin` / `keep_ids.json` as `orig2red`, and map only text embedding IDs before table indexing. Unlike ASR Phase C, Python-only is not enough for production TTS because embedding lookup is in C++ (`benchmark/cpp/tts_ort_models.cpp:198-213`), while ASR production lookup and argmax are Python-side (`docs/plans/vocab-pruning-phase-c-2026-04-28.md:1-7`). Keep tokenizer outputs original at API boundaries; only reduced IDs touch the pruned embedding table. No red-to-original mapping is needed on the output side because TTS outputs codec IDs, not text IDs.

## 5. Implementation Checklist

Step A, offline prune: read `/home/harve/qwen3-vocab-pruning/keep_ids.json`, sort ascending, load `text_embed_fp16.bin` as FP16 `[vocab,2048]`, slice rows, write `text_embed_fp16_pruned.bin`. Validate `pruned[orig2red[id]] == full[id]` bitwise. Use C++ dimensions from `benchmark/cpp/tts_ort_models.cpp:56-68` and projection evidence from `benchmark/test_all_fp32.py:24-28`.

Step B, runtime remap: in `benchmark/cpp/tts_ort_models.h:82-87`, add `red2orig`, direct `orig2red` array, and pruned flag. In `benchmark/cpp/tts_ort_models.cpp:56-74`, prefer `text_embed_fp16_pruned.bin` when token map exists; set `text_embed_vocab_ = n_keep`. In `benchmark/cpp/tts_ort_models.cpp:198-213`, map `orig_id -> red_id` before indexing; fail loudly if any token maps to `-1`.

Step C, ONNX surgery: none for the external split-embed path. If deployment falls back to old `text_project.onnx` (`benchmark/cpp/tts_ort_models.cpp:75-78`), either disable fallback in pruned mode or surgically prune the embedded `talker.model.text_embedding` initializer exported by `benchmark/export_sherpa_unified.py:96-110`.

Step D, rebuild TRT engine on Nano: No for Talker decode/prefill engines, because their interface remains `inputs_embeds [1,T,1024]` and codec logits unchanged (`benchmark/export_talker_no_if.py:143-153`). Rebuild only C++ pybind `.so` and copy the pruned `.bin`/map.

## 6. Risks

Voice-clone reference text uses the same Python tokenizer path, so OOV reference text can fail the new map (`app/backends/qwen3_trt.py:193-210`). Tokenizer mismatch risk is real because TTS uses `vocab.json`/`merges.txt` and ASR uses `tokenizer.json`; verify first 1000 `vocab.json` IDs and all special IDs exist in `keep_ids`. Codec/audio collision risk is low: codec IDs are separate 2048/3072 ranges inside Talker/CP sampling and embedding (`benchmark/cpp/tts_pipeline.h:20-23`, `benchmark/cpp/tts_pipeline.cpp:870-915`), and must not be remapped.

## 7. Smoke Test

Use fixed seed and run original vs pruned: `你好，今天天气不错。`; `请把灯打开，然后调低音量。`; `Hello, this is a Jetson voice test.`; `The price is twenty three dollars and fifty cents.`; `请用 English 回答这个问题。` Measure text-project tensor equality first, then full log-mel L2 against baseline (`docs/plans/vocab-pruning-2026-04-28.md:170-175`), plus perceptual listening. Expected mel delta is near FP16/ORT noise; any large delta means wrong row map.

## 8. Savings Estimate

For 35,641 keep IDs and raw text embedding `[151936,2048]` FP16: before 151936*2048*2 = 593.5 MiB; after 35641*2048*2 = 139.2 MiB; saved about 454.3 MiB. If actual WSL vocab is `[152064,2048]`, before is 594.0 MiB and savings is about 454.8 MiB. Other text-vocab weights: none found in current split path. Total GPU/host unified-memory pressure saved should be about 455 MiB, assuming this table is resident during TTS load; it directly targets the Nano shortfall documented at `docs/plans/handover-2026-04-28-nano-multilang.md:32-41`.

## 9. Phase Plan

Phase A, Python-only minimum, 1-2h: do offline artifact work only. Prune `text_embed_fp16.bin`, write `red2orig`/`orig2red`, and run a Python equivalence harness that emulates `ORTModels::TextProject` by mapping original tokenizer IDs to reduced rows before `text_projection_only.onnx`. This validates the map and `.bin` without touching production source, but it is not sufficient to run production TTS because the live embedding lookup is C++ (`benchmark/cpp/tts_ort_models.cpp:198-213`).

Phase B, C++ production change if needed: patch `benchmark/cpp/tts_ort_models.*` as in Step B, rebuild pybind `.so`, and keep `app/backends/qwen3_trt.py:170-177` / `app/backends/qwen3_trt.py:203-210` returning original token IDs. Avoid Python remap in production unless C++ is explicitly taught that incoming IDs are already reduced.

Phase C: copy pruned artifacts to Nano `/home/harvest/voice_test/models/qwen3-tts/onnx`, restart with LAZY_TTS, trigger `/tts`, and compare memory logs using the handover workflow (`docs/plans/handover-2026-04-28-nano-multilang.md:142-168`). No Talker TRT rebuild unless `text_project.onnx` fallback is the active deployed path.
