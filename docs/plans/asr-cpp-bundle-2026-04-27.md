## Goal

Bundle all `.so`-touching ASR C++ optimizations into one rebuild and one deploy of `qwen3_speech_engine.cpython-310-aarch64-linux-gnu.so`, while keeping every change opt-in and fallback-safe.

Baseline target: V2V steady median 366 ms, mean 388 ms. Current ASR breakdown is encoder ORT 38 ms + TRT BF16 dual-profile prefill/decode 218 ms for 10 tokens (~22 ms/token) + overhead 65 ms. The deployed module is `/opt/speech/app/qwen3_speech_engine.cpython-310-aarch64-linux-gnu.so`.

Non-negotiable constraints:

- Build via `benchmark/cpp/build.sh` only. Do not run bare `cmake`.
- Default OFF for every new feature; fallback to current ORT encoder / existing TRT decoder path on missing engine, missing symbol, numerical mismatch, or runtime error.
- No same-process ORT+TRT stream sharing. Keep the current isolated ORT CUDA EP `user_compute_stream` pattern and synchronize after ASR GPU work before other TRT graph work.
- CUDA Graph capture must use `cudaStreamCaptureModeThreadLocal`, never Global.
- No nested TRT context lock while holding the Python GIL. Pybind methods that run TRT work must release the GIL before acquiring native locks or calling `enqueueV3`.
- Keep TRT engine pool size at N=2 where new decoder contexts or graph variants are introduced, to reduce TensorRT 10.3 Myelin/906 exposure.
- Do not reintroduce rejected paths: no silero-vad same-process ORT+TRT, no A1 encoder frame reuse without proven receptive-field/eviction correctness.
- No cleanup scope creep. This spec covers exactly C1-C5.

## Changes (ranked by ROI)

### C2 — GPU argmax

Current (file:line) | Python production path receives full logits from pybind and computes `np.argmax`: streaming TRT path at `app/backends/qwen3_asr.py:617-627`, offline TRT path at `app/backends/qwen3_asr.py:1003-1016`. Pybind `TRTDecoder.decode_step` allocates a CPU `std::vector<float> logits`, calls `TRTTalkerEngine::DecodeStep`, copies it into a NumPy array, and returns `[1,1,vocab]` at `benchmark/cpp/tts_binding.cpp:83-95`. Pybind `TRTDecoder.prefill` returns a NumPy logits array at `benchmark/cpp/tts_binding.cpp:99-119`. C++ `TRTTalkerEngine::DecodeStep` copies the full vocab logits D2H at `benchmark/cpp/tts_trt_engine.cpp:1010-1023`, while the C++-only `ASRPipeline` performs CPU argmax over full logits at `benchmark/cpp/asr_pipeline.cpp:342-360` and `benchmark/cpp/asr_pipeline.cpp:377-385`.

Change | Add a CUDA argmax kernel for ASR vocab logits and a device scalar `d_next_token`. Add `TRTTalkerEngine::DecodeStepArgmax(...) -> int64_t` and `PrefillLastArgmax(...) -> int64_t` or an ASR-specific wrapper that runs argmax on `d_logits_` / prefill logits before host transfer. For Python, add pybind methods such as `prefill_next_token(input_embeds) -> dict(seq_len,next_token)` and `decode_step_next_token(input_embeds, vocab_size=151936) -> int`, keeping existing `prefill` and `decode_step` unchanged for fallback. The production caller uses the new methods only when enabled; otherwise it keeps the current `result["logits"]` and `np.argmax` path. C++ must return only an int64 token ID after a 4/8-byte D2H scalar transfer, not the 151936-float logits buffer. Release the GIL in pybind before TRT execution.

Env var | `ASR_GPU_ARGMAX=1`

Engine rebuild? | No new engine required. Requires `.so` rebuild only.

Expected ms | -20 to -120 ms per ~10-token call by eliminating ~600 KB D2H per step plus CPU conversion/argmax stalls.

Risk | Medium. The argmax kernel must handle FP32, FP16, and BF16 logits exactly enough for greedy token parity. Acceptance requires token ID parity on 8 wavs before enabling by default.

### C1 — TRTEncoder C++

Current (file:line) | Production Python encoder uses ORT CUDA EP with isolated `user_compute_stream` at `app/backends/qwen3_asr.py:28-51` and loads encoder sessions at `app/backends/qwen3_asr.py:774-827`. A2 Path A is already opt-in via `ASR_ENCODER_BACKEND=ort_trt` at `app/backends/qwen3_asr.py:784-819`. Offline transcription calls `self._encoder.run(None, {"mel": mel})[0]` at `app/backends/qwen3_asr.py:974-978`. The existing C++ `ASRPipeline` still constructs an ORT CUDA encoder at `benchmark/cpp/asr_pipeline.cpp:17-25` and `benchmark/cpp/asr_pipeline.cpp:42-46`, then runs it at `benchmark/cpp/asr_pipeline.cpp:146-175`.

Change | Add a pure TensorRT `TRTEncoder` C++ class, preferably in `benchmark/cpp/tts_trt_engine.{h,cpp}` or new `asr_trt_encoder.{h,cpp}` included by `CMakeLists.txt`. It should own runtime, engine, execution context, dedicated nonblocking CUDA stream, input/output GPU buffers, and a `Run(mel, mel_len)` method returning `[1,T',1024]` features in the dtype expected by existing prompt/prefill construction. Add pybind `TRTEncoder(engine_path, max_mel_len, hidden_dim=1024).run(mel)` or wire it into `ASRPipeline`. Production Python should load this only when `ASR_TRT_ENCODER=1` and `asr_encoder_fp16.engine`/`asr_encoder_bf16.engine` exists; otherwise fallback to current ORT encoder. Reuse the current stream isolation concept: no sharing ORT CUDA EP stream with TRT encoder. If ORT remains loaded for decoder fallback, synchronize the TRT encoder stream before passing features to ORT or Python.

Env var | `ASR_TRT_ENCODER=1`

Engine rebuild? | Yes. New encoder engine file, recommended `/opt/models/qwen3-asr-v2/asr_encoder_fp16.engine` or `/opt/models/qwen3-asr-v2/asr_encoder_bf16.engine`.

Expected ms | 38 -> ~15 ms, about -23 ms. If A2 Path A lands first and encoder is already ~20-25 ms, incremental C1 gain is only ~5-10 ms; keep C1 in this bundle because it removes ORT encoder dependency from the hot path.

Risk | Medium-high. Tensor shape/profile coverage and ORT/TRT same-process stability are the main risks. Do not share streams. Keep fallback immediate on missing engine or any TensorRT enqueue error.

### C4 — Prefill prefix KV cache

Current (file:line) | Prompt constants include `ASR_TEXT` at `app/backends/qwen3_asr.py:57-64`. Streaming builds prompt embeddings every decode at `app/backends/qwen3_asr.py:588-611`; offline builds prompt embeddings at `app/backends/qwen3_asr.py:980-991`. The TRT prefill call recomputes the full prompt+audio sequence every request at `app/backends/qwen3_asr.py:1000-1005`. `TRTTalkerEngine::Prefill` resets and processes the full sequence at `benchmark/cpp/tts_trt_engine.cpp:578-585`. In C++ `ASRPipeline`, prefill embeds are rebuilt and audio is injected at `benchmark/cpp/asr_pipeline.cpp:316-333`.

Change | Add a prefix-KV cache for the fixed system prompt and `<asr_text>` anchor prefix, excluding `AUDIO_PAD`/encoder-dependent positions. On first enabled request, run prefill for the fixed prefix and retain KV on GPU in a cache object keyed by language/prompt template/model hash/engine path. For each ASR request, clone or seed the decoder KV from the prefix cache, then prefill only the audio-token span and suffix boundary. The KV reuse boundary must be exact: position IDs, cache length, and the first token after audio insertion must match the current full-prefill logits. Add pybind methods to create/clear the prefix cache and report cache hits. Python caller should enable this only after token parity validation for the full-prompt baseline on warmup.

Env var | `ASR_PREFIX_KV_CACHE=1`

Engine rebuild? | No new engine expected if existing decoder supports seeding KV and correct position IDs. `.so` rebuild required. If the engine lacks needed KV input/output bindings for partial prefill, defer rather than changing model semantics.

Expected ms | -10 to -20 ms per request by avoiding ~30 fixed prompt tokens of prefill.

Risk | Medium. A one-token boundary error changes output text silently. Acceptance must compare first-token logits/top-1 and final CER against baseline.

### C5 — KV cache BF16 IO

Current (file:line) | Python loads `asr_decoder_bf16.engine` or `asr_decoder_fp16.engine` at `app/backends/qwen3_asr.py:766-772` and constructs `TRTDecoder` at `app/backends/qwen3_asr.py:837-845`. `TRTTalkerEngine` detects logits dtype at `benchmark/cpp/tts_trt_engine.cpp:181-184`, but decode still copies/returns FP32 host logits on the common path at `benchmark/cpp/tts_trt_engine.cpp:1010-1023`. `TRTASRPrefillEngine` comments that prefill returns FP32 CPU logits and stores KV on GPU at `benchmark/cpp/tts_trt_engine.h:661-675`; its implementation copies prefill logits to FP32 CPU at `benchmark/cpp/tts_trt_engine.cpp:2663-2709` and converts KV to FP32 CPU before seeding at `benchmark/cpp/tts_trt_engine.cpp:2729-2782`.

Change | Rebuild decoder/prefill engines so KV tensors and logits/input embeddings use BF16 IO where supported. Use `trtexec` IO format flags for every relevant binding: `--inputIOFormats=bf16:chw --outputIOFormats=bf16:chw`, covering 28 `past_key_*`, 28 `past_value_*`, 28 `new_past_key_*`, 28 `new_past_value_*`, logits, and `input_embeds`/`inputs_embeds`. Update C++ binding code to be dtype-agnostic and avoid assuming host FP32 for KV seed/copy. C2 should consume BF16 logits directly on GPU, so C5 and C2 should be validated together.

Env var | `ASR_BF16_KV_IO=1`

Engine rebuild? | Yes. New decoder engine recommended as `/opt/models/qwen3-asr-v2/asr_decoder_bf16_kvio.engine`; Python should prefer it only when `ASR_BF16_KV_IO=1`, otherwise keep current `asr_decoder_bf16.engine`/`asr_decoder_fp16.engine`.

Expected ms | -5 to -15 ms from lower KV bandwidth and less conversion.

Risk | Medium. TensorRT may keep selected tensors FP32 if formats are incompatible, and app-side dtype assumptions can corrupt KV. Must inspect engine bindings at startup and log actual dtype per KV/logits binding.

### C3 — Multi-step CUDA Graph

Current (file:line) | Python enables decoder CUDA Graph at load time with `self._decoder.enable_cuda_graph(True)` at `app/backends/qwen3_asr.py:841-845`. The binding exposes `enable_cuda_graph` at `benchmark/cpp/tts_binding.cpp:125-131`. Current `TRTTalkerEngine::DecodeStep` caches a graph per `(kv_len, parity)` and launches one decode step at a time at `benchmark/cpp/tts_trt_engine.cpp:929-984`. Capture already uses `cudaStreamCaptureModeThreadLocal` at `benchmark/cpp/tts_trt_engine.cpp:960-964`.

Change | Add an experimental ASR-only path that captures N=3-5 decode iterations as one graph, including per-step KV parity transitions, token argmax, embedding lookup, and next `position_ids` update entirely on GPU. This depends on C2 because a multi-step graph cannot round-trip logits to Python between steps. Expose as a separate pybind method such as `decode_n_steps_graph(max_steps, eos_ids)` returning token IDs after one scalar/vector D2H at graph completion. Keep the current one-step graph as default and fallback.

Env var | `ASR_MULTI_STEP_GRAPH=1`

Engine rebuild? | No new engine expected. `.so` rebuild only.

Expected ms | -10 to -30 ms for 10-token calls.

Risk | High. Cross-step graph state, EOS early-stop behavior, KV parity, and CUDA capture rules are complex. The P0 906 experience shows stream-capture footguns. Treat as backlog unless the team is confident and has time for isolated stress testing.

## Build & Deploy steps

- Pre-build backup command: `cp /opt/speech/app/qwen3_speech_engine.cpython-310-aarch64-linux-gnu.so /opt/speech/app/qwen3_speech_engine.cpython-310-aarch64-linux-gnu.so.bak`
- CMake env vars/values:
  - `export ORT_ROOT=/home/recomputer/ort-from-container` (also hardcoded in `benchmark/cpp/build.sh:6-8` as `-DORT_ROOT=/home/recomputer/ort-from-container`)
  - `export CUDA_HOME=/usr/local/cuda` and `export CUDA_ROOT=/usr/local/cuda` (CMake uses `CUDA_ROOT` default `/usr/local/cuda` at `benchmark/cpp/CMakeLists.txt:7-18`; `build.sh` uses `/usr/local/cuda/bin/nvcc` at `benchmark/cpp/build.sh:6-8`)
  - TensorRT headers/libs are discovered from `/usr/include/aarch64-linux-gnu` and `/usr/lib/aarch64-linux-gnu` at `benchmark/cpp/CMakeLists.txt:24-36`.
- Build invocation: `cd /Users/harvest/project/jetson-voice/benchmark/cpp && ./build.sh`
- `.so` copy command: `cp /Users/harvest/project/jetson-voice/benchmark/cpp/build_cmake/qwen3_speech_engine.cpython-310-aarch64-linux-gnu.so /opt/speech/app/qwen3_speech_engine.cpython-310-aarch64-linux-gnu.so`
- Engine file deploy:
  - C1: `cp asr_encoder_fp16.engine /opt/models/qwen3-asr-v2/asr_encoder_fp16.engine`
  - C5: `cp asr_decoder_bf16_kvio.engine /opt/models/qwen3-asr-v2/asr_decoder_bf16_kvio.engine`
  - Do not replace the current decoder engine until acceptance passes with `ASR_BF16_KV_IO=1`.
- Restart after deploy: `cd /home/recomputer/jetson-voice/reachy_speech && docker compose restart speech`

## Acceptance criteria

- C2 GPU argmax: token IDs match baseline for prefill first token and every decode step on 8 wavs; per-call ASR latency improves by at least 20 ms when `ASR_GPU_ARGMAX=1`.
- C1 TRTEncoder C++: encoder output shape matches ORT; max absolute/relative tolerance must be documented from 8 wavs; encoder median <= 18 ms, or <= 12 ms incremental over A2 Path A if A2 is already active.
- C4 prefix KV cache: first generated token matches full-prefill baseline on 8 wavs; prefill median improves by at least 10 ms with cache warm.
- C5 BF16 KV IO: engine binding log confirms BF16 for intended KV/logits/embed IO; decode/prefill median improves by at least 5 ms with no token parity regression beyond accepted numeric tolerance.
- C3 multi-step graph: if attempted, 5x cold->warm runs must pass without CUDA capture errors, EOS mistakes, or token mismatch versus one-step graph.
- V2V steady mean and median improvement must be >= 50% of the expected lower bound for the enabled set. For must-do C2+C1+C4+C5, lower bound is 58 ms, so measured mean and median must improve by at least 29 ms.
- CER on 8 wavs must be <= baseline + 0.5 percentage points.
- Multi-utterance unit test must pass.
- 5x cold->warm V2V must show no 906/Myelin crash.

## Rollback plan

- Exact restore command: `cp /opt/speech/app/qwen3_speech_engine.cpython-310-aarch64-linux-gnu.so.bak /opt/speech/app/qwen3_speech_engine.cpython-310-aarch64-linux-gnu.so`
- Docker restart command: `cd /home/recomputer/jetson-voice/reachy_speech && docker compose restart speech`
- STOP condition: any TensorRT 906/Myelin crash, CUDA stream-capture error, nested-lock deadlock, CER > baseline + 0.5 pp, multi-utterance test failure, or V2V cold->warm instability. On STOP, disable all ASR_* feature env vars, restore `.bak`, restart, and preserve logs/engine files for offline analysis.

## Sequencing recommendation

- Must-do (this sprint): C2 GPU argmax, C1 TRTEncoder C++, C4 prefill prefix KV cache, C5 KV cache BF16 IO. Implement behind env vars in one `.so` branch, then enable one flag at a time during acceptance.
- Backlog: C3 multi-step CUDA Graph. Do only after C2 is stable and after the one-step graph path has passed repeated cold->warm stress.
- Estimated total engineering hours: 46-74 hours. C2: 8-14h; C1: 14-22h plus engine export/build time; C4: 10-16h; C5: 8-14h; C3 backlog: 16-28h if attempted.
- Estimated total V2V improvement: must-do C2+C1+C4+C5 is -58 to -178 ms before overlap, or practical -50 to -130 ms after overlap and Python overhead. If A2 Path A lands first, subtract ~13-18 ms from C1's upside. C3 backlog adds a possible -10 to -30 ms but with high stability risk.
