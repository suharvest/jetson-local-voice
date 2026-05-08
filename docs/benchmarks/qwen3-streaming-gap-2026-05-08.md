# Qwen3 Streaming Gap Notes - 2026-05-08

Target: `orin-nano` 8GB-class Jetson.

Goal: quantify the memory gap for streaming Qwen3-ASR + streaming Qwen3-TTS dual residency without sacrificing first-audio latency.

## Run 0 - Existing Container Review

Container:

```text
jetson_voice_hlm_nano8g_verify
image: jetson-voice-speech:v3.5-clean
command: python3 -m uvicorn main:app --host 0.0.0.0 --port 18084
status: Exited (3)
```

Evidence from `docker logs --tail 80`:

- `LANGUAGE_MODE=custom` log line still prints "Using Sherpa TTS + ASR", but actual backend selection is EdgeLLM:
  - ASR: `TRTEdgeLLMASRBackend`
  - TTS: `TRTEdgeLLMTTSBackend`
- ASR reached streaming-ready state:
  - ASR backend preload OK
  - capabilities include `streaming`
  - ASR streaming executor warmed up
- TTS product config reached Python preload:
  - `talker_backend=product_explicit_kv`
  - `direct_talker_engine=/home/harvest/voice_test/models/qwen3-tts/engines/talker_decode_bf16.engine`
  - `text_projection=host_fp32`
  - `code_predictor_backend=qwen3_tts_native`
  - `async_code2wav=false`
  - `segment_text=false`
  - seed `42`
- TTS worker did not emit the expected JSON `ready` event. Python raised:

```text
RuntimeError: TTS worker failed to start
```

Captured worker stderr only contained warnings:

```text
rope_scaling is not specified in the model config
TensorRT: Using an engine plan file across different models of devices is not recommended
TensorRT: Using an engine plan file across different models of devices is not recommended
```

Current interpretation:

- This is not the old `voice_nano_test` exit 137 OOM record.
- This run exits the application with code 3 during startup because TTS worker exits before `ready`.
- Need tegrastats during a rerun to decide whether the worker is silently OOM-killed, hits an engine/runtime mismatch, or exits for another reason after the TensorRT warnings.

## Operational Notes

- Do not run full `docker inspect` in this workflow unless explicitly approved; full inspect can expose environment variables. Prefer narrow status commands or filtered logs.
- `fleet exec ... docker inspect --format '{{...}}'` is fragile because Go-template braces and shell quoting can be split incorrectly. Use simpler safe commands where possible.
- `fleet exec ... sh -lc '...'` also behaved unexpectedly with complex quoting in this session. Prefer simple argv-style commands, or validate with a trivial command before relying on quoted remote shell commands.
- Foreground `tegrastats` through `fleet exec --timeout` did not stop cleanly at the requested timeout in this run. Use `tegrastats --logfile` plus explicit `pkill -f tegrastats`, or be prepared to stop it manually.

## Run 1 - Restart Existing Container With Tegrastats

Command shape:

```text
fleet exec --timeout 170 orin-nano -- tegrastats --interval 1000
fleet exec --sudo --timeout 180 orin-nano -- docker start -a jetson_voice_hlm_nano8g_verify
```

Outcome:

- Container reproduced the same failure.
- ASR EdgeLLM worker reached streaming-ready state and the ASR streaming executor warmed.
- TTS config again selected product explicit-KV BF16 Talker and host FP32 text projection.
- TTS worker did not emit JSON `ready`; FastAPI startup failed with exit code 3.
- Worker stderr again only exposed:
  - `rope_scaling is not specified`
  - two TensorRT cross-device engine-plan warnings

Timing and memory from tegrastats:

| Approx stage | Time | RAM | Swap | Note |
|---|---|---:|---:|---|
| Idle before app | 04:17:43 | 1620 / 7620 MB | 488 / 12002 MB | Baseline. |
| App start | 04:17:49 | 1683 / 7620 MB | 488 / 12002 MB | Uvicorn startup. |
| ASR loading/warmup | 04:17:50-04:17:56 | 1953 -> 3018 MB | 488 MB | ASR backend preload and streaming executor warmup. |
| TTS worker startup begins | 04:17:57 | 3251 / 7620 MB | 488 MB | TTS preload begins. |
| TTS engine/context growth | 04:18:08-04:18:18 | 5090 -> 6578 MB | 488 MB | TensorRT warnings appear in this window. |
| Near failure | 04:18:23-04:18:30 | 7148 -> 7465 / 7620 MB | 580 -> 2180 MB | Only about 155 MB physical RAM left at peak. |
| After failure | 04:18:31-04:18:32 | 3532 -> 1537 MB | 1060 -> 641 MB | Worker/container memory released. |

Current interpretation:

- The current `v3.5-clean` path fails before the TTS worker reaches `ready`, so we still do not have a true `resident floor`, first-stream peak, or warm-stream peak for this configuration.
- The relevant gap is earlier than first `/tts/stream`: TTS worker initialization itself peaks at about `7465/7620 MB`, with severe swap pressure and very fragmented large free blocks.
- Kernel `dmesg` confirms this class of failure: `oom-kill ... task=qwen3_tts_worke` and `Out of memory: Killed process ... (qwen3_tts_worke)`. The OOM task list also showed `qwen3_asr_worke` still resident.
- Low-latency implication: this is not yet a streaming TTFT problem. The system cannot enter streaming-ready dual residency with the current BF16 Talker + Code2Wav worker startup footprint.

Updated gap statement:

- At peak, only about `155 MB` physical RAM was left (`7465/7620 MB` used), and swap had climbed to about `2180 MB`.
- Because the process dies before TTS `ready`, the required improvement is not merely "avoid first-stream OOM"; it must first reduce TTS worker startup residency enough to produce a stable streaming-ready floor.
- A practical next target is at least `400-600 MB` reduction before TTS worker ready, so the system is not operating inside the last 100-200 MB of physical RAM before any streaming request.

Next candidates ranked by latency risk:

1. No-latency memory reductions:
   - verify no unpruned vocab-sized allocations remain in TTS worker startup
   - remove duplicate token maps / projection side tables
   - share or mmap immutable CPU-side assets
   - reduce worker startup allocations that are only needed after first request, as long as they can be warmed before measured traffic
2. Low-risk startup residency reductions:
   - split Code2Wav context creation from engine weight load and measure each separately
   - keep Code2Wav resident only if memory can support streaming-ready floor
   - add finer MEM tags inside worker before/after Talker runtime, Code2Wav runner, graph capture, and first context allocation
3. Conditional compression:
   - W8A16 Talker with BF16 attention only after the gap is confirmed too large for lossless cuts
   - must report memory saved and per-step TTFT cost; prior notes expect a Jetson small-batch latency penalty

## Run 2 - Lazy Code2Wav Diagnostic Attempt

Command shape:

```text
docker run ... \
  -e LANGUAGE_MODE=custom \
  -e ASR_BACKEND=trt_edgellm \
  -e TTS_BACKEND=trt_edgellm \
  -e EDGE_LLM_ASR_MANIFEST=/home/harvest/project/jetson-voice-hlm-current/configs/qwen3_asr_edgellm_nano8g.json \
  -e EDGE_LLM_TTS_MANIFEST=/home/harvest/project/jetson-voice-hlm-current/configs/qwen3_tts_edgellm_reference.json \
  -e EDGE_LLM_TTS_LAZY_CODE2WAV=1 \
  jetson-voice-speech:v3.5-clean ...
```

Outcome:

- The first attempt without explicit manifests was invalid because it fell back to different ASR/TTS runtime paths.
- The manifest-correct rerun reproduced the same failure class:
  - ASR reached ready and warmup completed.
  - TTS selected the same product explicit-KV BF16 Talker path.
  - TTS worker was OOM-killed before emitting JSON `ready`.
- Peak from tegrastats was about `7490 / 7620 MB`, with swap rising to about `2416 MB` before the process was released.
- `dmesg -T --since 2026-05-08T04:33:00` confirmed:
  - `qwen3_tts_worke invoked oom-killer`
  - `Out of memory: Killed process ... (qwen3_tts_worke)`

Important correction:

- The deployed worker binary at `/tmp/qwen3tts_ref_0507_from_nano/build/examples/omni/qwen3_tts_worker` does **not** contain the string `EDGE_LLM_TTS_LAZY_CODE2WAV`.
- It does contain `EDGE_LLM_TTS_CUDA_GRAPH`.
- Therefore this run does not measure Code2Wav laziness. The lazy variable is ignored by this deployed worker build.

Implication:

- Do not use `EDGE_LLM_TTS_LAZY_CODE2WAV=1` as evidence that Code2Wav is or is not the main memory contributor until the actual deployed worker implements the switch or finer MEM tags are added.
- The immediate finding is operational: the current deployed reference worker lacks the lazy Code2Wav diagnostic hook.

## Run 3 - Disable TTS CUDA Graph

Command delta:

```text
EDGE_LLM_TTS_CUDA_GRAPH=0
```

Outcome:

- ASR reached ready and warmup completed.
- TTS worker again failed before `ready`.
- Peak from tegrastats was about `7468 / 7620 MB`, with swap rising to about `2334 MB`.
- This is effectively the same ceiling as Run 1 (`7465 MB`) and Run 2 (`7490 MB`).

Interpretation:

- TTS CUDA graph capture is not the current pre-ready blocker.
- The OOM is happening during TTS worker engine/context construction before graph capture can materially matter, or the graph capture allocation is not the dominant part of this failure.
- Disabling graph cannot be counted as a memory-fit strategy for the current gap.

## Run 4 - Skip ASR Streaming Warmup

Command delta:

```text
SKIP_ASR_WARMUP=1
```

Outcome:

- ASR reached ready and skipped the streaming executor warmup:

```text
ASR streaming warmup skipped (SKIP_ASR_WARMUP set).
```

- TTS worker still failed before `ready`.
- The pre-TTS floor dropped to about `2594 / 7620 MB`, versus about `3251 / 7620 MB` in Run 1 when ASR streaming warmup was paid.
- The run still climbed to about `7497 / 7620 MB` and OOMed before TTS ready.

Interpretation:

- Skipping ASR warmup freed roughly `600-700 MB` before TTS startup, but even that was not enough for TTS worker to reach ready.
- The true missing memory for a streaming-ready dual-resident state is therefore larger than the earlier `400-600 MB` practical target.
- Updated working gap:
  - minimum missing memory before TTS ready is `>650 MB` relative to the fully warmed ASR baseline;
  - production needs additional headroom beyond merely avoiding OOM, so a safer target is now `900-1200 MB` reclaimed before TTS ready.
- This knob is not acceptable as the final low-latency answer by itself because ASR warmup is part of the reuse strategy. It is useful only for gap accounting unless a later design repays ASR warmup before measured traffic without reintroducing OOM.

## Current Gap Ledger Update

| Config | Pre-TTS floor | Peak before failure | Ready? | Meaning |
|---|---:|---:|---|---|
| Baseline | ~3251 / 7620 MB | ~7465 / 7620 MB | No | TTS worker OOMs before ready with ASR warmed. |
| Lazy Code2Wav env | ~3250 MB class | ~7490 / 7620 MB | No | Invalid as lazy test; deployed worker ignores the env var. |
| TTS graph off | ~3250 MB class | ~7468 / 7620 MB | No | Graph capture is not the dominant pre-ready gap. |
| Skip ASR warmup | ~2594 / 7620 MB | ~7497 / 7620 MB | No | Even freeing ~600-700 MB is insufficient. |

Revised next candidates:

1. Add MEM tags or rebuild the deployed TTS worker so startup can be split at:
   - before/after Talker runtime construction
   - before/after CP runtime construction
   - before/after Code2Wav construction
   - before/after graph capture
   - before ready
2. Verify active vocab pruning inside this deployed `/tmp/qwen3tts_ref_0507_from_nano` worker, not only in local source.
3. Search for remaining vocab-sized or duplicate product explicit-KV allocations in the TTS worker path.
4. If lossless cuts cannot reach about `900-1200 MB`, move to conditional W8A16 Talker work. Keep CP BF16, 15 active groups, host FP32 projection, and exact token-map semantics unchanged.

Operational note:

- `docker inspect --format={{.Config.Env}} ...` exposed image-level credentials in the command output. Do not use that form again in normal notes. Prefer explicit manifests from logs, or a redacted helper that only emits safe key prefixes and never prints full env.

## Run 5 - Instrumented Worker, Lazy Code2Wav, ASR Warmed

Setup:

- Rebuilt the product TTS worker from `/home/harvest/project/tensorrt-edge-llm-hlm-current` with stage memory tags and a real `EDGE_LLM_TTS_LAZY_CODE2WAV=1` hook.
- Restored `talkerMLPKernels` in `hlm-current` to the known low-latency 9-row prefill + `activeGroups` version. The current `hlm-current` tree had a mismatched full-prefill kernel signature.
- Added explicit `cublas/cublasLt` link deps for the restored Jetson cuBLAS fallback.
- Deployed the diagnostic binary to `/tmp/qwen3tts_ref_0507_from_nano/build/examples/omni/qwen3_tts_worker`.

Container:

```text
qwen3_diag_memtags_lazy_c2w_0508a
EDGE_LLM_TTS_LAZY_CODE2WAV=1
ASR warmup enabled
TTS streaming warmup enabled by app startup
```

Stage memory tags:

| Stage | MemAvailable |
|---|---:|
| `worker_entry_before_plugin` | 4079 MB |
| `worker_after_cuda_stream` | 4003 MB |
| `worker_before_tts_runtime` | 4003 MB |
| `worker_after_tts_runtime` | 159 MB |
| `worker_skip_code2wav_lazy` | 159 MB |
| `worker_before_ready` | 159 MB |
| `worker_after_ready` | 158 MB |
| `worker_before_lazy_code2wav` during startup stream warmup | 134 MB |

Outcome:

- `/health` eventually reported both Qwen3 ASR and Qwen3 TTS as ready.
- This is not a valid streaming-ready success: the first TTS streaming warmup failed immediately after `worker_before_lazy_code2wav`, and the worker exited.
- The app still marked speech service ready after logging the warmup failure, so `/health` can be a false positive for this diagnostic. For this goal, readiness must include successful first streaming chunk or explicit resident Code2Wav warmup.

Interpretation:

- Lazy Code2Wav only moves the OOM point from pre-ready to first stream.
- Ready with `158 MB` available is below the practical streaming floor.
- This violates the low-latency requirement because the deferred Code2Wav load lands on the first user-visible chunk path.

## Run 6 - TTS-Only Code2Wav Contribution

Purpose:

- Separate ASR dual-resident pressure from Code2Wav's own first-use memory cost.

Command shape:

```text
direct qwen3_tts_worker
EDGE_LLM_TTS_LAZY_CODE2WAV=1
no ASR process
stream_only=true
first_chunk_frames=25
chunk_frames=25
max_audio_length=50
```

Stage memory tags with the current reference Code2Wav engine:

| Stage | MemAvailable |
|---|---:|
| `worker_before_tts_runtime` | 5756 MB |
| `worker_after_tts_runtime` | 1914 MB |
| `worker_after_ready` | 1914 MB |
| `worker_before_lazy_code2wav` | 1906 MB |
| `worker_after_lazy_code2wav` | 617 MB |
| `worker_after_code2wav_final_chunk` | 583 MB |

Streaming result:

```json
{"first_chunk_ms":3562.175161,"generation_ms":1365.314504,"code2wav_ms":678.957846,"rtf":3.427816955769231}
```

Interpretation:

- TTS runtime construction costs about `3.84 GB` available memory in this TTS-only run.
- Code2Wav first resident load costs about `1.29 GB` available memory (`1906 -> 617 MB`) before chunk execution.
- In the dual-resident run, first-stream Code2Wav started with only `134 MB` available.
- Minimum missing memory for first streaming Code2Wav is therefore about `1.15 GB`.
- Practical target is closer to `1.5 GB` reclaimed, because the service also needs several hundred MB of headroom for stable warm streaming and allocator variance.

## Run 7 - Existing Small-Profile Code2Wav Engines

Tested existing alternative engines:

| Engine | Memory result | Runtime result | Decision |
|---|---:|---|---|
| `/home/harvest/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder/code2wav_stream100_nx_ws2048/code2wav` | `2025 -> 660 MB` on lazy Code2Wav load, about `1.36 GB` | first stream failed with illegal memory access | Reject as memory-fit strategy. |
| `/tmp/qwen3tts-old-vocoder-as-code2wav` | did not reach a valid post-load memory tag | failed with `Construction of Tensor object with zero volume is prohibited` | Reject as incompatible with current `Code2WavRunner`. |

Updated gap ledger:

| Config | Reaches worker ready? | Reaches first streaming chunk? | Key memory point | Meaning |
|---|---|---|---:|---|
| Baseline eager Code2Wav | No | No | peak ~7465 / 7620 MB | OOM before ready. |
| Lazy Code2Wav with real hook | Yes | No | `134 MB` before lazy Code2Wav | Defers failure to first stream; not low latency. |
| TTS-only lazy Code2Wav | Yes | Yes | Code2Wav load costs ~1.29 GB | Quantifies missing dual-resident memory. |
| Existing stream100 Code2Wav | Yes | No | load costs ~1.36 GB | No memory win and unstable. |

Current conclusion:

- The true streaming gap is no longer the earlier `900-1200 MB` startup estimate. With first-stream Code2Wav included, the minimum missing memory is about `1.15 GB`, and the practical recovery target is about `1.5 GB`.
- Existing lazy/eager toggles cannot meet the low-latency goal. Code2Wav must either be resident and warmed before traffic, or be replaced by a materially smaller streaming vocoder path.
- Further ASR warmup skipping remains only an accounting tool; it conflicts with the reuse/low-latency requirement and would still not cover the full Code2Wav load with enough headroom.

Lessons to carry forward:

- Do not count `/health` as success until a fixed `/tts/stream` request produces the first PCM chunk while ASR remains resident.
- Do not retry `code2wav_stream100_nx_ws2048` for memory fit unless the illegal memory access is fixed and a new memory measurement proves it saves at least several hundred MB.
- The next useful strategy should target Code2Wav/vocoder resident memory or a large TTS runtime allocation. Small ASR-only cuts are unlikely to close a `~1.5 GB` practical gap by themselves.

## Code2Wav Runner/Profile Inspection

Findings:

- The current reference `config.json` has `min_code_len=1`, `opt_code_len=300`, `max_code_len=1000`, and `max_time_steps=6000`.
- The existing small-profile engine changes code length to `min=1`, `opt=50`, `max=100`, but keeps `max_time_steps=6000`.
- `trtexec` on the reference engine automatically used input shape `codes=1x16x1`, yet created output binding `waveform=1x1x576000`.
- `Code2WavRunner::allocateBuffer()` allocates `mInputCodesDevice`, `mInputCodesHost`, and `mOutputWaveform` using the engine profile max. These explicit buffers are MB-scale, not the observed `~1.29 GB`.
- The Qwen3-TTS tokenizer decoder ONNX was exported with dummy `codes=(2,16,300)`. Its graph still contains fixed constants derived from that dummy length:
  - `/pre_conv/Constant_1 = 300`
  - several decoder constants equal `576000`, which is `300 * 1920`.
- ONNX Runtime confirms the issue is already present before TensorRT: feeding `codes` with `L=1` or `L=25` both returns output shape `(1,1,576000)`.

Interpretation:

- The observed Code2Wav memory cost is dominated by TensorRT engine/context/activation memory and the exported graph's fixed large output/time-step shape, not by the C++ runner's visible input/output tensors.
- Reducing only `max_code_len` is insufficient if the ONNX graph was still exported from the 300-frame dummy. That matches the measured `stream100_nx_ws2048` result: no memory win and worse stability.
- The next Code2Wav strategy should first try a streaming-specific export with a smaller dummy/window length, not just a smaller TensorRT profile. For the current streaming worker defaults, `first_chunk_frames=25`, `chunk_frames=25`, and `leftContextSize=25`, so the maximum normal vocoder window is `50` frames.
- A minimal valid spike is: export ONNX with dummy `code_len=50`, build with `opt_code_len=25`, `max_code_len=50`, verify `codes=1x16x1` no longer yields `waveform=1x1x576000`, then measure TTS-only lazy Code2Wav memory and first chunk latency.
- If the dummy-length export still produces fixed output, that is acceptable only if the fixed output is the streaming window size, e.g. `50 * 1920 = 96000`, and quality/short-input cropping match the reference.

## Run 8 - Code2Wav Streaming Export Spike

Artifacts:

| Artifact | Path |
|---|---|
| Failed direct 50-frame graph-surgery ONNX | `/home/harvest/qwen3-tts-trt-edge-llm-export/tokenizer_decoder_stream50_gs` |
| Runner-compatible vocoder100 ONNX | `/home/harvest/qwen3-tts-trt-edge-llm-export/tokenizer_decoder_vocoder100_compat` |
| Runner-compatible vocoder100 engine | `/home/harvest/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder_vocoder100_compat/code2wav` |
| Diagnostic TTS manifest | `/home/harvest/project/jetson-voice-hlm-current/configs/qwen3_tts_edgellm_vocoder100_compat.json` |

What worked:

- A minimal constant rewrite of the current 300-frame tokenizer decoder ONNX changed one `300` constant and seven `576000` constants, but ONNX Runtime exited during `sess.run`. Do not build from this artifact unless the shape/crop dependencies are fixed more completely.
- The older `vocoder_fp16.onnx` has input `audio_codes [1,T,16]`, output `audio_values [1,192000]`, and a `lengths` output. It was wrapped for the current `Code2WavRunner` contract by:
  - adding `codes [1,16,T] -> Transpose -> audio_codes [1,T,16]`
  - exposing `waveform [1,1,192000]`
  - dropping the unused `lengths` binding
- The resulting engine has:
  - engine size `328 MB` versus `439 MB` reference
  - input profile `codes`: min `1x16x1`, opt `1x16x25`, max `1x16x100`
  - output binding `waveform`: `1x1x192000`
  - builder activation memory `323,659,776` bytes
  - builder weights memory `339,309,712` bytes

TTS-only result with lazy Code2Wav:

| Stage | MemAvailable |
|---|---:|
| `worker_before_tts_runtime` | 5609 MB |
| `worker_after_tts_runtime` | 2028 MB |
| `worker_after_ready` | 2028 MB |
| `worker_before_lazy_code2wav` | 2016 MB |
| `worker_after_lazy_code2wav` | 1337 MB |
| `worker_after_code2wav_final_chunk` | 1233 MB |

Streaming result:

```json
{"first_chunk_ms":3321.719275,"generation_ms":1922.376254,"code2wav_ms":434.888324,"rtf":2.1873274289473685,"frames":19,"samples":36480}
```

Interpretation:

- The vocoder100 engine reduces Code2Wav lazy-load cost from about `1.29 GB` to about `679 MB`, saving roughly `600 MB`.
- Code2Wav execution for the first/final chunk dropped from about `679 ms` to about `435 ms` in this comparable TTS-only smoke.
- The output binding is still fixed-size, but fixed to 100 codec frames (`192000` samples) rather than 300 frames (`576000` samples). This is a valid intermediate streaming export, not the ideal 50-frame export.

Dual-resident result with ASR warmup enabled:

| Stage | MemAvailable |
|---|---:|
| `worker_entry_before_plugin` | 4079 MB |
| `worker_before_tts_runtime` | 4007 MB |
| `worker_after_tts_runtime` | 137 MB |
| `worker_skip_code2wav_lazy` | 137 MB |
| `worker_before_ready` | 137 MB |
| `worker_after_ready` | 138 MB |
| `worker_before_lazy_code2wav` | 143 MB |

Outcome:

- The app again reported speech service ready after the TTS streaming warmup failed, so `/health` remains a false-positive readiness signal for this failure mode.
- The TTS worker exited during lazy Code2Wav initialization. `dmesg` did not show a new kernel OOM record for this run.
- Minimum remaining gap after strategy 3 is approximately `679 - 143 = 536 MB` before first-stream Code2Wav load.
- Practical remaining target is still closer to `800 MB` once stable headroom is included.

Lessons to carry forward:

- Strategy 3 is useful and measurably improves both memory and Code2Wav chunk time, but it does not close the dual-resident gap alone.
- Do not retry "only change TensorRT profile" or "only rewrite the 300/576000 constants" as the next Code2Wav path.
- A true 50-frame re-export could save more, but the current proven artifact is vocoder100. The next highest-probability pairing is vocoder100 plus TTS text vocab pruning.

## Run 9 - Vocoder100 + TTS Text Vocab Pruning

Artifacts:

| Artifact | Path |
|---|---|
| Pruned TTS talker dir | `/tmp/qwen3tts_ref_0507_from_nano/talker_pruned_text_0508` |
| Pruned text embedding | `talker_pruned_text_0508/text_embedding.safetensors` |
| Token map | `talker_pruned_text_0508/token_map.bin` |
| Token-map-aware worker | `/home/harvest/project/tensorrt-edge-llm-hlm-current/build_hlm_current/examples/omni/qwen3_tts_worker` |
| Vocoder100 Code2Wav | `/home/harvest/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder_vocoder100_compat/code2wav` |

Implementation notes:

- Full TTS text embedding was `151936 x 2048` F16, about `594 MB`.
- Historical pruned embedding has `35669 x 2048` F16 rows, about `140 MB`.
- `token_map.bin` is `pruned_row -> original_token_id`; the runtime now builds the inverse map and remaps text token ids before embedding lookup.
- TTS special tokens are covered:
  - `151671 -> row 35433`
  - `151672 -> row 35434`
  - `151673 -> row 35435`
- Missing tokens fail loudly. Example: `你好，今天天气很好。` hit missing token id `104307`; `你好` is covered.

TTS-only result with vocoder100 and pruned text embedding:

| Stage | Full text embed | Pruned text embed | Delta |
|---|---:|---:|---:|
| `worker_after_tts_runtime` | 2028 MB | 2526 MB | +498 MB available |
| `worker_before_lazy_code2wav` | 2016 MB | 2515 MB | +499 MB available |
| `worker_after_lazy_code2wav` | 1337 MB | 1769 MB | +432 MB available |

TTS-only streaming result for `你好`:

```json
{"first_chunk_ms":2652.116416,"generation_ms":1166.729342,"code2wav_ms":448.039091,"rtf":3.31795163125,"frames":10,"samples":19200}
```

Dual-resident result with ASR warmup enabled:

| Stage | MemAvailable |
|---|---:|
| `worker_entry_before_plugin` | 4096 MB |
| `worker_after_cuda_stream` | 4030 MB |
| `worker_before_tts_runtime` | 4030 MB |
| `worker_after_tts_runtime` | 196 MB |
| `worker_skip_code2wav_lazy` | 196 MB |
| `worker_before_ready` | 196 MB |
| `worker_after_ready` | 196 MB |
| `worker_before_lazy_code2wav` during startup warmup | 164 MB |
| `worker_after_lazy_code2wav` | 112 MB |
| `worker_after_code2wav_final_chunk` during startup warmup | 27 MB |

Real `/tts/stream` request while ASR remained resident:

```text
POST /tts/stream {"text":"你好","language":"chinese"}
HTTP 200
downloaded: 49,924 bytes
time_starttransfer: 0.079 s
time_total: 1.934 s
output: /tmp/qwen3_pruned_stream_0508.pcm
```

Service-state memory during the real request:

| Stage | MemAvailable |
|---|---:|
| `worker_before_code2wav_final_chunk` | 89 MB |
| `worker_after_code2wav_final_chunk` | 121 MB |

Outcome:

- This is the first measured Qwen3-ASR + Qwen3-TTS dual-resident streaming success on `orin-nano` for the current target stack.
- It keeps ASR warmup enabled, TTS worker resident, and produces real PCM through `/tts/stream`.
- The current configuration is not yet production-safe: it runs inside roughly `27-121 MB` available-memory headroom and uses swap heavily.
- The TTS worker stderr pipe must be drained while waiting for `ready`; otherwise TensorRT and memory-tag logs can block worker startup before the JSON `ready` line.

Lessons to carry forward:

- Vocab pruning plus vocoder100 closes the measured first-stream fit gap for the short covered text case.
- This is still a "fits once" proof, not a stable production gate. The next target should be at least several hundred MB additional headroom before soak testing.
- The current pruning corpus/token map is too narrow for arbitrary Chinese text. It is acceptable for memory proof and covered prompts only; production needs a wider pruned vocab or a fallback policy that does not silently change embeddings.
- Do not overwrite the remote app's manifest-aware TTS wrapper with an older local file. For this run, explicit environment variables were used to bypass the lost manifest parser.

## Run 10 - Vocoder50 + TTS Text Vocab Pruning

Artifacts:

| Artifact | Path |
|---|---|
| Vocoder50 ONNX wrapper | `/home/harvest/qwen3-tts-trt-edge-llm-export/tokenizer_decoder_vocoder50_compat/model.onnx` |
| Vocoder50 Code2Wav engine | `/home/harvest/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder_vocoder50_compat/code2wav/code2wav.engine` |
| Pruned TTS talker dir | `/tmp/qwen3tts_ref_0507_from_nano/talker_pruned_text_0508` |

Implementation notes:

- The vocoder50 wrapper slices the verified vocoder100 wrapper output from `192000` samples to `96000` samples and exposes `waveform [batch,1,96000]`.
- The TensorRT profile is `codes: min=1x16x1, opt=1x16x25, max=1x16x50`.
- Engine binding inspection confirmed output `waveform (1,1,96000) FLOAT`.
- Engine size is `266 MB`, compared with `328 MB` for vocoder100.
- Build lesson: TensorRT 10.3 parsed `--memPoolSize=workspace:128MiB` as about 128 bytes. Use pure MiB numbers such as `--memPoolSize=workspace:512`.

TTS-only result with vocoder50 and pruned text embedding:

| Stage | MemAvailable |
|---|---:|
| `worker_before_lazy_code2wav` | 2494 MB |
| `worker_after_lazy_code2wav` | 2039 MB |
| `worker_after_code2wav_final_chunk` | 1937 MB |

TTS-only streaming result for `你好`:

```json
{"first_chunk_ms":2101.925,"generation_ms":1145.366,"code2wav_ms":484.546,"frames":10,"samples":19200,"pcm_bytes":38400,"total_ms":2103.709}
```

Dual-resident result with ASR warmup enabled:

| Stage | MemAvailable |
|---|---:|
| TTS `worker_entry_before_plugin` | 4114 MB |
| TTS `worker_before_tts_runtime` | 4042 MB |
| TTS `worker_after_tts_runtime` | 420 MB |
| TTS `worker_after_ready` | 420 MB |
| startup warmup `worker_before_lazy_code2wav` | 423 MB |
| startup warmup `worker_after_lazy_code2wav` | 100 MB |
| startup warmup `worker_after_code2wav_final_chunk` | 86 MB |
| first real request `worker_after_code2wav_final_chunk` | 105 MB |
| second real request `worker_after_code2wav_final_chunk` | 96 MB |

Real `/tts/stream` requests while ASR remained resident:

```text
POST /tts/stream {"text":"你好","language":"chinese"}
Run A: HTTP 200, downloaded 46,084 bytes, time_starttransfer 0.058 s, time_total 1.893 s
Run B: HTTP 200, downloaded 19,204 bytes, time_starttransfer 0.004 s, time_total 1.083 s
```

Outcome:

- Vocoder50 plus TTS text pruning is the best measured Nano 8GB fit so far for the covered short text case.
- It is still not production-safe. The service survives startup warmup and repeated `/tts/stream`, but steady-state headroom is only about `86-126 MB` and swap remains active.
- The dual-resident memory win from vocoder50 is much smaller than the TTS-only estimate. TTS-only suggested about `270-290 MB` more headroom than vocoder100; the dual-resident startup minimum improved only from `27 MB` to `86 MB`.
- The next pruned-vocab expansion must budget from dual-resident headroom, not TTS-only headroom.

Lessons to carry forward:

- Code2Wav output-size reduction is real and low-latency-compatible, but it does not remove the larger issue that Talker/CP runtime initialization already pushes Nano into tens of MB available before lazy Code2Wav.
- Full TTS text embedding restore is not viable on this budget: `151936 -> 35669` rows would add back roughly `454-476 MB`, far beyond the measured headroom.
- Text vocab expansion should be staged. Each added token row costs `2048 * 2 = 4096` bytes, so `+5k` tokens is about `19.5 MB`, `+10k` about `39 MB`, and `+20k` about `78 MB` before allocator overhead. Given the measured `~86-126 MB` headroom, start at `+5k` or at most `+10k`, then re-run dual-resident `/tts/stream`.
- ASR worker also needs stderr drain when waiting for `ready`; otherwise it has the same pipe-blocking risk as TTS. Preserve manifest parsing when patching this file because the Nano ASR paths come from `EDGE_LLM_ASR_MANIFEST`.

## Run 11 - Talker W8A16 + Vocoder50 + TTS Text Vocab Pruning

Artifacts tested:

| Artifact | Path | Size |
|---|---|---:|
| BF16 Talker baseline | `/home/harvest/voice_test/models/qwen3-tts/engines/talker_decode_bf16.engine` | 876 MB |
| W8A16 Talker | `/home/harvest/voice_test/models/qwen3-tts/engines/talker_decode_int8.engine` | 445 MB |
| W8A16 Talker v13 | `/home/harvest/voice_test/models/qwen3-tts/engines/talker_decode_int8_v13.engine` | 445 MB |
| W8A16 Talker v11 | `/home/harvest/voice_test/models/qwen3-tts/engines/talker_decode_int8_v11.engine` | 621 MB |

Runtime compatibility notes:

- The 445 MB W8A16 engines are single-profile engines with `inputs_embeds` max seqLen `1`.
- The current explicit-KV runtime originally used batch Talker prefill for the 9-row Qwen3-TTS prompt and failed with:
  - `Qwen3-TTS direct Talker prefill seqLen out of range: 9 (max 1)`
- A temporary runtime patch made W8A16 runnable by:
  - falling back to iterative prefill when `maxSeqLen == 1`;
  - separating single-token input max from past-KV max, because the same engine still supports past KV length around 200.

TTS-only W8A16 result after the runtime patch:

| Engine | `worker_after_tts_runtime` | `worker_after_lazy_code2wav` | Output |
|---|---:|---:|---|
| `talker_decode_int8.engine` | 3248 MB | 2840 MB | runs, but 1 frame / 3840 PCM bytes |
| `talker_decode_int8_v13.engine` | 3211 MB | 2795 MB | runs, but 1 frame / 3840 PCM bytes |
| `talker_decode_int8_v11.engine` | 2834 MB | 2431 MB | runs, but 1 frame / 3840 PCM bytes |

TTS-only comparison:

- BF16 + pruned text + vocoder50 had `worker_after_tts_runtime=2503 MB`.
- W8A16 445 MB Talker improves TTS-only available memory by roughly `700-900 MB`, depending on run noise and engine variant.
- Runtime generation latency is not comparable yet because W8A16 exits after one codec frame.

Dual-resident W8A16 result with ASR warmup enabled:

| Stage | MemAvailable |
|---|---:|
| TTS `worker_entry_before_plugin` | 4140 MB |
| TTS `worker_before_tts_runtime` | 4065 MB |
| TTS `worker_after_tts_runtime` | 977 MB |
| TTS `worker_after_ready` | 977 MB |
| startup warmup `worker_before_lazy_code2wav` | 971 MB |
| startup warmup `worker_after_lazy_code2wav` | 598 MB |
| startup warmup `worker_after_code2wav_final_chunk` | 485 MB |
| real request `worker_after_code2wav_final_chunk` | 475 MB |

Real `/tts/stream` request while ASR remained resident:

```text
POST /tts/stream {"text":"你好","language":"chinese"}
HTTP 200
downloaded: 3,844 bytes
time_starttransfer: 0.061 s
time_total: 0.828 s
output: /tmp/qwen3_w8_vocoder50_pruned_stream_0508.pcm
```

Outcome:

- W8A16 is very promising for memory: it raises real dual-resident steady headroom from about `86-126 MB` to about `475-485 MB`.
- Current W8A16 output is not usable: all tested engines produce only one 80 ms codec frame for `你好`, even with `QWEN3_TTS_MIN_EOS_FRAMES=10` and auto EOS bias disabled in TTS-only smoke.
- This must not be promoted as the low-latency solution yet. It is a memory-fit branch requiring quality/logit repair.

Lessons to carry forward:

- For single-profile Talker engines, do not treat `inputs_embeds` max seqLen as KV capacity. The runtime must track input max and past-KV max separately.
- W8A16 needs a quality gate before any dual-resident soak: compare BF16 vs W8 prefill/decode logits, first codec token sequence, frame count, and audio duration before listening tests.
- The next W8 step should be numerical diagnosis or a dual-profile W8A16 rebuild, not tuning EOS sampling knobs.
