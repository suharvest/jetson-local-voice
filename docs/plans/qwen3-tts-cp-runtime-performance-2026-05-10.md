# Qwen3-TTS CodePredictor Runtime Performance

## Scope

Investigate the current high-performance Qwen3-TTS CodePredictor runtime on Orin Nano. The goal is lower first-audio latency and RTF without changing the quality contract. ASR/TTS vocab pruning is out of scope for the default path.

## Current Production Path

The active runtime is the highperf EdgeLLM `Qwen3TTSCodePredictorEngine` in `cpp/runtime/qwen3OmniTTSRuntime.cpp`, not the older local `benchmark/cpp` prototype. Current stateful TTS product defaults:

- `EDGE_LLM_TTS_STATEFUL_CODE2WAV=1`
- `QWEN3_TTS_CP_DECODE_CUDA_GRAPH=1`
- quality default: `QWEN3_TTS_ACTIVE_CP_GROUPS=15`, first chunk `8`
- balanced: `QWEN3_TTS_ACTIVE_CP_GROUPS=13`, first chunk `6`
- fast/V2V: `QWEN3_TTS_ACTIVE_CP_GROUPS=13`, first chunk `4`

The CP runtime uses device hidden (`device_hidden > 0`, `host_hidden=0`) and a device embedding table when decode graph is enabled. CPU sampling remains the quality-preserving default.

## Fresh Measurements

2026-05-10 Orin Nano, same product text `请关闭卧室的空调。`, stateful Code2Wav, full TTS vocab, W8A16 output-k Talker.

### CP=15 + Decode Graph

- warm first chunk: about `678-681ms`
- warm RTF: about `0.84-0.87`
- CP profile:
  - `frame_ms`: about `38.9-39.2ms`
  - `input_copy_ms`: about `0.019ms/frame`
  - `prefill_setup_ms`: about `0.081ms/frame`
  - `decode_setup_ms/group`: `0.000ms`
  - `embed_copy_ms/group`: about `0.010ms`
  - `sample_wait_ms/group`: about `2.42ms`
  - `sample_cpu_ms/group`: about `0.025ms`

### CP=13 + Decode Graph

- warm first chunk: about `638ms`
- warm RTF: about `0.78-0.79`
- CP profile:
  - `frame_ms`: about `34.1ms`
  - `decode_setup_ms/group`: `0.000ms`
  - `embed_copy_ms/group`: about `0.010ms`
  - `sample_wait_ms/group`: about `2.42ms`
  - `sample_cpu_ms/group`: about `0.027ms`

### CP=13 Without Decode Graph

- warm first chunk: about `708ms`
- warm RTF: about `0.89-0.92`
- CP profile:
  - `frame_ms`: about `41.9ms`
  - `decode_setup_ms/group`: about `0.066ms`
  - `embed_copy_ms/group`: about `0.013ms`
  - `sample_wait_ms/group`: about `1.73ms`
  - `sample_cpu_ms/group`: about `0.029ms`

Interpretation: decode CUDA graph is still a real win, about `7.8ms/frame` for CP=13 and about `70ms` on an 8-frame first chunk. `sample_wait_ms/group` is larger with graph because it now includes graph-launched GPU decode completion waiting; the CPU sampling work itself remains tiny.

### Per-Group Profile

Added a gated profiler switch in the highperf runtime:

- `QWEN3_TTS_CP_PROFILE=1`: existing aggregate profile.
- `QWEN3_TTS_CP_PROFILE_GROUPS=1`: enables aggregate profile and additionally logs average wait/cpu time per CP residual group.

CP=13 + decode graph, 50-frame aggregate:

- `g0`: wait about `1.84ms`, CPU about `0.035ms`
- `g1-g12`: wait about `2.44-2.49ms`, CPU about `0.024-0.030ms`

CP=15 + decode graph, 75-frame aggregate:

- `g0`: wait about `1.85ms`, CPU about `0.035ms`
- `g1-g10`: wait about `2.44-2.47ms`
- `g11-g14`: wait about `2.49-2.51ms`
- CPU sampling stays about `0.024-0.030ms/group`

Interpretation: there is no single abnormal residual group. The tail groups are only slightly slower, consistent with longer KV/past length. The remaining latency is a uniform sequential decode floor, so per-group special casing is unlikely to help.

### Existing Engine Variant A/B

Compared the current CP directory engine with the historical single-head engine already on the device:

- current: `/tmp/qwen3tts_ref_0507_from_nano/cp_product_unified_0506/qwen3_tts_cp.engine`
- alt single: `/home/harvest/voice_test/models/qwen3-tts/engines/cp_unified_bf16_single.engine`

Both are single-head `gen_step` engines with `logits [1,2048]`. CP=13 + decode graph results were effectively equal:

- current: first chunk about `637-639ms`, CP `frame_ms≈33.99ms`, `sample_wait_ms/group≈2.413ms`
- alt single: first chunk about `637-639ms`, CP `frame_ms≈33.88ms`, `sample_wait_ms/group≈2.403ms`

The difference is too small to justify switching engines, and the sampled frame trajectory can differ. Existing engine swaps are not a meaningful optimization path.

### Native CP Rebuild / Tactic Sweep

Built four native single-profile CP candidates from the available single-head ONNX:

- ONNX: `/tmp/qwen3-tts-cp-singlehead-edgellm/code_predictor/qwen3_tts_cp.onnx`
- variants: `ws256/512/1024 + opt_past=8`, plus `ws512 + opt_past=10`
- output: `/tmp/qwen3_tts_cp_engine_sweep_0510/*/qwen3_tts_cp.engine`
- engine size: all about `222.2MB`

These candidates showed a small apparent CP speed gain under CP=13 + decode graph:

- current production engine: `frame_ms≈34.0ms`, `sample_wait_ms/group≈2.41ms`
- rebuilt candidates: `frame_ms≈32.7-33.1ms`, `sample_wait_ms/group≈2.32-2.33ms`

But they are not equivalent to the current production engine. The available ONNX has an extra `past_length` input and the sampled trajectory changed materially; generated frame counts varied across repeats. A concrete quality gate failed:

- candidate: `single_ws512_opt10`
- prompt: `请关闭卧室的空调。`
- ASR round-trip: `有啊，说有说，其实大部分啊，现在都都没有。`

Conclusion: do not deploy these rebuilt CP engines. The measured 1-1.5ms/frame improvement is not a pure tactic win; it comes with semantic drift from a non-equivalent ONNX/runtime contract. A valid tactic sweep needs the exact no-`past_length` production ONNX or a fresh export proven code/logit-equivalent before engine performance numbers count.

### Recovered No-`past_length` CP Export

Recovered the correct export/build source from git history:

- export source: `benchmark/export_cp_unified.py` at `686d76b`, function `export_cp_single_head()`
- build source: `benchmark/build_cp_single_head.py` at `686d76b`
- bad follow-up commit: `969e4bc`, which added `past_length`

The production single-head CP contract does not need `past_length`; KV length is already represented by the `past_key/value` tensor shapes, and `cache_position` carries the absolute position. Do not use ONNX files whose inputs include `past_length` for the current runtime path.

Machine split used for recovery:

- WSL `wsl2-local`: ONNX export from the HF snapshot.
- `orin-nano`: final TensorRT build and all reported quality/performance numbers.
- `orin-nx`: can be used as a build scratch machine, but its engine plan should not be used for final nano measurements because TensorRT warns on cross-device plan reuse.

Artifacts:

- WSL ONNX: `/home/harve/qwen3-tts-cp-nopast-0510/out_single_head_686d76b/cp_single_head.onnx`
- ONNX md5: `8ee68dc005e091133b2d763bc88cc6a6`
- nano ONNX: `/tmp/qwen3-tts-cp-nopast-0510/onnx/cp_single_head_nopast_686d76b.onnx`
- nano-built engine: `/tmp/qwen3-tts-cp-nopast-0510/engines/cp_single_head_nopast_686d76b_bf16_nano.engine`
- engine md5: `564583ec4a76cf322fb89edd4ef4003c`
- runtime cp dir: `/tmp/qwen3-tts-cp-nopast-0510/cp_dir`

The CP runtime directory must include the auxiliary files from the production CP directory, not only `qwen3_tts_cp.engine`: `config.json`, `lm_heads.safetensors`, `codec_embeddings.safetensors`, `cp_embed_fp32.bin`, and `small_to_mtp_projection.safetensors`.

Nano-built engine quality/perf gate with full-vocab TTS, W8A16 output-k talker, stateful Code2Wav, `ACTIVE_CP_GROUPS=13`, `first_chunk_frames=8`, `chunk_frames=10`:

- 6/6 ASR gate passed.
- First chunk: `957-1013ms`.
- RTF: `0.845-0.959`.
- ASR outputs matched command intent for zh/en; punctuation/wording normalization only, e.g. `现在几点了？今天的天气怎么样？`.

With the same nano-built engine, reducing only `first_chunk_frames` from `8` to `4` on `请关闭卧室的空调。` produced:

- first chunk: `776.7ms`
- RTF: `0.986`
- ASR: exact `请关闭卧室的空调。`

This restores a correct no-`past_length` CP baseline and shows the sub-0.8s first-package target is reachable through chunk scheduling, but first=4 still needs the full multilingual/listening gate before becoming the default.

Follow-up continuous-streaming gate showed the important distinction between
aggressive TTFA and sustainable streaming:

- `first=4`, `chunk=10`, warm resident worker, long sentence:
  - first chunk about `555ms`
  - RTF about `0.712`
  - 11 chunks, continuous playback cadence
  - ASR gate was not stable on the sentence tail, so this remains `fast` only.
- `first=8`, `chunk=10`, warm resident worker, same long sentence:
  - first chunk about `759-760ms`
  - RTF about `0.710-0.715`
  - 9-10 chunks, continuous playback cadence
  - better quality/scheduling tradeoff and still under the 0.8s warm first-package target.

Production default was updated accordingly:

- stateful Code2Wav enabled by default
- full vocabulary by default (`QWEN3_TTS_VOCAB_PRUNED=0`)
- CP decode CUDA graph enabled by default for stateful streaming
- `QWEN3_TTS_ACTIVE_CP_GROUPS=13`
- quality/default streaming: `first_chunk_frames=8`, `chunk_frames=10`, `max_chunk_frames=10`
- explicit `EDGE_LLM_TTS_PERF_PROFILE=fast`: `first_chunk_frames=4`, still experimental for quality

## Bottleneck

The remaining CP bottleneck is not CPU top-k/top-p sampling:

- CPU sampling is only about `0.025-0.03ms/group`.
- Embedding copy is only about `0.010-0.013ms/group`.
- Hidden input is already device-resident.
- Decode graph has eliminated most host setup: `decode_setup_ms/group=0`.

The real floor is sequential per-residual-group decode. Each group depends on the previous sampled code, so the runtime must run:

`decode group j -> wait logits -> sample code j -> gather embedding -> decode group j+1`

That dependency chain makes ordinary overlap difficult. Reducing `ACTIVE_CP_GROUPS` saves almost linearly, but quality risk rises below the validated defaults.

## Optimization Options

### 1. Keep Decode Graph Default

This is already implemented and should remain default for stateful Code2Wav. It is quality-preserving because sampling stays on CPU and only decode enqueue/embedding gather are captured.

Risk: graph capture adds resident memory and warmup cost. Keep it tied to stateful/low-latency profiles, not generic TTS.

### 2. Fuse Multiple CP Groups Into One TensorRT Engine

Most promising remaining latency path if quality must stay identical.

Idea: export/build a CP engine that unrolls multiple residual groups internally, including embedding gather from the previously sampled token if sampling is greedy or otherwise device-resident. For default stochastic top-k/top-p, this is hard because CPU sampling is in the loop.

Practical variants:

- Greedy-only fused engine: easiest technically, but changes sampling quality and must stay experimental.
- Device sampling fused engine: possible, but previous GPU top-k/top-p path gave small or negative gains and needs strict codes/logits parity.
- Two-stage hybrid: keep CPU sampling, but batch multiple TRT bindings/graphs is blocked by the dependency on sampled token.

Assessment: large implementation, high quality risk unless limited to greedy/experimental profile.

### 3. Rebuild CP Engine To Emit Smaller / Faster Logits

Current CP vocab is only `2048`, so D2H logits are small. The measured wait is decode completion, not transfer or CPU sampling. Reducing logits dtype or copy size is unlikely to move the needle unless it also makes TRT compute faster.

Assessment: low priority.

### 4. CP Kernel/Engine Tactic Rebuild

Worth a controlled benchmark if we can rebuild CP with different tactic constraints, precision, or profile shapes. Previous notes say BF16 is the stable low-precision floor; FP16/INT8 had quality or NaN risk. Any rebuild must pass the existing ASR/listening gates.

Assessment: medium effort, uncertain gain. Safer than fused sampling, but may only recover a few ms/frame.

### 5. More Aggressive Active Group Policy

CP=13 is already validated on several short/medium gates and is the current balanced/fast profile. CP=12 improved speed but had quality concerns on repeated/longer text.

Assessment: product-policy knob, not a kernel/runtime optimization. Do not lower default below CP=15 without broader multilingual/listening gates.

## Recommendation

Short term:

- Keep `QWEN3_TTS_CP_DECODE_CUDA_GRAPH=1` for stateful Code2Wav profiles.
- Keep quality default CP=15, balanced/fast CP=13.
- Do not spend more time on CPU sampling, hidden copies, or simple GPU top-k kernels.

Next engineering step if more CP speed is required:

1. Rebuild/sweep CP engine tactics and precision while preserving BF16 quality floor.
2. Add CUDA event timing inside the graph path if tactic sweep is inconclusive, to split graph launch overhead from TRT decode GPU time.
3. Only after that, evaluate a fused multi-group CP path under an explicit experimental profile, because exact stochastic sampling keeps a hard dependency between groups.

## 2026-05-10 CP Engine Findings

Correct no-`past_length` ONNX:

- Source commit: `686d76b`
- Nano ONNX: `/tmp/qwen3-tts-cp-nopast-0510/onnx/cp_single_head_nopast_686d76b.onnx`
- ONNX md5: `8ee68dc005e091133b2d763bc88cc6a6`

TensorRT tactic/profile sweep on Orin Nano did not materially improve the current BF16 engine:

| Variant | Warm first chunk | RTF | CP frame |
|---|---:|---:|---:|
| baseline | `759.7ms` | `0.716` | `32.30ms` |
| ws256 opt-past8 lvl3 | `763.4ms` | `0.715` | `32.67ms` |
| ws512 opt-past8 lvl3 | `760.5ms` | `0.714` | `32.43ms` |
| ws1024 opt-past8 lvl3 | `761.6ms` | `0.715` | `32.46ms` |
| ws512 opt-past10 lvl3 | `760.2ms` | `0.717` | `32.40ms` |
| ws512 opt-past12 lvl3 | `760.0ms` | `0.714` | `32.37ms` |
| ws512 opt-past8 lvl5 | `759.9ms` | `0.716` | `32.41ms` |

Layer profiling with `trtexec` shows decode is dominated by fixed GEMM/MLP work, not KV length or sampling:

- Decode `past=2`: GPU compute about `2.74ms`, layer-profile sum `3.51ms`.
- Decode `past=8`: GPU compute about `2.75ms`, layer-profile sum `3.52ms`.
- Decode `past=13`: layer-profile sum about `3.57ms`.
- Per 5-layer decode, category sums are roughly:
  - MLP `up_proj`: `0.74ms`
  - MLP `gate_proj`: `0.52ms`
  - MLP `down_proj`: `0.39ms`
  - fused QKV: `0.62ms`
  - attention MHA kernels: only about `0.07ms`

Precision sweep:

- FP16 CP engine is faster in microbench: BF16 `2.76ms` GPU compute vs FP16 `2.51ms` at decode `past=8`.
- End-to-end stateful streaming also improves slightly: warm first chunk about `760.0ms -> 746.7ms`, RTF `0.750 -> 0.730`.
- But FP16 fails quality badly. Fixed-seed long-text gate:
  - BF16 ASR: `我们正在验证语音合成系统的流式输。`
  - FP16 ASR: repeated `嘿嘿...`
- Conclusion: FP16 CP is not acceptable for default or quality-sensitive profiles. Keep BF16 floor.

Graph surgery experiment:

- Added `scripts/optimize_qwen3_tts_cp_onnx.py` to fuse each MLP `gate_proj` + `up_proj` pair into one wider MatMul plus Slice.
- The transformation passed ONNX checker and fused all 5 MLP pairs.
- BF16 fused-Mlp engine was performance-neutral: baseline GPU `2.748ms`, fused `2.751ms` at decode `past=8`.
- Conclusion: TensorRT/Orin is not meaningfully launch-bound on those two MatMuls; do not prioritize MLP gate/up fusion.

BF16 I/O experiment:

- TensorRT layer info showed internal MatMul kernels already use BF16 xmma tactics, but the engine boundary was still FP32 for `past_key/value_*`, `new_past_key/value_*`, and `logits`.
- Rebuilding the same no-`past_length` ONNX with BF16 KV/logits I/O removes repeated boundary casts while preserving internal BF16 math.
- Microbench decode `past=8` improved:
  - baseline BF16 I/O-float: GPU `2.756ms`, layer-profile sum `3.517ms`
  - BF16 KV/logits I/O: GPU `2.566ms`, layer-profile sum `3.282ms`
- End-to-end fixed-seed streaming output was byte-identical to baseline:
  - both WAV md5 `eb289cd5c11651edb3f13e041c9fd147`
  - ASR text identical: `我们正在验证语音合成系统的流式输。`
  - warm first chunk improved `856.0ms -> 848.0ms`, RTF `0.775 -> 0.765` on the fixed quality sentence.
- Reproducible build script: `scripts/build_qwen3_tts_cp_engine.py --bf16-io`.
- Runtime default now prefers `EDGE_LLM_TTS_CP_BF16_IO_DIR` when present, defaulting to `/tmp/qwen3_tts_cp_lmhead_pretranspose_0510/cp_dir`, while explicit `EDGE_LLM_TTS_CP_DIR` still wins.

Aux stream sweep:

- BF16 I/O engines were rebuilt with `--max-aux-streams=0/1/2`.
- `trtexec` microbench at decode `past=8`:
  - aux0: GPU `2.520ms`, layer total `2.627ms`
  - aux1: GPU `2.529ms`, layer total `3.281ms` (parallel profile caveat)
  - aux2: GPU `2.564ms`, layer total `3.289ms`
- Real worker repeat with stateful streaming showed aux1 was the best end-to-end compromise:
  - aux0: first chunk `~756.8ms`, RTF `~0.7425`
  - aux1: first chunk `~755.1ms`, RTF `~0.7403`
  - aux2: first chunk `~755.7ms`, RTF `~0.7408`
- All fixed-seed aux variants produced the same WAV md5 as baseline. Use aux1 as the BF16 I/O default; the gain is small but free.

First chunk sweep after BF16 I/O aux1:

- `first_chunk_frames=5`: first chunk `603ms`, but second chunk arrives `115ms` after first audio is exhausted. Too risky for continuous playback.
- `first_chunk_frames=6`: first chunk `653ms`, second chunk gap about `35ms`. Better but still tight.
- `first_chunk_frames=7`: first chunk `702-705ms`, second chunk arrives about `46ms` before first audio is exhausted. This is the lowest validated continuous-playback default.
- `first_chunk_frames=8`: first chunk `754ms`, second chunk overlap about `125ms`.
- Default quality/stateful path was moved from first=8 to first=7. Balanced remains first=6 and fast remains first=4 as explicit lower-latency profiles.

Current next useful direction:

- Keep BF16 no-`past_length` engine, BF16 KV/logits I/O aux1, first=7/chunk=10, and stateful defaults.
- If continuing CP optimization, inspect detailed MatMul tactics or try targeted BF16 GEMM plugin only after proving TensorRT selected suboptimal kernels.
- Do not change stochastic sampling order or fuse across residual groups unless it is explicitly an experimental quality-gated branch.

Custom BF16 GEMV operator probe:

- Added `scripts/bf16_gemv_microbench.cu` as a standalone Orin Nano gate before investing in a TensorRT plugin.
- A naive `[K,N]` kernel with one thread per output column was not viable: about `0.209ms` for `K=1024,N=3072`, versus cuBLAS `0.082ms`.
- A decode-specific `[N,K]` layout with one warp per output column is much better but still does not beat cuBLAS:
  - fused QKV shape `K=1024,N=4096`: warp4 `0.115ms`, cuBLAS `0.105ms`
  - MLP gate/up shape `K=1024,N=3072`: warp4 `0.0885ms`, cuBLAS `0.0815ms`
  - MLP down shape `K=3072,N=1024`: warp4 `0.0885ms`, cuBLAS `0.0868ms`
  - lm_head shape `K=1024,N=2048`: warp4 `0.0616ms`, cuBLAS `0.0582ms`
- TensorRT layer info already shows BF16 xmma GEMM tactics such as `sm80_xmma_gemm_bf16bf16_bf16f32...` for QKV/MLP/lm_head. The custom GEMV probe confirms a hand-written non-Tensor-Core M=1 kernel is not a good CP plugin target on SM87.
- Recommendation: do not implement a TensorRT plugin around this hand-written GEMV. If we still pursue a custom operator, make it a targeted BF16 Linear plugin that calls cuBLAS/cuBLASLt with the fixed CP shapes, or a larger fused block whose microbench first proves a real gain over the existing TRT xmma tactic.

Fused MLP plugin probe:

- Built a prototype `Qwen3TtsCpMlpPlugin` in the EdgeLLM highperf tree plus an ONNX rewrite script during the investigation. The code is intentionally not kept in the active repo path because the result was a negative prototype.
- The plugin replaces each layer's `down_proj(silu(gate_proj(x)) * up_proj(x))` with one custom node and uses cuBLAS BF16 GEMMs plus a small SwiGLU kernel.
- Standalone fixed-shape MLP block looked promising: `~0.256ms/layer` after correcting stream dependencies, versus rough TRT layer-profile MLP sum above `0.33ms/layer`.
- Engine build succeeded and TensorRT imported all five plugin nodes. Artifact:
  - ONNX: `/tmp/qwen3_tts_cp_mlp_plugin_0510/onnx/cp_single_head_nopast_mlp_plugin.onnx`
  - engine dir: `/tmp/qwen3_tts_cp_mlp_plugin_0510/cp_dir`
- But engine-level performance failed badly:
  - baseline BF16 I/O aux1: GPU compute `2.53ms`, enqueue `0.50ms`
  - MLP plugin with internal aux streams: GPU compute `6.07ms`, enqueue `1.31ms`
  - MLP plugin single-stream cuBLAS: GPU compute `5.87ms`, enqueue `1.20ms`
- Conclusion: cuBLAS inside a TensorRT plugin is not a viable CP decode optimization here. It loses TensorRT's native scheduling/tactic advantages and adds plugin enqueue/dispatch overhead. Do not quality-gate, deploy, or carry this plugin in the maintained code path.
- Next custom-op work should avoid cuBLAS-in-plugin for CP decode. Only a true TensorRT-integrated tactic/plugin using Tensor Core kernels directly, or a larger graph-level change that preserves sampling semantics, is worth more effort.

LM head pre-transpose graph optimization:

- Detailed baseline layer profile exposed a runtime tail cost:
  - `Gather(stacked_weights, gen_step)`: about `0.058ms`
  - runtime `Transpose` of selected lm_head: about `0.102ms`
- Added `scripts/optimize_qwen3_tts_cp_lm_head_transpose.py`.
- The script changes `stacked_weights` from `[15,2048,1024]` to pre-transposed `[15,1024,2048]` and wires final `MatMul` directly to the `Gather` output.
- This is mathematically equivalent and does not change sampling order.
- Built artifact:
  - ONNX: `/tmp/qwen3_tts_cp_lmhead_pretranspose_0510/onnx/cp_single_head_nopast_lmhead_pretranspose.onnx`
  - engine dir: `/tmp/qwen3_tts_cp_lmhead_pretranspose_0510/cp_dir`
- Decode `past=8` microbench:
  - baseline BF16 I/O aux1: GPU compute `2.530ms`, layer total `3.286ms`
  - lm_head pretranspose: GPU compute `2.441ms`, layer total `3.112ms`
- Fixed-seed streaming gate produced byte-identical WAV vs baseline:
  - both md5 `03d3b92cfe283300e7a279d0db515e25`
  - measured warm smoke on the same text improved first chunk `608.6ms -> 602.4ms` and RTF `0.719 -> 0.706`
- Runtime default was moved to `/tmp/qwen3_tts_cp_lmhead_pretranspose_0510/cp_dir`.

Post-pretranspose profile/builder sweep:

- Rebuilt pretranspose CP with profile variants `opt6/max13`, `opt8/max13`, `opt10/max13`, `opt6/max15`, `opt8/max15`, `opt10/max15`, and `opt8/max20`.
- `past=8` trtexec results were all noise-level around GPU compute `2.439-2.581ms`; the best narrow profile was `opt8/max15` at GPU `2.439ms`, latency `2.572ms`, versus `opt8/max20` at GPU `2.440ms`, latency `2.576ms`.
- `past=2` and `past=13` did not show a stable narrow-profile win; `max20` stayed effectively tied and keeps full CP=15 headroom.
- Builder/workspace sweep on the pretranspose graph also had no material gain:
  - `ws512_lvl5`: GPU `2.442ms`, enqueue `0.493ms`, latency `2.576ms`
  - `ws1024_lvl3`: GPU `2.441ms`, enqueue `0.492ms`, latency `2.579ms`
  - `ws1024_lvl5`: GPU `2.439ms`, enqueue `0.494ms`, latency `2.574ms`
- Conclusion: do not change the default profile for this tiny delta. Keep `/tmp/qwen3_tts_cp_lmhead_pretranspose_0510/cp_dir` as the current best default; further CP work needs a new graph-level equivalent optimization or a true TensorRT-integrated kernel, not more routine tactic/profile sweeping.
