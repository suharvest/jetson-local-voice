# Qwen3 W8A16 Tensor Core Plugin Design

## Goal

Serve the current Qwen3-ASR thinker and Qwen3-TTS Talker W8A16 engines with a reusable TensorRT plugin that does not regress latency against the FP16 engines. Memory savings are useful only if the low-latency streaming path is preserved.

## Current Evidence

- Shapes are fixed and narrow:
  - `(K,N)=(1024,1024)` x56
  - `(1024,2048)` x28
  - `(2048,1024)` x28
  - `(1024,3072)` x56 or x57
  - `(3072,1024)` x28
  - ASR `lm_head=(1024,151936)` should stay FP16 unless proven otherwise.
- Current generic W8A16 plugin is correct but slow without the HMMA path:
  - FP16 ASR + FP8 embedding: about `183ms`, CUDA graph about `168ms`
  - optimized shared-input W8 full thinker: about `467ms`, CUDA graph about `452ms`
  - W8 excluding `lm_head`: about `443ms`, CUDA graph about `429ms`
- The first useful Tensor Core path is ASR `M>=16` HMMA on the original `[K,N]` layout:
  - W8 excluding `lm_head`, graph off: about `340ms` warm worker roundtrip
  - W8 excluding `lm_head`, graph on: about `326ms` warm worker roundtrip
  - three-sample ASR quality gate stayed exact
- The first material HMMA kernel optimization is to group multiple N tiles for the same M tile:
  - old single-N HMMA repeated the same A tile load/dequant work across N tiles
  - multi-N8 HMMA is now the default for `M>=16` when `EDGE_LLM_W8A16_HMMA=1`
  - full internal W8 no-lm-head product wav improved from about `330-331ms` to about `255ms`
  - `attn_only_w8` improved from about `233-234ms` to about `206ms`
  - quality gates stayed exact
- TTS W8 Talker remains quality-valid with current plugin path:
  - stateful Code2Wav + W8 Talker + FP8 text embedding warm first chunk about `0.776s`
  - product ASR gate exact on `请关闭卧室的空调。`

## Profile Results

### ASR W8 no-lm-head

`EDGE_LLM_W8A16_PROFILE=1` on `/tmp/qwen3_quality_product_set1.smoke_1.wav` shows both decode and prefill-like shapes:

- `M=1` decode:
  - `(1024,1024)`: 1008 calls, avg about `0.041ms`
  - `(1024,2048)`: 504 calls, avg about `0.065ms`
  - `(1024,3072)`: 1008 calls, avg about `0.080ms`
  - `(2048,1024)`: 504 calls, avg about `0.064ms`
  - `(3072,1024)`: 504 calls, avg about `0.097ms`
- `M=39` context/prefill:
  - `(1024,1024)`: 112 calls; one first-run outlier, normal min about `0.576ms`
  - `(1024,2048)`: 56 calls, avg about `1.16ms`
  - `(1024,3072)`: 112 calls, avg about `2.19ms`
  - `(2048,1024)`: 56 calls, avg about `1.14ms`
  - `(3072,1024)`: 56 calls, avg about `1.77ms`

This means ASR has a real HMMA candidate: `M=39` uses enough rows to amortize a 16-row Tensor Core tile.

### TTS W8 Talker

`EDGE_LLM_W8A16_PROFILE=1` on `请关闭卧室的空调。` shows only `M=1` Talker calls:

- `(1024,1024)`: min about `0.035ms`, one first-run outlier
- `(1024,2048)`: about `0.065-0.081ms`
- `(1024,3072)`: about `0.077-0.116ms`
- `(2048,1024)`: about `0.064-0.086ms`
- `(3072,1024)`: about `0.084-0.111ms`

So TTS should not be forced through a padded 16x16 Tensor Core tile. The practical path is a better `M=1` GEMV kernel, using the existing int4 AWQ GEMV structure but with W8 dequantization.

### Small-M Experiments

The first low-risk experiment was to tune the plain `[K,N]` shared-input tile:

- `EDGE_LLM_W8A16_OUTPUT_TILE=32`: best current setting.
- `64`: TTS first chunk regressed to about `0.81s`; ASR no-lm-head warm roundtrip about `452ms`.
- `128`: TTS first chunk regressed to about `0.91s`; ASR no-lm-head warm roundtrip about `469ms`.

So the default remains 32.

The second experiment added `weight_layout=1`, where qweight is transposed from `[K,N]` to `[N,K]` and `M=1` runs a contiguous per-output warp GEMV. This is not a full AWQ/Marlin interleave layout yet, but it avoids strided per-output weight reads.

TTS explicit Talker output-k engine:

- ONNX: `/tmp/qwen3_talker_decode_w8a16_outputk_0510/model.onnx`
- Engine: `/tmp/qwen3_talker_decode_w8a16_outputk_0510/talker_decode_w8a16_outputk.engine`
- Warm first chunk: about `0.759s`
- Warm RTF: about `0.97-0.99`
- Quality gate: two output wavs both ASR exact `请关闭卧室的空调。`

This is a small win over the previous layout's about `0.774s` first chunk. It is not enough to claim a large W8A16 speedup, but it meets the no-regression rule for the current TTS smoke. The next small-M step should be an AWQ-style interleaved layout to preserve shared input reuse while making weight reads more cache-friendly.

ASR no-lm-head output-k engine:

- ONNX: `/tmp/qwen3_asr_thinker_w8a16_no_lm_head_outputk_onnx_0510/model.onnx`
- Engine: `/home/harvest/qwen3-asr-edgellm-runtime/engines/thinker_full_in128_kv256_w8a16_no_lm_head_outputk_0510`
- Converted W8A16 nodes: `196`
- Quality gate: exact on the same three ASR samples
- Latency with `EDGE_LLM_W8A16_HMMA=1`:
  - graph off: about `344ms`
  - graph on: about `329ms`

This is slightly slower than original `[K,N]` layout plus HMMA (`340ms` graph off, `326ms` graph on). Do not use output-k for ASR by default. Its current value is TTS `M=1` only.

## Important Constraint

There is no native FP16-activation x INT8-weight Tensor Core instruction on Orin SM87. True W8A16 weight-only kernels normally use one of these strategies:

- Dequantize packed weights into FP16 fragments, then use FP16 HMMA.
- Use a Marlin/AWQ-style layout so dequantization and `ldmatrix` feed HMMA efficiently.
- Change the math to W8A8 by quantizing activations too, then use INT8 Tensor Cores. That is no longer W8A16 and needs a separate accuracy gate.

A naive WMMA kernel that converts `[K,N]` INT8 weights to FP16 inside every tile is not a good production path. For `M=1` decode it wastes a 16-row Tensor Core tile and repeats dequantization too often.

## Proposed Architecture

### 1. Keep The Plugin ABI Stable

Keep `W8A16LinearPlugin(activation, qweight, scales) -> output` so current ONNX rewriter and engines stay usable.

Add optional layout metadata later:

- `weight_layout=0`: current plain `[K,N]` int8, correctness and fallback path.
- `weight_layout=1`: prepacked W8 HMMA layout for Tensor Core path.
- `scale_mode=0`: per-output scale, current mode.
- `scale_mode=1`: group128 scale, optional if accuracy requires.

Default runtime should choose by shape and layout. Add env overrides only for experiments:

- `EDGE_LLM_W8A16_FORCE_PATH=plain_gemv|hmma|auto`
- `EDGE_LLM_W8A16_PROFILE=1`

### 2. Add Low-Overhead Shape Profiling First

Before changing kernels again, instrument plugin enqueue under `EDGE_LLM_W8A16_PROFILE=1`:

- count `(M,K,N)` calls
- total elapsed GPU time per shape
- max/min/p50 if feasible
- emit summary on process exit or every N calls

This tells us whether ASR/TTS latency is dominated by:

- decode `M=1`
- TTS prompt prefill `M=9`
- ASR input/prompt `M=128`
- large `N=3072` MLP projections

Without this, Tensor Core work can optimize the wrong region.

### 3. Two Kernel Families

#### A. Small-M GEMV Path, M <= 6

This is the current hot path for autoregressive decode. Tensor Core is usually not the first choice here because a 16x16 tile wastes most rows.

Implementation:

- Reuse EdgeLLM `int4WoQGemvCuda.cu` structure.
- Keep one CTA per small output block.
- Prepack weights into output-interleaved layout for coalesced vector loads.
- Use `half2` FMA after converting INT8 pairs to FP16 in registers.
- Support `Batch/M=1..6` exactly like existing int4 GEMV.
- Current implemented subset: `weight_layout=1` output-k layout for `M=1`; old layout remains fallback.

Success target:

- Must beat current shared-input W8 GEMV.
- Must get close enough to FP16 GEMV that ASR W8 no-lm-head is not worse than FP16 by more than a small margin before we consider enabling it.
- This is the first priority for TTS, because the profiled Talker path is entirely `M=1`.

#### B. HMMA Path, M >= 8 or M >= 16

This is the only path where Tensor Core can realistically pay off.

Implementation:

- Add prepacked W8 layout that matches `ldmatrix`/HMMA access.
- Convert INT8 weights to FP16 fragments per K tile with vectorized loads.
- Apply scale during dequantization once per tile.
- Use `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`, following existing int4 GEMM kernel style.
- Specialize the five known `(K,N)` shapes to avoid generic overhead.

Success target:

- For prefill-like `M>=16`, W8 HMMA should be at least as fast as TensorRT FP16 MatMul or not used.
- If it cannot beat FP16 for these model shapes, keep W8 as memory-only experimental mode.
- This is the first Tensor Core target for ASR because the profiled thinker path has heavy `M=39` work.

### 4. Optional W8A8 Path Is Separate

If W8A16 cannot meet the no-regression target, evaluate W8A8 separately:

- dynamic per-token activation quantization
- INT8 Tensor Core `mma.sync ... s8.s8.s32`
- dequant output back to FP16

This may be faster, but it changes activation precision and must pass ASR/TTS quality gates independently. It should not be marketed as W8A16.

## Implementation Order

1. Add `EDGE_LLM_W8A16_PROFILE=1` shape/timing summary. Done on remote stable and highperf plugin builds.
2. Run profile on ASR W8 no-lm-head and TTS W8 Talker product smoke. Done; results above.
3. Implement prepack metadata in ONNX rewriter without changing default layout.
4. Implement small-M W8 GEMV first, because TTS is all `M=1` and ASR decode also uses many `M=1` calls.
5. Implement HMMA path only for ASR-like `M>=16` and the known Qwen3 shapes. Initial generic WMMA/HMMA path is implemented and validated for layout 0 and layout 1, but ASR should stay layout 0 because it is faster.
6. Gate each change:
   - plugin unit numeric check against FP16 dequant reference
   - ASR three-sample full-vocab gate
   - TTS product ASR gate
   - latency comparison against FP16 baseline and current W8 baseline

## Current Decision

Use layout 0 + HMMA + CUDA graph as the best current full ASR W8 no-lm-head path, but do not make ASR W8 the low-latency default. Do not switch ASR to output-k. For TTS, output-k is a small no-regression win for `M=1`, but the next meaningful small-M change still needs a real AWQ/Marlin-style interleaved GEMV layout.

2026-05-10 ASR selective rollback results, same product wav, `EDGE_LLM_ASR_CUDA_GRAPH=1`, `EDGE_LLM_W8A16_HMMA=1`:

- FP16 thinker + FP8 embedding: engine about `1.2GB`, warm worker roundtrip about `167-168ms`.
- Full internal W8 no-lm-head: engine about `729MB`, warm about `331ms`; quality exact.
- `mlp_only_w8`: keep 84 MLP W8 nodes, rollback Attention to FP16. Engine about `896MB`, warm about `266-267ms`; three ASR text gates exact.
- `attn_only_w8`: keep 112 Attention W8 nodes, rollback MLP to FP16. Engine about `981MB`, warm about `233-234ms`; three ASR text gates exact.

Initial single-N HMMA interpretation: the MLP W8 path is the larger performance problem. `attn_only_w8` was the only useful selective-memory candidate, but it still cost about `+66ms` vs FP16+FP8 while saving only about `200-250MB` of engine size. The multi-N8 update below narrows that gap, but does not change the default low-latency decision.

CUDA graph capture instrumentation showed why graph only gives modest ASR W8 wins: under graph=1, `M=1` decode calls are partially captured, but `M=39` calls are not captured. Example periodic summary from `attn_only_w8`:

- `M=1 K=1024 N=1024`: 56 captured / 56 not captured.
- `M=1 K=1024 N=2048`: 28 captured / 28 not captured.
- `M=1 K=2048 N=1024`: 28 captured / 28 not captured.
- `M=39 K=1024 N=1024`: 0 captured / 388 not captured.
- `M=39 K=1024 N=2048`: 0 captured / 194 not captured.
- `M=39 K=2048 N=1024`: 0 captured / 194 not captured.

Next ASR direction at this point was fixed-shape prefill/M=39 capture or a runner-level graph for the ASR prompt step, to separate host scheduling overhead from kernel compute.

Follow-up implementation: added an experimental stable-runtime switch `EDGE_LLM_ASR_PREFILL_CUDA_GRAPH=1` in `LLMEngineRunner::executePrefillStep`. It lazily captures the prefill TRT enqueue per input shape/address after the first regular enqueue. This confirmed `M=39` can enter graph capture, but the latency win is small:

- `attn_only_w8`, product wav, no prefill graph: warm about `233.6-234.0ms`.
- `attn_only_w8`, product wav, prefill graph: warm about `231.0-231.7ms`.
- `full_w8`, product wav, prefill graph: warm about `328.8-329.3ms`, only marginally better than the previous about `330-331ms`.
- Three ASR text gates remained exact with prefill graph enabled.

Conclusion: prefill graph removes some host scheduling overhead but does not change the ASR W8 decision. The dominant W8 cost is still GPU compute in the W8 kernels, especially MLP. Keep `EDGE_LLM_ASR_PREFILL_CUDA_GRAPH` experimental; it is not worth making default unless combined with a better W8 kernel or another graph-level runtime cleanup.

Follow-up kernel implementation: replaced the old one-warp-per-`16x16` output tile HMMA dispatch with a multi-N HMMA kernel. A block now handles the same M tile across multiple N tiles, so the A tile is loaded once into shared memory and reused by 4 or 8 warps. The default policy is:

- `EDGE_LLM_W8A16_HMMA=1`, `M>=16`: use multi-N8 HMMA.
- `EDGE_LLM_W8A16_HMMA_MULTI_N4=1`: force the 4-warp variant.
- `EDGE_LLM_W8A16_HMMA_SINGLE_N=1`: force the previous single-N HMMA for rollback.

2026-05-10 Orin Nano ASR worker results, same product wav, stable plugin/runtime rebuilt:

- FP16 thinker + FP8 embedding baseline: warm about `166.9-167.3ms`.
- Full internal W8 no-lm-head, previous single-N HMMA: about `330-331ms`.
- Full internal W8 no-lm-head, multi-N4: about `260.2-260.5ms`.
- Full internal W8 no-lm-head, multi-N8/default: about `255.3-255.7ms`.
- Full internal W8 no-lm-head, multi-N16 experiment: about `257.0-258.1ms`.
- `attn_only_w8`, previous HMMA: about `233-234ms`.
- `attn_only_w8`, multi-N8/default: about `205.9-206.4ms`.
- `attn_only_w8`, multi-N16 experiment: about `207.4-208.1ms`.

Quality gate with multi-N8/default stayed exact:

- Full W8: `请打开客厅的灯。` about `236.8ms`, `今天我们继续验证低延迟流式生成的效果。` about `331.2ms`, `请关闭卧室的空调。` about `255.7ms`.
- `attn_only_w8`: same three texts exact, about `204.2ms`, `279.9ms`, `206.3ms`.

Updated decision: the previous kernel was not optimal, and multi-N8 should stay as the default HMMA path. Multi-N16 reuses more of the A tile but loses the small gain to 512-thread block occupancy/synchronization overhead, so it is kept only as `EDGE_LLM_W8A16_HMMA_MULTI_N16=1` for experiments. ASR W8 still does not beat FP16+FP8. `attn_only_w8` is now a more credible memory-pressure mode because the latency gap is about `+39ms` instead of `+66ms`, while saving roughly `200-250MB` engine size. Full W8 is still a memory-only mode: it saves about `500MB` engine size but costs about `+88ms` on the product wav.

2026-05-11 ASR MLP-only INT8/W8A8 feasibility check on Orin NX:

- Added experimental builder `scripts/build_qwen3_asr_thinker_mlp_int8.py`.
- Targeted only the 84 ASR MLP MatMul layers (`/mlp/{gate,up,down}_proj/MatMul`) for TensorRT implicit INT8/PTQ; kept 112 attention MatMul layers and the lm_head conservative.
- The ASR thinker ONNX uses external data, so the builder must use TensorRT parser-from-file rather than parsing ONNX bytes.
- Result: TensorRT build fails before engine generation with `AttentionPlugin: could not find any supported formats consistent with input/output data types`.
- The failure reproduces even when non-MLP MatMuls are forced FP16 and AttentionPlugin precision is pinned FP16. In implicit INT8/PTQ mode, TensorRT still propagates INT8/calibration format choices through the EdgeLLM AttentionPlugin boundary.

Decision from this check: do not pursue `BuilderFlag.INT8 + IInt8EntropyCalibrator` for ASR thinker MLP-only quantization. It is not a low-effort reuse path with the current EdgeLLM AttentionPlugin graph. If W8A8 is still needed, use one of these scoped routes instead:

- Explicit Q/DQ around only the MLP MatMuls, with attention plugin inputs proven FP16 by TensorRT layer info before any quality run.
- A dedicated MLP-only W8A8 TensorRT plugin that consumes FP16 activation, performs controlled activation quantization internally, and emits FP16 output. This keeps the AttentionPlugin ABI untouched but is no longer a simple TensorRT PTQ experiment.

Do not spend more time tuning calibrator data, workspace size, or synthetic-vs-real batches for the implicit INT8 path; those do not address the plugin format failure.

## TTS CP Policy

Stateful Code2Wav moves the main latency bottleneck back to Talker/CP generation. The current product policy is:

- Quality default: `QWEN3_TTS_ACTIVE_CP_GROUPS=15` plus `QWEN3_TTS_CP_DECODE_CUDA_GRAPH=1`
- Balanced profile: `QWEN3_TTS_ACTIVE_CP_GROUPS=13`, first chunk `6` frames, plus `QWEN3_TTS_CP_DECODE_CUDA_GRAPH=1`
- Fast/V2V profile: `QWEN3_TTS_ACTIVE_CP_GROUPS=13`, first chunk `4` frames, plus `QWEN3_TTS_CP_DECODE_CUDA_GRAPH=1`
- Explicit `QWEN3_TTS_ACTIVE_CP_GROUPS` always wins over the profile default.

2026-05-10 sweep with stateful Code2Wav and TTS output-k Talker:

- CP=13, product short sentence: ASR exact x2, first chunk about `638-640ms`, RTF about `0.77-0.81`
- CP=13, latency sentence: ASR exact x2, first chunk about `675ms`, RTF about `0.733-0.735`
- CP=15, product short sentence: ASR exact x3, first chunk about `679-682ms`, RTF about `0.84-0.87`
- CP=15, latency sentence: ASR exact x2, first chunk about `719ms`, RTF about `0.805-0.815`

CP=13 is promising for V2V latency, but CP=15 remains the quality-first default until a broader multilingual and longer-utterance quality gate passes.

Additional first-chunk sweep with CP=13:

- First chunk `4` frames, product sentence: ASR exact x2, first chunk about `428-430ms`, RTF about `0.788-0.789`
- First chunk `4` frames, latency sentence: ASR exact x2, first chunk about `465-466ms`, RTF about `0.736-0.742`
- First chunk `6` frames, product sentence: ASR exact x2, first chunk about `533-534ms`, RTF about `0.767-0.779`
- First chunk `6` frames, latency sentence: ASR exact x2, first chunk about `570-572ms`, RTF about `0.737`

This is why the product policy uses `first=4` only for the fast/V2V profile and `first=6` for balanced.
