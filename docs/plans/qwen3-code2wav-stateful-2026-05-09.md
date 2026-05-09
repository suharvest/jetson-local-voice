# Qwen3 Code2Wav Stateful Streaming Plan

Date: 2026-05-09

## Current State

The current Qwen3 TTS streaming path is not stateful. It uses bounded stateless overlap:

- collect RVQ frames from Talker/CodePredictor
- run Code2Wav on a window of `context + new frames`
- discard the waveform samples that correspond to the context
- emit only new PCM

The current low-latency dual-resident default is:

- `first_chunk_frames=50`
- `chunk_frames=97`
- `max_chunk_frames=97`
- `EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES=3`
- `vocoder100_compat`

This keeps each Code2Wav invocation at `<=100` input frames, which is the current fast TensorRT profile boundary.

## Verified Baseline

On Orin Nano 8GB, ASR+TTS dual resident:

- W8A16 Talker
- `QWEN3_TTS_ACTIVE_CP_GROUPS=12`
- `vocoder100_compat`
- Code2Wav preloaded
- async Code2Wav off
- TTS CUDA graph off

Measured result:

- `RTF=0.951`
- `total_ms=7378ms`
- `audio_s=7.76s`
- `code2wav_ms=614ms`
- lowest observed `MemAvailable` around `1.20GB`

## Actual Code2Wav Interface

Current C++ runner:

- file: `cpp/multimodal/code2WavRunner.cpp`
- public API: `generateWaveform(codes, outputAudio, stream)`
- engine input: `codes [batch, 16, code_len]`
- engine output: `waveform [batch, 1, waveform_len]`
- no state input
- no state output
- no per-session cache

Current TensorRT engine allocates max output from profile:

- max code len: `100`
- total upsample: `3840`
- max waveform per invocation: `100 * 3840 = 384000` FP32 samples

## Model Structure Relevant To Stateful Streaming

Current `qwen3_tts_code2wav` config:

- `num_quantizers=16`
- `codebook_size=2048`
- `hidden_size=512`
- `decoder_dim=1536`
- `num_hidden_layers=8`
- `num_attention_heads=16`
- `sliding_window=72`
- `upsampling_ratios=[2, 2]`
- `upsample_rates=[8, 5, 4, 3]`
- total upsample: `2 * 2 * 8 * 5 * 4 * 3 = 3840`

The exported ONNX has only `codes -> waveform`. It contains:

- sliding-window causal attention in the pre-transformer
- causal Conv1d pads
- ConvTranspose1d upsampling with trim
- residual decoder stages with dilated causal convolutions

Therefore stateful Code2Wav is not just a small runner change. The model graph itself must expose state.

## Required State

### 1. Pre-transformer attention state

For each transformer layer, cache K/V for the previous `sliding_window - 1 = 71` code frames.

Approximate FP16 memory:

```text
8 layers * 2(K,V) * 71 tokens * 512 hidden * 2 bytes ~= 1.16 MB
```

Inputs needed:

- `attn_k_state_in[layer]`
- `attn_v_state_in[layer]`
- `position_offset`

Outputs needed:

- `attn_k_state_out[layer]`
- `attn_v_state_out[layer]`

The chunk path must use absolute RoPE positions, not positions starting at zero every chunk.

### 2. Causal Conv1d state

Every causal Conv1d currently performs left padding inside the graph. Stateful mode must replace that implicit zero/history padding with an explicit history buffer.

For a Conv1d with effective kernel `K_eff = (kernel - 1) * dilation + 1`, keep `K_eff - stride` previous input samples on that layer's time grid.

State needed:

- one input-history tensor per causal Conv1d
- one output-history tensor if a following residual block needs aligned previous activations

### 3. ConvTranspose1d phase/state

ConvTranspose is the risky part. It is causal but trims `kernel_size - stride` from both left and right edges. In streaming, this means:

- some output samples near the chunk boundary depend on the chunk alignment
- each transposed-conv stage needs phase-aware state
- the implementation may need a small output delay or overlap-add equivalent

This is the main correctness risk for a true stateful vocoder.

## Phase A Progress

Reference script:

- `scripts/qwen3_code2wav_stateful_reference.py`
- `scripts/qwen3_tts_code2wav_stateful_real_gate.py`

Validated on CPU with random weights:

- `CausalConv1d` streaming history matches full forward.
- `CausalTransposeConv1d` overlap-add matches full forward when bias is applied once globally.
- `CausalTransposeConv1d` online emit can output stable samples per chunk and match full forward.
- Sliding-window attention KV state matches full forward when chunks use absolute `position_offset`.
- Code embedding + 3-layer pre-transformer stack + final RMSNorm matches full forward.
- A small complete `Code2WavModel` can be run through the same chunk-state rules and match full forward in collect mode.
- The same small complete `Code2WavModel` also matches full forward in online mode, where each layer receives only stable chunks emitted by the previous layer.

Latest probe results:

```text
float32 worst max_abs: 2.384185791015625e-07
float16 worst max_abs: 0.0
```

This locks down the reference state rules for the full model structure. The
remaining Phase A work is to run the online reference against real Code2Wav
weights/codes, then turn the Python reference state layout into an exportable
stateful graph interface.

Real product decoder gate:

- Source decoder: `/home/harvest/qwen3-tts-trt-edge-llm-export/tokenizer_decoder`
- Transfer path: Orin -> `wsl2-local` direct fleet transfer, because Mac -> Orin was slow.
- The product ONNX uses external data: `model.onnx` + `onnx_model.data`.
- The actual decoder is Qwen3-TTS tokenizer-12Hz `Qwen3TTSTokenizerV2Decoder`, not the older Qwen3-Omni standalone `Code2WavModel`.
- The ONNX initializer mapping needs two special cases:
  - `quantizer.rvq_first.input_proj.weight` and `quantizer.rvq_rest.input_proj.weight` are absent because decode does not use input projection.
  - `onnx::Exp_*` SnakeBeta initializers are raw alpha/beta parameters in this export; do not apply `log()`.

Validated with real RVQ codes from `qwen3tts-listen-0506/clean-allprs-0507/short_cn/rvq_req0.safetensors`:

```text
8 frames, chunk=3:  max_abs 9.128125384449959e-07
16 frames, chunk=5: max_abs 5.010515451431274e-06
44 frames, chunk=8: max_abs 3.077089786529541e-06
16 frames, chunk=1: max_abs 8.761882781982422e-06
```

All passed the float32 tolerance `2e-4`, with exact waveform length matches.

## Feasible Implementation Path

### Phase A: PyTorch reference, no TensorRT

Goal: prove stateful chunked output can numerically match full-window output.

Add a stateful wrapper around the Python Code2Wav model:

- `forward_chunk(codes_chunk, state_in, position_offset) -> waveform_chunk, state_out`
- process chunks such as `1`, `5`, `25`, `50`, `97`
- compare concatenated chunk output against full `forward(codes)` output
- measure max error and boundary error

Acceptance:

- no obvious boundary discontinuity
- full vs chunk waveform error small enough for ASR/subjective gates
- output length exactly matches `sum(chunk_len) * 3840`

### Phase B: Stateful ONNX export

Only after Phase A passes:

- export stateful ONNX with explicit `state_in/state_out`
- use fixed state shapes
- keep chunk length dynamic or provide profiles for common chunk sizes
- include `position_offset` as an explicit scalar input

Expected engine inputs:

- `codes_chunk`
- `position_offset`
- `attn_k_state_in_*`
- `attn_v_state_in_*`
- `conv_state_in_*`
- `transconv_state_in_*`

Expected engine outputs:

- `waveform_chunk`
- `attn_k_state_out_*`
- `attn_v_state_out_*`
- `conv_state_out_*`
- `transconv_state_out_*`

### Phase C: TensorRT runner

Add a new runner instead of changing the existing stateless runner in place:

- `StatefulCode2WavRunner`
- owns one state set per stream/session
- `reset()` at utterance start
- `generateChunk(codes_chunk, stream)` emits only new waveform
- no overlap/discard in the worker

Keep `Code2WavRunner` as fallback.

Phase C production hook started:

- Added C++ runner files in TensorRT-Edge-LLM:
  - `cpp/multimodal/statefulCode2WavRunner.h`
  - `cpp/multimodal/statefulCode2WavRunner.cpp`
- Worker integration:
  - `examples/omni/qwen3_tts_worker.cpp`
  - `EDGE_LLM_TTS_STATEFUL_CODE2WAV=1` enables the stateful path.
  - `EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR=<dir>` overrides the engine dir.
  - Expected engine file: `<dir>/code2wav_stateful.engine`.
  - Default stateless path is unchanged.
- Runtime contract:
  - input `codes`
  - output `waveform`
  - optional scalar input `position_offset`
  - optional scalar input `is_final`
  - state tensors are paired as `<name>_in` and `<name>_out`
- Worker behavior:
  - on each emitted streaming chunk, stateful mode feeds only `[lastEmittedFrames, totalFrames)` new codes.
  - no left context window and no context sample discard.
  - `reset()` is called per request before Talker generation.
  - `async_code2wav` is rejected in stateful mode for now.

Build and smoke status on Orin:

```text
cmake configure + qwen3_tts_worker build: passed
default stateless streaming smoke: passed
stateful flag without code2wav_stateful.engine: fails clearly at worker init
```

Build note: because `cpp/CMakeLists.txt` uses `GLOB_RECURSE`, adding the new
runner `.cpp` requires rerunning `cmake ..` before `cmake --build`, otherwise
the worker links with undefined `StatefulCode2WavRunner` symbols.

Stateful ONNX/export status:

- Export script: `scripts/qwen3_tts_code2wav_stateful_export.py`
- Product decoder: `Qwen3TTSTokenizerV2Decoder`
- Explicit state gate with real ONNX weights and real RVQ codes:
  - 16 frames, chunk=4: `max_abs=5.36e-6`
  - 16 frames, chunk=1: `max_abs=8.99e-6`
  - 44 frames, chunk=8: `max_abs=2.94e-6`
  - tolerance: `2e-4`
- Final state interface:
  - inputs: `codes`, `position_offset`, plus 37 state tensors
  - outputs: `waveform`, plus 37 state tensors
  - zero-length conv states and zero-length transposed-conv pending tails are not exported.
- WSL artifact:
  - `/tmp/qwen3_code2wav_stateful_gate/qwen3_tts_code2wav_stateful.onnx`
- Orin artifact:
  - `/tmp/qwen3_code2wav_stateful_engine/code2wav_stateful.onnx`
  - `/tmp/qwen3_code2wav_stateful_engine/code2wav_stateful.engine`
  - `/tmp/qwen3_code2wav_stateful_engine/config.json`

TensorRT build status on Orin:

```text
trtexec fp16 profile: min=1, opt=4, max=16 frames
engine size: 222.843 MiB
activation memory: 129.253 MiB
opt=4 GPU compute: ~15.99 ms
```

Production worker stateful smoke:

```text
EDGE_LLM_TTS_STATEFUL_CODE2WAV=1
EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR=/tmp/qwen3_code2wav_stateful_engine
stateful_code2wav: true
warm stable first chunk: ~0.766-0.768 s
warm stable RTF: ~0.976-0.984
warm stable Code2Wav total: ~81-87 ms
```

Compared with the previous stateless small-chunk path, this removes the large
per-chunk vocoder recompute cost. The remaining latency is now mostly Talker/CP
generation and per-chunk orchestration, not Code2Wav full-window recompute.

### Phase D: Quality and performance gates

Required gates:

- compare waveform against full Code2Wav on fixed code sequences
- ASR roundtrip with the same Qwen3 ASR path
- listen test for chunk boundaries
- dual-resident RTF and lowest MemAvailable

## Expected Benefit

Stateful Code2Wav would mainly improve:

- smaller chunks than `50/97`
- lower first-audio latency after enough RVQ frames exist
- long streaming with many chunks
- less repeated activation work around chunk boundaries

It will not remove Talker/CodePredictor generation time. Current dual-resident RTF is already below 1 with bounded stateless overlap, so stateful is an incremental latency/headroom project, not a blocker for the current target.

## Main Risks

- ConvTranspose state alignment can produce clicks or phase shifts.
- RoPE position reset will silently degrade attention output if `position_offset` is missed.
- A stateful ONNX with many state tensors can be hard to maintain unless names and shapes are generated systematically.
- TensorRT profile gains are not guaranteed for very tiny chunks because enqueue/fixed overhead remains.

## Recommendation

Do not attempt to patch the current TensorRT engine into stateful mode. Build a PyTorch stateful reference first. If Phase A cannot match full-window output, stop there and keep the current `50/97/context=3` strategy.
