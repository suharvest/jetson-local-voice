# EdgeLLM Qwen3-TTS Streaming PR Design — 2026-05-03

## Goal

Make Qwen3-TTS usable for conversational low-latency playback in TensorRT-Edge-LLM without coupling the core runtime to any product-specific HTTP or IPC protocol.

The PR shape should be:

1. `Qwen3OmniTTSRuntime` emits complete RVQ frames as they are generated.
2. `Code2WavRunner` can vocode a sliding window and return only newly generated PCM.
3. Example binaries demonstrate NDJSON/file-based streaming, while integrators can plug their own transport.

## Current EdgeLLM Limitation

Current flow:

```text
Qwen3OmniTTSRuntime::handleAudioGeneration()
  -> generates all RVQ frames
qwen3_tts_worker
  -> transposes all frames
  -> Code2WavRunner::generateWaveform()
  -> writes one full WAV
```

This is resident but not truly streaming. TTFT waits for the whole Talker/CodePredictor generation plus full Code2Wav.

## Proposed Runtime Boundary

### 1. RVQ Frame Callback

Add an optional callback overload:

```cpp
using RvqFrameCallback =
    std::function<void(std::vector<int32_t> const& frameCodes, int32_t totalFrames)>;

bool handleAudioGeneration(
    TalkerGenerationRequest const& request,
    TalkerGenerationResponse& response,
    cudaStream_t stream,
    RvqFrameCallback const& frameCallback);
```

The callback is invoked after each complete frame of RVQ codes is generated and before the next Talker decode step. This lets callers start vocoding the first chunk as soon as `first_chunk_frames` are available.

The existing non-streaming API remains unchanged and delegates to the callback overload with an empty callback.

### 2. Sliding-Window Code2Wav

Add:

```cpp
bool generateWaveformChunk(
    std::vector<std::vector<int32_t>> const& codes,
    int64_t skipContextFrames,
    std::vector<float>& outputSamples,
    cudaStream_t stream);
```

Caller provides a window:

```text
window_start = max(0, last_emitted_frames - left_context)
window = frames[window_start : total_frames]
skip_context = last_emitted_frames - window_start
```

`Code2WavRunner` runs normal vocoding on the window and returns only samples after `skip_context * upsample_rate`.

## Example Worker Protocol

The example worker keeps backward compatibility:

```json
{"id":"1","text":"你好。","output_file":"/tmp/out.wav"}
```

Streaming is opt-in:

```json
{
  "id": "1",
  "text": "你好。",
  "output_file": "/tmp/out.wav",
  "stream": true,
  "first_chunk_frames": 5,
  "chunk_frames": 25
}
```

For service integrations that do not need a final WAV, use streaming-only PCM:

```json
{
  "id": "1",
  "text": "你好。",
  "stream": true,
  "stream_only": true,
  "first_chunk_frames": 1,
  "chunk_frames": 25,
  "chunk_format": "pcm_s16le",
  "chunk_transport": "base64"
}
```

The worker emits chunk events before final `done`:

```json
{"event":"chunk","chunk_file":"/tmp/out.chunk0.wav","frames":5,"is_final":false,"elapsed_ms":...}
{"event":"chunk","chunk_file":"/tmp/out.chunk1.wav","frames":30,"is_final":true,"elapsed_ms":...}
{"event":"done","output_file":"/tmp/out.wav","first_chunk_ms":...}
```

File-based WAV chunks are example-only. The Jetson voice backend now uses inline base64 `pcm_s16le` chunks and yields raw PCM bytes from `/tts/stream`.

## Why This Matches EdgeLLM Direction

- Keeps Talker RVQ generation and Code2Wav vocoding modular.
- Does not add HTTP, Python, or product-specific state to core runtime.
- Preserves all existing non-streaming behavior.
- Reuses existing `Code2WavRunner` engine/profile handling.
- Makes streaming a caller concern through a minimal callback.

## Nano Validation

Build target:

```bash
make qwen3_tts_worker -j2
```

Streaming run, 1-frame first chunk, 8-frame follow-up chunks, resident worker:

```json
{"event":"chunk","frames":1,"elapsed_ms":636.3,"code2wav_ms":578.7,"samples":1920}
{"event":"done","first_chunk_ms":636.3,"frames":27,"audio_s":2.16,"rtf":2.28}
```

Cold first request still measured around `1150 ms` first chunk because TensorRT/Code2Wav first execution needs warmup. A conversational service should keep the worker resident and issue a warmup request before accepting user traffic.

After adding `stream_only=true` and inline PCM chunks, the final full Code2Wav pass is skipped. With `first_chunk_frames=1` and `chunk_frames=25`, the hot resident run measured:

```json
{"event":"done","first_chunk_ms":638.2,"audio_s":1.76,"rtf":1.38}
```

The Python `TRTEdgeLLMTTSBackend.generate_streaming()` integration consumed the same protocol through the resident worker. In a two-request hot test, the second request yielded the first PCM chunk at `0.64 s`.

Adaptive chunking now keeps the first chunk small but grows follow-up chunks to reduce Code2Wav calls:

```json
{
  "adaptive_chunks": true,
  "first_chunk_frames": 1,
  "chunk_frames": 25,
  "chunk_growth_frames": 25,
  "max_chunk_frames": 100
}
```

On a longer Chinese test sentence, hot resident metrics improved:

```json
{"mode":"fixed_25","first_chunk_ms":639.4,"audio_s":8.88,"rtf":1.10}
{"mode":"adaptive_25_50_75_100","first_chunk_ms":638.8,"audio_s":8.88,"rtf":0.98}
```

This is the current best default for conversational streaming: one tiny first chunk for TTFT, then larger chunks to amortize the roughly fixed Code2Wav invocation cost.

A follow-up sweep on the old `min=1,opt=300,max=1000` Code2Wav engine showed that faster chunk growth is better for the Nano runtime:

```json
{"mode":"fixed_25","first_chunk_ms":636.7,"audio_s":6.96,"code2wav_ms":2877.3,"rtf":1.128}
{"mode":"adaptive_25_50_100","first_chunk_ms":638.7,"audio_s":6.96,"code2wav_ms":2306.3,"rtf":1.047}
{"mode":"adaptive_25_75_150","first_chunk_ms":636.1,"audio_s":6.96,"code2wav_ms":1726.2,"rtf":0.962}
{"mode":"adaptive_50_100_200","first_chunk_ms":635.9,"audio_s":6.96,"code2wav_ms":1729.6,"rtf":0.963}
```

Keeping `first_chunk_frames=1` preserved the `~0.64s` TTFT. Changing only the first chunk size from `1` to `5/10/15/25` did not materially change total RTF, but delayed TTFT to `0.86s/1.15s/1.44s/2.00s`. This is useful for an "instant feedback" profile, but it is not the right default for conversational playback because the first chunk contains too little audio to bridge to the next chunk.

There is a separate playback-continuity tradeoff. `first_chunk_frames=1` gives the lowest TTFT, but the first chunk contains only `80 ms` of audio, so a naive player will underrun before the next chunk arrives. A smoother mode can use a larger first chunk:

```json
{"mode":"first10_growth15","first_chunk_ms":1147.6,"audio_s":6.96,"rtf":1.129}
{"mode":"first15_growth20","first_chunk_ms":1433.1,"audio_s":6.96,"rtf":1.045}
{"mode":"first20_growth30","first_chunk_ms":1714.8,"audio_s":6.96,"rtf":0.962}
```

For V2V latency reporting, keep two metrics: first emitted audio and playable start time. Playable start time is the first chunk arrival time from which a client can begin playback without underrunning before the next chunk arrives. On the current Nano worker, `first=1,chunk=25,growth=50,max=150` emits audio early but does not become self-sustaining until about `4.1s`. The best measured continuous-playback policy was `first=25,chunk=25,adaptive=false`, with first usable playback at about `2.34s`.

The Python backend exposes this as `EDGE_LLM_TTS_STREAMING_PROFILE`:

```text
continuous_playback: first=25, chunk=25, adaptive=false
instant_feedback:    first=1,  chunk=25, growth=50, max=150
playback/smooth:     first=20, chunk=20, growth=30, max=120
```

`continuous_playback` is now the default for the service path. `instant_feedback` remains available when a caller wants the lowest first-byte/first-audio time and will buffer client-side before real playback.

The remaining RTF cost is mostly from Code2Wav itself. A background Code2Wav queue with a separate CUDA stream was tested as an experimental `async_code2wav` path, but it did not materially improve Nano hot metrics:

```json
{"event":"done","first_chunk_ms":667.2,"audio_s":1.76,"rtf":1.38}
```

This suggests Talker/CP and Code2Wav do not overlap effectively on the Nano GPU, or Code2Wav dominates scheduling enough that a second stream cannot hide it. The async path should remain opt-in for now.

`Code2WavRunner::generateWaveformChunk()` was also tightened so short streaming windows copy only the newly emitted samples back to host, instead of materializing a complete CPU waveform and slicing it. This keeps the API cleaner and reduces host-copy overhead, but the measured hot RTF remained around `1.38`, confirming that the TensorRT vocoder enqueue is the dominant cost.

An attempted Code2Wav CUDA Graph cache for fixed `seqLen` shapes did not improve Nano performance. On the same hot prompt, graph disabled measured `first_chunk_ms=636.3, rtf=0.962`, while graph enabled measured `first_chunk_ms=637.5, rtf=0.964`. Keep this out of the default path unless a future TensorRT/JetPack build shows a measurable benefit.

The existing `audio_build` tool already supports Code2Wav profile knobs:

```bash
./examples/multimodal/audio_build \
  --onnxDir=/home/harvest/qwen3-tts-trt-edge-llm-export/tokenizer_decoder \
  --engineDir=/home/harvest/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder/code2wav_stream50 \
  --minCodeLen=1 \
  --optCodeLen=50 \
  --maxCodeLen=300
```

On the 8GB Nano this build was interrupted after about 35 minutes because TensorRT was still tactic-searching and repeatedly skipping tactics that requested `4.5GB` to `13.5GB` of device memory while only about `3.6GB` was available. The next Code2Wav profile rebuild should run on Orin NX or another device with more available memory, or use stricter builder memory/tactic limits if we add those to `audio_build`.

An Orin NX rebuild with `EDGE_LLM_TRT_WORKSPACE_MB=2048` succeeded for the tighter streaming profile:

```bash
./examples/multimodal/audio_build \
  --onnxDir=/home/harvest/qwen3-tts-trt-edge-llm-export/tokenizer_decoder \
  --engineDir=/home/harvest/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder/code2wav_stream100_nx_ws2048 \
  --minCodeLen=1 \
  --optCodeLen=50 \
  --maxCodeLen=100
```

Build time was about `64.6 min`. TensorRT reported peak builder allocator usage of about `9.0GB` GPU and `2.9GB` CPU. The produced engine was copied back to the Nano at:

```text
/home/harvest/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder/code2wav_stream100_nx_ws2048/code2wav
```

However, this tighter profile should not become the default yet. On the same Nano prompt, the old `min=1,opt=300,max=1000` engine was faster:

```json
{"engine":"old_opt300_max1000","first_chunk_ms":636.0,"audio_s":6.96,"code2wav_ms":2309.7,"rtf":1.045}
{"engine":"new_opt50_max100","first_chunk_ms":642.4,"audio_s":6.96,"code2wav_ms":2900.6,"rtf":1.132}
```

The regression came from larger streaming chunks: the new profile ran the 75-frame chunk at about `1158 ms`, while the old engine handled the same shape at about `578 ms`. Keep the old Code2Wav engine for the runtime path until profile/tactic selection is understood.

The old self-implemented path was checked again as a possible escape hatch. Its ORT CUDA EP fallback is not automatically faster on the current Nano image. Host Python only has CPU ORT:

```text
onnxruntime 1.23.2 providers ['AzureExecutionProvider', 'CPUExecutionProvider']
```

The old `jetson-voice-speech:v3.4-slim` container exposes CUDA/TensorRT providers, but CUDA EP initially failed because the container lacked JetPack 6 CUDA libraries:

```text
Failed to load library libonnxruntime_providers_cuda.so with error: libcublasLt.so.12: cannot open shared object file
```

Mounting the host CUDA/TensorRT libraries fixes provider loading:

```bash
docker run --rm --runtime nvidia --network host \
  -v /home/harvest:/home/harvest \
  -v /usr/local/cuda-12.6:/usr/local/cuda-12.6:ro \
  -v /usr/lib/aarch64-linux-gnu:/host-usr-lib-aarch64:ro \
  -e LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:/host-usr-lib-aarch64 \
  jetson-voice-speech:v3.4-slim \
  python3 /home/harvest/vocoder_ort_bench.py \
    --model /home/harvest/voice_test/models/qwen3-tts/onnx/vocoder_fp16.onnx \
    --provider cuda
```

Steady-state CUDA EP timing with zero RVQ codes was about `548 ms` per invocation for `1/5/10/20/25/50/75/100/150` input frames, and the ONNX output tensor stayed at `8.0 s` of audio. This means the old ORT path is effectively a fixed-cost full decoder call for this exported model. It can match the rough `~0.55 s` Code2Wav floor, but it does not directly provide a `100 ms` first-vocoder path.

The old native streaming parameters were also replayed on EdgeLLM. The old runtime used `left_context=25`, `first_chunk_frames=10` originally, a planned `first_chunk_frames=5`, and `chunk_frames=25`; its TRT wrapper used `max_frames=100` for `vocoder_fp16.engine`. On the current EdgeLLM worker, switching only to those old chunk parameters did not improve latency:

```json
{"mode":"oldlike_first5_fixed25_oldeng","first_chunk_ms":1143.0,"audio_s":5.92,"code2wav_ms":2452.3,"rtf":1.157}
{"mode":"oldlike_first10_fixed25_oldeng","first_chunk_ms":1463.7,"audio_s":5.92,"code2wav_ms":2398.1,"rtf":1.164}
{"mode":"current_first1_adapt25_50_150_oldeng","first_chunk_ms":876.2,"audio_s":5.92,"code2wav_ms":1932.0,"rtf":1.050}
```

The already-built `min=1,opt=50,max=100` Code2Wav engine also did not reproduce the old `~100 ms` vocoder behavior:

```json
{"mode":"oldlike_first5_fixed25_max100eng","first_chunk_ms":1155.3,"audio_s":5.92,"code2wav_ms":2452.2,"rtf":1.161}
{"mode":"current_first1_adapt25_50_150_max100eng","first_chunk_ms":888.8,"audio_s":5.92,"code2wav_ms":2549.0,"rtf":1.159}
```

For playable start time rather than first emitted audio, the measured policy comparison was:

```json
{"mode":"oldlike_first5_fixed25_oldeng","first_chunk_ms":1102.0,"playable_start_ms":5122.0}
{"mode":"oldlike_first10_fixed25_oldeng","first_chunk_ms":1478.7,"playable_start_ms":5506.3}
{"mode":"current_first1_adapt25_50_150_oldeng","first_chunk_ms":845.4,"playable_start_ms":6202.7}
{"mode":"playback_first20_adapt20_30_120_oldeng","first_chunk_ms":2025.0,"playable_start_ms":6299.4}
{"mode":"fixed_first25_chunk25_oldeng","first_chunk_ms":2324.9,"playable_start_ms":4340.4}
```

Because `fixed_first25_chunk25` has a 2s first buffer, the next chunk arrives only about `15.6 ms` after that buffer would end. With a small client-side safety buffer, it can be treated as playable from roughly `2.34s`. Conclusion: do not switch EdgeLLM defaults back to the old `first=5/10,fixed25` policy. Use `first=25,fixed25` for conversational playback and keep `first=1,adaptive` only for instant feedback. The missing deeper piece is still the old native `vocoder_fp16.engine`/wrapper behavior, or a different export/profile that actually specializes the short streaming shapes.

Current conclusion: keep EdgeLLM TRT Code2Wav as the default PR path, keep `async_code2wav` opt-in, and treat sub-500ms V2V as a pipeline problem rather than a simple vocoder backend swap. The useful next optimizations are:

1. overlap Code2Wav with subsequent Talker/CP only on devices where a second CUDA stream shows real overlap;
2. add a dedicated two-stage TTS worker mode for 16GB devices so Talker/CP and Code2Wav can be scheduled independently;
3. investigate graph surgery or a new exported vocoder that returns only the needed streaming samples instead of an 8s fixed output;
4. hide the remaining first-vocoder floor with ASR partials/speculative TTS in the voice-to-voice pipeline.

TTS correctness requires the special CodePredictor path on the current Nano runtime:

```bash
QWEN3_TTS_CP_ENGINE=/home/harvest/voice_test/models/qwen3-tts/engines/cp_bf16.engine
QWEN3_TTS_CP_EMBED_FP32=/home/harvest/voice_test/models/qwen3-tts/onnx/cp_embed_fp32.bin
```

With that path enabled, `/tmp/edgellm_cp_default.wav` generated from `你好。` was transcribed by the EdgeLLM ASR backend as:

```json
{"text":"你好。","language":"Chinese"}
```

Without this CP path, the same short prompt can drift semantically, for example ASR read one generated WAV as `是的，是的。`.

## Memory Impact

Sliding-window Code2Wav reduces the temporary memory peak that grows with utterance length, because the vocoder no longer needs to process the full generated RVQ sequence at once. It does not materially reduce fixed resident memory from TensorRT engines, weights, context memory, and KV caches.

For dual-resident ASR + TTS on the 8GB Nano, the main memory gap is still fixed residency. Sliding-window vocoding helps long replies stay bounded and enables low TTFT, but it is not enough by itself to close the previous `~1.3-2GB` free-memory gap for robust dual residency.

## Follow-Up Optimizations

1. Investigate Code2Wav tactic selection for small profiles; the `opt=50,max=100` engine builds successfully but is slower than the old `opt=300,max=1000` engine on 75-frame chunks.
2. Keep adaptive chunks enabled by default for the service path while using the old Code2Wav engine: `1 -> 25 -> 75 -> 150`.
3. Add more builder controls to `audio_build` if needed, beyond `EDGE_LLM_TRT_WORKSPACE_MB`, so Nano/NX can avoid pathological tactics deterministically.
4. Add binary stdout or socket transport for PCM to avoid base64 expansion in high-throughput services.
5. Warm common Code2Wav shapes (`1`, `25` frames) when memory allows.
6. Add a playback-aware mode that sets `first_chunk_frames=20` for smoother local playback when TTFT below one second is less important than avoiding underrun.
7. Reuse the special CodePredictor path by default when matching assets are present.
8. Move chunk assembly into a reusable helper if more examples need it.
9. Add CP graph cache / KV-zero optimizations from the old native runner after streaming correctness is stable.
