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

The remaining RTF cost is mostly from synchronous Code2Wav chunking: the Talker waits while each chunk is vocoded. Moving Code2Wav to a separate CUDA stream/thread is the next major latency/RTF optimization.

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

1. Move Code2Wav chunking to a separate CUDA stream/thread so Talker generation can continue while the previous chunk is vocoded.
2. Add binary stdout or socket transport for PCM to avoid base64 expansion in high-throughput services.
3. Warm common Code2Wav shapes (`1`, `25` frames) when memory allows.
4. Reuse the special CodePredictor path by default when matching assets are present.
5. Move chunk assembly into a reusable helper if more examples need it.
6. Add CP graph cache / KV-zero optimizations from the old native runner after streaming correctness is stable.
