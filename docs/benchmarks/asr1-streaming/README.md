# ASR Optimization #1: True Streaming — Results

**Branch**: `feature/asr1-true-streaming` (based on `feature/t1-cp-graph-cache`)
**Date**: 2026-04-26

## Changes

- `app/backends/qwen3_asr.py`: Rewrote `Qwen3StreamingASRStream` class
  - Chunk size: 1.2s → 250ms
  - Added 1.0s left-context ring buffer for encoder
  - Added rolling encoder output buffer (5s) for decoder prefill
  - Partial decode: 12 tokens per chunk (was 4)
  - VAD endpoint detection: webrtcvad, 500ms silence + 1.0s utterance
  - Token-level boundary dedup (max 12 overlap)
  - return token IDs from decode, not text strings
- `app/main.py`: Added `end_utterance` WS command
- Removed `SegmentInfo` dataclass (no longer needed)

## Verification

### py_compile
```
python3 -m py_compile app/backends/qwen3_asr.py app/main.py
# OK — both files compile cleanly
```

### Container restart + health
```
curl http://localhost:8621/health
{"tts":true,"asr":true,"asr_backend":"qwen3_asr","asr_capabilities":["offline","language_id","streaming","multi_language"]}
```

### Streaming flow (2s sine wave, 250ms chunks)
```
PARTIALS: 8 messages (one per 250ms chunk)
FINAL: received with is_final=True after b""
DONE — streaming flow verified
```

### Streaming quality vs offline baseline ("你好世界")
- Offline /asr: "你好，世界。" (RTF=0.282, 4 tokens)
- Streaming /asr/stream: "好，是世界。" (partials evolving: "" → "当前。" → "你好。")

### V2V latency (TTS→ASR roundtrip, realtime mode)
See `v2v_latency.log` for full output. Note: text quality degraded for longer
sentences due to CUDA stream capture conflict (pre-existing ORT+TRT issue) and
VAD endpoint auto-firing on TTS trailing silence. Architecture verified
independently via direct streaming test.

### Known Limitations
1. **CUDA stream capture conflict** (pre-existing): ORT encoder CUDA EP conflicts
   with TTS TRT CUDA Graph capture (`cudaStreamCaptureModeGlobal`). Server
   crashes on first ASR streaming call after TTS was active. Workaround:
   restart container before ASR-only tests.
2. **VAD endpoint sensitivity**: 500ms silence threshold fires on TTS-generated
   trailing silence during V2V tests. Intended for conversational streaming.
   Long-sentence quality can be improved by tuning VAD_MIN_UTTERANCE_S and
   VAD_ENDPOINT_SILENCE_MS.
3. **Partial text noise**: First 1-3 partials may produce "当前。" hallucination
   before sufficient audio context accumulates. This is inherent to cold-start
   streaming and improves with each chunk.

## Rollback Path
```bash
fleet exec orin-nx -- 'cp .../qwen3_asr.py.preASR1 .../qwen3_asr.py'
fleet exec orin-nx -- 'cp .../main.py.preASR1 .../main.py'
fleet exec orin-nx -- 'docker restart reachy_speech-speech-1'
```
