# ASR Architecture Cleanup

**Date:** 2026-04-15
**Goal:** Remove dead code, merge Sherpa's two-layer wrapping, add unit tests.
**Constraint:** No changes to external interfaces (WebSocket protocol, HTTP API, env vars).

## Problem

Sherpa ASR has three layers of wrapping that evolved historically:
1. `streaming_asr_service.py` — wraps sherpa_onnx (model loading, feed_and_decode, finalize)
2. `backends/sherpa_asr.py` — wraps streaming_asr_service into ASRStream interface
3. `main.py:_asr_stream_sherpa()` — calls streaming_asr_service directly (dead code path)

Plus: `asr_service.py` wraps SenseVoice for offline, also wrapped by SherpaASRBackend.

## Changes

### 1. Delete dead code (~150 lines)

- `main.py:_asr_stream_sherpa()` — unreachable when SherpaASRBackend has STREAMING capability
- `main.py` `/asr` legacy fallback — unreachable when any backend loads
- `main.py` startup defensive `streaming_asr_service.preload()` — handled by backend
- `vc_service.py` — unimplemented stubs, no endpoints use it

### 2. Merge streaming_asr_service.py into backends/sherpa_asr.py

Move into `SherpaASRBackend`:
- `get_recognizer()` → `SherpaASRBackend._get_recognizer()` (model loading + config)
- Endpoint detection config (rule1/rule2/rule3 silence thresholds)
- Language mode handling (Paraformer vs Zipformer)

Move into `SherpaASRStream`:
- `feed_and_decode()` logic → `accept_waveform()`
- `finalize()` logic → `finalize()`
- Endpoint state tracking (already done in prior commit)

Delete: `streaming_asr_service.py`, `asr_service.py`

### 3. Add unit tests

New `app/tests/` directory, runnable without CUDA/sherpa_onnx:

- `test_sherpa_stream.py` — mock recognizer, verify endpoint state propagation and stream reset
- `test_qwen3_stream.py` — sliding window, LocalAgreement, EOS endpoint detection
- `test_ws_reset.py` — WebSocket reset protocol (mock WebSocket)
- `test_backend_factory.py` — create_asr_backend selection logic

### 4. Final file structure

```
app/
  main.py              — routes + glue (slimmed)
  asr_backend.py       — ASR abstract layer + factory (unchanged)
  tts_backend.py       — TTS abstract layer (unchanged)
  tts_service.py       — TTS proxy (unchanged)
  backends/
    sherpa_asr.py       — complete Sherpa ASR (model loading + streaming + offline)
    sherpa.py           — Sherpa TTS (unchanged)
    qwen3_asr.py        — Qwen3 ASR (unchanged)
    qwen3_trt.py        — Qwen3 TTS (unchanged)
  tests/
    test_sherpa_stream.py
    test_qwen3_stream.py
    test_ws_reset.py
    test_backend_factory.py

  # Deleted:
  # streaming_asr_service.py
  # asr_service.py
  # vc_service.py
```

## Non-goals

- No TTS changes
- No Qwen3 code changes
- No env var interface changes
- No main.py router split (lower priority, future work)
