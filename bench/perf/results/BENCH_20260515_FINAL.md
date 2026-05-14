# Streaming Benchmark — 2026-05-15 Final Report

## Test Configuration

- **Test harness**: `bench/perf/perf.py` — ASR / TTS / V2V unified streaming benchmarks
- **Parameters**: warmup=3, runs=10, chunk_ms=250, realtime=True, eos=forced, llm_delay=0
- **Corpus**: 20 WAV files (10 short zh/en + 10 long zh/en) from `bench/perf/corpus/`
- **TTS prompts**: 20 prompts from `bench/perf/corpus/tts_prompts.json`

## Device Matrix

| Device | Arch | ASR Backend | TTS Backend | Port |
|--------|------|------------|-------------|------|
| **RK3576** (cat-remote) | aarch64, 8GB | rk:qwen3_asr_rk (RKLLM w4a16) | rk:matcha_rknn | 8000 |
| **RK3588** (radxa) | aarch64, 16GB | rk:qwen3_asr_rk (RKLLM w8a8) | rk:matcha_rknn | 8000 |
| **Orin Nano** | aarch64, 8GB | trt_edgellm | trt_edgellm | 8000 |
| **Orin NX** | aarch64, 16GB | trt_edgellm | trt_edgellm | 18092 |
| **Pi 5** (harvest-pi) | aarch64, 8GB | sherpa_asr | sherpa | 8765/8766 |

---

## 1. ASR Streaming

**Key metric**: fRTF (EOS→final time / audio duration) — pure compute, cross-device comparable.
Wall RTF includes realtime pacing sleep and is not meaningful for comparison.

### fRTF

| Device | short zh | long zh | short en | long en |
|--------|---------|---------|---------|---------|
| **RK3576** | 0.511 | 0.409 | 0.301 | 0.233 |
| **RK3588** | 0.347 | 0.154 | 0.350 | 0.150 |
| **Orin Nano** | 0.077 | 0.064 | 0.071 | 0.063 |
| **Orin NX** | 0.106 | 0.101 | 0.098 | 0.080 |
| **Pi 5** | ~0.000 | ~0.000 | ~0.000 | ~0.000 |

### TFD (Time to First Dictation)

| Device | short zh | long zh | short en | long en |
|--------|---------|---------|---------|---------|
| **RK3576** | 1685ms | 1516ms | 1517ms | 2106ms |
| **RK3588** | 4713ms | 2905ms | 6719ms | 2438ms |
| **Orin Nano** | 3532ms | 13922ms | 4290ms | 12054ms |
| **Orin NX** | 3702ms | 16330ms | 4409ms | 12403ms |
| **Pi 5** | 1618ms | 3166ms | 1395ms | 1879ms |

> TFD on Jetson is high because the TRT-EdgeLLM `_TRTEdgeLLMAccumulatingASRStream` buffers all audio and only transcribes on finalize — no incremental partial decode. TFD ≈ audio duration.

### Accuracy

| Device | zh CER | en WER | Notes |
|--------|--------|--------|-------|
| **RK3576** | 24-31% | 16-25% | zh_long CER improved from 80%→17% after fix |
| **RK3588** | 22-25% | 16-25% | Similar to RK3576 |
| **Orin Nano** | 13-61% | 0-6% | Best English, zh variance high |
| **Orin NX** | 10-61% | 0-11% | Best English, same zh variance |
| **Pi 5** | 28-37% | 28-37% | Sherpa no-punctuation output hurts CER |

### Errors

| Device | Before Fix | After Fix |
|--------|-----------|-----------|
| Orin NX | **6/13** (long audio) | **0/13** |
| RK3576 | 0 | 0 |
| RK3588 | 0 | 0 |
| Orin Nano | 0 | 0 |
| Pi 5 | 0 | 0 |

---

## 2. TTS

| Device | RTF (range) | TFD | Total latency (mean) | Notes |
|--------|------------|-----|---------------------|-------|
| **RK3576** | 0.14-0.22 | 5-8ms | 500-1810ms | Matcha RKNN, good |
| **RK3588** | 0.07-0.17 | 3-4ms | 214-821ms | Fastest TTS overall |
| **Orin Nano** | 0.41-0.43 | 4ms | 1593-7158ms | TRT-EdgeLLM, consistent |
| **Orin NX** | 0.40-0.43 | 4-6ms | 1633-4753ms | Same as Nano |
| **Pi 5** | 0.46-0.59 | 6-15ms | 1197-6208ms | Sherpa, below realtime |

**TTS ranking**: RK3588 > RK3576 > Orin NX ≈ Orin Nano > Pi 5

---

## 3. V2V End-to-End (EOS → First TTS Audio)

### Short utterances (zh/en, 2-4s)

| Device | EOS→Audio (mean) | ASR Finalize | TTS TFD | Best case |
|--------|-----------------|--------------|---------|-----------|
| **RK3576** | 1481ms | 1467ms | 15ms | 1121ms |
| **RK3588** | 545ms | 537ms | 8ms | 15ms |
| **Orin Nano** | 255ms | 250ms | 4ms | 183ms |
| **Orin NX** | 294ms | 289ms | 4ms | 193ms |
| **Pi 5** | 128ms | 124ms | 4ms | 4ms |

### Long utterances (zh/en, 6-15s)

| Device | EOS→Audio (mean) | ASR Finalize | TTS TFD | Worst case |
|--------|-----------------|--------------|---------|------------|
| **RK3576** | 4657ms | 4384ms | 273ms | **9510ms** |
| **RK3588** | 1659ms | 1643ms | 16ms | 4339ms |
| **Orin Nano** | 817ms | 812ms | 5ms | 1196ms |
| **Orin NX** | 982ms | 977ms | 4ms | 1293ms |
| **Pi 5** | 40ms | 34ms | 6ms | 70ms |

> EOS→Audio is dominated by ASR finalize. TTS TFD contributes <5% in all cases except RK3576 long zh (6%).

---

## 4. Issues Found & Fixed

### Fixed

| Issue | Device | Root Cause | Fix | Commit |
|-------|--------|-----------|-----|--------|
| ASR crash on long audio (6/13 fails) | **Orin NX** | Stale `trt_edge_llm_asr.py` (503 lines, no chunking) + stale `qwen3_asr.py` (missing `_split_at_silence_energy`) | Rebuild via `Dockerfile.jetson`, image `seeed-local-voice:jetson-2026-05-14` | — |
| Per-segment offline workaround for audio >5s | **RK3576/3588** | `_LONG_AUDIO_THRESHOLD_S=5.0` triggered slow re-transcription; `ROLLING_BUFFER_SEC=5.0` dropped encoder frames past 5s | Raise threshold to 15s, env `QWEN3_ASR_TRUE_ROLL_SEC=15`, `VAD_ENDPOINT_SILENCE_MS=800` | `8955336` |
| NX flat import structure | **Orin NX** | Slim image used `from asr_backend import` vs standard `from app.core.asr_backend` | Rebuild unifies with standard `app/` package structure | — |
| Orin Nano code stale | **Orin Nano** | Container had older app code | Updated `app/` from latest repo | — |

### Remaining

| Issue | Device | Severity | Notes |
|-------|--------|----------|-------|
| RK3576 V2V long zh 9.5s | RK3576 | Medium | ASR finalize=9498ms, single-NPU-core RKLLM bottleneck |
| RK ASR fRTF 0.2-0.5 | RK3576/3588 | Medium | 2-5x slower than Jetson fRTF 0.05-0.10 |
| Orin Nano TFD 3-17s | Orin Nano | Low | Buffered stream, no incremental partials |
| zh CER 60% on certain files | All devices | Low | Corpus-specific (zh_short_02, zh_long_02), not regression |

---

## 5. Key Takeaways

1. **Orin Nano/NX are best all-round**: ASR fRTF 0.05-0.10, V2V 200-1300ms, excellent English WER (0-6%)
2. **RK3588 TTS is fastest**: RTF 0.07, TFD 3ms — production-ready
3. **RK3576 needs ASR optimization**: Single NPU core + w4a16 quantization → fRTF 0.2-0.5, V2V up to 9.5s
4. **Pi is speed king for simple pipelines**: V2V 4-70ms, but accuracy suffers (28-37% CER/WER, no punctuation)
5. **TTS is solved across all devices**: All well below realtime, TFD <10ms on RK/Jetson

## 6. Commit History

| Commit | Description |
|--------|-------------|
| `8955336` | fix(rk): raise long-audio threshold 5s→15s, add rolling-buffer/VAD env vars |
| `b336fdf` | perf(rk): slim Dockerfile — drop librosa + transformers, save ~500MB |
| `bdb410a` | config(nx-profile): switch to orin-nx-highperf-2026-05-14 artifact set |

## 7. Raw Result Files

| Device | ASR | TTS | V2V |
|--------|-----|-----|-----|
| RK3576 | `_from_cat-remote/asr_streaming_local.json` | `_from_cat-remote/tts_local.json` | `_from_cat-remote/v2v_forced_llm0_local.json` |
| RK3588 | `_from_radxa/` | `_from_radxa/` | `_from_radxa/` |
| Orin Nano | `_from_orin-nano/` | `_from_orin-nano/` | `_from_orin-nano/` |
| Orin NX | `_from_orin-nx/` | `_from_orin-nx/` | `_from_orin-nx/` |
| Pi 5 | `_from_harvest-pi/` | `_from_harvest-pi/` | `_from_harvest-pi/` |

All result files at `bench/perf/results/`.
