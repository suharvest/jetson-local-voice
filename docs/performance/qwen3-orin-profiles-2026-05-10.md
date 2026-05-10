# Qwen3 Orin Profile Performance

Date: 2026-05-10

This document tracks the two Qwen3 runtime lines separately:

- `official`: minimal-diff EdgeLLM example/upstream line. It is evaluated for semantic correctness and upstream compatibility, not Orin 8GB dual residency.
- `highperf`: Jetson Voice product line. It is evaluated for Qwen3-ASR + Qwen3-TTS dual residency, V2V latency, memory floor, and TTS/ASR quality.

## Branch Map

| Line | Repository | Branch |
| --- | --- | --- |
| EdgeLLM official/minimal | TensorRT-Edge-LLM fork | `official-qwen3-tts-upstream-runtime` |
| EdgeLLM PR split | TensorRT-Edge-LLM fork | `pr-jetson-build-compat`, `pr-export-builder-robustness`, `pr-qwen3-tts-runtime-correctness` |
| Jetson Voice official integration | `jetson-voice` | `product-qwen3-tts-official-backend` |
| Jetson Voice highperf/product | `jetson-voice` | `qwen3tts-accurate-20260507` |

## Current Orin 8GB Baseline

Device: `orin-nano`, Jetson Orin NX Super, 8GB RAM.

Power modes:

| Mode | CPU | GPU | EMC |
| --- | ---: | ---: | ---: |
| 25W | `1.344GHz` | `918MHz` | `3.199GHz` |
| MAXN_SUPER | `1.728GHz` | `1020MHz` | `3.199GHz` |

Highperf frozen artifacts are recorded in `docs/plans/qwen3-current-frozen-baseline-2026-05-10.md`.

### Highperf V2V

Source WAV: `/tmp/qwen3_quality_product_set1.smoke_1.wav`

ASR text: `请关闭卧室的空调。`

| Device | Power | Round | ASR finalize | TTS first chunk | EOS -> first audio | Memory floor |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `orin-nano` | 25W | warm | `189.7ms` | `480.5ms` | `670.2ms` | `~850MB` |
| `orin-nano` | MAXN_SUPER | warm | `174.3ms` | `445.7ms` | `620.0ms` | `~759MB` |
| `orin-nx` | 40W | round 0 | `184.6ms` | `486.9ms` | `671.5ms` | not sampled |
| `orin-nx` | 40W | round 1 | `184.3ms` | `486.0ms` | `670.3ms` | not sampled |

### Highperf TTS-Only

Text: `今天我们继续验证低延迟流式生成的效果。`

Streaming policy: stateful Code2Wav, `first_chunk_frames=7`, `chunk_frames=10`, `max_chunk_frames=10`, `ACTIVE_CP_GROUPS=13`.

| Device | Power | First chunk | RTF | Notes |
| --- | --- | ---: | ---: | --- |
| `orin-nano` | 25W | `~601-602ms` | `0.696-0.703` | ASR round-trip exact on two WAVs |
| `orin-nano` | MAXN_SUPER | `~540-542ms` | `0.641-0.648` | Same highperf engine set |
| `orin-nx` | 40W | `~588-591ms` | `0.702-0.711` | From-zero transferred highperf artifacts; ASR round-trip exact |

## Official Smoke

Official/minimal is not the 8GB dual-resident target. It should stay close to upstream and only prove that the official-style runtime can start and synthesize with semantic fixes.

Latest smoke:

| Line | Device | Result | Notes |
| --- | --- | --- | --- |
| `official` TTS worker | `orin-nano` | pass | Generated `/tmp/qwen3_profile_regression_0510/official_like.wav`; short `你好`, `audio_s=0.24`, `total_ms=740.4`; not a quality or dual-resident gate |
| `official` TTS worker | `orin-nx` | pass | Generated `/tmp/qwen3_profile_regression_0510_nx/official_nx.wav`; short `你好`, `audio_s=0.72`, `total_ms=1633.1`, `rtf=2.27`; not a quality or dual-resident gate |
| `highperf` V2V | `orin-nano` | pass | MAXN warm `EOS -> first audio=620.0ms`, ASR text exact |
| `highperf` V2V | `orin-nx` | pass | 40W warm `EOS -> first audio=670.3-671.5ms`, ASR text exact |

## NX From-Zero Replication Log

Target device: `orin-nx`, 16GB RAM, 77GB free at session start, `nvpmodel` 40W mode.

Before highperf replication, two stale root-owned worker processes from an older run were cleared because they kept the device at only `~2.1GB MemAvailable`. After clearing them, idle availability was `~10GB`.

### Official

Command path:

- Worker: `/home/harvest/project/tensorrt-edge-llm/build_hlm_bf16_verify_0507/examples/omni/qwen3_tts_worker`
- Plugin: `/home/harvest/project/tensorrt-edge-llm/build_hlm_bf16_verify_0507/libNvInfer_edgellm_plugin.so`
- Engines: `/tmp/qwen3tts_ref_0507_from_nano/{talker,cp_product_unified_0506,code2wav}`

Result: pass. This is an official-style smoke only; it does not test dual residency.

### Highperf

Transferred from `orin-nano` to `orin-nx`:

- Stateful Code2Wav: `/tmp/qwen3_code2wav_stateful_engine`
- CP pretranspose path: `/tmp/qwen3_tts_cp_lmhead_pretranspose_0510` plus symlink target `/tmp/qwen3-tts-cp-nopast-0510`
- Talker W8A16: `/tmp/qwen3_talker_decode_w8a16_outputk_0510`
- Talker FP8 text embedding overlay: `/tmp/qwen3tts_ref_0507_from_nano/talker_text_embedding_fp8_0510` plus symlink target `/tmp/qwen3tts_text_embedding_fp8_0510`
- ASR thinker full-vocab FP8 embedding path: `/home/harvest/qwen3-asr-edgellm-runtime/engines/thinker_full_in128_kv256_fp8embed_0510` plus symlink target `/home/harvest/qwen3-asr-edgellm-runtime/engines/thinker_full_in128_kv256_0510`
- ASR audio encoder: `/home/harvest/qwen3-asr-trt-edge-llm-export/engines/audio_encoder`
- TTS highperf worker/plugin: `/tmp/qwen3_highperf_bin/qwen3_tts_worker`, `/tmp/qwen3_highperf_bin/libNvInfer_edgellm_plugin.so`
- ASR worker/plugin: `/tmp/qwen3_highperf_bin/qwen3_asr_worker`, `/tmp/qwen3_highperf_bin/libNvInfer_edgellm_plugin_asr.so`

Highperf policy:

- ASR vocab pruning: off
- TTS vocab pruning: off
- TTS streaming: stateful Code2Wav, `first_chunk_frames=7`, `chunk_frames=10`, `max_chunk_frames=10`
- CP decode: `QWEN3_TTS_ACTIVE_CP_GROUPS=13`, `QWEN3_TTS_CP_DECODE_CUDA_GRAPH=1`

Quality check:

| WAV | ASR text | Result |
| --- | --- | --- |
| `/tmp/qwen3_quality_product_set1.smoke_1.wav` | `请关闭卧室的空调。` | exact |
| `/tmp/qwen3_profile_regression_0510_nx/highperf_tts_nx.smoke_1.wav` | `今天我们继续验证低延迟流式生成的效果。` | exact |

Note: the Python backend still logs `Code2Wav not found at .../tokenizer_decoder/code2wav.engine` because the highperf path supplies stateful Code2Wav via `EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR`. The warning is expected for this configuration, but should be cleaned up later to reduce confusion.
