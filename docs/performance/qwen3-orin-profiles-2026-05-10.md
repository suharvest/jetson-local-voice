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
| `orin-nx` | 40W locked | round 0 | `185.1ms` | `426.4ms` | `611.5ms` | `~8.8GB` |
| `orin-nx` | 40W locked | round 1 | `172.0ms` | `425.8ms` | `597.8ms` | `~8.8GB` |
| `orin-nx` | 40W locked, NX-native TTS engines | round 0 | `173.0ms` | `422.7ms` | `595.8ms` | `~8.2GB` |
| `orin-nx` | 40W locked, NX-native TTS engines | round 1 | `172.8ms` | `422.3ms` | `595.1ms` | `~8.2GB` |

### Highperf TTS-Only

Text: `今天我们继续验证低延迟流式生成的效果。`

Streaming policy: stateful Code2Wav, `first_chunk_frames=7`, `chunk_frames=10`, `max_chunk_frames=10`, `ACTIVE_CP_GROUPS=13`.

| Device | Power | First chunk | RTF | Notes |
| --- | --- | ---: | ---: | --- |
| `orin-nano` | 25W | `~601-602ms` | `0.696-0.703` | ASR round-trip exact on two WAVs |
| `orin-nano` | MAXN_SUPER | `~540-542ms` | `0.641-0.648` | Same highperf engine set |
| `orin-nx` | 40W | `~588-591ms` | `0.702-0.711` | From-zero transferred highperf artifacts; ASR round-trip exact |
| `orin-nx` | 40W locked | `~533-535ms` | `0.619-0.628` | `jetson_clocks` locked; nano-imported engines; ASR round-trip exact |
| `orin-nx` | 40W locked, NX-native TTS engines | `~530-532ms` | `0.613-0.623` | NX-built Talker W8A16, CP pretranspose, and stateful Code2Wav; ASR round-trip exact |

## Official Smoke

Official/minimal is not the 8GB dual-resident target. It should stay close to upstream and only prove that the official-style runtime can start and synthesize with semantic fixes.

Latest smoke:

| Line | Device | Result | Notes |
| --- | --- | --- | --- |
| `official` TTS worker | `orin-nano` | pass | Generated `/tmp/qwen3_profile_regression_0510/official_like.wav`; short `你好`, `audio_s=0.24`, `total_ms=740.4`; not a quality or dual-resident gate |
| `official` TTS worker | `orin-nx` | pass | Generated `/tmp/qwen3_profile_regression_0510_nx/official_nx.wav`; short `你好`, `audio_s=0.72`, `total_ms=1633.1`, `rtf=2.27`; not a quality or dual-resident gate |
| `highperf` V2V | `orin-nano` | pass | MAXN warm `EOS -> first audio=620.0ms`, ASR text exact |
| `highperf` V2V | `orin-nx` | pass | 40W warm `EOS -> first audio=670.3-671.5ms`, ASR text exact |
| `highperf` V2V | `orin-nx` | pass | 40W locked warm `EOS -> first audio=597.8-611.5ms`, ASR text exact |
| `highperf` V2V | `orin-nx` | pass | 40W locked NX-native TTS engines `EOS -> first audio=595.1-595.8ms`, ASR text exact |

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

### NX 40W Locked Run

Date: 2026-05-11.

Runtime state:

- `nvpmodel`: 40W mode id `4`
- `jetson_clocks`: enabled
- CPU: `1497MHz`
- GPU: `1173MHz`
- EMC: `3199MHz`
- Idle before run: `~10GB MemAvailable`

Engine set: `nano-imported-on-nx-40w-locked`. These engines were transferred from `orin-nano` and must be preserved separately from any later NX-native rebuild.

Artifacts and logs:

- Output directory: `/tmp/qwen3_profile_opt_0511_nx`
- TTS-only log: `/tmp/qwen3_profile_opt_0511_nx/tts_only_nano_imported_40w_locked.log`
- V2V log: `/tmp/qwen3_profile_opt_0511_nx/v2v_nano_imported_40w_locked.log`
- Quality round-trip log: `/tmp/qwen3_profile_opt_0511_nx/quality_roundtrip_tts_wav_40w_locked.log`
- Engine manifest: `/tmp/qwen3_profile_opt_0511_nx/engine_manifest_nano_imported_40w_locked.txt`

Quality check:

| WAV | ASR text | Result |
| --- | --- | --- |
| `/tmp/qwen3_quality_product_set1.smoke_1.wav` | `请关闭卧室的空调。` | exact |
| `/tmp/qwen3_profile_opt_0511_nx/highperf_tts_nx_40w_locked.smoke_1.wav` | `今天我们继续验证低延迟流式生成的效果。` | exact |

### NX-Native TTS Engine Run

Date: 2026-05-11.

Runtime state:

- `nvpmodel`: 40W mode id `4`
- `jetson_clocks`: enabled
- Idle before TTS run: `~11GB MemAvailable`

Engine set: `nx-native-tts-0511`. These engines were built on `orin-nx` and must be preserved separately from the `nano-imported-on-nx-40w-locked` set.

Build script:

- Local source: `scripts/build_qwen3_nx_native_engines.sh`
- Remote build command: `bash /tmp/build_qwen3_nx_native_engines.sh`
- Output root: `/tmp/qwen3_native_engines_nx_0511`

Built TTS engines:

| Component | Path | Size | md5 |
| --- | --- | ---: | --- |
| Talker W8A16 output-k | `/tmp/qwen3_native_engines_nx_0511/talker_w8a16_outputk/talker_decode_w8a16_outputk.engine` | `439M` | `267e8fdfc782172c0df0eac8a92a04af` |
| CP lm-head pretranspose | `/tmp/qwen3_native_engines_nx_0511/cp_lmhead_pretranspose/cp_dir/qwen3_tts_cp.engine` | `212M` | `b15ea75ca399a6ba39be72a59552d288` |
| Stateful Code2Wav | `/tmp/qwen3_native_engines_nx_0511/code2wav_stateful/code2wav_stateful.engine` | `224M` | `a06f96ec00cc25351965e43c9d9f077a` |

Artifacts and logs:

- TTS-only output directory: `/tmp/qwen3_profile_opt_0511_nx_native`
- TTS-only log: `/tmp/qwen3_profile_opt_0511_nx_native/tts_only_nx_native_40w_locked.log`
- V2V log: `/tmp/qwen3_profile_opt_0511_nx_native/v2v_nx_native_40w_locked.log`
- Quality round-trip log: `/tmp/qwen3_profile_opt_0511_nx_native/quality_roundtrip_nx_native_40w_locked.log`
- Engine manifest: `/tmp/qwen3_native_engines_nx_0511/engine_manifest_nx_native_0511.txt`

Quality check:

| WAV | ASR text | Result |
| --- | --- | --- |
| `/tmp/qwen3_quality_product_set1.smoke_1.wav` | `请关闭卧室的空调。` | exact |
| `/tmp/qwen3_profile_opt_0511_nx_native/highperf_tts_nx_native_40w_locked.smoke_1.wav` | `今天我们继续验证低延迟流式生成的效果。` | exact |

Comparison against nano-imported TTS engines on the same locked NX runtime:

| Engine set | TTS first chunk | TTS RTF | V2V EOS -> first audio |
| --- | ---: | ---: | ---: |
| `nano-imported-on-nx-40w-locked` | `~533-535ms` | `0.619-0.628` | `597.8-611.5ms` |
| `nx-native-tts-0511` | `~530-532ms` | `0.613-0.623` | `595.1-595.8ms` |

Conclusion: NX-native TTS engine rebuild is valid and slightly faster, but the gain is small (`~2-4ms` on TTS first chunk and about `2-3ms` on the best V2V run). The main performance jump was still the runtime state (`40W` + `jetson_clocks`) rather than tactic rebuild alone.
