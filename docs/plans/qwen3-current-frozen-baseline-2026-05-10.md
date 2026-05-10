# Qwen3 ASR/TTS Frozen Baseline - 2026-05-10

Scope: Qwen3 ASR + Qwen3 TTS on Orin Nano/NX 8GB, streaming-first, full ASR/TTS vocab, dual-resident low-latency path.

## Default Runtime

- TTS Talker: W8A16 output-k engine
  - `/tmp/qwen3_talker_decode_w8a16_outputk_0510/talker_decode_w8a16_outputk.engine`
  - size `439M`, md5 `1fce66380b504cb008a153d9318ccc65`
- TTS text embedding: FP8 full-vocab directory
  - `/tmp/qwen3tts_ref_0507_from_nano/talker_text_embedding_fp8_0510`
- TTS CP: BF16 I/O + lm_head pretranspose
  - `/tmp/qwen3_tts_cp_lmhead_pretranspose_0510/cp_dir/qwen3_tts_cp.engine`
  - size `212M`, md5 `937d922794a91e542b338a6bd1510e4a`
- Code2Wav: stateful engine
  - `/tmp/qwen3_code2wav_stateful_engine/code2wav_stateful.engine`
  - size `223M`, md5 `72b579d62e5f2bfef0601b32cbe4d79a`
- ASR thinker: full-vocab KV256 FP8-embedding runtime directory
  - `/home/harvest/qwen3-asr-edgellm-runtime/engines/thinker_full_in128_kv256_fp8embed_0510/llm.engine`
  - symlink target md5 `5a15871064e77988041ca09a48f2b3ea`
- ASR audio encoder:
  - `/home/harvest/qwen3-asr-trt-edge-llm-export/engines/audio_encoder/audio/audio_encoder.engine`
  - size `361M`, md5 `0a43ed1492b9ede18a75ce5933383428`

Runtime knobs:

- `EDGE_LLM_TTS_STATEFUL_CODE2WAV=1`
- `EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR=/tmp/qwen3_code2wav_stateful_engine`
- `EDGE_LLM_TTS_LAZY_CODE2WAV=0`
- `EDGE_LLM_TTS_CUDA_GRAPH=0`
- `QWEN3_TTS_CP_DECODE_CUDA_GRAPH=1`
- `QWEN3_TTS_ACTIVE_CP_GROUPS=13`
- `EDGE_LLM_TTS_VOCAB_PRUNED=0`
- `EDGE_LLM_ASR_VOCAB_PRUNED=0`
- ASR uses the stable plugin path `/home/harvest/project/tensorrt-edge-llm/build_sm87/libNvInfer_edgellm_plugin.so`.

## TTS Accuracy Gate

Command output is saved on Nano under `/tmp/qwen3_current_frozen_0510`.

Input text:

`今天我们继续验证低延迟流式生成的效果。`

Generated WAVs:

- `/tmp/qwen3_current_frozen_0510/qwen3_tts_quality_default.smoke_1.wav`
- `/tmp/qwen3_current_frozen_0510/qwen3_tts_quality_default.smoke_2.wav`

Qwen3-ASR round-trip:

| WAV | Duration | RMS | Silence | ASR text | Result |
| --- | ---: | ---: | ---: | --- | --- |
| smoke_1 | `4.32s` | `3020.2` | `0.1435` | `今天我们继续验证低延迟流式生成的效果。` | pass |
| smoke_2 | `3.92s` | `1729.3` | `0.1497` | `今天我们继续验证低延迟流式生成的效果。` | pass |

## TTS Streaming Performance

TTS-only warm run, quality/default first chunk policy (`first_chunk_frames=7`, `chunk_frames=10`, `max_chunk_frames=10`):

| Run | First chunk wall | First chunk event | Total wall | Audio | RTF | Chunks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke_1 | `602.4ms` | `601.2ms` | `3.01s` | `4.32s` | `0.696` | `6` |
| smoke_2 | `601.4ms` | `600.3ms` | `2.76s` | `3.92s` | `0.703` | `6` |

TTS-only memory from worker stderr:

- before plugin: `MemAvailable=6120MB`
- after TTS runtime: `3972MB`
- after stateful Code2Wav: `3521MB`
- warm/request floor: about `3410MB`

## Dual-Resident V2V

Direct resident benchmark:

- ASR and TTS both preloaded and kept resident in one process.
- Source WAV: `/tmp/qwen3_quality_product_set1.smoke_1.wav`
- ASR text: `请关闭卧室的空调。`
- ASR plugin: stable build_sm87 plugin. The highperf plugin build failed this ASR path with `There must be one kernel to implement the MHA`.

| Round | ASR finalize | TTS first chunk | EOS -> first audio | Total measured | First chunk bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `229.5ms` | `481.0ms` | `710.5ms` | `1897.3ms` | `26880` |
| 1 | `189.7ms` | `480.5ms` | `670.2ms` | `1548.5ms` | `26880` |

Memory sampler during dual-resident benchmark:

- max `MemAvailable=5881MB`
- min `MemAvailable=850MB`
- last sample `867MB`
- samples `157` at about `0.2s` interval

## Fixed Decisions

- Keep full vocab for both ASR and TTS by default; do not use ASR/TTS vocab pruning for this baseline.
- Keep TTS CP at BF16; FP16 CP is faster in microbench but fails quality.
- Keep lm_head pretranspose CP as default; profile/workspace/builder sweeps were noise-level.
- Do not deploy the fused MLP cuBLAS-in-plugin prototype; it regressed CP decode.
- For ASR, keep the stable plugin/runtime pairing unless the highperf plugin is separately validated on ASR. The current highperf plugin pairing fails the direct resident ASR worker MHA path.
