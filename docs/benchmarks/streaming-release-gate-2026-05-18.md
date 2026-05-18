# Streaming Release Gate — 2026-05-18

Client ran on each target device against `http://127.0.0.1:8621`, so these
numbers exclude Mac-to-device network latency. TTS uses `/tts/stream`; ASR and
V2V use `/asr/stream` with forced end-of-utterance (`vad=none`) to measure the
compute-bound finalize path.

## RK True-Streaming Rerun (2026-05-18 10:06)

The original RK rows below were forced-EOS measurements and exposed a release
configuration bug: `QWEN3_ASR_STREAM_TRUE=1` was set, but
`QWEN3_ASR_CHUNK_CONFIRM` defaulted to `1`, so the backend selected
chunk-confirm before the true-streaming branch. The RK profiles and compose
files now set `QWEN3_ASR_CHUNK_CONFIRM=0` explicitly.

This rerun was started from the Mac over Tailscale with `NO_PROXY=*`, warmup 1,
steady 5, `--eos vad --vad-silence-ms 800`, against the same running release
containers on `radxa` and `cat-remote`.

| Target | TTS RTF p50 | TTS TTFB p50 | ASR EOS→final p50 | Split V2V EOS→audio p50 | `/v2v/stream` total p50 | Report set |
|---|---:|---:|---:|---:|---:|---|
| RK3588 `rk3588-default` | 0.124 | 51 ms | 0 ms | 394 ms | 1266 ms | `asr_streaming_remote_20260518-095612-972795.json`, `tts_remote_20260518-100434-434379.json`, `v2v_vad_llm0_remote_20260518-100547-864868.json`, `v2v_stream_remote_20260518-100242-740726.json` |
| RK3576 `rk3576-default` | 0.290 | 65 ms | 1016 ms | 1099 ms | 1270 ms | `asr_streaming_remote_20260518-095618-793667.json`, `tts_remote_20260518-100443-890747.json`, `v2v_vad_llm0_remote_20260518-100553-267470.json`, `v2v_stream_remote_20260518-100243-899652.json` |

Interpretation:

- RK3588 single-stream V2V is now sub-500 ms on the split endpoint path for the
  mixed steady sample median. The earlier 1.3 s / 5.9 s numbers were not TTS
  latency and were not the intended true-streaming path.
- `/v2v/stream` total p50 is about 1.27 s on both RK devices because this run
  used `vad_silence_ms=800`; endpoint hangover dominates and ASR finalize is
  effectively 0 ms for most samples.
- RK3576 remains materially slower in ASR finalization than RK3588, especially
  under split `/asr/stream` + `/tts/stream`, but it is no longer a 5-6 s path in
  the true-streaming configuration.
- Parallel `asr_tts_simul`, `parallel=2`, still serializes on the shared NPU
  resource and makes TTS first-byte latency large: RK3588 TTS TTFB p50 4186 ms,
  RK3576 TTS TTFB p50 5347 ms. Treat p=2 on RK as a stability smoke, not a
  recommended low-latency operating point.

## Device And Profile Matrix

| Target | Fleet node | Profile | TTS backend | ASR backend | Notes |
|---|---|---|---|---|---|
| Jetson Orin Nano | `orin-nano` | `jetson-multilang-highperf` | `trt_edgellm` | `trt_edgellm` | Qwen3 ASR + Qwen3 TTS voice-clone path. |
| Jetson Orin NX | `orin-nx` | `jetson-multilang-highperf-nx` | `trt_edgellm` | `trt_edgellm` | NX-native Qwen3 artifact set. |
| Jetson Orin Nano | `orin-nano` | `jetson-qwen3asr-matcha` | `matcha_trt` | `trt_edgellm` | Multilingual ASR with lightweight Matcha TTS. |
| Jetson Orin NX | `orin-nx` | `jetson-qwen3asr-matcha-nx` | `matcha_trt` | `trt_edgellm` | Multilingual ASR with lightweight Matcha TTS. |
| Jetson Orin Nano | `orin-nano` | `jetson-zh-en` | `matcha_trt` | `paraformer_trt` | Encoder fell back to ORT/CPU on the current Nano engine bundle. |
| Jetson Orin NX | `orin-nx` | `jetson-zh-en` | `matcha_trt` | `paraformer_trt` | Encoder and decoder both loaded as TRT after model resources were completed. |
| RK3588 | `radxa` | `rk3588-default` | `rk:matcha_rknn` | `rk:qwen3_asr_rk` | Validated hybrid Matcha path: acoustic on ORT, Vocos on RKNN/NPU. |
| RK3576 | `cat-remote` | `rk3576-default` | `rk:matcha_rknn` | `rk:qwen3_asr_rk` | Same RK release path, slower ASR finalize. |
| Raspberry Pi 5 | `harvest-pi` | `rpi5-default` | `sherpa` | `sherpa_asr` | CPU-only ONNX path. |

## Streaming TTS

Short Chinese prompts, warmup 1, steady 4.

| Target / profile | RTF p50 | TTFB p50 | Total p50 | Report |
|---|---:|---:|---:|---|
| Orin Nano Qwen3+Qwen3 | 0.470 | 7.3 ms | 1918.0 ms | `bench/perf/results/_from_orin-nano/tts_local_20260517-192746-535712.json` |
| Orin NX Qwen3+Qwen3 | 0.417 | 4.4 ms | 1444.6 ms | `bench/perf/results/_from_orin-nx/tts_local_20260517-192737-619602.json` |
| Orin Nano Qwen3 ASR + Matcha | 0.024 | 7.0 ms | 72.7 ms | `bench/perf/results/_from_orin-nano/tts_local_20260517-190703-935268.json` |
| Orin NX Qwen3 ASR + Matcha | 0.018 | 4.6 ms | 55.8 ms | `bench/perf/results/_from_orin-nx/tts_local_20260517-190834-035468.json` |
| Orin Nano Paraformer+Matcha | 0.023 | 7.5 ms | 68.2 ms | `bench/perf/results/_from_orin-nano/tts_local_20260517-195122-081187.json` |
| Orin NX Paraformer+Matcha | 0.018 | 4.6 ms | 53.4 ms | `bench/perf/results/_from_orin-nx/tts_local_20260517-195126-919620.json` |
| RK3588 | 0.075 | 4.0 ms | 214.1 ms | `bench/perf/results/_from_radxa/tts_local_20260517-222507.json` |
| RK3576 | 0.166 | 5.8 ms | 483.5 ms | `bench/perf/results/_from_cat-remote/tts_local_20260518-062513.json` |
| RPi5 | 0.078 | 2.6 ms | 207.3 ms | `bench/perf/results/_from_harvest-pi/tts_local_20260518-062507.json` |

## Streaming ASR

Short Chinese corpus, warmup 1, steady 5.

| Target / profile | Wall RTF p50 | Finalize RTF p50 | TFD p50 | CER p50 | EOS→final p50 | Report |
|---|---:|---:|---:|---:|---:|---|
| Orin Nano Qwen3+Qwen3 | 1.105 | 0.076 | 505 ms | 5.3% | 272 ms | `bench/perf/results/_from_orin-nano/asr_streaming_local_20260517-192924-856732.json` |
| Orin NX Qwen3+Qwen3 | 1.069 | 0.042 | 504 ms | 5.3% | 152 ms | `bench/perf/results/_from_orin-nx/asr_streaming_local_20260517-192924-279046.json` |
| Orin Nano Qwen3 ASR + Matcha | 1.105 | 0.075 | 505 ms | 5.3% | 267 ms | `bench/perf/results/_from_orin-nano/asr_streaming_local_20260517-190727-556275.json` |
| Orin NX Qwen3 ASR + Matcha | 1.069 | 0.042 | 504 ms | 5.3% | 152 ms | `bench/perf/results/_from_orin-nx/asr_streaming_local_20260517-190856-468338.json` |
| Orin Nano Paraformer+Matcha | 1.106 | 0.077 | 2016 ms | 13.3% | 267 ms | `bench/perf/results/_from_orin-nano/asr_streaming_local_20260517-195231-865818.json` |
| Orin NX Paraformer+Matcha | 1.044 | 0.015 | 2265 ms | 10.5% | 54 ms | `bench/perf/results/_from_orin-nx/asr_streaming_local_20260517-195231-604179.json` |
| RK3588 | 1.329 | 0.301 | 1259 ms | 20.0% | 661 ms | `bench/perf/results/_from_radxa/asr_streaming_local_20260517-235806-102379.json` |
| RK3576 | 2.346 | 1.316 | 1518 ms | 20.0% | 2347 ms | `bench/perf/results/_from_cat-remote/asr_streaming_local_20260518-075830-610334.json` |
| RPi5 | 1.025 | 0.000 | 1257 ms | 20.0% | 251 ms | `bench/perf/results/_from_harvest-pi/asr_streaming_local_20260518-075800-393227.json` |

## Streaming V2V

Short Chinese corpus, warmup 1, steady 3, `llm_delay=0`.

| Target / profile | EOS→audio p50 | ASR finalize p50 | TTS TTFB p50 | TTS total p50 | Report |
|---|---:|---:|---:|---:|---|
| Orin Nano Qwen3+Qwen3 | 251 ms | 245 ms | 7 ms | 4041 ms | `bench/perf/results/_from_orin-nano/v2v_forced_llm0_local_20260517-193101-834044.json` |
| Orin NX Qwen3+Qwen3 | 157 ms | 152 ms | 5 ms | 3685 ms | `bench/perf/results/_from_orin-nx/v2v_forced_llm0_local_20260517-193053-417256.json` |
| Orin Nano Qwen3 ASR + Matcha | 286 ms | 278 ms | 8 ms | 89 ms | `bench/perf/results/_from_orin-nano/v2v_forced_llm0_local_20260517-191049-519783.json` |
| Orin NX Qwen3 ASR + Matcha | 162 ms | 152 ms | 5 ms | 64 ms | `bench/perf/results/_from_orin-nx/v2v_forced_llm0_local_20260517-191050-863680.json` |
| Orin Nano Paraformer+Matcha | 327 ms | 321 ms | 6 ms | 86 ms | `bench/perf/results/_from_orin-nano/v2v_forced_llm0_local_20260517-195442-509259.json` |
| Orin NX Paraformer+Matcha | 58 ms | 52 ms | 5 ms | 63 ms | `bench/perf/results/_from_orin-nx/v2v_forced_llm0_local_20260517-195442-062492.json` |
| RK3588 | 1293 ms | 1276 ms | 17 ms | 350 ms | `bench/perf/results/_from_radxa/v2v_forced_llm0_local_20260517-235939-539754.json` |
| RK3576 | 5920 ms | 5893 ms | 28 ms | 725 ms | `bench/perf/results/_from_cat-remote/v2v_forced_llm0_local_20260518-080000-150029.json` |
| RPi5 | 3 ms | 1 ms | 2 ms | 340 ms | `bench/perf/results/_from_harvest-pi/v2v_forced_llm0_local_20260518-075935-230620.json` |

## Deployment Footprint

Collected with `docker images`, `docker stats --no-stream`, and
`docker system df -v` on the same devices.

| Target | Image | Image size | Model / engine volume | Resident memory | Startup to ready |
|---|---|---:|---:|---:|---:|
| Orin Nano | `jetson-v1.12-highperf` | 2.02 GB | `speech-models` 5.137 GB | 2.143 GiB | 14 s |
| Orin NX | `jetson-v1.12-highperf` | 2.02 GB | `speech-models` 5.449 GB | 1.021 GiB | 13 s |
| RK3588 | `rk-v1.4-closedloop` | 767 MB | 3.306 GB ASR + 300.6 MB TTS | 4.09 GiB | 9 s |
| RK3576 | `rk-v1.4-closedloop` | 767 MB | 2.205 GB ASR + 350.6 MB TTS | 2.714 GiB | 15 s |
| RPi5 | `rpi-v1.0-onnx` | 568 MB | `speech-models` 2.192 GB | n/a from Docker stats | 9 s |

## Concurrency Smoke

`parallel=2`, `mode=asr_tts_simul`, `runs=2`. This is a small stability smoke,
not a capacity limit test.

| Target | Status | ASR RTF p50 | TTS RTF p50 | Report |
|---|---|---:|---:|---|
| Orin Nano Paraformer+Matcha | PASS | 1.124 | 1.424 | `bench/perf/results/_from_orin-nano/concurrent_asr_tts_simul_p2_local_20260517-203829-567057.json` |
| Orin NX Paraformer+Matcha | PASS | 1.053 | 1.333 | `bench/perf/results/_from_orin-nx/concurrent_asr_tts_simul_p2_local_20260517-203829-613550.json` |
| RK3588 | PASS | 1.247 | 1.729 | `bench/perf/results/_from_radxa/concurrent_asr_tts_simul_p2_local_20260518-000333-495816.json` |
| RK3576 | PASS | 1.764 | 2.511 | `bench/perf/results/_from_cat-remote/concurrent_asr_tts_simul_p2_local_20260518-080823-881958.json` |
| RPi5 | PASS | 1.100 | 0.102 | `bench/perf/results/_from_harvest-pi/concurrent_asr_tts_simul_p2_local_20260518-080818-554861.json` |

## Productization Findings

- `bench/perf/run_on_device.sh` now stages the perf harness without historical
  `results/`; otherwise repeated runs eventually push megabytes of old JSON and
  can stall before benchmark execution.
- Jetson Paraformer model readiness must check both `encoder.onnx` and
  `tokens.txt`. On NX, the engine bundle existed but `tokens.txt` was missing,
  so ASR stayed disabled until base model resources were restored.
- `concurrent` benchmark ASR workers must use the same explicit EOS mode as the
  release gate. The old default used VAD, which left Paraformer WebSocket
  sessions open and prevented p=2 JSON reports from being written. Current
  concurrent release runs use `--eos forced`.
- Nano currently loads the Paraformer decoder through TRT but the encoder falls
  back after NaN validation on the available bundle. NX loaded both encoder and
  decoder as TRT after the model resource fix.
- RK release path remains the hybrid Matcha configuration. Full RKNN Matcha is
  still not the release path.
- RK Qwen3-ASR true-streaming must set `QWEN3_ASR_CHUNK_CONFIRM=0` together
  with `QWEN3_ASR_STREAM_TRUE=1`. Otherwise the chunk-confirm default shadows
  the true-streaming branch even though the env says true-streaming is enabled.
