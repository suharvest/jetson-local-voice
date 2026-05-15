# Open Dialogue VAD Follow-ups — 2026-05-15

Scope: open-mic dialogue benchmarking and default endpointing for Jetson/RK
speech services.

## Current Baseline

- Default service VAD backend is `silero`.
- Benchmark default EOS mode is `vad`, not `eou`.
- `eou` and `forced` are diagnostic lower-bound modes only.
- `silero_vad.onnx` is present in the repo at `app/core/assets/silero_vad.onnx`.
- Current running containers on `orin-nx`, `orin-nano`, `cat-remote`, and `radxa`
  were hot-patched with the Silero ONNX asset and restarted successfully.

## Smoke Results

- `orin-nx` ASR default `vad+silero`: final received.
- `cat-remote` RK3576 ASR default `vad+silero`: final received.
- `radxa` RK3588 ASR default `vad+silero`: final received.
- `orin-nx` V2V default `vad+silero`: `EOS->Audio ~= 772 ms`.
- `cat-remote` RK3576 V2V default `vad+silero`: `EOS->Audio ~= 1670 ms`.
- `radxa` RK3588 V2V default `vad+silero`: `EOS->Audio ~= 963 ms`.

## Work Items For Another Agent

1. Run full open-dialogue benchmark on NX/RK
   - Use default `--eos vad --vad-backend silero --vad-silence-ms 400`.
   - Run ASR and V2V with `--warmup 3 --runs 10`.
   - For RK, run inside the service container, not host Python.
   - Save and copy results back from container paths.

2. Add a true `/v2v/stream` benchmark
   - Current `bench/perf/perf.py v2v` is a composite lower-level harness:
     ASR websocket finalize, placeholder LLM delay, then TTS request.
   - Add a benchmark that drives the actual `/v2v/stream` protocol with config,
     binary mic chunks, natural silence, LLM text frames, and TTS flush.
   - Report endpoint latency, ASR final latency, first audio latency, and
     total turn latency.

3. Verify Silero full-corpus endpoint quality
   - Compare `silero` vs `webrtcvad` on short/long zh/en corpus.
   - Track false early cut, no-final timeout, CER/WER regression, and latency.
   - Keep `webrtcvad` as fallback only if Silero has device-specific issues.

4. Investigate RK ASR finalize tail latency
   - RK3576 showed much higher V2V latency than RK3588.
   - Prior forced-EOS full run had RK3576 V2V p50 around 2s and long-tail up to
     9s; RK3588 was closer to sub-second p50.
   - Focus on RK ASR streaming/finalize path before TTS.

5. Rebuild images to make Silero asset permanent
   - Dockerfiles already copy `app/`, so new images should include
     `app/core/assets/silero_vad.onnx`.
   - Rebuild/push Jetson and RK images, then verify the model exists at the
     runtime import path printed by `app.core.vad._SILERO_ONNX_PATH`.

6. Add timeout handling for VAD benchmarks
   - If VAD does not emit final after the configured silence tail, record a
     structured timeout row instead of letting a full sweep hang.
   - This is important for unattended A/B runs.
