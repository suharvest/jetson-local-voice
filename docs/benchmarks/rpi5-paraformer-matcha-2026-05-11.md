# Raspberry Pi 5 — Paraformer streaming + Matcha TTS (CPU-only)

Date: 2026-05-11
Device: `harvest-pi` (Raspberry Pi 5 Model B Rev 1.0, ARM64, 8GB RAM, Debian 12)
Stack: Docker `jetson-voice:rpi5` (debian:12-slim + sherpa-onnx + onnxruntime CPU EP)
Backends: `ASR_BACKEND=sherpa` (Paraformer streaming) + `TTS_BACKEND=sherpa` (Matcha)
Threads: `STREAMING_ASR_NUM_THREADS=4`, `TTS_NUM_THREADS=4`
Models:
- `sherpa-onnx-streaming-paraformer-bilingual-zh-en` (encoder.onnx 607MB + decoder.onnx 218MB, fp32)
- `matcha-icefall-zh-en` + `vocos-16khz-univ.onnx` (model-steps-3.onnx 73MB + vocos 52MB)

Image: 559MB on disk / 128MB content.

## Headline numbers

| Metric | Value |
|---|---|
| **ASR RTF** (Paraformer streaming, 6.0s sine wav, 100ms chunks, n=5) | **0.179** mean, 0.177 p50 (decode 1.07s) |
| **TTS RTF** Matcha zh ("今天天气真好…", 3.09s audio, n=5) | **0.095** mean (synth 293ms) |
| **TTS RTF** Matcha en ("Hello, this is a test…", 4.06s audio, n=5) | **0.092** mean (synth 374ms) |
| **V2V** ASR EOS → TTS first audio byte (short final "昨天是 mon", n=3) | **139ms** mean (134–143ms) |
| ASR total (push 3s wav + finalize) | ~3.16s |

## TTS streaming first-byte vs total (after `generate_streaming` fix)

`/tts/stream` chunked, first non-header audio byte vs full synthesis time:

| Text length | chars | first audio | total | audio duration | RTF |
|---|---|---|---|---|---|
| short | 7 | 125ms | 125ms | 1.14s | 0.110 |
| mid | 12 | 181ms | 182ms | 2.33s | 0.078 |
| long | 44 | **148ms** | 703ms | 8.49s | 0.083 |

Long-text scaling: 8.49s of audio synthesised in 703ms (RTF 0.083); first chunk lands at 148ms — **4.7× faster than waiting for full synthesis**. Short-text first-byte is dominated by Matcha acoustic-model (`model-steps-3.onnx`) which runs once before vocoder streaming begins (~100–150ms floor on Pi 5 CPU).

## Conclusion

- Paraformer streaming + Matcha both run **comfortably real-time** on RPi 5 CPU (ASR ~5.6× real-time, TTS ~10× real-time).
- True end-to-end voice loop EOS-to-first-audio: ~139ms for short utterances, scales linearly via vocoder streaming for longer utterances.
- No GPU / NPU / quantization needed for this workload.

## Reproduction

```bash
# Build image (on RPi)
cd ~/jetson-voice-rpi && docker build -f Dockerfile.rpi -t jetson-voice:rpi5 .

# Run
docker run -d --name jvrpi -p 8621:8000 \
  -v ~/jetson-voice-rpi/models:/opt/models \
  jetson-voice:rpi5

# Health
curl http://localhost:8621/health

# Bench
docker exec jvrpi python3 /opt/speech/bench/rpi_bench.py --mode asr
docker exec jvrpi python3 /opt/speech/bench/rpi_bench.py --mode tts
docker exec jvrpi python3 /opt/speech/bench/rpi_bench.py --mode v2v
```

Files added: `Dockerfile.rpi`, `bench/rpi_bench.py`.
Code change: `app/backends/sherpa.py:198` — `generate_streaming()` moved `OfflineTts.generate()` to background thread for true per-chunk streaming + fixed `sid/speed/pitch=None` coalescing bug.

## Gotchas

1. **Docker Hub unreachable from `harvest-pi`** — added `https://docker.m.daocloud.io` registry mirror to `/etc/docker/daemon.json` (persistent host change).
2. **`tar tjf | head` over SSH truncates output** — looked like 999MB Paraformer tarball was corrupt with only 4 entries; actual file is fine, extract works. Use `tar tjf > file && wc -l file` instead.
3. Sherpa Paraformer model expects `encoder.onnx`/`decoder.onnx` filenames; release tarball ships `encoder.int8.onnx`/`encoder.onnx` both — symlink rather than `mv` so int8 variant remains available.
4. Matcha `vocos-16khz-univ.onnx` is the file referenced in `app/backends/sherpa.py:124`; download separately from `sherpa-onnx/releases/download/vocoder-models/`.
5. Pre-existing `sherpa.py` bug: caller passing `speaker_id=None` to `generate_streaming` raised TypeError; fixed in this session.
