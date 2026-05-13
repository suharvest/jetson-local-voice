# Performance Comparison — Choosing a Device

Seeed Local Voice runs the same HTTP/WS API across Jetson, Rockchip, and
Raspberry Pi. This page shows **measured numbers** for each platform so
you can pick the device that fits your product, not your spec sheet.

All numbers are **local-mode** (client runs on the device, talks to
`localhost:8000`), so they reflect what your application code will see —
no network noise.

---

## Quick pick

| Your priority | Pick this device |
|---|---|
| Highest accuracy on English + multilingual | **Jetson Orin Nano** |
| Best Chinese accuracy, mid-tier latency | **Radxa ROCK 5T (RK3588)** |
| Lowest cost, decent zh+en for voice commands | **Raspberry Pi 5** |
| Multilingual on a tight budget (ja/ko/es/de/fr) | **RK3576** (ASR only — TTS limitation, see below) |
| Voice cloning (your own voice synthesised) | **Jetson Orin Nano** (the only platform that supports it today) |

---

## What we measure (and what it means for your users)

### Speech recognition (ASR) quality

| Metric | What it is | Why your user cares |
|---|---|---|
| **CER** (character error rate, zh) | % of Chinese characters mis-recognised | Lower = fewer "what did you mean?" prompts |
| **WER** (word error rate, en) | % of English words mis-recognised | Same — drives the "is this assistant smart?" feeling |
| **Finalize RTF** | How long the device takes to finish recognising AFTER you stop speaking, divided by the audio length | Lower = the assistant feels snappier. Anything under 0.5 is sub-second on a typical sentence |

### Speech synthesis (TTS) speed

| Metric | What it is | Why your user cares |
|---|---|---|
| **TTS RTF** | How long synthesis takes vs the length of audio produced. RTF=0.1 means 1s of audio takes 100ms to generate | Lower = the assistant doesn't pause before speaking. Under 0.3 is conversation-grade |
| **TTS TFD** (time to first byte) | Delay until the first audio chunk is ready | Drives the feel of "instant reply" vs "thinking..." |

### End-to-end conversation latency

| Metric | What it is | Why your user cares |
|---|---|---|
| **V2V EOS → first audio** | From when the user stops speaking, to when the assistant's first audio comes out (excludes the LLM you put in the middle) | Below 1 second feels conversational; above 2 seconds feels like waiting on hold |

### Capacity under load

| Metric | What it is | Why your user cares |
|---|---|---|
| **Concurrent RTF (parallel=2)** | RTF measured while 2 streams run at once | Tells you if the device can handle 2 simultaneous users, or if 1 is the ceiling |

---

## Measured numbers (2026-05-13)

### Speech recognition

| Group | **Orin Nano** (voice_clone) | **RK3588** (multilang) | **RK3576** (multilang) | **RPi5** (lite_zh_en) |
|---|---:|---:|---:|---:|
| Short Chinese CER | 5.3 % | **2.6 %** | 5.3 % | 10.5 % |
| Short English WER | **0.0 %** | 10.0 % | 13.1 % | 35.7 % |
| Long Chinese CER¹  | 8.4 % | 10.8 % | **7.8 %** | 14.5 % |
| Long English WER  | **3.0 %** | 5.5 % | 5.5 % | 23.6 % |
| Short Finalize RTF | 0.08 | 0.22 | 0.40 | **0.00** |
| Long Finalize RTF  | 0.06 | **0.03** | 0.54 | **0.00** |

¹ With Chinese-number normalisation (Arabic ↔ 一二三). Without
normalisation, the raw CER on `long/zh` looks 3-4× worse purely from
"15" vs "十五" mismatches — see methodology.

### Speech synthesis

| Group | Orin Nano (Qwen3 voice_clone) | RK3588 (matcha_rknn) | RK3576 (multilang) | RPi5 (sherpa matcha) |
|---|---:|---:|---:|---:|
| Short Chinese TTS RTF | 0.42 | **0.07** | n/a² | **0.08** |
| Long Chinese TTS RTF  | 0.41 | 0.14 | n/a² | **0.08** |
| Short English TTS RTF | 0.42 | 0.09 | n/a² | 0.11 |
| First-byte delay (any) | 4 ms | 4 ms | n/a² | 2 ms |

² **RK3576 has a known TTS limitation today** — neither in-tree backend
ships a working setup: the lightweight `matcha_rknn` model crashes when
co-loaded with the RKLLM ASR decoder (NPU resource conflict that we're
tracking), and the heavyweight `qwen3_rknn` runs at ~1.8 fps which is
not usable. Workaround in development: ARM-CPU sherpa-onnx Matcha as the
RK3576 TTS path. RK3576 ASR is unaffected and works well.

### End-to-end conversation latency (LLM excluded)

V2V means: user says something, the device transcribes it, then speaks
back. The number below is purely the speech part of that round-trip —
your LLM adds however long it takes to respond on top.

| Group | Orin Nano | RPi5 |
|---|---:|---:|
| Short Chinese | 325 ms | **5 ms**³ |
| Long Chinese  | 909 ms | **4 ms**³ |
| Short English | 277 ms | **3 ms**³ |
| Long English  | 810 ms | **4 ms**³ |

³ RPi5 looks unbelievably fast because sherpa-onnx is *fully streaming* —
the transcription is already done by the time the user stops speaking,
so finalize is essentially free. Nano's Qwen3 model is more accurate but
processes audio in one batch at the end, paying ~300-900 ms there.

### Concurrent load (parallel=2, ASR+TTS simultaneous)

| Device | ASR RTF | TTS RTF | Verdict |
|---|---:|---:|---|
| Orin Nano | 1.10 | 1.23 | GPU saturates at 2 concurrent users — still works, but slows down |
| RPi5      | 1.03 | **0.12** | Loads of headroom — sherpa-onnx on RPi5 can handle 4+ concurrent streams |

---

## Choosing a device — by use case

### "Speech-controlled appliance / robot, 1 user, mostly commands"

→ **Raspberry Pi 5** with `lite_zh_en` preset. Costs ~$80, 1-2 s ASR
latency, decent zh+en for command vocabulary. WER 36 % on free-form
English is noticeable but fine for fixed command sets.

### "Conversational AI in a kiosk, English-heavy, you want it to sound smart"

→ **Jetson Orin Nano** with `voice_clone` preset. ~$250, 0 % WER on
short English, supports voice cloning so the kiosk has its own voice.
Finalize latency ~300-900 ms is fine for a "press to talk" flow.

### "Multilingual customer service (Chinese + Japanese + Spanish + ...)"

→ **Radxa ROCK 5T (RK3588)** with `multilang` preset. ~$200, best CER
on Chinese short sentences (2.6 %), supports 50+ languages via Qwen3.
TTS is on par with RPi5 (Matcha 0.07 RTF). Mature, well-tested.

### "Tightest BOM, multilingual ASR only (push-to-talk transcription)"

→ **RK3576**. ~$80, multilingual Qwen3 ASR works well (5.3 % short
Chinese CER, 7.8 % long Chinese CER). **Plan around the current TTS
limitation** if you need synthesis — see footnote 2.

### "I need voice cloning"

→ **Jetson Orin Nano**. Today, this is the only device in the matrix
with a working voice clone path. The trade-off: slower TTS (RTF 0.42
vs 0.07-0.08 on RK / RPi), so your speakers wait ~half the audio
duration before they hear anything.

---

## Methodology

- **Corpus**: 20 short + long Chinese & English clips from Google
  FLEURS (CC BY 4.0), plus 10 multilingual smoke clips (ja/ko/es/de/fr).
  Same bytes on every device (SHA-256 locked). Hosted at
  [huggingface.co/datasets/harvestsu/seeed-local-voice-perf-corpus](https://huggingface.co/datasets/harvestsu/seeed-local-voice-perf-corpus).
- **Mode**: client runs on the device against `127.0.0.1:8000` (loopback).
  Eliminates network from the measurement.
- **Sample size**: warmup 5 + 10 steady runs per group. p50 reported.
- **Reproducible**: `bench/perf/run_on_device.sh <node> -- <scenario>`
  reruns the whole thing on any device. CER/WER computed with
  `jiwer` + `cn2an` (Chinese number normalisation).

Full methodology notes, raw run data, and the internal runbook live in
[`docs/perf-test-runbook.md`](perf-test-runbook.md).

---

## What we do NOT measure here

- **TTS naturalness / voice clone similarity** — those need subjective
  ratings (MOS) which are expensive to do properly. Voice clone embed
  + similarity scoring is wired into the harness (`perf.py clone`) and
  reported separately on request.
- **Noise robustness** — `bench/perf/perf.py noise` runs the corpus
  with synthetic babble at SNR 20/10/5/0 dB. Numbers TBD per device;
  reach out if you have a specific deployment-noise profile in mind.
- **30-min stability / thermal drift** — wired (`perf.py stability`),
  not run yet across all devices.
- **Power draw** — out of scope today; ask if you need it for battery
  / fanless designs.

---

## Honest caveats

1. **English mixed-script proper nouns** (e.g. "Oravec" in a Chinese
   sentence) are mis-recognised by every Qwen3 variant in our test, in
   the same way. This is a model-level limitation, not a device
   problem; it'll bias the English WER upward on real product copy
   that mixes Chinese names with English brand names.
2. **The "raw CER" you see in our internal logs is higher** than the
   numbers above because our normalisation maps "15" to "十五" before
   comparing. We picked normalisation because it matches what a user
   would judge correct ("the model said the number, just in a
   different form"). Both numbers are recorded in the raw JSON if you
   need the unforgiving version.
3. **RPi5's 0 ms finalize is a methodology artefact** of forced-EOS
   mode (we tell the server "the audio is done now"). Real users
   tend to trail off and rely on the server's VAD to detect that;
   add ~300-500 ms of VAD hangover for product latency budgets.
