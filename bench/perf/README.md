# OpenVoiceStream — Perf Test Harness

Cross-device, reproducible performance & latency benchmark.

## Scenarios

| Cmd | What it measures | Output |
|---|---|---|
| `perf.py asr`        | ASR RTF + TFD + WER/CER (vs manifest transcript) | per `(category,lang)` |
| `perf.py tts`        | TTS RTF + TFD (first audio chunk latency) + total | per `(category,lang)` |
| `perf.py v2v`        | End-to-end EOS → first TTS audio (LLM = `--llm-delay ms`) | per `(category,lang)` × eos mode |
| `perf.py concurrent` | N-way parallel (`asr_only` / `tts_only` / `asr_tts_simul`) | per `kind` (asr vs tts) |
| `perf.py clone`      | Voice clone embed + synth + speaker similarity (resemblyzer) | per `lang` |
| `perf.py noise`      | ASR @ SNR 20/10/5/0 dB (white / pink / babble) → WER 退化曲线 | per `(snr_db,category,lang)` |
| `perf.py stability`  | 30 min 持续负载 → RTF drift + RSS growth | drift report |
| `perf.py boot`       | docker restart → /health 200 的冷启动时间 | per-run |
| `perf.py matrix`     | Full sweep: all of the above with sensible defaults | one file per scenario |

All scenarios:
- Discard `--warmup` runs (default 3), record `--runs` runs (default 10)
- Sample container memory peak if `--container <name>` given
- Write `results/<scenario>_<timestamp>.{json,md}`

## Setup

```bash
# 1. Populate the fixed corpus (one-time per workspace)
cd bench/perf/corpus
python fetch.py --from ~/bench/wavs          # or --from cdn  once Seeed CDN bundle exists
python fetch.py --verify                     # checks SHA256 against manifest.json
#  OR — bootstrap by synthesizing from our own TTS on one stable device:
# python synthesize_from_tts.py --base-url http://localhost:8621
# python fetch.py --recompute-hashes && git add -p manifest.json

# 2. Install client deps (one-time)
pip install websocket-client requests numpy
```

## Quick start

```bash
# ASR streaming, 10 runs after 3 warmups
python bench/perf/perf.py asr --base-url http://localhost:8621

# TTS, short prompts only
python bench/perf/perf.py tts --base-url http://localhost:8621 --category short

# V2V net latency (no LLM)
python bench/perf/perf.py v2v --base-url http://localhost:8621 --llm-delay 0

# V2V with 800ms LLM placeholder + VAD-driven EOS (realistic)
python bench/perf/perf.py v2v --base-url http://localhost:8621 --llm-delay 800 --eos vad

# 2-way concurrent ASR+TTS simultaneous interpretation
python bench/perf/perf.py concurrent --parallel 2 --mode asr_tts_simul \
  --base-url http://localhost:8621

# Full matrix (10-15 min on Jetson; longer on RPi)
python bench/perf/perf.py matrix --base-url http://localhost:8621 \
  --container seeed-nano-v111

# === Tier-1 (customer eval) ===

# ASR with WER/CER per file (needs `pip install jiwer`)
python bench/perf/perf.py asr --runs 10 --warmup 3

# Voice clone (only meaningful with voice_clone preset; needs `pip install resemblyzer`)
# Re-use the corpus long-form clips as reference voices
python bench/perf/perf.py clone --refs bench/perf/corpus/long \
  --texts bench/perf/corpus/clone_texts.json --runs 5

# Cold-start time (restart container, poll /health)
python bench/perf/perf.py boot --container seeed-nano-v111 --runs 3

# === Tier-2 (differentiation) ===

# Noise robustness: WER at SNR 20/10/5/0 dB babble (default)
python bench/perf/perf.py noise --snr 20 10 5 0 --noise-type babble

# 30-min stability (v2v mode)
python bench/perf/perf.py stability --duration-min 30 --container seeed-nano-v111
```

Default benchmark endpointing matches open-dialogue deployment:
`--eos vad --vad-backend silero --vad-silence-ms 400`. Tune the silence
threshold per run with `--vad-silence-ms`, or in service deployment with
`OVS_VAD_SILENCE_MS`.

## Two test modes (important!)

The same metrics carry **very different meaning** depending on where the
perf client runs:

| Mode | When to use | Setup | Result tag |
|---|---|---|---|
| **local** (on-device) | "How fast is *this device* at compute?" Fair cross-device comparison. | client + service on the same box, talks to `localhost:8621` | `*_local_<timestamp>.{json,md}` |
| **remote** (over network) | "What latency does the user actually feel?" Real product SLA. | client elsewhere (your laptop, the robot brain, etc.) → device:8621 | `*_remote_<timestamp>.{json,md}` |

Mode is auto-inferred from `--base-url` (loopback = local, anything else =
remote) and stamped into result files + meta. Force with `--mode-label`.

### Local-mode one-liner (recommended for device benchmarks)

```bash
# Pushes bench/perf/ to device, fetches corpus, runs perf with loopback client,
# pulls results back to results/_from_<node>/
bench/perf/run_on_device.sh orin-nano -- asr --warmup 2 --runs 10
bench/perf/run_on_device.sh radxa     -- v2v --llm-delay 0
bench/perf/run_on_device.sh orin-nano -- matrix
```

### Remote-mode (current Mac→device path)

```bash
python bench/perf/perf.py asr --base-url http://orin-nano.local:8621
```

Remote-mode numbers will include client↔device RTT in every TFD / EOS metric.
Don't mix local-mode and remote-mode numbers in the same cross-device table.

## Per-device launch examples

```bash
# Jetson Orin Nano (voice_clone preset on :8621)
python bench/perf/perf.py matrix --base-url http://orin-nano.local:8621 \
  --container seeed-nano-v111

# Radxa ROCK 5T (RK3588, multilang preset)
python bench/perf/perf.py matrix --base-url http://radxa.local:8621 \
  --container seeed-rk-v11

# Raspberry Pi 5 (lite_zh_en preset)
python bench/perf/perf.py matrix --base-url http://harvest-pi.local:8621 \
  --container seeed-rpi-litezhen
```

## Metric definitions

| Symbol | What |
|---|---|
| **Wall RTF** | `processing_ms / audio_dur_ms`. In `--realtime` streaming this is **≥ 1.0 by construction** (client paces chunks). Use this only to confirm "can keep up with realtime". |
| **Finalize RTF** | `eos_to_final_ms / audio_dur_ms`. Compute-bound; **the right number for cross-device comparison**. Lower is better. |
| **TFD** (Time to First Decoded/Audio) | First user-visible byte. For ASR streaming = first partial after first PCM chunk sent. For TTS = first audio chunk after request. |
| **EOS → first audio** (V2V) | The only number the end user feels. = ASR finalize + LLM delay + TTS TFD. |
| **VAD EOS** vs **Forced EOS** | Forced = client sends `b""` immediately after last PCM. VAD = server waits for silence; adds ~300-700ms. Forced is for comparing pipelines; VAD is the real product SLA. |
| **Concurrent wall** | Total wall-clock for the whole scenario (start of first worker to last); useful for throughput. |

## Files

```
bench/perf/
├── perf.py                CLI dispatcher
├── client.py              ASR/TTS clients + V2V composite
├── runners.py             4 runner functions
├── stats.py               p50/p95/mean + markdown render
├── memory.py              docker stats sampler
├── corpus/
│   ├── manifest.json      20-file contract (transcripts + SHA256)
│   ├── tts_prompts.json   20 TTS prompts (committed)
│   ├── fetch.py           populate from local dir | CDN | verify SHA256
│   ├── synthesize_from_tts.py  bootstrap via live TTS service
│   ├── SOURCES.md         how to populate WAVs the first time
│   ├── short/             *.wav (gitignored — 16 kHz mono 16-bit)
│   └── long/              *.wav (gitignored)
└── results/               per-run output (gitignored)
```

## Filling the perf table

For the cross-device comparison table in `docs/perf-test-runbook.md`, the
key numbers to pick out of each scenario file:

- **ASR**: `summary["short/zh"]["rtf"]["p50"]`, `summary["long/zh"]["rtf"]["p50"]`
- **TTS**: same shape
- **V2V**: `summary["short/zh"]["eos_to_first_audio_ms"]["p50"]` with `llm-delay=0`
- **Concurrent**: degradation = `summary[k]["rtf"]["p50"]` at parallel=2 vs parallel=1
- **Memory**: `memory.peak_mib`

Each device runs the *exact same* corpus bytes (SHA256-verified), so RTF
deltas are pure compute, not input variation.
