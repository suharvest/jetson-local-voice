---
license: cc-by-4.0
language:
  - zh
  - en
size_categories:
  - n<1K
task_categories:
  - automatic-speech-recognition
  - text-to-speech
tags:
  - benchmark
  - perf
  - latency
  - rtf
  - jetson
  - edge
pretty_name: OpenVoiceStream Perf Corpus
---

# OpenVoiceStream — Perf Test Corpus

Fixed 20-file audio corpus used to benchmark
[`Seeed-Projects/openvoicestream`](https://github.com/Seeed-Projects/openvoicestream)
across Jetson, Rockchip, and Raspberry Pi deployments.

The same `.wav` bytes are pulled by every device, so RTF / latency deltas
between devices are pure compute — not input variation.

## Contents

- 5× zh short (1.5 – 4.0 s)
- 5× zh long (10 – 16 s)
- 5× en short (1.5 – 4.0 s)
- 5× en long (10 – 12 s)

Audio spec: **16 kHz mono 16-bit WAV**.

Each file's SHA256 + transcript is recorded in `manifest.json`. Fetchers
should verify SHA256 on every pull.

## Source

All clips are sampled from [**Google FLEURS**](https://huggingface.co/datasets/google/fleurs)
(Conneau et al. 2022, *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*),
licensed CC BY 4.0. We resampled / re-bundled a tiny subset; no audio
content was altered.

```bibtex
@article{conneau2022fleurs,
  title={FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  author={Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  year={2022}
}
```

## Usage

```bash
# From the openvoicestream repo
python bench/perf/corpus/fetch.py --from hf
python bench/perf/corpus/fetch.py --verify    # SHA256 must match manifest

python bench/perf/perf.py matrix --base-url http://<device>:8621
```

## License

CC BY 4.0 (inherited from FLEURS). Use freely with attribution.
