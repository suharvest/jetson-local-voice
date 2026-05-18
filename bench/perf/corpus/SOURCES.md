# Perf corpus sources

Cross-device perf testing requires **bit-identical** WAV inputs. The 20-file
corpus is described in `manifest.json` (5 zh-short + 5 zh-long + 5 en-short +
5 en-long, 16 kHz mono 16-bit). The WAV files themselves are **not** in git
(license + size); fetch them with one of the methods below, then commit the
SHA256 fingerprints back to `manifest.json` so future fetches verify integrity.

## Recommended source

Bundle the 20 WAVs as `models-perf-corpus.tar.gz` on Seeed CDN:

  https://sensecraft-statics.seeed.cc/solution-app/jetson-voice/models-perf-corpus.tar.gz

The upload is a one-time step (deferred): record on a clean Jetson and upload
through the same path as the matcha / paraformer tarballs.

## Recommended: curate from Google FLEURS (CC BY 4.0)

`curate_public_corpus.py` streams from HuggingFace (no 19 GB tarball pull):

```bash
pip install datasets soundfile numpy
python bench/perf/corpus/curate_public_corpus.py
# overwrites manifest.json transcripts/durations + writes 20 WAVs
python bench/perf/corpus/fetch.py --recompute-hashes
git add bench/perf/corpus/manifest.json
tar czf models-perf-corpus.tar.gz -C bench/perf/corpus short long
# upload to Seeed CDN as solution-app/jetson-voice/models-perf-corpus.tar.gz
```

Other devices then `python fetch.py --from cdn` to pull bit-identical bytes.

Credit: Conneau et al. 2022, "FLEURS" — CC BY 4.0.

## Building the corpus yourself (alternatives)

Pick whichever path is most convenient — the key invariant is **once the 20
WAVs exist, every device fetches the same bytes** (`fetch.py` verifies SHA256
against `manifest.json`).

### Option A — Synthesize with our own TTS (deterministic, no licensing)

```bash
# On a Jetson with the voice_clone preset running on :8621
python bench/perf/corpus/synthesize_from_tts.py \
  --base-url http://localhost:8621 \
  --voice default \
  --out-dir bench/perf/corpus
```

Pros: zero licensing risk, exact reproducibility. Cons: synthesized speech is
clean (no noise), ASR numbers may be optimistic vs real mic input.

### Option B — Record on a Mac (more realistic)

Read each `transcript` aloud, save as 16 kHz mono WAV using sox:

```bash
sox -d -r 16000 -c 1 -b 16 short/zh_short_01.wav
```

### Option C — Pull from a public dataset

- AISHELL-3 (Apache 2.0): https://www.openslr.org/93/ — 19 GB, pick 10 utterances
- LibriSpeech dev-clean (CC BY 4.0): https://www.openslr.org/12/ — 337 MB, pick 10
- Common Voice (CC0): https://commonvoice.mozilla.org/ — needs Mozilla account

Re-encode to 16 kHz mono 16-bit and rename to the manifest IDs.

## After populating

```bash
python bench/perf/corpus/fetch.py --recompute-hashes
git diff bench/perf/corpus/manifest.json   # commit the new sha256 values
```

`fetch.py --verify` will then fail loudly on any device whose WAVs drift from
the committed fingerprints.
