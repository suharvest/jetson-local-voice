# ASR Mel Norm Fair Comparison — Whisper Norm (B) vs Min-Max Norm (A)

Date: 2026-04-26
Device: orin-nx (Jetson AGX Orin)
Container: reachy_speech-speech-1 (both phases restarted cold)
Test: 8 WAV files via eval_asr.py, 2 warm runs each after restart + warmup

## Overview

| Metric                | Baseline (A) | Whisper (B) | Ratio/Delta |
|-----------------------|-------------|-------------|-------------|
| Mean Inference (cold) | 730.5 ms    | 713.6 ms    | 0.977       |
| Mean Inference (warm avg) | 417.75 ms | 423.55 ms | 1.014     |
| Mean Wall (warm avg)  | 436.9 ms    | 442.4 ms    | 1.013       |
| Mean PerToken (warm avg) | 27.8 ms  | 28.1 ms     | 1.009       |
| Mean CER (warm avg)   | 14.85%      | 16.33%      | +1.48%      |

## Per-Wav Inference Time (warm avg)

| ID  | Dur(s) | A (ms) | B (ms) | B/A   | A CER% | B CER% |
|-----|--------|--------|--------|-------|--------|--------|
| S0  | 2.32   | 255.5  | 261.0  | 1.022 | 0.00   | 0.00   |
| S1  | 2.80   | 331.5  | 334.0  | 1.008 | 0.00   | 0.00   |
| S2  | 6.40   | 413.5  | 412.0  | 0.996 | 33.33  | 33.33  |
| S3  | 9.76   | 605.0  | 634.5  | 1.049 | 33.93  | 35.71  |
| S4  | 1.60   | 205.5  | 204.0  | 0.993 | 0.00   | 10.00  |
| S5  | 3.04   | 296.5  | 297.0  | 1.002 | 5.26   | 5.26   |
| S6  | 7.44   | 563.0  | 564.5  | 1.003 | 13.71  | 13.71  |
| S7  | 10.16  | 671.5  | 681.5  | 1.015 | 32.58  | 32.58  |

## Verdict

- **Speed check**: B_warm/A_warm = 1.014 (≤1.10: ✅ PASS)
- **CER check**: B_cer - A_cer = +1.48% (≤+2.00%: ✅ PASS)
- **Overall**: ✅ PASS — committed as `jetson-voice-speech:v3.2-librosa`

## Notable Observations

- S4 (Hello): CER 0%→10% — Whisper norm changes output from "Hello, nice to meet you." to "Hello. Nice to meet you." (punctuation/cap change, not recognition failure)
- S3 (long Chinese): CER 33.93%→35.71% — slight regression on long Chinese audio
- Speed impact is negligible (~1.4%), well within noise floor
