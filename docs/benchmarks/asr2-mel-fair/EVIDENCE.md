# EVIDENCE Report — ASR Mel Norm Fair Comparison

## 1. Patch Diff (_compute_mel normalization change)

Before (min-max norm):
```python
        # Whisper normalization: mel = mel - mel.max(), then scale to [0, 1]
        log_mel = log_mel - log_mel.max()
        mel_range = log_mel.max() - log_mel.min()
        if mel_range > 1e-5:
            log_mel = (log_mel - log_mel.min()) / mel_range
```

After (Whisper-style clip+scale):
```python
        # Whisper-style: clip then normalize to N(0,1) via log-mel clipping
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
```

## 2. Raw Results Summary

### Phase A — Baseline (min-max norm)

| Run  | Mean Inference | Mean CER | Mean Wall | Mean PerToken |
|------|---------------|----------|-----------|--------------|
| Cold | 730.5 ms      | 14.85%   | 751.1 ms  | 53.4 ms      |
| Warm1| 417.0 ms      | 14.85%   | 436.6 ms  | 27.8 ms      |
| Warm2| 418.5 ms      | 14.85%   | 437.2 ms  | 27.8 ms      |

### Phase B — Whisper Norm

| Run  | Mean Inference | Mean CER | Mean Wall | Mean PerToken |
|------|---------------|----------|-----------|--------------|
| Cold | 713.6 ms      | 16.33%   | 732.3 ms  | 50.5 ms      |
| Warm1| 424.2 ms      | 16.33%   | 443.1 ms  | 28.1 ms      |
| Warm2| 422.9 ms      | 16.33%   | 441.7 ms  | 28.0 ms      |

### Warm Average Comparison

| Metric         | A (avg)   | B (avg)   | Ratio/Delta |
|---------------|-----------|-----------|-------------|
| Inference     | 417.75 ms | 423.55 ms | 1.014       |
| Wall          | 436.9 ms  | 442.4 ms  | 1.013       |
| PerToken      | 27.8 ms   | 28.1 ms   | 1.009       |
| CER           | 14.85%    | 16.33%    | +1.48%      |

## 3. Acceptance Criteria

| Check | Threshold | Actual | Result |
|-------|-----------|--------|--------|
| Speed | B_warm/A_warm ≤ 1.10 | 1.014 | ✅ PASS |
| CER   | B_cer - A_cer ≤ +2.00% | +1.48% | ✅ PASS |

## 4. Docker Image

- Committed: `jetson-voice-speech:v3.2-librosa` (ID 633563265417, 606MB content)
- Backup of old code retained at `qwen3_asr.py.preB`

## 5. Result Files

```
/Users/harvest/project/jetson-voice/docs/benchmarks/asr2-mel-fair/
├── A_cold.md
├── A_warm1.md
├── A_warm2.md
├── B_cold.md
├── B_warm1.md
├── B_warm2.md
├── COMPARISON.md
└── EVIDENCE.md
```
