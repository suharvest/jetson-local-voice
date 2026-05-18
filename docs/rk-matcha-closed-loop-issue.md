# RK Matcha Closed-Loop Issue

Date: 2026-05-17

## Summary

RK Qwen3-ASR itself can run, but the current RK Matcha/Vocos TTS artifact path
does not produce audio that passes TTS-to-ASR closed-loop validation.

The release-safe RK3588 NPU-assisted path is:

- TTS: `rk.tts` adapter with `TTS_BACKEND=matcha_rknn`,
  `MATCHA_USE_ORT=1` (Matcha acoustic on ORT CPU) and RKNN Vocos on NPU
- ASR: `rk:qwen3_asr_rk`

The full RKNN Matcha/Vocos path (`MATCHA_USE_ORT=0`, RK3588
`MATCHA_MODEL_SEQ_LEN=96`, `MATCHA_MODEL_FRAMES=256`) is faster, but should not
be treated as closed-loop release-ready until the audio quality is fixed and
revalidated.

## Reproduction

On RK3588, generate TTS audio from:

```text
你好，今天天气真不错。
```

Then feed the generated WAV back to ASR.

Observed with RK Matcha/Vocos TTS:

| ASR language | Result |
|---|---|
| `auto` | empty text |
| `zh` | `You are a helpful assistant` |
| `Chinese` | `You are a helpful assistant` |
| `en` | `You are a helpful assistant` |

Cross-checking the same RK-generated TTS WAV against non-RK ASR also failed or
returned unusable output, so this is not just an RK ASR HTTP/prompt issue.

## Key Findings

The current `matcha-s64.rknn` artifact shape does not match the old profile
assumptions:

- Old profile assumed `MATCHA_MODEL_SEQ_LEN=80`.
- Runtime probing showed token input expects sequence length `96`.
- `length_scale` is not accepted as scalar input for this artifact.
- The full RKNN model accepts `length_scale` as `[1, 80, 256]`.
- Full RKNN Matcha output is `[1, 80, 256]`.

After adapting the full RKNN input shape, full RKNN Matcha can run and is fast
(about RTF `0.075` in the quick test), but closed-loop recognition still only
returned partial/empty text, e.g. `你`.

This points to an audio quality / vocoder / export compatibility problem in the
current RKNN Matcha/Vocos path, not just a shape or API bug.

## Release Configuration

Use the `rkvoice-stream` Matcha backend in its validated hybrid mode while
keeping RK Qwen3-ASR:

```text
tts_backend = rk.tts
asr_backend = rk.asr
TTS_BACKEND=matcha_rknn
MATCHA_USE_ORT=1
MATCHA_MODEL_SEQ_LEN=80
VOCOS_MODEL=vocos-16khz-600.rknn
VOCOS_FRAMES=256
```

Validated RK3588 closed-loop result:

```json
{
  "expected": "你好，今天天气真不错。",
  "asr_text": "你好，今天天气真不错",
  "similarity": 1.0,
  "tts_backend": "rk:matcha_rknn",
  "asr_backend": "rk:qwen3_asr_rk"
}
```

Product eval snapshot:

```text
Target: RK3588
TTS backend: rk:matcha_rknn (MATCHA_USE_ORT=1, RKNN Vocos)
ASR backend: rk:qwen3_asr_rk
TTS short zh WAV size: 77356 bytes for standard phrase
ASR short zh error p50: 30.8%
TTS-to-ASR round-trip: PASS
```

## What Other Agents Should Verify

1. Reproduce the failure using the RKNN Matcha/Vocos path:
   - `TTS_BACKEND=matcha_rknn`
   - `MATCHA_USE_ORT=0`, `MATCHA_MODEL_SEQ_LEN=96`,
     `MATCHA_MODEL_FRAMES=256`, `VOCOS_FRAMES=256` for full RKNN.
2. Confirm the validated hybrid path still passes:
   - `MATCHA_USE_ORT=1`, `MATCHA_MODEL_SEQ_LEN=80`, `VOCOS_FRAMES=256`.
3. Listen to or inspect generated audio from both RKNN paths.
4. Compare mel/audio against sherpa Matcha reference output for the same text.
5. Check whether `vocos-16khz-600.rknn` is compatible with the 256-frame
   Matcha output path or if a matching vocoder export is needed.
6. Re-export Matcha/Vocos RKNN artifacts with explicit metadata:
   - token sequence length
   - mel frame count
   - scalar/broadcast input shapes
   - vocoder frame count
7. Only switch RK release profile back to full RKNN TTS after TTS-to-ASR round-trip
   passes with similarity at least `0.65` on the standard phrase and product
   eval remains green.
