## Goal

Replace the runtime dependency on `transformers.WhisperFeatureExtractor` inside `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:1121` with a local `librosa` + `numpy` Whisper log-mel implementation, while preserving the exact tensor contract currently consumed by the Qwen3 ASR encoder:

- Input: mono float audio already resampled to 16 kHz by `_bytes_to_float`.
- Current chunk selection: `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:1124-1125`, `chunk_len = min(30, int(audio_secs) + 1)`. Do not change this logic.
- Output: `np.ndarray` with shape `[1, 128, T]`, where `T = chunk_len * 16000 // 160`; for example 1 s -> 100 frames, 3 s -> 300 frames, 30 s -> 3000 frames.
- Runtime switch: `MEL_BACKEND=librosa|transformers`, default `librosa`, so production can roll back without rebuilding.

Do not touch CUDA/PyCUDA imports, encoder/decoder code, chunk splitting, prompt construction, mel normalization constants, or any unrelated ASR cleanup.

Recommendation: add `app/utils/whisper_mel.py` instead of inlining the implementation into `_compute_mel`. Rationale: `_compute_mel` should remain orchestration and backend selection; the Whisper math needs A/B tests, comments, and cacheable helpers without making `qwen3_asr.py` larger. This also isolates the future removal of the fallback.

## Whisper log-mel pipeline (confirmed spec from transformers source)

Confirmed against Hugging Face `transformers` main branch:

- `WhisperFeatureExtractor.__init__` sets `self.n_samples = chunk_length * sampling_rate`, `self.nb_max_frames = self.n_samples // hop_length`, and builds `self.mel_filters` using `mel_filter_bank(num_frequency_bins=1 + n_fft // 2, num_mel_filters=feature_size, min_frequency=0.0, max_frequency=8000.0, sampling_rate=16000, norm="slaney", mel_scale="slaney")`.
- `WhisperFeatureExtractor.__call__` pads/truncates raw speech to `max_length if max_length else self.n_samples`; with current Qwen3 usage this is exactly `chunk_length * 16000` samples.
- The feature extractor transposes the padded batch and calls `_torch_extract_fbank_features` when torch is available; otherwise `_np_extract_fbank_features`. Both paths implement the same Whisper pipeline.
- Window type is Hann: torch path uses `torch.hann_window(self.n_fft)`; NumPy path uses `window_function(self.n_fft, "hann")`, whose default is periodic Hann.
- STFT is centered with `n_fft=400`, `hop_length=160`, and default reflect padding in both Transformers paths (`torch.stft(..., center=True, pad_mode="reflect")` by PyTorch default; Transformers NumPy `spectrogram(..., center=True, pad_mode="reflect")` by default).
- Spectrum is power, not magnitude: torch path computes `stft[..., :-1].abs() ** 2`; NumPy path calls `spectrogram(..., power=2.0, ...)`.
- The final STFT frame is removed before mel/log normalization: torch path uses `stft[..., :-1]`; NumPy path computes `log_spec = log_spec[:, :-1]`. This converts the centered `n_samples // hop_length + 1` frames to exactly `n_samples // hop_length`.
- Mel scale is Slaney, not HTK: Transformers uses `norm="slaney", mel_scale="slaney"`. In librosa terms, use `htk=False, norm="slaney"`.
- Log scale is base-10, not natural log: torch path uses `.log10()` and NumPy path uses `log_mel="log10"`.
- Epsilon/floor is `1e-10`: torch path clamps `mel_spec` with `min=1e-10`; NumPy `spectrogram` uses `mel_floor=1e-10`.
- Whisper dynamic range and normalization are:

```python
log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
log_spec = (log_spec + 4.0) / 4.0
```

Qwen3 ASR currently uses this same path because `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:1130-1134` constructs `WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, n_fft=400, hop_length=160, chunk_length=chunk_len)` and returns `features["input_features"]`.

## librosa equivalence (step-by-step mapping with exact function calls)

Implement `app/utils/whisper_mel.py` with constants matching current Qwen3 usage:

```python
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
FMIN = 0.0
FMAX = 8000.0
MEL_FLOOR = 1e-10
```

Step 1: coerce audio and pad/trim to the exact Whisper chunk sample length:

```python
audio = np.asarray(audio, dtype=np.float32)
n_samples = int(chunk_length) * SAMPLE_RATE
if audio.shape[0] < n_samples:
    audio = np.pad(audio, (0, n_samples - audio.shape[0]), mode="constant", constant_values=0.0)
else:
    audio = audio[:n_samples]
```

Step 2: cache the mel filter matrix by `chunk_length` key to match the existing `_mel_cache` lifecycle. The filter itself does not mathematically depend on `chunk_length`, but preserving `chunk_length` keys makes fallback and future per-chunk scratch buffers straightforward.

```python
def get_librosa_mel_state(cache: dict, chunk_length: int) -> dict:
    state = cache.get(chunk_length)
    if state is None or state.get("backend") != "librosa":
        mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            htk=False,
            norm="slaney",
            dtype=np.float32,
        )
        state = {"backend": "librosa", "mel_basis": mel_basis}
        cache[chunk_length] = state
    return state
```

Step 3: compute centered STFT using librosa, not hand-rolled `np.fft.rfft`.

```python
stft = librosa.stft(
    y=audio,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=N_FFT,
    window="hann",
    center=True,
    dtype=np.complex64,
    pad_mode="reflect",
)
```

`librosa.stft` is safer than manual `np.fft.rfft` for this migration because it already matches the high-level PyTorch/Transformers STFT contract: centered frames, periodic Hann window, one-sided complex output, and reflect padding. A custom `rfft` loop can be made exact, but it creates more surface for off-by-one frame, window periodicity, buffer reuse, and padding mistakes. If A/B diff exceeds threshold, the fallback implementation should be a local copy of the Transformers NumPy `spectrogram` logic, not an ad hoc rewrite.

Step 4: remove the final STFT frame before power/mel projection, matching Transformers torch path:

```python
magnitudes = np.abs(stft[:, :-1]).astype(np.float32) ** 2.0
```

Step 5: apply Slaney mel filters, floor, base-10 log, Whisper dynamic range clamp, and normalization:

```python
mel_spec = np.matmul(mel_basis, magnitudes)
log_spec = np.log10(np.maximum(mel_spec, MEL_FLOOR))
log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
log_spec = (log_spec + 4.0) / 4.0
return log_spec[np.newaxis, :, :].astype(np.float32, copy=False)
```

Expected output shape:

```python
assert log_spec.shape == (N_MELS, n_samples // HOP_LENGTH)
assert output.shape == (1, N_MELS, n_samples // HOP_LENGTH)
```

## Implementation plan (file:line references + diff sketch)

Scope is intentionally small: one new utility file plus a narrow edit in `_compute_mel`.

1. Add `/Users/harvest/project/jetson-voice/app/utils/whisper_mel.py`.
2. Update `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:1121-1135` only.
3. Preserve `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:1124-1125` exactly.
4. Preserve the existing `self._mel_cache` pattern at `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:1127-1129`.
5. Support fallback via `MEL_BACKEND`, defaulting to `librosa`.

Diff sketch:

```diff
 def _compute_mel(self, audio):
-    from transformers import WhisperFeatureExtractor
     # Use chunk_length matching actual audio to avoid excessive padding
     audio_secs = len(audio) / 16000
     chunk_len = min(30, int(audio_secs) + 1)  # Round up, max 30s
     # Cache the feature extractor for common chunk lengths
     if not hasattr(self, '_mel_cache'):
         self._mel_cache = {}
-    if chunk_len not in self._mel_cache:
-        self._mel_cache[chunk_len] = WhisperFeatureExtractor(
-            feature_size=128, sampling_rate=16000,
-            n_fft=400, hop_length=160, chunk_length=chunk_len)
-    fe = self._mel_cache[chunk_len]
-    features = fe(audio, sampling_rate=16000, return_tensors="np")
-    return features["input_features"]  # [1, 128, T]
+    mel_backend = os.environ.get("MEL_BACKEND", "librosa").strip().lower()
+    if mel_backend == "transformers":
+        from transformers import WhisperFeatureExtractor
+        cache_key = ("transformers", chunk_len)
+        if cache_key not in self._mel_cache:
+            self._mel_cache[cache_key] = WhisperFeatureExtractor(
+                feature_size=128, sampling_rate=16000,
+                n_fft=400, hop_length=160, chunk_length=chunk_len)
+        fe = self._mel_cache[cache_key]
+        features = fe(audio, sampling_rate=16000, return_tensors="np")
+        return features["input_features"]  # [1, 128, T]
+    if mel_backend != "librosa":
+        raise ValueError(f"Unsupported MEL_BACKEND={mel_backend!r}; expected 'librosa' or 'transformers'")
+    from app.utils.whisper_mel import compute_whisper_log_mel
+    return compute_whisper_log_mel(audio, chunk_len, self._mel_cache)
```

Utility sketch:

```python
import numpy as np
import librosa

def compute_whisper_log_mel(audio, chunk_length: int, cache: dict) -> np.ndarray:
    ...
```

Note: `os` is already imported near the top of `qwen3_asr.py`; if not, add only that import. Do not touch CUDA imports or mel norm constants elsewhere.

## Validation (A/B test methodology + thresholds)

Do not ship the default flip until both feature parity and ASR behavior pass.

Feature A/B:

- Collect 8 WAVs that cover short speech, near-1-second boundary, 2-4 second steady-state, 10+ second speech, silence/trailing zero padding, loud speech, quiet speech, and a VAD-cut segment.
- For each WAV, decode through the existing `_bytes_to_float` path so resampling and mono conversion are identical.
- For each sample, compute:

```python
os.environ["MEL_BACKEND"] = "transformers"
mel_tf = backend._compute_mel(audio)
os.environ["MEL_BACKEND"] = "librosa"
mel_lb = backend._compute_mel(audio)
diff = np.abs(mel_tf - mel_lb)
max_abs = float(diff.max())
mean_abs = float(diff.mean())
```

- Required thresholds:
  - Overall 8-wav max abs diff `< 1e-4`.
  - Pad-boundary-only diff `< 1e-3`, measured on the last 2 frames and on samples whose source audio length is not an exact hop multiple.
  - Shapes must match exactly for every WAV.

ASR quality:

- Run the existing CER evaluation set against baseline and librosa default.
- Current baseline to preserve: mean CER `17.22%`, median CER `13.71%`.
- Acceptance: no statistically meaningful regression; any individual high-regression file must be inspected with both mel diff and transcript diff.

Latency:

- Run the existing V2V steady benchmark before and after.
- Current target to preserve: V2V steady median approximately `327 ms`.
- Acceptance: median must not regress; p95 should be checked for import/cache spikes.
- Measure first-call separately from warmed steady-state because importing librosa may be expensive. If first-call matters for the product path, prewarm mel during ASR backend initialization using the existing dummy-audio warmup flow.

## Acceptance criteria

- `MEL_BACKEND=librosa` is the default and returns `[1, 128, T]` float32 arrays for all existing ASR paths.
- `MEL_BACKEND=transformers` preserves the current implementation path and can be used as an immediate production rollback.
- Exact pad/trim length remains `chunk_length * 16000` samples.
- STFT uses Hann window, `n_fft=400`, `hop_length=160`, `win_length=400`, centered reflect padding, and drops the final frame.
- Mel filter bank uses Slaney scale and Slaney area normalization: `htk=False`, `norm="slaney"`.
- Spectrum is power: `abs(stft) ** 2.0`.
- Log is base-10 with floor `1e-10`.
- Normalization remains `maximum(max - 8.0)` then `(log_spec + 4.0) / 4.0`.
- Existing `chunk_len` selection remains unchanged.
- Cache continues to live on `self._mel_cache`, keyed by backend plus `chunk_length` to avoid collisions between Transformers objects and librosa state.
- No CUDA import, encoder, decoder, prompt, chunking, or mel normalization constants are changed.
- Validation passes: 8-wav A/B max abs diff `< 1e-4`, pad boundary `< 1e-3`, CER mean/median do not regress from `17.22%` / `13.71%`, and V2V steady median does not regress from approximately `327 ms`.

## Rollback plan

Immediate rollback:

```bash
MEL_BACKEND=transformers
```

This must select the existing `WhisperFeatureExtractor` behavior inside `_compute_mel` and use the same constructor arguments currently present at `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:1130-1132`.

Code rollback:

- Revert only the `_compute_mel` edit and delete `app/utils/whisper_mel.py`.
- Do not revert unrelated ASR performance, encoder, decoder, CUDA, or mel norm changes.

Operational watch points after rollout:

- First-call latency spike from importing librosa.
- Any transcript degradation near segment boundaries or exact pad boundaries.
- Any shape mismatch into encoder input `mel` at `/Users/harvest/project/jetson-voice/app/backends/qwen3_asr.py:933-976`.

## Anti-patterns

- Do not use `librosa.feature.melspectrogram` as a one-liner unless every argument is explicitly pinned; its defaults can hide STFT padding, power, and dtype differences.
- Do not use HTK mel filters (`htk=True`); Transformers Whisper uses Slaney.
- Do not use natural log or decibel conversion (`librosa.power_to_db`) for this path; Transformers Whisper uses `np.log10` directly followed by Whisper normalization.
- Do not skip the final-frame drop. Without `stft[:, :-1]`, output frame count becomes `chunk_length * 16000 // 160 + 1` and the encoder shape changes.
- Do not change `chunk_len = min(30, int(audio_secs) + 1)`.
- Do not change mel norm constants, clamp range, or tuned ASR constants while migrating the backend.
- Do not import `librosa` at module import time in `qwen3_asr.py`; keep it inside `app/utils/whisper_mel.py` so the fallback path and non-ASR imports stay cleaner.
- Do not cache only by bare `chunk_len` after adding fallback; it risks mixing `WhisperFeatureExtractor` instances and librosa cache dicts. Use keys like `("transformers", chunk_len)` and `("librosa", chunk_len)`.
- Do not hand-roll `np.fft.rfft` first. Use `librosa.stft` with pinned arguments for the initial migration; only move to a local Transformers-style NumPy STFT if A/B diff proves librosa is not close enough.
- Do not run quality validation only on mel arrays. The acceptance gate must include transcript CER and V2V latency because small mel differences can still affect decoder behavior.
