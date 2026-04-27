"""Whisper log-mel feature extractor implemented with librosa + numpy.

This is a drop-in equivalent of the path used by
``transformers.WhisperFeatureExtractor`` for Qwen3 ASR (feature_size=128,
sampling_rate=16000, n_fft=400, hop_length=160). It is intentionally a
narrow port of the Whisper feature pipeline so we can drop the
``transformers`` runtime dependency.

Spec: docs/plans/asr-mel-librosa-2026-04-27.md
"""

from __future__ import annotations

import numpy as np

# Constants pinned to Whisper / Qwen3 ASR usage.
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
FMIN = 0.0
FMAX = 8000.0
MEL_FLOOR = 1e-10


def _get_mel_state(cache: dict, chunk_length: int) -> dict:
    """Return cached mel filter state, building it lazily.

    The mel filter matrix only depends on (sr, n_fft, n_mels, fmin, fmax),
    not on chunk_length. We still key by ("librosa", chunk_length) so this
    cache slot does not collide with the transformers fallback path which
    keys by ("transformers", chunk_length) (or legacy bare chunk_length).
    """
    import librosa

    key = ("librosa", chunk_length)
    state = cache.get(key)
    if state is not None and state.get("backend") == "librosa":
        return state

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
    cache[key] = state
    return state


def compute_whisper_log_mel(
    audio: np.ndarray,
    chunk_length: int,
    cache: dict,
) -> np.ndarray:
    """Compute Whisper log-mel features as ``[1, 128, T]`` float32.

    Mirrors the transformers ``WhisperFeatureExtractor`` numpy path:
      * pad/trim to ``chunk_length * 16000`` samples
      * centered STFT with ``n_fft=400``, ``hop_length=160``, periodic Hann,
        reflect padding
      * drop the final STFT frame
      * power spectrum (``|stft|**2``)
      * Slaney mel filter bank, base-10 log, floor ``1e-10``
      * Whisper dynamic range clamp + ``(x + 4) / 4`` normalization
    """
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        audio = audio.reshape(-1)

    n_samples = int(chunk_length) * SAMPLE_RATE
    if audio.shape[0] < n_samples:
        audio = np.pad(
            audio,
            (0, n_samples - audio.shape[0]),
            mode="constant",
            constant_values=0.0,
        )
    else:
        audio = audio[:n_samples]

    state = _get_mel_state(cache, chunk_length)
    mel_basis = state["mel_basis"]

    # Centered STFT, periodic Hann window, reflect padding to match
    # transformers (numpy path uses spectrogram(center=True, pad_mode="reflect")).
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

    # Drop final frame to match transformers torch path (stft[..., :-1]).
    magnitudes = np.abs(stft[:, :-1]).astype(np.float32) ** 2.0

    mel_spec = mel_basis @ magnitudes
    log_spec = np.log10(np.maximum(mel_spec, MEL_FLOOR))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec[np.newaxis, :, :].astype(np.float32, copy=False)
