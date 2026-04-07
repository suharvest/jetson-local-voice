"""Mel spectrogram feature extractor (NumPy + Librosa based)."""

import numpy as np


class MelExtractor:
    """
    Whisper-compatible mel spectrogram extractor.
    
    Uses pre-computed mel filter bank (from Qwen3-ASR / Whisper).
    Parameters: n_fft=400, hop_length=160, window='hann', center=True.
    Output: (128, T) mel spectrogram, log-scaled and normalized.
    """

    def __init__(self, filter_path: str):
        """
        Args:
            filter_path: Path to mel_filters.npy (shape: [201, 128])
        """
        self.filters = np.load(filter_path)  # (201, 128)

    def __call__(self, audio: np.ndarray, dtype=np.float32) -> np.ndarray:
        """
        Extract mel spectrogram from audio waveform.
        
        Args:
            audio: 1D float32 waveform at 16kHz
            dtype: Output dtype (float32 for RKNN)
            
        Returns:
            (128, T) mel spectrogram where T = len(audio)//160 + 1
        """
        import librosa
        stft = librosa.stft(audio, n_fft=400, hop_length=160,
                            window='hann', center=True)
        magnitudes = np.abs(stft) ** 2
        mel_spec = np.dot(self.filters.T, magnitudes)
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        # Frame alignment: discard extra frame from center=True padding
        # (matches Qwen3-ASR official implementation)
        n_frames = audio.shape[-1] // 160
        log_spec = log_spec[:, :n_frames]
        return log_spec.astype(dtype)
