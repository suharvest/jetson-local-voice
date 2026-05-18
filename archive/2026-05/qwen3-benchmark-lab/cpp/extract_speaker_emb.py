#!/usr/bin/env python3
"""Extract speaker embedding from reference audio for voice cloning.

Uses exact mel spectrogram parameters from official Qwen3-TTS:
  n_fft=1024, hop_size=256, win_size=1024, n_mels=128, fmin=0, fmax=12000
  librosa slaney-norm mel filterbank, hann window, center=False

Usage:
    python3 extract_speaker_emb.py --audio ref.wav --model speaker_encoder.onnx --output spk.bin
"""
import argparse
import numpy as np
import wave
import onnxruntime as ort


def librosa_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """Slaney-norm mel filterbank (matches librosa.filters.mel with norm='slaney')."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    fmin_mel = hz_to_mel(fmin)
    fmax_mel = hz_to_mel(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, n_mels + 2)
    freqs = mel_to_hz(mels)

    n_bins = n_fft // 2 + 1
    fftfreqs = np.linspace(0, sr / 2, n_bins)
    fb = np.zeros((n_mels, n_bins))

    for i in range(n_mels):
        lower = freqs[i]
        center = freqs[i + 1]
        upper = freqs[i + 2]
        for j in range(n_bins):
            if lower <= fftfreqs[j] <= center and center > lower:
                fb[i, j] = (fftfreqs[j] - lower) / (center - lower)
            elif center < fftfreqs[j] <= upper and upper > center:
                fb[i, j] = (upper - fftfreqs[j]) / (upper - center)

        # Slaney normalization
        enorm = 2.0 / (freqs[i + 2] - freqs[i])
        fb[i] *= enorm

    return fb


def mel_spectrogram(audio, sr=24000, n_fft=1024, hop_size=256, win_size=1024,
                    n_mels=128, fmin=0, fmax=12000):
    """Compute mel spectrogram matching official Qwen3-TTS implementation."""
    # Reflect padding (center=False)
    padding = (n_fft - hop_size) // 2  # 384
    audio_padded = np.pad(audio, (padding, padding), mode='reflect')

    # STFT with hann window
    window = np.hanning(win_size + 1)[:-1]  # periodic hann = hann_window in torch
    n_frames = 1 + (len(audio_padded) - n_fft) // hop_size
    frames = np.lib.stride_tricks.as_strided(
        audio_padded,
        shape=(n_frames, n_fft),
        strides=(audio_padded.strides[0] * hop_size, audio_padded.strides[0])
    ).copy()
    frames *= window
    spec_complex = np.fft.rfft(frames, n=n_fft)
    spec = np.sqrt(np.real(spec_complex)**2 + np.imag(spec_complex)**2 + 1e-9)

    # Mel filterbank (slaney norm, matches librosa)
    mel_basis = librosa_mel_filterbank(sr, n_fft, n_mels, fmin, fmax)

    # mel_spec = mel_basis @ spec.T
    mel_spec = mel_basis @ spec.T  # [n_mels, n_frames]

    # Dynamic range compression: log(clamp(x, 1e-5))
    mel_spec = np.log(np.maximum(mel_spec, 1e-5))

    return mel_spec.T  # [n_frames, n_mels]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="Reference audio WAV (24kHz)")
    p.add_argument("--model", default="/tmp/qwen3-v2/speaker_encoder.onnx")
    p.add_argument("--output", default="/tmp/spk_emb.bin")
    args = p.parse_args()

    # Load audio
    with wave.open(args.audio) as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    print(f"Audio: {len(audio)/sr:.1f}s at {sr}Hz")

    assert sr == 24000, f"Expected 24kHz, got {sr}Hz"

    # Compute mel
    mel = mel_spectrogram(audio)
    print(f"Mel: {mel.shape}")

    # Run speaker encoder
    sess = ort.InferenceSession(args.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    result = sess.run(None, {"mel": mel[np.newaxis].astype(np.float32)})
    spk = result[0].flatten().astype(np.float32)
    print(f"Speaker embedding: {spk.shape}, norm={np.linalg.norm(spk):.3f}")

    # Save
    spk.tofile(args.output)
    print(f"Saved: {args.output} ({spk.size * 4} bytes)")


if __name__ == "__main__":
    main()
