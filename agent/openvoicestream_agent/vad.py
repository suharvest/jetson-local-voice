"""Client-side VAD for utterance segmentation.

Two backends:
- silero: silero-vad onnx (preferred, accurate). pip extra: `silero-vad`
- energy: pure numpy energy threshold (fallback, always available)
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class EnergyVAD:
    """Simple RMS-based VAD. Threshold tuned for typical built-in mics."""

    name = "energy"

    def __init__(self, threshold: float = 0.012, sample_rate: int = 16000) -> None:
        self.threshold = threshold
        self.sample_rate = sample_rate

    def is_speech(self, pcm: bytes) -> bool:
        if not pcm:
            return False
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if len(samples) == 0:
            return False
        rms = float(np.sqrt(np.mean(samples * samples)))
        return rms > self.threshold

    def reset(self) -> None:
        pass


class SileroVAD:
    """silero-vad onnx via the silero_vad package. Accurate but heavier."""

    name = "silero"

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000) -> None:
        from silero_vad import load_silero_vad  # late import

        self.threshold = threshold
        self.sample_rate = sample_rate
        self._model = load_silero_vad(onnx=True)
        # silero expects 32ms windows at 16kHz = 512 samples
        self._win = 512 if sample_rate == 16000 else 256
        self._buf = np.zeros(0, dtype=np.float32)

    def is_speech(self, pcm: bytes) -> bool:
        import torch

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, samples])
        any_speech = False
        while len(self._buf) >= self._win:
            win = self._buf[: self._win]
            self._buf = self._buf[self._win :]
            t = torch.from_numpy(win).float()
            prob = float(self._model(t, self.sample_rate).item())
            if prob >= self.threshold:
                any_speech = True
        return any_speech

    def reset(self) -> None:
        try:
            self._model.reset_states()
        except Exception:  # pragma: no cover
            pass
        self._buf = np.zeros(0, dtype=np.float32)


def create_vad(backend: str, sample_rate: int = 16000, threshold: float | None = None):
    """Build a VAD by name. `auto` tries silero, falls back to energy."""
    if backend in ("silero", "auto"):
        try:
            return SileroVAD(
                threshold=threshold if threshold is not None else 0.5,
                sample_rate=sample_rate,
            )
        except Exception as e:
            if backend == "silero":
                raise
            logger.info("silero VAD unavailable (%s), falling back to energy VAD", e)
    return EnergyVAD(
        threshold=threshold if threshold is not None else 0.012,
        sample_rate=sample_rate,
    )


__all__ = ["EnergyVAD", "SileroVAD", "create_vad"]
