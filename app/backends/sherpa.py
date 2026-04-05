"""Sherpa-onnx TTS backend (Matcha / Kokoro).

Supports: BASIC_TTS, STREAMING, MULTI_SPEAKER
"""

from __future__ import annotations

import io
import logging
import os
import struct
import time
from typing import Optional

import numpy as np

from tts_backend import TTSBackend, TTSCapability

logger = logging.getLogger(__name__)

LANGUAGE_MODE = os.environ.get("LANGUAGE_MODE", "zh_en")
_DEFAULT_TTS_DIRS = {
    "zh_en": "/opt/models/matcha-icefall-zh-en",
    "en": "/opt/models/kokoro-multi-lang-v1_0",
}
MODEL_DIR = os.environ.get(
    "TTS_MODEL_DIR", _DEFAULT_TTS_DIRS.get(LANGUAGE_MODE, _DEFAULT_TTS_DIRS["zh_en"])
)
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "cuda")
TTS_NUM_THREADS = int(os.environ.get("TTS_NUM_THREADS", "4"))
_DEFAULT_SIDS = {"zh_en": "0", "en": "52"}
DEFAULT_SPEAKER_ID = int(
    os.environ.get("TTS_DEFAULT_SID", _DEFAULT_SIDS.get(LANGUAGE_MODE, "0"))
)
DEFAULT_SPEED = float(os.environ.get("TTS_DEFAULT_SPEED", "1.0"))
PITCH_SHIFT = float(os.environ.get("TTS_PITCH_SHIFT", "0"))


def _pitch_shift_samples(samples: list, semitones: float) -> list:
    if semitones == 0:
        return samples
    ratio = 2 ** (semitones / 12)
    arr = np.array(samples, dtype=np.float32)
    new_len = int(len(arr) / ratio)
    indices = np.linspace(0, len(arr) - 1, new_len)
    return np.interp(indices, np.arange(len(arr)), arr).tolist()


def _samples_to_wav(samples: list, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    num_samples = len(samples)
    data_size = num_samples * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    arr = np.array(samples, dtype=np.float32)
    np.clip(arr, -1.0, 1.0, out=arr)
    buf.write((arr * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


class SherpaBackend(TTSBackend):
    """Sherpa-onnx TTS (Matcha Chinese+English or Kokoro English)."""

    def __init__(self):
        self._tts = None
        self._ready = False

    @property
    def name(self) -> str:
        return "sherpa"

    @property
    def capabilities(self) -> set[TTSCapability]:
        caps = {TTSCapability.BASIC_TTS, TTSCapability.STREAMING}
        if LANGUAGE_MODE == "en":
            caps.add(TTSCapability.MULTI_SPEAKER)
        return caps

    @property
    def sample_rate(self) -> int:
        if self._tts:
            return self._tts.sample_rate
        return 24000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        self._load_model()
        self._warmup()
        self._ready = True

    def _load_model(self):
        import sherpa_onnx

        if LANGUAGE_MODE == "en":
            logger.info("Loading Kokoro TTS from %s", MODEL_DIR)
            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=os.path.join(MODEL_DIR, "model.onnx"),
                        voices=os.path.join(MODEL_DIR, "voices.bin"),
                        tokens=os.path.join(MODEL_DIR, "tokens.txt"),
                        lexicon=os.path.join(MODEL_DIR, "lexicon-us-en.txt"),
                        data_dir=os.path.join(MODEL_DIR, "espeak-ng-data"),
                        dict_dir=MODEL_DIR,
                    ),
                    provider=TTS_PROVIDER,
                    num_threads=TTS_NUM_THREADS,
                ),
            )
        else:
            logger.info("Loading Matcha TTS from %s", MODEL_DIR)
            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                        acoustic_model=os.path.join(MODEL_DIR, "model-steps-3.onnx"),
                        vocoder=os.path.join(MODEL_DIR, "vocos-16khz-univ.onnx"),
                        lexicon=os.path.join(MODEL_DIR, "lexicon.txt"),
                        tokens=os.path.join(MODEL_DIR, "tokens.txt"),
                        data_dir=os.path.join(MODEL_DIR, "espeak-ng-data"),
                        dict_dir=MODEL_DIR,
                    ),
                    provider=TTS_PROVIDER,
                    num_threads=TTS_NUM_THREADS,
                ),
            )

        self._tts = sherpa_onnx.OfflineTts(config)
        logger.info("TTS loaded (sample_rate=%d).", self._tts.sample_rate)

    def _warmup(self):
        if LANGUAGE_MODE == "en":
            texts = ["OK", "Sure.", "Hello, nice to meet you."]
        else:
            texts = ["好", "你好", "今天天气不错", "OK", "Hello."]
        n_rounds = 5 if TTS_PROVIDER == "cuda" else 1
        start = time.time()
        for _ in range(n_rounds):
            for t in texts:
                self._tts.generate(t, sid=DEFAULT_SPEAKER_ID, speed=1.0)
        logger.info("TTS warmup: %.1fs", time.time() - start)

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if speaker_id is None:
            speaker_id = DEFAULT_SPEAKER_ID
        if speed is None:
            speed = DEFAULT_SPEED
        if pitch_shift is None:
            pitch_shift = PITCH_SHIFT

        start = time.time()
        audio = self._tts.generate(text, sid=speaker_id, speed=speed)
        if not audio.samples or len(audio.samples) == 0:
            logger.warning("Speaker %d empty, fallback to %d", speaker_id, DEFAULT_SPEAKER_ID)
            audio = self._tts.generate(text, sid=DEFAULT_SPEAKER_ID, speed=speed)
        elapsed = time.time() - start

        samples = _pitch_shift_samples(audio.samples, pitch_shift)
        duration = len(samples) / audio.sample_rate
        wav_bytes = _samples_to_wav(samples, audio.sample_rate)

        meta = {
            "duration": round(duration, 3),
            "inference_time": round(elapsed, 3),
            "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
            "sample_rate": audio.sample_rate,
        }
        return wav_bytes, meta

    def generate_streaming(self, text: str, **kwargs):
        """Yield PCM int16 chunks (sentence-level streaming)."""
        import queue

        sid = kwargs.get("speaker_id", DEFAULT_SPEAKER_ID)
        speed = kwargs.get("speed", DEFAULT_SPEED)
        pitch = kwargs.get("pitch_shift", PITCH_SHIFT)

        audio_queue: queue.Queue[bytes | None] = queue.Queue()

        def callback(samples, progress):
            shifted = _pitch_shift_samples(samples, pitch)
            arr = np.array(shifted, dtype=np.float32)
            np.clip(arr, -1.0, 1.0, out=arr)
            pcm = (arr * 32767).astype(np.int16).tobytes()
            audio_queue.put(pcm)
            return 1

        self._tts.generate(text, sid=sid, speed=speed, callback=callback)
        audio_queue.put(None)

        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            yield chunk
