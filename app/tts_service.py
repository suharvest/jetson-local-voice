"""Matcha TTS service using sherpa-onnx (CUDA accelerated).

Matcha TTS + Vocos vocoder: lightweight (125MB), fast TTFT (~60ms streaming),
sentence-level streaming via sherpa-onnx callback.
"""

from __future__ import annotations

import io
import logging
import os
import struct
import time

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "/opt/models/matcha-icefall-zh-en")
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "cuda")
TTS_NUM_THREADS = int(os.environ.get("TTS_NUM_THREADS", "4"))
DEFAULT_SPEAKER_ID = int(os.environ.get("TTS_DEFAULT_SID", "0"))

_tts_instance = None


def get_tts():
    """Lazy-initialize Matcha TTS model."""
    global _tts_instance
    if _tts_instance is not None:
        return _tts_instance

    import sherpa_onnx

    acoustic_model = os.path.join(MODEL_DIR, "model-steps-3.onnx")
    vocoder = os.path.join(MODEL_DIR, "vocos-16khz-univ.onnx")
    lexicon = os.path.join(MODEL_DIR, "lexicon.txt")
    tokens = os.path.join(MODEL_DIR, "tokens.txt")
    data_dir = os.path.join(MODEL_DIR, "espeak-ng-data")

    logger.info("Loading Matcha TTS from %s (provider=%s)", MODEL_DIR, TTS_PROVIDER)
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=acoustic_model,
                vocoder=vocoder,
                lexicon=lexicon,
                tokens=tokens,
                data_dir=data_dir,
                dict_dir=MODEL_DIR,
            ),
            provider=TTS_PROVIDER,
            num_threads=TTS_NUM_THREADS,
        ),
    )
    _tts_instance = sherpa_onnx.OfflineTts(config)
    logger.info("Matcha TTS loaded (sample_rate=%d).", _tts_instance.sample_rate)
    return _tts_instance


def samples_to_wav(samples: list, sample_rate: int) -> bytes:
    """Convert float32 samples to WAV bytes (16-bit PCM)."""
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

    import numpy as np
    arr = np.array(samples, dtype=np.float32)
    np.clip(arr, -1.0, 1.0, out=arr)
    buf.write((arr * 32767).astype(np.int16).tobytes())

    return buf.getvalue()


def synthesize(
    text: str,
    speaker_id: int | None = None,
    speed: float = 1.0,
    **kwargs,
) -> tuple[bytes, dict]:
    """Synthesize text to WAV bytes. Returns (wav_bytes, metadata)."""
    if speaker_id is None:
        speaker_id = DEFAULT_SPEAKER_ID

    tts = get_tts()
    start = time.time()
    audio = tts.generate(text, sid=speaker_id, speed=speed)
    elapsed = time.time() - start

    duration = len(audio.samples) / audio.sample_rate
    wav_bytes = samples_to_wav(audio.samples, audio.sample_rate)

    meta = {
        "duration": round(duration, 3),
        "inference_time": round(elapsed, 3),
        "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
        "sample_rate": audio.sample_rate,
    }
    return wav_bytes, meta


def preload() -> None:
    """Pre-load TTS model and warmup CUDA kernels.

    CUDA kernels are JIT-compiled per unique tensor shape. Matcha TTS
    generates different shapes based on phoneme sequence length, so we
    warm up with varied text lengths (ZH + EN) across multiple rounds
    to pre-compile all common kernel variants.
    """
    tts = get_tts()

    warmup_texts = [
        # ZH: cover 1-15 char lengths for full phoneme shape coverage
        "好",
        "你好",
        "好的呢",
        "没问题。",
        "今天天气不错",
        "你好，很高兴认识你。",
        "没问题，我来看一下。",
        "我来帮你看一下这个问题",
        "今天天气真不错，我们出去走走吧。",
        "你好，我是你的智能助手，很高兴认识你。",
        # EN: cover short to medium
        "OK",
        "Sure, no problem.",
        "Hello, nice to meet you.",
        "Let me help you with that.",
        "I'd be happy to help you with that question.",
    ]
    n_rounds = 5 if TTS_PROVIDER == "cuda" else 1

    start = time.time()
    for _ in range(n_rounds):
        for text in warmup_texts:
            tts.generate(text, sid=DEFAULT_SPEAKER_ID, speed=1.0)
    elapsed = time.time() - start
    logger.info(
        "TTS warmup done: %d texts x %d rounds = %d calls in %.1fs",
        len(warmup_texts), n_rounds, len(warmup_texts) * n_rounds, elapsed,
    )


def get_sample_rate() -> int:
    """Return the model's audio sample rate."""
    return get_tts().sample_rate


def is_ready() -> bool:
    return _tts_instance is not None
