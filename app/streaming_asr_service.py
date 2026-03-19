"""Streaming ASR service using sherpa-onnx OnlineRecognizer.

Supports two modes via LANGUAGE_MODE env var:
  - "zh_en" (default): Paraformer bilingual Chinese+English
  - "en": Zipformer English-only
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

LANGUAGE_MODE = os.environ.get("LANGUAGE_MODE", "zh_en")  # "zh_en" or "en"
_DEFAULT_ASR_DIRS = {
    "zh_en": "/opt/models/paraformer-streaming",
    "en": "/opt/models/zipformer-en",
}
MODEL_DIR = os.environ.get("STREAMING_ASR_MODEL_DIR", _DEFAULT_ASR_DIRS.get(LANGUAGE_MODE, _DEFAULT_ASR_DIRS["zh_en"]))
ASR_PROVIDER = os.environ.get("STREAMING_ASR_PROVIDER", "cuda")
ASR_NUM_THREADS = int(os.environ.get("STREAMING_ASR_NUM_THREADS", "4"))

_recognizer = None

# Zipformer BPE tokenizer sometimes splits common words across tokens.
# This dict merges them back. Only applied in "en" mode.
_MERGE_WORDS = {
    "TO DAY": "TODAY",
    "TO NIGHT": "TONIGHT",
    "TO MORROW": "TOMORROW",
    "TO GETHER": "TOGETHER",
    "TO WARD": "TOWARD",
    "TO WARDS": "TOWARDS",
    "SOME THING": "SOMETHING",
    "SOME ONE": "SOMEONE",
    "SOME WHERE": "SOMEWHERE",
    "SOME HOW": "SOMEHOW",
    "SOME TIMES": "SOMETIMES",
    "SOME TIME": "SOMETIME",
    "ANY THING": "ANYTHING",
    "ANY ONE": "ANYONE",
    "ANY WHERE": "ANYWHERE",
    "ANY WAY": "ANYWAY",
    "EVERY THING": "EVERYTHING",
    "EVERY ONE": "EVERYONE",
    "EVERY WHERE": "EVERYWHERE",
    "EVERY BODY": "EVERYBODY",
    "NO THING": "NOTHING",
    "NO WHERE": "NOWHERE",
    "NO BODY": "NOBODY",
    "MY SELF": "MYSELF",
    "YOUR SELF": "YOURSELF",
    "HIM SELF": "HIMSELF",
    "HER SELF": "HERSELF",
    "IT SELF": "ITSELF",
    "OUR SELVES": "OURSELVES",
    "THEM SELVES": "THEMSELVES",
    "MEAN WHILE": "MEANWHILE",
    "AL READY": "ALREADY",
    "AL THOUGH": "ALTHOUGH",
    "AL WAYS": "ALWAYS",
    "AL MOST": "ALMOST",
    "AL TOGETHER": "ALTOGETHER",
    "BREAK FAST": "BREAKFAST",
    "UNDER STAND": "UNDERSTAND",
    "OUT SIDE": "OUTSIDE",
    "IN SIDE": "INSIDE",
    "WITH OUT": "WITHOUT",
    "BE CAUSE": "BECAUSE",
    "BE COME": "BECOME",
    "BE FORE": "BEFORE",
    "BE TWEEN": "BETWEEN",
    "BE HIND": "BEHIND",
}


def _fix_bpe_splits(text: str) -> str:
    """Merge BPE-split words back together."""
    for split, merged in _MERGE_WORDS.items():
        text = text.replace(split, merged)
    return text


def get_recognizer():
    """Lazy-init the streaming OnlineRecognizer (Paraformer or Zipformer)."""
    global _recognizer
    if _recognizer is not None:
        return _recognizer

    import sherpa_onnx

    tokens = os.path.join(MODEL_DIR, "tokens.txt")

    if LANGUAGE_MODE == "en":
        # Zipformer transducer: encoder + decoder + joiner
        encoder = os.path.join(MODEL_DIR, "encoder-epoch-99-avg-1-chunk-16-left-128.onnx")
        decoder = os.path.join(MODEL_DIR, "decoder-epoch-99-avg-1-chunk-16-left-128.onnx")
        joiner = os.path.join(MODEL_DIR, "joiner-epoch-99-avg-1-chunk-16-left-128.onnx")

        logger.info("Loading streaming Zipformer (en) from %s (provider=%s)", MODEL_DIR, ASR_PROVIDER)
        _recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            provider=ASR_PROVIDER,
            num_threads=ASR_NUM_THREADS,
            enable_endpoint_detection=True,
            rule2_min_trailing_silence=0.6,
        )
        logger.info("Streaming Zipformer (en) loaded.")
    else:
        # Paraformer: encoder + decoder
        encoder = os.path.join(MODEL_DIR, "encoder.onnx")
        decoder = os.path.join(MODEL_DIR, "decoder.onnx")

        logger.info("Loading streaming Paraformer from %s (provider=%s)", MODEL_DIR, ASR_PROVIDER)
        _recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
            encoder=encoder,
            decoder=decoder,
            tokens=tokens,
            provider=ASR_PROVIDER,
            num_threads=ASR_NUM_THREADS,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=0.6,
            rule3_min_utterance_length=20,
        )
        logger.info("Streaming Paraformer loaded.")

    return _recognizer


def create_stream():
    """Create a new online stream for one utterance."""
    recognizer = get_recognizer()
    return recognizer.create_stream()


def feed_and_decode(stream, samples: np.ndarray, sample_rate: int = 16000):
    """Feed audio samples and decode. Returns (text, is_final)."""
    recognizer = get_recognizer()

    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    if np.abs(samples).max() > 1.0:
        samples = samples / 32768.0

    stream.accept_waveform(sample_rate, samples)

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    text = recognizer.get_result(stream).strip()
    if LANGUAGE_MODE == "en":
        text = _fix_bpe_splits(text)
    # sherpa-onnx OnlineRecognizer: result includes partial until endpoint
    is_endpoint = recognizer.is_endpoint(stream)

    return text, is_endpoint


def finalize(stream, sample_rate: int = 16000) -> str:
    """Finalize the stream (flush remaining audio). Returns final text.

    For Paraformer (zh_en): the patched sherpa-onnx EOF fix makes
    input_finished() sufficient — no silence padding needed.

    For Zipformer (en): the transducer decoder needs trailing silence
    to flush the last few tokens. We pad 0.3s of silence before
    calling input_finished().
    """
    recognizer = get_recognizer()

    if LANGUAGE_MODE == "en":
        # Zipformer transducer needs silence padding to flush tail tokens
        silence = np.zeros(int(sample_rate * 0.8), dtype=np.float32)
        stream.accept_waveform(sample_rate, silence)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    text = recognizer.get_result(stream).strip()
    if LANGUAGE_MODE == "en":
        text = _fix_bpe_splits(text)
    return text


def preload() -> None:
    """Pre-load model."""
    try:
        get_recognizer()
    except Exception as e:
        logger.warning(f"Streaming ASR preload failed (model may not be installed): {e}")


def is_ready() -> bool:
    return _recognizer is not None
