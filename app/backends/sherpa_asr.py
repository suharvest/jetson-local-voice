"""Sherpa-onnx ASR backend (Paraformer/Zipformer streaming + SenseVoice offline).

Consolidates logic from streaming_asr_service.py and asr_service.py.
Supports: OFFLINE, STREAMING
"""

from __future__ import annotations

import glob
import io
import logging
import os
from typing import Optional

import numpy as np

from asr_backend import ASRBackend, ASRCapability, ASRStream, TranscriptionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level config
# ---------------------------------------------------------------------------

LANGUAGE_MODE = os.environ.get("LANGUAGE_MODE", "zh_en")  # "zh_en" or "en"
_DEFAULT_ASR_DIRS = {
    "zh_en": "/opt/models/paraformer-streaming",
    "en": "/opt/models/zipformer-en",
}
STREAMING_MODEL_DIR = os.environ.get(
    "STREAMING_MODEL_DIR",
    _DEFAULT_ASR_DIRS.get(LANGUAGE_MODE, _DEFAULT_ASR_DIRS["zh_en"]),
)
ASR_PROVIDER = os.environ.get("STREAMING_ASR_PROVIDER", "cuda")
ASR_NUM_THREADS = int(os.environ.get("STREAMING_ASR_NUM_THREADS", "4"))

# ---------------------------------------------------------------------------
# BPE merge table (Zipformer / "en" mode only)
# ---------------------------------------------------------------------------

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
    """Merge BPE-split words back together (en mode only)."""
    for split, merged in _MERGE_WORDS.items():
        text = text.replace(split, merged)
    return text


# ---------------------------------------------------------------------------
# SherpaASRStream
# ---------------------------------------------------------------------------


class SherpaASRStream(ASRStream):
    """Streaming ASR session backed by a sherpa_onnx OnlineRecognizer."""

    def __init__(self, recognizer, language_mode: str = LANGUAGE_MODE):
        self._recognizer = recognizer
        self._language_mode = language_mode
        self._stream = recognizer.create_stream()
        self._last_text = ""
        self._is_endpoint = False

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        recognizer = self._recognizer

        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if np.abs(samples).max() > 1.0:
            samples = samples / 32768.0

        self._stream.accept_waveform(sample_rate, samples)

        while recognizer.is_ready(self._stream):
            recognizer.decode_stream(self._stream)

        text = recognizer.get_result(self._stream).strip()
        if self._language_mode == "en":
            text = _fix_bpe_splits(text)

        is_endpoint = recognizer.is_endpoint(self._stream)

        self._last_text = text
        self._is_endpoint = is_endpoint

        if is_endpoint:
            # Reset stream for next utterance
            self._stream = self._recognizer.create_stream()

    def finalize(self) -> str:
        recognizer = self._recognizer
        stream = self._stream

        if self._language_mode == "en":
            silence = np.zeros(int(16000 * 0.8), dtype=np.float32)
            stream.accept_waveform(16000, silence)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        text = recognizer.get_result(stream).strip()
        if self._language_mode == "en":
            text = _fix_bpe_splits(text)
        return text

    def get_partial(self) -> tuple[str, bool]:
        text = self._last_text
        is_endpoint = self._is_endpoint
        if is_endpoint:
            self._is_endpoint = False
            self._last_text = ""
        return text, is_endpoint


# ---------------------------------------------------------------------------
# SherpaASRBackend
# ---------------------------------------------------------------------------


class SherpaASRBackend(ASRBackend):

    def __init__(self):
        self._online_recognizer = None
        self._offline_recognizer = None

    @property
    def name(self) -> str:
        return "sherpa_asr"

    @property
    def capabilities(self) -> set[ASRCapability]:
        caps = {ASRCapability.OFFLINE}
        if self._online_recognizer is not None:
            caps.add(ASRCapability.STREAMING)
        return caps

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._offline_recognizer is not None or self._online_recognizer is not None

    def preload(self) -> None:
        try:
            self._online_recognizer = self._load_online_recognizer()
            logger.info("Sherpa streaming ASR loaded")
        except Exception as e:
            logger.info("Streaming ASR not available: %s", e)
        try:
            self._offline_recognizer = self._load_offline_recognizer()
            logger.info("Sherpa offline ASR loaded")
        except Exception as e:
            logger.info("Offline ASR not available: %s", e)

    def create_stream(self, language: str = "auto") -> ASRStream:
        if self._online_recognizer is None:
            raise RuntimeError("Online recognizer not loaded; call preload() first")
        return SherpaASRStream(self._online_recognizer, LANGUAGE_MODE)

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if self._offline_recognizer is None:
            raise RuntimeError("Offline recognizer not loaded; call preload() first")

        import soundfile as sf

        data, sample_rate = sf.read(io.BytesIO(audio_bytes))

        # Convert to mono float32
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)

        # Resample to 16 kHz using numpy linear interpolation (no subprocess/sox)
        if sample_rate != 16000:
            target_len = int(len(data) * 16000 / sample_rate)
            data = np.interp(
                np.linspace(0, len(data) - 1, target_len),
                np.arange(len(data)),
                data,
            ).astype(np.float32)
            sample_rate = 16000

        recognizer = self._offline_recognizer
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, data)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        return TranscriptionResult(text=text, language=language)

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load_online_recognizer():
        """Load Paraformer (zh_en) or Zipformer (en) streaming recognizer."""
        import sherpa_onnx

        model_dir = STREAMING_MODEL_DIR
        tokens = os.path.join(model_dir, "tokens.txt")

        if LANGUAGE_MODE == "en":
            encoder = os.path.join(model_dir, "encoder-epoch-99-avg-1-chunk-16-left-128.onnx")
            decoder = os.path.join(model_dir, "decoder-epoch-99-avg-1-chunk-16-left-128.onnx")
            joiner = os.path.join(model_dir, "joiner-epoch-99-avg-1-chunk-16-left-128.onnx")

            logger.info("Loading streaming Zipformer (en) from %s (provider=%s)", model_dir, ASR_PROVIDER)
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
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
            encoder = os.path.join(model_dir, "encoder.onnx")
            decoder = os.path.join(model_dir, "decoder.onnx")

            logger.info("Loading streaming Paraformer from %s (provider=%s)", model_dir, ASR_PROVIDER)
            recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
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

        return recognizer

    @staticmethod
    def _load_offline_recognizer():
        """Load SenseVoice offline recognizer."""
        import sherpa_onnx

        base = os.path.join(os.environ.get("MODEL_DIR", "/opt/models"), "sensevoice")
        dirs = glob.glob(os.path.join(base, "sherpa-onnx-sense-voice-*"))
        model_dir = dirs[0] if dirs else base

        model_path = os.path.join(model_dir, "model.int8.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")

        logger.info("Loading SenseVoice model from %s (provider=cuda)", model_dir)
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            use_itn=True,
            provider="cuda",
        )
        logger.info("SenseVoice model loaded.")
        return recognizer
