"""Qwen3-TTS backend via C++ TRT native engine (pybind11).

Supports: BASIC_TTS, VOICE_CLONE, MULTI_LANGUAGE
Models loaded once at preload(), C++ engine stays resident in memory.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from tts_backend import TTSBackend, TTSCapability

logger = logging.getLogger(__name__)


def _detect_language(text: str) -> str:
    """Simple language detection — returns config-compatible language strings."""
    for ch in text:
        cp = ord(ch)
        # CJK Unified Ideographs
        if 0x4E00 <= cp <= 0x9FFF:
            return "chinese"
        # Japanese Hiragana / Katakana
        if 0x3040 <= cp <= 0x30FF:
            return "japanese"
        # Korean Hangul
        if 0xAC00 <= cp <= 0xD7AF:
            return "korean"
    return "english"

# Paths — all under /opt/models/qwen3-tts (persistent volume)
_BASE = os.environ.get("QWEN3_MODEL_BASE", "/opt/models/qwen3-tts")
QWEN3_SHERPA_DIR = os.environ.get("QWEN3_SHERPA_DIR", os.path.join(_BASE, "onnx"))
QWEN3_MODEL_DIR = os.environ.get("QWEN3_MODEL_DIR", os.path.join(_BASE, "onnx"))
QWEN3_TALKER_ENGINE = os.environ.get("QWEN3_TALKER_ENGINE", os.path.join(_BASE, "engines", "talker_decode_bf16.engine"))
QWEN3_CP_ENGINE = os.environ.get("QWEN3_CP_ENGINE", os.path.join(_BASE, "engines", "cp_bf16.engine"))
QWEN3_SPEAKER_ENCODER = os.environ.get("QWEN3_SPEAKER_ENCODER", os.path.join(_BASE, "onnx", "speaker_encoder.onnx"))
QWEN3_TOKENIZER_DIR = os.environ.get("QWEN3_TOKENIZER_DIR", os.path.join(_BASE, "tokenizer"))
QWEN3_EXTRACT_SCRIPT = os.environ.get("QWEN3_EXTRACT_SCRIPT", os.path.join(_BASE, "extract_speaker_emb.py"))


class Qwen3TRTBackend(TTSBackend):
    """Qwen3-TTS via C++ TRT native inference (pybind11 module, models resident)."""

    def __init__(self):
        self._engine = None  # qwen3_speech_engine.Pipeline
        self._tokenizer = None
        self._ready = False

    @property
    def name(self) -> str:
        return "qwen3_trt"

    @property
    def capabilities(self) -> set[TTSCapability]:
        caps = {TTSCapability.BASIC_TTS, TTSCapability.MULTI_LANGUAGE,
                TTSCapability.STREAMING}
        if os.path.exists(QWEN3_SPEAKER_ENCODER):
            caps.add(TTSCapability.VOICE_CLONE)
        return caps

    @property
    def sample_rate(self) -> int:
        return 24000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        """Load C++ TRT engine + tokenizer. Models stay resident."""
        # Verify files
        for path, desc in [
            (QWEN3_TALKER_ENGINE, "talker engine"),
            (QWEN3_CP_ENGINE, "CP engine"),
            (os.path.join(_BASE, "engines", "vocoder_fp16.engine"), "vocoder engine"),
            (os.path.join(QWEN3_SHERPA_DIR, "config.json"), "config.json (authoritative)"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {desc}: {path}")

        # Load tokenizer
        self._load_tokenizer()

        # Load C++ engine (this is the heavy part: ~25s for model loading + embed table)
        logger.info("Loading Qwen3 TRT engine (this takes ~25s)...")
        t0 = time.time()

        import qwen3_speech_engine
        self._engine = qwen3_speech_engine.Pipeline(
            QWEN3_MODEL_DIR, QWEN3_SHERPA_DIR,
            QWEN3_TALKER_ENGINE, QWEN3_CP_ENGINE,
        )
        logger.info("Qwen3 TRT engine loaded in %.1fs", time.time() - t0)

        # Enable cached CUDA Graph for talker decode:
        # First request populates cache (~10ms extra per unique seq_len),
        # subsequent requests replay cached graphs (~3ms vs 26ms baseline).
        try:
            self._engine.enable_cuda_graph(True)
            logger.info("CUDA Graph enabled for talker decode (cached mode)")
        except Exception as e:
            logger.warning("CUDA Graph enable failed (non-fatal): %s", e)

        self._ready = True

    def _load_tokenizer(self):
        vocab_path = os.path.join(QWEN3_TOKENIZER_DIR, "vocab.json")
        merges_path = os.path.join(QWEN3_TOKENIZER_DIR, "merges.txt")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Tokenizer not found: {vocab_path}")

        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel

        self._tokenizer = Tokenizer(BPE(vocab_path, merges_path))
        self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        logger.info("Tokenizer loaded from %s", QWEN3_TOKENIZER_DIR)

    def _tokenize(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self._tokenizer.encode(text).ids

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if language is None:
            language = _detect_language(text)

        token_ids = self._tokenize(text)

        start = time.time()
        result = self._engine.synthesize(
            text=text,
            lang=language,
            token_ids=token_ids,
        )
        elapsed = time.time() - start

        wav_bytes = result["wav_bytes"]
        duration = result.get("duration", 0)

        meta = {
            "duration": round(duration, 3),
            "inference_time": round(elapsed, 3),
            "rtf": round(result.get("rtf", 0), 3),
            "sample_rate": self.sample_rate,
            "n_frames": result.get("n_frames", 0),
            "per_step_ms": round(result.get("per_step_ms", 0), 1),
        }
        return wav_bytes, meta

    def clone_voice(
        self,
        text: str,
        speaker_embedding: bytes,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if language is None:
            language = _detect_language(text)

        token_ids = self._tokenize(text)

        start = time.time()
        result = self._engine.synthesize_clone(
            text=text,
            lang=language,
            token_ids=token_ids,
            speaker_emb_bytes=speaker_embedding,
        )
        elapsed = time.time() - start

        wav_bytes = result["wav_bytes"]
        duration = result.get("duration", 0)

        meta = {
            "duration": round(duration, 3),
            "inference_time": round(elapsed, 3),
            "rtf": round(result.get("rtf", 0), 3),
            "sample_rate": self.sample_rate,
        }
        return wav_bytes, meta

    def generate_streaming(self, text: str, **kwargs):
        """Yield PCM int16 chunks via C++ callback-based streaming.

        The C++ engine calls our callback per chunk during generation,
        and we yield each chunk as it arrives via a thread-safe queue.

        Args:
            text: Text to synthesize
            language: Language code (auto-detected if not specified)
            speaker_embedding: Optional speaker embedding bytes for voice cloning
            first_chunk_frames: Frames in first chunk (default 10)
            chunk_frames: Frames in subsequent chunks (default 25)
            max_frames: Maximum total frames (default 200)
        """
        import queue as queue_mod
        import threading

        language = kwargs.get("language") or _detect_language(text)
        speaker_embedding = kwargs.get("speaker_embedding")
        first_chunk_frames = kwargs.get("first_chunk_frames", 5)
        chunk_frames = kwargs.get("chunk_frames", 25)
        max_frames = kwargs.get("max_frames", 200)

        token_ids = self._tokenize(text)

        chunk_queue: queue_mod.Queue = queue_mod.Queue()
        SENTINEL = object()

        def _on_chunk(chunk_dict):
            """Called from C++ thread per audio chunk."""
            wav_bytes = chunk_dict["wav_bytes"]
            if len(wav_bytes) > 44:
                chunk_queue.put(wav_bytes[44:])  # Strip WAV header -> raw PCM

        def _run_engine():
            try:
                if speaker_embedding:
                    self._engine.synthesize_streaming_clone_callback(
                        text=text,
                        lang=language,
                        token_ids=token_ids,
                        speaker_emb_bytes=speaker_embedding,
                        callback=_on_chunk,
                        first_chunk_frames=first_chunk_frames,
                        chunk_frames=chunk_frames,
                        max_frames=max_frames,
                    )
                else:
                    self._engine.synthesize_streaming_callback(
                        text=text,
                        lang=language,
                        token_ids=token_ids,
                        callback=_on_chunk,
                        first_chunk_frames=first_chunk_frames,
                        chunk_frames=chunk_frames,
                        max_frames=max_frames,
                    )
            finally:
                chunk_queue.put(SENTINEL)

        threading.Thread(target=_run_engine, daemon=True).start()

        while True:
            item = chunk_queue.get()
            if item is SENTINEL:
                break
            yield item

    def extract_speaker_embedding(self, audio_wav_bytes: bytes) -> bytes:
        """Extract speaker embedding using Python mel computation + ORT."""
        import tempfile
        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            wf.write(audio_wav_bytes)
            wav_path = wf.name
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as ef:
            emb_path = ef.name

        try:
            result = subprocess.run(
                ["python3", QWEN3_EXTRACT_SCRIPT,
                 "--audio", wav_path,
                 "--model", QWEN3_SPEAKER_ENCODER,
                 "--output", emb_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Embedding extraction failed: {result.stderr}")
            return open(emb_path, "rb").read()
        finally:
            for p in [wav_path, emb_path]:
                if os.path.exists(p):
                    os.unlink(p)
