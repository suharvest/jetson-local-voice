"""Qwen3-TTS backend via C++ TRT native engine.

Supports: BASIC_TTS, VOICE_CLONE, MULTI_LANGUAGE
Requires: compiled qwen3_tts binary + TRT engines + ONNX models
"""

from __future__ import annotations

import io
import logging
import os
import struct
import subprocess
import tempfile
import time
from typing import Optional

import numpy as np

from tts_backend import TTSBackend, TTSCapability

logger = logging.getLogger(__name__)

# Paths (configurable via env)
QWEN3_BINARY = os.environ.get("QWEN3_TTS_BINARY", "/tmp/qwen3_tts")
QWEN3_SHERPA_DIR = os.environ.get("QWEN3_SHERPA_DIR", "/tmp/qwen3-v2")
QWEN3_MODEL_DIR = os.environ.get("QWEN3_MODEL_DIR", "/tmp/qwen3-v2")
QWEN3_TALKER_ENGINE = os.environ.get("QWEN3_TALKER_ENGINE", "/tmp/talker_decode_fp16.engine")
QWEN3_CP_ENGINE = os.environ.get("QWEN3_CP_ENGINE", "/tmp/cp_bf16.engine")
QWEN3_SPEAKER_ENCODER = os.environ.get("QWEN3_SPEAKER_ENCODER", "/tmp/qwen3-v2/speaker_encoder.onnx")
QWEN3_EXTRACT_SCRIPT = os.environ.get("QWEN3_EXTRACT_SCRIPT", "/tmp/extract_speaker_emb.py")
QWEN3_TOKENIZER_DIR = os.environ.get("QWEN3_TOKENIZER_DIR", "/tmp/qwen3-tts-bench/model/tokenizer")


class Qwen3TRTBackend(TTSBackend):
    """Qwen3-TTS via C++ TRT native inference."""

    def __init__(self):
        self._ready = False
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "qwen3_trt"

    @property
    def capabilities(self) -> set[TTSCapability]:
        caps = {TTSCapability.BASIC_TTS, TTSCapability.MULTI_LANGUAGE}
        if os.path.exists(QWEN3_SPEAKER_ENCODER):
            caps.add(TTSCapability.VOICE_CLONE)
        return caps

    @property
    def sample_rate(self) -> int:
        return 24000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        # Verify binary and engines exist
        missing = []
        for path, desc in [
            (QWEN3_BINARY, "C++ binary"),
            (QWEN3_TALKER_ENGINE, "talker engine"),
            (QWEN3_CP_ENGINE, "CP engine"),
            (QWEN3_SHERPA_DIR + "/config.json", "config.json"),
        ]:
            if not os.path.exists(path):
                missing.append(f"{desc}: {path}")
        if missing:
            raise FileNotFoundError(
                f"Qwen3 TRT backend missing files:\n" + "\n".join(missing)
            )

        # Load tokenizer
        self._load_tokenizer()

        # Warmup: run one short synthesis
        logger.info("Qwen3 TRT warmup...")
        try:
            self.synthesize("OK", language="english")
            logger.info("Qwen3 TRT backend ready.")
        except Exception as e:
            logger.warning("Warmup failed (non-fatal): %s", e)

        self._ready = True

    def _load_tokenizer(self):
        vocab_path = os.path.join(QWEN3_TOKENIZER_DIR, "vocab.json")
        merges_path = os.path.join(QWEN3_TOKENIZER_DIR, "merges.txt")
        if os.path.exists(vocab_path):
            try:
                from tokenizers import Tokenizer
                from tokenizers.models import BPE
                from tokenizers.pre_tokenizers import ByteLevel

                self._tokenizer = Tokenizer(BPE(vocab_path, merges_path))
                self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
                logger.info("Tokenizer loaded from %s", QWEN3_TOKENIZER_DIR)
            except Exception as e:
                logger.warning("Failed to load tokenizer: %s", e)
        else:
            logger.warning("Tokenizer not found at %s", QWEN3_TOKENIZER_DIR)

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
            language = "english"

        token_ids = self._tokenize(text)
        token_ids_str = ",".join(str(i) for i in token_ids)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            start = time.time()
            self._run_binary(
                token_ids_str=token_ids_str,
                language=language,
                output=wav_path,
                text=text,
            )
            elapsed = time.time() - start

            wav_bytes = open(wav_path, "rb").read()
            duration = self._wav_duration(wav_bytes)

            meta = {
                "duration": round(duration, 3),
                "inference_time": round(elapsed, 3),
                "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
                "sample_rate": self.sample_rate,
            }
            return wav_bytes, meta
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    def clone_voice(
        self,
        text: str,
        speaker_embedding: bytes,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if language is None:
            language = "english"

        token_ids = self._tokenize(text)
        token_ids_str = ",".join(str(i) for i in token_ids)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            wav_path = wf.name
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as ef:
            ef.write(speaker_embedding)
            emb_path = ef.name

        try:
            start = time.time()
            self._run_binary(
                token_ids_str=token_ids_str,
                language=language,
                output=wav_path,
                text=text,
                speaker_emb=emb_path,
            )
            elapsed = time.time() - start

            wav_bytes = open(wav_path, "rb").read()
            duration = self._wav_duration(wav_bytes)

            meta = {
                "duration": round(duration, 3),
                "inference_time": round(elapsed, 3),
                "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
                "sample_rate": self.sample_rate,
            }
            return wav_bytes, meta
        finally:
            for p in [wav_path, emb_path]:
                if os.path.exists(p):
                    os.unlink(p)

    def extract_speaker_embedding(self, audio_wav_bytes: bytes) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            wf.write(audio_wav_bytes)
            wav_path = wf.name
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as ef:
            emb_path = ef.name

        try:
            result = subprocess.run(
                [
                    "python3", QWEN3_EXTRACT_SCRIPT,
                    "--audio", wav_path,
                    "--model", QWEN3_SPEAKER_ENCODER,
                    "--output", emb_path,
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Speaker embedding extraction failed: {result.stderr}")

            return open(emb_path, "rb").read()
        finally:
            for p in [wav_path, emb_path]:
                if os.path.exists(p):
                    os.unlink(p)

    def _run_binary(self, token_ids_str: str, language: str, output: str,
                    text: str = "", speaker_emb: Optional[str] = None):
        cmd = [
            QWEN3_BINARY,
            "--sherpa-dir", QWEN3_SHERPA_DIR,
            "--model-dir", QWEN3_MODEL_DIR,
            "--talker-engine", QWEN3_TALKER_ENGINE,
            "--cp-engine", QWEN3_CP_ENGINE,
            "--token-ids", token_ids_str,
            "--lang", language,
            "--output", output,
            "--text", text,
        ]
        if speaker_emb:
            cmd += ["--speaker-emb", speaker_emb]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"qwen3_tts failed: {result.stderr[-500:]}")

    @staticmethod
    def _wav_duration(wav_bytes: bytes) -> float:
        try:
            with io.BytesIO(wav_bytes) as bio:
                import wave
                with wave.open(bio) as w:
                    return w.getnframes() / w.getframerate()
        except Exception:
            return 0.0
