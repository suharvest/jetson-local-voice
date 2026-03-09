"""SenseVoice ASR service using sherpa-onnx."""

from __future__ import annotations

import glob
import logging
import os

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_recognizer = None


def _find_model_dir() -> str:
    """Locate the extracted SenseVoice model directory."""
    base = os.path.join(os.environ.get("MODEL_DIR", "/opt/models"), "sensevoice")
    # Find the extracted directory (name may vary)
    dirs = glob.glob(os.path.join(base, "sherpa-onnx-sense-voice-*"))
    if dirs:
        return dirs[0]
    return base


def get_recognizer():
    """Lazy-init the SenseVoice recognizer with CUDA."""
    global _recognizer
    if _recognizer is not None:
        return _recognizer

    import sherpa_onnx

    model_dir = _find_model_dir()
    model_path = os.path.join(model_dir, "model.int8.onnx")
    tokens_path = os.path.join(model_dir, "tokens.txt")

    logger.info("Loading SenseVoice model from %s (provider=cuda)", model_dir)
    _recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model_path,
        tokens=tokens_path,
        use_itn=True,
        provider="cuda",
    )
    logger.info("SenseVoice model loaded.")
    return _recognizer


def transcribe_audio(audio_bytes: bytes, language: str = "auto") -> str:
    """Transcribe audio bytes (WAV format) to text."""
    import io

    data, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # Convert to mono float32
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            sf.write(tmp_in.name, data, sample_rate)
            tmp_in_path = tmp_in.name

        tmp_out_path = tmp_in_path + ".16k.wav"
        try:
            subprocess.run(
                ["sox", tmp_in_path, "-r", "16000", tmp_out_path],
                check=True,
                capture_output=True,
            )
            data, sample_rate = sf.read(tmp_out_path)
            data = data.astype(np.float32)
        finally:
            for p in (tmp_in_path, tmp_out_path):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass

    recognizer = get_recognizer()
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, data)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def is_ready() -> bool:
    """Check if ASR model can be loaded."""
    try:
        get_recognizer()
        return True
    except Exception:
        return False
