import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.asr_backend import ASRCapability
from app.backends.jetson.trt_edge_llm_asr import (
    TRTEdgeLLMASRBackend,
    _float_audio_to_wav_bytes,
)


def test_float_audio_to_wav_bytes_roundtrip_header():
    wav_bytes = _float_audio_to_wav_bytes(np.zeros(16000, dtype=np.float32), 16000)

    assert wav_bytes[:4] == b"RIFF"
    assert b"WAVE" in wav_bytes[:16]


def test_trt_edgellm_asr_stream_accumulates_and_finalizes(monkeypatch):
    backend = TRTEdgeLLMASRBackend()
    backend._ready = True
    calls = []

    def fake_transcribe(wav_bytes, language="auto"):
        calls.append((wav_bytes, language))
        return type("Result", (), {"text": "你好"})()

    monkeypatch.setattr(backend, "transcribe", fake_transcribe)
    stream = backend.create_stream(language="Chinese")
    stream.accept_waveform(16000, np.zeros(8000, dtype=np.float32))
    stream.accept_waveform(16000, np.zeros(8000, dtype=np.float32))

    assert stream.get_partial() == ("", False)
    assert stream.finalize() == "你好"
    assert calls[0][1] == "Chinese"
    assert calls[0][0][:4] == b"RIFF"


def test_trt_edgellm_asr_advertises_streaming_capability():
    backend = TRTEdgeLLMASRBackend()

    assert ASRCapability.STREAMING in backend.capabilities
