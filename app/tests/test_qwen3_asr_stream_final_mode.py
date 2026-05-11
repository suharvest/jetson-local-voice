import os
import sys
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backends.jetson.qwen3_asr import Qwen3StreamingASRStream


class FakeTokenizer:
    def decode(self, ids):
        return "<asr_text>" + "".join(str(i) for i in ids)


class FakeBackend:
    def __init__(self):
        self._tokenizer = FakeTokenizer()
        self._embed_tokens = None
        self._decoder = None
        self._decoder_ort = None
        self.transcribe_audio = MagicMock(return_value=MagicMock(text="offline"))


def test_stream_final_mode_defaults_to_offline(monkeypatch):
    monkeypatch.delenv("QWEN3_ASR_STREAM_FINAL_MODE", raising=False)
    backend = FakeBackend()
    stream = Qwen3StreamingASRStream(backend)
    stream._utterance_audio_buffer = [np.zeros(160, dtype=np.float32)]
    stream._encoder_frames = [np.zeros((1, 2, 1024), dtype=np.float32)]
    stream._decode_final = MagicMock(return_value=[1, 2])

    assert stream._final_text() == "offline"
    backend.transcribe_audio.assert_called_once()


def test_stream_final_mode_reuses_encoder_frames(monkeypatch):
    monkeypatch.setenv("QWEN3_ASR_STREAM_FINAL_MODE", "reuse")
    backend = FakeBackend()
    stream = Qwen3StreamingASRStream(backend)
    stream._utterance_audio_buffer = [np.zeros(160, dtype=np.float32)]
    stream._encoder_frames = [np.zeros((1, 2, 1024), dtype=np.float32)]
    stream._decode_final = MagicMock(return_value=[1, 2])

    assert stream._final_text() == "12"
    backend.transcribe_audio.assert_not_called()


def test_stream_final_mode_falls_back_when_reuse_empty(monkeypatch):
    monkeypatch.setenv("QWEN3_ASR_STREAM_FINAL_MODE", "reuse")
    backend = FakeBackend()
    stream = Qwen3StreamingASRStream(backend)
    stream._utterance_audio_buffer = [np.zeros(160, dtype=np.float32)]
    stream._encoder_frames = []

    assert stream._final_text() == "offline"
    backend.transcribe_audio.assert_called_once()
