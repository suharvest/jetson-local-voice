"""Test Qwen3StreamingASRStream buffer, window, and agreement logic.

Run: cd /Users/harvest/project/jetson-voice && python3 -m pytest tests/test_streaming_asr.py -v
"""
import sys
sys.path.insert(0, "app")

import numpy as np
import pytest
from collections import deque
from unittest.mock import MagicMock, patch

from backends.qwen3_asr import (
    Qwen3StreamingASRStream,
    SegmentInfo,
    _is_cjk,
    CHUNK_SIZE_SEC,
    MEMORY_NUM,
)


class TestCJKHelper:
    def test_cjk_chinese(self):
        assert _is_cjk("中") is True

    def test_cjk_english(self):
        assert _is_cjk("A") is False

    def test_cjk_japanese(self):
        assert _is_cjk("あ") is True


class TestLocalAgreement:
    def test_identical(self):
        result = Qwen3StreamingASRStream._local_agreement("hello", "hello")
        assert result == "hello"

    def test_common_prefix(self):
        result = Qwen3StreamingASRStream._local_agreement("hello world", "hello there")
        # Snaps to space boundary for English
        assert result == "hello "

    def test_cjk_char_boundary(self):
        result = Qwen3StreamingASRStream._local_agreement("今天天气", "今天不错")
        assert result == "今天"

    def test_empty_prev(self):
        result = Qwen3StreamingASRStream._local_agreement("", "新文本")
        assert result == "新文本"

    def test_no_common(self):
        result = Qwen3StreamingASRStream._local_agreement("abc", "xyz")
        assert result == ""


class TestBuffering:
    def _make_stream(self):
        backend = MagicMock()
        backend._tokenizer = MagicMock()
        stream = Qwen3StreamingASRStream(backend, language="auto")
        # Prevent actual encoding
        stream._process_chunk = MagicMock()
        return stream

    def test_buffer_accumulates(self):
        stream = self._make_stream()
        samples = np.zeros(8000, dtype=np.float32)  # 0.5s
        stream.accept_waveform(16000, samples)
        assert len(stream._sample_buf) == 8000
        stream._process_chunk.assert_not_called()

    def test_chunk_triggers_at_threshold(self):
        stream = self._make_stream()
        chunk_samples = int(CHUNK_SIZE_SEC * 16000)
        samples = np.zeros(chunk_samples, dtype=np.float32)
        stream.accept_waveform(16000, samples)
        stream._process_chunk.assert_called_once()
        assert len(stream._sample_buf) == 0

    def test_multiple_chunks(self):
        stream = self._make_stream()
        chunk_samples = int(CHUNK_SIZE_SEC * 16000)
        samples = np.zeros(chunk_samples * 2 + 1000, dtype=np.float32)
        stream.accept_waveform(16000, samples)
        assert stream._process_chunk.call_count == 2
        assert len(stream._sample_buf) == 1000

    def test_resampling(self):
        stream = self._make_stream()
        # 48kHz -> 16kHz, so 72000 samples -> 24000 = 1 chunk (CHUNK_SIZE_SEC=1.5s * 16000=24000)
        samples = np.zeros(72000, dtype=np.float32)
        stream.accept_waveform(48000, samples)
        stream._process_chunk.assert_called_once()


class TestEndpointDetection:
    def test_get_partial_default(self):
        backend = MagicMock()
        backend._tokenizer = MagicMock()
        stream = Qwen3StreamingASRStream(backend)
        text, is_ep = stream.get_partial()
        assert text == ""
        assert is_ep is False

    def test_eos_count_triggers_endpoint(self):
        backend = MagicMock()
        backend._tokenizer = MagicMock()
        stream = Qwen3StreamingASRStream(backend)
        stream._eos_count = 2
        stream._stable_text = "测试"
        text, is_ep = stream.get_partial()
        assert text == "测试"
        assert is_ep is True
