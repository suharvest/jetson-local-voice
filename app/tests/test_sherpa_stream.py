"""Unit tests for SherpaASRStream — safety net before refactoring."""

from unittest.mock import MagicMock, call
import numpy as np
import pytest

from backends.sherpa_asr import SherpaASRStream


def _make_recognizer(text="", is_endpoint=False):
    """Build a mock sherpa_onnx OnlineRecognizer with sensible defaults."""
    recognizer = MagicMock()
    recognizer.create_stream.return_value = MagicMock(name="stream")
    recognizer.is_ready.return_value = False  # decode loop won't run
    recognizer.get_result.return_value = text
    recognizer.is_endpoint.return_value = is_endpoint
    recognizer.decode_stream.return_value = None
    return recognizer


def _dummy_samples(n=160):
    return np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Partial text is surfaced after feeding audio
# ---------------------------------------------------------------------------

def test_partial_text_returned():
    recognizer = _make_recognizer(text="hello world", is_endpoint=False)

    stream = SherpaASRStream(recognizer)
    stream.accept_waveform(16000, _dummy_samples())

    text, is_endpoint = stream.get_partial()
    assert text == "hello world"
    assert is_endpoint is False


# ---------------------------------------------------------------------------
# 2. Endpoint detection triggers create_stream (stream reset)
# ---------------------------------------------------------------------------

def test_endpoint_detected_and_stream_reset():
    recognizer = _make_recognizer(text="done", is_endpoint=True)

    stream = SherpaASRStream(recognizer)
    # create_stream called once during __init__
    assert recognizer.create_stream.call_count == 1

    stream.accept_waveform(16000, _dummy_samples())

    # should have been called a second time to reset the stream
    assert recognizer.create_stream.call_count == 2


# ---------------------------------------------------------------------------
# 3. Endpoint flag clears after reading via get_partial
# ---------------------------------------------------------------------------

def test_endpoint_clears_on_next_get_partial():
    recognizer = _make_recognizer(text="sentence", is_endpoint=True)

    stream = SherpaASRStream(recognizer)
    stream.accept_waveform(16000, _dummy_samples())

    # First read — should report endpoint and clear it
    text1, ep1 = stream.get_partial()
    assert text1 == "sentence"
    assert ep1 is True

    # Second read — flag should be cleared, text should be empty
    text2, ep2 = stream.get_partial()
    assert text2 == ""
    assert ep2 is False


# ---------------------------------------------------------------------------
# 4. finalize uses the current inner stream object
# ---------------------------------------------------------------------------

def test_finalize_delegates():
    recognizer = _make_recognizer(text="final text", is_endpoint=False)
    inner_stream = MagicMock(name="inner_stream")
    recognizer.create_stream.return_value = inner_stream

    stream = SherpaASRStream(recognizer)
    result = stream.finalize()

    # input_finished must be called on the inner stream
    inner_stream.input_finished.assert_called_once()
    assert result == "final text"


# ---------------------------------------------------------------------------
# 5. get_partial returns empty string before any waveform is fed
# ---------------------------------------------------------------------------

def test_no_text_before_any_waveform():
    recognizer = _make_recognizer()

    stream = SherpaASRStream(recognizer)
    text, is_endpoint = stream.get_partial()

    assert text == ""
    assert is_endpoint is False
