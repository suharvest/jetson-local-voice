"""Unit tests for SherpaASRStream — safety net before refactoring."""

from unittest.mock import MagicMock, call
import numpy as np
import pytest

from backends.sherpa_asr import SherpaASRStream


def _make_svc(feed_returns=None):
    """Build a mock svc with sensible defaults."""
    svc = MagicMock()
    svc.create_stream.return_value = MagicMock(name="stream")
    if feed_returns is not None:
        svc.feed_and_decode.side_effect = feed_returns
    else:
        svc.feed_and_decode.return_value = ("", False)
    return svc


def _dummy_samples(n=160):
    return np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Partial text is surfaced after feeding audio
# ---------------------------------------------------------------------------

def test_partial_text_returned():
    svc = _make_svc()
    svc.feed_and_decode.return_value = ("hello world", False)

    stream = SherpaASRStream(svc)
    stream.accept_waveform(16000, _dummy_samples())

    text, is_endpoint = stream.get_partial()
    assert text == "hello world"
    assert is_endpoint is False


# ---------------------------------------------------------------------------
# 2. Endpoint detection triggers create_stream (stream reset)
# ---------------------------------------------------------------------------

def test_endpoint_detected_and_stream_reset():
    svc = _make_svc()
    svc.feed_and_decode.return_value = ("done", True)

    stream = SherpaASRStream(svc)
    # create_stream called once during __init__
    assert svc.create_stream.call_count == 1

    stream.accept_waveform(16000, _dummy_samples())

    # should have been called a second time to reset the stream
    assert svc.create_stream.call_count == 2


# ---------------------------------------------------------------------------
# 3. Endpoint flag clears after reading via get_partial
# ---------------------------------------------------------------------------

def test_endpoint_clears_on_next_get_partial():
    svc = _make_svc()
    svc.feed_and_decode.return_value = ("sentence", True)

    stream = SherpaASRStream(svc)
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
# 4. finalize delegates to svc.finalize with the current stream object
# ---------------------------------------------------------------------------

def test_finalize_delegates():
    svc = _make_svc()
    inner_stream = MagicMock(name="inner_stream")
    svc.create_stream.return_value = inner_stream
    svc.finalize.return_value = "final text"

    stream = SherpaASRStream(svc)
    result = stream.finalize()

    svc.finalize.assert_called_once_with(inner_stream)
    assert result == "final text"


# ---------------------------------------------------------------------------
# 5. get_partial returns empty string before any waveform is fed
# ---------------------------------------------------------------------------

def test_no_text_before_any_waveform():
    svc = _make_svc()

    stream = SherpaASRStream(svc)
    text, is_endpoint = stream.get_partial()

    assert text == ""
    assert is_endpoint is False
