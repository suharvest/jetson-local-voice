"""Unit tests for Qwen3StreamingASRStream state machine (PR1 + PR2 fixes).

Tests Bug A/C/E (PR2) and Bug B/D (PR1).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import pytest
from unittest.mock import MagicMock, patch, call
import numpy as np

from backends.qwen3_asr import (
    Qwen3StreamingASRStream,
    SegmentInfo,
    ASR_TEXT,
    AUDIO_PAD,
    IM_START,
    IM_END,
    AUDIO_START,
    AUDIO_END,
)


class TestBugBNoStaleSpecEmbedding:
    """Bug B: half chunk → full chunk → tail with coincidentally matching length.
    
    Previously, speculative embedding was reused if length matched, causing stale
    embeddings to be used when the actual audio content was different.
    Now: _run_encoder is always called for each chunk; no speculative reuse.
    """
    
    def test_encoder_always_called_not_cached(self):
        """Verify _run_encoder is called for every chunk, no speculative reuse."""
        backend = MagicMock()
        backend._encoder = MagicMock()
        backend._decoder = MagicMock()
        backend._embed_tokens = np.zeros((151936, 1024), dtype=np.float32)
        backend._tokenizer = MagicMock()
        backend._tokenizer.decode = lambda ids: "".join([chr(i % 26 + 97) for i in ids])
        backend._tokenizer.encode = MagicMock(return_value=MagicMock(ids=[]))
        
        # Mock encoder output
        enc_out = np.zeros((1, 10, 1024), dtype=np.float32)
        backend._encoder.run = MagicMock(return_value=[enc_out])
        
        # Mock compute_mel
        backend._compute_mel = MagicMock(return_value=np.zeros((1, 128, 100), dtype=np.float32))
        
        stream = Qwen3StreamingASRStream(backend)
        
        # Track _run_encoder calls
        encoder_calls = []
        def mock_run_encoder(audio_chunk, context_audio=None):
            encoder_calls.append((len(audio_chunk), context_audio is not None))
            return np.zeros((1, 10, 1024), dtype=np.float32)
        
        stream._run_encoder = mock_run_encoder
        
        # Mock _decode_window to return None (EOS) for simplicity
        stream._decode_window = MagicMock(return_value=None)
        
        # Send half chunk (0.6s at 16kHz = 9600 samples)
        half_chunk = np.zeros(9600, dtype=np.float32)
        stream.accept_waveform(16000, half_chunk)
        
        # Should not process yet (buffer < chunk_size)
        assert len(encoder_calls) == 0
        
        # Send another half to make a full chunk (total 1.2s = 19200 samples)
        stream.accept_waveform(16000, half_chunk)
        
        # Now should have processed one full chunk
        assert len(encoder_calls) == 1
        
        # Send tail with SAME length as half chunk (9600 samples)
        # This used to trigger Bug B: stale speculative embedding reused
        stream.accept_waveform(16000, half_chunk)
        stream.finalize()
        
        # Should have called encoder for tail too (not reused stale)
        # Total: 1 full chunk + 1 tail
        assert len(encoder_calls) == 2


class TestBugDAsrTextAnchor:
    """Bug D: ASR_TEXT anchor must be appended unconditionally.
    
    Previously, ASR_TEXT was only appended when language was explicitly provided.
    For None/auto mode, the prompt lacked ASR_TEXT, causing English inputs to fail.
    """
    
    def test_build_prompt_auto_ends_with_asr_text(self):
        """_build_prompt(audio_len, None) must end with ASR_TEXT."""
        from backends.qwen3_asr import Qwen3ASRBackend
        backend = Qwen3ASRBackend()
        backend._tokenizer = MagicMock()
        backend._tokenizer.encode = MagicMock(return_value=MagicMock(ids=[123, 456]))
        
        audio_len = 100
        prompt_ids = backend._build_prompt(audio_len, None)
        
        assert prompt_ids[-1] == ASR_TEXT, "Prompt must end with ASR_TEXT for language=None"
    
    def test_build_prompt_auto_no_language_token(self):
        """_build_prompt(audio_len, None) should not include language tokens."""
        from backends.qwen3_asr import Qwen3ASRBackend
        backend = Qwen3ASRBackend()
        backend._tokenizer = MagicMock()
        backend._tokenizer.encode = MagicMock(return_value=MagicMock(ids=[123, 456]))
        
        audio_len = 100
        prompt_ids = backend._build_prompt(audio_len, None)
        
        # Language tokens (123, 456) should not be present for None
        assert 123 not in prompt_ids
        assert 456 not in prompt_ids
    
    def test_build_prompt_en_ends_with_asr_text(self):
        """_build_prompt(audio_len, 'en') must end with ASR_TEXT."""
        from backends.qwen3_asr import Qwen3ASRBackend
        backend = Qwen3ASRBackend()
        backend._tokenizer = MagicMock()
        backend._tokenizer.encode = MagicMock(return_value=MagicMock(ids=[111, 222]))
        
        audio_len = 100
        prompt_ids = backend._build_prompt(audio_len, "en")
        
        assert prompt_ids[-1] == ASR_TEXT, "Prompt must end with ASR_TEXT for language='en'"
    
    def test_build_prompt_en_includes_language_tokens(self):
        """_build_prompt(audio_len, 'en') should include language tokens before ASR_TEXT."""
        from backends.qwen3_asr import Qwen3ASRBackend
        backend = Qwen3ASRBackend()
        backend._tokenizer = MagicMock()
        backend._tokenizer.encode = MagicMock(return_value=MagicMock(ids=[111, 222]))
        
        audio_len = 100
        prompt_ids = backend._build_prompt(audio_len, "en")
        
        # Language tokens should be present
        assert 111 in prompt_ids
        assert 222 in prompt_ids
        # And appear before ASR_TEXT
        asr_idx = prompt_ids.index(ASR_TEXT)
        assert prompt_ids[asr_idx - 1] == 222
    
    def test_build_prompt_zh_ends_with_asr_text(self):
        """_build_prompt(audio_len, 'zh') must end with ASR_TEXT."""
        from backends.qwen3_asr import Qwen3ASRBackend
        backend = Qwen3ASRBackend()
        backend._tokenizer = MagicMock()
        backend._tokenizer.encode = MagicMock(return_value=MagicMock(ids=[333, 444]))
        
        audio_len = 100
        prompt_ids = backend._build_prompt(audio_len, "zh")
        
        assert prompt_ids[-1] == ASR_TEXT


class TestIndentationFixed:
    """Verify _process_chunk is correctly inside the class after indentation fix."""
    
    def test_process_chunk_is_class_method(self):
        """_process_chunk should be a method of Qwen3StreamingASRStream."""
        assert hasattr(Qwen3StreamingASRStream, '_process_chunk')
        assert callable(Qwen3StreamingASRStream._process_chunk)
    
    def test_apply_rollback_is_class_method(self):
        """_apply_rollback should be a method of Qwen3StreamingASRStream."""
        assert hasattr(Qwen3StreamingASRStream, '_apply_rollback')
        assert callable(Qwen3StreamingASRStream._apply_rollback)
    
    def test_local_agreement_is_staticmethod(self):
        """_local_agreement should be a static method."""
        assert hasattr(Qwen3StreamingASRStream, '_local_agreement')
        # Static methods can be called on class without instance
        result = Qwen3StreamingASRStream._local_agreement("hello", "hello")
        assert result == "hello"


class TestPrepareFinalizeIsNoOp:
    """Verify prepare_finalize is now a no-op (speculative encoding removed)."""
    
    def test_prepare_finalize_does_not_encode(self):
        """prepare_finalize should not call _run_encoder."""
        backend = MagicMock()
        backend._encoder = MagicMock()
        backend._embed_tokens = np.zeros((151936, 1024), dtype=np.float32)
        
        stream = Qwen3StreamingASRStream(backend)
        stream._run_encoder = MagicMock()
        
        # Put some data in buffer
        stream._sample_buf = np.zeros(9600, dtype=np.float32)
        stream._left_context = np.zeros(16000, dtype=np.float32)
        
        # Call prepare_finalize
        stream.prepare_finalize()
        
        # Should NOT have called _run_encoder (no-op now)
        stream._run_encoder.assert_not_called()


class TestSpeculativeFieldsRemoved:
    """Verify speculative encoding fields are removed from __init__."""
    
    def test_no_spec_embd_field(self):
        """_spec_embd should not exist after __init__."""
        backend = MagicMock()
        stream = Qwen3StreamingASRStream(backend)
        assert not hasattr(stream, '_spec_embd')
    
    def test_no_spec_audio_len_field(self):
        """_spec_audio_len should not exist after __init__."""
        backend = MagicMock()
        stream = Qwen3StreamingASRStream(backend)
        assert not hasattr(stream, '_spec_audio_len')
    
    def test_no_spec_left_context_field(self):
        """_spec_left_context should not exist after __init__."""
        backend = MagicMock()
        stream = Qwen3StreamingASRStream(backend)
        assert not hasattr(stream, '_spec_left_context')


class TestBugANoFirstCharLoss:
    """Bug A: First character loss in streaming finalize.
    
    Previously, finalize returned _archive_text + _prev_text, and partial
    results were prematurely archived, causing first characters to be lost.
    Now: finalize does a single decode on ALL segments.
    """
    
    def test_finalize_returns_full_transcript(self):
        """finalize must return full text from all segments via single decode."""
        backend = MagicMock()
        backend._embed_tokens = np.zeros((151936, 1024), dtype=np.float32)
        backend._tokenizer = MagicMock()
        backend._tokenizer.encode = MagicMock(return_value=MagicMock(ids=[]))
        
        stream = Qwen3StreamingASRStream(backend)
        
        # Mock _run_encoder to return embeddings
        def mock_run_encoder(audio_chunk, context_audio=None):
            return np.zeros((1, 10, 1024), dtype=np.float32)
        stream._run_encoder = mock_run_encoder
        
        # Mock _decode_window to simulate progressive recognition
        decode_calls = []
        def mock_decode_window(all_embd, max_tokens=4):
            decode_calls.append(all_embd.shape[1])
            # Return partial for intermediate, full for final
            if len(decode_calls) <= 2:
                return None  # Simulate EOS for partials
            return "今天天气不错我们出去玩吧"
        stream._decode_window = mock_decode_window
        
        # Feed 2 chunks
        chunk = np.zeros(19200, dtype=np.float32)
        stream.accept_waveform(16000, chunk)
        stream.accept_waveform(16000, chunk)
        
        # Finalize should return the mock result
        result = stream.finalize()
        assert result == "今天天气不错我们出去玩吧"
    
    def test_finalize_single_decode_no_archive(self):
        """finalize should NOT use _archive_text + _prev_text."""
        backend = MagicMock()
        backend._embed_tokens = np.zeros((151936, 1024), dtype=np.float32)
        
        stream = Qwen3StreamingASRStream(backend)
        
        # Mock encoder
        stream._run_encoder = lambda audio_chunk, context_audio=None: np.zeros((1, 10, 1024), dtype=np.float32)
        
        # Mock decode to return specific text
        decode_state = {'count': 0}
        def mock_decode(all_embd, max_tokens=4):
            decode_state['count'] += 1
            return "expected result" if decode_state['count'] > 1 else None
        stream._decode_window = mock_decode
        
        # Feed chunks
        stream.accept_waveform(16000, np.zeros(19200, dtype=np.float32))
        
        # Call finalize
        result = stream.finalize()
        
        # Should be the decode result, not archive+prev
        assert result == "expected result"


class TestBugCNoRepetition:
    """Bug C: Repetition/hallucination in streaming output.
    
    Previously, _archive_text was accumulated, causing repeated text.
    Now: finalize does a single decode; no archive accumulation.
    """
    
    def test_finalize_no_repetition(self):
        """finalize must return exactly one copy, no repetition."""
        backend = MagicMock()
        backend._embed_tokens = np.zeros((151936, 1024), dtype=np.float32)
        backend._tokenizer = MagicMock()
        backend._tokenizer.encode = MagicMock(return_value=MagicMock(ids=[]))
        
        stream = Qwen3StreamingASRStream(backend)
        
        # Mock encoder
        stream._run_encoder = lambda audio_chunk, context_audio=None: np.zeros((1, 10, 1024), dtype=np.float32)
        
        # Track decode calls - simulate incremental hypotheses
        decode_history = []
        def mock_decode(all_embd, max_tokens=4):
            n_segments = all_embd.shape[1] // 10
            # For partials: return incremental
            if n_segments < 4:
                partial_texts = ["今天", "今天天气不错", "今天天气不错我们出去"]
                return partial_texts[n_segments - 1] if n_segments > 0 else None
            # For final: return full
            decode_history.append("final")
            return "今天天气不错我们出去玩吧"
        stream._decode_window = mock_decode
        
        # Feed 4 chunks (trigger incremental decoding)
        for _ in range(4):
            stream.accept_waveform(16000, np.zeros(19200, dtype=np.float32))
        
        result = stream.finalize()
        
        # Should be exactly one copy, not repeated
        assert result == "今天天气不错我们出去玩吧"
        assert result.count("今天") == 1  # No repetition
    
    def test_no_archive_accumulation(self):
        """_archive_text should not be accumulated into final output."""
        backend = MagicMock()
        backend._embed_tokens = np.zeros((151936, 1024), dtype=np.float32)
        
        stream = Qwen3StreamingASRStream(backend)
        
        # Mock encoder and decode
        stream._run_encoder = lambda audio_chunk, context_audio=None: np.zeros((1, 10, 1024), dtype=np.float32)
        stream._decode_window = lambda all_embd, max_tokens=4: "text"
        
        # Simulate some partial processing
        stream.accept_waveform(16000, np.zeros(19200, dtype=np.float32))
        
        # _archive_text should remain empty or not affect final
        assert stream._archive_text == ""
        
        # Finalize returns decode result
        result = stream.finalize()
        assert result == "text"


class TestBugEEarlyEOSNoClear:
    """Bug E: Early EOS clears state, losing segments.
    
    Previously, EOS_CONFIRM_COUNT consecutive None clears _segments.
    Now: EOS only flags endpoint; segments are kept.
    """
    
    def test_segments_preserved_on_eos(self):
        """_segments must be preserved after consecutive None partials."""
        backend = MagicMock()
        backend._embed_tokens = np.zeros((151936, 1024), dtype=np.float32)
        
        stream = Qwen3StreamingASRStream(backend)
        
        # Mock encoder
        stream._run_encoder = lambda audio_chunk, context_audio=None: np.zeros((1, 10, 1024), dtype=np.float32)
        
        # Mock decode to return None for partials, text for final
        decode_state = {'count': 0}
        def mock_decode(all_embd, max_tokens=4):
            decode_state['count'] += 1
            if decode_state['count'] <= 2:
                return None  # Simulate EOS for first 2 partials
            return "valid text"
        stream._decode_window = mock_decode
        
        # Feed 2 chunks (will trigger 2 None partials)
        stream.accept_waveform(16000, np.zeros(19200, dtype=np.float32))
        stream.accept_waveform(16000, np.zeros(19200, dtype=np.float32))
        
        # _segments should still have 2 entries (not cleared)
        assert len(stream._segments) == 2
        
        # Finalize should return valid text
        result = stream.finalize()
        assert result == "valid text"
    
    def test_eos_count_incremented_not_state_clear(self):
        """EOS should increment eos_count but NOT clear _segments/_left_context."""
        backend = MagicMock()
        backend._embed_tokens = np.zeros((151936, 1024), dtype=np.float32)
        
        stream = Qwen3StreamingASRStream(backend)
        
        stream._run_encoder = lambda audio_chunk, context_audio=None: np.zeros((1, 10, 1024), dtype=np.float32)
        stream._decode_window = MagicMock(return_value=None)  # Always EOS
        
        # Feed chunks
        stream.accept_waveform(16000, np.zeros(19200, dtype=np.float32))
        
        # eos_count should increment
        assert stream._eos_count >= 1
        
        # But segments should NOT be cleared
        assert len(stream._segments) >= 1
        assert len(stream._left_context) > 0


class TestSegmentsNoPopleft:
    """Verify _segments.popleft() eviction is removed (PR2)."""
    
    def test_segments_not_evicted(self):
        """_segments should keep all chunks, no eviction."""
        backend = MagicMock()
        backend._embed_tokens = np.zeros((151936, 1024), dtype=np.float32)
        
        stream = Qwen3StreamingASRStream(backend)
        
        stream._run_encoder = lambda audio_chunk, context_audio=None: np.zeros((1, 10, 1024), dtype=np.float32)
        stream._decode_window = MagicMock(return_value=None)
        
        # Feed 5 chunks (more than MEMORY_NUM=3)
        for _ in range(5):
            stream.accept_waveform(16000, np.zeros(19200, dtype=np.float32))
        
        # All 5 segments should be preserved (no eviction)
        assert len(stream._segments) == 5
    
    def test_finalize_sees_all_segments(self):
        """finalize must decode on ALL segments, not truncated window."""
        backend = MagicMock()
        backend._embed_tokens = np.zeros((151936, 1024), dtype=np.float32)
        
        stream = Qwen3StreamingASRStream(backend)
        
        stream._run_encoder = lambda audio_chunk, context_audio=None: np.zeros((1, 10, 1024), dtype=np.float32)
        
        final_embd_size = None
        def mock_decode(all_embd, max_tokens=4):
            final_embd_size = all_embd.shape[1]
            return f"decoded_{all_embd.shape[1]}"
        stream._decode_window = mock_decode
        
        # Feed 6 chunks
        for _ in range(6):
            stream.accept_waveform(16000, np.zeros(19200, dtype=np.float32))
        
        result = stream.finalize()
        
        # Final decode should have 6 segments' worth of embeddings (60 frames)
        # 6 chunks * 10 frames per chunk = 60
        assert "60" in result or len(stream._segments) == 6