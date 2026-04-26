"""Unit tests for _dedup_boundary_tokens in Qwen3StreamingASRStream."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends.qwen3_asr import Qwen3StreamingASRStream, DEDUP_MAX_OVERLAP


class MockBackend:
    """Minimal mock to instantiate Qwen3StreamingASRStream."""
    def __init__(self):
        self._tokenizer = None
        self._embed_tokens = None
        self._decoder = None
        self._decoder_ort = None

    def _build_prompt(self, audio_len, language=None):
        return []

    def _compute_mel(self, audio):
        return None


def test_dedup_empty_archive():
    """Empty archive should return all new tokens."""
    stream = Qwen3StreamingASRStream(MockBackend())
    result = stream._dedup_boundary_tokens([], [1, 2, 3])
    assert result == [1, 2, 3], f"Expected [1,2,3], got {result}"
    print("PASS: test_dedup_empty_archive")


def test_dedup_empty_new():
    """Empty new should return empty."""
    stream = Qwen3StreamingASRStream(MockBackend())
    result = stream._dedup_boundary_tokens([1, 2, 3], [])
    assert result == [], f"Expected [], got {result}"
    print("PASS: test_dedup_empty_new")


def test_dedup_full_overlap():
    """Full overlap: archive suffix == new prefix."""
    stream = Qwen3StreamingASRStream(MockBackend())
    # archive=['你好'], new=['你好','世界'] -> should return ['世界']
    result = stream._dedup_boundary_tokens([100, 101], [100, 101, 102, 103])
    assert result == [102, 103], f"Expected [102,103], got {result}"
    print("PASS: test_dedup_full_overlap")


def test_dedup_partial_overlap():
    """Partial overlap at boundary."""
    stream = Qwen3StreamingASRStream(MockBackend())
    # archive=['a','b','c'], new=['b','c','d','e'] -> should return ['d','e']
    result = stream._dedup_boundary_tokens([1, 2, 3], [2, 3, 4, 5])
    assert result == [4, 5], f"Expected [4,5], got {result}"
    print("PASS: test_dedup_partial_overlap")


def test_dedup_no_overlap():
    """No overlap should return all new tokens."""
    stream = Qwen3StreamingASRStream(MockBackend())
    result = stream._dedup_boundary_tokens([1, 2, 3], [4, 5, 6])
    assert result == [4, 5, 6], f"Expected [4,5,6], got {result}"
    print("PASS: test_dedup_no_overlap")


def test_dedup_single_token_overlap():
    """Single token overlap at boundary."""
    stream = Qwen3StreamingASRStream(MockBackend())
    # archive ends with [X], new starts with [X] -> should clip X
    result = stream._dedup_boundary_tokens([1, 2, 99], [99, 3, 4])
    assert result == [3, 4], f"Expected [3,4], got {result}"
    print("PASS: test_dedup_single_token_overlap")


def test_dedup_respects_max_overlap():
    """Overlap beyond max_overlap limit should not be deduped."""
    stream = Qwen3StreamingASRStream(MockBackend())
    # archive has 15 tokens, new has 15 tokens, all new == archive[-15:]
    # But max_overlap=12, so only check up to 12 tokens
    archive = list(range(100, 115))  # 15 tokens
    new = list(range(103, 118))      # archive[-12:] == new[:12] = [103..114]
    # With max_overlap=12, should find overlap of 12 and return [115,116,117]
    result = stream._dedup_boundary_tokens(archive, new, max_overlap=12)
    # archive[-12:] = [103..114], new[:12] = [103..114] -> match!
    assert result == [115, 116, 117], f"Expected [115,116,117], got {result}"
    print("PASS: test_dedup_respects_max_overlap")


def test_dedup_longest_match_priority():
    """Should find longest overlap, not just any overlap."""
    stream = Qwen3StreamingASRStream(MockBackend())
    # archive = [A, B, C, D, E]
    # new = [D, E, F] -> overlap at k=2 (D,E)
    # Should NOT match k=1 (E vs D)
    archive = [10, 20, 30, 40, 50]
    new = [40, 50, 60]
    result = stream._dedup_boundary_tokens(archive, new)
    assert result == [60], f"Expected [60] (k=2 overlap), got {result}"
    print("PASS: test_dedup_longest_match_priority")


def test_dedup_first_utterance():
    """First utterance: archive empty -> return all new."""
    stream = Qwen3StreamingASRStream(MockBackend())
    # Simulate first utterance: committed_token_ids is empty
    committed = []
    final_ids = [1000, 1001, 1002, 1003]  # tokens for "你好，世界。"
    result = stream._dedup_boundary_tokens(committed, final_ids)
    assert result == final_ids, f"First utterance should return all tokens, got {result}"
    print("PASS: test_dedup_first_utterance")


def test_dedup_second_utterance_with_overlap():
    """Second utterance: should dedup if overlap exists."""
    stream = Qwen3StreamingASRStream(MockBackend())
    # First utterance committed: "你好" -> tokens [100, 101]
    committed = [100, 101]
    # Second utterance decode: might include carryover -> [101, 200, 201]
    # If encoder context carryover causes first token to match last token of prev utterance
    final_ids = [101, 200, 201]
    result = stream._dedup_boundary_tokens(committed, final_ids)
    # Should dedup the overlapping 101
    assert result == [200, 201], f"Expected [200,201], got {result}"
    print("PASS: test_dedup_second_utterance_with_overlap")


if __name__ == "__main__":
    test_dedup_empty_archive()
    test_dedup_empty_new()
    test_dedup_full_overlap()
    test_dedup_partial_overlap()
    test_dedup_no_overlap()
    test_dedup_single_token_overlap()
    test_dedup_respects_max_overlap()
    test_dedup_longest_match_priority()
    test_dedup_first_utterance()
    test_dedup_second_utterance_with_overlap()
    print("\nAll tests passed!")