"""Unit tests for Qwen3StreamingASRStream._local_agreement and _is_cjk."""

import sys
import os

# Ensure app/ is on path (also set in conftest.py, but explicit here for clarity)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import only the pure-Python pieces — no ONNX/GPU deps required
from backends.qwen3_asr import _is_cjk, Qwen3StreamingASRStream

_la = Qwen3StreamingASRStream._local_agreement


class TestLocalAgreement:
    def test_identical_strings(self):
        assert _la("hello", "hello") == "hello"

    def test_common_prefix(self):
        # Diverges at index 6 ('w' vs 't'); last space before index 6 is at index 5
        # result before snap = "hello " (indices 0-5 inclusive → "hello ")
        # prev char at index 5 is ' ' which is not CJK, but i==6 < len("hello there")
        # rfind(" ") on "hello " → 5, result = "hello " (5+1=6)
        assert _la("hello world", "hello there") == "hello "

    def test_no_common_prefix(self):
        assert _la("abc", "xyz") == ""

    def test_empty_prev_returns_curr(self):
        assert _la("", "new text") == "new text"

    def test_cjk_character_boundary(self):
        # "你好世界" vs "你好地球": common prefix is "你好" (indices 0-1)
        # Diverges at index 2; prev char is '好' which IS CJK → no space snap
        assert _la("你好世界", "你好地球") == "你好"

    def test_full_prefix_match_no_snap(self):
        # When i == len(curr), condition `i < len(curr)` is False → no snap
        assert _la("hello world", "hello") == "hello"

    def test_snap_to_word_boundary(self):
        # prev="foo bar baz", curr="foo bar qux"
        # Diverges at index 8 ('b' vs 'q'); result[:8] = "foo bar "
        # prev char at 7 is ' ', not CJK → rfind(" ") on "foo bar " → 7, result = "foo bar "
        assert _la("foo bar baz", "foo bar qux") == "foo bar "

    def test_no_space_in_prefix_returns_empty(self):
        # prev="abcdef", curr="abcxyz": diverge at 3, result[:3]="abc"
        # prev char 'c' not CJK, rfind(" ") on "abc" → -1 (not > 0) → no snap, return "abc"
        # Wait: the code checks `last_space > 0`, so if rfind returns -1 or 0 it does NOT snap
        # So result stays "abc"
        assert _la("abcdef", "abcxyz") == "abc"


class TestIsCjk:
    def test_chinese(self):
        assert _is_cjk("你") is True

    def test_latin(self):
        assert _is_cjk("A") is False

    def test_japanese_hiragana(self):
        assert _is_cjk("あ") is True

    def test_japanese_katakana(self):
        assert _is_cjk("ア") is True

    def test_korean_hangul(self):
        assert _is_cjk("한") is True

    def test_digit(self):
        assert _is_cjk("5") is False

    def test_space(self):
        assert _is_cjk(" ") is False

    def test_cjk_boundary_low(self):
        # U+4E00 is the first CJK Unified Ideograph
        assert _is_cjk("\u4e00") is True

    def test_cjk_boundary_high(self):
        # U+9FFF is the last in the basic CJK block
        assert _is_cjk("\u9fff") is True
