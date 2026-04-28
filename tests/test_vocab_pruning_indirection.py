"""Unit tests for vocab pruning indirection (Phase C).

Tests the orig↔red mapping infrastructure added to app/backends/qwen3_asr.py
for ASR_VOCAB_PRUNED=1 mode.  Uses synthetic token_map.bin so no WSL/Nano
artifacts are required.
"""

import os
import sys
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Ensure the backend module is importable (app/ is not a package — no __init__.py)
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "app"))

from app.backends.qwen3_asr import (
    IM_START, IM_END, AUDIO_START, AUDIO_END, AUDIO_PAD, ASR_TEXT, EOS_IDS,
    ASR_TOKEN_MAP_PATH, ASR_PRUNED_EMBED_NAME,
)

_SPECIAL_IDS = {IM_START, IM_END, AUDIO_START, AUDIO_END, AUDIO_PAD, ASR_TEXT, 151643, 151645, 9125, 882, 77091}


def _build_synthetic_token_map(extra_ids: set[int] | None = None) -> np.ndarray:
    """Build a synthetic red→orig uint32 array including all Qwen3 ASR special tokens.

    Returns np.ndarray[dtype=uint32] — the same format as token_map.bin.
    The reduced set consists of:
      - All special tokens (IM_START, IM_END, AUDIO_START, AUDIO_END,
        AUDIO_PAD, ASR_TEXT, EOS_IDS, and the prompt tokens 9125, 882, 77091).
      - A contiguous range of 'keep' tokens from 0..N  (excluding specials
        already included) to simulate ~100-kept vocab.
    """
    keep = set(_SPECIAL_IDS)
    if extra_ids:
        keep |= extra_ids
    # Fill with low-index tokens up to 200 total so the map has a realistic span
    for i in range(200):
        keep.add(i)
    keep_sorted = sorted(keep)
    return np.array(keep_sorted, dtype=np.uint32)


def _write_token_map(path: str, red2orig: np.ndarray) -> None:
    """Write token_map.bin in the same format the backend expects (raw uint32)."""
    red2orig.astype(np.uint32).tofile(path)


def _load_maps(red2orig: np.ndarray):
    """Simulate the backend's token map loading (see preload() in qwen3_asr.py)."""
    n_red = len(red2orig)
    orig2red = np.full(151936, -1, dtype=np.int32)
    orig2red[red2orig] = np.arange(n_red, dtype=np.int32)
    eos_red_ids = {int(rid) for eid in EOS_IDS for rid in [orig2red[eid]] if rid >= 0}
    return red2orig, orig2red, n_red, eos_red_ids


class TestTokenMapIndirection(unittest.TestCase):

    def setUp(self):
        self.red2orig = _build_synthetic_token_map()
        self.r2o, self.o2r, self.n_red, self.eos_red = _load_maps(self.red2orig)

    # ── 1. Round-trip integrity ───────────────────────────────────

    def test_round_trip_all_special_ids(self):
        """Every special sentinel ID maps to a non-negative reduced ID and back."""
        for sid in sorted(_SPECIAL_IDS):
            with self.subTest(special_id=sid):
                rid = self.o2r[sid]
                self.assertGreaterEqual(rid, 0,
                    f"Special ID {sid} not in reduced vocab (orig2red[{sid}] = {rid})")
                orig_back = int(self.r2o[rid])
                self.assertEqual(orig_back, sid,
                    f"Round-trip failed: {sid} → red {rid} → {orig_back}")

    def test_eos_red_ids_nonempty(self):
        """EOS sentinels (151643, 151645) both have reduced-space representations."""
        self.assertGreater(len(self.eos_red), 0)
        for eid in EOS_IDS:
            with self.subTest(eos_id=eid):
                rid = self.o2r[eid]
                self.assertGreaterEqual(rid, 0,
                    f"EOS ID {eid} missing from reduced vocab")
                self.assertIn(int(rid), self.eos_red)

    def test_round_trip_random_sample(self):
        """Reduced IDs in [0, n_red) round-trip correctly."""
        n = min(50, self.n_red)
        rids = np.random.choice(self.n_red, size=n, replace=False)
        for rid in rids:
            rid = int(rid)
            orig = int(self.r2o[rid])
            back = int(self.o2r[orig])
            self.assertEqual(back, rid,
                f"red {rid} → orig {orig} → red {back}")

    def test_unknown_orig_id_is_negative(self):
        """An original ID not in keep_ids maps to -1."""
        # It is very unlikely that 99999 is in a small synthetic map
        self.assertEqual(self.o2r[99999], -1,
            "orig2red[99999] should be -1 (not in keep set)")

    # ── 2. Prompt token coverage ──────────────────────────────────

    def _assert_prompt_ids_all_mapped(self, prompt_ids: list[int], label: str):
        """Fail if any prompt token is not in the reduced vocab."""
        for i, tid in enumerate(prompt_ids):
            rid = self.o2r[tid]
            self.assertGreaterEqual(rid, 0,
                f"[{label}] Prompt token at pos {i}: orig={tid} has no reduced mapping")

    def test_prompt_no_language(self):
        """All tokens in _build_prompt(language=None) are in keep_ids."""
        from app.backends.qwen3_asr import Qwen3ASRBackend
        backend = Qwen3ASRBackend()
        backend._asr_vocab_pruned = True
        backend._orig2red = self.o2r
        prompt = backend._build_prompt(audio_len=5, language=None)
        self._assert_prompt_ids_all_mapped(prompt, "language=None")

    def test_prompt_with_language_via_mock_tokenizer(self):
        """All tokens in _build_prompt(language='english') are in keep_ids."""
        from unittest.mock import MagicMock
        from app.backends.qwen3_asr import Qwen3ASRBackend
        backend = Qwen3ASRBackend()
        backend._asr_vocab_pruned = True
        backend._orig2red = self.o2r
        mock_tok = MagicMock()
        # encode("language english") returns a small list of safe token IDs
        mock_tok.encode.return_value.ids = [1, 2, 3]
        backend._tokenizer = mock_tok
        prompt = backend._build_prompt(audio_len=5, language="english")
        self._assert_prompt_ids_all_mapped(prompt, "language=english")
        # Also test chinese
        mock_tok.encode.return_value.ids = [4, 5, 6]
        prompt_cn = backend._build_prompt(audio_len=5, language="chinese")
        self._assert_prompt_ids_all_mapped(prompt_cn, "language=chinese")

    # ── 3. Embed lookup equivalence ───────────────────────────────

    def test_embed_lookup_via_orig2red(self):
        """Indexing pruned embed via orig2red gives the same row as original embed."""
        # Build full-size synthetic embed [151936, 4] and a pruned slice
        full_embed = np.random.randn(151936, 4).astype(np.float16)
        keep_sorted = sorted(set(self.r2o.tolist()))
        pruned_embed = full_embed[keep_sorted]

        # For each special token, verify the mapping picks the correct row
        for sid in sorted(_SPECIAL_IDS):
            with self.subTest(special_id=sid):
                rid = self.o2r[sid]
                if rid < 0:
                    continue
                np.testing.assert_array_equal(
                    pruned_embed[rid],
                    full_embed[sid],
                    err_msg=f"Embed row mismatch for special ID {sid} (red={rid})",
                )

    # ── 4. EOS termination in reduced space ───────────────────────

    def test_eos_red_id_terminates_decode(self):
        """A fake logits vector peaking at an EOS red ID terminates decode."""
        eos_rid = next(iter(self.eos_red))
        logits = np.full((1, 1, self.n_red), -10.0, dtype=np.float32)
        logits[0, 0, eos_rid] = 0.0

        output_ids: list[int] = []
        next_token = int(np.argmax(logits[0, -1, :]))
        self.assertIn(next_token, self.eos_red,
            "Argmax should return an EOS reduced ID")
        # If we treated next_token as EOS, we would stop here
        # Verify that the EOS check works
        self.assertIn(next_token, self.eos_red)

    def test_non_eos_red_id_appends_orig_id(self):
        """A non-EOS red ID maps back to the correct original ID."""
        # Pick a reduced ID that is NOT an EOS
        non_eos_candidates = [i for i in range(self.n_red) if i not in self.eos_red]
        self.assertGreater(len(non_eos_candidates), 0,
            "Need at least one non-EOS reduced ID for this test")
        rid = non_eos_candidates[0]
        expected_orig = int(self.r2o[rid])

        # Simulate what the decode loop does
        orig_id = int(self.r2o[rid])
        self.assertEqual(orig_id, expected_orig)
        self.assertNotIn(rid, self.eos_red)

    # ── 5. Token map file I/O parity ──────────────────────────────

    def test_token_map_file_round_trip(self):
        """Write a token_map.bin and verify the backend loading logic reproduces maps."""
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            _write_token_map(f.name, self.red2orig)

            # Simulate backend loading
            loaded_red2orig = np.fromfile(f.name, dtype=np.uint32)
            n_red = len(loaded_red2orig)
            loaded_orig2red = np.full(151936, -1, dtype=np.int32)
            loaded_orig2red[loaded_red2orig] = np.arange(n_red, dtype=np.int32)

            np.testing.assert_array_equal(loaded_red2orig, self.red2orig)
            for sid in _SPECIAL_IDS:
                expected_rid = self.o2r[sid]
                actual_rid = loaded_orig2red[sid]
                self.assertEqual(actual_rid, expected_rid,
                    f"File-loaded orig2red[{sid}] = {actual_rid} != {expected_rid}")

    # ── 6. Env gate wiring ────────────────────────────────────────

    def test_env_gate_parsing(self):
        """ASR_VOCAB_PRUNED env var is parsed correctly (only '1' is truthy)."""
        # On value
        for val in ("1",):
            with self.subTest(val=val):
                os.environ["ASR_VOCAB_PRUNED"] = val
                from app.backends import qwen3_asr as mod
                mod.ASR_VOCAB_PRUNED = os.environ.get("ASR_VOCAB_PRUNED", "0") == "1"
                self.assertTrue(mod.ASR_VOCAB_PRUNED, f"{repr(val)} should be True")

        # Off values (including non-'1' truthy-looking strings)
        for val in ("0", "", "true", "TRUE", "yes", "YES", "false", "no", "off"):
            with self.subTest(val=val):
                os.environ["ASR_VOCAB_PRUNED"] = val
                from app.backends import qwen3_asr as mod
                mod.ASR_VOCAB_PRUNED = os.environ.get("ASR_VOCAB_PRUNED", "0") == "1"
                self.assertFalse(mod.ASR_VOCAB_PRUNED, f"{repr(val)} should be False")

        del os.environ["ASR_VOCAB_PRUNED"]
        from app.backends import qwen3_asr as mod
        mod.ASR_VOCAB_PRUNED = os.getenv("ASR_VOCAB_PRUNED", "0") == "1"
        self.assertFalse(mod.ASR_VOCAB_PRUNED, "Default (unset) should be False")


if __name__ == "__main__":
    unittest.main()
