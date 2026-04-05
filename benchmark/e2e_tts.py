#!/usr/bin/env python3
"""
End-to-end Qwen3-TTS 0.6B inference pipeline for Jetson Orin Nano.
Produces a WAV file from text input using ONNX Runtime only.

Models (ALL FP32 from elbruno, same source — no INT8 mixing):
  - FP32 talker_prefill / talker_decode (CPU)
  - FP32 code_predictor with KV cache (CPU)
  - FP32 vocoder (CUDA preferred, CPU fallback)
  - NPY embedding lookup tables (same source as models)

Key design decisions (validated in test_all_fp32.py):
  1. Code predictor MUST use sampling (top-k=50, temp=0.9), NOT greedy.
     Greedy sub-codes degenerate to a repeating pattern → vocoder silence.
  2. FP32 code_predictor.onnx has past_keys/past_values KV cache inputs.
     The INT8 code_predictor_q.onnx does NOT have KV cache → wrong sub-codes.
  3. Codec embeddings use NPY lookup tables from the same source as models.
     Do NOT use INT8 ONNX models for embedding lookup.
  4. FP32 talker needs position_ids: [3,1,seq] for prefill, [3,1,1] for decode.
  5. FP32 prefill outputs separate KV per layer (present_key_0..27).
     Decode expects stacked past_keys [28,B,8,T,128]. Must stack after prefill.
  6. FP32 outputs `hidden_states` not `last_hidden`.
     Use hidden_states[:, -1:, :] for last token's hidden state.
  7. Next talker input = sum(codec_embed(first_code) + cp_embed_i(sub_code_i)
     for all 16 groups) + tts_pad_embed.
  8. Input embeddings use ref_input_embeds.npy (pre-computed from PyTorch SDK).

Usage:
    python e2e_tts.py
    python e2e_tts.py --text "Hello world" --lang english
    python e2e_tts.py --text "你好世界" --lang chinese --output hello_zh.wav
"""

import argparse
import json
import os
import struct
import sys
import time
import gc
import re
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FP32_DIR = "/tmp/qwen3-tts-bench/model"
EMB_DIR = os.path.join(FP32_DIR, "embeddings")
TOK_DIR = os.path.join(FP32_DIR, "tokenizer")
CORRECT_EMB_DIR = "/tmp/correct-emb"
REF_INPUT_EMBEDS = "/tmp/ref_input_embeds.npy"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 1024
NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_CODE_GROUPS = 16  # 1 first code + 15 sub-codes
NUM_CP_STEPS = 15     # code predictor runs 15 steps (groups 1-15)
SAMPLE_RATE = 24000
CODEC_HZ = 12.5

# Special token IDs
TTS_PAD_ID = 151671
CODEC_EOS_ID = 2150

LANGUAGE_IDS = {
    "chinese": 2055,
    "english": 2050,
    "japanese": 2058,
    "korean": 2059,
}

# Max generation tokens (safety limit)
MAX_CODEC_FRAMES = 100  # ~8 seconds at 12.5 Hz


# ---------------------------------------------------------------------------
# WAV writer (no scipy/librosa dependency)
# ---------------------------------------------------------------------------
def write_wav(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    """Write a mono WAV file from float32 audio [-1, 1]."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    n_samples = len(pcm)
    with open(path, "wb") as f:
        data_size = n_samples * 2
        f.write(struct.pack("<4sI4s", b"RIFF", 36 + data_size, b"WAVE"))
        f.write(struct.pack("<4sIHHIIHH",
                            b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(struct.pack("<4sI", b"data", data_size))
        f.write(pcm.tobytes())
    print(f"  Saved: {path} ({n_samples / sr:.2f}s, {n_samples} samples)")


# ---------------------------------------------------------------------------
# BPE Tokenizer (from vocab.json + merges.txt)
# ---------------------------------------------------------------------------
class BPETokenizer:
    """Minimal BPE tokenizer compatible with Qwen2 vocab.json + merges.txt."""

    def __init__(self, vocab_path: str, merges_path: str):
        t0 = time.perf_counter()

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        if lines and lines[0].startswith("#"):
            lines = lines[1:]
        self.bpe_ranks = {}
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) == 2:
                self.bpe_ranks[tuple(parts)] = i

        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self._cache = {}

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  BPE tokenizer loaded: {len(self.encoder)} tokens, "
              f"{len(self.bpe_ranks)} merges ({elapsed:.0f}ms)")

    @staticmethod
    def _bytes_to_unicode():
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("\xa1"), ord("\xac") + 1))
            + list(range(ord("\xae"), ord("\xff") + 1))
        )
        cs = list(bs)
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def _get_pairs(self, word):
        pairs = set()
        prev = word[0]
        for sym in word[1:]:
            pairs.add((prev, sym))
            prev = sym
        return pairs

    def _bpe(self, token: str) -> str:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        result = " ".join(word)
        self._cache[token] = result
        return result

    def encode(self, text: str) -> list:
        pat = re.compile(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
            re.UNICODE,
        )
        try:
            tokens_raw = pat.findall(text)
        except Exception:
            pat = re.compile(
                r"""[a-zA-Z]+|[\u4e00-\u9fff]|[0-9]+|[^\sa-zA-Z0-9\u4e00-\u9fff]+|\s+""",
                re.UNICODE,
            )
            tokens_raw = pat.findall(text)

        bpe_tokens = []
        for token in tokens_raw:
            encoded = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_result = self._bpe(encoded)
            for bpe_tok in bpe_result.split(" "):
                if bpe_tok in self.encoder:
                    bpe_tokens.append(self.encoder[bpe_tok])
                else:
                    for ch in bpe_tok:
                        if ch in self.encoder:
                            bpe_tokens.append(self.encoder[ch])
        return bpe_tokens


# ---------------------------------------------------------------------------
# ONNX Runtime session helpers
# ---------------------------------------------------------------------------
def create_session(path: str, providers: list, threads: int = 6):
    """Create an ORT InferenceSession with standard options."""
    import onnxruntime as ort

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 2
    opts.log_severity_level = 3

    sess = ort.InferenceSession(path, sess_options=opts, providers=providers)
    actual = sess.get_providers()[0]
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Loaded {os.path.basename(path)} ({size_mb:.1f}MB) on {actual}")
    return sess


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------
class Qwen3TTSPipeline:
    """
    End-to-end Qwen3-TTS 0.6B inference pipeline (ALL FP32).

    Steps:
        1. Load pre-computed input embeddings (ref_input_embeds.npy)
        2. Run talker prefill (get initial logits + KV cache)
        3. Autoregressive decode loop (talker + code predictor with sampling)
        4. Vocoder: codec codes -> waveform
    """

    def __init__(self, use_fp32_vocoder: bool = True):
        import onnxruntime as ort

        self.timings = {}
        t_total = time.perf_counter()

        print("\n[init] Loading Qwen3-TTS pipeline (ALL FP32)...")

        # Detect providers
        available = ort.get_available_providers()
        self.has_cuda = "CUDAExecutionProvider" in available
        cpu = ["CPUExecutionProvider"]
        cuda = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.has_cuda else cpu

        print(f"  ORT {ort.__version__} | CUDA: {self.has_cuda}")

        # --- Load tokenizer ---
        t0 = time.perf_counter()
        self._try_load_tokenizers_lib()
        if self.tokenizer is None:
            self.tokenizer = BPETokenizer(
                os.path.join(TOK_DIR, "vocab.json"),
                os.path.join(TOK_DIR, "merges.txt"),
            )
        self.timings["tokenizer_load"] = time.perf_counter() - t0

        # --- Load NPY embedding lookup tables ---
        t0 = time.perf_counter()
        self._load_embeddings()
        self.timings["embeddings_load"] = time.perf_counter() - t0

        # --- Load ONNX sessions (ALL FP32) ---
        print("\n[init] Loading ONNX models (ALL FP32 from elbruno)...")

        # Talker prefill: processes full input sequence
        t0 = time.perf_counter()
        self.talker_prefill = create_session(
            f"{FP32_DIR}/talker_prefill.onnx", cpu)
        self.timings["talker_prefill_load"] = time.perf_counter() - t0

        # Talker decode: autoregressive single-step with KV cache
        t0 = time.perf_counter()
        self.talker_decode = create_session(
            f"{FP32_DIR}/talker_decode.onnx", cpu)
        self.timings["talker_decode_load"] = time.perf_counter() - t0

        # Code predictor: FP32 with KV cache (past_keys/past_values)
        t0 = time.perf_counter()
        self.code_predictor = create_session(
            f"{FP32_DIR}/code_predictor.onnx", cpu)
        self.timings["code_predictor_load"] = time.perf_counter() - t0

        # Vocoder: codec codes -> waveform (FP32 on CUDA for speed)
        t0 = time.perf_counter()
        if use_fp32_vocoder and os.path.exists(f"{FP32_DIR}/vocoder.onnx"):
            self.vocoder = create_session(f"{FP32_DIR}/vocoder.onnx", cuda)
        else:
            self.vocoder = create_session(f"{FP32_DIR}/vocoder.onnx", cpu)
        self.timings["vocoder_load"] = time.perf_counter() - t0

        # Cache output names
        self._prefill_output_names = [o.name for o in self.talker_prefill.get_outputs()]
        self._decode_output_names = [o.name for o in self.talker_decode.get_outputs()]
        self._cp_output_names = [o.name for o in self.code_predictor.get_outputs()]
        self._vocoder_input_name = self.vocoder.get_inputs()[0].name

        self.timings["total_load"] = time.perf_counter() - t_total
        print(f"\n[init] Pipeline ready ({self.timings['total_load']:.1f}s)")

    def _try_load_tokenizers_lib(self):
        """Try loading HuggingFace tokenizers library (faster, more accurate)."""
        self.tokenizer = None
        self._use_hf_tokenizer = False
        try:
            from tokenizers import Tokenizer
            tok_json = os.path.join(TOK_DIR, "tokenizer.json")
            if os.path.exists(tok_json):
                self.tokenizer = Tokenizer.from_file(tok_json)
                self._use_hf_tokenizer = True
                print(f"  HuggingFace tokenizer loaded from tokenizer.json")
                return
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel
            vocab_path = os.path.join(TOK_DIR, "vocab.json")
            merges_path = os.path.join(TOK_DIR, "merges.txt")
            if os.path.exists(vocab_path) and os.path.exists(merges_path):
                tok = Tokenizer(BPE(vocab_path, merges_path))
                tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
                self.tokenizer = tok
                self._use_hf_tokenizer = True
                print(f"  HuggingFace tokenizer built from vocab.json + merges.txt")
                return
        except ImportError:
            pass
        except Exception as e:
            print(f"  [warn] HuggingFace tokenizer failed: {e}")

    def _load_embeddings(self):
        """Load NPY embedding lookup tables (same source as FP32 models)."""
        print("\n[init] Loading NPY embeddings...")

        # Talker codec embedding [3072, 1024] — from correct-emb directory
        path = os.path.join(CORRECT_EMB_DIR, "talker_codec_embedding.npy")
        self.talker_codec_emb = np.load(path).astype(np.float32)
        print(f"  talker_codec_embedding: {self.talker_codec_emb.shape}")

        # Code predictor per-group embeddings [2048, 1024] x 15
        self.cp_codec_embs = []
        for i in range(NUM_CP_STEPS):
            path = os.path.join(CORRECT_EMB_DIR, f"cp_codec_embedding_{i}.npy")
            emb = np.load(path).astype(np.float32)
            self.cp_codec_embs.append(emb)
        print(f"  cp_codec_embeddings: {NUM_CP_STEPS} x {self.cp_codec_embs[0].shape}")

        # Text projection weights (for building input embeddings from text)
        self.text_emb_weights = np.load(
            os.path.join(EMB_DIR, "text_embedding.npy"))
        self.text_proj_fc1_w = np.load(
            os.path.join(EMB_DIR, "text_projection_fc1_weight.npy"))
        self.text_proj_fc1_b = np.load(
            os.path.join(EMB_DIR, "text_projection_fc1_bias.npy"))
        self.text_proj_fc2_w = np.load(
            os.path.join(EMB_DIR, "text_projection_fc2_weight.npy"))
        self.text_proj_fc2_b = np.load(
            os.path.join(EMB_DIR, "text_projection_fc2_bias.npy"))
        print(f"  text_embedding: {self.text_emb_weights.shape}")
        print(f"  text_projection: fc1={self.text_proj_fc1_w.shape}, "
              f"fc2={self.text_proj_fc2_w.shape}")

        # Pre-compute tts_pad embedding (used every decode step)
        self.tts_pad_emb = self._text_project_token(TTS_PAD_ID)
        print(f"  tts_pad_emb: {self.tts_pad_emb.shape}")

    def _text_project_token(self, token_id: int) -> np.ndarray:
        """text_embedding -> text_projection (fc1 -> silu -> fc2) for one token."""
        x = self.text_emb_weights[token_id].astype(np.float32)  # [2048]
        h = x @ self.text_proj_fc1_w.T + self.text_proj_fc1_b   # [2048]
        h = h * (1.0 / (1.0 + np.exp(-h)))                      # SiLU
        out = h @ self.text_proj_fc2_w.T + self.text_proj_fc2_b  # [1024]
        return out.reshape(1, 1, 1024)

    def _codec_embed_lookup(self, token_id: int) -> np.ndarray:
        """Lookup talker codec embedding for a token ID. Returns [1,1,1024]."""
        return self.talker_codec_emb[token_id:token_id+1].reshape(
            1, 1, 1024).astype(np.float32)

    def _cp_embed_lookup(self, token_id: int, group: int) -> np.ndarray:
        """Lookup code predictor embedding for a token at a given group. Returns [1,1,1024]."""
        return self.cp_codec_embs[group][token_id:token_id+1].reshape(
            1, 1, 1024).astype(np.float32)

    def tokenize(self, text: str) -> list:
        if self._use_hf_tokenizer:
            return self.tokenizer.encode(text).ids
        else:
            return self.tokenizer.encode(text)

    def _sample_top_k(self, logits_1d: np.ndarray, k: int = 50,
                      temperature: float = 0.9) -> int:
        """Sample from logits with top-k filtering and temperature.

        MUST use sampling, NOT greedy — greedy sub-codes degenerate to a
        repeating pattern that makes vocoder output silence.
        """
        logits = logits_1d.flatten().astype(np.float64) / temperature

        # Top-k filtering
        top_k_indices = np.argpartition(logits, -k)[-k:]
        mask = np.full_like(logits, -np.inf)
        mask[top_k_indices] = logits[top_k_indices]

        # Softmax
        exp_logits = np.exp(mask - np.max(mask))
        probs = exp_logits / exp_logits.sum()

        if np.any(np.isnan(probs)):
            return int(np.argmax(logits_1d))

        return int(np.random.choice(len(probs), p=probs))

    def _get_input_embeddings(self, text: str, language: str) -> np.ndarray:
        """Get input embeddings for the talker.

        Currently uses pre-computed ref_input_embeds.npy. Later this can be
        built from tokenizer + embedding lookup.
        """
        if os.path.exists(REF_INPUT_EMBEDS):
            emb = np.load(REF_INPUT_EMBEDS).astype(np.float32)
            print(f"  Using pre-computed ref_input_embeds: {emb.shape}")
            return emb

        # Fallback: build from text (not yet validated end-to-end)
        print("  [warn] ref_input_embeds.npy not found, building from text...")
        return self._build_input_embeddings_from_text(text, language)

    def _build_input_embeddings_from_text(self, text: str,
                                          language: str) -> np.ndarray:
        """Build input embeddings from scratch using tokenizer + text_project.

        NOTE: This path is not yet validated end-to-end. The ref_input_embeds.npy
        path is the known-working approach.
        """
        from functools import reduce

        lang_id = LANGUAGE_IDS.get(language, LANGUAGE_IDS["english"])

        # Role prefix: <|im_start|>assistant\n
        IM_START_ID = 151644
        ROLE_PREFIX_IDS = [IM_START_ID, 77091, 198]
        role_prefix = np.concatenate(
            [self._text_project_token(tid) for tid in ROLE_PREFIX_IDS], axis=1)

        # Codec prefix: [nothink, think_bos, lang, think_eos, pad, bos]
        CODEC_NOTHINK_ID = 2155
        CODEC_THINK_BOS_ID = 2156
        CODEC_THINK_EOS_ID = 2157
        CODEC_PAD_ID = 2148
        CODEC_BOS_ID = 2149
        codec_ids = [CODEC_NOTHINK_ID, CODEC_THINK_BOS_ID, lang_id,
                     CODEC_THINK_EOS_ID, CODEC_PAD_ID, CODEC_BOS_ID]
        codec_prefix = np.concatenate(
            [self._codec_embed_lookup(cid) for cid in codec_ids], axis=1)

        # Text tokens
        text_ids = self.tokenize(text)
        print(f"  Text tokens: {len(text_ids)} IDs")
        text_emb = np.concatenate(
            [self._text_project_token(tid) for tid in text_ids], axis=1)

        # Role suffix: <|im_end|>\n<|im_start|>assistant\n
        IM_END_ID = 151645
        ROLE_SUFFIX_IDS = [IM_END_ID, 198, IM_START_ID, 77091, 198]
        role_suffix = np.concatenate(
            [self._text_project_token(tid) for tid in ROLE_SUFFIX_IDS], axis=1)

        input_emb = np.concatenate(
            [role_prefix, codec_prefix, text_emb, role_suffix], axis=1)
        print(f"  Built input embedding: {input_emb.shape}")
        return input_emb

    def synthesize(self, text: str, language: str = "english",
                   output_path: str = "output.wav",
                   top_k: int = 50, temperature: float = 0.9,
                   max_frames: int = MAX_CODEC_FRAMES) -> Optional[str]:
        """
        Full TTS pipeline: text -> WAV file.

        Args:
            text: Input text to synthesize
            language: Language hint (english, chinese, japanese, korean)
            output_path: Path for output WAV file
            top_k: Top-k sampling parameter (must use sampling, not greedy)
            temperature: Sampling temperature (0.9 recommended)
            max_frames: Maximum number of codec frames to generate

        Returns:
            Path to output WAV file, or None on failure
        """
        print(f"\n{'='*65}")
        print(f"Synthesizing: \"{text}\"")
        print(f"Language: {language} | top_k={top_k} | temp={temperature}")
        print(f"{'='*65}")

        t_pipeline = time.perf_counter()

        # ---------------------------------------------------------------
        # Step 1: Get input embeddings
        # ---------------------------------------------------------------
        print("\n[step 1] Getting input embeddings...")
        t0 = time.perf_counter()
        input_emb = self._get_input_embeddings(text, language)
        seq_len = input_emb.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        self.timings["build_emb"] = (time.perf_counter() - t0) * 1000

        # ---------------------------------------------------------------
        # Step 2: Talker prefill
        # ---------------------------------------------------------------
        print("\n[step 2] Talker prefill...")
        t0 = time.perf_counter()

        # FP32 talker needs position_ids: [3, 1, seq_len]
        pos_ids = np.arange(seq_len, dtype=np.int64).reshape(
            1, 1, seq_len).repeat(3, axis=0)  # [3, 1, seq_len]

        prefill_out = self.talker_prefill.run(None, {
            "inputs_embeds": input_emb,
            "attention_mask": attention_mask,
            "position_ids": pos_ids,
        })

        # Parse prefill outputs
        pm = dict(zip(self._prefill_output_names, prefill_out))

        logits = pm["logits"]
        # FP32 outputs `hidden_states` not `last_hidden`
        hidden = pm["hidden_states"][:, -1:, :]  # last token hidden

        # FP32 prefill outputs separate KV per layer (present_key_0..27).
        # Decode expects stacked: past_keys [28, B, 8, T, 128].
        pk_list = [pm[f"present_key_{i}"] for i in range(NUM_LAYERS)]
        pv_list = [pm[f"present_value_{i}"] for i in range(NUM_LAYERS)]
        kv = {
            "present_keys": np.stack(pk_list, axis=0),
            "present_values": np.stack(pv_list, axis=0),
        }

        elapsed_prefill = (time.perf_counter() - t0) * 1000
        self.timings["prefill"] = elapsed_prefill
        print(f"  Prefill done: logits={logits.shape}, "
              f"kv={kv['present_keys'].shape} ({elapsed_prefill:.1f}ms)")

        # ---------------------------------------------------------------
        # Step 3: Autoregressive decode loop
        # ---------------------------------------------------------------
        print("\n[step 3] Autoregressive decode loop...")
        t0 = time.perf_counter()

        all_codes = []
        total_seq = seq_len
        decode_times = []
        cp_times = []

        for frame in range(max_frames):
            # --- Sample first code (talker) with temperature + top-k ---
            first_code = self._sample_top_k(
                logits[0, -1, :], k=top_k, temperature=temperature)

            if first_code == CODEC_EOS_ID:
                print(f"  EOS at frame {frame}")
                break

            # --- Code predictor: predict 15 sub-codes ---
            t_cp = time.perf_counter()

            # Start with [hidden_state, codec_embed(first_code)]
            fce = self._codec_embed_lookup(first_code)
            ci = np.concatenate([hidden, fce], axis=1)  # [1, 2, 1024]

            # FP32 code_predictor has KV cache (past_keys/past_values)
            cpk = np.zeros((5, 1, 8, 0, 128), dtype=np.float32)
            cpv = np.zeros((5, 1, 8, 0, 128), dtype=np.float32)

            subs = []
            for step in range(NUM_CP_STEPS):
                co = self.code_predictor.run(None, {
                    "inputs_embeds": ci,
                    "generation_steps": np.array([step], dtype=np.int64),
                    "past_keys": cpk,
                    "past_values": cpv,
                })
                cm = dict(zip(self._cp_output_names, co))

                # MUST sample (not greedy) — greedy sub-codes degenerate
                sub_code = self._sample_top_k(
                    cm["logits"][0, -1, :], k=top_k, temperature=temperature)
                subs.append(sub_code)

                # Update KV cache for next step
                cpk = cm["present_keys"]
                cpv = cm["present_values"]

                # Next input = cp_embed_lookup(sub_code, step)
                ci = self._cp_embed_lookup(sub_code, step)

            cp_times.append(time.perf_counter() - t_cp)

            codes = [first_code] + subs
            all_codes.append(codes)

            # --- Build next talker input ---
            # sum(codec_embed(first_code) + cp_embed_i(sub_code_i)) + tts_pad
            next_emb = self._codec_embed_lookup(first_code)
            for i, sc in enumerate(subs):
                next_emb = next_emb + self._cp_embed_lookup(sc, i)
            next_emb = next_emb + self.tts_pad_emb

            if frame == 0:
                print(f"  next_emb mean: {next_emb.mean():.6f} "
                      f"hidden mean: {hidden.mean():.6f}")

            # --- Talker decode step ---
            t_decode = time.perf_counter()
            total_seq += 1
            attn_mask = np.ones((1, total_seq), dtype=np.int64)
            # FP32 decode position_ids: [3, 1, 1] with current position
            pos = np.array([[[total_seq - 1]]] * 3, dtype=np.int64)

            feed = {
                "inputs_embeds": next_emb,
                "attention_mask": attn_mask,
                "position_ids": pos,
                "past_keys": kv["present_keys"],
                "past_values": kv["present_values"],
            }

            do = self.talker_decode.run(None, feed)
            dm = dict(zip(self._decode_output_names, do))
            decode_times.append(time.perf_counter() - t_decode)

            # FP32 outputs hidden_states, not last_hidden
            logits = dm["logits"]
            hidden = dm["hidden_states"][:, -1:, :]
            kv = {
                "present_keys": dm["present_keys"],
                "present_values": dm["present_values"],
            }

            # Progress
            if frame < 5 or (frame + 1) % 10 == 0:
                print(f"  Frame {frame}: fc={first_code} "
                      f"h_mean={hidden.mean():.6f} subs[:3]={subs[:3]}")

        elapsed_decode = (time.perf_counter() - t0) * 1000
        n_frames = len(all_codes)
        self.timings["decode_loop"] = elapsed_decode
        self.timings["n_frames"] = n_frames

        if n_frames == 0:
            print("  [error] No frames generated!")
            return None

        avg_decode_ms = np.mean(decode_times) * 1000 if decode_times else 0
        avg_cp_ms = np.mean(cp_times) * 1000 if cp_times else 0
        print(f"\n  Decode loop: {n_frames} frames in {elapsed_decode:.0f}ms")
        print(f"  Avg per frame: decode={avg_decode_ms:.1f}ms + cp={avg_cp_ms:.1f}ms "
              f"= {avg_decode_ms + avg_cp_ms:.1f}ms")
        print(f"  Audio duration: {n_frames / CODEC_HZ:.2f}s")

        rtf_step = (avg_decode_ms + avg_cp_ms) / (1000 / CODEC_HZ)
        print(f"  Step RTF: {rtf_step:.3f} "
              f"({'real-time OK' if rtf_step < 1 else 'TOO SLOW'})")

        # ---------------------------------------------------------------
        # Step 4: Vocoder — convert codes to waveform
        # ---------------------------------------------------------------
        print("\n[step 4] Vocoder...")
        t0 = time.perf_counter()

        codes_array = np.array(all_codes, dtype=np.int64)  # [n_frames, 16]
        # FP32 vocoder expects [1, 16, N]
        codes_input = codes_array.T[np.newaxis, :, :]  # [1, 16, n_frames]
        voc_out = self.vocoder.run(None, {self._vocoder_input_name: codes_input})
        waveform = voc_out[0].flatten()

        elapsed_vocoder = (time.perf_counter() - t0) * 1000
        self.timings["vocoder"] = elapsed_vocoder
        rms = np.sqrt(np.mean(waveform**2))
        print(f"  Vocoder: {len(waveform)} samples, rms={rms:.4f} ({elapsed_vocoder:.1f}ms)")

        # ---------------------------------------------------------------
        # Step 5: Save WAV
        # ---------------------------------------------------------------
        print("\n[step 5] Saving WAV...")
        waveform = waveform.astype(np.float32)

        peak = np.max(np.abs(waveform))
        if peak > 0:
            if peak > 1.0:
                waveform = waveform / peak * 0.95
        else:
            print("  [warn] Silent output!")

        write_wav(output_path, waveform, SAMPLE_RATE)

        # ---------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------
        elapsed_total = (time.perf_counter() - t_pipeline) * 1000
        self.timings["total_synthesis"] = elapsed_total
        audio_duration = len(waveform) / SAMPLE_RATE

        print(f"\n{'='*65}")
        print(f"SYNTHESIS COMPLETE")
        print(f"{'='*65}")
        print(f"  Text:           \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        print(f"  Language:       {language}")
        print(f"  Codec frames:   {n_frames}")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Output:         {output_path}")
        print(f"\n  Timing breakdown:")
        print(f"    Embedding build:  {self.timings.get('build_emb', 0):>8.1f}ms")
        print(f"    Prefill (TTFA):   {self.timings.get('prefill', 0):>8.1f}ms")
        print(f"    Decode loop:      {self.timings.get('decode_loop', 0):>8.1f}ms  "
              f"({n_frames} frames)")
        print(f"    Vocoder:          {self.timings.get('vocoder', 0):>8.1f}ms")
        print(f"    Total synthesis:  {elapsed_total:>8.1f}ms")
        print(f"    Overall RTF:      "
              f"{elapsed_total / 1000 / max(audio_duration, 0.001):.3f}")

        return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS 0.6B end-to-end inference (ONNX Runtime, ALL FP32)")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to synthesize (default: run test sentences)")
    parser.add_argument("--lang", type=str, default="english",
                        choices=list(LANGUAGE_IDS.keys()),
                        help="Language hint")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output WAV path")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling (default: 50)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature (default: 0.9)")
    parser.add_argument("--max-frames", type=int, default=MAX_CODEC_FRAMES,
                        help=f"Max codec frames (default: {MAX_CODEC_FRAMES})")
    parser.add_argument("--cpu-vocoder", action="store_true",
                        help="Force vocoder on CPU instead of CUDA")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print("=" * 65)
    print("Qwen3-TTS 0.6B -- End-to-End Inference Pipeline (ALL FP32)")
    print("=" * 65)

    # Verify required paths exist
    required_paths = [
        (FP32_DIR, "FP32 models"),
        (EMB_DIR, "Embeddings"),
        (TOK_DIR, "Tokenizer"),
        (CORRECT_EMB_DIR, "Correct embeddings (NPY)"),
    ]
    for d, label in required_paths:
        exists = os.path.isdir(d)
        status = "OK" if exists else "MISSING"
        print(f"  {label:30s}: {d} [{status}]")
        if not exists:
            print(f"\n[ERROR] Required directory not found: {d}")
            sys.exit(1)

    if not os.path.exists(REF_INPUT_EMBEDS):
        print(f"\n  [warn] {REF_INPUT_EMBEDS} not found.")
        print(f"         Will attempt to build input embeddings from text.")

    # Initialize pipeline
    pipeline = Qwen3TTSPipeline(use_fp32_vocoder=not args.cpu_vocoder)

    if args.text:
        pipeline.synthesize(
            text=args.text,
            language=args.lang,
            output_path=args.output,
            top_k=args.top_k,
            temperature=args.temperature,
            max_frames=args.max_frames,
        )
    else:
        test_sentences = [
            ("english", "Hello, welcome to the voice synthesis system."),
            ("chinese", "你好，欢迎使用语音合成系统。"),
        ]

        for lang, text in test_sentences:
            safe_name = text[:20].replace(" ", "_").replace(",", "")
            safe_name = re.sub(r"[^\w]", "", safe_name)
            out_path = f"/tmp/qwen3_e2e_{lang}_{safe_name}.wav"

            try:
                pipeline.synthesize(
                    text=text,
                    language=lang,
                    output_path=out_path,
                    top_k=args.top_k,
                    temperature=args.temperature,
                    max_frames=args.max_frames,
                )
            except Exception as e:
                print(f"\n[ERROR] Synthesis failed for \"{text}\": {e}")
                import traceback
                traceback.print_exc()

            gc.collect()

    print(f"\n{'='*65}")
    print("Done.")


if __name__ == "__main__":
    main()
