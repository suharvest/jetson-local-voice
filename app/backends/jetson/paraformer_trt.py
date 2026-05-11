"""Paraformer streaming ASR — encoder via TensorRT, decoder via ONNX Runtime CUDA.

Supports: OFFLINE, STREAMING
Uses numpy-only fbank extraction + TRT encoder (CUDA) + ORT decoder (CUDA EP).
CIF (Continuous Integrate-and-Fire) handles token timing and endpoint detection.

Provider architecture: encoder=trt, decoder=ort_cuda

Based on M1 manifest (docs/plans/matcha-paraformer-trt-m1-manifest-2026-04-28.md):
- Encoder input:  speech [1, feats_length, 560] float32
- Encoder input:  speech_lengths [1] int32
- Encoder output: enc [1, feats_length, 512] float32
- Encoder output: enc_len [1] int32
- Encoder output: alphas [1, feats_length] float32
- Decoder input:  enc, enc_len, acoustic_embeds [1, token_length, 512], acoustic_embeds_len [1]
- Decoder input:  in_cache_0..15 [1, 512, 10] float32 (fixed-depth causal window)
- Decoder output: logits [1, token_length, 8404], sample_ids [1, token_length] int64
- Decoder output: out_cache_0..15 [1, 512, 10]
- Vocab: 8404 tokens (0=blank, 1=<s>, 2=</s>, 8403=<unk>)
"""

from __future__ import annotations

import io
import logging
import os
import time
from typing import Optional

import numpy as np

from app.core.asr_backend import ASRBackend, ASRCapability, ASRStream, TranscriptionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARAFORMER_MODEL_DIR = os.environ.get(
    "PARAFORMER_MODEL_DIR",
    "/opt/models/paraformer-streaming",
)
ENC_ENGINE_PATH = os.environ.get(
    "PARAFORMER_ENC_ENGINE",
    os.path.join(PARAFORMER_MODEL_DIR, "engines", "paraformer_encoder_sp1_80.plan"),
)
ENC_ONNX_PATH = os.environ.get(
    "PARAFORMER_ENC_ONNX",
    os.path.join(PARAFORMER_MODEL_DIR, "encoder.onnx"),
)
DEC_ONNX_PATH = os.environ.get(
    "PARAFORMER_DEC_ONNX",
    os.path.join(PARAFORMER_MODEL_DIR, "decoder.onnx"),
)
TOKENS_PATH = os.environ.get(
    "PARAFORMER_TOKENS",
    os.path.join(PARAFORMER_MODEL_DIR, "tokens.txt"),
)

# Streaming parameters — match sherpa-onnx training distribution
# (https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en)
CHUNK_SIZE_SEC = 0.67      # 670ms chunk for low-latency partials
LEFT_CONTEXT_SEC = 2.68    # 4 prior chunks of left context (encoder_chunk_look_back=4)

# FBank parameters (kaldi-compatible)
SAMPLE_RATE = 16000
FFT_SIZE = 512
WINDOW_SIZE = 400           # 25ms at 16kHz
HOP_SIZE = 160              # 10ms at 16kHz
NUM_MEL_BINS = 80
NUM_STACKED = 7
NUM_STRIDE = 6   # LFR stride (downsample 10ms -> 60ms)
PRE_EMPH = 0.97
LOW_FREQ = 20
HIGH_FREQ = 8000

# CIF parameters
CIF_THRESHOLD = 1.0
# Defer the last N LFR frames of each chunk's CIF to the next chunk:
# these frames have no right-context in the current encoder run; they'll
# reach full context once they become history in the next chunk's run.
RIGHT_LOOKAHEAD_LFR = 15
CIF_TAIL_THRESHOLD = 0.5    # Minimum weight to fire tail token on finalize

# Tokens
BLANK_ID = 0
SOS_ID = 1
EOS_ID = 2
VOCAB_SIZE = 8404

# Engine names (from M1 manifest)
ENC_INPUT_NAMES = ["speech", "speech_lengths"]
ENC_OUTPUT_NAMES = ["enc", "enc_len", "alphas"]
DEC_INPUT_NAMES = ["enc", "enc_len", "acoustic_embeds", "acoustic_embeds_len"] + \
                  [f"in_cache_{i}" for i in range(16)]
DEC_OUTPUT_NAMES = ["logits", "sample_ids"] + \
                   [f"out_cache_{i}" for i in range(16)]

# ---------------------------------------------------------------------------
# FBank extraction (numpy-only, kaldi-style)
# ---------------------------------------------------------------------------

_MEL_FILTERBANK: Optional[np.ndarray] = None


def _get_mel_filterbank() -> np.ndarray:
    """Create 80-dim mel filterbank matrix: [80, 257] for 512-pt FFT at 16kHz."""
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is not None:
        return _MEL_FILTERBANK

    num_bins = NUM_MEL_BINS
    fft_size = FFT_SIZE
    sr = SAMPLE_RATE
    low_mel = 2595.0 * np.log10(1.0 + LOW_FREQ / 700.0)
    high_mel = 2595.0 * np.log10(1.0 + HIGH_FREQ / 700.0)
    mel_points = np.linspace(low_mel, high_mel, num_bins + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_indices = np.floor(hz_points * (fft_size // 2 + 1) / (sr / 2.0)).astype(np.int32)

    fbank = np.zeros((num_bins, fft_size // 2 + 1), dtype=np.float32)
    for i in range(num_bins):
        left, center, right = bin_indices[i], bin_indices[i + 1], bin_indices[i + 2]
        for j in range(left, center):
            fbank[i, j] = (j - left) / (center - left) if center != left else 1.0
        for j in range(center, right):
            fbank[i, j] = (right - j) / (right - center) if right != center else 1.0

    _MEL_FILTERBANK = fbank
    return fbank


def compute_fbank(audio: np.ndarray) -> np.ndarray:
    """Compute 80-dim log-fbank features from 16kHz audio.

    Returns:
        features: [num_frames, 80] float32
    """
    if len(audio) < WINDOW_SIZE:
        audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)))

    # Pre-emphasis
    audio = np.concatenate([[audio[0]], audio[1:] - PRE_EMPH * audio[:-1]])

    # Framing: [num_frames, window_size]
    num_frames = (len(audio) - WINDOW_SIZE) // HOP_SIZE + 1
    frames = np.zeros((num_frames, WINDOW_SIZE), dtype=np.float32)
    for i in range(num_frames):
        start = i * HOP_SIZE
        frames[i] = audio[start:start + WINDOW_SIZE]

    # Hamming window
    hamming = np.hamming(WINDOW_SIZE).astype(np.float32)
    frames = frames * hamming

    # Power spectrum: [num_frames, fft_size // 2 + 1]
    spectrum = np.fft.rfft(frames, n=FFT_SIZE)
    power = (spectrum.real ** 2 + spectrum.imag ** 2) / FFT_SIZE

    # Mel filterbank: [num_frames, 80]
    fbank = _get_mel_filterbank()
    mel_feats = power @ fbank.T  # [num_frames, 80]

    # Log (with floor)
    mel_feats = np.maximum(mel_feats, 1e-10)
    mel_feats = np.log(mel_feats)

    # Utterance-level CMVN
    mean = mel_feats.mean(axis=0, keepdims=True)
    std = mel_feats.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-10)
    mel_feats = (mel_feats - mean) / std

    return mel_feats.astype(np.float32)


def stack_frames(feats: np.ndarray) -> np.ndarray:
    """Apply Paraformer LFR (low-frame-rate): stack 7 frames with stride 6.

    Downsamples 10ms fbank frames to 60ms LFR frames. Output length =
    ceil(N / 6). Last partial window is padded by repeating the last frame.

    Returns:
        stacked: [ceil(N/6), 560] float32
    """
    n, d = feats.shape
    out_n = (n + NUM_STRIDE - 1) // NUM_STRIDE
    needed = (out_n - 1) * NUM_STRIDE + NUM_STACKED
    if needed > n:
        pad = np.repeat(feats[-1:], needed - n, axis=0)
        feats = np.concatenate([feats, pad], axis=0)
    stacked = np.zeros((out_n, d * NUM_STACKED), dtype=np.float32)
    for i in range(out_n):
        start = i * NUM_STRIDE
        stacked[i] = feats[start:start + NUM_STACKED].ravel()
    return stacked


# ---------------------------------------------------------------------------
# CIF (Continuous Integrate-and-Fire)
# ---------------------------------------------------------------------------

def cif(
    enc: np.ndarray,
    alphas: np.ndarray,
    threshold: float = CIF_THRESHOLD,
    tail_threshold: float = CIF_TAIL_THRESHOLD,
    carry_weight: float = 0.0,
    carry_embed: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Continuous Integrate-and-Fire for token boundary detection.

    Integrates alphas per frame. When cumulative weight crosses threshold,
    emits an acoustic embedding (weighted avg of encoder frames).

    Args:
        enc: encoder output [feats_length, 512]
        alphas: per-frame CIF weights [feats_length] (after activation)
        threshold: firing threshold (typically 1.0)
        tail_threshold: minimum accumulated weight to fire tail token on finalize
        carry_weight: accumulated weight carried from previous chunk
        carry_embed: accumulated embedding carried from previous chunk

    Returns:
        acoustic_embeds: [num_tokens, 512] (empty if no tokens)
        tail_weight: remaining accumulated weight for next chunk
        tail_embed: remaining accumulated embedding for next chunk
    """
    if carry_embed is None:
        carry_embed = np.zeros(512, dtype=np.float32)

    acoustic_embeds = []
    accum_weight = carry_weight
    accum_embed = carry_embed.copy()

    for t in range(len(enc)):
        alpha = float(alphas[t])
        if alpha <= 0:
            continue

        accum_weight += alpha
        accum_embed += alpha * enc[t]

        while accum_weight >= threshold:
            excess = accum_weight - threshold
            token_embed = (accum_embed - excess * enc[t]) / threshold
            acoustic_embeds.append(token_embed)
            accum_weight = excess
            accum_embed = excess * enc[t]

    return np.stack(acoustic_embeds) if acoustic_embeds else np.empty((0, 512), dtype=np.float32), \
           accum_weight, accum_embed


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def load_tokens(path: str) -> list[str]:
    """Load token-to-string mapping from tokens.txt.

    Supports two line formats:
      1. Plain:     <token_text>
      2. FunASR/k2: <token_text> <integer_id>
    In format 2, the trailing integer id is stripped.
    """
    with open(path, "r", encoding="utf-8") as f:
        tokens = []
        for line in f:
            token = line.rstrip("\n")
            parts = token.rsplit(None, 1)
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                token = parts[0]
            else:
                token = token.strip()
            tokens.append(token)
    return tokens


def decode_ids(token_ids: list[int], tokens: list[str]) -> str:
    """Decode token IDs to text, filtering special tokens.

    Skips BLANK/SOS/EOS and suppresses immediate adjacent-id repeats
    (e.g. [6049, 6049] → one "好"). EOS is skipped, NOT used as a stop:
    Paraformer streaming may emit EOS mid-stream as cache-flush artifact.
    BPE continuation suffix "@@" is stripped and merged with next token.
    """
    pieces = []
    prev_tid: Optional[int] = None
    for tid in token_ids:
        if tid in (BLANK_ID, SOS_ID, EOS_ID):
            continue
        if tid == prev_tid:
            continue
        if 0 <= tid < len(tokens):
            token = tokens[tid]
            if token.startswith("<") and token.endswith(">"):
                continue
            if token.endswith("@@"):
                token = token[:-2]
            pieces.append(token)
            prev_tid = tid
    return "".join(pieces)


# ---------------------------------------------------------------------------
# TRT helpers
# ---------------------------------------------------------------------------

_HAS_TRT = False
try:
    import tensorrt as trt
    from cuda import cudart
    _HAS_TRT = True
except ImportError:
    logger.warning("TensorRT or CUDA Python not available; paraformer_trt backend disabled")


def _load_trt_engine(path: str):
    """Load a TensorRT engine from a plan file."""
    logger_obj = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger_obj)
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


# ---------------------------------------------------------------------------
# ParaformerTRTStream
# ---------------------------------------------------------------------------

class ParaformerTRTStream(ASRStream):
    """Streaming ASR session backed by TRT encoder + ORT CUDA decoder."""

    def __init__(self, backend: "ParaformerTRTBackend"):
        self._backend = backend
        self._tokens = backend._tokens

        # Audio accumulation
        self._audio_buf = np.array([], dtype=np.float32)
        self._processed_chunks = 0

        # History audio for feature normalization continuity (Bug G fix)
        left_ctx_samples = int(LEFT_CONTEXT_SEC * SAMPLE_RATE)  # 16000
        self._history_audio = np.array([], dtype=np.float32)
        self._left_ctx_samples = left_ctx_samples
        # Full utterance audio for CMVN (matches offline normalization)
        self._all_audio = np.array([], dtype=np.float32)
        self._prev_total_frames = 0
        # Absolute LFR-frame index of next CIF-eligible frame (for right look-ahead defer)
        self._cif_processed_lfr = 0

        # Per-utterance state
        self._all_token_ids: list[int] = []
        self._partial_text: str = ""
        self._is_endpoint: bool = False

        # CIF cross-chunk carry-over
        self._carry_weight: float = 0.0
        self._carry_embed: np.ndarray = np.zeros(512, dtype=np.float32)

        # Decoder persistent cache (updated across chunks)
        self._cache = [np.zeros((1, 512, 10), dtype=np.float32) for _ in range(16)]

        # Timing
        self._chunk_count = 0
        self._total_enc_ms = 0.0
        self._total_dec_ms = 0.0

    def _reset_utterance_state(self) -> None:
        self._audio_buf = np.array([], dtype=np.float32)
        self._processed_chunks = 0
        self._all_token_ids = []
        self._partial_text = ""
        self._is_endpoint = False
        self._carry_weight = 0.0
        self._carry_embed = np.zeros(512, dtype=np.float32)
        self._cache = [np.zeros((1, 512, 10), dtype=np.float32) for _ in range(16)]
        self._chunk_count = 0
        self._total_enc_ms = 0.0
        self._total_dec_ms = 0.0
        self._history_audio = np.array([], dtype=np.float32)
        self._all_audio = np.array([], dtype=np.float32)
        self._prev_total_frames = 0
        self._cif_processed_lfr = 0

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if sample_rate != SAMPLE_RATE:
            ratio = SAMPLE_RATE / sample_rate
            new_len = int(len(samples) * ratio)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, new_len),
                np.arange(len(samples)), samples,
            ).astype(np.float32)

        self._audio_buf = np.concatenate([self._audio_buf, samples])
        self._process_chunks()

    def _process_chunks(self) -> None:
        """Process complete 400ms audio chunks."""
        chunk_samples = int(CHUNK_SIZE_SEC * SAMPLE_RATE)  # 6400
        while len(self._audio_buf) >= chunk_samples:
            chunk_audio = self._audio_buf[:chunk_samples]
            self._audio_buf = self._audio_buf[chunk_samples:]
            self._process_one_chunk(chunk_audio)

    def _process_one_chunk(self, audio: np.ndarray) -> None:
        """Process a single 400ms chunk: fbank -> encoder -> CIF -> decoder.

        Bug G fix: concat history_audio before fbank to get utterance-level CMVN
        across chunks. Only process new frames (skip history_frames).
        """
        t0 = time.perf_counter()

        # 1. Accumulate full-utterance audio (full-utterance CMVN matches
        # offline normalization)
        self._all_audio = np.concatenate([self._all_audio, audio])

        # 2. FBank + LFR stack over FULL utterance — keeps LFR alignment
        # identical to offline path (chunk-boundary independent)
        all_feats = compute_fbank(self._all_audio)
        all_lfr = stack_frames(all_feats)
        cur_total_lfr = all_lfr.shape[0]
        new_lfr = cur_total_lfr - self._prev_total_frames
        if new_lfr <= 0:
            return
        self._prev_total_frames = cur_total_lfr

        # 3. Run encoder over FULL accumulated LFR (max engine: 400 frames).
        # Decoder then has full-utterance cross-attention memory like offline.
        # Encoder cost is bounded; profile 1 covers up to 400 LFR (~24s).
        feats = all_lfr if cur_total_lfr <= 400 else all_lfr[-400:]
        hist_stacked = feats.shape[0] - new_lfr
        new_stacked = new_lfr

        t1 = time.perf_counter()
        enc, alphas = self._backend._run_encoder(feats)
        enc_time = (time.perf_counter() - t1) * 1000
        self._total_enc_ms += enc_time

        logger.debug(
            f'[paraformer-stream] enc_call frames_total={feats.shape[0]} '
            f'new_frames={new_stacked} history_frames={hist_stacked}'
        )

        if enc is None or alphas is None:
            logger.warning("Encoder returned None for chunk %d", self._chunk_count)
            self._chunk_count += 1
            return

        # 4. CIF on NEW frames, but DEFER the last RIGHT_LOOKAHEAD LFR frames
        # (they have no right context in this chunk; they'll get full context
        # in the next chunk's encoder run and become "history" then).
        enc_t = enc[0]
        alphas_t = alphas[0]

        # cif_processed_lfr is absolute LFR index from utterance start.
        # Encoder runs on full all_lfr, so cif_start_local == absolute index.
        cif_end = enc_t.shape[0] - RIGHT_LOOKAHEAD_LFR
        cif_start = self._cif_processed_lfr
        if cif_end <= cif_start:
            self._chunk_count += 1
            return

        cif_enc = enc_t[cif_start:cif_end]
        cif_alphas = alphas_t[cif_start:cif_end]
        self._cif_processed_lfr = cif_end

        acoustic_embeds, self._carry_weight, self._carry_embed = cif(
            cif_enc, cif_alphas,
            carry_weight=self._carry_weight,
            carry_embed=self._carry_embed,
        )

        if len(acoustic_embeds) == 0:
            self._chunk_count += 1
            return

        # 6. Decoder cross-attends acoustic_embeds over FULL enc window
        # (history + new) — matches offline cross-attention memory shape
        t2 = time.perf_counter()
        sample_ids = self._backend._run_decoder(
            enc, enc.shape[1],
            acoustic_embeds, len(acoustic_embeds),
            self._cache,
        )
        dec_time = (time.perf_counter() - t2) * 1000
        self._total_dec_ms += dec_time
        self._chunk_count += 1

        if sample_ids is None:
            return

        # 7. Decode token IDs
        new_ids = sample_ids.tolist()
        self._all_token_ids.extend(new_ids)
        old_len = len(self._partial_text)
        self._partial_text = decode_ids(self._all_token_ids, self._tokens)

        if old_len != len(self._partial_text):
            logger.debug(
                "Chunk %d: %d new tokens, text += '%s'",
                self._chunk_count, len(new_ids),
                self._partial_text[old_len:],
            )

    def get_partial(self) -> tuple[str, bool]:
        text = self._partial_text
        is_endpoint = self._is_endpoint
        if is_endpoint:
            self._reset_utterance_state()
        return text, is_endpoint

    def finalize(self) -> str:
        """Process remaining audio tail + flush CIF -> final text.

        Bug G fix: full-utterance CMVN, slice (history+new) for encoder.
        """
        residual_audio = self._audio_buf
        if len(residual_audio) > 0:
            self._all_audio = np.concatenate([self._all_audio, residual_audio])

        if len(self._all_audio) >= WINDOW_SIZE:
            all_feats = compute_fbank(self._all_audio)
            all_lfr = stack_frames(all_feats)
            cur_total_lfr = all_lfr.shape[0]

            # At finalize, run encoder on the full utterance (or last 400 LFR
            # cap) and drain ALL pending CIF frames (no right-defer — there's
            # no more future audio).
            feats = all_lfr if cur_total_lfr <= 400 else all_lfr[-400:]
            enc, alphas = self._backend._run_encoder(feats)
            if enc is not None and alphas is not None:
                # CIF from where streaming left off to the very end
                cif_start = max(self._cif_processed_lfr,
                                feats.shape[0] - cur_total_lfr + cur_total_lfr - feats.shape[0])
                # Map absolute LFR index to position within feats window
                cif_start_local = max(0, cif_start - (cur_total_lfr - feats.shape[0]))
                cif_enc = enc[0][cif_start_local:]
                cif_alphas = alphas[0][cif_start_local:]
                self._cif_processed_lfr = cur_total_lfr

                acoustic_embeds, self._carry_weight, self._carry_embed = cif(
                    cif_enc, cif_alphas,
                    carry_weight=self._carry_weight,
                    carry_embed=self._carry_embed,
                )

                if len(acoustic_embeds) > 0:
                    sample_ids = self._backend._run_decoder(
                        enc, enc.shape[1],
                        acoustic_embeds, len(acoustic_embeds),
                        self._cache,
                    )
                    if sample_ids is not None:
                        self._all_token_ids.extend(sample_ids.tolist())
                        self._partial_text = decode_ids(self._all_token_ids, self._tokens)

        self._audio_buf = np.array([], dtype=np.float32)

        self._flush_cif_tail()

        text = self._partial_text
        chunk_count = self._chunk_count
        total_enc_ms = self._total_enc_ms
        total_dec_ms = self._total_dec_ms
        self._is_endpoint = True
        self._reset_utterance_state()

        logger.info(
            "Paraformer finalize: %d chunks, enc=%.0fms dec=%.0fms, text='%s'",
            chunk_count, total_enc_ms, total_dec_ms, text,
        )
        return text

    def _flush_cif_tail(self) -> None:
        """Fire final token for accumulated CIF weight if above threshold."""
        if self._carry_weight >= CIF_TAIL_THRESHOLD:
            acoustic_embed = self._carry_embed / self._carry_weight
            acoustic_embeds = acoustic_embed[np.newaxis, :]  # [1, 512]
            dummy_enc = np.zeros((1, 1, 512), dtype=np.float32)
            sample_ids = self._backend._run_decoder(
                dummy_enc, 1,
                acoustic_embeds, 1,
                self._cache,
            )
            if sample_ids is not None:
                new_ids = sample_ids.tolist()
                self._all_token_ids.extend(new_ids)
                self._partial_text = decode_ids(self._all_token_ids, self._tokens)

    def force_endpoint(self) -> str:
        """Trigger endpoint on demand (end_utterance WS command)."""
        self._flush_cif_tail()
        text = self._partial_text
        self._is_endpoint = True
        self._reset_utterance_state()
        return text


# ---------------------------------------------------------------------------
# ParaformerTRTBackend
# ---------------------------------------------------------------------------

class ParaformerTRTBackend(ASRBackend):

    def __init__(self):
        self._engines: dict[str, trt.IEngine] = {}
        self._contexts: dict[str, trt.IExecutionContext] = {}
        self._bindings: dict[str, dict] = {}
        self._dec_session = None  # ORT InferenceSession
        self._enc_ort_session = None  # ORT fallback for encoder
        self._enc_provider = "trt"  # "trt" or "ort_cuda"
        self._tokens: list[str] = []
        self._ready = False
        self._enc_active_profile: Optional[int] = None

    # -- Properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "paraformer_trt"

    @property
    def providers(self) -> dict[str, str]:
        """Return provider labels for each component."""
        return {"encoder": self._enc_provider, "decoder": "ort_cuda"}

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.STREAMING}

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def is_ready(self) -> bool:
        return self._ready

    # -- Lifecycle ----------------------------------------------------------

    def preload(self) -> None:
        if not _HAS_TRT:
            raise RuntimeError("TensorRT + CUDA Python not available")

        # Validate files
        for label, path in [("encoder engine", ENC_ENGINE_PATH),
                            ("decoder ONNX", DEC_ONNX_PATH)]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Paraformer {label} not found: {path}")
        if not os.path.isfile(TOKENS_PATH):
            raise FileNotFoundError(f"Paraformer tokens not found: {TOKENS_PATH}")

        # Load tokens
        self._tokens = load_tokens(TOKENS_PATH)
        logger.info("Loaded %d tokens from %s", len(self._tokens), TOKENS_PATH)

        # -- Load encoder TRT engine --
        self._engines["enc"] = _load_trt_engine(ENC_ENGINE_PATH)
        eng = self._engines["enc"]
        tensor_names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
        logger.info("Encoder engine (%d I/O): %s", len(tensor_names), tensor_names)
        self._contexts["enc"] = eng.create_execution_context()

        # Validate encoder TRT engine with a warmup run. If it produces NaN,
        # fall back to ORT CUDA EP (FP16 engines can overflow on Jetson).
        warmup_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        warmup_feats = compute_fbank(warmup_audio)
        warmup_feats = stack_frames(warmup_feats)
        n_warmup = min(warmup_feats.shape[0], 40)
        warmup_feats = warmup_feats[:n_warmup]

        enc, alphas = self._run_encoder_trt(warmup_feats)
        if enc is not None and alphas is not None and not np.isnan(alphas).any():
            logger.info("Encoder TRT engine validated (no NaN)")
            self._enc_provider = "trt"
        else:
            logger.warning(
                "Encoder TRT engine produces NaN, falling back to ORT CUDA EP. "
                "Rebuild with --bf16 or --best precision to restore TRT."
            )
            self._enc_provider = "ort_cuda"
            import onnxruntime
            enc_ort_opts = onnxruntime.SessionOptions()
            enc_ort_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            enc_ort_opts.log_severity_level = 3
            self._enc_ort_session = onnxruntime.InferenceSession(
                ENC_ONNX_PATH,
                sess_options=enc_ort_opts,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            logger.info("Encoder ORT session loaded (providers: %s)", self._enc_ort_session.get_providers())
            # Warmup ORT encoder
            self._run_encoder_ort(warmup_feats)

        # -- Load decoder ONNX via ORT CUDA EP --
        import onnxruntime
        dec_opts = onnxruntime.SessionOptions()
        dec_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        dec_opts.log_severity_level = 3  # only errors
        self._dec_session = onnxruntime.InferenceSession(
            DEC_ONNX_PATH,
            sess_options=dec_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        dec_inputs = [(n.name, n.shape, n.type) for n in self._dec_session.get_inputs()]
        dec_outputs = [(n.name, n.shape, n.type) for n in self._dec_session.get_outputs()]
        logger.info("Decoder ORT session (%d in / %d out):",
                    len(dec_inputs), len(dec_outputs))
        for n, s, t in dec_inputs:
            logger.info("  IN  %s %s %s", n, s, t)
        for n, s, t in dec_outputs:
            logger.info("  OUT %s %s %s", n, s, t)

        # Warmup decoder with minimal shapes
        dummy_enc = np.zeros((1, 1, 512), dtype=np.float32)
        dummy_ae = np.zeros((1, 512), dtype=np.float32)
        dummy_cache = [np.zeros((1, 512, 10), dtype=np.float32) for _ in range(16)]
        self._run_decoder(dummy_enc, 1, dummy_ae, 1, dummy_cache)

        logger.info("Paraformer TRT backend ready (encoder=%s, decoder=ort_cuda)", self._enc_provider)
        self._ready = True

    # -- Public API ---------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if not self._ready:
            raise RuntimeError("Paraformer TRT backend not loaded; call preload() first")

        # Load WAV from bytes
        try:
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception:
            import wave
            with wave.open(io.BytesIO(audio_bytes)) as w:
                sr = w.getframerate()
                raw = w.readframes(w.getnframes())
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to 16kHz
        if sr != SAMPLE_RATE:
            ratio = SAMPLE_RATE / sr
            new_len = int(len(data) * ratio)
            data = np.interp(
                np.linspace(0, len(data) - 1, new_len),
                np.arange(len(data)), data,
            ).astype(np.float32)

        # FBank -> stack -> encoder (full audio)
        feats = compute_fbank(data)
        feats = stack_frames(feats)

        # Use largest chunk that fits engine profile (max=400 in BF16 dual-profile build)
        # to give FSMN encoder full left context. Falls back to 40-frame chunking
        # only when audio exceeds engine max.
        ENGINE_MAX_FRAMES = 400
        chunk_frames = min(ENGINE_MAX_FRAMES, max(40, feats.shape[0]))
        all_token_ids: list[int] = []
        carry_w = 0.0
        carry_e = np.zeros(512, dtype=np.float32)
        cache = [np.zeros((1, 512, 10), dtype=np.float32) for _ in range(16)]

        for start in range(0, feats.shape[0], chunk_frames):
            chunk = feats[start:start + chunk_frames]
            if chunk.shape[0] < chunk_frames:
                pad = np.zeros((chunk_frames - chunk.shape[0], 560), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)

            enc, alphas = self._run_encoder(chunk)
            if enc is None:
                continue

            enc_t = enc[0]
            alphas_t = alphas[0]

            acoustic_embeds, carry_w, carry_e = cif(
                enc_t, alphas_t, carry_weight=carry_w, carry_embed=carry_e,
            )

            if len(acoustic_embeds) == 0:
                continue

            sample_ids = self._run_decoder(
                enc, alphas.shape[1],
                acoustic_embeds, len(acoustic_embeds),
                cache,
            )
            if sample_ids is not None:
                all_token_ids.extend(sample_ids.tolist())

        # Flush tail
        if carry_w >= CIF_TAIL_THRESHOLD:
            acoustic_embeds = (carry_e / carry_w)[np.newaxis, :]
            dummy_enc = np.zeros((1, 1, 512), dtype=np.float32)
            sample_ids = self._run_decoder(
                dummy_enc, 1, acoustic_embeds, 1, cache,
            )
            if sample_ids is not None:
                all_token_ids.extend(sample_ids.tolist())

        full_text = decode_ids(all_token_ids, self._tokens)
        return TranscriptionResult(text=full_text, language=language)

    def transcribe_audio(self, audio: np.ndarray, language: str = "auto") -> TranscriptionResult:
        """Transcribe float32 audio array (16kHz, [-1,1])."""
        if not self._ready:
            raise RuntimeError("Paraformer TRT backend not loaded; call preload() first")

        feats = compute_fbank(audio)
        feats = stack_frames(feats)

        ENGINE_MAX_FRAMES = 400
        chunk_frames = min(ENGINE_MAX_FRAMES, max(40, feats.shape[0]))
        all_text_parts = []
        carry_w = 0.0
        carry_e = np.zeros(512, dtype=np.float32)
        cache = [np.zeros((1, 512, 10), dtype=np.float32) for _ in range(16)]

        for start in range(0, feats.shape[0], chunk_frames):
            chunk = feats[start:start + chunk_frames]
            if chunk.shape[0] < chunk_frames:
                pad = np.zeros((chunk_frames - chunk.shape[0], 560), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)

            enc, alphas = self._run_encoder(chunk)
            if enc is None:
                continue

            enc_t = enc[0]
            alphas_t = alphas[0]

            acoustic_embeds, carry_w, carry_e = cif(
                enc_t, alphas_t, carry_weight=carry_w, carry_embed=carry_e,
            )

            if len(acoustic_embeds) == 0:
                continue

            sample_ids = self._run_decoder(
                enc, alphas.shape[1],
                acoustic_embeds, len(acoustic_embeds),
                cache,
            )
            if sample_ids is not None:
                new_ids = sample_ids.tolist()
                text = decode_ids(new_ids, self._tokens)
                if text:
                    all_text_parts.append(text)

        # Flush tail (mirror of transcribe() L674-684)
        if carry_w >= CIF_TAIL_THRESHOLD:
            acoustic_embeds = (carry_e / carry_w)[np.newaxis, :]
            dummy_enc = np.zeros((1, 1, 512), dtype=np.float32)
            sample_ids = self._run_decoder(
                dummy_enc, 1, acoustic_embeds, 1, cache,
            )
            if sample_ids is not None:
                text = decode_ids(sample_ids.tolist(), self._tokens)
                if text:
                    all_text_parts.append(text)

        full_text = "".join(all_text_parts)
        return TranscriptionResult(text=full_text, language=language)

    def create_stream(self, language: str = "auto") -> ASRStream:
        if not self._ready:
            raise RuntimeError("Paraformer TRT backend not loaded")
        return ParaformerTRTStream(self)

    # -- Internal: Encoder inference -----------------------------------

    @staticmethod
    def _cuda_err(result):
        """Normalize cuda-python return value to cudaError_t."""
        if isinstance(result, tuple):
            return result[0]
        return result

    def _run_encoder(self, feats: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Dispatch encoder to TRT or ORT based on runtime validation."""
        if self._enc_provider == "ort_cuda":
            return self._run_encoder_ort(feats)
        return self._run_encoder_trt(feats)

    def _run_encoder_ort(self, feats: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run encoder via ONNX Runtime CUDA EP.

        Args:
            feats: [feats_length, 560] float32

        Returns:
            enc: [1, feats_length, 512]
            alphas: [1, feats_length]
        """
        n_frames = feats.shape[0]
        enc_min_frames = 40
        orig_n_frames = n_frames

        if n_frames < enc_min_frames:
            pad_len = enc_min_frames - n_frames
            feats = np.pad(feats, ((0, pad_len), (0, 0)), mode="edge")
            n_frames = enc_min_frames

        speech = np.ascontiguousarray(feats[np.newaxis, :].astype(np.float32))
        speech_len = np.array([n_frames], dtype=np.int32)

        outputs = self._enc_ort_session.run(
            output_names=["enc", "enc_len", "alphas"],
            input_feed={"speech": speech, "speech_lengths": speech_len},
        )

        enc_out, enc_len_out, alphas_out = outputs

        if orig_n_frames < n_frames:
            enc_out = enc_out[:, :orig_n_frames, :]
            alphas_out = alphas_out[:, :orig_n_frames]

        return enc_out, alphas_out

    def _run_encoder_trt(self, feats: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run encoder TRT inference via set_tensor_address + execute_async_v3.

        TRT 10.x requires explicit tensor address registration before execute_async_v3.
        execute_v2(bindings) does NOT work with dynamic shapes in TRT 10.3.

        Args:
            feats: [feats_length, 560] float32

        Returns:
            enc: [1, feats_length, 512] or None on failure
            alphas: [1, feats_length] or None on failure
        """
        ctx = self._contexts["enc"]
        n_frames = feats.shape[0]
        n_profiles = getattr(self._engines["enc"], "num_optimization_profiles", 1)
        # Profile 1 covers 40..400 frames and matches streaming-with-history
        # (~140) plus offline chunks. Skip profile 0 to avoid TRT single-context
        # profile-switch races; pad short inputs up to profile 1 min instead.
        profile_idx = 1 if n_profiles > 1 else 0
        enc_min_frames = 40

        # Pad to engine min shape if needed
        orig_n_frames = n_frames
        if n_frames < enc_min_frames:
            pad_len = enc_min_frames - n_frames
            feats = np.pad(feats, ((0, pad_len), (0, 0)), mode="edge")
            n_frames = enc_min_frames

        key = f"enc_p{profile_idx}_{n_frames}"
        if key not in self._bindings:
            self._bindings[key] = self._alloc_enc_buffers(n_frames)

        bufs = self._bindings[key]

        # Execute on a fresh CUDA stream. Profile changes are asynchronous in
        # TRT 10, so synchronize the previous work before switching and the
        # profile-switch stream before setting dynamic input shapes.
        err, stream = cudart.cudaStreamCreate()
        if self._cuda_err(err) != 0:
            logger.error("cudaStreamCreate failed: %s", err)
            return None, None

        if self._enc_active_profile != profile_idx:
            cudart.cudaDeviceSynchronize()
            if hasattr(ctx, "set_optimization_profile_async"):
                success = ctx.set_optimization_profile_async(profile_idx, stream)
                if not success:
                    logger.error(
                        "Encoder TRT set_optimization_profile_async failed "
                        "(profile=%d, n_frames=%d)",
                        profile_idx, n_frames,
                    )
                    cudart.cudaStreamDestroy(stream)
                    return None, None
                cudart.cudaStreamSynchronize(stream)
            elif hasattr(ctx, "active_optimization_profile"):
                ctx.active_optimization_profile = profile_idx
            self._enc_active_profile = profile_idx

        # TRT 10.x: register tensor addresses
        ctx.set_tensor_address("speech", bufs["speech"])
        ctx.set_tensor_address("speech_lengths", bufs["speech_lengths"])
        ctx.set_tensor_address("enc", bufs["enc"])
        ctx.set_tensor_address("enc_len", bufs["enc_len"])
        ctx.set_tensor_address("alphas", bufs["alphas"])

        # Set dynamic shape profile
        ctx.set_input_shape("speech", (1, n_frames, 560))
        ctx.set_input_shape("speech_lengths", (1,))

        # Copy inputs to device
        speech = np.ascontiguousarray(feats[np.newaxis, :])
        err = cudart.cudaMemcpy(
            bufs["speech"], speech.ctypes.data, speech.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        if self._cuda_err(err) != 0:
            cudart.cudaStreamDestroy(stream)
            return None, None

        speech_len = np.array([n_frames], dtype=np.int32)
        err = cudart.cudaMemcpy(
            bufs["speech_lengths"], speech_len.ctypes.data, speech_len.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        if self._cuda_err(err) != 0:
            cudart.cudaStreamDestroy(stream)
            return None, None

        success = ctx.execute_async_v3(stream)
        cudart.cudaStreamSynchronize(stream)
        cudart.cudaStreamDestroy(stream)

        if not success:
            logger.error("Encoder TRT execute_async_v3 failed (n_frames=%d)", n_frames)
            return None, None

        enc_out = np.empty((1, n_frames, 512), dtype=np.float32)
        err = cudart.cudaMemcpy(
            enc_out.ctypes.data, bufs["enc"], enc_out.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        if self._cuda_err(err) != 0:
            return None, None

        alphas_out = np.empty((1, n_frames), dtype=np.float32)
        err = cudart.cudaMemcpy(
            alphas_out.ctypes.data, bufs["alphas"], alphas_out.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        if self._cuda_err(err) != 0:
            return None, None

        # Trim padded frames
        if orig_n_frames < n_frames:
            enc_out = enc_out[:, :orig_n_frames, :]
            alphas_out = alphas_out[:, :orig_n_frames]

        return enc_out, alphas_out

    def _alloc_enc_buffers(self, n_frames: int) -> dict:
        bufs = {}
        bufs["speech"] = self._cuda_malloc(1 * n_frames * 560 * 4)
        bufs["speech_lengths"] = self._cuda_malloc(4)
        bufs["enc"] = self._cuda_malloc(1 * n_frames * 512 * 4)
        bufs["enc_len"] = self._cuda_malloc(4)
        bufs["alphas"] = self._cuda_malloc(1 * n_frames * 4)
        return bufs

    def _cuda_malloc(self, nbytes: int) -> int:
        err, ptr = cudart.cudaMalloc(nbytes)
        if err != 0:
            raise RuntimeError(f"cudaMalloc({nbytes}) failed: {err}")
        return int(ptr)

    # -- Internal: Decoder ORT-CUDA inference ------------------------------

    def _run_decoder(
        self,
        enc: np.ndarray,
        enc_len: int,
        acoustic_embeds: np.ndarray,
        acoustic_embeds_len: int,
        cache: list[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Run decoder via ONNX Runtime CUDA EP.

        Args:
            enc: [1, enc_len, 512] encoder output
            enc_len: int
            acoustic_embeds: [num_tokens, 512]
            acoustic_embeds_len: int
            cache: list of 16 [1, 512, 10] cache tensors

        Returns:
            sample_ids: [n_tokens] int64 token IDs, or None on failure
        """
        if self._dec_session is None:
            logger.error("Decoder ORT session not loaded")
            return None

        n_tokens = acoustic_embeds.shape[0]

        # Pad acoustic_embeds to 1 token if empty (shouldn't happen at call site)
        if n_tokens == 0:
            return np.array([], dtype=np.int64)

        try:
            # Build ORT input dict
            ort_inputs = {
                "enc": np.ascontiguousarray(enc),
                "enc_len": np.array([enc_len], dtype=np.int32),
                "acoustic_embeds": np.ascontiguousarray(acoustic_embeds[np.newaxis, :]),
                "acoustic_embeds_len": np.array([acoustic_embeds_len], dtype=np.int32),
            }
            for i in range(16):
                ort_inputs[f"in_cache_{i}"] = np.ascontiguousarray(cache[i])

            # Single run: fetch sample_ids + all updated caches
            output_names = ["sample_ids"] + [f"out_cache_{i}" for i in range(16)]
            outputs = self._dec_session.run(
                output_names=output_names,
                input_feed=ort_inputs,
            )

            sample_ids = outputs[0][0]  # [n_tokens] int64
            for i in range(16):
                oc = outputs[1 + i]
                if oc.shape != cache[i].shape:
                    cache[i] = oc
                else:
                    cache[i][:] = oc

            return sample_ids

        except Exception as e:
            logger.error("Decoder ORT inference failed: %s", e)
            return None
