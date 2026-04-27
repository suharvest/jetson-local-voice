"""Qwen3-ASR backend — C++ pipeline (encoder + prefill + TRT decoder) via pybind11.

Supports: OFFLINE, STREAMING, MULTI_LANGUAGE, LANGUAGE_ID
Models loaded once at preload(), stays resident.

The C++ pipeline (ASRPipeline) handles encoder, prefill, and TRT decode loop.
Python handles: audio loading, mel computation, prompt construction, tokenizer decode.
Falls back to pure-Python ORT if C++ module not available.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import wave
from collections import deque
from typing import Optional

import numpy as np

from asr_backend import ASRBackend, ASRCapability, ASRStream, TranscriptionResult

logger = logging.getLogger(__name__)

# Dedicated CUDA stream for ORT CUDA EP to avoid legacy stream conflict with TRT capture
_ASR_CUDA_STREAM = None
_ASR_CUDA_STREAM_HANDLE = None


def _get_asr_cuda_stream_handle() -> str:
    global _ASR_CUDA_STREAM, _ASR_CUDA_STREAM_HANDLE
    if _ASR_CUDA_STREAM_HANDLE is None:
        from packaging import version
        import onnxruntime as ort

        if version.parse(ort.__version__) < version.parse("1.14"):
            raise RuntimeError(
                f"ORT {ort.__version__} < 1.14 does not support user_compute_stream"
            )
        from cuda import cudart

        err, stream = cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamCreateWithFlags failed: {err}")
        _ASR_CUDA_STREAM = stream
        _ASR_CUDA_STREAM_HANDLE = str(int(stream))
        logger.info("ASR ORT CUDA EP user_compute_stream=%s", _ASR_CUDA_STREAM_HANDLE)
    return _ASR_CUDA_STREAM_HANDLE


# Self-exported v2 with per-layer KV cache (validated with ORT 1.20 CUDA EP)
_BASE = os.environ.get("QWEN3_ASR_MODEL_BASE", "/opt/models/qwen3-asr-v2")

# Prompt constants (from andrewleech/qwen3-asr-onnx/src/prompt.py)
IM_START = 151644
IM_END = 151645
AUDIO_START = 151669
AUDIO_END = 151670
AUDIO_PAD = 151676
ASR_TEXT = 151704
EOS_IDS = {151643, 151645}

# ── True streaming parameters ──
CHUNK_SIZE_SEC = 0.4           # 400ms chunks (reduce partial frequency vs latency)
LEFT_CONTEXT_SEC = 1.0         # left-context audio ring buffer
ENCODER_HOP_SAMPLES = 1280     # hop_length(160) × encoder conv stride(8)
ROLLING_BUFFER_SEC = 5.0       # encoder output buffer for decoder prefill
PARTIAL_MAX_TOKENS = 12        # tokens per partial decode
DEDUP_MAX_OVERLAP = 12         # max token overlap for boundary dedup

# ── VAD endpoint parameters ──
VAD_ENDPOINT_SILENCE_MS = int(os.environ.get("VAD_ENDPOINT_SILENCE_MS", "1000"))  # trailing silence to trigger endpoint (default 1000ms)
VAD_MIN_UTTERANCE_S = 1.0      # min speech before endpoint eligible

# ── Legacy (still used by Qwen3ASRStream / offline path) ──
MEMORY_NUM = 3
ROLLBACK_TOKENS = 3
EOS_CONFIRM_COUNT = 2
STREAMING_MAX_TOKENS = 4

# Long-audio VAD split parameters (Qwen3-ASR emits premature '。'+EOS after
# ~6.5s of continuous speech; split longer audio at silence to stay under
# the safe boundary and avoid deterministic truncation.)
LONG_AUDIO_THRESHOLD_SEC = 6.0
VAD_MAX_SEG_SEC = 4.5        # conservative — leaves margin below Bug A boundary
VAD_MIN_SEG_SEC = 0.5        # allow finer splits when silence is available
VAD_FRAME_MS = 20           # webrtcvad frame size (10/20/30 supported)
VAD_AGGRESSIVENESS = 2      # 0-3; 2 = balanced for mixed noise conditions
VAD_MIN_SILENCE_MS = 150    # minimum silence run to count as a cut candidate


def _split_at_silence_vad(audio: np.ndarray, sr: int = 16000) -> list[np.ndarray]:
    """Split a long audio into segments at natural silence points via webrtcvad.

    Greedy: walk forward, try to cut at the silence point closest to
    VAD_MAX_SEG_SEC from the last cut, within [MIN_SEG, MAX_SEG] window. Falls
    back to a hard cut at MAX_SEG if no silence is found (e.g. continuous
    speech in noisy environments).
    """
    import webrtcvad

    max_seg = int(VAD_MAX_SEG_SEC * sr)
    min_seg = int(VAD_MIN_SEG_SEC * sr)

    if len(audio) <= max_seg:
        return [audio]

    frame_len = int(VAD_FRAME_MS * sr / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return [audio]

    pcm16 = (np.clip(audio[:n_frames * frame_len], -1.0, 1.0) * 32767).astype(np.int16)
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    is_speech = np.zeros(n_frames, dtype=bool)
    frame_bytes = frame_len * 2
    raw = pcm16.tobytes()
    for i in range(n_frames):
        is_speech[i] = vad.is_speech(raw[i * frame_bytes:(i + 1) * frame_bytes], sr)

    # Silence-run start indices (sample offset at run center) where run length
    # >= VAD_MIN_SILENCE_MS.
    min_run = max(1, VAD_MIN_SILENCE_MS // VAD_FRAME_MS)
    cut_candidates = []
    run_start = None
    for i in range(n_frames):
        if not is_speech[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_run:
                mid = (run_start + i) // 2
                cut_candidates.append(mid * frame_len)
            run_start = None
    if run_start is not None and n_frames - run_start >= min_run:
        mid = (run_start + n_frames) // 2
        cut_candidates.append(mid * frame_len)
    cut_candidates = np.array(cut_candidates, dtype=np.int64)

    cuts = [0]
    while len(audio) - cuts[-1] > max_seg:
        target = cuts[-1] + max_seg
        lo = cuts[-1] + min_seg
        hi = target
        mask = (cut_candidates >= lo) & (cut_candidates <= hi)
        if mask.any():
            # Cut at candidate closest to target (prefer later cuts for max chunk usage)
            pick = int(cut_candidates[mask][np.argmax(cut_candidates[mask])])
        else:
            # Fallback: hard cut at max boundary
            pick = int(target)
        cuts.append(pick)
    cuts.append(len(audio))

    # Post-process: merge short fragments into neighbors. Tiny chunks produce
    # hallucinations ("当前。") because the model misbehaves on incomplete
    # audio context. Merged segment may slightly exceed max_seg but stays
    # under the 6s Bug A boundary (max_seg + min_frag < 6.0s).
    min_frag = int(1.0 * sr)    # drop any mid segment <1.0s
    min_tail = int(2.0 * sr)    # tail must be >=2.0s, else merge into previous
    # 1. Merge mid fragments <min_frag into the preceding segment
    i = 1
    while i < len(cuts) - 1:
        if (cuts[i + 1] - cuts[i]) < min_frag:
            cuts.pop(i)  # drop cut point, merge into prev
        else:
            i += 1
    # 2. Tail merge if last segment too short
    while len(cuts) >= 3 and (cuts[-1] - cuts[-2]) < min_tail:
        cuts.pop(-2)
    return [audio[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]


def _join_segments(parts: list[str]) -> str:
    """Concatenate segment transcripts, stripping duplicate trailing
    punctuation that each segment's decoder may have added.
    """
    if not parts:
        return ""
    out = [parts[0].rstrip()]
    for p in parts[1:]:
        prev = out[-1]
        # If previous ended with '。'/'!'/'?' and next starts without one,
        # keep the period; otherwise just concat with no extra separator.
        out.append(p.strip())
    return "".join(out)


class Qwen3ASRStream(ASRStream):
    """Accumulate-then-transcribe streaming session for Qwen3-ASR.

    Audio chunks are buffered; full pipeline runs on finalize().
    """

    def __init__(self, backend: "Qwen3ASRBackend", language: str = "auto"):
        self._backend = backend
        self._language = language
        self._chunks: list[np.ndarray] = []
        self._total_samples = 0

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_len = int(len(samples) * ratio)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, new_len),
                np.arange(len(samples)),
                samples,
            ).astype(np.float32)
        self._chunks.append(samples)
        self._total_samples += len(samples)

    def finalize(self) -> str:
        if not self._chunks:
            return ""
        audio = np.concatenate(self._chunks)
        duration = len(audio) / 16000
        logger.info("Qwen3ASR stream finalize: %.1fs audio (%d samples)",
                     duration, len(audio))

        result = self._backend.transcribe_audio(audio, language=self._language)
        return result.text

    def get_partial(self) -> tuple[str, bool]:
        # V1: no partial results; could add duration-based hints later
        return "", False


def _is_cjk(ch: str) -> bool:
    """Check if character is CJK (Chinese/Japanese/Korean)."""
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF     # CJK Unified
            or 0x3040 <= cp <= 0x30FF   # Hiragana/Katakana
            or 0xAC00 <= cp <= 0xD7AF)  # Hangul


class Qwen3StreamingASRStream(ASRStream):
    """True streaming ASR: 250ms chunks + left-context encoder + VAD endpoint.

    Per-chunk: encode (left_context + new) → trim context frames → partial
    decode (8-12 tokens).  VAD detects trailing-silence endpoint and triggers
    a final decode with full token budget + EOS termination.
    """

    def __init__(self, backend: "Qwen3ASRBackend", language: str = "auto"):
        self._backend = backend
        self._language = language
        self._chunk_size_samples = int(CHUNK_SIZE_SEC * 16000)       # 400ms
        self._left_context_samples = int(LEFT_CONTEXT_SEC * 16000)   # 1.0s
        self._encoder_hop_samples = ENCODER_HOP_SAMPLES             # 1280

        # Audio accumulation (kept for left-context; trimmed periodically)
        self._audio_buf = np.array([], dtype=np.float32)
        self._utterance_audio_buffer: list[np.ndarray] = []
        self._processed_samples = 0

        # Rolling encoder output buffer (frames for decoder prefill)
        self._encoder_frames: list[np.ndarray] = []
        self._total_encoder_frames = 0
        self._max_encoder_frames = int(ROLLING_BUFFER_SEC * 13)

        # Output state
        self._committed_token_ids: list[int] = []
        self._archive_text: str = ""
        self._partial_text: str = ""
        self._partial_token_ids: list[int] = []
        self._episode_final: bool = False

        # Timing
        self._total_audio_s: float = 0.0
        self._total_enc_ms: float = 0.0
        self._total_dec_ms: float = 0.0
        self._n_chunks: int = 0

        # VAD (webrtcvad, 20ms frames)
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        except Exception:
            self._vad = None
        self._vad_speech_samples = 0
        self._vad_silence_samples = 0

        # Legacy tail-optimisation (reuse pre-encoded tail in finalize)
        self._tail_embd: Optional[np.ndarray] = None
        self._tail_audio_len = 0

    # ── Public API ────────────────────────────────────────────────

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_len = int(len(samples) * ratio)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, new_len),
                np.arange(len(samples)), samples,
            ).astype(np.float32)

        # Check for new utterance BEFORE appending — VAD state reset
        # must not discard current chunk (M1 fix).
        self._check_new_utterance_resume(samples)

        self._utterance_audio_buffer.append(samples.copy())
        self._audio_buf = np.concatenate([self._audio_buf, samples])

        # Track VAD on incoming audio (no longer resets buffer)
        self._run_vad(samples)

        # Process complete 250ms chunks
        chunk_sz = self._chunk_size_samples
        while len(self._audio_buf) - self._processed_samples >= chunk_sz:
            self._process_streaming_chunk()

        # Check VAD endpoint (trailing silence ≥500ms + utterance ≥1s)
        if self._check_vad_endpoint():
            self._do_final_decode()

        # Trim old audio periodically (keep 2× left-context headroom)
        max_prefix = self._left_context_samples + chunk_sz
        if self._processed_samples > max_prefix * 2:
            trim = self._processed_samples - max_prefix
            self._audio_buf = self._audio_buf[trim:]
            self._processed_samples -= trim

    def get_partial(self) -> tuple[str, bool]:
        text = self._archive_text
        if self._partial_text and not self._episode_final:
            sep = " " if not (self._archive_text and _is_cjk(self._archive_text[-1])) else ""
            text = (self._archive_text + sep + self._partial_text).strip()
        return text, self._episode_final

    def prepare_finalize(self) -> None:
        """No-op: offline final decode uses full-audio, not encoder tail."""
        pass

    def finalize(self) -> str:
        # Drain remaining unprocessed audio into encoder buffer
        while len(self._audio_buf) - self._processed_samples >= self._chunk_size_samples:
            self._process_streaming_chunk()

        tail_len = len(self._audio_buf) - self._processed_samples
        if tail_len > 0:
            if (self._tail_embd is not None
                    and self._tail_audio_len == tail_len):
                enc_out = self._tail_embd
            else:
                ctx_audio, n_ctx = self._get_left_context(self._processed_samples)
                new_audio = self._audio_buf[self._processed_samples:]
                enc_out = self._encode_with_context(ctx_audio, new_audio, n_ctx)
            if enc_out is not None and enc_out.shape[1] > 0:
                self._encoder_frames.append(enc_out)
                self._total_encoder_frames += enc_out.shape[1]
            self._processed_samples = len(self._audio_buf)

        # Force final decode if not already endpointed (offline full-audio)
        if not self._episode_final:
            self._archive_text = self._offline_final_text()
            self._episode_final = True

        logger.info(
            "Qwen3 streaming finalize: %d chunks, %.1fs audio, "
            "enc=%.0fms dec=%.0fms",
            self._n_chunks, self._total_audio_s,
            self._total_enc_ms, self._total_dec_ms,
        )

        # Sync ASR ORT user_compute_stream so all ASR GPU work is done
        # before the caller hands off to TTS TRT CUDA Graph capture.
        # Without this, V2V triggers CUDA error 906 (legacy stream
        # depend on capturing blocking stream).
        if _ASR_CUDA_STREAM is not None:
            try:
                from cuda import cudart
                cudart.cudaStreamSynchronize(_ASR_CUDA_STREAM)
            except Exception as e:  # pragma: no cover
                logger.warning("ASR stream sync on finalize failed: %s", e)

        return self._archive_text.strip()

    def force_endpoint(self) -> str:
        """Trigger endpoint on demand (e.g. end_utterance WS command)."""
        if not self._episode_final:
            self._do_final_decode()
        return self._archive_text

    # ── Internal: audio / encoder ─────────────────────────────────

    def _get_left_context(self, chunk_start: int) -> tuple[np.ndarray, int]:
        """Return (left-context audio, context_sample_count) for chunk_start."""
        ctx_start = max(0, chunk_start - self._left_context_samples)
        return self._audio_buf[ctx_start:chunk_start], chunk_start - ctx_start

    def _encode_with_context(
        self, ctx_audio: np.ndarray, new_audio: np.ndarray, n_context: int,
    ) -> Optional[np.ndarray]:
        """Encode (context + new) → [1, T_new, 1024], trimming context frames."""
        if len(ctx_audio) > 0:
            audio = np.concatenate([ctx_audio, new_audio])
        else:
            audio = new_audio
        mel = self._backend._compute_mel(audio)
        enc_out = self._backend._encoder.run(None, {"mel": mel})[0]  # [1, T', 1024]
        trim = int(n_context / self._encoder_hop_samples)
        if trim >= enc_out.shape[1]:
            return None
        return enc_out[:, trim:, :]

    # ── Internal: VAD ─────────────────────────────────────────────

    def _check_new_utterance_resume(self, samples: np.ndarray) -> bool:
        """Check if *samples* contain speech starting a new utterance.

        Must be called BEFORE appending to _utterance_audio_buffer so that
        the first chunk of the new utterance is preserved. When a new
        utterance is detected (previous episode finalized + current chunk
        contains speech), all per-utterance state is reset so the new
        chunk is the first sample of the next episode.

        Returns True if a new-utterance reset was performed.
        """
        if not self._episode_final:
            return False
        # Detect speech presence in the incoming chunk. Prefer webrtcvad if
        # available; otherwise fall back to a simple energy heuristic so
        # the bug-fix path still triggers in test/CI environments without
        # the VAD installed.
        has_speech = False
        if self._vad is not None:
            frame_len = int(VAD_FRAME_MS * 16000 / 1000)
            n = len(samples) // frame_len
            if n > 0:
                pcm = (np.clip(samples[:n * frame_len], -1, 1)
                       * 32767).astype(np.int16).tobytes()
                fb = frame_len * 2
                for i in range(n):
                    if self._vad.is_speech(pcm[i * fb:(i + 1) * fb], 16000):
                        has_speech = True
                        break
        else:
            # Energy fallback: RMS above a small threshold counts as speech.
            if len(samples) > 0:
                rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
                has_speech = rms > 1e-3
        if not has_speech:
            return False

        # Full per-utterance state reset.
        self._utterance_audio_buffer = []
        self._partial_text = ""
        self._partial_token_ids = []
        self._committed_token_ids = []
        self._encoder_frames = []
        self._total_encoder_frames = 0
        self._vad_speech_samples = 0
        self._vad_silence_samples = 0
        self._archive_text = ""
        self._episode_final = False
        return True

    def _run_vad(self, samples: np.ndarray) -> None:
        if self._vad is None:
            return
        frame_len = int(VAD_FRAME_MS * 16000 / 1000)
        n = len(samples) // frame_len
        if n == 0:
            return
        pcm = (np.clip(samples[:n * frame_len], -1, 1)
               * 32767).astype(np.int16).tobytes()
        fb = frame_len * 2
        for i in range(n):
            if self._vad.is_speech(pcm[i * fb:(i + 1) * fb], 16000):
                self._vad_speech_samples += frame_len
                self._vad_silence_samples = 0
            else:
                self._vad_silence_samples += frame_len

    def _check_vad_endpoint(self) -> bool:
        if self._vad is None or self._episode_final:
            return False
        return (
            self._vad_speech_samples >= int(VAD_MIN_UTTERANCE_S * 16000)
            and self._vad_silence_samples >= int(VAD_ENDPOINT_SILENCE_MS * 16000 / 1000)
        )

    # ── Internal: chunk processing ────────────────────────────────

    def _process_streaming_chunk(self) -> None:
        chunk_sz = self._chunk_size_samples
        new_start = self._processed_samples
        new_end = new_start + chunk_sz
        new_audio = self._audio_buf[new_start:new_end]
        ctx_audio, n_context = self._get_left_context(new_start)

        t0 = time.perf_counter()
        enc_out = self._encode_with_context(ctx_audio, new_audio, n_context)
        self._total_enc_ms += (time.perf_counter() - t0) * 1000

        if enc_out is None or enc_out.shape[1] == 0:
            self._processed_samples = new_end
            return

        # Append to rolling encoder buffer
        self._encoder_frames.append(enc_out)
        self._total_encoder_frames += enc_out.shape[1]
        while (self._total_encoder_frames > self._max_encoder_frames
               and len(self._encoder_frames) > 1):
            removed = self._encoder_frames.pop(0)
            self._total_encoder_frames -= removed.shape[1]

        self._processed_samples = new_end
        self._n_chunks += 1
        self._total_audio_s += chunk_sz / 16000

        # Partial decode (skip if endpoint already finalized)
        if self._episode_final or not self._encoder_frames:
            return

        all_frames = np.concatenate(self._encoder_frames, axis=1)
        t0 = time.perf_counter()
        partial_ids = self._decode_partial(all_frames)
        self._total_dec_ms += (time.perf_counter() - t0) * 1000

        if partial_ids:
            self._partial_token_ids = partial_ids
            decoded = self._backend._tokenizer.decode(partial_ids)
            if "<asr_text>" in decoded:
                decoded = decoded.split("<asr_text>", 1)[1]
            self._partial_text = decoded.strip()

        logger.debug("Chunk %d: partial='%s'", self._n_chunks,
                      self._partial_text[:50] if self._partial_text else "<empty>")

    def _offline_final_text(self) -> str:
        """Full-audio single-pass decode via offline transcribe_audio."""
        if not self._utterance_audio_buffer:
            return ''
        audio = np.concatenate(self._utterance_audio_buffer)
        return self._backend.transcribe_audio(audio, language=self._language).text

    def _do_final_decode(self) -> None:
        """Run final decode via offline full-audio single-pass decode."""
        self._archive_text = self._offline_final_text()

        # Reset state for next utterance
        self._partial_text = ""
        self._partial_token_ids = []
        self._committed_token_ids = []
        self._encoder_frames.clear()
        self._total_encoder_frames = 0
        self._vad_speech_samples = 0
        self._vad_silence_samples = 0
        self._utterance_audio_buffer.clear()
        self._episode_final = True

        logger.info(
            "VAD endpoint (offline): text='%s'",
            self._archive_text[-60:] if self._archive_text else "",
        )

    # ── Internal: decode ─────────────────────────────────────────

    def _decode_partial(self, all_frames: np.ndarray) -> list[int]:
        """Partial decode: limited budget, EOS-break OK but not committed."""
        return self._decode_window_internal(all_frames,
                                            max_tokens=PARTIAL_MAX_TOKENS)

    def _decode_final(self, all_frames: np.ndarray) -> list[int]:
        """Final decode: budget scales with audio duration, EOS terminates."""
        window_sec = all_frames.shape[1] / 13  # ~13 encoder fps
        max_tok = min(200, max(20, int(window_sec * 15)))
        return self._decode_window_internal(all_frames, max_tokens=max_tok)

    def _decode_window_internal(
        self, all_embd: np.ndarray, max_tokens: int,
    ) -> list[int]:
        """Prefill + autoregressive decode.  Returns token-ID list.

        TRT path preferred (40ms prefill); ORT fallback (200ms prefill).
        """
        audio_len = all_embd.shape[1]
        lang = self._language if self._language != "auto" else None
        prompt_ids = self._backend._build_prompt(audio_len, lang)
        seq_len = len(prompt_ids)
        audio_offset = prompt_ids.index(AUDIO_PAD)

        trt_dec = self._backend._decoder
        ort_dec = self._backend._decoder_ort
        if ort_dec:
            first_input = ort_dec.get_inputs()[0]
            model_dtype = (
                np.float16 if first_input.type == "tensor(float16)" else np.float32
            )
        else:
            model_dtype = np.float32

        embed_tokens = self._backend._embed_tokens
        input_embeds = np.zeros((1, seq_len, 1024), dtype=np.float32)
        for i, tid in enumerate(prompt_ids):
            input_embeds[0, i] = embed_tokens[tid]
        audio_end = min(audio_offset + audio_len, seq_len)
        input_embeds[0, audio_offset:audio_end] = (
            all_embd[0, :audio_end - audio_offset].astype(np.float32)
        )

        output_ids: list[int] = []

        # ── TRT prefill ──
        if trt_dec and seq_len <= getattr(self._backend, '_trt_max_seq', 500):
            result = trt_dec.prefill(input_embeds)
            logits = result["logits"]
            for _step in range(max_tokens):
                next_token = int(np.argmax(logits[0, -1, :]))
                if next_token in EOS_IDS:
                    break
                output_ids.append(next_token)
                embeds = embed_tokens[next_token].astype(np.float32)[
                    np.newaxis, np.newaxis, :
                ]
                logits = trt_dec.decode_step(embeds, 151936)

        # ── ORT fallback ──
        elif ort_dec:
            n_layers, H, dh = 28, 8, 128
            valid_names = [i.name for i in ort_dec.get_inputs()]
            prefill_in: dict = {
                "input_embeds": input_embeds.astype(model_dtype),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
            }
            for layer in range(n_layers):
                for kv_type in ("past_key_", "past_value_"):
                    name = f"{kv_type}{layer}"
                    if name in valid_names:
                        prefill_in[name] = np.zeros(
                            (1, H, 0, dh), dtype=model_dtype,
                        )
            valid = {n: v for n, v in prefill_in.items() if n in valid_names}
            outputs = ort_dec.run(None, valid)
            out_map = dict(zip([o.name for o in ort_dec.get_outputs()], outputs))
            logits = out_map.get("logits")
            kv: dict = {}
            for k, v in out_map.items():
                if k.startswith("new_past_"):
                    kv[k.replace("new_past_", "past_")] = v
                elif k.startswith("present_"):
                    kv[k.replace("present_", "past_")] = v
                elif k.startswith("past_"):
                    kv[k] = v

            for _step in range(max_tokens):
                next_token = int(np.argmax(logits[0, -1, :]))
                if next_token in EOS_IDS:
                    break
                output_ids.append(next_token)
                embeds = embed_tokens[next_token].astype(model_dtype)[
                    np.newaxis, np.newaxis, :
                ]
                cur_pos = seq_len + len(output_ids)
                step_in = {
                    "input_embeds": embeds,
                    "position_ids": np.array([[cur_pos]], dtype=np.int64),
                }
                step_in.update(kv)
                step_out = ort_dec.run(None, step_in)
                step_map = dict(
                    zip([o.name for o in ort_dec.get_outputs()], step_out)
                )
                logits = step_map.get("logits")
                kv = {}
                for k, v in step_map.items():
                    if k.startswith("new_past_"):
                        kv[k.replace("new_past_", "past_")] = v
                    elif k.startswith("present_"):
                        kv[k.replace("present_", "past_")] = v
                    elif k.startswith("past_"):
                        kv[k] = v
        else:
            logger.warning("No decoder available for streaming")
            return []

        return output_ids

    # ── Internal: dedup ───────────────────────────────────────────

    def _dedup_boundary_tokens(
        self, archive_ids: list[int], new_ids: list[int],
        max_overlap: int = DEDUP_MAX_OVERLAP,
    ) -> list[int]:
        """Return suffix of *new_ids* with the longest prefix overlap removed.

        Compares token IDs (not strings), so the match is exact.
        """
        if not archive_ids or not new_ids:
            return new_ids
        limit = min(len(archive_ids), len(new_ids), max_overlap)
        for k in range(limit, 0, -1):
            if archive_ids[-k:] == new_ids[:k]:
                return new_ids[k:]
        return new_ids

    # ── Legacy helpers (retained for reference) ──────────────────

    def _apply_rollback(self, text: str) -> str:
        """Strip last tokens to remove boundary jitter. Adaptive for short text."""
        if not text or ROLLBACK_TOKENS <= 0:
            return text
        ids = self._backend._tokenizer.encode(text).ids
        n_rollback = min(ROLLBACK_TOKENS, max(1, len(ids) // 3))
        if len(ids) <= n_rollback:
            return ""
        return self._backend._tokenizer.decode(ids[:-n_rollback])

    @staticmethod
    def _local_agreement(prev: str, curr: str) -> str:
        """Longest common prefix between two outputs for stability."""
        if not prev:
            return curr
        min_len = min(len(prev), len(curr))
        i = 0
        while i < min_len and prev[i] == curr[i]:
            i += 1
        result = curr[:i]
        if i < len(curr) and i > 0 and not _is_cjk(curr[i - 1]):
            last_space = result.rfind(" ")
            if last_space > 0:
                result = result[:last_space + 1]
        return result


class Qwen3ASRBackend(ASRBackend):

    def __init__(self):
        self._encoder = None
        self._decoder = None      # C++ TRT decoder (qwen3_speech_engine.TRTDecoder)
        self._decoder_ort = None  # ORT fallback decoder
        self._embed_tokens = None
        self._tokenizer = None
        self._ready = False

    @property
    def name(self) -> str:
        return "qwen3_asr"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.STREAMING, ASRCapability.MULTI_LANGUAGE, ASRCapability.LANGUAGE_ID}

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        logger.info("Loading Qwen3-ASR from %s", _BASE)
        t0 = time.time()

        # Find TRT engine path
        engine_path = None
        for engine_name in ["asr_decoder_bf16.engine", "asr_decoder_fp16.engine"]:
            p = os.path.join(_BASE, engine_name)
            if os.path.exists(p):
                engine_path = p
                break

        # ── 1. ORT encoder (CUDA EP) — load before TRT to avoid CUDA state pollution ──
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_options = [
            {"device_id": 0, "user_compute_stream": _get_asr_cuda_stream_handle()},
            {},
        ]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        logger.info("Loading encoder...")
        for enc_name in ["encoder_fp16.onnx", "encoder.onnx"]:
            enc_path = os.path.join(_BASE, enc_name)
            if os.path.exists(enc_path):
                self._encoder = ort.InferenceSession(
                    enc_path, so, providers=providers, provider_options=provider_options
                )
                logger.info("Encoder loaded: %s", enc_name)
                break

        # ── 2. Embed tokens ──
        emb_path = os.path.join(_BASE, "embed_tokens.bin")
        if os.path.exists(emb_path):
            self._embed_tokens = np.fromfile(emb_path, dtype=np.float16).reshape(-1, 1024).astype(np.float32)

        # ── 3. TRT decoder (preferred — supports prefill + decode_step) ──
        if engine_path:
            try:
                import qwen3_speech_engine
                self._decoder = qwen3_speech_engine.TRTDecoder(
                    engine_path, 28, 1024, 8, 128, 151936, 500)
                self._trt_max_seq = 500
                self._decoder.enable_cuda_graph(True)
                logger.info("TRT decoder loaded (CUDA Graph enabled): %s", engine_path)
            except Exception as e:
                logger.warning("TRT decoder %s failed: %s", engine_path, e)

        # ── 4. ORT decoder fallback (only if TRT decoder not available) ──
        if self._decoder is None:
            so_dec = ort.SessionOptions()
            so_dec.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            for dec_name in ["decoder_unified.onnx", "decoder_step.onnx"]:
                path = os.path.join(_BASE, dec_name)
                if os.path.exists(path):
                    try:
                        self._decoder_ort = ort.InferenceSession(
                            path, so_dec, providers=providers, provider_options=provider_options
                        )
                        logger.info("ORT decoder loaded (fallback): %s", dec_name)
                        break
                    except Exception as e:
                        logger.warning("ORT decoder %s failed: %s", dec_name, e)

        # ── 5. Tokenizer ──
        tok_path = os.path.join(_BASE, "tokenizer.json")
        if os.path.exists(tok_path):
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(tok_path)

        offline_backend = "TRT" if self._decoder else "ORT" if self._decoder_ort else "none"
        stream_backend = offline_backend
        logger.info("Qwen3-ASR loaded in %.1fs (backend: %s)",
                     time.time() - t0, offline_backend)

        # ── 6. Warm-up ──
        if self._encoder and (self._decoder or self._decoder_ort):
            logger.info("Warming up encoder + decoder...")
            t_warm = time.time()
            try:
                dummy_audio = np.zeros(16000, dtype=np.float32)  # 1s silence
                mel = self._compute_mel(dummy_audio)
                self._encoder.run(None, {"mel": mel})
                if self._decoder_ort:
                    sess = self._decoder_ort
                    valid_names = [i.name for i in sess.get_inputs()]
                    first_input = sess.get_inputs()[0]
                    dtype = np.float16 if first_input.type == "tensor(float16)" else np.float32
                    warm_in = {
                        "input_embeds": np.zeros((1, 2, 1024), dtype=dtype),
                        "position_ids": np.arange(2, dtype=np.int64).reshape(1, -1),
                    }
                    for layer in range(28):
                        for prefix in ("past_key_", "past_value_"):
                            name = f"{prefix}{layer}"
                            if name in valid_names:
                                warm_in[name] = np.zeros((1, 8, 0, 128), dtype=dtype)
                    valid = {n: v for n, v in warm_in.items() if n in valid_names}
                    sess.run(None, valid)
                logger.info("Warm-up done in %.1fs", time.time() - t_warm)
            except Exception as e:
                logger.warning("Warm-up failed: %s", e)

        self._ready = True

    def create_stream(self, language: str = "auto") -> ASRStream:
        if not self._ready:
            raise RuntimeError("Qwen3-ASR backend not ready")
        has_encoder = self._encoder is not None
        has_decoder = (self._decoder is not None or self._decoder_ort is not None)
        has_embeds = self._embed_tokens is not None
        if has_encoder and has_decoder and has_embeds:
            logger.info("Creating real streaming ASR session (sliding window)")
            return Qwen3StreamingASRStream(self, language=language)
        logger.info("Creating accumulate-then-transcribe ASR session")
        return Qwen3ASRStream(self, language=language)

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        audio = self._bytes_to_float(audio_bytes)
        return self.transcribe_audio(audio, language=language)

    def transcribe_audio(self, audio: np.ndarray, language: str = "auto") -> TranscriptionResult:
        """Transcribe float32 audio array (16kHz, [-1,1]).

        Audio longer than LONG_AUDIO_THRESHOLD_SEC is split at silence via
        webrtcvad before transcription: Qwen3-ASR emits a premature '。' then
        EOS after ~6.5s of continuous speech, causing deterministic mid-audio
        truncation. Each segment stays under VAD_MAX_SEG_SEC and is
        transcribed independently; results are concatenated.
        """
        t_total = time.perf_counter()
        if len(audio) / 16000 <= LONG_AUDIO_THRESHOLD_SEC:
            mel = self._compute_mel(audio)
            return self._transcribe_python(mel, audio, language, t_total)
        return self._transcribe_segmented(audio, language, t_total)

    def _transcribe_segmented(self, audio, language, t_total):
        """Split long audio at silence, transcribe each segment, concatenate."""
        try:
            segments = _split_at_silence_vad(audio)
        except Exception as e:
            logger.warning("VAD split failed (%s); falling back to single-pass", e)
            mel = self._compute_mel(audio)
            return self._transcribe_python(mel, audio, language, t_total)

        logger.info("Long audio %.2fs split into %d segments: %s",
                    len(audio) / 16000, len(segments),
                    [round(len(s) / 16000, 2) for s in segments])

        parts, total_tokens = [], 0
        for seg in segments:
            if len(seg) < 800:  # skip too-short segments (<50ms)
                continue
            mel = self._compute_mel(seg)
            sub = self._transcribe_python(mel, seg, language, time.perf_counter())
            if sub.text:
                parts.append(sub.text)
            total_tokens += sub.meta.get("n_tokens", 0)

        audio_dur = len(audio) / 16000
        total_ms = (time.perf_counter() - t_total) * 1000
        return TranscriptionResult(
            text=_join_segments(parts),
            duration=round(audio_dur, 3),
            inference_time=round(total_ms / 1000, 3),
            rtf=round(total_ms / 1000 / audio_dur, 3) if audio_dur > 0 else 0,
            n_tokens=total_tokens,
            per_token_ms=round(total_ms / max(total_tokens, 1), 1),
            backend="TRT" if self._decoder else "ORT",
        )

    def _transcribe_python(self, mel, audio, language, t_total):
        """Python ORT encoder + TRT/ORT prefill + decode."""
        # 1. Encoder
        t0 = time.perf_counter()
        enc_out = self._encoder.run(None, {"mel": mel})[0]  # [1, T', 1024]
        enc_ms = (time.perf_counter() - t0) * 1000
        audio_len = enc_out.shape[1]

        # 2. Prompt
        lang = language if language != "auto" else None
        prompt_ids = self._build_prompt(audio_len, lang)
        seq_len = len(prompt_ids)
        audio_offset = prompt_ids.index(AUDIO_PAD)

        # 3. Build input_embeds
        input_embeds = np.zeros((1, seq_len, 1024), dtype=np.float32)
        for i, tid in enumerate(prompt_ids):
            input_embeds[0, i] = self._embed_tokens[tid]
        audio_end = min(audio_offset + audio_len, seq_len)
        input_embeds[0, audio_offset:audio_end] = enc_out[0, :audio_end - audio_offset]

        # 4. Prefill + decode
        t0 = time.perf_counter()
        output_ids = []

        # === TRT prefill path (fast, preferred) ===
        if self._decoder and seq_len <= getattr(self, '_trt_max_seq', 500):
            result = self._decoder.prefill(input_embeds)
            logits = result["logits"]  # [1, S, vocab_size]

            for step in range(200):
                next_token = int(np.argmax(logits[0, -1, :]))
                if next_token in EOS_IDS:
                    break
                output_ids.append(next_token)
                embeds = self._embed_tokens[next_token].astype(np.float32)[np.newaxis, np.newaxis, :]
                logits = self._decoder.decode_step(embeds, 151936)

        # === ORT fallback path ===
        elif self._decoder_ort:
            ort_dec = self._decoder_ort
            first_input = ort_dec.get_inputs()[0]
            model_dtype = np.float16 if first_input.type == "tensor(float16)" else np.float32
            valid_names = [i.name for i in ort_dec.get_inputs()]

            n_layers, H, dh = 28, 8, 128
            prefill_in = {
                "input_embeds": input_embeds.astype(model_dtype),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
            }
            for layer in range(n_layers):
                for kv_type in ("past_key_", "past_value_"):
                    name = f"{kv_type}{layer}"
                    if name in valid_names:
                        prefill_in[name] = np.zeros((1, H, 0, dh), dtype=model_dtype)

            valid = {n: v for n, v in prefill_in.items() if n in valid_names}
            outputs = ort_dec.run(None, valid)
            out_map = dict(zip([o.name for o in ort_dec.get_outputs()], outputs))
            logits = out_map.get("logits")
            kv = {}
            for k, v in out_map.items():
                if k.startswith("new_past_"):
                    kv[k.replace("new_past_", "past_")] = v
                elif k.startswith("present_"):
                    kv[k.replace("present_", "past_")] = v
                elif k.startswith("past_"):
                    kv[k] = v

            for step in range(200):
                next_token = int(np.argmax(logits[0, -1, :]))
                if next_token in EOS_IDS:
                    break
                output_ids.append(next_token)
                embeds = self._embed_tokens[next_token].astype(model_dtype)[np.newaxis, np.newaxis, :]
                cur_pos = seq_len + step
                step_in = {"input_embeds": embeds,
                           "position_ids": np.array([[cur_pos]], dtype=np.int64)}
                step_in.update(kv)
                step_out = ort_dec.run(None, step_in)
                step_map = dict(zip([o.name for o in ort_dec.get_outputs()], step_out))
                logits = step_map.get("logits")
                kv = {}
                for k, v in step_map.items():
                    if k.startswith("new_past_"):
                        kv[k.replace("new_past_", "past_")] = v
                    elif k.startswith("present_"):
                        kv[k.replace("present_", "past_")] = v
                    elif k.startswith("past_"):
                        kv[k] = v
        else:
            logger.warning("No decoder available for offline transcription")

        decode_ms = (time.perf_counter() - t0) * 1000
        total_ms = (time.perf_counter() - t_total) * 1000

        # A0 instrumentation: per-call enc / prefill+decode split for offline path
        logger.info(
            "offline transcribe: enc=%.0fms prefill+dec=%.0fms tokens=%d audio=%.2fs",
            enc_ms, decode_ms, len(output_ids), len(audio) / 16000,
        )

        # Decode text
        text = self._tokenizer.decode(output_ids) if self._tokenizer else f"[{len(output_ids)} tokens]"
        if "<asr_text>" in text:
            text = text.split("<asr_text>", 1)[1]

        audio_dur = len(audio) / 16000
        per_tok = decode_ms / max(len(output_ids), 1)
        backend = "TRT" if self._decoder else "ORT"

        return TranscriptionResult(
            text=text.strip(),
            duration=round(audio_dur, 3),
            inference_time=round(total_ms / 1000, 3),
            rtf=round(total_ms / 1000 / audio_dur, 3) if audio_dur > 0 else 0,
            n_tokens=len(output_ids),
            per_token_ms=round(per_tok, 1),
            backend=backend,
        )

    def _build_prompt(self, audio_len, language=None):
        ids = [
            IM_START, 9125, 198, IM_END, 198,
            IM_START, 882, 198,
            AUDIO_START, *([AUDIO_PAD] * audio_len), AUDIO_END, IM_END, 198,
            IM_START, 77091, 198,
        ]
        # ASR_TEXT anchor MUST be appended unconditionally. Without it on
        # language=None/auto, decoder has no anchor and hallucinates loops
        # like "CurrentCurrentCurrent..." on English audio. Confirmed via
        # LibriSpeech eval: 15/15 real WAVs produced that garbage.
        if language and self._tokenizer:
            lang_ids = self._tokenizer.encode(f"language {language}").ids
            ids.extend(lang_ids)
        ids.append(ASR_TEXT)
        return ids

    def _compute_mel(self, audio):
        from transformers import WhisperFeatureExtractor
        # Use chunk_length matching actual audio to avoid excessive padding
        audio_secs = len(audio) / 16000
        chunk_len = min(30, int(audio_secs) + 1)  # Round up, max 30s
        # Cache the feature extractor for common chunk lengths
        if not hasattr(self, '_mel_cache'):
            self._mel_cache = {}
        if chunk_len not in self._mel_cache:
            self._mel_cache[chunk_len] = WhisperFeatureExtractor(
                feature_size=128, sampling_rate=16000,
                n_fft=400, hop_length=160, chunk_length=chunk_len)
        fe = self._mel_cache[chunk_len]
        features = fe(audio, sampling_rate=16000, return_tensors="np")
        return features["input_features"]  # [1, 128, T]

    @staticmethod
    def _bytes_to_float(audio_bytes):
        try:
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        except ImportError:
            bio = io.BytesIO(audio_bytes)
            with wave.open(bio) as w:
                sr = w.getframerate()
                raw = w.readframes(w.getnframes())
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if sr != 16000:
            ratio = 16000 / sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(np.linspace(0, len(audio)-1, new_len), np.arange(len(audio)), audio)
        return audio.astype(np.float32)
