"""Qwen3-TTS backend via C++ TRT native engine (pybind11).

Supports: BASIC_TTS, VOICE_CLONE, MULTI_LANGUAGE
Models loaded once at preload(), C++ engine stays resident in memory.
"""

from __future__ import annotations

import logging
import io
import os
import time
import wave
from typing import Optional

import numpy as np

from tts_backend import TTSBackend, TTSCapability

logger = logging.getLogger(__name__)


def _sampling_uniforms(seed: int, max_frames: int) -> list[float]:
    if seed == 0 or os.environ.get("QWEN3_TTS_NUMPY_SAMPLING", "1").lower() in ("0", "false", "no"):
        return []
    # One primary sample plus up to 15 CP residual samples per frame, with
    # slack for EOS/edge cases. This preserves the old Python reference
    # RandomState(seed).choice() consumption order without calling Python from C++.
    n = max(64, (max_frames + 4) * 16)
    return np.random.RandomState(seed).random_sample(n).astype(float).tolist()


def _detect_language(text: str) -> str:
    """Simple language detection — returns config-compatible language strings."""
    for ch in text:
        cp = ord(ch)
        # CJK Unified Ideographs
        if 0x4E00 <= cp <= 0x9FFF:
            return "chinese"
        # Japanese Hiragana / Katakana
        if 0x3040 <= cp <= 0x30FF:
            return "japanese"
        # Korean Hangul
        if 0xAC00 <= cp <= 0xD7AF:
            return "korean"
    return "english"


def _pcm16_to_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return out.getvalue()


def _contains_cjk(text: str) -> bool:
    return any(0x4E00 <= ord(ch) <= 0x9FFF for ch in text)


def _is_ascii_word_char(ch: str) -> bool:
    return ch.isascii() and (ch.isalnum() or ch in "_+-./")


def _is_ascii_word_boundary(text: str, idx: int) -> bool:
    before = text[idx - 1] if idx > 0 else ""
    after = text[idx] if idx < len(text) else ""
    return not (_is_ascii_word_char(before) and _is_ascii_word_char(after))


def _safe_product_tts_cut(text: str, max_chars: int) -> int:
    if len(text) <= max_chars:
        return len(text)

    # Prefer whitespace or mixed CJK/ASCII boundaries. Splitting inside English
    # product/device names such as "Jetson" is audible and hurts ASR.
    floor = max(1, int(max_chars * 0.55))
    for idx in range(max_chars, floor - 1, -1):
        if not _is_ascii_word_boundary(text, idx):
            continue
        prev_ch = text[idx - 1]
        next_ch = text[idx] if idx < len(text) else ""
        if prev_ch.isspace() or next_ch.isspace():
            return idx
        if (prev_ch.isascii() and not next_ch.isascii()) or (not prev_ch.isascii() and next_ch.isascii()):
            return idx

    idx = max_chars
    while idx < len(text) and not _is_ascii_word_boundary(text, idx):
        idx += 1
    return idx if idx < len(text) else len(text)


def _split_product_tts_text(text: str, max_chars: int = 20) -> list[str]:
    """Split only where the Qwen3-TTS product path is known to lose conditioning.

    Chinese comma/full-stop continuations regress in a single Talker session on
    the current Jetson path. Keep punctuation with each segment and synthesize
    segments with the same seed for ASR-correct output.
    """
    text = text.strip()
    if not text or not _contains_cjk(text):
        return [text] if text else []
    breaks = set("，,、。！？!?；;：:\n")
    raw_parts: list[str] = []
    current: list[str] = []
    for ch in text:
        current.append(ch)
        part = "".join(current).strip()
        if ch in breaks:
            if part:
                raw_parts.append(part)
            current.clear()
    tail = "".join(current).strip()
    if tail:
        raw_parts.append(tail)

    parts: list[str] = []
    punctuation_only = set("，,、。！？!?；;：:")
    for raw in raw_parts:
        rest = raw
        while len(rest) > max_chars:
            cut = _safe_product_tts_cut(rest, max_chars)
            part = rest[:cut]
            rest = rest[cut:]
            if part.strip():
                parts.append(part)
        if rest.strip():
            if parts and all(ch in punctuation_only for ch in rest):
                parts[-1] += rest
            else:
                parts.append(rest)
    return parts or [text]


def _segment_pause_ms(segment: str) -> int:
    comma_pause = int(os.environ.get("QWEN3_TTS_PRODUCT_COMMA_PAUSE_MS", "120"))
    hard_pause = int(os.environ.get("QWEN3_TTS_PRODUCT_HARD_PAUSE_MS", "180"))
    if segment.rstrip().endswith(("。", "！", "？", "!", "?", "；", ";")):
        return max(0, hard_pause)
    return max(0, comma_pause)


def _concat_wav_bytes(parts: list[bytes], pauses_ms: Optional[list[int]] = None) -> bytes:
    non_empty = [part for part in parts if part]
    if not non_empty:
        return b""
    if len(non_empty) == 1:
        return non_empty[0]

    params = None
    frames: list[bytes] = []
    for idx, part in enumerate(non_empty):
        with wave.open(io.BytesIO(part), "rb") as reader:
            current = (
                reader.getnchannels(),
                reader.getsampwidth(),
                reader.getframerate(),
                reader.getcomptype(),
                reader.getcompname(),
            )
            if params is None:
                params = current
            elif current != params:
                raise RuntimeError(f"Cannot concatenate WAV segments with different formats: {current} != {params}")
            frames.append(reader.readframes(reader.getnframes()))
            if pauses_ms and idx < len(non_empty) - 1:
                pause_samples = int(current[2] * max(0, pauses_ms[idx]) / 1000)
                if pause_samples > 0:
                    frames.append(b"\x00" * pause_samples * current[0] * current[1])

    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(params[0])
        writer.setsampwidth(params[1])
        writer.setframerate(params[2])
        writer.setcomptype(params[3], params[4])
        for frame_bytes in frames:
            writer.writeframes(frame_bytes)
    return out.getvalue()

# Paths — all under /opt/models/qwen3-tts (persistent volume)
_BASE = os.environ.get("QWEN3_MODEL_BASE", "/opt/models/qwen3-tts")
QWEN3_SHERPA_DIR = os.environ.get("QWEN3_SHERPA_DIR", os.path.join(_BASE, "onnx"))
QWEN3_MODEL_DIR = os.environ.get("QWEN3_MODEL_DIR", os.path.join(_BASE, "onnx"))
QWEN3_TALKER_ENGINE = os.environ.get("QWEN3_TALKER_ENGINE", os.path.join(_BASE, "engines", "talker_decode_bf16.engine"))
QWEN3_CP_ENGINE = os.environ.get("QWEN3_CP_ENGINE", os.path.join(_BASE, "engines", "cp_bf16.engine"))
QWEN3_SPEAKER_ENCODER = os.environ.get("QWEN3_SPEAKER_ENCODER", os.path.join(_BASE, "onnx", "speaker_encoder.onnx"))
QWEN3_TOKENIZER_DIR = os.environ.get("QWEN3_TOKENIZER_DIR", os.path.join(_BASE, "tokenizer"))
QWEN3_EXTRACT_SCRIPT = os.environ.get("QWEN3_EXTRACT_SCRIPT", os.path.join(_BASE, "extract_speaker_emb.py"))


class Qwen3TRTBackend(TTSBackend):
    """Qwen3-TTS via C++ TRT native inference (pybind11 module, models resident)."""

    def __init__(self):
        self._engine = None  # qwen3_speech_engine.Pipeline
        self._tokenizer = None
        self._ready = False

    @property
    def name(self) -> str:
        return "qwen3_trt"

    @property
    def capabilities(self) -> set[TTSCapability]:
        caps = {TTSCapability.BASIC_TTS, TTSCapability.MULTI_LANGUAGE,
                TTSCapability.STREAMING}
        if os.path.exists(QWEN3_SPEAKER_ENCODER):
            caps.add(TTSCapability.VOICE_CLONE)
        return caps

    @property
    def sample_rate(self) -> int:
        return 24000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        """Load C++ TRT engine + tokenizer. Models stay resident."""
        def _meminfo(tag):
            try:
                with open("/proc/self/status") as f:
                    s = {l.split(":")[0]: l.split(":")[1].strip() for l in f if ":" in l}
                with open("/proc/meminfo") as f:
                    sys_avail = next((l for l in f if l.startswith("MemAvailable:")), "?").strip()
                logger.info("[MEM:%s] RSS=%s HWM=%s | sys %s", tag,
                            s.get("VmRSS","?"), s.get("VmHWM","?"), sys_avail)
            except Exception as e:
                logger.warning("[MEM:%s] failed: %s", tag, e)

        # Background sampler: every 500ms during heavy load, prints RSS + sys avail
        import threading
        _stop = threading.Event()
        def _sampler():
            i = 0
            while not _stop.wait(0.5):
                _meminfo(f"tts_load_t{i*500}ms")
                i += 1
        sampler = threading.Thread(target=_sampler, daemon=True)

        _meminfo("tts_start")

        # Verify files
        for path, desc in [
            (QWEN3_TALKER_ENGINE, "talker engine"),
            (QWEN3_CP_ENGINE, "CP engine"),
            (os.path.join(_BASE, "engines", "vocoder_fp16.engine"), "vocoder engine"),
            (os.path.join(QWEN3_SHERPA_DIR, "config.json"), "config.json (authoritative)"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {desc}: {path}")

        # Load tokenizer
        self._load_tokenizer()
        _meminfo("tts_after_tokenizer")

        # Load C++ engine (this is the heavy part: ~25s for model loading + embed table)
        logger.info("Loading Qwen3 TRT engine (this takes ~25s)...")
        t0 = time.time()

        # INT8 EOS compensation: subtract from EOS logit before sampling
        # Set via TTS_INT8_EOS_LOGIT_OFFSET env var (negative = harder EOS)
        os.environ.setdefault("TTS_INT8_EOS_LOGIT_OFFSET", "-10.0")

        sampler.start()
        try:
            import qwen3_speech_engine
            self._engine = qwen3_speech_engine.Pipeline(
                QWEN3_MODEL_DIR, QWEN3_SHERPA_DIR,
                QWEN3_TALKER_ENGINE, QWEN3_CP_ENGINE,
            )
        finally:
            _stop.set()
        logger.info("Qwen3 TRT engine loaded in %.1fs", time.time() - t0)
        _meminfo("tts_after_pipeline")

        # Enable cached CUDA Graph for talker decode:
        # First request populates cache (~10ms extra per unique seq_len),
        # subsequent requests replay cached graphs (~3ms vs 26ms baseline).
        # Memory-tight setups (Orin Nano 8GB multilanguage) can disable via
        # TTS_TALKER_CUDA_GRAPH=0 to save ~150-300 MB graph cache during decode.
        if os.environ.get("TTS_TALKER_CUDA_GRAPH", "1") == "1":
            try:
                self._engine.enable_cuda_graph(True)
                logger.info("CUDA Graph enabled for talker decode (cached mode)")
            except Exception as e:
                logger.warning("CUDA Graph enable failed (non-fatal): %s", e)
        else:
            logger.info("CUDA Graph disabled for talker decode (TTS_TALKER_CUDA_GRAPH=0)")

        self._ready = True
        _meminfo("tts_ready")

    def _load_tokenizer(self):
        vocab_path = os.path.join(QWEN3_TOKENIZER_DIR, "vocab.json")
        merges_path = os.path.join(QWEN3_TOKENIZER_DIR, "merges.txt")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Tokenizer not found: {vocab_path}")

        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel

        self._tokenizer = Tokenizer(BPE(vocab_path, merges_path))
        self._tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        logger.info("Tokenizer loaded from %s", QWEN3_TOKENIZER_DIR)

    def _tokenize(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self._tokenizer.encode(text).ids

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if language is None:
            language = _detect_language(text)

        token_ids = self._tokenize(text)
        requested_max_frames = int(kwargs.get("max_audio_length", kwargs.get("max_frames", 200)))
        vocoder_cap = int(os.environ.get("TTS_TRT_VOCODER_MAX_FRAMES", "100"))
        use_trt_vocoder = os.environ.get("TTS_VOCODER_TRT", "1").lower() not in ("0", "false", "no")
        expected_frames = max(50, len(token_ids) * 3)
        collect_streaming = (
            use_trt_vocoder
            and requested_max_frames > vocoder_cap
            and expected_frames > vocoder_cap
            and os.environ.get("QWEN3_TTS_OFFLINE_STREAMING_FOR_LONG", "1").lower()
            not in ("0", "false", "no")
        )
        seed = int(kwargs.get("seed", os.environ.get("SEEED_LOCAL_VOICE_TTS_SEED", "0")))
        segment_text = kwargs.get("product_segment_text", True)
        if isinstance(segment_text, str):
            segment_text = segment_text.lower() not in ("0", "false", "no")
        if segment_text and os.environ.get("QWEN3_TTS_PRODUCT_SEGMENT_TEXT", "0").lower() not in ("0", "false", "no"):
            max_chars = int(os.environ.get("QWEN3_TTS_PRODUCT_SEGMENT_MAX_CHARS", "20"))
            segments = _split_product_tts_text(text, max_chars=max_chars)
            if len(segments) > 1:
                start = time.time()
                wav_parts: list[bytes] = []
                segment_meta: list[dict] = []
                segment_kwargs = dict(kwargs)
                segment_kwargs["product_segment_text"] = False
                segment_kwargs.pop("seed", None)
                segment_kwargs.setdefault("max_audio_length", min(requested_max_frames, vocoder_cap if use_trt_vocoder else requested_max_frames))
                for segment in segments:
                    wav, meta = self.synthesize(
                        segment,
                        speaker_id=speaker_id,
                        speed=speed,
                        pitch_shift=pitch_shift,
                        language=language,
                        seed=seed,
                        **segment_kwargs,
                    )
                    wav_parts.append(wav)
                    segment_meta.append({"text": segment, **meta})
                pauses_ms = [_segment_pause_ms(segment) for segment in segments[:-1]]
                wav_bytes = _concat_wav_bytes(wav_parts, pauses_ms=pauses_ms)
                duration = 0.0
                samples = 0
                with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
                    samples = reader.getnframes()
                    duration = samples / reader.getframerate() if reader.getframerate() else 0.0
                elapsed = time.time() - start
                return wav_bytes, {
                    "duration": round(duration, 3),
                    "inference_time": round(elapsed, 3),
                    "rtf": round(elapsed / duration, 3) if duration else 0,
                    "sample_rate": self.sample_rate,
                    "samples": samples,
                    "seed": seed,
                    "product_segmented": True,
                    "segment_count": len(segments),
                    "segment_pauses_ms": pauses_ms,
                    "segments": segment_meta,
                }

        if collect_streaming:
            start = time.time()
            pcm = b"".join(
                self.generate_streaming(
                    text,
                    language=language,
                    max_frames=requested_max_frames,
                    seed=seed,
                    first_chunk_frames=int(kwargs.get("first_chunk_frames", 25)),
                    chunk_frames=int(kwargs.get("chunk_frames", 25)),
                )
            )
            elapsed = time.time() - start
            duration = len(pcm) / 2 / self.sample_rate if pcm else 0.0
            return _pcm16_to_wav(pcm, self.sample_rate), {
                "duration": round(duration, 3),
                "inference_time": round(elapsed, 3),
                "rtf": round(elapsed / duration, 3) if duration else 0,
                "sample_rate": self.sample_rate,
                "samples": len(pcm) // 2,
                "seed": seed,
                "offline_collected_streaming": True,
            }

        max_frames = requested_max_frames
        if use_trt_vocoder:
            max_frames = min(max_frames, vocoder_cap)
        random_values = _sampling_uniforms(seed, max_frames)

        start = time.time()
        result = self._engine.synthesize(
            text=text,
            lang=language,
            token_ids=token_ids,
            max_frames=max_frames,
            seed=seed,
            random_values=random_values,
        )
        elapsed = time.time() - start

        wav_bytes = result["wav_bytes"]
        duration = result.get("duration", 0)

        meta = {
            "duration": round(duration, 3),
            "inference_time": round(elapsed, 3),
            "rtf": round(result.get("rtf", 0), 3),
            "sample_rate": self.sample_rate,
            "n_frames": result.get("n_frames", 0),
            "per_step_ms": round(result.get("per_step_ms", 0), 1),
            "seed": seed,
        }
        return wav_bytes, meta

    def clone_voice(
        self,
        text: str,
        speaker_embedding: bytes,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if language is None:
            language = _detect_language(text)

        token_ids = self._tokenize(text)

        start = time.time()
        result = self._engine.synthesize_clone(
            text=text,
            lang=language,
            token_ids=token_ids,
            speaker_emb_bytes=speaker_embedding,
        )
        elapsed = time.time() - start

        wav_bytes = result["wav_bytes"]
        duration = result.get("duration", 0)

        meta = {
            "duration": round(duration, 3),
            "inference_time": round(elapsed, 3),
            "rtf": round(result.get("rtf", 0), 3),
            "sample_rate": self.sample_rate,
        }
        return wav_bytes, meta

    def generate_streaming(self, text: str, **kwargs):
        """Yield PCM int16 chunks via C++ callback-based streaming.

        The C++ engine calls our callback per chunk during generation,
        and we yield each chunk as it arrives via a thread-safe queue.

        Args:
            text: Text to synthesize
            language: Language code (auto-detected if not specified)
            speaker_embedding: Optional speaker embedding bytes for voice cloning
            first_chunk_frames: Frames in first chunk (default 10)
            chunk_frames: Frames in subsequent chunks (default 25)
            max_frames: Maximum total frames (default 200)
        """
        import queue as queue_mod
        import threading

        language = kwargs.get("language") or _detect_language(text)
        speaker_embedding = kwargs.get("speaker_embedding")
        first_chunk_frames = kwargs.get("first_chunk_frames", 5)
        chunk_frames = kwargs.get("chunk_frames", 25)
        max_frames = kwargs.get("max_frames", 200)
        seed = int(kwargs.get("seed", os.environ.get("SEEED_LOCAL_VOICE_TTS_SEED", "0")))
        random_values = _sampling_uniforms(seed, int(max_frames))

        token_ids = self._tokenize(text)

        chunk_queue: queue_mod.Queue = queue_mod.Queue()
        SENTINEL = object()

        def _on_chunk(chunk_dict):
            """Called from C++ thread per audio chunk."""
            wav_bytes = chunk_dict["wav_bytes"]
            if len(wav_bytes) > 44:
                chunk_queue.put(wav_bytes[44:])  # Strip WAV header -> raw PCM

        def _run_engine():
            try:
                if speaker_embedding:
                    self._engine.synthesize_streaming_clone_callback(
                        text=text,
                        lang=language,
                        token_ids=token_ids,
                        speaker_emb_bytes=speaker_embedding,
                        callback=_on_chunk,
                        first_chunk_frames=first_chunk_frames,
                        chunk_frames=chunk_frames,
                        max_frames=max_frames,
                        seed=seed,
                    )
                else:
                    self._engine.synthesize_streaming_callback(
                        text=text,
                        lang=language,
                        token_ids=token_ids,
                        callback=_on_chunk,
                        first_chunk_frames=first_chunk_frames,
                        chunk_frames=chunk_frames,
                        max_frames=max_frames,
                        seed=seed,
                        random_values=random_values,
                    )
            finally:
                chunk_queue.put(SENTINEL)

        threading.Thread(target=_run_engine, daemon=True).start()

        while True:
            item = chunk_queue.get()
            if item is SENTINEL:
                break
            yield item

    def extract_speaker_embedding(self, audio_wav_bytes: bytes) -> bytes:
        """Extract speaker embedding using Python mel computation + ORT."""
        import tempfile
        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            wf.write(audio_wav_bytes)
            wav_path = wf.name
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as ef:
            emb_path = ef.name

        try:
            result = subprocess.run(
                ["python3", QWEN3_EXTRACT_SCRIPT,
                 "--audio", wav_path,
                 "--model", QWEN3_SPEAKER_ENCODER,
                 "--output", emb_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Embedding extraction failed: {result.stderr}")
            return open(emb_path, "rb").read()
        finally:
            for p in [wav_path, emb_path]:
                if os.path.exists(p):
                    os.unlink(p)
