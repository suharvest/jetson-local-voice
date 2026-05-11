"""TTS backend via TRT-Edge-LLM C++ binary (qwen3_tts_inference).

Calls the binary per-request with temp-file I/O.
Supports: BASIC_TTS, MULTI_LANGUAGE
Audio output: WAV via Code2Wav (vocoder) engine.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import base64
import io
import wave
from collections import deque
from typing import Optional
import importlib
import uuid

from tts_backend import TTSBackend, TTSCapability

from backends.trt_edge_llm_ipc import (
    TTS_BINARY,
    TTS_WORKER_BINARY,
    TTS_TALKER_DIR,
    TTS_CODE_PREDICTOR_DIR,
    TTS_CODE2WAV_DIR,
    TTS_TOKENIZER_DIR,
    PLUGIN_PATH,
    QWEN3_RUNTIME_PROFILE,
    qwen3_highperf_enabled,
    run_binary,
)

logger = logging.getLogger(__name__)

_QWEN3_TTS_MODEL_BASE = os.environ.get("JETSON_VOICE_TTS_MODEL_BASE", os.environ.get("QWEN3_MODEL_BASE", "/opt/models/qwen3-tts"))
_QWEN3_SPEAKER_ENCODER = os.environ.get(
    "QWEN3_SPEAKER_ENCODER", os.path.join(_QWEN3_TTS_MODEL_BASE, "onnx", "speaker_encoder.onnx")
)
_QWEN3_EXTRACT_SCRIPT = os.environ.get(
    "QWEN3_EXTRACT_SCRIPT", os.path.join(_QWEN3_TTS_MODEL_BASE, "extract_speaker_emb.py")
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no", "off")


def _tts_perf_profile() -> str:
    return os.environ.get("EDGE_LLM_TTS_PERF_PROFILE", "quality").strip().lower()


def _tts_fast_profile(profile: str) -> bool:
    return profile in ("fast", "v2v", "low_latency")


def _tts_balanced_or_fast_profile(profile: str) -> bool:
    return profile in ("balanced", "fast", "v2v", "low_latency")


def _tts_stateful_code2wav_enabled() -> bool:
    # Stateful Code2Wav is the validated low-latency streaming path. It keeps
    # subsequent chunks small and continuous instead of optimizing only TTFA.
    return _env_flag("EDGE_LLM_TTS_STATEFUL_CODE2WAV", default=qwen3_highperf_enabled())


def _detect_language(text: str) -> str:
    """Simple language detection — returns config-compatible language strings."""
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            return "chinese"
        if 0x3040 <= cp <= 0x30FF:
            return "japanese"
        if 0xAC00 <= cp <= 0xD7AF:
            return "korean"
    return "english"


def _contains_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF:
            return True
    return False


def _split_tts_text(text: str, max_chars: Optional[int] = None) -> list[str]:
    """Split long TTS text into independently stable synthesis requests."""
    normalized = " ".join(text.split()) if not _contains_cjk(text) else text.strip()
    if not normalized:
        return []

    if max_chars is None:
        env_name = "EDGE_LLM_TTS_CJK_SEGMENT_MAX_CHARS" if _contains_cjk(normalized) else "EDGE_LLM_TTS_SEGMENT_MAX_CHARS"
        default_chars = "48" if _contains_cjk(normalized) else "120"
        max_chars = int(os.environ.get(env_name, default_chars))
    max_chars = max(8, max_chars)

    hard_breaks = set("。！？!?；;\n")
    soft_breaks = set("，,、：:")
    segments: list[str] = []
    current: list[str] = []
    is_cjk = _contains_cjk(normalized)
    max_overrun = max(2, min(8, max_chars // 2))
    abbreviations = {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "st.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
    }

    def is_nonterminal_period(buffer: str, next_ch: str) -> bool:
        stripped = buffer.strip().lower()
        if next_ch.isdigit() and len(stripped) >= 2 and stripped[-2].isdigit():
            return True
        return any(stripped.endswith(abbrev) for abbrev in abbreviations)

    def flush() -> None:
        part = "".join(current).strip()
        current.clear()
        if part:
            segments.append(part)

    for idx, ch in enumerate(normalized):
        next_ch = normalized[idx + 1] if idx + 1 < len(normalized) else ""
        current.append(ch)
        if ch in hard_breaks:
            if not is_cjk and ch == "." and is_nonterminal_period("".join(current), next_ch):
                continue
            flush()
            continue
        current_text = "".join(current).strip()
        if len(current_text) >= max_chars:
            text_so_far = "".join(current)
            cut = max(text_so_far.rfind(p) for p in soft_breaks)
            if cut >= max_chars // 3:
                head = text_so_far[: cut + 1].strip()
                tail = text_so_far[cut + 1 :].lstrip()
                current.clear()
                if head:
                    segments.append(head)
                if tail:
                    current.extend(tail)
            else:
                if is_cjk and len(current_text) < max_chars + max_overrun:
                    continue
                flush()
    flush()

    if not is_cjk:
        packed: list[str] = []
        for part in segments:
            if len(part) <= max_chars:
                packed.append(part)
                continue
            words = part.split(" ")
            buf: list[str] = []
            for word in words:
                candidate = " ".join(buf + [word]).strip()
                if buf and len(candidate) > max_chars:
                    packed.append(" ".join(buf))
                    buf = [word]
                else:
                    buf.append(word)
            if buf:
                packed.append(" ".join(buf))
        segments = packed

    merged: list[str] = []
    min_chars = max(4, min(12, max_chars // 3))
    for part in segments:
        if merged and all(ch in hard_breaks or ch in soft_breaks for ch in part):
            merged[-1] = f"{merged[-1]}{part}"
            continue
        if merged and len(part) < min_chars and len(merged[-1]) + 1 + len(part) <= max_chars:
            sep = "" if _contains_cjk(part + merged[-1]) else " "
            merged[-1] = f"{merged[-1]}{sep}{part}"
        else:
            merged.append(part)
    return merged


def _segment_pause_ms(segment: str) -> int:
    if not segment:
        return 0
    pause_ms = int(os.environ.get("EDGE_LLM_TTS_SEGMENT_PAUSE_MS", "80"))
    hard_pause_ms = int(os.environ.get("EDGE_LLM_TTS_HARD_SEGMENT_PAUSE_MS", "120"))
    stripped = segment.rstrip()
    if stripped.endswith(("。", "！", "？", "!", "?", ";", "；")):
        return max(0, hard_pause_ms)
    if stripped.endswith(("，", ",", "、", "：", ":")):
        return max(0, pause_ms)
    return max(0, pause_ms)


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
            current = reader.getparams()
            comparable = (current.nchannels, current.sampwidth, current.framerate, current.comptype, current.compname)
            if params is None:
                params = comparable
            elif comparable != params:
                raise RuntimeError(f"Cannot concatenate WAV segments with different formats: {comparable} != {params}")
            frames.append(reader.readframes(reader.getnframes()))
            if pauses_ms and idx < len(non_empty) - 1:
                pause_samples = int(current.framerate * max(0, pauses_ms[idx]) / 1000)
                if pause_samples > 0:
                    frames.append(b"\x00" * pause_samples * current.nchannels * current.sampwidth)

    nchannels, sampwidth, framerate, comptype, compname = params
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(nchannels)
        writer.setsampwidth(sampwidth)
        writer.setframerate(framerate)
        writer.setcomptype(comptype, compname)
        for frame_bytes in frames:
            writer.writeframes(frame_bytes)
    return out.getvalue()


def _wav_duration_and_samples(wav_bytes: bytes) -> tuple[float, int]:
    if not wav_bytes:
        return 0.0, 0
    with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
        samples = reader.getnframes()
        rate = reader.getframerate()
    return (samples / rate if rate > 0 else 0.0), samples


def _pcm16_to_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw mono int16 little-endian PCM in a minimal WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


# Default sampling parameters
_DEFAULT_TEMPERATURE = float(os.environ.get("TTS_TALKER_TEMPERATURE", "0.9"))
_DEFAULT_TOP_K = int(os.environ.get("TTS_TALKER_TOP_K", "50"))
_DEFAULT_TOP_P = float(os.environ.get("TTS_TOP_P", "1.0"))
_DEFAULT_PREDICTOR_TEMPERATURE = float(os.environ.get("TTS_PREDICTOR_TEMPERATURE", "0.9"))
_DEFAULT_PREDICTOR_TOP_K = int(os.environ.get("TTS_PREDICTOR_TOP_K", "50"))
_DEFAULT_PREDICTOR_TOP_P = float(os.environ.get("TTS_PREDICTOR_TOP_P", "1.0"))
_DEFAULT_MAX_AUDIO_LENGTH = int(os.environ.get("TTS_MAX_AUDIO_LENGTH", "1024"))
_DEFAULT_MIN_AUDIO_LENGTH = int(os.environ.get("TTS_MIN_AUDIO_LENGTH", "30"))
_DEFAULT_REPETITION_PENALTY = float(os.environ.get("TTS_REPETITION_PENALTY", "1.05"))
_DEFAULT_CODEC_EOS_LOGIT_OFFSET = float(os.environ.get("TTS_CODEC_EOS_LOGIT_OFFSET", "0"))
_DEFAULT_SEGMENT_TEXT = os.environ.get("EDGE_LLM_TTS_SEGMENT_TEXT", "1").lower() not in ("0", "false", "no")


class TRTEdgeLLMTTSBackend(TTSBackend):
    """TTS via TRT-Edge-LLM qwen3_tts_inference subprocess."""

    def __init__(self):
        self._ready = False
        self._product_backend = None
        self._worker: Optional[subprocess.Popen] = None
        self._worker_lock = threading.Lock()
        self._worker_ready_meta: dict = {}
        self._worker_stderr_tail = deque(maxlen=80)

    # -- TTSBackend interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "trt_edgellm"

    @property
    def capabilities(self) -> set[TTSCapability]:
        caps = {TTSCapability.BASIC_TTS, TTSCapability.MULTI_LANGUAGE, TTSCapability.STREAMING}
        if self._product_backend is not None:
            caps |= self._product_backend.capabilities
        else:
            caps.add(TTSCapability.VOICE_CLONE)
        return caps

    @property
    def sample_rate(self) -> int:
        return 24000

    def is_ready(self) -> bool:
        return self._ready

    def _backend_mode(self) -> str:
        mode = os.environ.get("JETSON_VOICE_TTS_BACKEND", os.environ.get("EDGE_LLM_TTS_BACKEND", "edgellm_worker"))
        return mode.strip().lower().replace("-", "_")

    def _load_product_explicit_kv_backend(self):
        model_base = os.environ.get("JETSON_VOICE_TTS_MODEL_BASE", "/home/harvest/voice_test/models/qwen3-tts")
        overlay = os.environ.get("JETSON_VOICE_TTS_NATIVE_MODULE_DIR", "/home/harvest/voice_test/app_overlay")
        overlay_backends = os.path.join(overlay, "backends")
        for path in (overlay, overlay_backends):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

        os.environ.setdefault("QWEN3_MODEL_BASE", model_base)
        os.environ.setdefault("QWEN3_MODEL_DIR", os.path.join(model_base, "onnx"))
        os.environ.setdefault("QWEN3_SHERPA_DIR", os.path.join(model_base, "onnx"))
        os.environ.setdefault("QWEN3_TALKER_ENGINE", os.path.join(model_base, "engines", "talker_decode_bf16.engine"))
        os.environ.setdefault("QWEN3_CP_ENGINE", os.path.join(model_base, "engines", "cp_bf16.engine"))
        os.environ.setdefault("TTS_TALKER_CUDA_GRAPH", "0")

        module = importlib.import_module("backends.qwen3_trt")
        module = importlib.reload(module)
        backend = module.Qwen3TRTBackend()
        logger.info(
            "Using Jetson Voice product_explicit_kv TTS backend (model_base=%s, talker=%s)",
            model_base,
            os.environ.get("QWEN3_TALKER_ENGINE"),
        )
        backend.preload()
        return backend

    def preload(self) -> None:
        """Verify all required files exist."""
        mode = self._backend_mode()
        if mode in ("product_explicit_kv", "explicit_kv"):
            self._product_backend = self._load_product_explicit_kv_backend()
            self._ready = True
            return
        if mode not in ("edgellm", "edgellm_worker", "official"):
            raise ValueError(
                "Unsupported JETSON_VOICE_TTS_BACKEND/EDGE_LLM_TTS_BACKEND value "
                f"{mode!r}; expected edgellm_worker or product_explicit_kv"
            )

        explicit_talker_engine = os.environ.get("EDGE_LLM_TTS_TALKER_ENGINE")
        required = [
            (TTS_WORKER_BINARY if self._use_worker() else TTS_BINARY, "TTS binary"),
            (PLUGIN_PATH, "TRT-Edge-LLM plugin"),
            (os.path.join(TTS_TALKER_DIR, "config.json"), "talker config"),
            (os.path.join(TTS_TOKENIZER_DIR, "tokenizer.json"), "tokenizer"),
        ]
        if explicit_talker_engine:
            required.append((explicit_talker_engine, "explicit Talker engine"))
        else:
            required.append((os.path.join(TTS_TALKER_DIR, "llm.engine"), "talker engine"))
        missing = []
        for path, label in required:
            if not os.path.exists(path):
                missing.append(f"{label}: {path}")
        if missing:
            raise FileNotFoundError(
                "TTS preload failed — missing:\n  " + "\n  ".join(missing)
            )

        # Code2Wav is optional (graceful fallback)
        c2w_path = os.path.join(TTS_CODE2WAV_DIR, "code2wav.engine")
        if os.path.exists(c2w_path):
            logger.info("Code2Wav engine found at %s", c2w_path)
        else:
            logger.warning(
                "Code2Wav not found at %s — will output RVQ codes only",
                c2w_path,
            )

        logger.info(
            "TTS backend preload OK (profile=%s binary=%s talker=%s)",
            QWEN3_RUNTIME_PROFILE,
            TTS_WORKER_BINARY if self._use_worker() else TTS_BINARY,
            TTS_TALKER_DIR,
        )
        if self._use_worker():
            self._ensure_worker()
        self._ready = True

    def _use_worker(self) -> bool:
        return os.environ.get("EDGE_LLM_TTS_WORKER", "1").lower() not in ("0", "false", "no")

    def _worker_env(self) -> dict:
        env = os.environ.copy()
        env["EDGELLM_PLUGIN_PATH"] = PLUGIN_PATH
        # On the current W8A16 Talker path CUDA graph has not shown measurable
        # RTF gain, while it costs extra resident memory during dual ASR+TTS.
        env.setdefault("EDGE_LLM_TTS_CUDA_GRAPH", "0")
        env.setdefault("EDGE_LLM_TTS_LAZY_CODE2WAV", "0")
        env.setdefault("EDGE_LLM_TTS_STATEFUL_CODE2WAV", "1" if qwen3_highperf_enabled() else "0")
        stateful_engine_dir = os.environ.get(
            "EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR",
            "/tmp/qwen3_code2wav_stateful_engine",
        )
        if os.path.isdir(stateful_engine_dir):
            env.setdefault("EDGE_LLM_TTS_STATEFUL_CODE2WAV_ENGINE_DIR", stateful_engine_dir)
        if _tts_stateful_code2wav_enabled():
            env.setdefault("EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES", "0")
            env.setdefault("QWEN3_TTS_CP_DECODE_CUDA_GRAPH", "1")
            env.setdefault("QWEN3_TTS_ACTIVE_CP_GROUPS", "13")
        else:
            # Legacy stateless vocoder100 path: first chunk 50 frames, then 97
            # new frames + 3 context.
            env.setdefault("EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES", "3")
        vocab_pruned = os.environ.get("EDGE_LLM_TTS_VOCAB_PRUNED")
        if vocab_pruned is not None:
            env.setdefault("QWEN3_TTS_VOCAB_PRUNED", vocab_pruned)
        else:
            env.setdefault("QWEN3_TTS_VOCAB_PRUNED", "0")
        return env

    def _worker_stderr_snip(self) -> str:
        return "".join(self._worker_stderr_tail)[-2000:] or "(empty)"

    def _drain_worker_stderr(self) -> None:
        worker = self._worker
        if worker is None or worker.stderr is None:
            return
        for line in worker.stderr:
            self._worker_stderr_tail.append(line)
            if "[JV_MEM]" in line:
                logger.info("TTS worker: %s", line.rstrip())
            else:
                logger.debug("TTS worker stderr: %s", line.rstrip())

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.poll() is None:
            return
        cmd = [
            TTS_WORKER_BINARY,
            "--talkerEngineDir",
            TTS_TALKER_DIR,
            "--codePredictorEngineDir",
            TTS_CODE_PREDICTOR_DIR,
            "--tokenizerDir",
            TTS_TOKENIZER_DIR,
            "--code2wavEngineDir",
            TTS_CODE2WAV_DIR,
        ]
        optional_flags = [
            ("EDGE_LLM_TTS_TALKER_BACKEND", "--qwen3TtsTalkerBackend"),
            ("EDGE_LLM_TTS_TALKER_ENGINE", "--qwen3TtsTalkerEngine"),
            ("EDGE_LLM_TTS_CODE_PREDICTOR_BACKEND", "--codePredictorBackend"),
            ("EDGE_LLM_TTS_TEXT_PROJECTION", "--qwen3TtsTextProjection"),
            ("EDGE_LLM_TTS_PROMPT_KV_CACHE", "--qwen3TtsPromptKvCache"),
        ]
        for env_name, flag in optional_flags:
            value = os.environ.get(env_name)
            if value:
                cmd.extend([flag, value])
        self._worker = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._worker_env(),
        )
        threading.Thread(target=self._drain_worker_stderr, name="tts-worker-stderr", daemon=True).start()
        assert self._worker.stdout is not None
        ready_line = self._worker.stdout.readline()
        if not ready_line:
            raise RuntimeError(f"TTS worker failed to start: {self._worker_stderr_snip()}")
        ready = json.loads(ready_line)
        if ready.get("event") != "ready":
            raise RuntimeError(f"TTS worker did not become ready: {ready}")
        self._worker_ready_meta = ready

    def _synthesize_worker(self, text: str, language: Optional[str], **kwargs) -> tuple[bytes, dict]:
        if _tts_stateful_code2wav_enabled():
            # The C++ worker rejects oneshot requests with "Stateful Code2Wav
            # currently requires stream=true". Drive the same streaming path
            # that /tts/stream uses, accumulate raw PCM, and wrap as a single
            # WAV so the public non-streaming /tts endpoint stays usable.
            return self._synthesize_worker_via_stream(text, language=language, **kwargs)
        req_id = uuid.uuid4().hex
        with tempfile.NamedTemporaryFile(prefix="trt_edgellm_tts_", suffix=".wav", delete=False) as f:
            output_file = f.name
        request = {
            "id": req_id,
            "text": text,
            "output_file": output_file,
            "language": language or _detect_language(text),
            "talker_temperature": _DEFAULT_TEMPERATURE,
            "talker_top_k": _DEFAULT_TOP_K,
            "talker_top_p": _DEFAULT_TOP_P,
            "repetition_penalty": _DEFAULT_REPETITION_PENALTY,
            "codec_eos_logit_offset": _DEFAULT_CODEC_EOS_LOGIT_OFFSET,
            "predictor_temperature": _DEFAULT_PREDICTOR_TEMPERATURE,
            "predictor_top_k": _DEFAULT_PREDICTOR_TOP_K,
            "predictor_top_p": _DEFAULT_PREDICTOR_TOP_P,
            "max_audio_length": kwargs.get("max_audio_length", _DEFAULT_MAX_AUDIO_LENGTH),
            "min_audio_length": kwargs.get("min_audio_length", _DEFAULT_MIN_AUDIO_LENGTH),
        }
        speaker_embedding = kwargs.get("speaker_embedding")
        if speaker_embedding:
            request["speaker_embedding_b64"] = base64.b64encode(speaker_embedding).decode("ascii")
        with self._worker_lock:
            self._ensure_worker()
            assert self._worker is not None and self._worker.stdin is not None and self._worker.stdout is not None
            t0 = time.time()
            self._worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._worker.stdin.flush()
            line = self._worker.stdout.readline()
            elapsed = time.time() - t0
        if not line:
            self._worker = None
            raise RuntimeError(f"TTS worker exited before response: {self._worker_stderr_snip()}")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"TTS worker failed: {response}")
        with open(response["output_file"], "rb") as f:
            wav_bytes = f.read()
        try:
            os.unlink(response["output_file"])
        except OSError:
            pass
        audio_s = float(response.get("audio_s", 0.0))
        meta = {
            "inference_time_s": round(elapsed, 3),
            "sample_rate": int(response.get("sample_rate", 24000)),
            "duration_s": audio_s,
            "samples": int(response.get("samples", 0)),
            "rtf": round(float(response.get("rtf", 0.0)), 3),
            "generation_ms": round(float(response.get("generation_ms", 0.0)), 1),
            "code2wav_ms": round(float(response.get("code2wav_ms", 0.0)), 1),
            "worker_init_ms": round(float(self._worker_ready_meta.get("init_ms", 0.0)), 1),
        }
        return wav_bytes, meta

    def _synthesize_worker_via_stream(
        self, text: str, language: Optional[str] = None, **kwargs
    ) -> tuple[bytes, dict]:
        """Aggregate streaming PCM chunks into a single WAV.

        Used by the non-streaming /tts endpoint when stateful Code2Wav is on
        (the C++ worker only accepts stream=true in that mode).
        """
        t0 = time.time()
        done_meta: dict = {}
        # Force segment_text=False so we drive a single streaming request
        # here; outer synthesize() already split into segments if needed.
        stream_kwargs = dict(kwargs)
        stream_kwargs["segment_text"] = False
        stream_kwargs["language"] = language
        pcm = bytearray()
        for chunk in self._generate_streaming_single(text, meta_out=done_meta, **stream_kwargs):
            pcm.extend(chunk)
        elapsed = time.time() - t0
        sample_rate = int(done_meta.get("sample_rate", 24000))
        wav_bytes = _pcm16_to_wav(bytes(pcm), sample_rate=sample_rate)
        meta = {
            "inference_time_s": round(elapsed, 3),
            "sample_rate": sample_rate,
            "duration_s": float(done_meta.get("audio_s", 0.0)),
            "samples": int(done_meta.get("samples", len(pcm) // 2)),
            "rtf": round(float(done_meta.get("rtf", 0.0)), 3),
            "generation_ms": round(float(done_meta.get("generation_ms", 0.0)), 1),
            "code2wav_ms": round(float(done_meta.get("code2wav_ms", 0.0)), 1),
            "first_chunk_ms": round(float(done_meta.get("first_chunk_ms", 0.0)), 1),
            "chunk_count": int(done_meta.get("chunk_count", 0)),
            "stateful_code2wav": bool(done_meta.get("stateful_code2wav", True)),
            "worker_init_ms": round(float(self._worker_ready_meta.get("init_ms", 0.0)), 1),
        }
        return wav_bytes, meta

    def generate_streaming(self, text: str, **kwargs):
        """Yield raw PCM int16 chunks from the resident EdgeLLM TTS worker."""
        if self._product_backend is not None:
            yield from self._product_backend.generate_streaming(text, **kwargs)
            return

        if _DEFAULT_SEGMENT_TEXT and kwargs.get("segment_text", True):
            segments = _split_tts_text(text, kwargs.get("segment_max_chars"))
            if len(segments) > 1:
                segment_kwargs = dict(kwargs)
                segment_kwargs["segment_text"] = False
                for segment in segments:
                    yield from self.generate_streaming(segment, **segment_kwargs)
                return

        yield from self._generate_streaming_single(text, **kwargs)

    def _generate_streaming_single(self, text: str, meta_out: Optional[dict] = None, **kwargs):
        """Yield raw PCM int16 chunks for one already-bounded TTS request.

        If ``meta_out`` is a dict, the worker's terminal ``done`` event JSON
        (rtf, total_ms, first_chunk_ms, etc.) is merged into it before the
        generator returns. Used by ``_synthesize_worker`` to assemble a
        single-shot WAV in stateful Code2Wav mode.
        """
        req_id = uuid.uuid4().hex
        streaming_profile = str(
            kwargs.get("streaming_profile", os.environ.get("EDGE_LLM_TTS_STREAMING_PROFILE", "continuous_playback"))
        ).lower()
        if streaming_profile in ("v2v", "voice_to_voice", "eos_to_first_audio"):
            default_first_chunk_frames = 1
            default_chunk_frames = 97
            default_chunk_growth_frames = 0
            default_max_chunk_frames = 97
            default_adaptive_chunks = False
        elif streaming_profile in ("instant_feedback", "low_latency"):
            default_first_chunk_frames = 1
            default_chunk_frames = 25
            default_chunk_growth_frames = 50
            default_max_chunk_frames = 150
            default_adaptive_chunks = True
        elif streaming_profile in ("playback", "smooth", "buffered"):
            default_first_chunk_frames = 20
            default_chunk_frames = 20
            default_chunk_growth_frames = 30
            default_max_chunk_frames = 120
            default_adaptive_chunks = True
        else:
            default_first_chunk_frames = 50
            default_chunk_frames = 97
            default_chunk_growth_frames = 0
            default_max_chunk_frames = 97
            default_adaptive_chunks = False
        if _tts_stateful_code2wav_enabled():
            perf_profile = _tts_perf_profile()
            if _tts_fast_profile(perf_profile):
                default_first_chunk_frames = 4
            elif perf_profile == "balanced":
                default_first_chunk_frames = 6
            else:
                # 7 frames is the smallest validated default that still gives
                # the second chunk overlap with stateful Code2Wav on Orin Nano.
                default_first_chunk_frames = 7
            default_chunk_frames = 10
            default_chunk_growth_frames = 0
            default_max_chunk_frames = 10
            default_adaptive_chunks = False
        request = {
            "id": req_id,
            "text": text,
            "output_file": f"/tmp/trt_edgellm_tts_stream_{req_id}.wav",
            "language": kwargs.get("language") or _detect_language(text),
            "talker_temperature": _DEFAULT_TEMPERATURE,
            "talker_top_k": _DEFAULT_TOP_K,
            "talker_top_p": _DEFAULT_TOP_P,
            "repetition_penalty": _DEFAULT_REPETITION_PENALTY,
            "codec_eos_logit_offset": _DEFAULT_CODEC_EOS_LOGIT_OFFSET,
            "predictor_temperature": _DEFAULT_PREDICTOR_TEMPERATURE,
            "predictor_top_k": _DEFAULT_PREDICTOR_TOP_K,
            "predictor_top_p": _DEFAULT_PREDICTOR_TOP_P,
            "max_audio_length": kwargs.get("max_audio_length", _DEFAULT_MAX_AUDIO_LENGTH),
            "min_audio_length": kwargs.get("min_audio_length", _DEFAULT_MIN_AUDIO_LENGTH),
            "stream": True,
            "stream_only": True,
            "first_chunk_frames": kwargs.get(
                "first_chunk_frames",
                int(os.environ.get("EDGE_LLM_TTS_FIRST_CHUNK_FRAMES", str(default_first_chunk_frames))),
            ),
            "chunk_frames": kwargs.get(
                "chunk_frames",
                int(os.environ.get("EDGE_LLM_TTS_CHUNK_FRAMES", str(default_chunk_frames))),
            ),
            "adaptive_chunks": kwargs.get(
                "adaptive_chunks",
                os.environ.get("EDGE_LLM_TTS_ADAPTIVE_CHUNKS", "1" if default_adaptive_chunks else "0").lower()
                not in ("0", "false", "no"),
            ),
            "max_chunk_frames": kwargs.get(
                "max_chunk_frames",
                int(os.environ.get("EDGE_LLM_TTS_MAX_CHUNK_FRAMES", str(default_max_chunk_frames))),
            ),
            "chunk_growth_frames": kwargs.get(
                "chunk_growth_frames",
                int(os.environ.get("EDGE_LLM_TTS_CHUNK_GROWTH_FRAMES", str(default_chunk_growth_frames))),
            ),
            "chunk_format": "pcm_s16le",
            "chunk_transport": "base64",
        }
        speaker_embedding = kwargs.get("speaker_embedding")
        if speaker_embedding:
            request["speaker_embedding_b64"] = base64.b64encode(speaker_embedding).decode("ascii")

        with self._worker_lock:
            self._ensure_worker()
            assert self._worker is not None and self._worker.stdin is not None and self._worker.stdout is not None
            self._worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._worker.stdin.flush()

            while True:
                line = self._worker.stdout.readline()
                if not line:
                    self._worker = None
                    raise RuntimeError(f"TTS worker exited during stream: {self._worker_stderr_snip()}")
                event = json.loads(line)
                if not event.get("ok"):
                    raise RuntimeError(f"TTS streaming worker failed: {event}")
                if event.get("event") == "chunk":
                    if event.get("chunk_transport") == "base64":
                        yield base64.b64decode(event.get("audio_b64", ""))
                    elif event.get("chunk_file"):
                        with open(event["chunk_file"], "rb") as f:
                            payload = f.read()
                        try:
                            os.unlink(event["chunk_file"])
                        except OSError:
                            pass
                        if event.get("chunk_format") == "wav" and len(payload) > 44:
                            payload = payload[44:]
                        yield payload
                elif event.get("event") == "done":
                    # Surface the worker's terminal metadata to callers that
                    # passed a `meta_out` dict (used by _synthesize_worker in
                    # stateful Code2Wav mode to assemble a single-shot WAV).
                    if meta_out is not None and isinstance(meta_out, dict):
                        meta_out.update(event)
                    break

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Run TTS inference via subprocess.

        Returns (wav_bytes, meta_dict).  ``wav_bytes`` will be empty if the
        Code2Wav engine is unavailable (the backend produced RVQ codes only).
        """
        if not self._ready:
            raise RuntimeError("TTS backend not preloaded")
        if self._product_backend is not None:
            return self._synthesize_single(
                text,
                speaker_id=speaker_id,
                speed=speed,
                pitch_shift=pitch_shift,
                language=language,
                **kwargs,
            )

        if _DEFAULT_SEGMENT_TEXT and kwargs.get("segment_text", True):
            segments = _split_tts_text(text, kwargs.get("segment_max_chars"))
            if len(segments) > 1:
                segment_kwargs = dict(kwargs)
                segment_kwargs["segment_text"] = False
                segment_kwargs.setdefault("seed", int(os.environ.get("JETSON_VOICE_TTS_SEED", "42")))
                wav_parts: list[bytes] = []
                segment_meta: list[dict] = []
                total_elapsed = 0.0
                total_duration = 0.0
                total_samples = 0
                for segment in segments:
                    wav, meta = self.synthesize(
                        segment,
                        speaker_id=speaker_id,
                        speed=speed,
                        pitch_shift=pitch_shift,
                        language=language,
                        **segment_kwargs,
                    )
                    wav_parts.append(wav)
                    segment_meta.append({"text": segment, **meta})
                    total_elapsed += float(meta.get("inference_time_s", 0.0))
                    wav_duration, wav_samples = _wav_duration_and_samples(wav)
                    total_duration += wav_duration
                    total_samples += wav_samples

                pauses_ms = [_segment_pause_ms(segment) for segment in segments[:-1]]
                wav_bytes = _concat_wav_bytes(wav_parts, pauses_ms)
                meta = {
                    "inference_time_s": round(total_elapsed, 3),
                    "sample_rate": self.sample_rate,
                    "duration_s": round(total_duration + sum(pauses_ms) / 1000.0, 3),
                    "samples": total_samples + int(self.sample_rate * sum(pauses_ms) / 1000.0),
                    "rtf": round(total_elapsed / total_duration, 3) if total_duration > 0 else 0.0,
                    "segmented": True,
                    "segment_count": len(segments),
                    "segment_pauses_ms": pauses_ms,
                    "segments": segment_meta,
                }
                return wav_bytes, meta

        return self._synthesize_single(
            text,
            speaker_id=speaker_id,
            speed=speed,
            pitch_shift=pitch_shift,
            language=language,
            **kwargs,
        )

    def clone_voice(
        self,
        text: str,
        speaker_embedding: bytes,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if self._product_backend is not None:
            return self._product_backend.clone_voice(
                text=text,
                speaker_embedding=speaker_embedding,
                language=language,
                speed=speed,
                **kwargs,
            )
        if len(speaker_embedding) % 4 != 0:
            raise ValueError("speaker_embedding must be a float32 byte vector")
        return self.synthesize(
            text,
            speed=speed,
            language=language,
            speaker_embedding=speaker_embedding,
            **kwargs,
        )

    def extract_speaker_embedding(self, audio_wav_bytes: bytes) -> bytes:
        if self._product_backend is not None:
            return self._product_backend.extract_speaker_embedding(audio_wav_bytes)
        if not os.path.exists(_QWEN3_SPEAKER_ENCODER):
            raise NotImplementedError(f"speaker encoder not found: {_QWEN3_SPEAKER_ENCODER}")
        if not os.path.exists(_QWEN3_EXTRACT_SCRIPT):
            raise NotImplementedError(f"speaker extraction script not found: {_QWEN3_EXTRACT_SCRIPT}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            wf.write(audio_wav_bytes)
            wav_path = wf.name
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as ef:
            emb_path = ef.name

        try:
            result = subprocess.run(
                [
                    "python3",
                    _QWEN3_EXTRACT_SCRIPT,
                    "--audio",
                    wav_path,
                    "--model",
                    _QWEN3_SPEAKER_ENCODER,
                    "--output",
                    emb_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Embedding extraction failed: {result.stderr}")
            with open(emb_path, "rb") as f:
                return f.read()
        finally:
            for path in (wav_path, emb_path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _synthesize_single(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Run one already-bounded TTS request."""
        if self._product_backend is not None:
            return self._product_backend.synthesize(
                text,
                speaker_id=speaker_id,
                speed=speed,
                pitch_shift=pitch_shift,
                language=language,
                **kwargs,
            )
        if self._use_worker():
            return self._synthesize_worker(text, language, **kwargs)

        # Build input JSON
        input_data = {
            "requests": [
                {
                    "messages": [{"role": "user", "content": text}],
                    "speaker": "",
                }
            ],
            "batch_size": 1,
            "apply_chat_template": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "talker_temperature": _DEFAULT_TEMPERATURE,
            "talker_top_k": _DEFAULT_TOP_K,
            "talker_top_p": _DEFAULT_TOP_P,
            "repetition_penalty": _DEFAULT_REPETITION_PENALTY,
            "codec_eos_logit_offset": _DEFAULT_CODEC_EOS_LOGIT_OFFSET,
            "predictor_temperature": _DEFAULT_PREDICTOR_TEMPERATURE,
            "predictor_top_k": _DEFAULT_PREDICTOR_TOP_K,
            "predictor_top_p": _DEFAULT_PREDICTOR_TOP_P,
            "max_audio_length": kwargs.get(
                "max_audio_length", _DEFAULT_MAX_AUDIO_LENGTH
            ),
            "min_audio_length": kwargs.get(
                "min_audio_length", _DEFAULT_MIN_AUDIO_LENGTH
            ),
        }

        with tempfile.TemporaryDirectory(prefix="trt_edgellm_tts_") as tmpdir:
            input_path = os.path.join(tmpdir, "input.json")
            output_path = os.path.join(tmpdir, "output.json")
            audio_dir = os.path.join(tmpdir, "audio_out")
            os.makedirs(audio_dir, exist_ok=True)

            with open(input_path, "w") as f:
                json.dump(input_data, f)

            # Build CLI args
            cli_args = [
                "--inputFile",
                input_path,
                "--talkerEngineDir",
                TTS_TALKER_DIR,
                "--codePredictorEngineDir",
                TTS_CODE_PREDICTOR_DIR,
                "--tokenizerDir",
                TTS_TOKENIZER_DIR,
                "--outputFile",
                output_path,
                "--outputAudioDir",
                audio_dir,
            ]

            # Add code2wav if engine exists
            c2w_path = os.path.join(TTS_CODE2WAV_DIR, "code2wav.engine")
            if os.path.exists(c2w_path):
                cli_args += ["--code2wavEngineDir", TTS_CODE2WAV_DIR]

            t0 = time.time()
            result = run_binary(TTS_BINARY, cli_args, timeout=120)
            elapsed = time.time() - t0

            # Parse output — fail loudly on errors
            if result.returncode != 0 or not os.path.exists(output_path):
                raise RuntimeError(
                    f"TTS subprocess failed (exit={result.returncode}): "
                    f"stdout={result.stdout[-300:]}, stderr={result.stderr[-300:]}"
                )

            with open(output_path) as f:
                output_data = json.load(f)

            responses = output_data.get("responses", [])
            if not responses:
                raise RuntimeError(f"TTS produced no responses: {output_data}")

            r = responses[0]
            audio_file = r.get("audio_file")
            wav_bytes = b""
            meta = {"inference_time_s": round(elapsed, 3), "sample_rate": 24000}

            if audio_file and os.path.exists(audio_file):
                with open(audio_file, "rb") as f:
                    wav_bytes = f.read()
                meta["duration_s"] = r.get("audio_duration_ms", 0) / 1000.0
                meta["samples"] = r.get("audio_samples", 0)
            else:
                logger.warning("No audio WAV in output, returning RVQ codes only")
                meta["rvq_file"] = r.get("rvq_file")
                if not meta.get("rvq_file"):
                    raise RuntimeError(
                        f"TTS output has neither audio nor RVQ: {list(r.keys())}"
                    )

            return wav_bytes, meta
