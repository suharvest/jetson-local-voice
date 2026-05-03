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
from typing import Optional
import importlib.util
import uuid

from tts_backend import TTSBackend, TTSCapability

from backends.trt_edge_llm_ipc import (
    TTS_BINARY,
    TTS_WORKER_BINARY,
    TTS_TALKER_DIR,
    TTS_CODE2WAV_DIR,
    TTS_TOKENIZER_DIR,
    PLUGIN_PATH,
    TTS_SPECIAL_CP_ENGINE,
    TTS_SPECIAL_CP_EMBED_FP32,
    run_binary,
)

logger = logging.getLogger(__name__)


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


# Default sampling parameters
_DEFAULT_TEMPERATURE = float(os.environ.get("TTS_TALKER_TEMPERATURE", "0.9"))
_DEFAULT_TOP_K = int(os.environ.get("TTS_TALKER_TOP_K", "50"))
_DEFAULT_TOP_P = float(os.environ.get("TTS_TOP_P", "1.0"))
_DEFAULT_MAX_AUDIO_LENGTH = int(os.environ.get("TTS_MAX_AUDIO_LENGTH", "1024"))
_DEFAULT_REPETITION_PENALTY = float(os.environ.get("TTS_REPETITION_PENALTY", "1.05"))


class TRTEdgeLLMTTSBackend(TTSBackend):
    """TTS via TRT-Edge-LLM qwen3_tts_inference subprocess."""

    def __init__(self):
        self._ready = False
        self._native_fallback = None
        self._worker: Optional[subprocess.Popen] = None
        self._worker_lock = threading.Lock()
        self._worker_ready_meta: dict = {}

    # -- TTSBackend interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "trt_edgellm"

    @property
    def capabilities(self) -> set[TTSCapability]:
        return {TTSCapability.BASIC_TTS, TTSCapability.MULTI_LANGUAGE, TTSCapability.STREAMING}

    @property
    def sample_rate(self) -> int:
        return 24000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        """Verify all required files exist."""
        fallback_mode = os.environ.get("EDGE_LLM_TTS_NATIVE_FALLBACK", "0").lower()
        fallback_base = os.environ.get("EDGE_LLM_TTS_FALLBACK_BASE", "/home/harvest/voice_test")
        fallback_backend = os.path.join(fallback_base, "app_overlay", "backends", "qwen3_trt.py")
        fallback_models = os.path.join(fallback_base, "models", "qwen3-tts")
        use_fallback = fallback_mode in ("1", "true", "yes") or (
            fallback_mode == "auto" and os.path.exists(fallback_backend)
        )
        if use_fallback:
            overlay = os.path.join(fallback_base, "app_overlay")
            overlay_backends = os.path.join(overlay, "backends")
            for path in (overlay, overlay_backends):
                if path not in sys.path:
                    sys.path.insert(0, path)

            os.environ.setdefault("QWEN3_MODEL_BASE", fallback_models)
            os.environ.setdefault("QWEN3_MODEL_DIR", os.path.join(fallback_models, "onnx"))
            os.environ.setdefault("QWEN3_SHERPA_DIR", os.path.join(fallback_models, "onnx"))
            os.environ.setdefault(
                "QWEN3_TALKER_ENGINE",
                os.path.join(fallback_models, "engines", "talker_decode_bf16.engine"),
            )
            os.environ.setdefault(
                "QWEN3_CP_ENGINE",
                os.path.join(fallback_models, "engines", "cp_bf16.engine"),
            )
            os.environ.setdefault("TTS_TALKER_CUDA_GRAPH", "0")

            spec = importlib.util.spec_from_file_location(
                "edge_llm_tts_native_fallback", fallback_backend
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load native fallback backend: {fallback_backend}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._native_fallback = module.Qwen3TRTBackend()
            logger.warning(
                "Using native Qwen3 TRT fallback for TTS; EdgeLLM talker logits are known-bad on this Nano build"
            )
            self._native_fallback.preload()
            self._ready = True
            return

        required = [
            (TTS_WORKER_BINARY if self._use_worker() else TTS_BINARY, "TTS binary"),
            (PLUGIN_PATH, "TRT-Edge-LLM plugin"),
            (os.path.join(TTS_TALKER_DIR, "config.json"), "talker config"),
            (os.path.join(TTS_TALKER_DIR, "llm.engine"), "talker engine"),
            (os.path.join(TTS_TOKENIZER_DIR, "tokenizer.json"), "tokenizer"),
        ]
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
            "TTS backend preload OK (binary=%s talker=%s)",
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
        env.setdefault("EDGE_LLM_TTS_CUDA_GRAPH", "1")
        env.setdefault("EDGE_LLM_TTS_LAZY_CODE2WAV", "0")
        if os.path.exists(TTS_SPECIAL_CP_ENGINE) and os.path.exists(TTS_SPECIAL_CP_EMBED_FP32):
            env.setdefault("QWEN3_TTS_CP_ENGINE", TTS_SPECIAL_CP_ENGINE)
            env.setdefault("QWEN3_TTS_CP_EMBED_FP32", TTS_SPECIAL_CP_EMBED_FP32)
        return env

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.poll() is None:
            return
        cmd = [
            TTS_WORKER_BINARY,
            "--talkerEngineDir",
            TTS_TALKER_DIR,
            "--tokenizerDir",
            TTS_TOKENIZER_DIR,
            "--code2wavEngineDir",
            TTS_CODE2WAV_DIR,
        ]
        self._worker = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._worker_env(),
        )
        assert self._worker.stdout is not None
        ready_line = self._worker.stdout.readline()
        if not ready_line:
            stderr = self._worker.stderr.read()[-2000:] if self._worker.stderr else ""
            raise RuntimeError(f"TTS worker failed to start: {stderr}")
        ready = json.loads(ready_line)
        if ready.get("event") != "ready":
            raise RuntimeError(f"TTS worker did not become ready: {ready}")
        self._worker_ready_meta = ready

    def _synthesize_worker(self, text: str, language: Optional[str], **kwargs) -> tuple[bytes, dict]:
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
            "max_audio_length": kwargs.get("max_audio_length", _DEFAULT_MAX_AUDIO_LENGTH),
        }
        with self._worker_lock:
            self._ensure_worker()
            assert self._worker is not None and self._worker.stdin is not None and self._worker.stdout is not None
            t0 = time.time()
            self._worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._worker.stdin.flush()
            line = self._worker.stdout.readline()
            elapsed = time.time() - t0
        if not line:
            stderr = self._worker.stderr.read()[-2000:] if self._worker.stderr else ""
            self._worker = None
            raise RuntimeError(f"TTS worker exited before response: {stderr}")
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

    def generate_streaming(self, text: str, **kwargs):
        """Yield raw PCM int16 chunks from the resident EdgeLLM TTS worker."""
        req_id = uuid.uuid4().hex
        request = {
            "id": req_id,
            "text": text,
            "output_file": f"/tmp/trt_edgellm_tts_stream_{req_id}.wav",
            "language": kwargs.get("language") or _detect_language(text),
            "talker_temperature": _DEFAULT_TEMPERATURE,
            "talker_top_k": _DEFAULT_TOP_K,
            "talker_top_p": _DEFAULT_TOP_P,
            "repetition_penalty": _DEFAULT_REPETITION_PENALTY,
            "max_audio_length": kwargs.get("max_audio_length", _DEFAULT_MAX_AUDIO_LENGTH),
            "stream": True,
            "stream_only": True,
            "first_chunk_frames": kwargs.get(
                "first_chunk_frames",
                int(os.environ.get("EDGE_LLM_TTS_FIRST_CHUNK_FRAMES", "1")),
            ),
            "chunk_frames": kwargs.get(
                "chunk_frames",
                int(os.environ.get("EDGE_LLM_TTS_CHUNK_FRAMES", "25")),
            ),
            "adaptive_chunks": kwargs.get(
                "adaptive_chunks",
                os.environ.get("EDGE_LLM_TTS_ADAPTIVE_CHUNKS", "1").lower()
                not in ("0", "false", "no"),
            ),
            "max_chunk_frames": kwargs.get(
                "max_chunk_frames",
                int(os.environ.get("EDGE_LLM_TTS_MAX_CHUNK_FRAMES", "150")),
            ),
            "chunk_growth_frames": kwargs.get(
                "chunk_growth_frames",
                int(os.environ.get("EDGE_LLM_TTS_CHUNK_GROWTH_FRAMES", "50")),
            ),
            "chunk_format": "pcm_s16le",
            "chunk_transport": "base64",
        }

        with self._worker_lock:
            self._ensure_worker()
            assert self._worker is not None and self._worker.stdin is not None and self._worker.stdout is not None
            self._worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._worker.stdin.flush()

            while True:
                line = self._worker.stdout.readline()
                if not line:
                    stderr = self._worker.stderr.read()[-2000:] if self._worker.stderr else ""
                    self._worker = None
                    raise RuntimeError(f"TTS worker exited during stream: {stderr}")
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
        if self._native_fallback is not None:
            return self._native_fallback.synthesize(
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
            "max_audio_length": kwargs.get(
                "max_audio_length", _DEFAULT_MAX_AUDIO_LENGTH
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
