"""ASR backend via TRT-Edge-LLM C++ binary (llm_inference).

Audio is converted to a Whisper-compatible log-mel spectrogram in Python
(scipy + numpy, no librosa), saved as a safetensors file, and passed to the
LLM binary via ``--multimodalEngineDir`` for the audio encoder.

Supports: OFFLINE, MULTI_LANGUAGE
Streaming: planned (Phase 2, requires llm_stream binary).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from typing import Optional

import numpy as np

from asr_backend import ASRBackend, ASRCapability, ASRStream, TranscriptionResult

from backends.trt_edge_llm_ipc import (
    ASR_BINARY,
    ASR_WORKER_BINARY,
    ASR_ENGINE_DIR,
    ASR_AUDIO_ENC_DIR,
    ASR_PLUGIN_PATH,
    audio_bytes_to_mel,
    run_binary,
    write_safetensors,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_GENERATE_LENGTH = int(
    os.environ.get("ASR_MAX_GENERATE_LENGTH", "200")
)
_DEFAULT_TEMPERATURE = float(os.environ.get("ASR_TEMPERATURE", "1.0"))
_DEFAULT_TOP_P = float(os.environ.get("ASR_TOP_P", "1.0"))
_DEFAULT_TOP_K = int(os.environ.get("ASR_TOP_K", "1"))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no")


class TRTEdgeLLMASRBackend(ASRBackend):
    """ASR via TRT-Edge-LLM llm_inference subprocess."""

    def __init__(self):
        self._config = self._load_config()
        self._ready = False
        self._worker: Optional[subprocess.Popen] = None
        self._worker_lock = threading.Lock()
        self._worker_ready_meta: dict = {}
        self._worker_stderr_tail: deque[str] = deque(maxlen=80)

    def _load_config(self) -> dict:
        manifest: dict = {}
        manifest_path = os.environ.get("EDGE_LLM_ASR_MANIFEST")
        if manifest_path:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        use_worker_default = bool(manifest.get("use_worker", True))
        return {
            "asr_binary": os.environ.get(
                "EDGE_LLM_ASR_BIN", manifest.get("asr_binary", ASR_BINARY)
            ),
            "worker_binary": os.environ.get(
                "EDGE_LLM_ASR_WORKER_BIN",
                manifest.get("worker_binary", ASR_WORKER_BINARY),
            ),
            "plugin_path": os.environ.get(
                "EDGE_LLM_ASR_PLUGIN_PATH",
                os.environ.get(
                    "EDGELLM_ASR_PLUGIN_PATH",
                    manifest.get(
                        "asr_plugin_path",
                        manifest.get("plugin_path", ASR_PLUGIN_PATH),
                    ),
                ),
            ),
            "engine_dir": os.environ.get(
                "EDGE_LLM_ASR_ENGINE_DIR", manifest.get("engine_dir", ASR_ENGINE_DIR)
            ),
            "audio_encoder_dir": os.environ.get(
                "EDGE_LLM_ASR_AUDIO_ENC_DIR",
                manifest.get("audio_encoder_dir", ASR_AUDIO_ENC_DIR),
            ),
            "use_worker": _env_bool("EDGE_LLM_ASR_WORKER", use_worker_default),
            "mel_tensor_name": os.environ.get(
                "EDGE_LLM_ASR_MEL_TENSOR_NAME",
                manifest.get("mel_tensor_name", "mel"),
            ),
            "max_mel_frames": int(
                os.environ.get(
                    "EDGE_LLM_ASR_MAX_MEL_FRAMES",
                    str(manifest.get("max_mel_frames", 6000)),
                )
            ),
            "manifest_path": manifest_path,
        }

    # -- ASRBackend interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "trt_edgellm"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.MULTI_LANGUAGE, ASRCapability.STREAMING}

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        """Verify all required files exist."""
        worker_binary = self._config["worker_binary"]
        asr_binary = self._config["asr_binary"]
        plugin_path = self._config["plugin_path"]
        engine_dir = self._config["engine_dir"]
        audio_encoder_dir = self._config["audio_encoder_dir"]
        required = [
            (worker_binary if self._use_worker() else asr_binary, "ASR binary"),
            (plugin_path, "TRT-Edge-LLM plugin"),
            (os.path.join(engine_dir, "config.json"), "LLM config"),
            (os.path.join(engine_dir, "llm.engine"), "LLM engine"),
            (os.path.join(
                audio_encoder_dir, "audio", "config.json"
            ), "audio encoder config"),
            (os.path.join(
                audio_encoder_dir, "audio", "audio_encoder.engine"
            ), "audio encoder engine"),
        ]
        missing = []
        for path, label in required:
            if not os.path.exists(path):
                missing.append(f"{label}: {path}")
        if missing:
            raise FileNotFoundError(
                "ASR preload failed — missing:\n  " + "\n  ".join(missing)
            )

        logger.info(
            "ASR backend preload OK (config=%s)",
            self._config,
        )
        if self._use_worker():
            self._ensure_worker()
        self._ready = True

    def _use_worker(self) -> bool:
        return bool(self._config["use_worker"])

    def _worker_env(self) -> dict:
        env = os.environ.copy()
        env["EDGELLM_PLUGIN_PATH"] = self._config["plugin_path"]
        env.setdefault("EDGE_LLM_ASR_CUDA_GRAPH", "0")
        return env

    def _drain_worker_stderr(self, worker: subprocess.Popen) -> None:
        if worker.stderr is None:
            return
        for line in worker.stderr:
            text = line.rstrip()
            self._worker_stderr_tail.append(text)
            if "[JV_MEM]" in text:
                logger.info("ASR worker: %s", text)
            else:
                logger.debug("ASR worker stderr: %s", text)

    def _stderr_tail_text(self) -> str:
        return "\n".join(self._worker_stderr_tail)

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.poll() is None:
            return
        cmd = [
            self._config["worker_binary"],
            "--engineDir",
            self._config["engine_dir"],
            "--multimodalEngineDir",
            self._config["audio_encoder_dir"],
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
        self._worker_stderr_tail.clear()
        threading.Thread(
            target=self._drain_worker_stderr,
            args=(self._worker,),
            name="trt-edgellm-asr-stderr",
            daemon=True,
        ).start()
        assert self._worker.stdout is not None
        ready_line = self._worker.stdout.readline()
        if not ready_line:
            stderr = self._stderr_tail_text()
            raise RuntimeError(f"ASR worker failed to start: {stderr}")
        ready = json.loads(ready_line)
        if ready.get("event") != "ready":
            raise RuntimeError(f"ASR worker did not become ready: {ready}")
        self._worker_ready_meta = ready

    @staticmethod
    def _strip_language_prefix(text: str) -> tuple[str, Optional[str]]:
        language_detected = None
        if text and len(text) > 9 and text[:9] == "language ":
            known_languages = (
                "Chinese", "English", "Cantonese", "Japanese", "Korean",
                "French", "German", "Italian", "Portuguese", "Russian",
                "Spanish",
            )
            for name in known_languages:
                prefix = f"language {name}"
                if text.startswith(prefix):
                    language_detected = name
                    text = text[len(prefix) :].lstrip()
                    break
            else:
                space = text.find(" ", 9)
                if space > 0:
                    language_detected = text[9:space]
                    text = text[space + 1 :].lstrip()
        return text, language_detected

    def _transcribe_worker(self, mel_path: str, elapsed_mel_s: float) -> TranscriptionResult:
        req_id = uuid.uuid4().hex
        input_data = {
            "id": req_id,
            "requests": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio",
                                    "audio": mel_path,
                                }
                            ],
                        }
                    ],
                }
            ],
            "batch_size": 1,
            "temperature": _DEFAULT_TEMPERATURE,
            "top_p": _DEFAULT_TOP_P,
            "top_k": _DEFAULT_TOP_K,
            "max_generate_length": _DEFAULT_MAX_GENERATE_LENGTH,
            "apply_chat_template": True,
            "add_generation_prompt": True,
        }
        with self._worker_lock:
            self._ensure_worker()
            assert self._worker is not None and self._worker.stdin is not None and self._worker.stdout is not None
            t0 = time.time()
            self._worker.stdin.write(json.dumps(input_data, ensure_ascii=False) + "\n")
            self._worker.stdin.flush()
            line = self._worker.stdout.readline()
            elapsed_worker = time.time() - t0

        if not line:
            stderr = self._stderr_tail_text()
            self._worker = None
            raise RuntimeError(f"ASR worker exited before response: {stderr}")
        output_data = json.loads(line)
        if not output_data.get("ok"):
            raise RuntimeError(f"ASR worker failed: {output_data}")

        responses = output_data.get("responses", [])
        if not responses:
            raise RuntimeError(f"ASR produced no responses: {output_data}")
        text = responses[0].get("output_text", "")
        if text == "TensorRT Edge LLM cannot handle this request. Fails.":
            raise RuntimeError(f"ASR inference failed (model returned error): {responses[0]}")
        text, language_detected = self._strip_language_prefix(text)
        total_s = elapsed_mel_s + elapsed_worker
        return TranscriptionResult(
            text=text,
            language=language_detected,
            inference_time_s=round(total_s, 3),
            mel_time_s=round(elapsed_mel_s, 3),
            worker_time_s=round(elapsed_worker, 3),
            worker_init_ms=round(float(self._worker_ready_meta.get("init_ms", 0.0)), 1),
        )

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "auto",
    ) -> TranscriptionResult:
        """Transcribe audio via subprocess.

        Workflow:
          1. Write incoming audio to a temp WAV file.
          2. Compute log-mel spectrogram (numpy+scipy).
          3. Save mel as FP16 safetensors.
          4. Build input JSON referencing the mel file.
          5. Run ``llm_inference --multimodalEngineDir ...``.
          6. Parse output JSON for transcribed text.
        """
        if not self._ready:
            raise RuntimeError("ASR backend not preloaded")

        with tempfile.TemporaryDirectory(
            prefix="trt_edgellm_asr_"
        ) as tmpdir:
            # -- 1. Compute mel spectrogram (with duration guard) --
            mel_t0 = time.time()
            mel = audio_bytes_to_mel(audio_bytes)  # [1, 128, T] float32
            max_mel_frames = int(self._config["max_mel_frames"])
            if mel.shape[2] > max_mel_frames:  # 10ms hop
                raise ValueError(
                    f"Audio too long: {mel.shape[2]} frames (~{mel.shape[2]*0.01:.0f}s). "
                    f"Max {max_mel_frames} frames (~{max_mel_frames*0.01:.0f}s). Split into smaller chunks."
                )

            # Convert to FP16 for TRT
            mel_fp16 = mel.astype(np.float16)

            mel_path = os.path.join(tmpdir, "mel.safetensors")
            write_safetensors(mel_fp16, self._config["mel_tensor_name"], mel_path)
            elapsed_mel_s = time.time() - mel_t0
            logger.info(
                "Mel computed: shape=%s size=%s -> %s",
                list(mel_fp16.shape),
                mel_fp16.nbytes,
                mel_path,
            )

            if self._use_worker():
                return self._transcribe_worker(mel_path, elapsed_mel_s)

            # -- 2. Build input JSON --
            input_data = {
                "requests": [
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "audio",
                                        "audio": mel_path,
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "batch_size": 1,
                "temperature": _DEFAULT_TEMPERATURE,
                "top_p": _DEFAULT_TOP_P,
                "top_k": _DEFAULT_TOP_K,
                "max_generate_length": _DEFAULT_MAX_GENERATE_LENGTH,
                "apply_chat_template": True,
                "add_generation_prompt": True,
            }

            input_path = os.path.join(tmpdir, "input.json")
            with open(input_path, "w") as f:
                json.dump(input_data, f)

            output_path = os.path.join(tmpdir, "output.json")

            # -- 3. Run binary --
            cli_args = [
                "--engineDir",
                self._config["engine_dir"],
                "--multimodalEngineDir",
                self._config["audio_encoder_dir"],
                "--inputFile",
                input_path,
                "--outputFile",
                output_path,
            ]

            t0 = time.time()
            result = run_binary(self._config["asr_binary"], cli_args, timeout=60)
            elapsed = time.time() - t0

            # -- 4. Parse output — fail loudly on errors
            if result.returncode != 0 or not os.path.exists(output_path):
                raise RuntimeError(
                    f"ASR subprocess failed (exit={result.returncode}): "
                    f"stdout={result.stdout[-300:]}, stderr={result.stderr[-300:]}"
                )

            with open(output_path) as f:
                output_data = json.load(f)

            responses = output_data.get("responses", [])
            if not responses:
                raise RuntimeError(f"ASR produced no responses: {output_data}")

            r = responses[0]
            text = r.get("output_text", "")
            if text == "TensorRT Edge LLM cannot handle this request. Fails.":
                raise RuntimeError(
                    f"ASR inference failed (model returned error): {r}"
                )

            text, language_detected = self._strip_language_prefix(text)

            meta = {
                "inference_time_s": round(elapsed, 3),
            }
            return TranscriptionResult(
                text=text, language=language_detected, **meta
            )

    def create_stream(self, language: str = "auto") -> ASRStream:
        """Streaming ASR is not yet implemented for the TRT-Edge-LLM backend.

        Phase 2 will use ``llm_stream --streamInterval N`` with per-token
        stdout parsing.
        """
        raise NotImplementedError(
            f"{self.name} does not support streaming yet; "
            "use llm_inference for offline transcription."
        )
