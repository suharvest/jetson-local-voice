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
import wave
import io
from collections import deque
from typing import Optional

import numpy as np

from app.core.asr_backend import ASRBackend, ASRCapability, ASRStream, TranscriptionResult

from app.backends.jetson.trt_edge_llm_ipc import (
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
        if self._use_worker():
            self._warm_worker()

    def _warm_worker(self) -> None:
        """Pre-warm TRT audio_encoder optimization profile for batch shapes 1..N.

        The audio_encoder engine has a single optimization profile with
        dynamic batch dim [1..60]. TRT silently re-selects tactics + reallocs
        workspace the FIRST time each unique batch shape is seen, costing
        ~3-4× extra ms on that call. Without pre-warming, a `long_audio →
        short_audio` switch in production triggers tactic recompilation
        ~once per unique short-audio length (3s, 4s, 3.5s, ...) and the user
        feels several extra seconds of latency.

        We pre-warm 1..12 seconds (batch=1..12) which covers >99% of expected
        utterances. Boot cost ~1-3s, RAM ~80MB. Bypass with env vars below.
        Set EDGE_LLM_ASR_PREWARM_MAX to extend coverage (max 60).
        """
        if os.environ.get("SKIP_ASR_WARMUP", "").lower() in ("1", "true", "yes"):
            logger.info("TRT-EdgeLLM ASR worker warmup skipped (SKIP_ASR_WARMUP set).")
            return
        if os.environ.get("EDGE_LLM_ASR_WORKER_WARMUP", "1").lower() in ("0", "false", "no"):
            logger.info("TRT-EdgeLLM ASR worker warmup skipped.")
            return
        # Default max = 6 seconds, covering production calls. The streaming
        # finalize path splits long audio at 4.5s, so worker never sees
        # batches > 5 sec in real traffic. The audio encoder engine starts
        # rejecting (attention_mask profile max ~780×780 ≈ 9s) somewhere
        # between batch=9 and batch=10, so warming above 9 is wasted anyway.
        try:
            prewarm_max = int(os.environ.get("EDGE_LLM_ASR_PREWARM_MAX", "6"))
        except ValueError:
            prewarm_max = 6
        prewarm_max = max(1, min(prewarm_max, 60))
        import time as _time
        t0 = _time.monotonic()
        warmed = 0
        for seconds in range(1, prewarm_max + 1):
            try:
                silence = np.zeros(16000 * seconds, dtype=np.float32)
                self.transcribe(_float_audio_to_wav_bytes(silence, 16000))
                warmed += 1
            except Exception as exc:
                # The encoder profile has a hard max shape; hitting it is the
                # natural stop signal, not a real warning. Log as INFO.
                msg = str(exc)
                if "cannot handle" in msg or "TensorRT Edge LLM" in msg:
                    logger.info(
                        "TRT-EdgeLLM ASR pre-warm: engine boundary at batch=%d "
                        "(expected, stopping)", seconds,
                    )
                else:
                    logger.warning(
                        "TRT-EdgeLLM ASR pre-warm batch=%d failed: %s", seconds, exc
                    )
                break  # larger shapes will also fail
        elapsed = _time.monotonic() - t0
        logger.info(
            "TRT-EdgeLLM ASR worker pre-warmed shapes 1..%d in %.1fs", warmed, elapsed
        )

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
        if text and len(text) >= 9 and text[:9] == "language ":
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
                # No known language matched. The model may emit "language None"
                # (no trailing text, hallucinated as a bailout for silent/noise
                # segments) or "language Xxx <text>". Find the trailing space:
                #   - found → strip "language <Xxx> " prefix
                #   - missing → entire string is just "language <Xxx>", drop it
                space = text.find(" ", 9)
                if space > 0:
                    language_detected = text[9:space]
                    text = text[space + 1 :].lstrip()
                else:
                    language_detected = text[9:]   # e.g. "None"
                    text = ""                      # empty out the bailout
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
        """Accumulate stream audio and run the resident worker on finalize.

        This matches the current V2V product behavior: the user has already
        stopped speaking before ASR is invoked, so partials are unnecessary.
        The resident C++ worker keeps the stop-to-final path warm.
        """
        if not self._ready:
            raise RuntimeError("ASR backend not preloaded")
        return _TRTEdgeLLMAccumulatingASRStream(self, language=language)


def _float_audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return out.getvalue()


class _TRTEdgeLLMAccumulatingASRStream(ASRStream):
    def __init__(self, backend: TRTEdgeLLMASRBackend, language: str = "auto"):
        self._backend = backend
        self._language = language
        self._chunks: list[np.ndarray] = []

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_len = int(len(samples) * ratio)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, new_len),
                np.arange(len(samples)),
                samples,
            ).astype(np.float32)
        self._chunks.append(samples.copy())

    def finalize(self) -> str:
        if not self._chunks:
            return ""
        audio = np.concatenate(self._chunks)
        # The TRT audio_encoder engine is built with a fixed optimization profile
        # (~800-1500 mel frames ≈ 10-15s). Forwarding >10s of audio in one shot
        # makes the worker reject the request ("cannot handle this request").
        # Split at natural silence points using the same VAD splitter the older
        # qwen3_asr backend uses, then concatenate the per-segment transcripts.
        from app.backends.jetson.qwen3_asr import (
            _split_at_silence_vad, _split_at_silence_energy, VAD_MAX_SEG_SEC,
        )
        try:
            segments = _split_at_silence_vad(audio)
        except ImportError:
            # webrtcvad not installed on this image — fall back to energy-RMS
            segments = _split_at_silence_energy(audio)
        except Exception:
            # Any other VAD failure: be safe, single segment (will only break
            # for very long audio; short audio still works).
            segments = [audio]

        # Skip very short / near-silent segments: they tend to push the
        # Qwen3-ASR model into bailout outputs ("language None"). Threshold
        # of 0.4s drops VAD-trailing fragments without removing real content
        # — natural speech segments split by the VAD are >=1.0s by design.
        MIN_SEG_S = 0.4
        texts: list[str] = []
        for seg in segments:
            if len(seg) / 16000 < MIN_SEG_S:
                continue
            wav_bytes = _float_audio_to_wav_bytes(seg, 16000)
            try:
                result = self._backend.transcribe(wav_bytes, language=self._language)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "TRT-EdgeLLM ASR segment failed (%.1fs): %s",
                    len(seg) / 16000, e,
                )
                continue
            if result.text:
                # Drop trailing CJK/Latin sentence punctuation when more
                # segments follow — the split was mid-utterance, the model
                # over-eagerly punctuated the end. Keeps "变得 更善于"
                # instead of "变得。 更善于".
                texts.append(result.text)

        # Join: use the request language to pick separator. CJK languages
        # never use spaces; everything else does. Don't infer from output
        # content (output may contain mixed-script proper nouns).
        cjk_langs = {"Chinese", "Japanese", "Korean", "Cantonese", "zh", "ja", "ko"}
        is_cjk = self._language in cjk_langs
        # Trim segment-trailing punctuation when more text follows
        if len(texts) > 1:
            trail_punct = "。，、！？；,.!?;"
            cleaned: list[str] = []
            for i, t in enumerate(texts):
                if i < len(texts) - 1:
                    cleaned.append(t.rstrip(trail_punct).rstrip())
                else:
                    cleaned.append(t)
            texts = cleaned
        separator = "" if is_cjk else " "
        return separator.join(texts).strip()

    def get_partial(self) -> tuple[str, bool]:
        return "", False
