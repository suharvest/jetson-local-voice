"""Common subprocess and I/O utilities for TRT-Edge-LLM backends.

Provides:
  - Path constants (override via env vars)
  - run_binary()  — one-shot subprocess invocation
  - write_safetensors() — numpy -> safetensors file (no PyPI dep needed)
  - audio_bytes_to_mel() — WAV bytes -> log-mel spectrogram (numpy-only)
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import tempfile
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# GPU subprocess gate: serialise binary launches to avoid concurrent GPU init OOM
_gpu_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Paths — all overridable via environment variables
# ---------------------------------------------------------------------------

_EDGE_LLM_BASE = os.environ.get(
    "EDGE_LLM_BASE", os.path.expanduser("~/project/tensorrt-edge-llm")
)
_EDGE_LLM_BUILD = os.path.join(
    _EDGE_LLM_BASE,
    os.environ.get("EDGE_LLM_BUILD_DIR", "build_sm87"),
)
_JETSON_VOICE_BASE = os.environ.get(
    "JETSON_VOICE_BASE", os.path.expanduser("~/project/jetson-voice")
)
_VOICE_WORKER_BUILD = os.environ.get(
    "JETSON_VOICE_WORKER_BUILD",
    os.path.join(_JETSON_VOICE_BASE, "build", "edgellm_voice_worker", "workers"),
)


def _prefer_existing(primary: str, fallback: str) -> str:
    return primary if os.path.exists(primary) else fallback

# Binaries
TTS_BINARY = os.environ.get(
    "EDGE_LLM_TTS_BIN",
    os.path.join(_EDGE_LLM_BUILD, "examples/omni/qwen3_tts_inference"),
)
TTS_WORKER_BINARY = os.environ.get(
    "EDGE_LLM_TTS_WORKER_BIN",
    _prefer_existing(
        os.path.join(_VOICE_WORKER_BUILD, "qwen3_tts_worker"),
        os.path.join(_EDGE_LLM_BUILD, "examples/omni/qwen3_tts_worker"),
    ),
)
ASR_BINARY = os.environ.get(
    "EDGE_LLM_ASR_BIN",
    os.path.join(_EDGE_LLM_BUILD, "examples/llm/llm_inference"),
)
ASR_WORKER_BINARY = os.environ.get(
    "EDGE_LLM_ASR_WORKER_BIN",
    _prefer_existing(
        os.path.join(_VOICE_WORKER_BUILD, "qwen3_asr_worker"),
        os.path.join(_EDGE_LLM_BUILD, "examples/llm/qwen3_asr_worker"),
    ),
)
PLUGIN_PATH = os.environ.get(
    "EDGELLM_PLUGIN_PATH",
    os.path.join(_EDGE_LLM_BUILD, "libNvInfer_edgellm_plugin.so"),
)
DEFAULT_PLUGIN_PATH = os.path.join(_EDGE_LLM_BUILD, "libNvInfer_edgellm_plugin.so")
ASR_PLUGIN_PATH = os.environ.get(
    "EDGE_LLM_ASR_PLUGIN_PATH",
    os.environ.get("EDGELLM_ASR_PLUGIN_PATH", DEFAULT_PLUGIN_PATH),
)

# TTS engine directories
_TTS_FIXED_RUNTIME = os.path.expanduser("~/qwen3-tts-edgellm-runtime")
_TTS_DEFAULT_ROOT = (
    _TTS_FIXED_RUNTIME
    if os.path.exists(os.path.join(_TTS_FIXED_RUNTIME, "engines", "talker", "llm.engine"))
    else os.path.expanduser("~/qwen3-tts-trt-edge-llm-export")
)
TTS_TALKER_DIR = os.environ.get(
    "EDGE_LLM_TTS_TALKER_DIR",
    os.path.join(_TTS_DEFAULT_ROOT, "engines", "talker"),
)
TTS_CODE_PREDICTOR_DIR = os.environ.get(
    "EDGE_LLM_TTS_CP_DIR",
    os.path.join(os.path.dirname(TTS_TALKER_DIR), "code_predictor"),
)
TTS_CODE2WAV_DIR = os.environ.get(
    "EDGE_LLM_TTS_CODE2WAV_DIR",
    os.path.join(_TTS_DEFAULT_ROOT, "engines", "code2wav")
    if os.path.exists(os.path.join(_TTS_DEFAULT_ROOT, "engines", "code2wav"))
    else os.path.expanduser("~/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder/code2wav"),
)
TTS_TOKENIZER_DIR = os.environ.get(
    "EDGE_LLM_TTS_TOKENIZER_DIR",
    _TTS_DEFAULT_ROOT
    if os.path.exists(os.path.join(_TTS_DEFAULT_ROOT, "processed_chat_template.json"))
    else os.path.expanduser("~/qwen3-tts-trt-edge-llm-export"),
)

# ASR engine directories
_ASR_PRUNED_ENGINE_DIR = os.path.expanduser(
    "~/qwen3-asr-edgellm-runtime/engines/thinker_prunedembed35k_kv512"
)
_ASR_OFFICIAL_PRUNED_ENGINE_DIR = os.path.expanduser(
    "~/qwen3-asr-edgellm-runtime/engines/thinker_pruned35k_kv512"
)
_ASR_DIALOG_ENGINE_DIR = os.path.expanduser(
    "~/qwen3-asr-edgellm-runtime/engines/thinker_kv512"
)
ASR_ENGINE_DIR = os.environ.get(
    "EDGE_LLM_ASR_ENGINE_DIR",
    _ASR_PRUNED_ENGINE_DIR
    if os.path.exists(os.path.join(_ASR_PRUNED_ENGINE_DIR, "llm.engine"))
    else _ASR_OFFICIAL_PRUNED_ENGINE_DIR
    if os.path.exists(os.path.join(_ASR_OFFICIAL_PRUNED_ENGINE_DIR, "llm.engine"))
    else _ASR_DIALOG_ENGINE_DIR
    if os.path.exists(os.path.join(_ASR_DIALOG_ENGINE_DIR, "llm.engine"))
    else os.path.expanduser("~/qwen3-asr-trt-edge-llm-export/engines/thinker"),
)
ASR_AUDIO_ENC_DIR = os.environ.get(
    "EDGE_LLM_ASR_AUDIO_ENC_DIR",
    os.path.expanduser(
        "~/qwen3-asr-trt-edge-llm-export/engines/audio_encoder"
    ),
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _build_env() -> dict:
    """Return a copy of os.environ with EDGELLM_PLUGIN_PATH set."""
    env = os.environ.copy()
    env["EDGELLM_PLUGIN_PATH"] = PLUGIN_PATH
    return env


# ---------------------------------------------------------------------------
# Binary runner
# ---------------------------------------------------------------------------


def run_binary(
    binary_path: str,
    args: list[str],
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a TRT-Edge-LLM binary and return the CompletedProcess.

    Raises RuntimeError on non-zero exit (unless ``check=False``).
    """
    cmd = [binary_path] + args
    logger.info("Running (acquiring GPU lock): %s", " ".join(cmd[:4]))
    with _gpu_lock:
        logger.info("GPU lock acquired, launching: %s", os.path.basename(binary_path))
        try:
            result = subprocess.run(
                cmd,
                env=_build_env(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"{os.path.basename(binary_path)} timed out after {timeout}s"
            ) from e

    if check and result.returncode != 0:
        stderr_snip = result.stderr[:1000] if result.stderr else "(empty)"
        raise RuntimeError(
            f"{os.path.basename(binary_path)} failed (exit={result.returncode}): "
            f"{stderr_snip}"
        )
    return result


# ---------------------------------------------------------------------------
# Safetensors writer (zero external deps)
# ---------------------------------------------------------------------------

_SAFETENSORS_DTYPE_MAP = {
    np.float16: "F16",
    np.float32: "F32",
    np.int32: "I32",
    np.int64: "I64",
    np.int8: "I8",
    np.uint8: "U8",
    np.bool_: "BOOL",
}


def write_safetensors(tensor: np.ndarray, name: str, path: str) -> None:
    """Write a single numpy array to a standard safetensors file.

    The tensor is written as-is (caller must cast to desired dtype first).
    """
    header = {
        name: {
            "dtype": _SAFETENSORS_DTYPE_MAP.get(
                tensor.dtype.type, str(tensor.dtype)
            ),
            "shape": list(tensor.shape),
            "data_offsets": [0, tensor.nbytes],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    with open(path, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(tensor.tobytes())


# ---------------------------------------------------------------------------
# Mel-spectrogram computation (scipy + numpy, no librosa needed)
# ---------------------------------------------------------------------------

# Whisper / Qwen3 ASR constants
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
FMIN = 0.0
FMAX = 8000.0
MEL_FLOOR = 1e-10
MIN_AUDIO_FRAMES = int(os.environ.get("EDGE_LLM_ASR_MIN_AUDIO_FRAMES", "100"))


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank() -> np.ndarray:
    """Build Slaney-norm mel filterbank [n_mels, n_fft//2+1]."""
    n_freq = N_FFT // 2 + 1
    low_mel = _hz_to_mel(np.float64(FMIN))
    high_mel = _hz_to_mel(np.float64(FMAX))
    mel_points = np.linspace(low_mel, high_mel, N_MELS + 2, dtype=np.float64)
    hz_points = _mel_to_hz(mel_points)

    bin = np.floor((n_freq - 1) * hz_points / FMAX).astype(np.int32)
    bin = np.clip(bin, 0, n_freq - 1)

    fb = np.zeros((N_MELS, n_freq), dtype=np.float64)
    for m in range(1, N_MELS + 1):
        left = int(bin[m - 1])
        center = int(bin[m])
        right = int(bin[m + 1])
        if left != center:
            for i in range(left, center):
                fb[m - 1, i] = (i - left) / (center - left)
        if center != right:
            for i in range(center, right):
                fb[m - 1, i] = (right - i) / (right - center)

    # Slaney norm: normalize each filter to unit area
    widths = hz_points[2:] - hz_points[:-2]
    fb *= (2.0 / widths)[:, np.newaxis]
    return fb.astype(np.float32)


# Build once at module level (cache)
_MEL_FILTERBANK = _build_mel_filterbank()


def audio_bytes_to_mel(
    audio_bytes: bytes,
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convert WAV bytes to log-mel spectrogram.

    Returns float32 array of shape ``[1, 128, T]`` (batch, mel, time),
    using a narrow numpy port of Whisper/Qwen3-ASR feature extraction.

    Dynamic range clamp uses max-8dB (old working behavior) instead of the
    fixed -4dB Whisper clamp (which was producing wrong mel for Qwen3 ASR).
    """
    import wave

    # -- Read WAV --
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        new_len = int(round(len(audio) * target_sr / sr))
        src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        audio = np.interp(dst_x, src_x, audio).astype(np.float32)

    # -- Centered STFT with periodic Hann window --
    pad = N_FFT // 2
    if audio.shape[0] <= 1:
        audio = np.pad(audio, (0, 2 - audio.shape[0]), mode="constant")
    audio = np.pad(audio, (pad, pad), mode="reflect")
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, N_FFT),
        strides=(audio.strides[0] * HOP_LENGTH, audio.strides[0]),
    )

    # Drop final frame (Whisper convention)
    stft = np.fft.rfft(frames * window[np.newaxis, :], n=N_FFT, axis=1)
    magnitudes = np.abs(stft[:-1].T).astype(np.float32) ** 2.0

    mel_spec = _MEL_FILTERBANK @ magnitudes
    # -- Log compression (old working: max-8dB dynamic range) --
    log_spec = np.log10(np.maximum(mel_spec, MEL_FLOOR))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    if log_spec.shape[1] < MIN_AUDIO_FRAMES:
        pad_width = MIN_AUDIO_FRAMES - log_spec.shape[1]
        log_spec = np.pad(log_spec, ((0, 0), (0, pad_width)), mode="constant")

    return log_spec[np.newaxis, :, :].astype(np.float32)  # [1, 128, T]


# ---------------------------------------------------------------------------
# Temp-file helpers
# ---------------------------------------------------------------------------


def write_temp_json(data: dict, suffix: str = ".json") -> str:
    """Write a JSON dict to a temporary file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False
    )
    json.dump(data, tmp)
    tmp.close()
    return tmp.name


def write_temp_wav(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Write audio bytes to a temporary WAV file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=suffix, delete=False
    )
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name
