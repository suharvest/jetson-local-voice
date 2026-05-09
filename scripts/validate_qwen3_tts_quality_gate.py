#!/usr/bin/env python3
"""Validate Qwen3-TTS outputs with duration/energy and Qwen3-ASR round-trip.

This script is intentionally self-contained and numpy-only for mel extraction.
It avoids scipy/librosa so Jetson host Python package drift cannot masquerade
as an ASR quality failure.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import tempfile
import time
import uuid
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
FMIN = 0.0
FMAX = 8000.0
MEL_FLOOR = 1e-10


def hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank() -> np.ndarray:
    n_freq = N_FFT // 2 + 1
    low_mel = hz_to_mel(np.float64(FMIN))
    high_mel = hz_to_mel(np.float64(FMAX))
    mel_points = np.linspace(low_mel, high_mel, N_MELS + 2, dtype=np.float64)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_freq - 1) * hz_points / FMAX).astype(np.int32)
    bins = np.clip(bins, 0, n_freq - 1)

    fb = np.zeros((N_MELS, n_freq), dtype=np.float64)
    for m in range(1, N_MELS + 1):
        left = int(bins[m - 1])
        center = int(bins[m])
        right = int(bins[m + 1])
        if left != center:
            for i in range(left, center):
                fb[m - 1, i] = (i - left) / (center - left)
        if center != right:
            for i in range(center, right):
                fb[m - 1, i] = (right - i) / (right - center)
    widths = hz_points[2:] - hz_points[:-2]
    fb *= (2.0 / widths)[:, np.newaxis]
    return fb.astype(np.float32)


MEL_FILTERBANK = build_mel_filterbank()


def wav_bytes_to_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
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
    return audio, sr


def audio_bytes_to_mel(audio_bytes: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = wav_bytes_to_audio(audio_bytes)
    if sr != target_sr:
        new_len = int(round(len(audio) * target_sr / sr))
        src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        audio = np.interp(dst_x, src_x, audio).astype(np.float32)

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
    stft = np.fft.rfft(frames * window[np.newaxis, :], n=N_FFT, axis=1)
    magnitudes = np.abs(stft[:-1].T).astype(np.float32) ** 2.0
    mel_spec = MEL_FILTERBANK @ magnitudes
    log_spec = np.log10(np.maximum(mel_spec, MEL_FLOOR))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    if log_spec.shape[1] < 100:
        log_spec = np.pad(log_spec, ((0, 0), (0, 100 - log_spec.shape[1])), mode="constant")
    return log_spec[np.newaxis, :, :].astype(np.float32)


def write_safetensors(tensor: np.ndarray, name: str, path: Path) -> None:
    dtype_map = {np.float16: "F16", np.float32: "F32"}
    header = {name: {"dtype": dtype_map[tensor.dtype.type], "shape": list(tensor.shape), "data_offsets": [0, tensor.nbytes]}}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    with path.open("wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(tensor.tobytes())


def wav_metrics(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        duration_s = wav.getnframes() / wav.getframerate()
    rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    silence_ratio = float(np.mean(np.abs(samples) < 64.0)) if samples.size else 1.0
    return {"duration_s": round(duration_s, 3), "rms": round(rms, 1), "peak": round(peak, 1), "silence_ratio": round(silence_ratio, 4)}


def transcribe_with_worker(args: argparse.Namespace, wav_path: Path) -> dict:
    audio_bytes = wav_path.read_bytes()
    mel = audio_bytes_to_mel(audio_bytes).astype(np.float16)
    with tempfile.TemporaryDirectory(prefix="qwen3_tts_quality_") as tmp:
        mel_path = Path(tmp) / "mel.safetensors"
        write_safetensors(mel, args.mel_tensor_name, mel_path)
        env = os.environ.copy()
        env["EDGELLM_PLUGIN_PATH"] = args.asr_plugin
        env.setdefault("EDGE_LLM_ASR_CUDA_GRAPH", "0")
        proc = subprocess.Popen(
            [args.asr_worker, "--engineDir", args.asr_engine, "--multimodalEngineDir", args.asr_audio_encoder],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdin is not None and proc.stdout is not None
        ready = proc.stdout.readline()
        if not ready:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"ASR worker failed to start: {stderr[-2000:]}")
        ready_json = json.loads(ready)
        if ready_json.get("event") != "ready":
            raise RuntimeError(f"Unexpected ASR ready event: {ready_json}")
        request = {
            "id": uuid.uuid4().hex,
            "requests": [{
                "messages": [{
                    "role": "user",
                    "content": [{"type": "audio", "audio": str(mel_path)}],
                }]
            }],
            "batch_size": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_generate_length": args.asr_max_generate_length,
            "apply_chat_template": True,
            "add_generation_prompt": True,
        }
        t0 = time.perf_counter()
        proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        elapsed_s = time.perf_counter() - t0
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if not line:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"ASR worker exited before response: {stderr[-2000:]}")
        result = json.loads(line)
        if not result.get("ok"):
            raise RuntimeError(f"ASR worker failed: {result}")
        text = result.get("responses", [{}])[0].get("output_text", "")
        text = strip_language_prefix(text)
        return {"text": text, "asr_worker_s": round(elapsed_s, 3), "asr_init_ms": ready_json.get("init_ms")}


def strip_language_prefix(text: str) -> str:
    if not text.startswith("language "):
        return text
    for language in (
        "Chinese",
        "English",
        "Cantonese",
        "Japanese",
        "Korean",
        "French",
        "German",
        "Italian",
        "Portuguese",
        "Russian",
        "Spanish",
    ):
        prefix = "language " + language
        if text.startswith(prefix):
            return text[len(prefix):].lstrip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", nargs="+", type=Path)
    parser.add_argument("--asr-worker", default="/home/harvest/project/jetson-voice/build/edgellm_voice_worker/workers/qwen3_asr_worker")
    parser.add_argument("--asr-plugin", default="/home/harvest/project/tensorrt-edge-llm/build_sm87/libNvInfer_edgellm_plugin.so")
    parser.add_argument("--asr-engine", default="/home/harvest/qwen3-asr-edgellm-runtime/engines/thinker_prunedembed35k_kv512")
    parser.add_argument("--asr-audio-encoder", default="/home/harvest/qwen3-asr-trt-edge-llm-export/engines/audio_encoder")
    parser.add_argument("--mel-tensor-name", default="mel")
    parser.add_argument("--asr-max-generate-length", type=int, default=200)
    args = parser.parse_args()

    report = []
    for wav_path in args.wav:
        item = {"path": str(wav_path), "metrics": wav_metrics(wav_path)}
        item["asr"] = transcribe_with_worker(args, wav_path)
        report.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    print(json.dumps({"summary": report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
