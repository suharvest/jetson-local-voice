"""Qwen3-ASR backend — C++ pipeline (encoder + prefill + TRT decoder) via pybind11.

Supports: OFFLINE, MULTI_LANGUAGE, LANGUAGE_ID
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
from typing import Optional

import numpy as np

from asr_backend import ASRBackend, ASRCapability, TranscriptionResult

logger = logging.getLogger(__name__)

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


class Qwen3ASRBackend(ASRBackend):

    def __init__(self):
        self._pipeline = None     # C++ ASRPipeline
        # Fallback: Python ORT sessions
        self._encoder = None
        self._prefill = None
        self._decoder = None      # C++ TRT ASRDecoder (legacy)
        self._decoder_ort = None  # ORT fallback
        self._embed_tokens = None
        self._tokenizer = None
        self._ready = False
        self._use_cpp_pipeline = False

    @property
    def name(self) -> str:
        return "qwen3_asr"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.MULTI_LANGUAGE, ASRCapability.LANGUAGE_ID}

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        logger.info("Loading Qwen3-ASR from %s", _BASE)
        t0 = time.time()

        # Try C++ pipeline first (encoder + prefill + TRT decode, all in C++)
        engine_path = None
        for engine_name in ["asr_decoder_bf16.engine", "asr_decoder_fp16.engine"]:
            p = os.path.join(_BASE, engine_name)
            if os.path.exists(p):
                engine_path = p
                break

        if engine_path:
            try:
                import qwen3_tts_engine
                self._pipeline = qwen3_tts_engine.ASRPipeline(
                    _BASE, engine_path, 0)
                self._use_cpp_pipeline = True
                logger.info("C++ ASR pipeline loaded (encoder+prefill+TRT)")
            except Exception as e:
                logger.warning("C++ ASR pipeline failed: %s, falling back to Python", e)
                self._pipeline = None

        # Fall back to Python ORT if C++ pipeline not available
        if not self._use_cpp_pipeline:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

            logger.info("Loading encoder...")
            self._encoder = ort.InferenceSession(
                os.path.join(_BASE, "encoder.onnx"), so, providers=providers)
            logger.info("Encoder OK")

            for name in ["decoder_prefill.onnx", "decoder_init.onnx"]:
                path = os.path.join(_BASE, name)
                if not os.path.exists(path):
                    path = os.path.join("/opt/models/qwen3-asr", name)
                if os.path.exists(path):
                    try:
                        self._prefill = ort.InferenceSession(path, so, providers=providers)
                        logger.info("Prefill loaded: %s", path)
                        break
                    except Exception as e:
                        logger.warning("Prefill %s failed: %s", name, e)

            # Embed tokens
            emb_path = os.path.join(_BASE, "embed_tokens.bin")
            if os.path.exists(emb_path):
                self._embed_tokens = np.fromfile(emb_path, dtype=np.float16).reshape(-1, 1024).astype(np.float32)

            # TRT decoder via pybind11
            if engine_path:
                try:
                    import qwen3_tts_engine
                    self._decoder = qwen3_tts_engine.ASRDecoder(
                        engine_path, 28, 1024, 8, 128, 151936, 500)
                    self._trt_max_seq = 500
                    logger.info("ASR TRT decoder loaded: %s", engine_path)
                except Exception as e:
                    logger.warning("TRT decoder %s failed: %s", engine_path, e)

            # ORT decoder fallback
            if self._decoder is None:
                path = os.path.join(_BASE, "decoder_step.onnx")
                if os.path.exists(path):
                    try:
                        self._decoder_ort = ort.InferenceSession(path, so, providers=providers)
                        logger.info("ASR ORT decoder loaded: %s", path)
                    except Exception as e:
                        logger.warning("ORT decoder load failed: %s", e)

        # Tokenizer (needed by both paths)
        tok_path = os.path.join(_BASE, "tokenizer.json")
        if os.path.exists(tok_path):
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(tok_path)

        backend_name = "C++" if self._use_cpp_pipeline else (
            "TRT" if self._decoder else "ORT" if self._decoder_ort else "none")
        logger.info("Qwen3-ASR loaded in %.1fs (decoder: %s)",
                     time.time() - t0, backend_name)
        self._ready = True

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        audio = self._bytes_to_float(audio_bytes)
        t_total = time.perf_counter()

        # 1. Mel spectrogram (always computed in Python)
        mel = self._compute_mel(audio)  # [1, 128, T]

        if self._use_cpp_pipeline:
            return self._transcribe_cpp(mel, audio, language, t_total)
        else:
            return self._transcribe_python(mel, audio, language, t_total)

    def _transcribe_cpp(self, mel, audio, language, t_total):
        """Full C++ pipeline: mel → encoder → prefill → TRT decode."""
        mel_len = mel.shape[2]

        # Get exact encoder output length via C++ run_encoder (avoids
        # guessing the downsampling ratio).  The encoder is fast (~80-280 ms)
        # and Transcribe() will re-run it, but correctness > speed here.
        # If run_encoder is not available, fall back to the known ratio.
        mel_c = np.ascontiguousarray(mel, dtype=np.float32)
        if hasattr(self._pipeline, 'run_encoder'):
            audio_len_est = self._pipeline.run_encoder(mel_c)
        else:
            # Qwen3-ASR encoder downsamples mel by ~7.69x (100 mel → 13 features)
            audio_len_est = mel_len * 13 // 100

        lang = language if language != "auto" else None
        prompt_ids = self._build_prompt(audio_len_est, lang)
        audio_offset = prompt_ids.index(AUDIO_PAD)

        result = self._pipeline.transcribe(
            mel=mel_c,
            prompt_ids=prompt_ids,
            audio_offset=audio_offset,
            max_tokens=200,
        )

        total_ms = (time.perf_counter() - t_total) * 1000
        text_ids = result["text_ids"]

        # Decode text
        text = self._tokenizer.decode(text_ids) if self._tokenizer else f"[{len(text_ids)} tokens]"
        if "<asr_text>" in text:
            text = text.split("<asr_text>", 1)[1]

        audio_dur = len(audio) / 16000

        return TranscriptionResult(
            text=text.strip(),
            duration=round(audio_dur, 3),
            inference_time=round(total_ms / 1000, 3),
            rtf=round(total_ms / 1000 / audio_dur, 3) if audio_dur > 0 else 0,
            n_tokens=result["n_tokens"],
            per_token_ms=round(result.get("per_token_ms", 0), 1),
            backend="C++",
        )

    def _transcribe_python(self, mel, audio, language, t_total):
        """Python ORT encoder/prefill + TRT/ORT decode (legacy path)."""
        import onnxruntime as ort

        # 2. Encoder
        t0 = time.perf_counter()
        enc_out = self._encoder.run(None, {"mel": mel})[0]  # [1, T', 1024]
        enc_ms = (time.perf_counter() - t0) * 1000
        audio_len = enc_out.shape[1]

        # 3. Prompt
        lang = language if language != "auto" else None
        prompt_ids = self._build_prompt(audio_len, lang)
        input_ids = np.array([prompt_ids], dtype=np.int64)
        seq_len = len(prompt_ids)
        audio_offset = prompt_ids.index(AUDIO_PAD)

        # 4. Prefill
        t0 = time.perf_counter()
        prefill_inputs = {
            "input_ids": input_ids,
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
            "audio_features": enc_out,
            "audio_offset": np.array([audio_offset], dtype=np.int64),
        }
        valid = {n: v for n, v in prefill_inputs.items()
                 if n in [i.name for i in self._prefill.get_inputs()]}
        outputs = self._prefill.run(None, valid)
        out_map = dict(zip([o.name for o in self._prefill.get_outputs()], outputs))
        prefill_ms = (time.perf_counter() - t0) * 1000

        logits = out_map.get("logits")
        kv = {k.replace("present_", "past_"): v for k, v in out_map.items()
              if k.startswith("past_") or k.startswith("present_")}

        # Seed TRT KV (only if seq fits in engine's max profile)
        use_trt = self._decoder and seq_len <= getattr(self, '_trt_max_seq', 500)
        if use_trt:
            self._decoder.reset()
            for k in list(kv.keys()):
                kv[k] = np.ascontiguousarray(kv[k].astype(np.float32))
            self._decoder.seed_kv(kv, seq_len)
        elif self._decoder:
            logger.debug("seq_len %d > TRT max %d, using ORT fallback", seq_len, self._trt_max_seq)

        # 5. Decode
        t0 = time.perf_counter()
        output_ids = []
        for step in range(200):
            next_token = int(np.argmax(logits[0, -1, :]))
            if next_token in EOS_IDS:
                break
            output_ids.append(next_token)

            embeds = self._embed_tokens[next_token][np.newaxis, np.newaxis, :]
            cur_pos = seq_len + step

            if use_trt:
                logits = self._decoder.decode_step(embeds, 151936)
            elif self._decoder_ort:
                step_in = {"input_embeds": embeds,
                           "position_ids": np.array([[cur_pos]], dtype=np.int64)}
                step_in.update(kv)
                step_out = self._decoder_ort.run(None, step_in)
                step_map = dict(zip([o.name for o in self._decoder_ort.get_outputs()], step_out))
                logits = step_map.get("logits")
                kv = {k.replace("present_", "past_").replace("new_", ""): v
                      for k, v in step_map.items() if "logits" not in k}
            else:
                break

        decode_ms = (time.perf_counter() - t0) * 1000
        total_ms = (time.perf_counter() - t_total) * 1000

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
        if language:
            if self._tokenizer:
                lang_ids = self._tokenizer.encode(f"language {language}").ids
            else:
                lang_ids = []
            ids.extend(lang_ids + [ASR_TEXT])
        return ids

    def _compute_mel(self, audio):
        from transformers import WhisperFeatureExtractor
        # Use chunk_length matching actual audio to avoid excessive padding
        audio_secs = len(audio) / 16000
        chunk_len = min(30, int(audio_secs) + 1)  # Round up, max 30s
        fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000,
                                     n_fft=400, hop_length=160, chunk_length=chunk_len)
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
