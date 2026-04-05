#!/usr/bin/env python3
"""Qwen3-ASR 0.6B — TRT decoder + ORT encoder/prefill pipeline.

Uses self-exported per-layer KV ONNX (same pattern as TTS).
Hot path (decoder step): TRT FP16 with GPU-resident KV cache
Cold path (encoder, prefill): ORT CUDA EP

Usage:
    python3 asr_qwen3_trt.py --audio test.wav --model-dir /opt/models/qwen3-asr-v2
"""
import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_audio(path, target_sr=16000):
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except ImportError:
        import wave
        with wave.open(path) as w:
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != target_sr:
        ratio = target_sr / sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def compute_mel(audio, sr=16000):
    """Compute mel using WhisperFeatureExtractor."""
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=sr, n_fft=400, hop_length=160, chunk_length=30)
    features = fe(audio, sampling_rate=sr, return_tensors="np")
    return features["input_features"][0]  # [128, T]


# ---------------------------------------------------------------------------
# TRT Decoder Engine (GPU-resident KV cache)
# ---------------------------------------------------------------------------
class TRTDecoderEngine:
    """TRT decoder_step with double-buffered GPU-resident KV (same as TTS TRTTalkerEngine)."""

    def __init__(self, engine_path, n_layers=28, n_heads=8, head_dim=128, hidden_dim=1024, vocab_size=151936, max_seq=500):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq = max_seq

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Double-buffered KV cache
        kv_bytes = 1 * n_heads * max_seq * head_dim * 4  # FP32
        self.kv_a = {}
        self.kv_b = {}
        for i in range(n_layers):
            for prefix in ["past_key_", "past_value_"]:
                self.kv_a[f"{prefix}{i}"] = cuda.mem_alloc(kv_bytes)
                self.kv_b[f"{prefix}{i}"] = cuda.mem_alloc(kv_bytes)

        # I/O buffers
        self.d_emb = cuda.mem_alloc(1 * 1 * hidden_dim * 4)
        self.d_pos = cuda.mem_alloc(1 * 1 * 8)  # int64
        self.d_logits = cuda.mem_alloc(1 * 1 * vocab_size * 4)

        # Pre-allocate host output
        self._h_logits = np.empty((1, 1, vocab_size), dtype=np.float32)

        # Cached tensor names
        self.kv_names = []
        self.new_kv_names = []
        for i in range(n_layers):
            self.kv_names.append(f"past_key_{i}")
            self.kv_names.append(f"past_value_{i}")
            self.new_kv_names.append(f"new_past_key_{i}")
            self.new_kv_names.append(f"new_past_value_{i}")

        self.seq_len = 0
        self.parity = 0
        self._first = True

    def seed_kv(self, kv_dict, seq_len):
        """Copy prefill KV output to GPU buffer A."""
        for name, arr in kv_dict.items():
            if name in self.kv_a:
                a = np.ascontiguousarray(arr.astype(np.float32))
                cuda.memcpy_htod_async(self.kv_a[name], a, self.stream)
        self.stream.synchronize()
        self.seq_len = seq_len
        self.parity = 0

    def decode_step(self, input_embeds, position_id):
        """Single decode step. Returns logits [1, 1, vocab_size]."""
        ctx = self.context
        read = self.kv_a if self.parity == 0 else self.kv_b
        write = self.kv_b if self.parity == 0 else self.kv_a

        # Copy input_embeds + position_ids
        cuda.memcpy_htod_async(self.d_emb, np.ascontiguousarray(input_embeds.astype(np.float32)), self.stream)
        pos = np.array([[position_id]], dtype=np.int64)
        cuda.memcpy_htod_async(self.d_pos, pos, self.stream)

        # Bind static on first call
        if self._first:
            ctx.set_input_shape("input_embeds", (1, 1, self.hidden_dim))
            ctx.set_tensor_address("input_embeds", int(self.d_emb))
            ctx.set_input_shape("position_ids", (1, 1))
            ctx.set_tensor_address("position_ids", int(self.d_pos))
            ctx.set_tensor_address("logits", int(self.d_logits))
            self._first = False

        # Bind KV cache
        kv_shape = (1, self.n_heads, self.seq_len, self.head_dim)
        for i in range(self.n_layers):
            ctx.set_input_shape(self.kv_names[2*i], kv_shape)
            ctx.set_tensor_address(self.kv_names[2*i], int(read[f"past_key_{i}"]))
            ctx.set_tensor_address(self.new_kv_names[2*i], int(write[f"past_key_{i}"]))
            ctx.set_input_shape(self.kv_names[2*i+1], kv_shape)
            ctx.set_tensor_address(self.kv_names[2*i+1], int(read[f"past_value_{i}"]))
            ctx.set_tensor_address(self.new_kv_names[2*i+1], int(write[f"past_value_{i}"]))

        ctx.execute_async_v3(stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(self._h_logits, self.d_logits, self.stream)
        self.stream.synchronize()

        self.seq_len += 1
        self.parity ^= 1
        return self._h_logits.copy()


# ---------------------------------------------------------------------------
# ASR Pipeline
# ---------------------------------------------------------------------------
class Qwen3ASRPipeline:
    def __init__(self, model_dir, trt_engine_path=None, provider="cuda"):
        self.model_dir = model_dir
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print("Loading encoder...")
        self.encoder = ort.InferenceSession(os.path.join(model_dir, "encoder.onnx"), so, providers=providers)

        # Prefill: try self-exported (decoder_prefill) first, fall back to community (decoder_init)
        prefill_path = os.path.join(model_dir, "decoder_prefill.onnx")
        if not os.path.exists(prefill_path):
            prefill_path = os.path.join(model_dir, "decoder_init.onnx")
        print(f"Loading prefill: {os.path.basename(prefill_path)}")
        self.prefill = ort.InferenceSession(prefill_path, so, providers=providers)
        self.prefill_type = "v2" if "prefill" in prefill_path else "v1"

        # Embed tokens
        emb_path = os.path.join(model_dir, "embed_tokens.bin")
        self.embed_tokens = np.fromfile(emb_path, dtype=np.float16).reshape(-1, 1024).astype(np.float32)
        print(f"Embed tokens: {self.embed_tokens.shape}")

        # TRT decoder or ORT fallback
        self.trt_decoder = None
        self.decoder_step = None
        engine_path = trt_engine_path or os.path.join(model_dir, "asr_decoder_fp16.engine")
        if HAS_TRT and os.path.exists(engine_path):
            print(f"Loading TRT decoder: {engine_path}")
            self.trt_decoder = TRTDecoderEngine(engine_path)
        else:
            step_path = os.path.join(model_dir, "decoder_step.onnx")
            if os.path.exists(step_path):
                try:
                    print("Loading ORT decoder_step (fallback)...")
                    self.decoder_step = ort.InferenceSession(step_path, so, providers=providers)
                except Exception as e:
                    print(f"WARNING: decoder_step ORT load failed: {e}")
                    print("ASR will use prefill-only mode (slower).")

        # Tokenizer
        tok_path = os.path.join(model_dir, "tokenizer.json")
        if os.path.exists(tok_path):
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(tok_path)
        else:
            self.tokenizer = None

        # Config
        cfg_path = os.path.join(model_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Print I/O info
        print(f"Encoder: {[i.name for i in self.encoder.get_inputs()]} → {[o.name for o in self.encoder.get_outputs()]}")
        print(f"Prefill: {[i.name for i in self.prefill.get_inputs()]} → {[o.name for o in self.prefill.get_outputs()][:3]}...")
        print("Ready.")

    def transcribe(self, audio, language=None, max_tokens=200):
        t_total = time.perf_counter()

        # 1. Mel
        t0 = time.perf_counter()
        mel = compute_mel(audio)  # [128, T]
        mel_input = mel[np.newaxis, :, :]  # [1, 128, T]
        mel_ms = (time.perf_counter() - t0) * 1000

        # 2. Encoder
        t0 = time.perf_counter()
        enc_out = self.encoder.run(None, {"mel": mel_input})[0]  # [1, T', 1024]
        enc_ms = (time.perf_counter() - t0) * 1000
        audio_len = enc_out.shape[1]
        print(f"  Mel: {mel_input.shape} ({mel_ms:.0f}ms), Encoder: {enc_out.shape} ({enc_ms:.0f}ms)")

        # 3. Build prompt
        AUDIO_PAD = 151676
        prompt_ids = [
            151644, 9125, 198, 151645, 198,     # <|im_start|>system\n<|im_end|>\n
            151644, 882, 198,                     # <|im_start|>user\n
            151669,                                # <|audio_start|>
            *([AUDIO_PAD] * audio_len),
            151670, 151645, 198,                  # <|audio_end|><|im_end|>\n
            151644, 77091, 198,                   # <|im_start|>assistant\n
        ]
        if language:
            if self.tokenizer:
                lang_ids = self.tokenizer.encode(f"language {language}").ids
            else:
                lang_ids = []
            prompt_ids.extend(lang_ids + [151704])  # <asr_text>

        input_ids = np.array([prompt_ids], dtype=np.int64)
        seq_len = input_ids.shape[1]
        audio_offset = prompt_ids.index(AUDIO_PAD)

        # 4. Prefill
        t0 = time.perf_counter()
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        prefill_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "audio_features": enc_out,
            "audio_offset": np.array([audio_offset], dtype=np.int64),
        }
        # Filter to only inputs that exist
        valid_inputs = {n: v for n, v in prefill_inputs.items()
                       if n in [i.name for i in self.prefill.get_inputs()]}
        prefill_outputs = self.prefill.run(None, valid_inputs)
        prefill_names = [o.name for o in self.prefill.get_outputs()]
        prefill_map = dict(zip(prefill_names, prefill_outputs))
        prefill_ms = (time.perf_counter() - t0) * 1000

        logits = prefill_map.get("logits")
        # Collect KV cache (handle both stacked and per-layer formats)
        kv = {}
        for name, val in prefill_map.items():
            if name.startswith("past_") or name.startswith("present_"):
                clean = name.replace("present_", "past_")
                kv[clean] = val
        # Detect format: stacked [28,1,8,S,128] vs per-layer [1,8,S,128]
        self._stacked_kv = "past_keys" in kv or "past_values" in kv
        print(f"  Prefill: {prefill_ms:.0f}ms, KV: {len(kv)} tensors ({'stacked' if self._stacked_kv else 'per-layer'}), logits: {logits.shape if logits is not None else '?'}")

        # Seed TRT KV cache
        if self.trt_decoder:
            self.trt_decoder.seed_kv(kv, seq_len)

        # 5. Decode loop
        t0 = time.perf_counter()
        output_ids = []
        eos_ids = {151643, 151645}

        for step in range(max_tokens):
            next_token = int(np.argmax(logits[0, -1, :]))
            if next_token in eos_ids:
                break
            output_ids.append(next_token)

            input_embeds = self.embed_tokens[next_token][np.newaxis, np.newaxis, :]
            cur_pos = seq_len + step

            if self.trt_decoder:
                logits = self.trt_decoder.decode_step(input_embeds, cur_pos)
            elif self.decoder_step:
                # ORT fallback
                step_inputs = {"input_embeds": input_embeds, "position_ids": np.array([[cur_pos]], dtype=np.int64)}
                step_inputs.update(kv)
                step_outputs = self.decoder_step.run(None, step_inputs)
                step_names = [o.name for o in self.decoder_step.get_outputs()]
                step_map = dict(zip(step_names, step_outputs))
                logits = step_map.get("logits")
                new_kv = {}
                for k, v in step_map.items():
                    if "logits" not in k:
                        clean = k.replace("present_", "past_").replace("new_", "")
                        new_kv[clean] = v
                kv = new_kv
            else:
                # No decoder available — use prefill logits only (greedy from last position)
                print("WARNING: No decoder_step available, using prefill logits only")
                break

        decode_ms = (time.perf_counter() - t0) * 1000
        total_ms = (time.perf_counter() - t_total) * 1000

        # Decode text
        text = self.tokenizer.decode(output_ids) if self.tokenizer else f"[{len(output_ids)} tokens]"

        # Strip language prefix
        if "<asr_text>" in text:
            text = text.split("<asr_text>", 1)[1]

        audio_dur = len(audio) / 16000
        per_tok = decode_ms / max(len(output_ids), 1)
        backend = "TRT" if self.trt_decoder else "ORT"
        print(f"\n  === Qwen3-ASR ({backend}) ===")
        print(f"  Audio: {audio_dur:.1f}s")
        print(f"  Encoder: {enc_ms:.0f}ms")
        print(f"  Prefill: {prefill_ms:.0f}ms")
        print(f"  Decode ({len(output_ids)} tok): {decode_ms:.0f}ms ({per_tok:.1f}ms/tok)")
        print(f"  Total: {total_ms:.0f}ms, RTF={total_ms/1000/audio_dur:.3f}")
        print(f"  Text: {text}")

        return text, {
            "duration": round(audio_dur, 3),
            "inference_time": round(total_ms / 1000, 3),
            "rtf": round(total_ms / 1000 / audio_dur, 3),
            "n_tokens": len(output_ids),
            "per_token_ms": round(per_tok, 1),
            "backend": backend,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--model-dir", default="/opt/models/qwen3-asr-v2")
    p.add_argument("--engine", default=None)
    p.add_argument("--provider", default="cuda")
    p.add_argument("--language", default=None)
    p.add_argument("--max-tokens", type=int, default=200)
    args = p.parse_args()

    audio = load_audio(args.audio)
    print(f"Audio: {len(audio)/16000:.1f}s")

    asr = Qwen3ASRPipeline(args.model_dir, trt_engine_path=args.engine, provider=args.provider)
    text, meta = asr.transcribe(audio, language=args.language, max_tokens=args.max_tokens)
    print(f"\nTranscription: {text}")
    print(f"Metadata: {meta}")


if __name__ == "__main__":
    main()
