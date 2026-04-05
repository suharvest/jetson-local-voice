#!/usr/bin/env python3
"""Qwen3-ASR 0.6B — ORT CUDA EP baseline inference.

Uses andrewleech's ONNX export: encoder + decoder_init + decoder_step.

Usage:
    python3 asr_qwen3_ort.py --audio test.wav --model-dir /opt/models/qwen3-asr
"""
import argparse
import json
import os
import time
import wave

import numpy as np
import onnxruntime as ort


def load_audio(path, target_sr=16000):
    """Load WAV and resample to 16kHz if needed."""
    with wave.open(path) as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != target_sr:
        # Simple linear interpolation resample (no scipy needed)
        ratio = target_sr / sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def compute_fbank(audio, sr=16000, n_mels=128, n_fft=400, hop=160, win=400, fmax=8000):
    """Compute 128-dim Fbank features at 100Hz (matching Qwen3-ASR)."""
    # STFT
    padding = (n_fft - hop) // 2
    audio_padded = np.pad(audio, (padding, padding), mode='reflect')
    window = np.hanning(win + 1)[:-1]
    # Pad window to n_fft if needed
    if len(window) < n_fft:
        window = np.pad(window, (0, n_fft - len(window)))

    n_frames = 1 + (len(audio_padded) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        audio_padded,
        shape=(n_frames, n_fft),
        strides=(audio_padded.strides[0] * hop, audio_padded.strides[0])
    ).copy()
    frames *= window[:n_fft]
    spec = np.fft.rfft(frames, n=n_fft)
    power = np.abs(spec) ** 2

    # Mel filterbank
    fmin = 0
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = 700 * (10 ** (mels / 2595) - 1)
    n_bins = n_fft // 2 + 1
    fftfreqs = np.linspace(0, sr / 2, n_bins)
    fb = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        for j in range(n_bins):
            if freqs[i] <= fftfreqs[j] <= freqs[i + 1] and freqs[i + 1] > freqs[i]:
                fb[i, j] = (fftfreqs[j] - freqs[i]) / (freqs[i + 1] - freqs[i])
            elif freqs[i + 1] < fftfreqs[j] <= freqs[i + 2] and freqs[i + 2] > freqs[i + 1]:
                fb[i, j] = (freqs[i + 2] - fftfreqs[j]) / (freqs[i + 2] - freqs[i + 1])
        # Slaney normalization
        enorm = 2.0 / (freqs[i + 2] - freqs[i]) if freqs[i + 2] > freqs[i] else 1.0
        fb[i] *= enorm

    mel_spec = fb @ power.T  # [n_mels, n_frames]
    mel_spec = np.log(np.maximum(mel_spec, 1e-10))
    return mel_spec.T.astype(np.float32)  # [n_frames, n_mels]


class Qwen3ASR:
    def __init__(self, model_dir, provider="cuda"):
        self.model_dir = model_dir
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print("Loading encoder...")
        self.encoder = ort.InferenceSession(
            os.path.join(model_dir, "encoder.onnx"), so, providers=providers)

        print("Loading decoder_init...")
        self.decoder_init = ort.InferenceSession(
            os.path.join(model_dir, "decoder_init.onnx"), so, providers=providers)

        print("Loading decoder_step...")
        self.decoder_step = ort.InferenceSession(
            os.path.join(model_dir, "decoder_step.onnx"), so, providers=providers)

        # Load embedding table
        emb_path = os.path.join(model_dir, "embed_tokens.bin")
        if os.path.exists(emb_path):
            self.embed_tokens = np.fromfile(emb_path, dtype=np.float16).reshape(-1, 1024).astype(np.float32)
            print(f"Embed tokens: {self.embed_tokens.shape}")
        else:
            self.embed_tokens = None

        # Load tokenizer
        tok_path = os.path.join(model_dir, "tokenizer.json")
        if os.path.exists(tok_path):
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(tok_path)
            print("Tokenizer loaded")
        else:
            self.tokenizer = None

        # Get I/O info
        print("Encoder inputs:", [(i.name, i.shape) for i in self.encoder.get_inputs()])
        print("Encoder outputs:", [(o.name, o.shape) for o in self.encoder.get_outputs()])
        print("Decoder init inputs:", [(i.name, i.shape) for i in self.decoder_init.get_inputs()])
        print("Decoder step inputs:", [(i.name, i.shape) for i in self.decoder_step.get_inputs()])
        print("Ready.")

    def transcribe(self, audio, language="auto", max_tokens=200):
        """Transcribe audio to text."""
        t_total = time.perf_counter()

        # 1. Compute Fbank features
        t0 = time.perf_counter()
        fbank = compute_fbank(audio)
        fbank_ms = (time.perf_counter() - t0) * 1000
        print(f"  Fbank: {fbank.shape} ({fbank_ms:.0f}ms)")

        # 2. Encoder
        t0 = time.perf_counter()
        # Encoder expects [batch, mel_bins, time] — transpose
        enc_input = fbank.T[np.newaxis, :, :]  # [1, 128, T]
        enc_in_name = self.encoder.get_inputs()[0].name
        enc_outputs = self.encoder.run(None, {enc_in_name: enc_input})
        encoder_hidden = enc_outputs[0]  # [1, T', D]
        enc_ms = (time.perf_counter() - t0) * 1000
        print(f"  Encoder: {encoder_hidden.shape} ({enc_ms:.0f}ms)")

        # 3. Build prompt tokens for Qwen3-ASR
        # Format: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        #         <|im_start|>user\n<|audio_start|>[audio_pad × N]<|audio_end|>\n
        #         <asr_text><|im_end|>\n<|im_start|>assistant\n
        audio_len = encoder_hidden.shape[1]  # number of audio tokens from encoder

        # Special token IDs
        im_start = 151644
        im_end = 151645
        audio_start = 151669
        audio_end = 151670
        audio_pad = 151676
        asr_text = 151704

        # Build token sequence
        system_text = "You are a helpful assistant."
        if self.tokenizer:
            sys_ids = self.tokenizer.encode(system_text).ids
        else:
            sys_ids = [2610, 525, 264, 10950, 17847, 13]

        prompt_ids = (
            [im_start] + [8948, 198] +  # system\n
            sys_ids + [im_end, 198] +    # text<|im_end|>\n
            [im_start] + [872, 198] +    # user\n
            [audio_start] + [audio_pad] * audio_len + [audio_end, 198] +  # audio
            [asr_text, im_end, 198] +    # <asr_text><|im_end|>\n
            [im_start] + [77091, 198]    # assistant\n
        )
        prompt_ids = np.array([prompt_ids], dtype=np.int64)
        print(f"  Prompt: {prompt_ids.shape[1]} tokens (audio_pad×{audio_len})")

        # 4. Decoder init (prefill)
        t0 = time.perf_counter()
        seq_len = prompt_ids.shape[1]
        # audio_offset = index of first audio_pad token in prompt
        audio_offset = list(prompt_ids[0]).index(audio_pad)
        init_inputs = {}
        for inp in self.decoder_init.get_inputs():
            if inp.name == "input_ids":
                init_inputs[inp.name] = prompt_ids
            elif inp.name == "position_ids":
                init_inputs[inp.name] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            elif inp.name == "audio_features":
                init_inputs[inp.name] = encoder_hidden
            elif inp.name == "audio_offset":
                init_inputs[inp.name] = np.array([audio_offset], dtype=np.int64)
        print(f"  Decoder init inputs: {[(k, v.shape) for k,v in init_inputs.items()]}")
        init_outputs = self.decoder_init.run(None, init_inputs)
        init_ms = (time.perf_counter() - t0) * 1000

        # Parse outputs
        init_out_names = [o.name for o in self.decoder_init.get_outputs()]
        init_map = dict(zip(init_out_names, init_outputs))
        logits = None
        kv = {}
        for name, val in init_map.items():
            if "logits" in name:
                logits = val
            else:
                # Map init output names to step input names
                # e.g., "present_keys" → "past_keys"
                step_name = name.replace("present_", "past_")
                kv[step_name] = val
        print(f"  Decoder init: {init_ms:.0f}ms, logits: {logits.shape if logits is not None else '?'}")
        print(f"  KV names: {list(kv.keys())}, shapes: {[v.shape for v in kv.values()]}")

        # 5. Autoregressive decode
        t0 = time.perf_counter()
        output_ids = []
        eos_token_id = 151645  # <|im_end|>

        for step in range(max_tokens):
            if logits is None:
                break
            # Sample (greedy)
            next_token = int(np.argmax(logits[0, -1, :]))
            if next_token == eos_token_id:
                print(f"  EOS at step {step}")
                break
            output_ids.append(next_token)

            # Prepare decoder_step input
            if self.embed_tokens is not None:
                input_embeds = self.embed_tokens[next_token][np.newaxis, np.newaxis, :]
            else:
                input_embeds = np.zeros((1, 1, 1024), dtype=np.float32)

            # Position ID = prefill length + step
            cur_pos = seq_len + audio_len + step
            position_ids = np.array([[cur_pos]], dtype=np.int64)

            step_inputs = {}
            for inp in self.decoder_step.get_inputs():
                if "input_embeds" in inp.name or "inputs_embeds" in inp.name:
                    step_inputs[inp.name] = input_embeds
                elif inp.name == "position_ids":
                    step_inputs[inp.name] = position_ids
                elif inp.name in kv:
                    step_inputs[inp.name] = kv[inp.name]

            step_outputs = self.decoder_step.run(None, step_inputs)
            step_out_names = [o.name for o in self.decoder_step.get_outputs()]
            step_map = dict(zip(step_out_names, step_outputs))

            logits = None
            new_kv = {}
            for name, val in step_map.items():
                if "logits" in name:
                    logits = val
                else:
                    # Map output KV names to input KV names for next step
                    # e.g., "present_keys" → "past_keys", "present_values" → "past_values"
                    clean_name = name.replace("present_", "past_").replace("new_", "")
                    new_kv[clean_name] = val
            kv = new_kv if new_kv else kv

        decode_ms = (time.perf_counter() - t0) * 1000
        total_ms = (time.perf_counter() - t_total) * 1000

        # Decode tokens to text
        if self.tokenizer:
            text = self.tokenizer.decode(output_ids)
        else:
            text = f"[{len(output_ids)} tokens]"

        audio_dur = len(audio) / 16000
        print(f"\n  === Qwen3-ASR Results ===")
        print(f"  Audio: {audio_dur:.1f}s")
        print(f"  Fbank: {fbank_ms:.0f}ms")
        print(f"  Encoder: {enc_ms:.0f}ms")
        print(f"  Decoder init: {init_ms:.0f}ms")
        print(f"  Decode ({len(output_ids)} tokens): {decode_ms:.0f}ms ({decode_ms/max(len(output_ids),1):.1f}ms/tok)")
        print(f"  Total: {total_ms:.0f}ms, RTF={total_ms/1000/audio_dur:.3f}")
        print(f"  Text: {text}")

        return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--model-dir", default="/opt/models/qwen3-asr")
    p.add_argument("--provider", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max-tokens", type=int, default=200)
    args = p.parse_args()

    audio = load_audio(args.audio)
    print(f"Audio: {len(audio)/16000:.1f}s")

    asr = Qwen3ASR(args.model_dir, provider=args.provider)
    text = asr.transcribe(audio, max_tokens=args.max_tokens)
    print(f"\nTranscription: {text}")


if __name__ == "__main__":
    main()
