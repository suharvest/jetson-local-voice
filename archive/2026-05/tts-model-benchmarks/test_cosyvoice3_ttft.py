"""CosyVoice3 ONNX TTFT benchmark on Jetson Orin NX.

Uses the ayousanz/cosy-voice3-onnx pure ONNX inference pipeline.
Measures per-component timing and TTFT.
"""
import sys
sys.path.insert(0, "/opt/models/cosyvoice3-onnx/scripts")

import time
import numpy as np
import soundfile as sf

# Monkey-patch the inference class to add timing
from onnx_inference_pure import PureOnnxCosyVoice3

MODEL_DIR = "/opt/models/cosyvoice3-onnx"
PROMPT_WAV = "/opt/models/cosyvoice3-onnx/prompts/en_female_nova_greeting.wav"
PROMPT_TEXT = "Hello, my name is Sarah. I'm excited to help you with your project today."

TEST_TEXTS = [
    ("好的。", "zh"),
    ("你好，很高兴认识你。", "zh"),
    ("今天天气真不错，我们出去走走吧。", "zh"),
    ("Hello, nice to meet you.", "en"),
]


def timed_inference(model, text, prompt_wav, prompt_text):
    """Run inference with per-stage timing."""
    timings = {}

    # Stage 0: Prompt processing
    t0 = time.time()
    embedding = model.extract_speaker_embedding(prompt_wav)
    timings["speaker_embed"] = time.time() - t0

    t0 = time.time()
    prompt_tokens = model.extract_speech_tokens(prompt_wav)
    timings["speech_tokenize"] = time.time() - t0

    t0 = time.time()
    prompt_mel = model.extract_speech_mel(prompt_wav)
    timings["mel_extract"] = time.time() - t0

    timings["prompt_total"] = timings["speaker_embed"] + timings["speech_tokenize"] + timings["mel_extract"]

    # Stage 1: LLM
    t0 = time.time()
    speech_tokens = model.llm_inference(
        text, prompt_text=prompt_text,
        prompt_speech_tokens=prompt_tokens,
        sampling_k=25, max_len=500, min_len=10,
    )
    timings["llm"] = time.time() - t0
    timings["llm_tokens"] = speech_tokens.shape[1]

    # Stage 2: Flow
    t0 = time.time()
    mel = model.flow_inference(
        speech_tokens, embedding,
        prompt_tokens=prompt_tokens,
        prompt_mel=prompt_mel,
        n_timesteps=10,
    )
    timings["flow"] = time.time() - t0

    # Stage 3: HiFT vocoder
    t0 = time.time()
    audio = model.hift_inference(mel)
    timings["hift"] = time.time() - t0

    timings["total"] = timings["prompt_total"] + timings["llm"] + timings["flow"] + timings["hift"]
    timings["audio_dur"] = len(audio.reshape(-1)) / 24000
    # TTFT = prompt + LLM (must finish all tokens) + flow + hift
    # For streaming: TTFT = prompt + first LLM token time (but not implemented)
    timings["ttft"] = timings["total"]  # no streaming, TTFT = total

    return audio, timings


def main():
    print("=" * 75)
    print("CosyVoice3 ONNX TTFT Benchmark — Jetson Orin NX (CUDA)")
    print("=" * 75)

    print("\nLoading models (this may take a while)...")
    t0 = time.time()
    model = PureOnnxCosyVoice3(MODEL_DIR, use_fp16=True)
    print(f"Models loaded in {time.time()-t0:.1f}s\n")

    # Warmup
    print("Warmup run...")
    _, _ = timed_inference(model, "Hello.", PROMPT_WAV, PROMPT_TEXT)
    print("Warmup done.\n")

    # Benchmark
    print(f"{'Text':<40} {'TTFT':>8} {'Prompt':>8} {'LLM':>8} {'Flow':>8} {'HiFT':>8} {'Tokens':>6} {'Audio':>7} {'RTF':>6}")
    print("-" * 100)

    for text, lang in TEST_TEXTS:
        results = []
        for _ in range(2):  # run twice, take second (warmed up)
            audio, t = timed_inference(model, text, PROMPT_WAV, PROMPT_TEXT)
            results.append(t)
        t = results[-1]
        short = text[:38] + ".." if len(text) > 40 else text
        print(f"{short:<40} {t['ttft']*1000:>7.0f}ms {t['prompt_total']*1000:>7.0f}ms "
              f"{t['llm']*1000:>7.0f}ms {t['flow']*1000:>7.0f}ms {t['hift']*1000:>7.0f}ms "
              f"{t['llm_tokens']:>5} {t['audio_dur']:>6.2f}s {t['total']/t['audio_dur']:>6.2f}")

    # Detailed breakdown for one text
    print(f"\n{'='*75}")
    print("Detailed breakdown: \"你好，很高兴认识你。\"")
    print(f"{'='*75}")
    _, t = timed_inference(model, "你好，很高兴认识你。", PROMPT_WAV, PROMPT_TEXT)
    for k, v in t.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v*1000:>8.1f}ms")
        else:
            print(f"  {k:<20}: {v}")

    # Save sample
    sf.write("/tmp/cosyvoice3_output.wav", audio.reshape(-1), 24000)
    print(f"\nSaved sample: /tmp/cosyvoice3_output.wav")

    print(f"\n{'='*75}")
    print("Comparison:")
    print(f"{'='*75}")
    print("  Kokoro:      TTFT ~130ms (short), ~1s (sentence)")
    print("  F5-TTS:      TTFT ~2500ms (shortest, NFE=8)")
    print(f"  CosyVoice3:  TTFT ~{t['ttft']*1000:.0f}ms (see above)")


if __name__ == "__main__":
    main()
