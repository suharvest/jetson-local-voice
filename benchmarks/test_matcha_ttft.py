"""Matcha TTS (sherpa-onnx) TTFT benchmark on Jetson Orin NX.

Uses sherpa-onnx OfflineTts with Matcha acoustic model + Vocos vocoder.
Measures TTFT and compares with Kokoro/F5-TTS/CosyVoice3.
"""
import time
import sherpa_onnx
import numpy as np
import struct
import io

MODEL_DIR = "/opt/models/matcha-icefall-zh-en"
MODEL_FILE = f"{MODEL_DIR}/model-steps-3.onnx"
VOCODER_FILE = f"{MODEL_DIR}/vocos-16khz-univ.onnx"
LEXICON = f"{MODEL_DIR}/lexicon.txt"
TOKENS = f"{MODEL_DIR}/tokens.txt"
DATA_DIR = f"{MODEL_DIR}/espeak-ng-data"
DICT_DIR = MODEL_DIR

SAMPLE_RATE = 16000

TEST_TEXTS = [
    "好的。",
    "没问题。",
    "你好，很高兴认识你。",
    "今天天气真不错，我们出去走走吧。",
    "你好，我是你的智能助手，很高兴认识你。今天天气真不错，我们聊聊吧。",
    "Hello, nice to meet you.",
]


def create_tts(provider="cuda", num_threads=4):
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=MODEL_FILE,
                vocoder=VOCODER_FILE,
                lexicon=LEXICON,
                tokens=TOKENS,
                data_dir=DATA_DIR,
                dict_dir=DICT_DIR,
            ),
            provider=provider,
            num_threads=num_threads,
            debug=False,
        ),
    )
    return sherpa_onnx.OfflineTts(config)


def measure_ttft(tts, text, speed=1.0, sid=0):
    """Measure TTFT = time to generate complete audio (no streaming for OfflineTts)."""
    t0 = time.time()
    audio = tts.generate(text, sid=sid, speed=speed)
    elapsed = time.time() - t0
    duration = len(audio.samples) / audio.sample_rate
    return elapsed, duration, len(audio.samples)


def measure_ttft_streaming(tts, text, speed=1.0, sid=0):
    """Measure TTFT using sentence callback (true streaming).

    sherpa-onnx Matcha supports sentence-level callback during generation.
    TTFT = time until first sentence callback fires.
    """
    first_chunk_time = None
    first_chunk_samples = 0
    chunks = []
    t0 = time.time()

    def on_sentence(samples, progress):
        nonlocal first_chunk_time, first_chunk_samples
        elapsed = time.time() - t0
        if first_chunk_time is None:
            first_chunk_time = elapsed
            first_chunk_samples = len(samples)
        chunks.append((len(samples), elapsed))
        return 1  # continue

    audio = tts.generate(text, sid=sid, speed=speed, callback=on_sentence)
    total_time = time.time() - t0
    total_dur = len(audio.samples) / audio.sample_rate

    return {
        "total_time": total_time,
        "total_dur": total_dur,
        "ttft": first_chunk_time if first_chunk_time else total_time,
        "first_chunk_samples": first_chunk_samples,
        "first_chunk_dur": first_chunk_samples / audio.sample_rate if first_chunk_samples else 0,
        "n_chunks": len(chunks),
        "chunks": chunks,
        "rtf": total_time / total_dur if total_dur > 0 else 0,
    }


def main():
    print("=" * 80)
    print("Matcha TTS (sherpa-onnx) TTFT Benchmark — Jetson Orin NX")
    print("=" * 80)

    # Test both CPU and CUDA
    for provider in ["cuda", "cpu"]:
        print(f"\n{'='*80}")
        print(f"Provider: {provider.upper()}")
        print(f"{'='*80}")

        print("Loading model...")
        t0 = time.time()
        tts = create_tts(provider=provider)
        print(f"Model loaded in {time.time()-t0:.2f}s, sample_rate={tts.sample_rate}")

        # Warmup
        tts.generate("你好。", sid=0, speed=1.0)
        tts.generate("你好。", sid=0, speed=1.0)
        print("Warmup done.\n")

        # === Batch mode (no callback) ===
        print(f"{'Text':<44} {'TTFT':>8} {'Audio':>7} {'RTF':>6}")
        print("-" * 70)

        for text in TEST_TEXTS:
            # Run twice, take second
            measure_ttft(tts, text)
            elapsed, dur, n_samples = measure_ttft(tts, text)
            short = text[:42] + ".." if len(text) > 44 else text
            print(f"{short:<44} {elapsed*1000:>7.0f}ms {dur:>6.2f}s {elapsed/dur:>6.3f}")

        # === Streaming mode (sentence callback) ===
        print(f"\n--- Streaming (sentence callback) ---")
        print(f"{'Text':<44} {'TTFT':>8} {'1stChunk':>10} {'Total':>8} {'Audio':>7} {'Chunks':>6}")
        print("-" * 90)

        for text in TEST_TEXTS:
            measure_ttft_streaming(tts, text)  # warmup
            r = measure_ttft_streaming(tts, text)
            short = text[:42] + ".." if len(text) > 44 else text
            print(f"{short:<44} {r['ttft']*1000:>7.0f}ms {r['first_chunk_dur']:>9.2f}s "
                  f"{r['total_time']*1000:>7.0f}ms {r['total_dur']:>6.2f}s {r['n_chunks']:>5}")

        del tts

    print(f"\n{'='*80}")
    print("Comparison (TTFT for shortest text '好的。'):")
    print(f"{'='*80}")
    print("  Kokoro:      ~130ms")
    print("  F5-TTS:      ~2500ms (NFE=8)")
    print("  CosyVoice3:  ~3800ms")
    print("  Matcha:      see above")


if __name__ == "__main__":
    main()
