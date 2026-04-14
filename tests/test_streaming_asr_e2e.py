"""End-to-end streaming ASR test — requires Qwen3-ASR models on device.

Run manually on Jetson:
  cd /path/to/jetson-voice
  python3 tests/test_streaming_asr_e2e.py [path/to/test.wav]
"""
import sys
import time
sys.path.insert(0, "app")

import numpy as np


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else None

    from backends.qwen3_asr import Qwen3ASRBackend

    print("Loading Qwen3-ASR backend...")
    backend = Qwen3ASRBackend()
    backend.preload()

    if not backend._encoder:
        print("ERROR: ORT encoder not loaded — streaming requires Python path")
        sys.exit(1)

    # Load or generate test audio
    if wav_path:
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    else:
        print("No WAV provided, generating 3s of silence as smoke test")
        audio = np.zeros(48000, dtype=np.float32)
        sr = 16000

    # Create streaming session
    stream = backend.create_stream(language="auto")
    print(f"Stream type: {type(stream).__name__}")

    # Feed audio in 0.5s chunks (simulating real-time microphone)
    chunk_size = int(sr * 0.5)
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        t0 = time.perf_counter()
        stream.accept_waveform(sr, chunk)
        elapsed = (time.perf_counter() - t0) * 1000

        text, is_endpoint = stream.get_partial()
        pos = (i + chunk_size) / sr
        print(f"  [{pos:.1f}s] ({elapsed:.0f}ms) partial='{text}' endpoint={is_endpoint}")

        if is_endpoint:
            print(f"  >>> ENDPOINT: '{text}'")

    # Finalize
    t0 = time.perf_counter()
    final = stream.finalize()
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\nFinal ({elapsed:.0f}ms): '{final}'")


if __name__ == "__main__":
    main()
