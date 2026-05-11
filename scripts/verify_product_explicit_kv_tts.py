from __future__ import annotations

import argparse
import json
import os
import wave

from backends.trt_edge_llm_tts import TRTEdgeLLMTTSBackend


def _write_pcm16_wav(path: str, pcm: bytes, sample_rate: int = 24000) -> None:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Seeed Local Voice product explicit-KV Qwen3-TTS path.")
    parser.add_argument("--text", default="语音合成的稳定性。")
    parser.add_argument("--output", default="/tmp/jetson_voice_product_explicit_kv.wav")
    parser.add_argument("--max-audio-length", type=int, default=160)
    parser.add_argument("--segment-max-chars", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-segment", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    os.environ["SEEED_LOCAL_VOICE_TTS_BACKEND"] = "product_explicit_kv"
    os.environ["SEEED_LOCAL_VOICE_TTS_SEED"] = str(args.seed)
    backend = TRTEdgeLLMTTSBackend()
    backend.preload()
    if args.streaming:
        product_backend = getattr(backend, "_product_backend", None)
        if product_backend is None or not hasattr(product_backend, "generate_streaming"):
            raise RuntimeError("product_explicit_kv backend does not expose generate_streaming")
        chunks = list(
            product_backend.generate_streaming(
                args.text,
                max_frames=args.max_audio_length,
                seed=args.seed,
            )
        )
        pcm = b"".join(chunks)
        _write_pcm16_wav(args.output, pcm)
        print(
            json.dumps(
                {
                    "output": args.output,
                    "bytes": os.path.getsize(args.output),
                    "chunks": len(chunks),
                    "pcm_bytes": len(pcm),
                    "seed": args.seed,
                },
                ensure_ascii=False,
            )
        )
        return

    wav, meta = backend.synthesize(
        args.text,
        max_audio_length=args.max_audio_length,
        segment_text=not args.no_segment,
        product_segment_text=not args.no_segment,
        segment_max_chars=args.segment_max_chars,
        seed=args.seed,
    )
    with open(args.output, "wb") as f:
        f.write(wav)
    print(json.dumps({"output": args.output, "bytes": len(wav), "meta": meta}, ensure_ascii=False))


if __name__ == "__main__":
    main()
