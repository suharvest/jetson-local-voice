#!/usr/bin/env python3
"""Test WAV files with streaming ASR WebSocket."""
import asyncio, json, sys
import numpy as np
import soundfile as sf

async def test_asr(wav_path):
    import websockets
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Simple resample to 16kHz
    if sr != 16000:
        ratio = 16000 / sr
        new_len = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, new_len)
        data = np.interp(indices, np.arange(len(data)), data)
        sr = 16000

    audio_int16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
    print(f"  Audio: {len(audio_int16)} samples, {len(audio_int16)/sr:.2f}s")

    uri = "ws://localhost:8000/asr/stream?sample_rate=16000"
    async with websockets.connect(uri) as ws:
        chunk_size = 3200
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i+chunk_size]
            await ws.send(chunk.tobytes())
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.05)
                r = json.loads(msg)
                if r.get("type") == "partial":
                    t = r.get("text", "")
                    if t:
                        print(f"  partial: {t}")
            except asyncio.TimeoutError:
                pass

        await ws.send(b"")
        # Consume messages until we get type=final (skip reset/empty messages)
        result = ""
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=10)
            r = json.loads(msg)
            t = r.get("type")
            if t == "final":
                result = r.get("text", "")
                print(f"  FINAL: {result}")
                return result
            # type=reset or missing type → skip, keep waiting

wavs = sys.argv[1:] if len(sys.argv) > 1 else ["/tmp/ref_output.wav", "/tmp/test_en2.wav"]
for wav in wavs:
    print(f"\n=== {wav} ===")
    try:
        asyncio.run(test_asr(wav))
    except Exception as e:
        print(f"  ERROR: {e}")
