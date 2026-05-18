# OpenVoiceStream Examples

Small clients for validating and integrating the public streaming APIs.

## Stream TTS to WAV

No third-party Python packages required. The script calls `/tts/stream`, reads
the 4-byte sample-rate prefix, and writes a playable WAV.

```bash
python3 examples/stream_tts_to_wav.py \
  --url http://device:8621 \
  --text "你好，欢迎使用 OpenVoiceStream。" \
  --out /tmp/ovs-tts.wav
```

Use `http://device:8621` for the default compose deployment on every device.

## V2V WebSocket TTS-Only

Requires `websockets`. This demonstrates the unified `/v2v/stream` protocol in
TTS-only mode by sending text chunks and collecting returned PCM.

```bash
uv run --with websockets python examples/v2v_tts_only.py \
  --url ws://device:8621/v2v/stream \
  --text "Hello from a streaming client." \
  --out /tmp/ovs-v2v-tts.wav
```

Use this as the smallest copy-paste starting point for feeding LLM tokens into
OpenVoiceStream TTS.
