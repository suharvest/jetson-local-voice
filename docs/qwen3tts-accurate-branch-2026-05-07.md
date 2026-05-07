# Qwen3-TTS Accurate Branch 2026-05-07

This branch freezes the Qwen3-TTS runtime state that generated correct Chinese,
English, and Japanese Seeed Studio conversational AI solution samples on Orin NX.

## Branches

- Product repo: `qwen3tts-accurate-20260507`
- EdgeLLM/HLM repo: `stable/qwen3tts-direct-bf16-nx-20260507`
- EdgeLLM/HLM commit: `b9c57c8 Freeze Qwen3-TTS direct BF16 reference runtime`

The EdgeLLM/HLM branch was recovered from:

```text
orin-nano:/tmp/tensorrt-edge-llm-upstream-runtime-0505
```

and imported locally into:

```text
/Users/harvest/project/tensorrt-edge-llm
```

from bundle:

```text
/private/tmp/qwen3tts-direct-bf16-nx-20260507.bundle
```

## Runtime Used For Validation

The runtime copied to Orin NX is:

```text
/tmp/qwen3tts_ref_0507_from_nano
```

Key paths:

```text
build/examples/omni/qwen3_tts_inference
build/libNvInfer_edgellm_plugin.so
talker/
cp_product_unified_0506/
cp_bf16_historical_0506/
code2wav/
/home/harvest/voice_test/models/qwen3-tts/engines/talker_decode_bf16.engine
```

Required environment:

```text
QWEN3_TTS_SEED=42
QWEN3_TTS_DIRECT_TALKER_ENGINE=/home/harvest/voice_test/models/qwen3-tts/engines/talker_decode_bf16.engine
QWEN3_TTS_HOST_TEXT_PROJECTION=1
EDGELLM_PLUGIN_PATH=/tmp/qwen3tts_ref_0507_from_nano/build/libNvInfer_edgellm_plugin.so
```

## Validation Outputs

Generated on `orin-nx` and copied back locally:

```text
/private/tmp/sherpa-tts/out/zh_qwen3tts_seeed_conversational_ai.wav
/private/tmp/sherpa-tts/out/en_qwen3tts_seeed_conversational_ai.wav
/private/tmp/sherpa-tts/out/ja_qwen3tts_seeed_conversational_ai.wav
```

All three files are 24 kHz, mono WAV files.

The earlier short Chinese reference was also ASR-verified on Orin NX with
Paraformer streaming ASR:

```text
Input:  语音合成的稳定性。
ASR:    语音合成的稳定性
```

## What This Freezes

This freezes a product-quality accurate path, not an upstream-ready minimal PR.

The key correctness constraints are:

- explicit-KV Qwen3-TTS Talker override
- BF16 Talker decode engine
- host FP32 text projection
- BF16 CodePredictor engine and weights
- complete tokenizer export directory, including `tokenizer.json`,
  `tokenizer_config.json`, and `processed_chat_template.json`
- Code2Wav engine copied from the previously validated Nano runtime

The corresponding upstream issue should describe this as a precision/runtime
semantics gap in the official Qwen3-TTS path, not as a product protocol request.
