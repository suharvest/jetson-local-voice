# EdgeLLM Voice Workers

This directory owns the Seeed Local Voice product worker protocol for TensorRT-Edge-LLM based ASR/TTS.

TensorRT-Edge-LLM remains the inference framework dependency. The workers here are product glue:

- resident process lifecycle
- stdin/stdout JSONL request protocol
- streaming TTS chunk policy
- PCM/base64/file transport
- Seeed Local Voice error and metric fields

The intended layering is:

```text
Seeed Local Voice worker
  -> references our EdgeLLM baseline build
     -> tracks NVIDIA EdgeLLM upstream plus minimal local patches
```

Do not put Seeed Local Voice product protocol into the EdgeLLM baseline. If Qwen3-TTS needs a local framework patch to run correctly on Jetson, keep the patch in the EdgeLLM baseline and document whether it is an upstream PR candidate or an NVIDIA issue reference.

Build the EdgeLLM baseline first, then build the workers from the Seeed Local Voice repo:

```bash
CUDACXX=/usr/local/cuda-12.6/bin/nvcc cmake \
  -S native/edgellm_voice_worker -B build/edgellm_voice_worker \
  -DEDGE_LLM_SOURCE_DIR=/Users/harvest/project/tensorrt-edge-llm \
  -DEDGE_LLM_BUILD_DIR=/Users/harvest/project/tensorrt-edge-llm/build_sm87 \
  -DTRT_PACKAGE_DIR=/usr \
  -DCUDA_DIR=/usr/local/cuda-12.6 \
  -DCUDA_CTK_VERSION=12.6
cmake --build build/edgellm_voice_worker --target qwen3_tts_worker qwen3_asr_worker -j2
```

The binaries are written to:

```text
build/edgellm_voice_worker/workers/
```

`app/backends/trt_edge_llm_ipc.py` prefers those binaries and falls back to the old EdgeLLM example paths for older deployments.

## Product Boundary

The workers may adapt EdgeLLM runtime behavior into product semantics:

- stateful resident sessions
- stable request/response schema
- streaming chunk and final `done` events
- product sampling and text segmentation policy
- model manifest based engine selection
- explicit-KV cache session policy for product-owned Qwen3-TTS backends

The EdgeLLM baseline should only expose generic lower-level capabilities, for example a codec-frame callback or a corrected Qwen3-TTS backend. If NVIDIA later provides an official streaming or precision-correct Qwen3-TTS API, the worker should switch adapters without changing the product protocol.

## Product Explicit-KV Verification

Use the explicit-KV backend only through the product backend selector:

```bash
PYTHONPATH=/home/harvest/project/jetson-voice/app:/home/harvest/voice_test/app_overlay \
SEEED_LOCAL_VOICE_TTS_BACKEND=product_explicit_kv \
SEEED_LOCAL_VOICE_TTS_MODEL_BASE=/home/harvest/voice_test/models/qwen3-tts \
SEEED_LOCAL_VOICE_TTS_NATIVE_MODULE_DIR=/home/harvest/voice_test/app_overlay \
python3 scripts/verify_product_explicit_kv_tts.py \
  --text "语音合成的稳定性。" \
  --output /tmp/jetson_voice_product_explicit_kv.wav
```

The older `EDGE_LLM_TTS_NATIVE_FALLBACK` switch is intentionally not part of the product path.
