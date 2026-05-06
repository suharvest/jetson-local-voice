# EdgeLLM Voice Workers

This directory owns the Jetson Voice product worker protocol for TensorRT-Edge-LLM based ASR/TTS.

TensorRT-Edge-LLM remains the inference framework dependency. The workers here are product glue:

- resident process lifecycle
- stdin/stdout JSONL request protocol
- streaming TTS chunk policy
- PCM/base64/file transport
- Jetson Voice error and metric fields

Build TensorRT-Edge-LLM first, then build the workers from the Jetson Voice repo:

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
