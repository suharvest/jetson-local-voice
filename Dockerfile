# Jetson Voice Assistant — all-in-one image (ASR + TTS + models baked in)
# Matcha TTS + Paraformer streaming ASR, CUDA accelerated
# Base: dustynv onnxruntime with CUDAExecutionProvider (JP6.x, CUDA 12.6)
FROM dustynv/onnxruntime:1.20-r36.4.0

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2 libsndfile1 sox patchelf \
    && rm -rf /var/lib/apt/lists/*

# FastAPI stack (use PyPI explicitly; base image may set a custom index)
RUN pip3 install --no-cache-dir --index-url https://pypi.org/simple \
    fastapi \
    "uvicorn[standard]" \
    soundfile \
    python-multipart \
    numpy

# sherpa-onnx CUDA wheel for aarch64
# The +cuda wheel bundles onnxruntime 1.11.0 (CUDA 10/11).
# We patch it below to use the system's onnxruntime 1.20.0 (CUDA 12.6).
ARG SHERPA_ONNX_VERSION=1.12.28
ARG SHERPA_WHEEL_URL=https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cuda/${SHERPA_ONNX_VERSION}/sherpa_onnx-${SHERPA_ONNX_VERSION}+cuda-cp310-cp310-linux_aarch64.whl
RUN pip3 install --no-cache-dir --index-url https://pypi.org/simple "${SHERPA_WHEEL_URL}"

# Patch sherpa-onnx to use system onnxruntime with CUDA 12.6
COPY scripts/patch_sherpa_ort.py /tmp/
RUN python3 /tmp/patch_sherpa_ort.py && rm /tmp/patch_sherpa_ort.py

# Install patched sherpa-onnx .so files (Paraformer streaming EOF fix)
# Pre-built from v1.12.28 source with IsReady/DecodeStream/CIF patches.
# See patches/README.md for details; to rebuild, see patches/paraformer-eof-fix.patch
COPY patches/sherpa-onnx-lib/ /tmp/sherpa-onnx-lib/
RUN SITE_LIB=$(python3 -c "import sherpa_onnx, os; print(os.path.join(os.path.dirname(sherpa_onnx.__file__), 'lib'))") && \
    cp /tmp/sherpa-onnx-lib/*.so "$SITE_LIB/" && \
    rm -rf /tmp/sherpa-onnx-lib && \
    python3 -c "import sherpa_onnx; print(f'sherpa_onnx {sherpa_onnx.__version__} patched OK')"

# Clean up build-only tools
RUN apt-get purge -y patchelf && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

# Application code
COPY app/ /opt/speech/app/

# Bake in models — separate layers to stay under registry upload limits
ARG CDN=https://sensecraft-statics.seeed.cc/solution-app/jetson-voice
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /opt/models && \
    wget -qO- "${CDN}/models-matcha.tar.gz" | tar xzf - -C /opt/models && \
    apt-get purge -y wget && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/*

RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/* && \
    wget -qO- "${CDN}/models-paraformer.tar.gz" | tar xzf - -C /opt/models && \
    apt-get purge -y wget && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/*

ENV MODEL_DIR=/opt/models \
    NVIDIA_VISIBLE_DEVICES=all \
    CUDA_MODULE_LOADING=LAZY \
    ORT_CUDA_ARENA_EXTEND_STRATEGY=kSameAsRequested

WORKDIR /opt/speech/app
EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
