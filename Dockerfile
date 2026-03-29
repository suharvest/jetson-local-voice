# Jetson Voice — GPU-accelerated speech service
# Models are NOT baked in — downloaded on first start based on LANGUAGE_MODE.
# zh_en (default): Matcha TTS + Paraformer ASR (~930 MB)
# en:              Kokoro TTS + Zipformer ASR  (~590 MB)
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

# sherpa-onnx for aarch64 (installed from PyPI)
# The +cuda wheel (if available) bundles onnxruntime 1.11.0 (CUDA 10/11).
# We patch it below to use the system's onnxruntime 1.20.0 (CUDA 12.6).
RUN pip3 install --no-cache-dir --index-url https://pypi.org/simple 'sherpa-onnx==1.12.28'

# Patch sherpa-onnx to use system onnxruntime with CUDA 12.6
COPY scripts/patch_sherpa_ort.py /tmp/
RUN python3 /tmp/patch_sherpa_ort.py && rm /tmp/patch_sherpa_ort.py

# Install patched sherpa-onnx .so files (Paraformer streaming EOF fix)
# Pre-built from v1.12.28 source with IsReady/DecodeStream/CIF patches.
# See patches/README.md for details; to rebuild, see patches/paraformer-eof-fix.patch
COPY patches/sherpa-onnx-lib/ /tmp/sherpa-onnx-lib/
RUN SITE_LIB=/usr/local/lib/python3.10/dist-packages/sherpa_onnx/lib && \
    cp /tmp/sherpa-onnx-lib/*.so "$SITE_LIB/" && \
    rm -rf /tmp/sherpa-onnx-lib && \
    python3 -c "import sherpa_onnx; print(f'sherpa_onnx {sherpa_onnx.__version__} patched OK')"

# Clean up build-only tools
RUN apt-get purge -y patchelf && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

# Application code
COPY app/ /opt/speech/app/

# Custom voice embeddings (patched into voices.bin on first start)
COPY voices/ /opt/speech/voices/

# No models baked in — model_downloader.py auto-downloads on first start
# based on LANGUAGE_MODE env var. Models cached in /opt/models volume.
RUN mkdir -p /opt/models

ENV MODEL_DIR=/opt/models \
    NVIDIA_VISIBLE_DEVICES=all \
    CUDA_MODULE_LOADING=LAZY \
    ORT_CUDA_ARENA_EXTEND_STRATEGY=kSameAsRequested

WORKDIR /opt/speech/app
EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
