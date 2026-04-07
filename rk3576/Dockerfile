# RK3576 Speech Service Docker Container (TTS + ASR)
#
# Build on cat-remote (aarch64):
#   docker build --network=host -t rk3576-speech -f Dockerfile .
#
# Run (--network=host required because RK3576 kernel lacks iptable_raw):
#   docker run --privileged --network=host \
#     -v /dev:/dev \
#     -v /home/cat/qwen3-tts-rknn:/opt/tts/models:ro \
#     -v /home/cat/models/talker_fullvocab_fixed_w4a16_rk3576.rkllm:/opt/tts/models/talker_fullvocab_fixed_w4a16_rk3576.rkllm:ro \
#     -v /home/cat/qwen3-asr-models:/opt/asr/models:ro \
#     -e ASR_BACKEND=qwen3_asr_rk \
#     rk3576-speech

FROM debian:12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# System deps (gcc for C engine compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libsndfile1 \
    gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Python venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Python deps (install in stages for better caching)
RUN pip install --no-cache-dir \
    'numpy<2' \
    soundfile \
    librosa

RUN pip install --no-cache-dir \
    fastapi \
    'uvicorn[standard]' \
    python-multipart

RUN pip install --no-cache-dir \
    transformers \
    tokenizers

# RKNN toolkit lite2 (aarch64 wheel from PyPI)
RUN pip install --no-cache-dir \
    rknn-toolkit-lite2

# RKLLM + RKNN runtime libraries (copied from host /usr/lib/)
COPY librkllmrt.so /usr/lib/librkllmrt.so
COPY librknnrt.so /usr/lib/librknnrt.so
RUN ldconfig

RUN mkdir -p /opt/tts/models /opt/tts/lib /opt/asr/models

# C engine: compile cp_engine (W4A16 mode for ~3x speedup)
# Non-fatal: tts_service.py currently uses 15x RKNN calls, not the C engine.
# Will be enabled in a future update.
COPY engine/ /opt/tts/engine/
RUN cd /opt/tts/engine && \
    gcc -O2 -fPIC -Wall -Wno-unused-result -march=armv8-a+fp16 \
    -DCP_USE_W4A16 \
    -I/opt/tts/engine/include \
    -shared -o /opt/tts/lib/libcp_engine.so \
    cp_engine.c \
    -lrknnrt -lm && \
    ldconfig || echo "WARNING: cp_engine.so build failed (non-fatal, C engine not used yet)"

# Application code
COPY app/ /opt/tts/app/

ENV MODEL_DIR=/opt/tts/models
ENV TTS_BACKEND=qwen3_rknn
ENV ASR_MODEL_DIR=/opt/asr/models
# ASR_BACKEND is intentionally unset here — enable at runtime via env var
# e.g. -e ASR_BACKEND=qwen3_asr_rk
ENV LD_LIBRARY_PATH=/opt/tts/lib:/usr/lib
WORKDIR /opt/tts/app
EXPOSE 8621

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8621"]
