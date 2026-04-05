# RK3576 Qwen3-TTS Docker Container
#
# Build on cat-remote (aarch64):
#   docker build --network=host -t rk3576-tts -f Dockerfile .
#
# Run (--network=host required because RK3576 kernel lacks iptable_raw):
#   docker run --privileged --network=host \
#     -v /dev:/dev \
#     -v /home/cat/qwen3-tts-rknn:/opt/tts/models:ro \
#     -v /home/cat/models/talker_fullvocab_fixed_w4a16_rk3576.rkllm:/opt/tts/models/talker_fullvocab_fixed_w4a16_rk3576.rkllm:ro \
#     rk3576-tts
#
# Docker daemon.json must have:
#   {"iptables": false, "registry-mirrors": ["https://docker.1ms.run"]}
# This is needed because the RK3576 kernel 6.1.99 lacks iptable_raw module.

FROM debian:12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Python deps (install in stages for better caching)
RUN pip install --no-cache-dir \
    'numpy<2' \
    soundfile

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

# RKLLM runtime library
COPY librkllmrt.so /usr/lib/librkllmrt.so
RUN ldconfig

# Application code
COPY app/ /opt/tts/app/

# Model directory structure:
# /opt/tts/models/
#   talker_fullvocab_fixed_w4a16_rk3576.rkllm
#   text_project.rknn
#   codec_embed.rknn
#   code_predictor.rknn
#   code_predictor_embed.rknn
#   decoder_ctx25_int8.rknn
#   codec_head_weight.npy
#   codebook_embeds/
#     codebook_{0..15}.npy
#     output_proj_first_weight.npy
#     output_proj_rest_weight.npy
#   tokenizer/
#     tokenizer.json, vocab.json, merges.txt, etc.
RUN mkdir -p /opt/tts/models

ENV MODEL_DIR=/opt/tts/models
WORKDIR /opt/tts/app
EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
