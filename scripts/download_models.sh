#!/usr/bin/env bash
# Download SenseVoice ASR + Kokoro TTS models (idempotent).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/opt/models}"
mkdir -p "$MODEL_DIR"

# ── SenseVoice (ASR) ────────────────────────────────────────────────────
SV_DIR="$MODEL_DIR/sensevoice"
SV_NAME="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
if [ ! -f "$SV_DIR/$SV_NAME/model.int8.onnx" ]; then
    echo "[models] Downloading SenseVoice ASR model..."
    mkdir -p "$SV_DIR"
    cd "$SV_DIR"
    wget -q --show-progress \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${SV_NAME}.tar.bz2" \
        -O model.tar.bz2
    tar xjf model.tar.bz2
    rm model.tar.bz2
    echo "[models] SenseVoice ready."
else
    echo "[models] SenseVoice already present."
fi

# ── Kokoro (TTS) ────────────────────────────────────────────────────────
KOKORO_NAME="kokoro-multi-lang-v1_1"
KOKORO_DIR="$MODEL_DIR/$KOKORO_NAME"
if [ ! -f "$KOKORO_DIR/model.onnx" ]; then
    echo "[models] Downloading Kokoro TTS model..."
    cd "$MODEL_DIR"
    wget -q --show-progress \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/${KOKORO_NAME}.tar.bz2" \
        -O "${KOKORO_NAME}.tar.bz2"
    tar xjf "${KOKORO_NAME}.tar.bz2"
    rm "${KOKORO_NAME}.tar.bz2"
    echo "[models] Kokoro TTS ready."
else
    echo "[models] Kokoro TTS already present."
fi

echo "[models] All models ready."
