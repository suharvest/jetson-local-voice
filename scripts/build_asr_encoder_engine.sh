#!/bin/bash
# Build Qwen3-ASR encoder FP16 TRT engine. TensorRT 10.3 / Jetson.
#
# Per-device profile (override via env):
#   Nano 8GB  (default): WS=256
#   NX 16GB           :  WS=2048
#   AGX 32GB          :  WS=4096
#
# Cross-device cubin compatibility verified for Ampere SM 8.7. For best
# tactic tuning, build on the target device with its WS profile.
#
# Encoder ONNX needs ONNX surgery first — the original encoder.onnx contains
# an `If` node whose two branches output mismatched shapes ([128,T] vs
# [1,128,T]) that TensorRT rejects. Surgery replaces the If with Squeeze
# (mathematically equivalent for batch=1 ASR). Run `surgery_encoder_onnx.py`
# to produce the surgeried .onnx if missing (see comments in that script).
#
# Input mel: [1, 128, T]  T=40..3000  (16kHz audio, 10ms hop, 3000 frames=30s)
# Output audio_features: [1, T_pooled, 1024]

set -euo pipefail

ONNX_PATH="${ONNX_PATH:-/home/harvest/qwen3-asr-v2/encoder_fp16_trt.onnx}"
OUT_DIR="${OUT_DIR:-/home/harvest/qwen3-asr-v2}"
ENGINE_NAME="${ENGINE_NAME:-asr_encoder_fp16.engine}"
WS="${WS:-256}"
MIN_T="${MIN_T:-40}"
OPT_T="${OPT_T:-200}"
MAX_T="${MAX_T:-3000}"

OUT_ENGINE="${OUT_DIR}/${ENGINE_NAME}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

if [ ! -f "$ONNX_PATH" ]; then
    echo "ERROR: ONNX not found: $ONNX_PATH"
    echo "       Run scripts/surgery_encoder_onnx.py to produce surgeried ONNX first."
    exit 1
fi

echo "ONNX:        $ONNX_PATH"
echo "OUT:         $OUT_ENGINE"
echo "Workspace:   ${WS}MiB"
echo "Mel T range: ${MIN_T}..${MAX_T} (opt=${OPT_T})"
echo

"$TRTEXEC" \
    --onnx="$ONNX_PATH" \
    --saveEngine="$OUT_ENGINE" \
    --fp16 \
    --memPoolSize=workspace:"${WS}MiB" \
    --minShapes=mel:1x128x"${MIN_T}" \
    --optShapes=mel:1x128x"${OPT_T}" \
    --maxShapes=mel:1x128x"${MAX_T}" \
    2>&1 | tail -25

echo
ls -lh "$OUT_ENGINE"
md5sum "$OUT_ENGINE"
