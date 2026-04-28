#!/bin/bash
# Build Qwen3-TTS vocoder FP16 TRT engine. Run on Jetson with TensorRT 10.3.
#
# Per-device profile (override via env):
#   Nano 8GB  (default): WS=256
#   NX 16GB           :  WS=2048
#   AGX 32GB          :  WS=4096
#
# Cross-device cubin compatibility verified for Ampere SM 8.7. For best
# tactic tuning, build on the target device with its WS profile. A
# Nano-built engine works on NX/AGX but uses conservative tactics.
#
# Vocoder has 1 dynamic input:
#   audio_codes [1, T, 16] int64  (T = n_frames, 1..200)
# Output: audio_values [1, 192000], lengths [1]

set -euo pipefail

ONNX="${ONNX:-/home/harvest/voice_test/models/qwen3-tts/onnx/vocoder_fp16.onnx}"
OUT_DIR="${OUT_DIR:-/home/harvest/voice_test/models/qwen3-tts/engines}"
ENGINE_NAME="${ENGINE_NAME:-vocoder_fp16.engine}"
WS="${WS:-256}"  # workspace in MiB

OUT_ENGINE="${OUT_DIR}/${ENGINE_NAME}"

echo "ONNX:        $ONNX"
echo "OUT:         $OUT_ENGINE"
echo "Workspace:   ${WS}MiB"
echo "min:         audio_codes:1x1x16"
echo "opt:         audio_codes:1x100x16"
echo "max:         audio_codes:1x200x16"
echo

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

"$TRTEXEC" \
    --onnx="$ONNX" \
    --saveEngine="$OUT_ENGINE" \
    --fp16 \
    --memPoolSize=workspace:"${WS}MiB" \
    --minShapes=audio_codes:1x1x16 \
    --optShapes=audio_codes:1x100x16 \
    --maxShapes=audio_codes:1x200x16 \
    2>&1 | tail -20

echo
ls -lh "$OUT_ENGINE"
md5sum "$OUT_ENGINE"
