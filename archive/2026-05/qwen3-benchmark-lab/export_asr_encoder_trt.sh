#!/bin/bash
# Build Qwen3-ASR encoder TRT engine.
# Usage:
#   ./benchmark/export_asr_encoder_trt.sh               # single profile (default)
#   DUAL_PROFILE=1 ./benchmark/export_asr_encoder_trt.sh # dual profile (streaming+offline)
#
# Run on Jetson with TensorRT 10.3. Prod must be stopped first (RAM contention).

set -euo pipefail

ONNX_PATH="${ONNX_PATH:-/home/harvest/qwen3-asr-v2/qwen3-asr-v2/encoder_fp16.onnx}"
OUT_DIR="${OUT_DIR:-/home/harvest/qwen3-asr-v2/qwen3-asr-v2}"

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

if [ "${DUAL_PROFILE:-0}" = "1" ]; then
    ENGINE_NAME="${ENGINE_NAME:-asr_encoder_fp16_dual.engine}"
    echo "Profile:    dual (streaming 40-200 + offline 200-3000)"
    MIN_SHAPES="mel:1x128x40,mel:1x128x200"
    OPT_SHAPES="mel:1x128x100,mel:1x128x1000"
    MAX_SHAPES="mel:1x128x200,mel:1x128x3000"
else
    ENGINE_NAME="${ENGINE_NAME:-asr_encoder_fp16.engine}"
    echo "Profile:    single (40-3000, opt=200)"
    MIN_SHAPES="mel:1x128x40"
    OPT_SHAPES="mel:1x128x200"
    MAX_SHAPES="mel:1x128x3000"
fi

OUT_ENGINE="${OUT_DIR}/${ENGINE_NAME}"

echo "ONNX:       $ONNX_PATH"
echo "OUT:        $OUT_ENGINE"
echo "MIN:        $MIN_SHAPES"
echo "OPT:        $OPT_SHAPES"
echo "MAX:        $MAX_SHAPES"
echo

"$TRTEXEC" \
    --onnx="$ONNX_PATH" \
    --saveEngine="$OUT_ENGINE" \
    --bf16 \
    --memPoolSize=workspace:2048MiB \
    --minShapes="$MIN_SHAPES" \
    --optShapes="$OPT_SHAPES" \
    --maxShapes="$MAX_SHAPES" \
    2>&1 | tail -50

echo
echo "=== Result ==="
ls -lh "$OUT_ENGINE"
md5sum "$OUT_ENGINE"
