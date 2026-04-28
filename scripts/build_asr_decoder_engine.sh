#!/bin/bash
# Build Qwen3-ASR decoder BF16 TRT engine. TensorRT 10.3 / Jetson.
#
# Per-device profile (override via env):
#   Nano 8GB  (default): MAX_SEQ=200 WS=256
#   NX 16GB           :  MAX_SEQ=500 WS=2048
#   AGX 32GB          :  MAX_SEQ=1000 WS=4096
#
# ASR sequences in practice rarely exceed 50 tokens; even MAX_SEQ=200 has plenty
# of headroom. Cross-device cubin compatibility verified for Ampere SM 8.7.

set -euo pipefail

ONNX_PATH="${ONNX_PATH:-/home/harvest/qwen3-asr-v2/decoder_step.onnx}"
MAX_SEQ="${MAX_SEQ:-200}"
OPT_SEQ="${OPT_SEQ:-30}"
WS="${WS:-256}"
OUT_DIR="${OUT_DIR:-/home/harvest/qwen3-asr-v2}"
ENGINE_NAME="${ENGINE_NAME:-asr_decoder_bf16_max${MAX_SEQ}.engine}"

KV_LAYERS=28

# Build per-layer past_key_N / past_value_N shape strings
build_shapes() {
    local seq=$1
    local parts=("input_embeds:1x1x1024" "position_ids:1x1")
    for i in $(seq 0 $((KV_LAYERS-1))); do
        parts+=("past_key_${i}:1x8x${seq}x128" "past_value_${i}:1x8x${seq}x128")
    done
    local IFS=','
    echo "${parts[*]}"
}

MIN_SHAPES=$(build_shapes 0)
OPT_SHAPES=$(build_shapes "$OPT_SEQ")
MAX_SHAPES=$(build_shapes "$MAX_SEQ")

OUT_ENGINE="${OUT_DIR}/${ENGINE_NAME}"

echo "ONNX:       $ONNX_PATH"
echo "OUT:        $OUT_ENGINE"
echo "MIN seq:    0"
echo "OPT seq:    $OPT_SEQ"
echo "MAX seq:    $MAX_SEQ"
echo "Workspace:  ${WS}MiB"
echo

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

"$TRTEXEC" \
    --onnx="$ONNX_PATH" \
    --saveEngine="$OUT_ENGINE" \
    --bf16 \
    --memPoolSize=workspace:"${WS}MiB" \
    --minShapes="$MIN_SHAPES" \
    --optShapes="$OPT_SHAPES" \
    --maxShapes="$MAX_SHAPES" \
    2>&1 | tail -30

echo
ls -lh "$OUT_ENGINE"
md5sum "$OUT_ENGINE"
