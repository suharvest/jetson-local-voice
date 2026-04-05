#!/bin/bash
# Build ASR decoder BF16 TRT engine inside the container
# Usage: docker exec asr_debug bash /host_tmp/build_asr_bf16_engine.sh

set -e

MODEL_DIR=/opt/models/qwen3-asr-v2
ONNX=${MODEL_DIR}/decoder_step.onnx
OUT=${MODEL_DIR}/asr_decoder_bf16.engine
TRTEXEC=/trtexec_dir/trtexec

# Profile shapes:
#   input_embeds: dynamic seq_len (1..500) for unified prefill+decode
#   position_ids: dynamic seq_len (1..500)
#   past_key/value: dynamic past_len (0..500), min=0 for prefill with empty KV
MIN_SHAPES="input_embeds:1x1x1024,position_ids:1x1"
OPT_SHAPES="input_embeds:1x1x1024,position_ids:1x1"
MAX_SHAPES="input_embeds:1x500x1024,position_ids:1x500"

for i in $(seq 0 27); do
    MIN_SHAPES="${MIN_SHAPES},past_key_${i}:1x8x0x128,past_value_${i}:1x8x0x128"
    OPT_SHAPES="${OPT_SHAPES},past_key_${i}:1x8x100x128,past_value_${i}:1x8x100x128"
    MAX_SHAPES="${MAX_SHAPES},past_key_${i}:1x8x500x128,past_value_${i}:1x8x500x128"
done

echo "Building BF16 engine..."
echo "  ONNX: ${ONNX}"
echo "  Output: ${OUT}"

$TRTEXEC \
    --onnx=${ONNX} \
    --saveEngine=${OUT} \
    --bf16 \
    --minShapes="${MIN_SHAPES}" \
    --optShapes="${OPT_SHAPES}" \
    --maxShapes="${MAX_SHAPES}" \
    --memPoolSize=workspace:2048MiB \
    2>&1 | tail -30

echo ""
echo "Engine built: $(ls -lh ${OUT})"
