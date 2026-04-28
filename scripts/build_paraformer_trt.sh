#!/usr/bin/env bash
# Build Paraformer TRT engines on Orin NX.
# Run inside the v3.4-slim container or on the host with trtexec.
#
# Usage:
#   bash scripts/build_paraformer_trt.sh [--container NAME] [--model-dir PATH]
#
# Options:
#   --container NAME   Stop v3.4-slim container during build (default: reachy_speech-speech-1)
#   --model-dir PATH   Paraformer model directory (default: /opt/models/paraformer-streaming)
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/opt/models/paraformer-streaming}"
ENGINE_DIR="${MODEL_DIR}/engines"
CONTAINER="${CONTAINER:-reachy_speech-speech-1}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --container) CONTAINER="$2"; shift 2 ;;
        --model-dir) MODEL_DIR="$2"; ENGINE_DIR="${MODEL_DIR}/engines"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p "${ENGINE_DIR}"

echo "=== Paraformer TRT Engine Build ==="
echo "Model dir:  ${MODEL_DIR}"
echo "Engine dir: ${ENGINE_DIR}"
echo "Container:  ${CONTAINER}"
echo ""

# ── Stop v3.4-slim container to free GPU memory ──
echo "=== Stopping container: ${CONTAINER} ==="
docker stop "${CONTAINER}" 2>/dev/null || true
echo ""

# ── Build encoder (dual-profile) ──
# Profile 1 (streaming chunk): min=40, opt=40, max=80
# Profile 2 (offline):         min=80, opt=200, max=400
ENC_PLAN="${ENGINE_DIR}/paraformer_encoder_dp4_400.plan"
echo "=== Building encoder engine ==="
echo "  → ${ENC_PLAN}"
trtexec --onnx="${MODEL_DIR}/encoder.onnx" \
  --minShapes=speech:1x40x560,speech_lengths:1 \
  --optShapes=speech:1x40x560,speech_lengths:1 \
  --maxShapes=speech:1x80x560,speech_lengths:1 \
  --optShapes=speech:1x200x560,speech_lengths:1 \
  --maxShapes=speech:1x400x560,speech_lengths:1 \
  --fp16 \
  --saveEngine="${ENC_PLAN}" \
  --workspace=8192 \
  2>&1 | tee "${ENGINE_DIR}/encoder_build.log"
echo "Encoder engine build exit: $?"
ls -lh "${ENC_PLAN}"
md5sum "${ENC_PLAN}"
echo ""

# ── Build decoder (single-profile) ──
DEC_PLAN="${ENGINE_DIR}/paraformer_decoder_sp1_400.plan"
echo "=== Building decoder engine ==="
echo "  → ${DEC_PLAN}"
trtexec --onnx="${MODEL_DIR}/decoder.onnx" \
  --minShapes=enc:1x1x512,enc_len:1,acoustic_embeds:1x1x512,acoustic_embeds_len:1 \
  --optShapes=enc:1x40x512,enc_len:1,acoustic_embeds:1x10x512,acoustic_embeds_len:1 \
  --maxShapes=enc:1x400x512,enc_len:1,acoustic_embeds:1x40x512,acoustic_embeds_len:1 \
  --fp16 \
  --saveEngine="${DEC_PLAN}" \
  --workspace=8192 \
  2>&1 | tee "${ENGINE_DIR}/decoder_build.log"
echo "Decoder engine build exit: $?"
ls -lh "${DEC_PLAN}"
md5sum "${DEC_PLAN}"
echo ""

# ── Restart container ──
echo "=== Restarting container: ${CONTAINER} ==="
docker start "${CONTAINER}"
echo ""

echo "=== Build complete ==="
echo "Encoder: ${ENC_PLAN}"
echo "Decoder: ${DEC_PLAN}"
