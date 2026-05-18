#!/usr/bin/env bash
# Build Paraformer TRT engines on Jetson Orin.
# Run inside the slim container with /usr/src/tensorrt mounted, or on the host
# with trtexec available.
#
# Usage:
#   bash scripts/build_paraformer_trt.sh [--container NAME] [--model-dir PATH]
#
# Options:
#   --container NAME   Legacy no-op, kept for compatibility
#   --model-dir PATH   Paraformer model directory (default: /opt/models/paraformer-streaming)
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/opt/models/paraformer-streaming}"
ENGINE_DIR="${MODEL_DIR}/engines"
CONTAINER="${CONTAINER:-}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

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
echo "trtexec:    ${TRTEXEC}"
echo ""

if [[ ! -x "${TRTEXEC}" ]]; then
    if command -v trtexec >/dev/null 2>&1; then
        TRTEXEC="$(command -v trtexec)"
    else
        echo "ERROR: trtexec not found. Mount /usr/src/tensorrt or set TRTEXEC." >&2
        exit 1
    fi
fi

# ── Build encoder ──
ENC_PLAN="${ENGINE_DIR}/paraformer_encoder_dp4_400.plan"
echo "=== Building encoder engine ==="
echo "  → ${ENC_PLAN}"
"${TRTEXEC}" --onnx="${MODEL_DIR}/encoder.onnx" \
  --minShapes=speech:1x40x560,speech_lengths:1 \
  --optShapes=speech:1x80x560,speech_lengths:1 \
  --maxShapes=speech:1x400x560,speech_lengths:1 \
  --saveEngine="${ENC_PLAN}" \
  --memPoolSize=workspace:2048 \
  --skipInference \
  2>&1 | tee "${ENGINE_DIR}/encoder_build.log"
echo "Encoder engine build exit: $?"
ls -lh "${ENC_PLAN}"
md5sum "${ENC_PLAN}"
echo ""

# ── Build decoder from surgically-modified ONNX ──
# Requires decoder-trt.onnx (make_pad_mask subgraphs externalized).
# Generate with: python3 scripts/surgery_paraformer_decoder.py
#   --input ${MODEL_DIR}/decoder.onnx --output ${MODEL_DIR}/decoder-trt.onnx
DEC_PLAN="${ENGINE_DIR}/paraformer_decoder_fp16.plan"
DEC_ONNX="${MODEL_DIR}/decoder-trt.onnx"
RAW_DEC_ONNX="${MODEL_DIR}/decoder.onnx"
echo "=== Building decoder engine ==="
echo "  ONNX: ${DEC_ONNX}"
echo "  Plan: ${DEC_PLAN}"
if [[ ! -f "${DEC_ONNX}" ]]; then
  echo "decoder-trt.onnx not found; generating from decoder.onnx ..."
  if [[ ! -f "${RAW_DEC_ONNX}" ]]; then
    echo "ERROR: neither ${DEC_ONNX} nor ${RAW_DEC_ONNX} exists." >&2
    exit 1
  fi
  python3 "$(dirname "$0")/surgery_paraformer_decoder.py" \
    --input "${RAW_DEC_ONNX}" \
    --output "${DEC_ONNX}" \
    --validate
fi
"${TRTEXEC}" --onnx="${DEC_ONNX}" \
  --minShapes=enc:1x1x512,acoustic_embeds:1x1x512,pad_mask:1x1,enc_pad_mask:1x1 \
  --optShapes=enc:1x40x512,acoustic_embeds:1x10x512,pad_mask:1x10,enc_pad_mask:1x40 \
  --maxShapes=enc:1x400x512,acoustic_embeds:1x40x512,pad_mask:1x40,enc_pad_mask:1x400 \
  --bf16 \
  --saveEngine="${DEC_PLAN}" \
  --memPoolSize=workspace:1024 \
  2>&1 | tee "${ENGINE_DIR}/decoder_build.log"
echo "Decoder engine build exit: $?"
ls -lh "${DEC_PLAN}"
md5sum "${DEC_PLAN}"
echo ""

echo "=== Build complete ==="
echo "Encoder: ${ENC_PLAN}"
echo "Decoder: ${DEC_PLAN}"
