#!/usr/bin/env bash
# Build the split Matcha ODE estimator TensorRT engines on Jetson.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="${ONNX_DIR:-/opt/models/matcha-icefall-zh-en/onnx}"
OUT_DIR="${OUT_DIR:-/opt/models/matcha-icefall-zh-en/engines}"
WS="${WS:-1024}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ESTIMATOR_PREFIX="${ESTIMATOR_PREFIX:-matcha_estimator_step}"
ESTIMATOR_SUFFIX="${ESTIMATOR_SUFFIX:-_bf16.engine}"
FULL_ONNX="${FULL_ONNX:-${ONNX_PATH:-}}"

if [[ ! -x "${TRTEXEC}" ]]; then
  if command -v trtexec >/dev/null 2>&1; then
    TRTEXEC="$(command -v trtexec)"
  else
    echo "ERROR: trtexec not found. Mount /usr/src/tensorrt or set TRTEXEC." >&2
    exit 1
  fi
fi

mkdir -p "${OUT_DIR}"
mkdir -p "${ONNX_DIR}"

if [[ ! -f "${ONNX_DIR}/matcha_estimator_step0_trt.onnx" ]]; then
  if [[ -z "${FULL_ONNX}" ]]; then
    FULL_ONNX="${ONNX_DIR}/../model-steps-3.onnx"
  fi
  if [[ ! -f "${FULL_ONNX}" ]]; then
    echo "ERROR: split Matcha ONNX not found in ${ONNX_DIR}, and full ONNX not found: ${FULL_ONNX}" >&2
    exit 1
  fi
  echo "Split Matcha ONNX not found; generating from ${FULL_ONNX}"
  python3 "${SCRIPT_DIR}/split_matcha_trt.py" \
    --input "${FULL_ONNX}" \
    --output-dir "${ONNX_DIR}" \
    --verify
fi

echo "=== Building split Matcha estimator ==="
echo "ONNX dir:  ${ONNX_DIR}"
echo "Engine dir:${OUT_DIR}"
echo "Workspace: ${WS} MiB"

for step in 0 1 2; do
  ONNX_PATH="${ONNX_DIR}/matcha_estimator_step${step}_trt.onnx"
  EST_OUT="${OUT_DIR}/${ESTIMATOR_PREFIX}${step}${ESTIMATOR_SUFFIX}"
  echo ""
  echo "=== Step ${step}: ${ONNX_PATH} -> ${EST_OUT}"
  "${TRTEXEC}" \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${EST_OUT}" \
    --bf16 \
    --minShapes=z:1x80x72,mu:1x80x72,mask:1x1x72 \
    --optShapes=z:1x80x256,mu:1x80x256,mask:1x1x256 \
    --maxShapes=z:1x80x600,mu:1x80x600,mask:1x1x600 \
    --memPoolSize=workspace:"${WS}" \
    2>&1 | tee "${OUT_DIR}/matcha_estimator_step${step}_build.log"

  ls -lh "${EST_OUT}"
  md5sum "${EST_OUT}"
done
