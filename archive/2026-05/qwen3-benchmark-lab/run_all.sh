#!/bin/bash
# Master script: deploy and run all Qwen3-TTS benchmarks on recomputer
# Usage: ./run_all.sh [onnx|cpp|sherpa|all]
set -e

REMOTE="recomputer@100.67.111.58"
REMOTE_DIR="/tmp/qwen3-tts-bench"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODE="${1:-all}"

echo "============================================================"
echo "Deploying Qwen3-TTS benchmarks to ${REMOTE}"
echo "Mode: ${MODE}"
echo "============================================================"

# Deploy scripts
echo "[deploy] Copying benchmark scripts..."
ssh ${REMOTE} "mkdir -p ${REMOTE_DIR}"
scp ${SCRIPT_DIR}/bench_onnx.py ${REMOTE}:${REMOTE_DIR}/
scp ${SCRIPT_DIR}/bench_cpp.sh ${REMOTE}:${REMOTE_DIR}/
scp ${SCRIPT_DIR}/bench_sherpa.sh ${REMOTE}:${REMOTE_DIR}/

# Run selected benchmarks
if [ "$MODE" = "onnx" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "======== APPROACH 1: ONNX Runtime ========"
    echo "Running inside existing jetson-voice container..."
    ssh ${REMOTE} "docker exec reachy_speech-speech-1 python3 ${REMOTE_DIR}/bench_onnx.py" 2>&1
fi

if [ "$MODE" = "cpp" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "======== APPROACH 2: qwen3-tts.cpp (new container) ========"
    ssh ${REMOTE} "bash ${REMOTE_DIR}/bench_cpp.sh" 2>&1
fi

if [ "$MODE" = "sherpa" ] || [ "$MODE" = "all" ]; then
    echo ""
    echo "======== APPROACH 3: sherpa-onnx (new container) ========"
    ssh ${REMOTE} "bash ${REMOTE_DIR}/bench_sherpa.sh" 2>&1
fi

echo ""
echo "============================================================"
echo "All benchmarks complete."
echo "============================================================"
