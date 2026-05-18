#!/bin/bash
# Benchmark: Qwen3-TTS via sherpa-onnx (community dev branch)
# Runs in a NEW container.
set -e

CONTAINER_NAME="qwen3-tts-sherpa-bench"
BASE_IMAGE="python:3.12-slim"

echo "============================================================"
echo "Qwen3-TTS sherpa-onnx Benchmark"
echo "============================================================"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[setup] Container ${CONTAINER_NAME} already exists"
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker start ${CONTAINER_NAME}
    fi
else
    echo "[setup] Creating new container: ${CONTAINER_NAME}"
    docker run -d \
        --name ${CONTAINER_NAME} \
        --runtime nvidia \
        --gpus all \
        -v /tmp/qwen3-models:/models \
        ${BASE_IMAGE} \
        sleep infinity
fi

echo "[setup] Installing Python + pip..."
docker exec ${CONTAINER_NAME} bash -c '
    apt-get update -qq
    apt-get install -y -qq python3 python3-pip python3-dev git wget 2>/dev/null | tail -1
    echo "Python: $(python3 --version)"
'

echo "[setup] Installing sherpa-onnx (latest from PyPI)..."
docker exec ${CONTAINER_NAME} bash -c '
    pip3 install -q sherpa-onnx numpy soundfile 2>&1 | tail -5

    python3 -c "
import sherpa_onnx
print(f\"sherpa-onnx version: {sherpa_onnx.__version__}\")

# Check if Qwen3-TTS is supported
import inspect
members = [m for m in dir(sherpa_onnx) if \"tts\" in m.lower() or \"qwen\" in m.lower()]
print(f\"TTS-related members: {members}\")

# Check OfflineTts config options
if hasattr(sherpa_onnx, \"OfflineTtsConfig\"):
    sig = inspect.signature(sherpa_onnx.OfflineTtsConfig.__init__)
    print(f\"OfflineTtsConfig params: {list(sig.parameters.keys())}\")

# Try to find model config for Qwen3-TTS
help_text = []
for attr in dir(sherpa_onnx):
    obj = getattr(sherpa_onnx, attr)
    if callable(obj) and \"model\" in attr.lower():
        try:
            sig = inspect.signature(obj.__init__)
            params = list(sig.parameters.keys())
            if any(\"qwen\" in p.lower() or \"tts\" in p.lower() for p in params):
                help_text.append(f\"{attr}: {params}\")
        except:
            pass
if help_text:
    print(f\"Qwen-related configs: {help_text}\")
else:
    print(\"[WARN] No Qwen3-TTS support found in this version of sherpa-onnx\")
    print(\"The community PR has not been merged yet.\")
    print(\"See: https://github.com/k2-fsa/sherpa-onnx/issues/3104\")
" 2>&1
'

echo ""
echo "[check] Testing if dev branch with Qwen3 support is available..."
docker exec ${CONTAINER_NAME} bash -c '
    # Try installing from the dev branch that has Qwen3-TTS support
    pip3 install -q git+https://github.com/HeiSir2014/sherpa-onnx.git@develop 2>&1 | tail -5 || \
    echo "[WARN] Dev branch install failed. Qwen3-TTS support not yet available in sherpa-onnx."
    echo ""
    echo "Status: sherpa-onnx Qwen3-TTS support is in development (PR pending)."
    echo "Monitor: https://github.com/k2-fsa/sherpa-onnx/issues/3104"
' 2>&1 || true

echo ""
echo "Container: ${CONTAINER_NAME}"
echo "To cleanup: docker rm -f ${CONTAINER_NAME}"
