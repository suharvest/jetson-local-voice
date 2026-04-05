#!/bin/bash
# Benchmark: Qwen3-TTS 0.6B via qwen3-tts.cpp (GGUF, pure C++)
# Runs in a NEW container to avoid breaking existing setup.
set -e

WORK_DIR="/tmp/qwen3-tts-cpp"
CONTAINER_NAME="qwen3-tts-cpp-bench"
# Use existing jetson-voice image (has CUDA + ONNX Runtime)
BASE_IMAGE="jetson-voice:v3.0-slim"

echo "============================================================"
echo "Qwen3-TTS 0.6B C++ (GGUF) Benchmark"
echo "============================================================"

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[setup] Container ${CONTAINER_NAME} already exists"
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "[setup] Starting existing container..."
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

echo "[setup] Installing build dependencies..."
docker exec ${CONTAINER_NAME} bash -c '
    apt-get update -qq && \
    apt-get install -y -qq git cmake g++ wget curl libcurl4-openssl-dev 2>/dev/null | tail -1
    echo "[setup] Build tools installed"
'

echo "[build] Cloning qwen3-tts.cpp..."
docker exec ${CONTAINER_NAME} bash -c '
    cd /tmp
    if [ -d "qwen3-tts.cpp" ]; then
        echo "Already cloned, pulling latest..."
        cd qwen3-tts.cpp && git pull 2>/dev/null || true
    else
        git clone --depth 1 https://github.com/predict-woo/qwen3-tts.cpp.git
    fi
'

echo "[build] Building qwen3-tts.cpp..."
docker exec ${CONTAINER_NAME} bash -c '
    cd /tmp/qwen3-tts.cpp
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
    make -j$(nproc) 2>&1 | tail -10
    echo ""
    echo "[build] Build result: $?"
    ls -la bin/ 2>/dev/null || ls -la qwen3-tts* 2>/dev/null || echo "Binary not found, checking..."
    find /tmp/qwen3-tts.cpp/build -type f -executable -name "*qwen*" -o -name "*tts*" 2>/dev/null | head -5
'

echo "[model] Downloading GGUF Q8_0 model (0.6B)..."
docker exec ${CONTAINER_NAME} bash -c '
    mkdir -p /models/qwen3-tts-0.6b
    cd /models/qwen3-tts-0.6b

    # Download from HuggingFace - cgisky repo has 1.7B, check for 0.6B
    # The predict-woo repo may have its own model download
    HF_BASE="https://huggingface.co/cgisky/qwen3-tts-custom-gguf/resolve/main"

    # Check what models are needed by the binary
    /tmp/qwen3-tts.cpp/build/bin/qwen3-tts --help 2>&1 | head -20 || \
    /tmp/qwen3-tts.cpp/build/qwen3-tts --help 2>&1 | head -20 || \
    find /tmp/qwen3-tts.cpp/build -type f -executable 2>/dev/null | while read f; do
        echo "Found: $f"
        $f --help 2>&1 | head -10
    done
    echo "[model] Check binary help output above for model requirements"
'

echo ""
echo "[NOTE] Build complete. Check output above for next steps."
echo "Container: ${CONTAINER_NAME}"
echo "To exec: docker exec -it ${CONTAINER_NAME} bash"
