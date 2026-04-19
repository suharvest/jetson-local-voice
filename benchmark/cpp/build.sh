#!/bin/bash
# Canonical build script for Qwen3 Speech TRT engine
# Usage: ./build.sh [target]

set -e

ORT_ROOT=/opt/onnxruntime
BUILD_TYPE=Release
BUILD_DIR=build_cmake

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DORT_ROOT="$ORT_ROOT"

# Build
make -j$(nproc) qwen3_speech_engine

echo "Build complete: $BUILD_DIR/qwen3_speech_engine.cpython-*-linux-gnu.so"
