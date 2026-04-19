#!/bin/bash
# Canonical in-host Release build. Run on Jetson.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build_cmake && mkdir build_cmake && cd build_cmake
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
         -DORT_ROOT=/home/recomputer/ort-from-container
make -j"$(nproc)" qwen3_speech_engine
md5sum qwen3_speech_engine.cpython-*.so
echo ''
echo 'Deploy: cp qwen3_speech_engine.cpython-*.so /home/recomputer/jetson-voice/app_overlay/'
echo 'Restart: cd /home/recomputer/jetson-voice/reachy_speech && docker compose restart speech'