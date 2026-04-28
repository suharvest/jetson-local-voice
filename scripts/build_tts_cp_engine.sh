#!/bin/bash
# Build Qwen3-TTS CP unified BF16 dual-profile engine. TensorRT 10.3 / Jetson.
#
# Per-device profile (override via env):
#   Nano 8GB  (default): WS=256
#   NX 16GB           :  WS=2048
#   AGX 32GB          :  WS=4096
#
# Cross-device cubin compatibility verified for Ampere SM 8.7. For best
# tactic tuning, build on the target device with its WS profile. A
# Nano-built engine works on NX/AGX but uses conservative tactics.
#
# Profile shapes (matched to existing live engine via trtexec --dumpOptimizationProfile):
#   Profile 0 (prefill): inputs_embeds=1x1..200x1024, cache_position=1..200,
#                        5 layers KV all empty 1x8x0x128
#   Profile 1 (decode):  inputs_embeds=1x1x1024 (fixed), cache_position=1 (fixed),
#                        5 layers KV 1x8x0..20x128

set -euo pipefail

ONNX="${ONNX:-/home/harvest/voice_test/models/qwen3-tts/onnx/code_predictor.onnx}"
OUT_DIR="${OUT_DIR:-/home/harvest/voice_test/models/qwen3-tts/engines}"
ENGINE_NAME="${ENGINE_NAME:-cp_unified_bf16_ws256.engine}"
WS="${WS:-256}"  # workspace MiB

OUT_ENGINE="${OUT_DIR}/${ENGINE_NAME}"

echo "ONNX:        $ONNX"
echo "OUT:         $OUT_ENGINE"
echo "Workspace:   ${WS}MiB (down from 2GiB default)"
echo "Profile 0 (prefill): seq=1..200, empty KV"
echo "Profile 1 (decode):  seq=1 fixed, KV 0..20"
echo

python3 -c "
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open('$ONNX', 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f'ONNX parse error: {parser.get_error(i)}')
        raise RuntimeError('ONNX parse failed')

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ${WS} << 20)
config.set_flag(trt.BuilderFlag.BF16)

n_layers = 5

# Profile 0: prefill (variable seq, empty KV)
p0 = builder.create_optimization_profile()
p0.set_shape('inputs_embeds',  (1, 1, 1024), (1, 20, 1024), (1, 200, 1024))
p0.set_shape('cache_position', (1,),         (20,),         (200,))
for i in range(n_layers):
    p0.set_shape(f'past_key_{i}',   (1, 8, 0, 128), (1, 8, 0, 128), (1, 8, 0, 128))
    p0.set_shape(f'past_value_{i}', (1, 8, 0, 128), (1, 8, 0, 128), (1, 8, 0, 128))
config.add_optimization_profile(p0)

# Profile 1: decode (seq=1 fixed, KV grows)
p1 = builder.create_optimization_profile()
p1.set_shape('inputs_embeds',  (1, 1, 1024), (1, 1, 1024), (1, 1, 1024))
p1.set_shape('cache_position', (1,),         (1,),         (1,))
for i in range(n_layers):
    p1.set_shape(f'past_key_{i}',   (1, 8, 0, 128), (1, 8, 10, 128), (1, 8, 20, 128))
    p1.set_shape(f'past_value_{i}', (1, 8, 0, 128), (1, 8, 10, 128), (1, 8, 20, 128))
config.add_optimization_profile(p1)

print('Building CP unified BF16 engine ...')
engine_bytes = builder.build_serialized_network(network, config)
if engine_bytes is None:
    raise RuntimeError('Engine build failed (returned None)')

data = bytes(engine_bytes)
with open('$OUT_ENGINE', 'wb') as f:
    f.write(data)
print(f'Saved: $OUT_ENGINE ({len(data)/1e6:.1f} MB)')
"

echo
ls -lh "$OUT_ENGINE"
md5sum "$OUT_ENGINE"
