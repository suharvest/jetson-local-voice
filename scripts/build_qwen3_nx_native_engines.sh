#!/usr/bin/env bash
# HISTORICAL — script used to build the orin-nx-highperf-2026-05-11 engines on
# the NX. ONNX/sidecar source paths below (/tmp/qwen3_talker_decode_*, /tmp/
# qwen3_tts_cp_*, etc.) were the build-time outputs of that session and are NOT
# expected to exist on a fresh checkout. Re-run only after re-exporting the
# matching ONNX with scripts/export_qwen3_tts_onnx.sh and friends.
# For PLUGIN, the canonical source-build path is
# ~/project/tensorrt-edge-llm/build_sm87/libNvInfer_edgellm_plugin.so .
set -euo pipefail

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
PLUGIN="${EDGELLM_PLUGIN_PATH:-$HOME/project/tensorrt-edge-llm/build_sm87/libNvInfer_edgellm_plugin.so}"
ROOT="${QWEN3_NX_ENGINE_ROOT:-/tmp/qwen3_native_engines_nx_0511}"

TALKER_SRC="/tmp/qwen3_talker_decode_w8a16_outputk_0510"
CP_ONNX="/tmp/qwen3_tts_cp_lmhead_pretranspose_0510/onnx/cp_single_head_nopast_lmhead_pretranspose.onnx"
CP_SIDECAR="/tmp/qwen3-tts-cp-nopast-0510/cp_dir"
CODE2WAV_SRC="/tmp/qwen3_code2wav_stateful_engine"

TALKER_OUT="$ROOT/talker_w8a16_outputk"
CP_OUT="$ROOT/cp_lmhead_pretranspose/cp_dir"
CODE2WAV_OUT="$ROOT/code2wav_stateful"
LOG_DIR="$ROOT/logs"

mkdir -p "$TALKER_OUT" "$CP_OUT" "$CODE2WAV_OUT" "$LOG_DIR"

echo "[nx-native] root=$ROOT"
echo "[nx-native] plugin=$PLUGIN"
"$TRTEXEC" --help | sed -n '1,3p' | tee "$LOG_DIR/trtexec_version.log" || true

echo "[nx-native] build talker W8A16 output-k"
sed \
  -e "s#--plugins=[^ ]*#--plugins=$PLUGIN#g" \
  -e "s#--saveEngine=[^ ]*#--saveEngine=$TALKER_OUT/talker_decode_w8a16_outputk.engine#g" \
  "$TALKER_SRC/build_outputk.cmd" > "$LOG_DIR/build_talker.cmd"
bash "$LOG_DIR/build_talker.cmd" 2>&1 | tee "$LOG_DIR/build_talker.log"

echo "[nx-native] build CP lm-head pretranspose"
python3 /tmp/build_qwen3_tts_cp_engine.py \
  --onnx "$CP_ONNX" \
  --output-dir "$CP_OUT" \
  --sidecar-dir "$CP_SIDECAR" \
  --workspace-mb 512 \
  --builder-opt-level 3 \
  --opt-past 8 \
  --max-past 20 \
  --max-aux-streams 1 \
  --bf16-io 2>&1 | tee "$LOG_DIR/build_cp.log"

echo "[nx-native] build stateful Code2Wav"
cp "$CODE2WAV_SRC/config.json" "$CODE2WAV_OUT/config.json"
"$TRTEXEC" \
  --onnx="$CODE2WAV_SRC/code2wav_stateful.onnx" \
  --saveEngine="$CODE2WAV_OUT/code2wav_stateful.engine" \
  --fp16 \
  --skipInference \
  --memPoolSize=workspace:1024 \
  --minShapes=codes:1x16x1 \
  --optShapes=codes:1x16x4 \
  --maxShapes=codes:1x16x16 \
  2>&1 | tee "$LOG_DIR/build_code2wav.log"

echo "[nx-native] manifest"
{
  echo "# Qwen3 NX-native engine manifest 2026-05-11"
  echo "root=$ROOT"
  echo "plugin=$PLUGIN"
  echo
  for p in "$TALKER_OUT" "$CP_OUT" "$CODE2WAV_OUT"; do
    echo "## $p"
    du -sh "$p"
    find "$p" -maxdepth 2 \( -name "*.engine" -o -name "*.safetensors" -o -name "*.bin" -o -name "config.json" \) -type f -print0 | xargs -0 -r md5sum
    echo
  done
} | tee "$ROOT/engine_manifest_nx_native_0511.txt"
