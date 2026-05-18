#!/usr/bin/env bash
# Run perf on a device with the client ALSO running on that device.
# This eliminates Mac↔device network latency from the measurement, giving
# the compute-bound "device intrinsic" numbers.
#
# Usage:
#   bench/perf/run_on_device.sh <fleet-node>  [-- perf.py args...]
#
# Examples:
#   bench/perf/run_on_device.sh orin-nano -- asr --warmup 2 --runs 10
#   bench/perf/run_on_device.sh radxa     -- v2v --llm-delay 0
#   bench/perf/run_on_device.sh orin-nano -- matrix

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <fleet-node> [-- perf.py args...]" >&2
  echo "Example: $0 orin-nano -- asr --warmup 2 --runs 10" >&2
  exit 2
fi

NODE="$1"; shift
# Drop leading -- if present
[[ "${1:-}" == "--" ]] && shift

# Default warmup=5 here (vs perf.py's 3): we just deployed/restarted,
# so the first few inferences are TRT engine warmup + CUDA cache fill.
PERF_ARGS="${*:-asr --warmup 5 --runs 10}"
REMOTE_DIR="/tmp/seeed-perf"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

echo ">> Target: $NODE"
echo ">> Staging perf harness without historical results..."
(
  cd "$LOCAL_DIR"
  tar --exclude='./results' --exclude='./__pycache__' -cf - .
) | (
  cd "$STAGE_DIR"
  tar -xf -
)

echo ">> Pushing staged harness -> $NODE:$REMOTE_DIR/"
fleet push "$NODE" "$STAGE_DIR"/  "$REMOTE_DIR"/

echo ">> Verifying corpus on device (fetch from HF if missing)..."
fleet exec "$NODE" -- "
  set -e
  cd $REMOTE_DIR
  if ! command -v uv >/dev/null 2>&1; then
    echo 'installing uv...'
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    export PATH=\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH
  fi
  test -f corpus/short/zh_short_01.wav || \\
    uv run --with numpy python corpus/fetch.py --from hf
  uv run --with numpy python corpus/fetch.py --verify --strict
"

# Inject --base-url AFTER the subcommand (it's a per-subcommand flag, not global).
# Only inject if the user didn't already pass --base-url themselves.
if [[ "$PERF_ARGS" != *"--base-url"* ]]; then
  SUB="${PERF_ARGS%% *}"
  REST="${PERF_ARGS#${SUB}}"      # everything after the subcommand, may be empty
  PERF_ARGS="$SUB --base-url http://127.0.0.1:${OVS_PORT:-${SEEED_LOCAL_VOICE_PORT:-8621}} --mode-label local$REST"
fi

echo ">> Running perf on-device: python perf.py $PERF_ARGS"
fleet exec "$NODE" -- "
  export PATH=\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH
  cd $REMOTE_DIR
  uv run --with websocket-client --with requests --with numpy --with jiwer \\
    python perf.py $PERF_ARGS
"

echo ">> Pulling results back to Mac..."
mkdir -p "$LOCAL_DIR/results"
fleet pull "$NODE" "$REMOTE_DIR/results/" "$LOCAL_DIR/results/_from_${NODE}/"

echo ">> Done. Local-mode results at $LOCAL_DIR/results/_from_${NODE}/"
