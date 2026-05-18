#!/usr/bin/env bash
# Build and verify the Jetson Qwen3 high-performance image.
#
# Intended to run on a Jetson Orin device. It bakes the Qwen3 runtime bundle
# into the image, verifies the model artifact set, runs HTTP TTS->ASR loopback,
# and runs the ASR streaming gate.
set -euo pipefail

IMAGE="${IMAGE:-openvoicestream:jetson-v1.12-highperf}"
REGISTRY_IMAGE="${REGISTRY_IMAGE:-sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:jetson-v1.12-highperf}"
ARTIFACT_SET="${QWEN3_ARTIFACT_SET:-orin-nano-highperf-2026-05-10}"
ARTIFACT_ROOT="${QWEN3_ARTIFACT_ROOT:-/opt/models/qwen3-edgellm}"
SERVICE_PORT="${SERVICE_PORT:-18621}"
CONTAINER="${CONTAINER:-ovs-highperf-verify}"
PROFILE="${OVS_PROFILE:-${SEEED_LOCAL_VOICE_PROFILE:-jetson-multilang-highperf}}"
PUSH=0
EXPORT_TAR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --registry-image) REGISTRY_IMAGE="$2"; shift 2 ;;
    --artifact-set) ARTIFACT_SET="$2"; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
    --service-port) SERVICE_PORT="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --push) PUSH=1; shift ;;
    --export-tar) EXPORT_TAR="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,28p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

case "$ARTIFACT_SET" in
  orin-nano-*)
    ASR_ENGINE_DIR="$ARTIFACT_ROOT/engines/orin-nano/highperf/asr_thinker_full_fp8embed"
    ASR_AUDIO_DIR="$ARTIFACT_ROOT/engines/orin-nano/highperf/asr_audio_encoder"
    ;;
  orin-nx-highperf-2026-05-14)
    ASR_ENGINE_DIR="$ARTIFACT_ROOT/engines/orin-nx/highperf-v2/asr_thinker_full_fp8embed"
    ASR_AUDIO_DIR="$ARTIFACT_ROOT/engines/orin-nx/highperf/asr_audio_encoder"
    ;;
  orin-nx-*)
    ASR_ENGINE_DIR="$ARTIFACT_ROOT/engines/orin-nx/highperf/asr_thinker_full_fp8embed"
    ASR_AUDIO_DIR="$ARTIFACT_ROOT/engines/orin-nx/highperf/asr_audio_encoder"
    ;;
  *)
    echo "unsupported artifact set: $ARTIFACT_SET" >&2
    exit 2
    ;;
esac

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "build $IMAGE"
docker build --network=host \
  -f deploy/docker/Dockerfile.jetson \
  --build-arg LANGUAGE_MODE=multilanguage \
  -t "$IMAGE" .

log "verify baked Qwen3 runtime files"
docker run --rm --network=host "$IMAGE" bash -lc '
  test -x /opt/jv-workers/qwen3_asr_worker
  test -x /opt/jv-workers/qwen3_tts_worker
  test -f /opt/edgellm-bin/libNvInfer_edgellm_plugin.so
  test -f /opt/edgellm-bin/libNvInfer_edgellm_plugin_asr.so
  test -f /opt/qwen3-edgellm-jetson/scripts/deploy_qwen3_artifacts.py
  test -f /opt/qwen3-edgellm-jetson/scripts/verify_reproduction.sh
  test -f /opt/qwen3-edgellm-jetson/scripts/verify_reproduction_streaming.sh
  nm /opt/edgellm-bin/libNvInfer_edgellm_plugin.so | grep -q w8a16_hmma_m16n16k16_kernel
'

log "artifact integrity gate: $ARTIFACT_SET at $ARTIFACT_ROOT"
docker run --rm --network=host \
  -v "$ARTIFACT_ROOT:$ARTIFACT_ROOT:ro" \
  "$IMAGE" \
  python3 /opt/qwen3-edgellm-jetson/scripts/deploy_qwen3_artifacts.py \
    --manifest /opt/qwen3-edgellm-jetson/deploy/artifacts/qwen3_manifest.json \
    --set "$ARTIFACT_SET" \
    --root "$ARTIFACT_ROOT" \
    --check-only \
    --verify-sha256

log "start verification service on :$SERVICE_PORT"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" --runtime nvidia --network host --ipc host \
  -v "$ARTIFACT_ROOT:/opt/models/qwen3-edgellm:ro" \
  -v /usr/local/cuda/lib64:/host-cuda:ro \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host-nvidia-libs:ro \
  -v /lib/aarch64-linux-gnu:/host-libs:ro \
  -v /usr/lib/python3.10/dist-packages/tensorrt:/usr/lib/python3.10/dist-packages/tensorrt:ro \
  -v /usr/src/tensorrt:/usr/src/tensorrt:ro \
  -e LANGUAGE_MODE=multilanguage \
  -e OVS_PROFILE="$PROFILE" \
  -e SEEED_LOCAL_VOICE_PROFILE="$PROFILE" \
  -e QWEN3_ARTIFACT_ROOT=/opt/models/qwen3-edgellm \
  -e QWEN3_ARTIFACT_VERIFY_SHA256=1 \
  -e CUDA_MODULE_LOADING=LAZY \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/host-cuda:/host-nvidia-libs:/host-libs \
  -e PYTHONPATH=/opt/speech:/usr/lib/python3.10/dist-packages \
  "$IMAGE" \
  python3 -m uvicorn app.main:app --host 0.0.0.0 --port "$SERVICE_PORT" >/dev/null

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$SERVICE_PORT/health" >/tmp/slv_highperf_health.json; then
    cat /tmp/slv_highperf_health.json
    break
  fi
  sleep 2
done
if ! curl -sf "http://127.0.0.1:$SERVICE_PORT/health" >/dev/null; then
  docker logs --tail 160 "$CONTAINER" >&2
  exit 1
fi

log "HTTP TTS->ASR loopback gate"
docker exec "$CONTAINER" bash /opt/qwen3-edgellm-jetson/scripts/verify_reproduction.sh \
  --plugin /opt/edgellm-bin/libNvInfer_edgellm_plugin.so \
  --artifact-root /opt/models/qwen3-edgellm \
  --set "$ARTIFACT_SET" \
  --service-url "http://127.0.0.1:$SERVICE_PORT" \
  --skip-clone

log "ASR streaming gate"
docker exec "$CONTAINER" bash /opt/qwen3-edgellm-jetson/scripts/verify_reproduction_streaming.sh \
  --repo-root /opt/qwen3-edgellm-jetson \
  --worker /opt/jv-workers/qwen3_asr_worker \
  --plugin /opt/edgellm-bin/libNvInfer_edgellm_plugin_asr.so \
  --engine-dir "$ASR_ENGINE_DIR" \
  --multimodal-engine-dir "$ASR_AUDIO_DIR" \
  --with-pcm \
  --latency-runs 2

if [ "$PUSH" -eq 1 ]; then
  log "push $REGISTRY_IMAGE"
  docker tag "$IMAGE" "$REGISTRY_IMAGE"
  if ! docker push "$REGISTRY_IMAGE"; then
    echo "push failed. If this Jetson is not logged into the registry, run with --export-tar or push from a logged-in machine." >&2
    exit 1
  fi
fi

if [ -n "$EXPORT_TAR" ]; then
  log "export $IMAGE -> $EXPORT_TAR"
  docker save "$IMAGE" | gzip -1 > "$EXPORT_TAR"
fi

log "PASS: $IMAGE"
