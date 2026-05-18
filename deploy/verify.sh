#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  deploy/verify.sh [--url http://host:port] [--timeout-sec N] [--tts-smoke] [--roundtrip]

Checks:
  - /health becomes reachable before timeout
  - at least one of ASR or TTS is ready
  - /tts/capabilities and /asr/capabilities are queried when ready (warn-only)
  - optional /tts smoke request writes a temporary WAV
  - optional TTS -> ASR round-trip verifies non-empty, similar ASR text
EOF
}

base_url="http://127.0.0.1:8621"
timeout_sec=180
tts_smoke=0
roundtrip=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)
      base_url="${2:-}"
      shift 2
      ;;
    --timeout-sec)
      timeout_sec="${2:-}"
      shift 2
      ;;
    --tts-smoke)
      tts_smoke=1
      shift
      ;;
    --roundtrip)
      roundtrip=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required." >&2
  exit 1
fi

deadline=$((SECONDS + timeout_sec))
health_json=""
while [[ "$SECONDS" -lt "$deadline" ]]; do
  if health_json="$(curl -fsS "${base_url}/health" 2>/dev/null)"; then
    break
  fi
  sleep 2
done

if [[ -z "$health_json" ]]; then
  echo "Timed out waiting for ${base_url}/health" >&2
  exit 1
fi

echo "$health_json" | python3 -c '
import json, sys
data = json.load(sys.stdin)
print(json.dumps(data, ensure_ascii=False, indent=2))
if not (data.get("asr") or data.get("tts")):
    raise SystemExit("neither ASR nor TTS is ready")
'

tts_ready="$(printf '%s' "$health_json" | python3 -c 'import json,sys; print("1" if json.load(sys.stdin).get("tts") else "0")')"
asr_ready="$(printf '%s' "$health_json" | python3 -c 'import json,sys; print("1" if json.load(sys.stdin).get("asr") else "0")')"

if [[ "$tts_ready" == "1" ]]; then
  echo "TTS capabilities:"
  if ! curl -fsS "${base_url}/tts/capabilities"; then
    echo "WARN: ${base_url}/tts/capabilities is unavailable; continuing with behavior checks." >&2
  fi
  echo
fi

if [[ "$asr_ready" == "1" ]]; then
  echo "ASR capabilities:"
  if ! curl -fsS "${base_url}/asr/capabilities"; then
    echo "WARN: ${base_url}/asr/capabilities is unavailable; continuing with behavior checks." >&2
  fi
  echo
fi

if [[ "$tts_smoke" -eq 1 && "$tts_ready" == "1" ]]; then
  out="$(mktemp -t openvoicestream-tts-XXXXXX.wav)"
  curl -fsS -X POST "${base_url}/tts" \
    -H "Content-Type: application/json" \
    -d '{"text":"你好，欢迎使用 OpenVoiceStream。"}' \
    --output "$out"
  bytes="$(wc -c < "$out" | tr -d ' ')"
  if [[ "$bytes" -lt 1000 ]]; then
    echo "TTS smoke output is too small: ${bytes} bytes" >&2
    exit 1
  fi
  echo "TTS smoke OK: ${out} (${bytes} bytes)"
fi

if [[ "$roundtrip" -eq 1 ]]; then
  if [[ "$tts_ready" != "1" || "$asr_ready" != "1" ]]; then
    echo "Round-trip skipped: TTS and ASR must both be ready." >&2
    exit 1
  fi
  script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
  python3 "${script_dir}/roundtrip_verify.py" --url "$base_url" --timeout-sec "$timeout_sec"
fi
