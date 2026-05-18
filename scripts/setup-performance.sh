#!/usr/bin/env bash
# Jetson Orin NX performance tuning for speech service.
# Run once after boot (or add to rc.local / systemd).
# Requires sudo.

set -euo pipefail

echo "=== Jetson Performance Tuning ==="

# 1. Set MAXN power mode (all cores, max clocks)
if command -v nvpmodel &>/dev/null; then
    echo "Setting nvpmodel to MAXN (mode 0)..."
    sudo nvpmodel -m 0
    echo "Current power mode:"
    sudo nvpmodel -q
else
    echo "nvpmodel not found, skipping"
fi

# 2. Lock clocks to max (bypass dynamic scaling)
if command -v jetson_clocks &>/dev/null; then
    echo "Locking clocks to max with jetson_clocks..."
    sudo jetson_clocks
else
    echo "jetson_clocks not found, skipping"
fi

# 3. Set GPU clock to max (Orin NX)
GPU_MAX_FREQ="/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/max_freq"
GPU_MIN_FREQ="/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/min_freq"
if [ -f "$GPU_MAX_FREQ" ] && [ -f "$GPU_MIN_FREQ" ]; then
    MAX=$(cat "$GPU_MAX_FREQ")
    echo "Pinning GPU freq to max: ${MAX}Hz"
    echo "$MAX" | sudo tee "$GPU_MIN_FREQ" > /dev/null
fi

echo "=== Done ==="
echo "Verify with: sudo jetson_clocks --show"
