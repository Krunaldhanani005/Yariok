#!/usr/bin/env bash
# ============================================================================
# YaariOK environment fix-up
# ----------------------------------------------------------------------------
# Resolves the two blockers identified in the live runtime audit:
#
#   1. No local MQTT broker on :1883 — vision events have nowhere to go.
#      Installs and enables Mosquitto + clients.
#
#   2. `torch.cuda.is_available() == False` despite cu130 wheels —
#      force-reinstalls PyTorch against a published CUDA index URL so
#      the runtime actually matches the host's NVIDIA driver.
#
# Target: Debian / Ubuntu / Raspberry Pi OS (apt-based).
# Run from project root:   bash setup_env.sh
# ============================================================================

set -euo pipefail

# ── Pretty logging ───────────────────────────────────────────────────────────
log() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || SUDO="sudo" && SUDO="${SUDO:-}"

# Default CUDA channel — override with: CUDA_CHANNEL=cu118 bash setup_env.sh
# Supported: cu121 (CUDA 12.1, most modern desktops) or cu118 (CUDA 11.8).
CUDA_CHANNEL="${CUDA_CHANNEL:-cu121}"

# Project venv path (must already exist).
VENV_DIR="${VENV_DIR:-$(pwd)/venv}"

# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Mosquitto MQTT broker
# ──────────────────────────────────────────────────────────────────────────────
install_mosquitto() {
    log "Installing mosquitto + mosquitto-clients via apt..."
    $SUDO apt-get update -y
    $SUDO apt-get install -y mosquitto mosquitto-clients

    log "Enabling and starting mosquitto service..."
    if command -v systemctl >/dev/null 2>&1; then
        $SUDO systemctl enable mosquitto
        $SUDO systemctl restart mosquitto
        $SUDO systemctl --no-pager --full status mosquitto | sed -n '1,10p' || true
    else
        warn "systemctl not available — start mosquitto manually with: mosquitto -d"
    fi

    log "Smoke-testing the broker on localhost:1883..."
    # mosquitto_sub blocks; use timeout to fail fast if the broker is down.
    if mosquitto_pub -h localhost -p 1883 -t "yaariok/setup/ping" -m "ok" 2>/dev/null; then
        log "Mosquitto is accepting publishes ✓"
    else
        warn "mosquitto_pub failed — check 'systemctl status mosquitto' and /etc/mosquitto/mosquitto.conf"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — PyTorch with a real CUDA runtime
# ──────────────────────────────────────────────────────────────────────────────
fix_torch_cuda() {
    [[ -x "$VENV_DIR/bin/python" ]] || die "venv not found at $VENV_DIR (set VENV_DIR=... to override)"

    log "Detecting NVIDIA driver..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi | sed -n '1,4p' || true
    else
        warn "nvidia-smi not found — no NVIDIA GPU driver detected. Skipping torch reinstall."
        warn "(YaariOK will keep running on CPU. Install the driver first, then re-run this script.)"
        return 0
    fi

    log "Force-reinstalling PyTorch from https://download.pytorch.org/whl/${CUDA_CHANNEL}"
    log "  → channel: ${CUDA_CHANNEL} (override with CUDA_CHANNEL=cu118 to switch to CUDA 11.8)"

    # --force-reinstall is critical: the current cu130 build was silently
    # falling back to CPU; we want it gone, replaced with cu121 / cu118.
    "$VENV_DIR/bin/pip" install --upgrade --force-reinstall \
        torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"

    log "Verifying torch sees CUDA..."
    "$VENV_DIR/bin/python" - <<'PY'
import torch
print(f"torch        {torch.__version__}")
print(f"CUDA build   {torch.version.cuda}")
print(f"CUDA avail   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device 0     {torch.cuda.get_device_name(0)}")
PY
}

# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Optional: install ONNX Runtime GPU so InsightFace also moves off CPU
# ──────────────────────────────────────────────────────────────────────────────
fix_onnxruntime() {
    [[ -x "$VENV_DIR/bin/python" ]] || return 0
    if "$VENV_DIR/bin/python" -c "import onnxruntime" 2>/dev/null; then
        log "Replacing onnxruntime → onnxruntime-gpu so InsightFace uses CUDA..."
        "$VENV_DIR/bin/pip" uninstall -y onnxruntime || true
        "$VENV_DIR/bin/pip" install --upgrade onnxruntime-gpu
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────
main() {
    log "YaariOK environment setup starting."
    install_mosquitto
    fix_torch_cuda
    fix_onnxruntime
    log "Done. Restart the app:  $VENV_DIR/bin/python backend/main.py"
}

main "$@"
