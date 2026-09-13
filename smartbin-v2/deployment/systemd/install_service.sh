#!/usr/bin/env bash
# ============================================================================
# SmartBin AI v2 — Systemd Service Installation Script for Raspberry Pi 5
# ============================================================================

set -e

SERVICE_NAME="smartbin.service"
SERVICE_SRC="$(dirname "$0")/${SERVICE_NAME}"
SYSTEMD_DIR="/etc/systemd/system"

echo "Installing SmartBin AI v2 Systemd Service..."

if [ "$EUID" -ne 0 ]; then
  echo "Error: Please run as root (sudo ./install_service.sh)"
  exit 1
fi

if [ ! -f "$SERVICE_SRC" ]; then
  echo "Error: Service unit file not found: $SERVICE_SRC"
  exit 1
fi

# Copy service unit
cp "$SERVICE_SRC" "$SYSTEMD_DIR/$SERVICE_NAME"
chmod 644 "$SYSTEMD_DIR/$SERVICE_NAME"

# Reload daemon and enable service on boot
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

echo "===================================================================="
echo "SmartBin AI v2 Service installed and enabled on boot!"
echo "Commands:"
echo "  Start:   sudo systemctl start smartbin"
echo "  Status:  sudo systemctl status smartbin"
echo "  Logs:    journalctl -u smartbin -f"
echo "  Stop:    sudo systemctl stop smartbin"
echo "===================================================================="
