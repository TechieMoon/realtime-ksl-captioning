#!/usr/bin/env bash
set -euo pipefail

DEVICE_NR="${DEVICE_NR:-20}"
CARD_LABEL="${CARD_LABEL:-KSL Caption Camera}"
DEVICE_PATH="/dev/video${DEVICE_NR}"

have_command() {
  command -v "$1" >/dev/null 2>&1
}

ensure_packages() {
  local needs_packages=0

  if ! modinfo v4l2loopback >/dev/null 2>&1; then
    needs_packages=1
  fi

  if ! have_command v4l2-ctl; then
    needs_packages=1
  fi

  if [ "${needs_packages}" -eq 0 ]; then
    echo "[virtual-camera] v4l2loopback module and v4l2-ctl are already available"
    return
  fi

  echo "[virtual-camera] installing missing v4l2loopback packages"
  sudo apt-get \
    -o Acquire::Retries=2 \
    -o Acquire::http::Timeout=15 \
    -o Acquire::https::Timeout=15 \
    update
  sudo apt-get install -y v4l2loopback-dkms v4l2loopback-utils
}

ensure_packages

if [ ! -e "${DEVICE_PATH}" ] && lsmod | awk '{print $1}' | grep -qx v4l2loopback; then
  echo "[virtual-camera] v4l2loopback is loaded but ${DEVICE_PATH} is missing"
  echo "[virtual-camera] reloading v4l2loopback with ${DEVICE_PATH}"
  if ! sudo modprobe -r v4l2loopback; then
    echo "[virtual-camera] failed to unload v4l2loopback" >&2
    echo "[virtual-camera] close apps using virtual cameras, then rerun this script" >&2
    exit 1
  fi
fi

if [ ! -e "${DEVICE_PATH}" ]; then
  echo "[virtual-camera] creating ${DEVICE_PATH}"
  sudo modprobe v4l2loopback \
    devices=1 \
    "video_nr=${DEVICE_NR}" \
    "card_label=${CARD_LABEL}" \
    exclusive_caps=1
  if have_command udevadm; then
    sudo udevadm settle
  fi
else
  echo "[virtual-camera] ${DEVICE_PATH} already exists"
fi

if id -nG "${USER}" | tr ' ' '\n' | grep -qx video; then
  echo "[virtual-camera] ${USER} is already in the video group"
else
  echo "[virtual-camera] adding ${USER} to the video group"
  sudo usermod -aG video "${USER}"
  echo "[virtual-camera] log out and log back in before using ${DEVICE_PATH}"
fi

echo "[virtual-camera] available video devices:"
v4l2-ctl --list-devices || true

if [ ! -e "${DEVICE_PATH}" ]; then
  echo "[virtual-camera] failed to create ${DEVICE_PATH}" >&2
  echo "[virtual-camera] run 'lsmod | grep v4l2loopback' and rerun this script" >&2
  exit 1
fi

echo "[virtual-camera] ${DEVICE_PATH} is ready"
echo "[virtual-camera] use ${DEVICE_PATH} in the frontend Virtual Camera device field"
