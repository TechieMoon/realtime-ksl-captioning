#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[setup] repository: ${ROOT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[setup] ${PYTHON_BIN} is not installed." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -m venv --help >/dev/null 2>&1; then
  echo "[setup] python venv support is missing."
  echo "[setup] On Ubuntu, run: sudo apt update && sudo apt install python3.12-venv"
  exit 1
fi

if ! "${PYTHON_BIN}" -m ensurepip --version >/dev/null 2>&1; then
  echo "[setup] python ensurepip is missing."
  echo "[setup] On Ubuntu, run: sudo apt update && sudo apt install python3.12-venv"
  exit 1
fi

echo "[setup] creating backend virtual environment"
cd "${ROOT_DIR}/server"
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[setup] writing server/.env from .env.example if needed"
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
fi

echo "[setup] checking Node.js"
if ! command -v node >/dev/null 2>&1; then
  echo "[setup] Node.js is not installed. Install Node 20.19+ before frontend setup." >&2
  echo "[setup] Recommended Ubuntu commands:"
  echo "curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"
  echo "sudo apt install -y nodejs"
  exit 1
fi

NODE_MAJOR="$(node -p "process.versions.node.split('.')[0]")"
NODE_MINOR="$(node -p "process.versions.node.split('.')[1]")"
if [ "${NODE_MAJOR}" -lt 20 ] || { [ "${NODE_MAJOR}" -eq 20 ] && [ "${NODE_MINOR}" -lt 19 ]; }; then
  echo "[setup] Node.js $(node -v) is too old. Vite requires Node 20.19+." >&2
  exit 1
fi

echo "[setup] installing frontend dependencies"
cd "${ROOT_DIR}/frontend"
npm install

echo "[setup] done"
echo "[setup] backend: cd ${ROOT_DIR}/server && source .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo "[setup] frontend: cd ${ROOT_DIR}/frontend && npm run dev -- --host 0.0.0.0 --port 5173"
