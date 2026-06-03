#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data4/songxinshuai/conda/envs/video-pclmm/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m uvicorn src.api.app:app --host "$HOST" --port "$PORT"
