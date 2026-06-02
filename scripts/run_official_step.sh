#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <script.py> [args...]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$PROJECT_ROOT/outputs/official_runtime}"
PYTHON_BIN="${PYTHON_BIN:-/data4/songxinshuai/conda/envs/video-pclmm/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
WHISPER_CACHE_DIR="${WHISPER_CACHE_DIR:-/data4/songxinshuai/cache/whisper}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/outputs/cache}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/outputs/cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/outputs/cache/torch}"

export CUDA_VISIBLE_DEVICES
export WHISPER_CACHE_DIR
export XDG_CACHE_HOME
export HF_HOME
export TRANSFORMERS_CACHE
export TORCH_HOME
export PYTHONPATH="$RUNTIME_ROOT/code:${PYTHONPATH:-}"

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

script="$1"
shift

cd "$RUNTIME_ROOT/code"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Running: $script $*"
exec "$PYTHON_BIN" "$script" "$@"
