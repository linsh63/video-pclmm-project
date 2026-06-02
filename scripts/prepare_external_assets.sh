#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/2] Preparing pretrained models..."
bash "$PROJECT_ROOT/scripts/prepare_pretrained_models.sh"

echo
echo "[2/2] Preparing PCLMM dataset..."
bash "$PROJECT_ROOT/scripts/prepare_pclmm_dataset.sh"

echo
echo "All external assets are prepared."
