#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

ANNOTATION_FILE="${ANNOTATION_FILE:-$PROJECT_ROOT/data/annotations/Annotation_Subset.csv}"
VIDEO_ROOT="${VIDEO_ROOT:-$PROJECT_ROOT/data/raw/videos}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/data/processed/video_manifest.csv}"

"$PYTHON_BIN" "$PROJECT_ROOT/src/data/build_manifest.py" \
  --annotation-file "$ANNOTATION_FILE" \
  --video-root "$VIDEO_ROOT" \
  --output "$OUTPUT" \
  "$@"
