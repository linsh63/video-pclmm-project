#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:-v0.1.0}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/release_assets}"
PART_SIZE="${PART_SIZE:-1900M}"
WHISPER_FILE="${WHISPER_FILE:-}"

CORE_ARCHIVE="video-pclmm-runtime-core-${VERSION}.tar.gz"
CORE_PATH="$OUT_DIR/$CORE_ARCHIVE"
MANIFEST="$OUT_DIR/runtime-assets-${VERSION}.sha256"

REQUIRED_PATHS=(
  "outputs/checkpoints/multi_modal_cross_attention_model.pth"
  "pretrained/googlevit-base-patch16-224-in21k"
  "pretrained/bert_chinese"
  "pretrained/FER-VT"
  "outputs/cache/torch/hub/checkpoints/resnet34-b627a593.pth"
)

mkdir -p "$OUT_DIR"

missing=0
for path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -e "$PROJECT_ROOT/$path" ]]; then
    echo "Missing required runtime asset: $path" >&2
    missing=1
  fi
done

if [[ "$missing" == "1" ]]; then
  exit 1
fi

rm -f "$CORE_PATH" "$MANIFEST" "$OUT_DIR"/whisper-large-v3.pt.part-* "$OUT_DIR"/whisper-large-v3.pt.sha256

echo "Packaging core runtime assets: $CORE_PATH"
(
  cd "$PROJECT_ROOT"
  tar \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -czf "$CORE_PATH" \
    "${REQUIRED_PATHS[@]}"
)

(
  cd "$OUT_DIR"
  sha256sum "$CORE_ARCHIVE" > "$MANIFEST"
)

if [[ -n "$WHISPER_FILE" ]]; then
  if [[ ! -f "$WHISPER_FILE" ]]; then
    echo "WHISPER_FILE does not exist: $WHISPER_FILE" >&2
    exit 1
  fi

  echo "Splitting Whisper model into GitHub Release sized parts..."
  split -b "$PART_SIZE" "$WHISPER_FILE" "$OUT_DIR/whisper-large-v3.pt.part-"
  whisper_sha="$(sha256sum "$WHISPER_FILE" | awk '{print $1}')"
  printf "%s  whisper-large-v3.pt\n" "$whisper_sha" > "$OUT_DIR/whisper-large-v3.pt.sha256"
  (
    cd "$OUT_DIR"
    sha256sum whisper-large-v3.pt.part-* >> "$MANIFEST"
    sha256sum whisper-large-v3.pt.sha256 >> "$MANIFEST"
  )
else
  echo "WHISPER_FILE was not set; skipping Whisper packaging."
  echo "Set WHISPER_FILE=/path/to/large-v3.pt if deployment machines cannot download Whisper."
fi

echo
echo "Release assets are ready under: $OUT_DIR"
ls -lh "$OUT_DIR"
