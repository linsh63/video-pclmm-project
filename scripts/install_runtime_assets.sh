#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="${ASSET_DIR:-$PROJECT_ROOT/outputs/release_assets}"
VERSION="${VERSION:-v0.1.0}"
CORE_ARCHIVE="${CORE_ARCHIVE:-$ASSET_DIR/video-pclmm-runtime-core-${VERSION}.tar.gz}"
WHISPER_DIR="${WHISPER_DIR:-$PROJECT_ROOT/outputs/cache/whisper}"

if [[ ! -f "$CORE_ARCHIVE" ]]; then
  echo "Core runtime archive not found: $CORE_ARCHIVE" >&2
  echo "Download the GitHub Release assets first, or set CORE_ARCHIVE=/path/to/archive." >&2
  exit 1
fi

echo "Extracting core runtime assets from: $CORE_ARCHIVE"
tar -xzf "$CORE_ARCHIVE" -C "$PROJECT_ROOT"

shopt -s nullglob
whisper_parts=("$ASSET_DIR"/whisper-large-v3.pt.part-*)
if (( ${#whisper_parts[@]} > 0 )); then
  mkdir -p "$WHISPER_DIR"
  echo "Reconstructing Whisper model at: $WHISPER_DIR/large-v3.pt"
  cat "${whisper_parts[@]}" > "$WHISPER_DIR/large-v3.pt"

  if [[ -f "$ASSET_DIR/whisper-large-v3.pt.sha256" ]]; then
    (
      cd "$WHISPER_DIR"
      sha256sum -c "$ASSET_DIR/whisper-large-v3.pt.sha256"
    )
  fi
else
  echo "No Whisper split parts found under $ASSET_DIR; skipping Whisper install."
fi

echo "Runtime assets installed."
