#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
ZIP_DIR="${ZIP_DIR:-$DATA_ROOT/raw/zips}"
ANNOTATION_DIR="${ANNOTATION_DIR:-$DATA_ROOT/annotations}"
ZENODO_RECORD="${ZENODO_RECORD:-15128981}"
ZENODO_BASE_URL="${ZENODO_BASE_URL:-https://zenodo.org/records/$ZENODO_RECORD/files}"
DOWNLOAD_JOBS="${DOWNLOAD_JOBS:-8}"
DOWNLOAD_SPLITS="${DOWNLOAD_SPLITS:-8}"
PARALLEL_DOWNLOADS="${PARALLEL_DOWNLOADS:-1}"

mkdir -p "$ZIP_DIR" "$ANNOTATION_DIR"

download_file() {
  local filename="$1"
  local output="$2"
  local url="$ZENODO_BASE_URL/$filename?download=1"

  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" ]]; then
    echo "Exists: $output"
    return 0
  fi

  echo "Downloading: $url"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --continue=true \
      --max-connection-per-server="$DOWNLOAD_JOBS" \
      --split="$DOWNLOAD_SPLITS" \
      --min-split-size=16M \
      --file-allocation=none \
      --retry-wait=10 \
      --max-tries=0 \
      --timeout=60 \
      --connect-timeout=30 \
      --summary-interval=30 \
      --dir "$(dirname "$output")" \
      --out "$(basename "$output")" \
      "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 8 --retry-delay 5 --connect-timeout 30 -C - -o "$output" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$output" "$url"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
}

verify_md5() {
  local expected="$1"
  local file="$2"

  if [[ ! -f "$file" ]]; then
    echo "Missing file for md5 check: $file" >&2
    return 1
  fi

  local actual
  actual="$(md5sum "$file" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "MD5 mismatch: $file" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    return 1
  fi
  echo "MD5 ok: $file"
}

ensure_file() {
  local filename="$1"
  local output="$2"
  local expected_md5="${3:-}"

  if [[ -s "$output" && "${SKIP_MD5:-0}" != "1" && -n "$expected_md5" ]]; then
    if verify_md5 "$expected_md5" "$output"; then
      return 0
    fi

    local bad_output="$output.bad-$(date +%Y%m%d-%H%M%S)"
    echo "Moving bad file to: $bad_output" >&2
    mv "$output" "$bad_output"
  fi

  download_file "$filename" "$output"

  if [[ "${SKIP_MD5:-0}" != "1" && -n "$expected_md5" ]]; then
    verify_md5 "$expected_md5" "$output"
  fi
}

FILES=(
  "Annotation_Subset.csv"
  "children.zip"
  "disabled.zip"
  "elderly.zip"
  "low_income.zip"
  "single_parent.zip"
  "women.zip"
)

declare -A MD5S=(
  ["Annotation_Subset.csv"]="1ce70c163654783fcf5bf1e17c600d54"
  ["children.zip"]="5d0af8a8ee07a4a934b2eccad8f7048f"
  ["disabled.zip"]="19556568f0ff132d307f5385c7890a62"
  ["elderly.zip"]="e85deaa3274aaf734318e65982bcbb48"
  ["low_income.zip"]="c76123e2d9de042735e8798562a462ae"
  ["single_parent.zip"]="bd1cce330f022b468a613ade8e3f8d1c"
  ["women.zip"]="763bb232496912807eb9a146398735ec"
)

if [[ -n "${DATASET_FILES:-}" ]]; then
  read -r -a FILES <<< "$DATASET_FILES"
fi

echo "Project root: $PROJECT_ROOT"
echo "Annotation dir: $ANNOTATION_DIR"
echo "Zip dir: $ZIP_DIR"
echo "Zenodo base URL: $ZENODO_BASE_URL"
echo "Parallel downloads: $PARALLEL_DOWNLOADS"
echo "Connections per file: $DOWNLOAD_JOBS"
echo

download_one() {
  local filename="$1"
  if [[ "$filename" == "Annotation_Subset.csv" ]]; then
    output="$ANNOTATION_DIR/$filename"
  else
    output="$ZIP_DIR/$filename"
  fi

  ensure_file "$filename" "$output" "${MD5S[$filename]}"
}

export PROJECT_ROOT DATA_ROOT ZIP_DIR ANNOTATION_DIR ZENODO_BASE_URL DOWNLOAD_JOBS DOWNLOAD_SPLITS SKIP_MD5

if (( PARALLEL_DOWNLOADS > 1 )); then
  for filename in "${FILES[@]}"; do
    download_one "$filename" &
    while (( "$(jobs -rp | wc -l)" >= PARALLEL_DOWNLOADS )); do
      wait -n
    done
  done
  wait
else
  for filename in "${FILES[@]}"; do
    download_one "$filename"
  done
fi

if [[ -f "$PROJECT_ROOT/third_party/PCLMM/data/Annotation_Link.csv" ]]; then
  cp "$PROJECT_ROOT/third_party/PCLMM/data/Annotation_Link.csv" "$ANNOTATION_DIR/Annotation_Link.csv"
  echo "Copied official Annotation_Link.csv to $ANNOTATION_DIR"
fi

echo
echo "Dataset download complete."
echo "Annotations: $ANNOTATION_DIR"
echo "Zips:        $ZIP_DIR"
