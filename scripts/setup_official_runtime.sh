#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-$PROJECT_ROOT/outputs/official_runtime}"
VIDEO_ROOT="${VIDEO_ROOT:-$PROJECT_ROOT/data/raw/videos}"
FEATURE_ROOT="${FEATURE_ROOT:-$PROJECT_ROOT/features}"
TEMP_ROOT="${TEMP_ROOT:-$PROJECT_ROOT/temp}"

CODE_SRC="$PROJECT_ROOT/third_party/PCLMM/code"
CODE_DST="$RUNTIME_ROOT/code"

mkdir -p "$RUNTIME_ROOT" "$FEATURE_ROOT" "$TEMP_ROOT"
mkdir -p "$CODE_DST"

cp "$CODE_SRC"/*.py "$CODE_DST"/
cp "$PROJECT_ROOT/data/annotations/Annotation_Subset.csv" "$CODE_DST/Annotation.csv"
cp "$PROJECT_ROOT/src/extraction/mfcc_compatible.py" "$CODE_DST/MFCC.py"
cp "$PROJECT_ROOT/src/extraction/extract_audio_text_compatible.py" "$CODE_DST/extract_audio_text.py"

find "$CODE_DST" -maxdepth 1 -type f -name '*.py' -print0 |
  xargs -0 sed -i "s#/root/autodl-tmp#$RUNTIME_ROOT#g"

ln -sfn "$VIDEO_ROOT" "$RUNTIME_ROOT/PCLMM"
ln -sfn "$FEATURE_ROOT" "$RUNTIME_ROOT/features"
ln -sfn "$TEMP_ROOT" "$RUNTIME_ROOT/temp"
ln -sfn "$PROJECT_ROOT/pretrained/googlevit-base-patch16-224-in21k" "$CODE_DST/googlevit-base-patch16-224-in21k"
ln -sfn "$PROJECT_ROOT/pretrained/bert_chinese" "$CODE_DST/bert_chinese"
ln -sfn "$PROJECT_ROOT/pretrained/FER-VT/model" "$CODE_DST/model"

echo "Runtime root: $RUNTIME_ROOT"
echo "Runtime code: $CODE_DST"
echo "Video root:   $VIDEO_ROOT"
echo "Feature root: $FEATURE_ROOT"
echo "Temp root:    $TEMP_ROOT"
