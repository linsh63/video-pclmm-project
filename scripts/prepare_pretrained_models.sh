#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

PRETRAINED_DIR="${PRETRAINED_DIR:-$PROJECT_ROOT/pretrained}"
VIT_REPO="${VIT_REPO:-google/vit-base-patch16-224-in21k}"
BERT_REPO="${BERT_REPO:-google-bert/bert-base-chinese}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/outputs/cache/huggingface}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/outputs/cache}"
WHISPER_CACHE_DIR="${WHISPER_CACHE_DIR:-$XDG_CACHE_HOME/whisper}"
FER_VT_DIR="${FER_VT_DIR:-$PRETRAINED_DIR/FER-VT}"

VIT_DIR="$PRETRAINED_DIR/googlevit-base-patch16-224-in21k"
BERT_DIR="$PRETRAINED_DIR/bert_chinese"

export HF_HOME
export XDG_CACHE_HOME
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN=/path/to/python if the conda env lives elsewhere." >&2
  exit 1
fi

mkdir -p "$PRETRAINED_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$WHISPER_CACHE_DIR"

download_file() {
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" ]]; then
    echo "Exists: $output"
    return 0
  fi
  echo "Downloading: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 3 --connect-timeout 30 -o "$output.tmp" "$url"
    mv "$output.tmp" "$output"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$output.tmp" "$url"
    mv "$output.tmp" "$output"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
}

echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "Pretrained dir: $PRETRAINED_DIR"
echo "HF_HOME: $HF_HOME"
echo "HF_ENDPOINT: $HF_ENDPOINT"
echo "MODEL_SOURCE: $MODEL_SOURCE"
echo

echo "[1/4] Downloading ViT model..."
if [[ "$MODEL_SOURCE" == "modelscope" ]]; then
  VIT_BASE_URL="https://modelscope.cn/models/${VIT_REPO}/resolve/master"
  download_file "$VIT_BASE_URL/config.json" "$VIT_DIR/config.json"
  download_file "$VIT_BASE_URL/preprocessor_config.json" "$VIT_DIR/preprocessor_config.json"
  download_file "$VIT_BASE_URL/model.safetensors" "$VIT_DIR/model.safetensors"

  "$PYTHON_BIN" - <<PY
from pathlib import Path
from transformers import ViTFeatureExtractor, ViTModel

out_dir = Path(r"""$VIT_DIR""")
ViTFeatureExtractor.from_pretrained(out_dir)
ViTModel.from_pretrained(out_dir)
print(f"Verified local ViT at {out_dir}")
PY
elif [[ "${DIRECT_VIT_DOWNLOAD:-1}" == "1" ]]; then
  VIT_BASE_URL="${HF_ENDPOINT%/}/${VIT_REPO}/resolve/main"
  download_file "$VIT_BASE_URL/config.json" "$VIT_DIR/config.json"
  download_file "$VIT_BASE_URL/preprocessor_config.json" "$VIT_DIR/preprocessor_config.json"
  download_file "$VIT_BASE_URL/model.safetensors" "$VIT_DIR/model.safetensors"

  "$PYTHON_BIN" - <<PY
from pathlib import Path
from transformers import ViTFeatureExtractor, ViTModel

out_dir = Path(r"""$VIT_DIR""")
ViTFeatureExtractor.from_pretrained(out_dir)
ViTModel.from_pretrained(out_dir)
print(f"Verified local ViT at {out_dir}")
PY
else
  "$PYTHON_BIN" - <<PY
from pathlib import Path
from transformers import ViTFeatureExtractor, ViTModel

repo = r"""$VIT_REPO"""
out_dir = Path(r"""$VIT_DIR""")
cache_dir = r"""$TRANSFORMERS_CACHE"""
out_dir.mkdir(parents=True, exist_ok=True)

extractor = ViTFeatureExtractor.from_pretrained(repo, cache_dir=cache_dir)
try:
    model = ViTModel.from_pretrained(repo, cache_dir=cache_dir, use_safetensors=True)
except OSError as exc:
    try:
        model = ViTModel.from_pretrained(repo, cache_dir=cache_dir, use_safetensors=False)
    except OSError as bin_exc:
        message = str(bin_exc)
        if "TensorFlow weights" not in message and "from_tf=True" not in message:
            raise
        print("PyTorch ViT weights were not found; trying TensorFlow-to-PyTorch conversion with from_tf=True.")
        try:
            model = ViTModel.from_pretrained(repo, cache_dir=cache_dir, from_tf=True)
        except Exception as tf_exc:
            raise RuntimeError(
                "ViT PyTorch/safetensors weights were not available from this endpoint, and TF conversion failed. "
                "Try rerunning with HF_ENDPOINT=https://huggingface.co, or clear the cached partial model under "
                f"{cache_dir!r}, or install tensorflow-cpu and rerun."
            ) from tf_exc
    except Exception as other_exc:
        raise RuntimeError(
            "ViT safetensors load failed, and fallback to pytorch_model.bin did not complete."
        ) from other_exc
extractor.save_pretrained(out_dir)
model.save_pretrained(out_dir)
print(f"Saved ViT to {out_dir}")
PY
fi

echo
echo "[2/4] Downloading BERT Chinese model..."
if [[ "$MODEL_SOURCE" == "modelscope" ]]; then
  BERT_BASE_URL="https://modelscope.cn/models/${BERT_REPO}/resolve/master"
  download_file "$BERT_BASE_URL/config.json" "$BERT_DIR/config.json"
  download_file "$BERT_BASE_URL/vocab.txt" "$BERT_DIR/vocab.txt"
  download_file "$BERT_BASE_URL/tokenizer_config.json" "$BERT_DIR/tokenizer_config.json"
  download_file "$BERT_BASE_URL/model.safetensors" "$BERT_DIR/model.safetensors"

  "$PYTHON_BIN" - <<PY
from pathlib import Path
from transformers import BertTokenizer, BertModel

out_dir = Path(r"""$BERT_DIR""")
BertTokenizer.from_pretrained(out_dir)
BertModel.from_pretrained(out_dir)
print(f"Verified local BERT Chinese at {out_dir}")
PY
else
  "$PYTHON_BIN" - <<PY
from pathlib import Path
from transformers import BertTokenizer, BertModel

repo = r"""$BERT_REPO"""
out_dir = Path(r"""$BERT_DIR""")
cache_dir = r"""$TRANSFORMERS_CACHE"""
out_dir.mkdir(parents=True, exist_ok=True)

tokenizer = BertTokenizer.from_pretrained(repo, cache_dir=cache_dir)
model = BertModel.from_pretrained(repo, cache_dir=cache_dir)
tokenizer.save_pretrained(out_dir)
model.save_pretrained(out_dir)
print(f"Saved BERT Chinese to {out_dir}")
PY
fi

if [[ "${SKIP_WHISPER:-0}" != "1" ]]; then
  echo
  echo "[3/4] Downloading Whisper large model..."
  "$PYTHON_BIN" - <<PY
from pathlib import Path
import whisper

download_root = Path(r"""$WHISPER_CACHE_DIR""")
download_root.mkdir(parents=True, exist_ok=True)

def download_large():
    if hasattr(whisper, "_MODELS") and "large" in whisper._MODELS:
        return whisper._download(whisper._MODELS["large"], str(download_root), False)
    model = whisper.load_model("large", download_root=str(download_root))
    return type(model).__name__

try:
    path = download_large()
except RuntimeError as exc:
    message = str(exc)
    if "SHA256 checksum" not in message:
        raise
    bad_name = "large-v3.pt"
    if hasattr(whisper, "_MODELS") and "large" in whisper._MODELS:
        bad_name = whisper._MODELS["large"].rsplit("/", 1)[-1]
    bad_path = download_root / bad_name
    if bad_path.exists():
        moved_path = bad_path.with_suffix(bad_path.suffix + ".bad")
        suffix = 1
        while moved_path.exists():
            moved_path = bad_path.with_suffix(bad_path.suffix + f".bad{suffix}")
            suffix += 1
        bad_path.rename(moved_path)
        print(f"Moved corrupt Whisper file to {moved_path}")
    path = download_large()

print(f"Saved Whisper large to {path}")
PY
else
  echo
  echo "[3/4] Skipping Whisper large because SKIP_WHISPER=1"
fi

if [[ "${SKIP_FER_VT:-0}" != "1" ]]; then
  echo
  echo "[4/4] Cloning FER-VT..."
  if [[ -d "$FER_VT_DIR/.git" ]]; then
    git -C "$FER_VT_DIR" pull --ff-only
  elif [[ -e "$FER_VT_DIR" ]]; then
    echo "FER-VT path already exists and is not a git repo: $FER_VT_DIR"
  else
    git clone --depth 1 https://github.com/ZBigFish/FER-VT "$FER_VT_DIR"
  fi

  if [[ -d "$FER_VT_DIR/model" ]]; then
    ln -sfn "$FER_VT_DIR/model" "$PROJECT_ROOT/third_party/PCLMM/code/model"
    echo "Linked FER-VT model package to third_party/PCLMM/code/model"
  else
    echo "FER-VT cloned, but no model/ directory was found at $FER_VT_DIR/model" >&2
    echo "Inspect the repository layout before running extract_face_fervt.py." >&2
  fi
else
  echo
  echo "[4/4] Skipping FER-VT because SKIP_FER_VT=1"
fi

echo
echo "Done. Key paths:"
echo "  ViT:     $VIT_DIR"
echo "  BERT:    $BERT_DIR"
echo "  Whisper: $WHISPER_CACHE_DIR"
echo "  FER-VT:  $FER_VT_DIR"
