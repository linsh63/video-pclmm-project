#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data4/songxinshuai/conda/envs/video-pclmm/bin/python}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/data/processed/video_manifest.csv}"
SMOKE_ROOT="${SMOKE_ROOT:-$PROJECT_ROOT/outputs/smoke_videos}"
GROUP="${GROUP:-disabled}"
N="${N:-1}"

mkdir -p "$SMOKE_ROOT/$GROUP"

"$PYTHON_BIN" - <<PY
from pathlib import Path
import os
import pandas as pd

manifest = Path(r"""$MANIFEST""")
smoke_root = Path(r"""$SMOKE_ROOT""")
group = r"""$GROUP"""
n = int(r"""$N""")

df = pd.read_csv(manifest)
df = df[df["group"] == group].sort_values("duration_sec").head(n)
if df.empty:
    raise SystemExit(f"No videos found for group={group}")

target_dir = smoke_root / group
target_dir.mkdir(parents=True, exist_ok=True)

for _, row in df.iterrows():
    src = Path(row["video_path"])
    dst = target_dir / src.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)
    print(f"{row['video_id']}: {row['duration_sec']:.2f}s -> {dst}")
PY

echo "Smoke video root: $SMOKE_ROOT"
