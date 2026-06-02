from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PCLMM video manifest.")
    parser.add_argument("--annotation-file", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--groups", nargs="*", default=None)
    return parser.parse_args()


def has_audio_stream(video_path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    return bool(payload.get("streams"))


def read_video_meta(video_path: Path) -> dict[str, float | int | bool]:
    cap = cv2.VideoCapture(str(video_path))
    opened = cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    cap.release()

    duration = frame_count / fps if fps else 0.0
    return {
        "can_open": opened,
        "frame_count": frame_count,
        "fps": fps,
        "duration_sec": duration,
        "width": width,
        "height": height,
        "has_audio": has_audio_stream(video_path),
    }


def main() -> None:
    args = parse_args()
    annotations = pd.read_csv(args.annotation_file)
    annotations = annotations.set_index("File", drop=False)

    rows = []
    groups = set(args.groups) if args.groups else None
    video_paths = sorted(args.video_root.glob("*/*.mp4"))
    for video_path in video_paths:
        group = video_path.parent.name
        if groups and group not in groups:
            continue

        video_id = video_path.stem
        ann = annotations.loc[video_id] if video_id in annotations.index else None
        meta = read_video_meta(video_path)

        rows.append(
            {
                "video_id": video_id,
                "group": group,
                "video_path": str(video_path.resolve()),
                "label": None if ann is None else int(ann["Annotation"]),
                "subset": None if ann is None or "Subset" not in ann else ann["Subset"],
                "title": None if ann is None or "Title" not in ann else ann["Title"],
                "source_link": None if ann is None or "Source_Link" not in ann else ann["Source_Link"],
                "annotation_source": str(args.annotation_file.resolve()),
                **meta,
            }
        )

    manifest = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    print(f"wrote {args.output}")
    print(f"videos: {len(manifest)}")
    if not manifest.empty:
        print("labels:")
        print(manifest["label"].value_counts(dropna=False).sort_index().to_string())
        print("groups:")
        print(manifest["group"].value_counts().to_string())
        print("subsets:")
        print(manifest["subset"].value_counts(dropna=False).to_string())
        print(f"missing labels: {manifest['label'].isna().sum()}")
        print(f"no audio: {(~manifest['has_audio']).sum()}")
        print(f"cannot open: {(~manifest['can_open']).sum()}")


if __name__ == "__main__":
    main()
