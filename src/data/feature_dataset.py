from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pickle

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FeatureRecord:
    file_id: str
    label: int
    subset: str
    text_path: Path
    audio_path: Path
    video_path: Path
    face_path: Path


def index_feature_files(folder: str | Path, extension: str = ".p") -> dict[str, Path]:
    folder = Path(folder)
    return {path.stem: path for path in folder.rglob(f"*{extension}") if path.is_file()}


def build_feature_records(
    annotation_csv: str | Path,
    text_dir: str | Path,
    audio_dir: str | Path,
    video_dir: str | Path,
    face_dir: str | Path,
    subset: str = "test",
    file_prefixes: Iterable[str] | None = None,
) -> tuple[list[FeatureRecord], dict[str, int]]:
    annotations = pd.read_csv(annotation_csv)
    required_columns = {"File", "Annotation", "Subset"}
    missing_columns = required_columns - set(annotations.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in annotation CSV: {sorted(missing_columns)}")

    if subset.lower() != "all":
        annotations = annotations[annotations["Subset"].str.lower() == subset.lower()]

    prefixes = tuple(file_prefixes or ())
    if prefixes:
        annotations = annotations[
            annotations["File"].astype(str).apply(lambda value: value.startswith(prefixes))
        ]

    text_files = index_feature_files(text_dir)
    audio_files = index_feature_files(audio_dir)
    video_files = index_feature_files(video_dir)
    face_files = index_feature_files(face_dir)

    records: list[FeatureRecord] = []
    missing = {"annotation": 0, "text": 0, "audio": 0, "video": 0, "face": 0}

    for row in annotations.itertuples(index=False):
        file_id = str(getattr(row, "File"))
        text_path = text_files.get(file_id)
        audio_path = audio_files.get(file_id)
        video_path = video_files.get(file_id)
        face_path = face_files.get(file_id)

        if not text_path:
            missing["text"] += 1
        if not audio_path:
            missing["audio"] += 1
        if not video_path:
            missing["video"] += 1
        if not face_path:
            missing["face"] += 1
        if not all([text_path, audio_path, video_path, face_path]):
            continue

        records.append(
            FeatureRecord(
                file_id=file_id,
                label=int(getattr(row, "Annotation")),
                subset=str(getattr(row, "Subset")).strip().lower(),
                text_path=text_path,
                audio_path=audio_path,
                video_path=video_path,
                face_path=face_path,
            )
        )

    missing["annotation"] = int(len(annotations) - len(records))
    records.sort(key=lambda record: record.file_id)
    return records, missing


class PCLMMFeatureDataset(Dataset):
    def __init__(self, records: list[FeatureRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        return {
            "file_id": record.file_id,
            "text": torch.from_numpy(load_feature_array(record.text_path)).float(),
            "audio": torch.from_numpy(load_feature_array(record.audio_path)).float(),
            "video": torch.from_numpy(load_feature_array(record.video_path)).float(),
            "face": torch.from_numpy(load_face_feature(record.face_path)).float(),
            "label": torch.tensor(record.label, dtype=torch.float32),
        }


def load_pickle(path: str | Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_feature_array(path: str | Path) -> np.ndarray:
    loaded = load_pickle(path)
    if isinstance(loaded, dict) and "features" in loaded:
        loaded = loaded["features"]
    return np.asarray(loaded, dtype=np.float32)


def load_face_feature(path: str | Path) -> np.ndarray:
    loaded = load_pickle(path)
    if isinstance(loaded, dict):
        if "all_zero" in loaded:
            return np.zeros((1, 1, 4, 192), dtype=np.float32)
        if "features" in loaded:
            loaded = loaded["features"]

    array = np.asarray(loaded, dtype=np.float32)
    if array.ndim == 2 and array.shape[-1] == 192:
        array = np.repeat(array[:, None, None, :], repeats=4, axis=2)
    return array


def collate_feature_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "file_id": [str(item["file_id"]) for item in batch],
        "text": pad_sequence([item["text"] for item in batch], batch_first=True),
        "audio": pad_sequence([item["audio"] for item in batch], batch_first=True),
        "video": pad_sequence([item["video"] for item in batch], batch_first=True),
        "face": pad_sequence([item["face"] for item in batch], batch_first=True),
        "label": torch.stack([item["label"] for item in batch]),
    }
