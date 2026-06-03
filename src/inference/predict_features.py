from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.feature_dataset import load_face_feature, load_feature_array  # noqa: E402
from src.models.fusion_model import FusionModelConfig, load_fusion_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one video from already extracted PCLMM features."
    )
    parser.add_argument("--checkpoint", default="outputs/checkpoints/multi_modal_cross_attention_model.pth")
    parser.add_argument("--file-id", default=None, help="Feature basename, for example disabled90.")
    parser.add_argument("--text-dir", default="features/TEXT_features")
    parser.add_argument("--audio-dir", default="features/AUDIO_features")
    parser.add_argument("--video-dir", default="features/VIT_features")
    parser.add_argument("--face-dir", default="features/extracted_features_without_xml")
    parser.add_argument("--text-path", default=None)
    parser.add_argument("--audio-path", default=None)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--face-path", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--target-length", type=int, default=512)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(device)


def find_feature_path(folder: str | Path, file_id: str) -> Path:
    folder = Path(folder)
    direct_path = folder / f"{file_id}.p"
    if direct_path.is_file():
        return direct_path

    matches = sorted(folder.rglob(f"{file_id}.p"))
    if not matches:
        raise FileNotFoundError(f"Could not find {file_id}.p under {folder}")
    if len(matches) > 1:
        match_list = ", ".join(str(path) for path in matches[:5])
        raise RuntimeError(f"Multiple feature files matched {file_id}.p: {match_list}")
    return matches[0]


def resolve_feature_paths(args: argparse.Namespace) -> dict[str, Path]:
    explicit_paths = {
        "text": args.text_path,
        "audio": args.audio_path,
        "video": args.video_path,
        "face": args.face_path,
    }

    if any(explicit_paths.values()):
        missing = [name for name, path in explicit_paths.items() if not path]
        if missing:
            raise ValueError(
                "When using explicit feature paths, all four paths are required. "
                f"Missing: {missing}"
            )
        return {name: Path(path) for name, path in explicit_paths.items() if path}

    if not args.file_id:
        raise ValueError("Provide either --file-id or all four explicit feature paths.")

    return {
        "text": find_feature_path(args.text_dir, args.file_id),
        "audio": find_feature_path(args.audio_dir, args.file_id),
        "video": find_feature_path(args.video_dir, args.file_id),
        "face": find_feature_path(args.face_dir, args.file_id),
    }


def tensor_from_feature(path: Path, feature_type: str) -> torch.Tensor:
    if feature_type == "face":
        array = load_face_feature(path)
    else:
        array = load_feature_array(path)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(array).float().unsqueeze(0)


class FusionFeaturePredictor:
    def __init__(
        self,
        checkpoint: str | Path,
        threshold: float = 0.5,
        device: str | torch.device = "cpu",
        target_length: int = 512,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.threshold = float(threshold)
        self.device = torch.device(device)
        self.model = load_fusion_model(
            self.checkpoint,
            config=FusionModelConfig(target_length=target_length),
            device=self.device,
        )

    def predict_paths(
        self,
        text_path: str | Path,
        audio_path: str | Path,
        video_path: str | Path,
        face_path: str | Path,
        threshold: float | None = None,
    ) -> dict:
        threshold = self.threshold if threshold is None else float(threshold)
        text = tensor_from_feature(Path(text_path), "text").to(self.device)
        audio = tensor_from_feature(Path(audio_path), "audio").to(self.device)
        video = tensor_from_feature(Path(video_path), "video").to(self.device)
        face = tensor_from_feature(Path(face_path), "face").to(self.device)

        with torch.no_grad():
            logit = self.model(text, audio, video, face).squeeze().item()
            score = float(torch.sigmoid(torch.tensor(logit)).item())

        return {
            "is_normal": bool(score < threshold),
            "score": score,
            "threshold": float(threshold),
            "prediction": int(score >= threshold),
            "label_semantics": {
                "0": "normal_or_non_pcl",
                "1": "abnormal_or_pcl",
                "score": "probability_of_label_1",
                "is_normal": "score < threshold",
            },
        }


def predict_feature_paths(
    checkpoint: str | Path,
    text_path: str | Path,
    audio_path: str | Path,
    video_path: str | Path,
    face_path: str | Path,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
    target_length: int = 512,
) -> dict:
    predictor = FusionFeaturePredictor(
        checkpoint=checkpoint,
        threshold=threshold,
        device=device,
        target_length=target_length,
    )
    return predictor.predict_paths(text_path, audio_path, video_path, face_path)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    paths = resolve_feature_paths(args)

    predictor = FusionFeaturePredictor(
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        device=device,
        target_length=args.target_length,
    )
    result = predictor.predict_paths(
        text_path=paths["text"],
        audio_path=paths["audio"],
        video_path=paths["video"],
        face_path=paths["face"],
    )
    result["file_id"] = args.file_id
    result["feature_paths"] = {name: str(path) for name, path in paths.items()}

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
