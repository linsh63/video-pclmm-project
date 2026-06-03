from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.feature_dataset import (  # noqa: E402
    PCLMMFeatureDataset,
    build_feature_records,
    collate_feature_batch,
)
from src.models.fusion_model import FusionModelConfig, load_fusion_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PCLMM fusion checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/multi_modal_cross_attention_model.pth")
    parser.add_argument("--annotation", default="data/annotations/Annotation_Subset.csv")
    parser.add_argument("--text-dir", default="features/TEXT_features")
    parser.add_argument("--audio-dir", default="features/AUDIO_features")
    parser.add_argument("--video-dir", default="features/VIT_features")
    parser.add_argument("--face-dir", default="features/extracted_features_without_xml")
    parser.add_argument("--subset", default="test", choices=["train", "test", "all"])
    parser.add_argument("--file-prefix", action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--target-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--metrics-out", default="outputs/metrics/fusion_eval.json")
    parser.add_argument("--no-save-predictions", action="store_true")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(device)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(set(labels.tolist())) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "auc": safe_auc(labels, scores),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels,
            preds,
            labels=[0, 1],
            target_names=["normal_or_non_pcl", "abnormal_or_pcl"],
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
    }


def evaluate(args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    records, missing = build_feature_records(
        annotation_csv=args.annotation,
        text_dir=args.text_dir,
        audio_dir=args.audio_dir,
        video_dir=args.video_dir,
        face_dir=args.face_dir,
        subset=args.subset,
        file_prefixes=args.file_prefix,
    )
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise RuntimeError("No aligned feature records were found for evaluation.")

    dataset = PCLMMFeatureDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_feature_batch,
    )

    config = FusionModelConfig(target_length=args.target_length)
    model = load_fusion_model(args.checkpoint, config=config, device=device)
    criterion = nn.BCEWithLogitsLoss()

    all_file_ids: list[str] = []
    all_labels: list[float] = []
    all_scores: list[float] = []
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            video = batch["video"].to(device)
            face = batch["face"].to(device)
            labels = batch["label"].to(device)

            logits = model(text, audio, video, face).squeeze(1)
            loss = criterion(logits, labels)
            scores = torch.sigmoid(logits)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            all_file_ids.extend(batch["file_id"])
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_scores.extend(scores.detach().cpu().numpy().tolist())

    labels_np = np.asarray(all_labels, dtype=np.int64)
    scores_np = np.asarray(all_scores, dtype=np.float32)
    metrics = compute_metrics(labels_np, scores_np, args.threshold)
    preds_np = (scores_np >= args.threshold).astype(np.int64)

    result = {
        "checkpoint": str(args.checkpoint),
        "subset": args.subset,
        "file_prefix": args.file_prefix,
        "samples": int(len(records)),
        "missing": missing,
        "threshold": float(args.threshold),
        "label_semantics": {
            "0": "normal_or_non_pcl",
            "1": "abnormal_or_pcl",
            "score": "probability_of_label_1",
            "is_normal": "score < threshold",
        },
        "average_loss": float(total_loss / max(total_items, 1)),
        "metrics": metrics,
    }

    if not args.no_save_predictions:
        result["predictions"] = [
            {
                "file_id": file_id,
                "label": int(label),
                "score": float(score),
                "prediction": int(pred),
                "is_normal": bool(score < args.threshold),
            }
            for file_id, label, score, pred in zip(
                all_file_ids, labels_np, scores_np, preds_np
            )
        ]

    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = result["metrics"]
    print(f"checkpoint: {result['checkpoint']}")
    print(f"subset: {result['subset']}")
    print(f"samples: {result['samples']}")
    print(f"average_loss: {result['average_loss']:.6f}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")
    print(f"positive_f1: {metrics['positive_f1']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"auc: {metrics['auc']}")
    print(f"confusion_matrix: {metrics['confusion_matrix']}")
    print(f"metrics_out: {metrics_out}")


if __name__ == "__main__":
    main()
