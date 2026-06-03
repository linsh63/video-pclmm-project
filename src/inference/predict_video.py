from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUP = "single"
EXTRACTION_STEPS = (
    "extract_video_vit.py",
    "extract_audio_wav.py",
    "MFCC.py",
    "extract_audio_text.py",
    "BERT.py",
    "extract_face_fervt.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-video feature extraction and fusion prediction."
    )
    parser.add_argument("--video", required=True, help="Input .mp4 video path.")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/multi_modal_cross_attention_model.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--target-length", type=int, default=512)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--runtime-root", default="outputs/inference/runtime")
    parser.add_argument("--cache-root", default="outputs/inference/cache")
    parser.add_argument("--file-id", default=None)
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--copy-video", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extraction-mode", choices=["sequential", "parallel"], default="sequential")
    parser.add_argument(
        "--step-cuda-visible-devices",
        action="append",
        default=[],
        help="Per-step GPU override, for example extract_video_vit.py=4.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def sanitize_file_id(value: str) -> str:
    value = Path(value).stem
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    if not value:
        raise ValueError("Could not derive a valid file id from the video path.")
    return value


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def prepare_video_link(
    source_video: Path,
    video_root: Path,
    group: str,
    file_id: str,
    copy_video: bool = False,
) -> Path:
    if source_video.suffix.lower() != ".mp4":
        raise ValueError(f"Only .mp4 videos are supported for now: {source_video}")
    if not source_video.is_file():
        raise FileNotFoundError(f"Input video not found: {source_video}")

    video_dir = video_root / group
    video_dir.mkdir(parents=True, exist_ok=True)
    target = video_dir / f"{file_id}.mp4"

    if target.exists() or target.is_symlink():
        target.unlink()

    if copy_video:
        shutil.copy2(source_video, target)
    else:
        target.symlink_to(source_video)
    return target


def setup_runtime(runtime_root: Path, video_root: Path, feature_root: Path, temp_root: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "RUNTIME_ROOT": str(runtime_root),
            "VIDEO_ROOT": str(video_root),
            "FEATURE_ROOT": str(feature_root),
            "TEMP_ROOT": str(temp_root),
        }
    )
    subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/setup_official_runtime.sh")],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def elapsed_seconds(started_at: float) -> float:
    return round(time.perf_counter() - started_at, 4)


def expected_feature_paths(feature_root: Path, temp_root: Path, group: str, file_id: str) -> dict[str, Path]:
    return {
        "video": feature_root / "VIT_features" / group / f"{file_id}.p",
        "audio": feature_root / "AUDIO_features" / group / f"{file_id}.p",
        "text": feature_root / "TEXT_features" / group / f"{file_id}.p",
        "face": feature_root / "extracted_features_without_xml" / group / f"{file_id}.p",
        "wav": temp_root / "WAV" / group / f"{file_id}.wav",
        "txt": temp_root / "TXT" / group / f"{file_id}.txt",
    }


def parse_step_cuda_visible_devices(values: list[str] | tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Invalid step CUDA mapping: {item!r}")
            step_name, cuda_devices = item.split("=", 1)
            step_name = step_name.strip()
            cuda_devices = cuda_devices.strip()
            if not step_name.endswith(".py"):
                step_name = f"{step_name}.py"
            if step_name not in EXTRACTION_STEPS:
                raise ValueError(f"Unknown extraction step in CUDA mapping: {step_name}")
            result[step_name] = cuda_devices
    return result


def step_cuda_devices(
    script_name: str,
    default_cuda_visible_devices: str | None,
    step_cuda_visible_devices: dict[str, str] | None,
) -> str | None:
    if step_cuda_visible_devices and script_name in step_cuda_visible_devices:
        return step_cuda_visible_devices[script_name]
    return default_cuda_visible_devices


def run_step(
    script_name: str,
    runtime_root: Path,
    cuda_visible_devices: str | None,
    python_bin: str,
) -> float:
    started_at = time.perf_counter()
    env = os.environ.copy()
    env.update(
        {
            "RUNTIME_ROOT": str(runtime_root),
            "PYTHON_BIN": python_bin,
        }
    )
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts/run_official_step.sh"), script_name],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )
    return elapsed_seconds(started_at)


def run_or_skip_step(
    script_name: str,
    outputs: list[Path],
    runtime_root: Path,
    default_cuda_visible_devices: str | None,
    step_cuda_visible_devices: dict[str, str] | None,
    python_bin: str,
    skip_existing: bool,
) -> tuple[float, bool]:
    if skip_existing and all(path.exists() for path in outputs):
        print(f"Skipping {script_name}; expected output already exists.")
        return 0.0, True

    duration = run_step(
        script_name=script_name,
        runtime_root=runtime_root,
        cuda_visible_devices=step_cuda_devices(
            script_name,
            default_cuda_visible_devices,
            step_cuda_visible_devices,
        ),
        python_bin=python_bin,
    )
    return duration, False


def ensure_features(
    runtime_root: Path,
    feature_root: Path,
    temp_root: Path,
    group: str,
    file_id: str,
    cuda_visible_devices: str | None,
    step_cuda_visible_devices: dict[str, str] | None,
    python_bin: str,
    skip_existing: bool,
    extraction_mode: str,
) -> tuple[dict[str, Path], dict[str, float | bool | str]]:
    paths = expected_feature_paths(feature_root, temp_root, group, file_id)
    step_outputs = {
        "extract_video_vit.py": [paths["video"]],
        "extract_audio_wav.py": [paths["wav"]],
        "MFCC.py": [paths["audio"]],
        "extract_audio_text.py": [paths["txt"]],
        "BERT.py": [paths["text"]],
        "extract_face_fervt.py": [paths["face"]],
    }

    timings: dict[str, float | bool | str] = {"extraction_mode": extraction_mode}

    def record_step(script_name: str) -> None:
        duration, skipped = run_or_skip_step(
            script_name=script_name,
            outputs=step_outputs[script_name],
            runtime_root=runtime_root,
            default_cuda_visible_devices=cuda_visible_devices,
            step_cuda_visible_devices=step_cuda_visible_devices,
            python_bin=python_bin,
            skip_existing=skip_existing,
        )
        timings[script_name] = duration
        timings[f"{script_name}:skipped"] = skipped

    if extraction_mode == "sequential":
        for script_name in EXTRACTION_STEPS:
            record_step(script_name)
    elif extraction_mode == "parallel":
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(record_step, "extract_video_vit.py"),
                executor.submit(run_audio_branch, record_step),
                executor.submit(record_step, "extract_face_fervt.py"),
            ]
            for future in as_completed(futures):
                future.result()
    else:
        raise ValueError(f"Unsupported extraction mode: {extraction_mode}")

    missing = [name for name in ("video", "audio", "text", "face") if not paths[name].exists()]
    if missing:
        detail = {name: str(paths[name]) for name in missing}
        raise RuntimeError(f"Feature extraction did not produce all required features: {detail}")
    return paths, timings


def run_audio_branch(record_step) -> None:
    record_step("extract_audio_wav.py")
    with ThreadPoolExecutor(max_workers=2) as executor:
        mfcc_future = executor.submit(record_step, "MFCC.py")
        asr_future = executor.submit(record_step, "extract_audio_text.py")
        mfcc_future.result()
        asr_future.result()
    record_step("BERT.py")


def prepare_video_features(
    source_video: str | Path,
    runtime_root: str | Path,
    cache_root: str | Path,
    group: str = DEFAULT_GROUP,
    file_id: str | None = None,
    cuda_visible_devices: str | None = None,
    step_cuda_visible_devices: dict[str, str] | None = None,
    copy_video: bool = False,
    skip_existing: bool = True,
    extraction_mode: str = "sequential",
    python_bin: str | None = None,
) -> tuple[str, dict[str, Path], dict[str, float | bool | str]]:
    timings: dict[str, float | bool | str] = {}
    source_video = resolve_path(source_video)
    runtime_root = resolve_path(runtime_root)
    cache_root = resolve_path(cache_root)
    video_root = cache_root / "videos"
    feature_root = cache_root / "features"
    temp_root = cache_root / "temp"
    file_id = sanitize_file_id(file_id or source_video.stem)

    started_at = time.perf_counter()
    prepare_video_link(
        source_video=source_video,
        video_root=video_root,
        group=group,
        file_id=file_id,
        copy_video=copy_video,
    )
    timings["prepare_video_link"] = elapsed_seconds(started_at)

    started_at = time.perf_counter()
    setup_runtime(
        runtime_root=runtime_root,
        video_root=video_root,
        feature_root=feature_root,
        temp_root=temp_root,
    )
    timings["setup_runtime"] = elapsed_seconds(started_at)

    started_at = time.perf_counter()
    feature_paths, extraction_timings = ensure_features(
        runtime_root=runtime_root,
        feature_root=feature_root,
        temp_root=temp_root,
        group=group,
        file_id=file_id,
        cuda_visible_devices=cuda_visible_devices,
        step_cuda_visible_devices=step_cuda_visible_devices,
        python_bin=python_bin or sys.executable,
        skip_existing=skip_existing,
        extraction_mode=extraction_mode,
    )
    timings.update(extraction_timings)
    timings["ensure_features_total"] = elapsed_seconds(started_at)
    return file_id, feature_paths, timings


def predict_video(args: argparse.Namespace) -> dict:
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    checkpoint = resolve_path(args.checkpoint)
    file_id, feature_paths, timings = prepare_video_features(
        source_video=args.video,
        runtime_root=args.runtime_root,
        cache_root=args.cache_root,
        group=args.group,
        file_id=args.file_id,
        cuda_visible_devices=args.cuda_visible_devices,
        step_cuda_visible_devices=parse_step_cuda_visible_devices(args.step_cuda_visible_devices),
        copy_video=args.copy_video,
        skip_existing=args.skip_existing,
        extraction_mode=args.extraction_mode,
        python_bin=sys.executable,
    )

    from src.inference.predict_features import FusionFeaturePredictor

    predictor = FusionFeaturePredictor(
        checkpoint=checkpoint,
        threshold=args.threshold,
        device=resolve_device(args.device),
        target_length=args.target_length,
    )
    started_at = time.perf_counter()
    result = predictor.predict_paths(
        text_path=feature_paths["text"],
        audio_path=feature_paths["audio"],
        video_path=feature_paths["video"],
        face_path=feature_paths["face"],
    )
    timings["fusion_predict"] = elapsed_seconds(started_at)
    result.update(
        {
            "file_id": file_id,
            "group": args.group,
            "video": str(resolve_path(args.video)),
            "feature_paths": {
                "text": str(feature_paths["text"]),
                "audio": str(feature_paths["audio"]),
                "video": str(feature_paths["video"]),
                "face": str(feature_paths["face"]),
            },
            "timings": timings,
        }
    )
    return result


def main() -> None:
    args = parse_args()
    result = predict_video(args)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
