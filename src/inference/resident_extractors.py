from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import BertModel, BertTokenizer, ViTFeatureExtractor, ViTModel

from src.extraction.mfcc_compatible import extract_mfcc
from src.inference.predict_video import (
    PROJECT_ROOT,
    DEFAULT_GROUP,
    elapsed_seconds,
    expected_feature_paths,
    prepare_video_link,
    resolve_device,
    resolve_path,
    sanitize_file_id,
)


@dataclass(frozen=True)
class ResidentExtractorConfig:
    cache_root: Path
    group: str = DEFAULT_GROUP
    extraction_mode: str = "parallel"
    vit_device: str = "auto"
    whisper_device: str = "auto"
    bert_device: str = "auto"
    face_device: str = "auto"
    vit_batch_size: int = 16
    face_batch_size: int = 4
    whisper_model: str = "large"
    whisper_cache_dir: Path = PROJECT_ROOT / "outputs/cache/whisper"
    torch_home: Path = PROJECT_ROOT / "outputs/cache/torch"
    vit_model_dir: Path = PROJECT_ROOT / "pretrained/googlevit-base-patch16-224-in21k"
    bert_model_dir: Path = PROJECT_ROOT / "pretrained/bert_chinese"
    fervt_root: Path = PROJECT_ROOT / "pretrained/FER-VT"


class ResidentFeatureExtractor:
    def __init__(self, config: ResidentExtractorConfig) -> None:
        self.config = config
        self.video_root = config.cache_root / "videos"
        self.feature_root = config.cache_root / "features"
        self.temp_root = config.cache_root / "temp"
        self.timings: dict[str, float] = {}

        self.vit_device = torch.device(resolve_device(config.vit_device))
        self.whisper_device = resolve_device(config.whisper_device)
        self.bert_device = torch.device(resolve_device(config.bert_device))
        self.face_device = torch.device(resolve_device(config.face_device))

        self.vit_feature_extractor = None
        self.vit_model = None
        self.whisper_model = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.face_detector = None
        self.fer_vt = None
        self._face_hook_features: list[np.ndarray] = []
        self._face_hook_handle = None

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def load(self) -> dict[str, float]:
        os.environ.setdefault("TORCH_HOME", str(self.config.torch_home))
        self.config.torch_home.mkdir(parents=True, exist_ok=True)

        timings: dict[str, float] = {}
        started_at = time.perf_counter()
        self.vit_feature_extractor = ViTFeatureExtractor.from_pretrained(
            str(self.config.vit_model_dir)
        )
        self.vit_model = ViTModel.from_pretrained(str(self.config.vit_model_dir)).to(
            self.vit_device
        )
        self.vit_model.eval()
        timings["load_vit"] = elapsed_seconds(started_at)

        started_at = time.perf_counter()
        import whisper

        self.whisper_model = whisper.load_model(
            self.config.whisper_model,
            device=self.whisper_device,
            download_root=str(self.config.whisper_cache_dir),
        )
        timings["load_whisper"] = elapsed_seconds(started_at)

        started_at = time.perf_counter()
        self.bert_tokenizer = BertTokenizer.from_pretrained(str(self.config.bert_model_dir))
        self.bert_model = BertModel.from_pretrained(str(self.config.bert_model_dir)).to(
            self.bert_device
        )
        self.bert_model.eval()
        timings["load_bert"] = elapsed_seconds(started_at)

        started_at = time.perf_counter()
        self._load_face_model()
        timings["load_face"] = elapsed_seconds(started_at)

        self.timings = timings
        return timings

    def _load_face_model(self) -> None:
        from facenet_pytorch import MTCNN

        fervt_parent = str(self.config.fervt_root)
        if fervt_parent not in sys.path:
            sys.path.insert(0, fervt_parent)

        from model.FERVT import FERVT

        self.face_detector = MTCNN(keep_all=False, device="cpu")
        if self.face_device.type == "cuda":
            torch.cuda.set_device(self.face_device)
        self.fer_vt = FERVT(device=self.face_device).to(self.face_device)
        self.fer_vt.eval()
        self._face_hook_handle = self.fer_vt.vta.layernorm.register_forward_hook(
            self._face_hook
        )

    def _face_hook(self, module, inputs, output) -> None:
        self._face_hook_features.append(output.detach().cpu().numpy())

    def extract(
        self,
        source_video: str | Path,
        file_id: str | None = None,
        group: str | None = None,
        copy_video: bool = True,
        skip_existing: bool = True,
        extraction_mode: str | None = None,
    ) -> tuple[str, dict[str, Path], dict[str, float | bool | str]]:
        source_video = resolve_path(source_video)
        group = group or self.config.group
        extraction_mode = extraction_mode or self.config.extraction_mode
        file_id = sanitize_file_id(file_id or source_video.stem)

        timings: dict[str, float | bool | str] = {"feature_backend": "resident"}

        started_at = time.perf_counter()
        prepare_video_link(
            source_video=source_video,
            video_root=self.video_root,
            group=group,
            file_id=file_id,
            copy_video=copy_video,
        )
        timings["prepare_video_link"] = elapsed_seconds(started_at)

        paths = expected_feature_paths(self.feature_root, self.temp_root, group, file_id)
        started_at = time.perf_counter()
        self._ensure_features(
            video_path=self.video_root / group / f"{file_id}.mp4",
            paths=paths,
            skip_existing=skip_existing,
            extraction_mode=extraction_mode,
            timings=timings,
        )
        timings["ensure_features_total"] = elapsed_seconds(started_at)

        missing = [name for name in ("video", "audio", "text", "face") if not paths[name].exists()]
        if missing:
            detail = {name: str(paths[name]) for name in missing}
            raise RuntimeError(f"Resident extraction did not produce all required features: {detail}")

        return file_id, paths, timings

    def _ensure_features(
        self,
        video_path: Path,
        paths: dict[str, Path],
        skip_existing: bool,
        extraction_mode: str,
        timings: dict[str, float | bool | str],
    ) -> None:
        timings["extraction_mode"] = extraction_mode

        def run_step(name: str, output_keys: tuple[str, ...], func) -> None:
            if skip_existing and all(paths[key].exists() for key in output_keys):
                timings[name] = 0.0
                timings[f"{name}:skipped"] = True
                return

            started_at = time.perf_counter()
            func()
            timings[name] = elapsed_seconds(started_at)
            timings[f"{name}:skipped"] = False

        def audio_branch() -> None:
            run_step("extract_audio_wav.py", ("wav",), lambda: self.extract_wav(video_path, paths["wav"]))
            with ThreadPoolExecutor(max_workers=2) as executor:
                mfcc_future = executor.submit(
                    run_step,
                    "MFCC.py",
                    ("audio",),
                    lambda: self.extract_audio_feature(paths["wav"], paths["audio"]),
                )
                asr_future = executor.submit(
                    run_step,
                    "extract_audio_text.py",
                    ("txt",),
                    lambda: self.extract_text(paths["wav"], paths["txt"]),
                )
                mfcc_future.result()
                asr_future.result()
            run_step("BERT.py", ("text",), lambda: self.extract_bert(paths["txt"], paths["text"]))

        if extraction_mode == "sequential":
            run_step("extract_video_vit.py", ("video",), lambda: self.extract_vit(video_path, paths["video"]))
            audio_branch()
            run_step("extract_face_fervt.py", ("face",), lambda: self.extract_face(video_path, paths["face"]))
        elif extraction_mode == "parallel":
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(
                        run_step,
                        "extract_video_vit.py",
                        ("video",),
                        lambda: self.extract_vit(video_path, paths["video"]),
                    ),
                    executor.submit(audio_branch),
                    executor.submit(
                        run_step,
                        "extract_face_fervt.py",
                        ("face",),
                        lambda: self.extract_face(video_path, paths["face"]),
                    ),
                ]
                for future in as_completed(futures):
                    future.result()
        else:
            raise ValueError(f"Unsupported extraction mode: {extraction_mode}")

    def extract_wav(self, video_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-b:a",
            "192k",
            str(output_path),
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def extract_audio_feature(self, wav_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature = extract_mfcc(wav_path)
        with output_path.open("wb") as handle:
            pickle.dump(feature, handle)

    def extract_text(self, wav_path: Path, output_path: Path) -> None:
        if self.whisper_model is None:
            raise RuntimeError("Whisper model is not loaded.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = self.whisper_model.transcribe(str(wav_path), task="transcribe")
        output_path.write_text(result["text"], encoding="utf-8")

    def extract_bert(self, txt_path: Path, output_path: Path) -> None:
        if self.bert_tokenizer is None or self.bert_model is None:
            raise RuntimeError("BERT model is not loaded.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text = txt_path.read_text(encoding="utf-8")
        inputs = self.bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
            add_special_tokens=True,
        )
        inputs = {key: value.to(self.bert_device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            feature = outputs.last_hidden_state[0][0].detach().cpu().numpy()
        with output_path.open("wb") as handle:
            pickle.dump(feature, handle)

    def extract_vit(self, video_path: Path, output_path: Path) -> None:
        if self.vit_feature_extractor is None or self.vit_model is None:
            raise RuntimeError("ViT model is not loaded.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        features: list[np.ndarray] = []
        frames: list[Image.Image] = []

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                if len(frames) >= self.config.vit_batch_size:
                    features.extend(self._extract_vit_batch(frames))
                    frames = []
            if frames:
                features.extend(self._extract_vit_batch(frames))
        finally:
            cap.release()

        with output_path.open("wb") as handle:
            pickle.dump(features, handle)

    def _extract_vit_batch(self, frames: list[Image.Image]) -> list[np.ndarray]:
        if self.vit_feature_extractor is None or self.vit_model is None:
            raise RuntimeError("ViT model is not loaded.")

        inputs = self.vit_feature_extractor(images=frames, return_tensors="pt")
        inputs = {key: value.to(self.vit_device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            batch_features = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        return [feature.astype(np.float32) for feature in batch_features]

    def extract_face(self, video_path: Path, output_path: Path) -> None:
        if self.face_detector is None or self.fer_vt is None:
            raise RuntimeError("Face model is not loaded.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_indices = generate_sample_indices(total_frames)
        video_features: list[np.ndarray] = []

        try:
            for frame_id in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap.read()
                if not ok:
                    video_features.append(np.zeros((1, 192), dtype=np.float32))
                    continue
                video_features.append(self._extract_face_frame(frame))
        finally:
            cap.release()

        video_features = [np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0) for feature in video_features]
        non_zero_features = [feature for feature in video_features if np.any(feature)]
        data = {"features": non_zero_features} if non_zero_features else {"all_zero": True}
        with output_path.open("wb") as handle:
            pickle.dump(data, handle)

    def _extract_face_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.face_detector is None or self.fer_vt is None:
            raise RuntimeError("Face model is not loaded.")

        faces = self.face_detector(frame)
        if faces is None:
            return np.zeros((1, 192), dtype=np.float32)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = self.image_transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(
            self.face_device
        )

        self._face_hook_features.clear()
        try:
            if self.face_device.type == "cuda":
                torch.cuda.set_device(self.face_device)
            with torch.no_grad():
                self.fer_vt(frame_tensor)
            if self._face_hook_features:
                feature = self._face_hook_features[0].astype(np.float32)
                feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
                if np.any(feature):
                    return feature
        except Exception:
            return np.zeros((1, 192), dtype=np.float32)
        return np.zeros((1, 192), dtype=np.float32)


def generate_sample_indices(total_frames: int) -> list[int]:
    sample_count = max(1, total_frames // 10)
    sample_interval = total_frames / sample_count
    return [int(i * sample_interval) for i in range(sample_count)]
