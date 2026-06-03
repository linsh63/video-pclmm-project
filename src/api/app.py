from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.inference.predict_video import (
    DEFAULT_GROUP,
    PROJECT_ROOT,
    elapsed_seconds,
    parse_step_cuda_visible_devices,
    prepare_video_features,
    resolve_device,
    resolve_path,
    sanitize_file_id,
)
from src.inference.resident_extractors import ResidentExtractorConfig, ResidentFeatureExtractor


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _step_env(step_name: str, env_name: str) -> str | None:
    value = os.environ.get(env_name)
    if not value:
        return None
    return f"{step_name}={value}"


@dataclass(frozen=True)
class ApiSettings:
    checkpoint: Path
    threshold: float
    device: str
    target_length: int
    cuda_visible_devices: str | None
    runtime_root: Path
    cache_root: Path
    upload_root: Path
    group: str
    skip_existing: bool
    cleanup_uploads: bool
    extraction_mode: str
    step_cuda_visible_devices: dict[str, str]
    feature_backend: str
    resident_vit_device: str
    resident_whisper_device: str
    resident_bert_device: str
    resident_face_device: str
    resident_vit_batch_size: int
    resident_face_batch_size: int
    resident_whisper_model: str
    resident_whisper_cache_dir: Path
    resident_torch_home: Path

    @classmethod
    def from_env(cls) -> "ApiSettings":
        cuda_visible_devices = os.environ.get("PCLMM_API_CUDA_VISIBLE_DEVICES") or None
        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        step_cuda_values = [
            value
            for value in [
                os.environ.get("PCLMM_API_STEP_CUDA_VISIBLE_DEVICES"),
                _step_env("extract_video_vit.py", "PCLMM_API_VIT_CUDA_VISIBLE_DEVICES"),
                _step_env("extract_audio_text.py", "PCLMM_API_WHISPER_CUDA_VISIBLE_DEVICES"),
                _step_env("BERT.py", "PCLMM_API_BERT_CUDA_VISIBLE_DEVICES"),
                _step_env("extract_face_fervt.py", "PCLMM_API_FACE_CUDA_VISIBLE_DEVICES"),
            ]
            if value
        ]

        return cls(
            checkpoint=resolve_path(
                os.environ.get(
                    "PCLMM_API_CHECKPOINT",
                    "outputs/checkpoints/multi_modal_cross_attention_model.pth",
                )
            ),
            threshold=float(os.environ.get("PCLMM_API_THRESHOLD", "0.5")),
            device=os.environ.get("PCLMM_API_DEVICE", "auto"),
            target_length=int(os.environ.get("PCLMM_API_TARGET_LENGTH", "512")),
            cuda_visible_devices=cuda_visible_devices,
            runtime_root=resolve_path(os.environ.get("PCLMM_API_RUNTIME_ROOT", "outputs/api/runtime")),
            cache_root=resolve_path(os.environ.get("PCLMM_API_CACHE_ROOT", "outputs/api/cache")),
            upload_root=resolve_path(os.environ.get("PCLMM_API_UPLOAD_ROOT", "outputs/api/uploads")),
            group=os.environ.get("PCLMM_API_GROUP", DEFAULT_GROUP),
            skip_existing=env_bool("PCLMM_API_SKIP_EXISTING", True),
            cleanup_uploads=env_bool("PCLMM_API_CLEANUP_UPLOADS", True),
            extraction_mode=os.environ.get("PCLMM_API_EXTRACTION_MODE", "parallel"),
            step_cuda_visible_devices=parse_step_cuda_visible_devices(step_cuda_values),
            feature_backend=os.environ.get("PCLMM_API_FEATURE_BACKEND", "resident"),
            resident_vit_device=os.environ.get("PCLMM_API_VIT_DEVICE", "auto"),
            resident_whisper_device=os.environ.get("PCLMM_API_WHISPER_DEVICE", "auto"),
            resident_bert_device=os.environ.get("PCLMM_API_BERT_DEVICE", "auto"),
            resident_face_device=os.environ.get("PCLMM_API_FACE_DEVICE", "auto"),
            resident_vit_batch_size=int(os.environ.get("PCLMM_API_VIT_BATCH_SIZE", "16")),
            resident_face_batch_size=int(os.environ.get("PCLMM_API_FACE_BATCH_SIZE", "4")),
            resident_whisper_model=os.environ.get("WHISPER_MODEL", "large"),
            resident_whisper_cache_dir=resolve_path(
                os.environ.get("WHISPER_CACHE_DIR", "/data4/songxinshuai/cache/whisper")
            ),
            resident_torch_home=resolve_path(
                os.environ.get("TORCH_HOME", "outputs/cache/torch")
            ),
        )


class VideoModelService:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.lock = threading.Lock()
        self.predictor = None
        self.resident_extractor = None
        self.load_timings = {}

    def load(self) -> None:
        from src.inference.predict_features import FusionFeaturePredictor

        if not self.settings.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.settings.checkpoint}")

        self.settings.upload_root.mkdir(parents=True, exist_ok=True)
        self.predictor = FusionFeaturePredictor(
            checkpoint=self.settings.checkpoint,
            threshold=self.settings.threshold,
            device=resolve_device(self.settings.device),
            target_length=self.settings.target_length,
        )
        if self.settings.feature_backend == "resident":
            extractor_config = ResidentExtractorConfig(
                cache_root=self.settings.cache_root,
                group=self.settings.group,
                extraction_mode=self.settings.extraction_mode,
                vit_device=self.settings.resident_vit_device,
                whisper_device=self.settings.resident_whisper_device,
                bert_device=self.settings.resident_bert_device,
                face_device=self.settings.resident_face_device,
                vit_batch_size=self.settings.resident_vit_batch_size,
                face_batch_size=self.settings.resident_face_batch_size,
                whisper_model=self.settings.resident_whisper_model,
                whisper_cache_dir=self.settings.resident_whisper_cache_dir,
                torch_home=self.settings.resident_torch_home,
            )
            self.resident_extractor = ResidentFeatureExtractor(extractor_config)
            self.load_timings = self.resident_extractor.load()
        elif self.settings.feature_backend != "subprocess":
            raise ValueError(
                "PCLMM_API_FEATURE_BACKEND must be 'resident' or 'subprocess'."
            )

    def predict(
        self,
        video_path: Path,
        file_id: str,
        threshold: float | None = None,
        skip_existing: bool | None = None,
    ) -> dict:
        if self.predictor is None:
            raise RuntimeError("Model predictor has not been loaded.")

        skip_existing = self.settings.skip_existing if skip_existing is None else skip_existing
        with self.lock:
            service_started_at = time.perf_counter()
            if self.settings.feature_backend == "resident":
                if self.resident_extractor is None:
                    raise RuntimeError("Resident feature extractor has not been loaded.")
                effective_file_id, feature_paths, timings = self.resident_extractor.extract(
                    source_video=video_path,
                    file_id=file_id,
                    group=self.settings.group,
                    copy_video=True,
                    skip_existing=skip_existing,
                    extraction_mode=self.settings.extraction_mode,
                )
            else:
                effective_file_id, feature_paths, timings = prepare_video_features(
                    source_video=video_path,
                    runtime_root=self.settings.runtime_root,
                    cache_root=self.settings.cache_root,
                    group=self.settings.group,
                    file_id=file_id,
                    cuda_visible_devices=self.settings.cuda_visible_devices,
                    step_cuda_visible_devices=self.settings.step_cuda_visible_devices,
                    copy_video=True,
                    skip_existing=skip_existing,
                    extraction_mode=self.settings.extraction_mode,
                )
            fusion_started_at = time.perf_counter()
            result = self.predictor.predict_paths(
                text_path=feature_paths["text"],
                audio_path=feature_paths["audio"],
                video_path=feature_paths["video"],
                face_path=feature_paths["face"],
                threshold=threshold,
            )
            timings["fusion_predict"] = elapsed_seconds(fusion_started_at)
            timings["service_predict_total"] = elapsed_seconds(service_started_at)

        result.update(
            {
                "file_id": effective_file_id,
                "group": self.settings.group,
                "feature_paths": {
                    "text": str(feature_paths["text"]),
                    "audio": str(feature_paths["audio"]),
                    "video": str(feature_paths["video"]),
                    "face": str(feature_paths["face"]),
                },
                "timings": timings,
                "cache_policy": {
                    "skip_existing": skip_existing,
                },
                "extraction_mode": self.settings.extraction_mode,
                "feature_backend": self.settings.feature_backend,
            }
        )
        return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = ApiSettings.from_env()
    service = VideoModelService(settings)
    service.load()
    app.state.settings = settings
    app.state.service = service
    yield


app = FastAPI(
    title="Video PCLMM Model API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    settings: ApiSettings = app.state.settings
    service: VideoModelService = app.state.service
    return {
        "ok": True,
        "project_root": str(PROJECT_ROOT),
        "checkpoint": str(settings.checkpoint),
        "device": settings.device,
        "cuda_visible_devices": settings.cuda_visible_devices,
        "feature_backend": settings.feature_backend,
        "extraction_mode": settings.extraction_mode,
        "step_cuda_visible_devices": settings.step_cuda_visible_devices,
        "resident_devices": {
            "vit": settings.resident_vit_device,
            "whisper": settings.resident_whisper_device,
            "bert": settings.resident_bert_device,
            "face": settings.resident_face_device,
            "fusion": settings.device,
        },
        "load_timings": service.load_timings,
        "predictor_loaded": service.predictor is not None,
        "resident_extractor_loaded": service.resident_extractor is not None,
    }


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    threshold: float | None = Form(default=None),
    file_id: str | None = Form(default=None),
    debug: bool = Form(default=False),
    force_recompute: bool = Form(default=False),
) -> dict:
    request_started_at = time.perf_counter()
    filename = file.filename or "uploaded.mp4"
    if not filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 videos are supported.")

    settings: ApiSettings = app.state.settings
    service: VideoModelService = app.state.service

    request_id = uuid.uuid4().hex
    safe_stem = sanitize_file_id(file_id or f"{Path(filename).stem}_{request_id[:8]}")
    upload_path = settings.upload_root / f"{safe_stem}.mp4"
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        upload_started_at = time.perf_counter()
        with upload_path.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        upload_save_seconds = elapsed_seconds(upload_started_at)

        result = service.predict(
            video_path=upload_path,
            file_id=safe_stem,
            threshold=threshold,
            skip_existing=settings.skip_existing and not force_recompute,
        )
        result.setdefault("timings", {})
        result["timings"]["upload_save"] = upload_save_seconds
        result["timings"]["request_total"] = elapsed_seconds(request_started_at)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        file.file.close()
        if settings.cleanup_uploads:
            upload_path.unlink(missing_ok=True)

    if debug:
        return result
    return {"is_normal": bool(result["is_normal"])}
