from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 0.0, maximum: float | None = None) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = max(minimum, float(value))
        return min(maximum, parsed) if maximum is not None else parsed
    except ValueError:
        return default


def _env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _local_model(relative_path: str) -> str | None:
    candidate = Path(__file__).resolve().parents[2] / relative_path
    if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
        return str(candidate)
    return None


def _local_file(relative_path: str) -> str | None:
    candidate = Path(__file__).resolve().parents[2] / relative_path
    return str(candidate) if candidate.is_file() else None


def _default_image_models() -> list[str]:
    for relative_path in (
        "training/models/truthshield-image-detector-v3",
        "training/models/truthshield-image-detector-v2",
        "training/models/truthshield-image-detector",
    ):
        local = _local_model(relative_path)
        if local:
            return [local]
    return ["Organika/sdxl-detector", "dima806/deepfake_vs_real_image_detection"]


def _configured_models(name: str, defaults: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return defaults
    configured = [item.strip() for item in raw.split(",") if item.strip()]
    if name == "AI_IMAGE_DETECTOR_MODELS":
        # Older deployment instructions populated this variable with generic
        # Hub models. Keep a packaged TruthShield model first when it exists so
        # that a stale environment variable cannot silently disable it.
        packaged = [item for item in defaults if Path(item).is_dir()]
        configured = [*packaged, *configured]
    return list(dict.fromkeys(configured))


@dataclass(frozen=True)
class EnhancedSettings:
    enable_enhanced_analysis: bool
    brave_search_api_key: str | None
    web_research_per_scan_limit: int
    web_research_monthly_limit: int
    enable_local_ai_models: bool
    local_reasoning_base_url: str | None
    ai_image_detector_models: list[str]
    ai_video_frame_detector_models: list[str]
    ai_video_temporal_model_path: str | None
    video_analysis_mode: str
    video_frame_stride: int
    video_max_frames: int
    video_tile_analysis: bool
    video_tile_size: int
    video_tile_overlap: float
    google_vision_api_key: str | None
    google_vision_max_results: int
    monthly_counter_path: Path


def get_settings() -> EnhancedSettings:
    image_models = _configured_models("AI_IMAGE_DETECTOR_MODELS", _default_image_models())
    video_model = _local_model("training/models/truthshield-video-frame-detector")
    video_models = _configured_models("AI_VIDEO_FRAME_DETECTOR_MODELS", [video_model] if video_model else image_models)
    requested_video_mode = (os.getenv("VIDEO_ANALYSIS_MODE") or "exhaustive").strip().lower()
    video_mode = requested_video_mode if requested_video_mode in {"exhaustive", "sampled"} else "exhaustive"
    return EnhancedSettings(
        enable_enhanced_analysis=_env_bool("ENABLE_ENHANCED_ANALYSIS", True),
        brave_search_api_key=os.getenv("BRAVE_SEARCH_API_KEY") or None,
        web_research_per_scan_limit=_env_int("WEB_RESEARCH_PER_SCAN_LIMIT", 2, minimum=0),
        web_research_monthly_limit=_env_int("WEB_RESEARCH_MONTHLY_LIMIT", 150, minimum=0),
        enable_local_ai_models=_env_bool("ENABLE_LOCAL_AI_MODELS", True),
        local_reasoning_base_url=(os.getenv("LOCAL_REASONING_BASE_URL") or "").rstrip("/") or None,
        ai_image_detector_models=image_models,
        ai_video_frame_detector_models=video_models,
        ai_video_temporal_model_path=(
            os.getenv("AI_VIDEO_TEMPORAL_MODEL_PATH")
            or _local_file("training/models/truthshield-video-temporal.joblib")
        ),
        video_analysis_mode=video_mode,
        video_frame_stride=_env_int("VIDEO_FRAME_STRIDE", 1, minimum=1),
        video_max_frames=_env_int("VIDEO_MAX_FRAMES", 0, minimum=0),
        video_tile_analysis=_env_bool("VIDEO_TILE_ANALYSIS", True),
        video_tile_size=_env_int("VIDEO_TILE_SIZE", 448, minimum=224),
        video_tile_overlap=_env_float("VIDEO_TILE_OVERLAP", 0.15, minimum=0.0, maximum=0.45),
        google_vision_api_key=os.getenv("GOOGLE_VISION_API_KEY") or os.getenv("GOOGLE_CLOUD_VISION_API_KEY") or None,
        google_vision_max_results=_env_int("GOOGLE_VISION_MAX_RESULTS", 10, minimum=1),
        monthly_counter_path=Path(
            os.getenv(
                "WEB_RESEARCH_COUNTER_PATH",
                str(Path(tempfile.gettempdir()) / "truthshield_web_research_counter.json"),
            )
        ),
    )
