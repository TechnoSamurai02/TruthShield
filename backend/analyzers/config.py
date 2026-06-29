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


def _env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class EnhancedSettings:
    enable_enhanced_analysis: bool
    brave_search_api_key: str | None
    web_research_per_scan_limit: int
    web_research_monthly_limit: int
    enable_local_ai_models: bool
    local_reasoning_base_url: str | None
    ai_image_detector_models: list[str]
    monthly_counter_path: Path


def get_settings() -> EnhancedSettings:
    return EnhancedSettings(
        enable_enhanced_analysis=_env_bool("ENABLE_ENHANCED_ANALYSIS", True),
        brave_search_api_key=os.getenv("BRAVE_SEARCH_API_KEY") or None,
        web_research_per_scan_limit=_env_int("WEB_RESEARCH_PER_SCAN_LIMIT", 2, minimum=0),
        web_research_monthly_limit=_env_int("WEB_RESEARCH_MONTHLY_LIMIT", 150, minimum=0),
        enable_local_ai_models=_env_bool("ENABLE_LOCAL_AI_MODELS", True),
        local_reasoning_base_url=(os.getenv("LOCAL_REASONING_BASE_URL") or "").rstrip("/") or None,
        ai_image_detector_models=_env_list(
            "AI_IMAGE_DETECTOR_MODELS",
            "dima806/deepfake_vs_real_image_detection",
        ),
        monthly_counter_path=Path(
            os.getenv(
                "WEB_RESEARCH_COUNTER_PATH",
                str(Path(tempfile.gettempdir()) / "truthshield_web_research_counter.json"),
            )
        ),
    )
