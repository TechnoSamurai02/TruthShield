from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from analyzers.config import get_settings
from analyzers.media_decision import load_media_policy


def model_health() -> Dict[str, Any]:
    settings = get_settings()
    policy = load_media_policy(settings.media_policy_path)
    generation = [_artifact_status(model, "generation") for model in settings.ai_image_detector_models]
    manipulation = [_artifact_status(model, "manipulation") for model in settings.ai_manipulation_detector_models]
    temporal = (
        [_artifact_status(settings.ai_video_temporal_model_path, "temporal")]
        if settings.ai_video_temporal_model_path
        else []
    )
    calibration = _artifact_status(settings.media_policy_path, "calibration") if settings.media_policy_path else {
        "id": None,
        "task": "calibration",
        "status": "missing",
        "checksum_sha256": None,
    }
    generation_ready = bool(generation) and all(item["status"] in {"available_for_lazy_load", "remote_configured"} for item in generation)
    manipulation_ready = bool(manipulation) and all(item["status"] in {"available_for_lazy_load", "remote_configured"} for item in manipulation)
    manipulation_screen_ready = manipulation_ready or _has_editing_screen(settings.ai_image_detector_models)
    calibration_ready = calibration["status"] == "available_for_lazy_load"
    return {
        "decision_policy_version": policy.get("policy_version"),
        "calibration_id": policy.get("calibration_id"),
        "calibration_status": policy.get("calibration_status", "unknown"),
        "required_models": {
            "generation": generation,
            "manipulation": manipulation,
            "video_temporal": temporal,
        },
        "calibration_artifact": calibration,
        "capabilities": {
            "decisive_generation_verdicts": bool(generation_ready and calibration_ready and _enabled(policy, "generation")),
            "decisive_manipulation_verdicts": bool(manipulation_ready and calibration_ready and _enabled(policy, "manipulation")),
            "decisive_authentic_verdicts": bool(generation_ready and manipulation_screen_ready and calibration_ready),
            "manipulation_screening_for_authentic_verdicts": manipulation_screen_ready,
            "inconclusive_fallback": True,
        },
        "loading_note": (
            "Local neural weights are lazy-loaded on first analysis. 'available_for_lazy_load' means the checked artifact "
            "is present; it does not claim that inference has already succeeded in this process."
        ),
    }


def _enabled(policy: Dict[str, Any], task: str) -> bool:
    value = policy.get(task)
    return bool(value.get("enabled", True)) if isinstance(value, dict) else True


def _artifact_status(identifier: str | None, task: str) -> Dict[str, Any]:
    if not identifier:
        return {"id": None, "task": task, "status": "missing", "checksum_sha256": None}
    path = Path(identifier).expanduser()
    if not path.exists():
        return {
            "id": identifier,
            "task": task,
            "status": "remote_configured" if not _looks_like_path(identifier) else "missing",
            "checksum_sha256": None,
        }
    files = _artifact_files(path)
    return {
        "id": identifier,
        "task": task,
        "status": "available_for_lazy_load",
        "checksum_sha256": _combined_checksum(tuple(str(item.resolve()) for item in files)),
        "files_checked": [item.name for item in files],
    }


def _artifact_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    preferred = [path / "config.json", path / "model.safetensors"]
    return [item for item in preferred if item.is_file()]


@functools.lru_cache(maxsize=32)
def _combined_checksum(files: tuple[str, ...]) -> str | None:
    if not files:
        return None
    digest = hashlib.sha256()
    for filename in files:
        path = Path(filename)
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _looks_like_path(value: str) -> bool:
    return any(marker in value for marker in ("\\", "/training/", ":\\")) or value.startswith(".")


def _has_editing_screen(identifiers: Iterable[str]) -> bool:
    for identifier in identifiers:
        config_path = Path(identifier).expanduser() / "config.json"
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        labels = config.get("id2label", {}) if isinstance(config, dict) else {}
        if isinstance(labels, dict) and any(
            any(marker in str(label).lower() for marker in ("edited", "manipulated", "tampered", "inpaint"))
            for label in labels.values()
        ):
            return True
    return False
