from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image


DEFAULT_RESOLUTION = 384
DEFAULT_PIXEL_THRESHOLD = 0.5
DEFAULT_MINIMUM_AREA_RATIO = 0.001


def run_manipulation_localizer(image: Image.Image, artifact_path: str) -> Dict[str, Any]:
    name = "truthshield-manipulation-localizer-v4"
    try:
        model, preprocess, torch = _load_localizer(str(Path(artifact_path).expanduser().resolve()))
        resolution = max(224, int(preprocess.get("resolution", DEFAULT_RESOLUTION)))
        pixel_threshold = float(preprocess.get("pixel_threshold", DEFAULT_PIXEL_THRESHOLD))
        tensor = _preprocess(image, resolution=resolution, preprocess=preprocess, torch=torch)
        with torch.inference_mode():
            logits = model(tensor)
            if isinstance(logits, dict):
                logits = logits.get("out")
            if not isinstance(logits, torch.Tensor):
                raise TypeError("The TorchScript localizer returned an unsupported output.")
            if logits.ndim == 3:
                logits = logits.unsqueeze(1)
            probabilities = torch.sigmoid(logits)
            probabilities = torch.nn.functional.interpolate(
                probabilities,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )
        probability = probabilities[0, 0].detach().cpu().numpy()
        score = _image_score(probability)
        region, area_ratio = _largest_region(
            probability,
            threshold=pixel_threshold,
            minimum_area_ratio=DEFAULT_MINIMUM_AREA_RATIO,
        )
        source_region = _scale_region(region, resolution, image.size) if region else None
        suspicious_regions = (
            [{"box": source_region, "manipulation_score": round(score, 4), "area_ratio": round(area_ratio, 6)}]
            if source_region
            else []
        )
        return {
            "name": name,
            "status": "completed",
            "label": (
                "localized_manipulation_signal"
                if suspicious_regions
                else "no_localized_manipulation_support"
            ),
            "score": round(score, 4),
            "synthetic_probability": None,
            "manipulation_probability": round(score, 4),
            "task": "manipulation",
            "model_version": str(preprocess.get("model_version") or name),
            "calibration_id": preprocess.get("calibration_id"),
            "suspicious_regions": suspicious_regions,
            "details": {
                "model_provider": "truthshield_torchscript_localizer",
                "input_resolution": resolution,
                "pixel_threshold": pixel_threshold,
                "localized_support": bool(suspicious_regions),
                "largest_region_area_ratio": round(area_ratio, 6),
                "suspicious_regions": suspicious_regions,
                "note": (
                    "The shared calibrated policy and controlled-view stability check determine the verdict; "
                    "this pixel score is not independently decisive."
                ),
            },
        }
    except ImportError as exc:
        return _unavailable(name, f"Install torch to enable the manipulation localizer: {exc}")
    except FileNotFoundError as exc:
        return _unavailable(name, str(exc))
    except Exception as exc:
        return _unavailable(name, str(exc)[:300], status="error")


@functools.lru_cache(maxsize=2)
def _load_localizer(artifact_path: str) -> tuple[Any, dict[str, Any], Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on deployment extras
        raise ImportError("PyTorch is unavailable") from exc
    artifact = Path(artifact_path)
    model_path = artifact / "model.ts" if artifact.is_dir() else artifact
    if not model_path.is_file():
        raise FileNotFoundError(f"Manipulation localizer not found: {model_path}")
    preprocess_path = model_path.parent / "preprocess.json"
    preprocess: dict[str, Any] = {}
    if preprocess_path.is_file():
        value = json.loads(preprocess_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            preprocess = value
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    return model, preprocess, torch


def _preprocess(image: Image.Image, *, resolution: int, preprocess: dict[str, Any], torch: Any) -> Any:
    rgb = image.convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values.transpose(2, 0, 1)).unsqueeze(0)
    mean = torch.tensor(preprocess.get("mean", [0.485, 0.456, 0.406]), dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(preprocess.get("std", [0.229, 0.224, 0.225]), dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _image_score(probability: np.ndarray, top_fraction: float = 0.01) -> float:
    flattened = np.asarray(probability, dtype=np.float32).reshape(-1)
    if not len(flattened):
        return 0.0
    count = max(1, int(round(len(flattened) * top_fraction)))
    start = max(0, len(flattened) - count)
    return float(np.partition(flattened, start)[start:].mean())


def _largest_region(
    probability: np.ndarray,
    *,
    threshold: float,
    minimum_area_ratio: float,
) -> tuple[list[int] | None, float]:
    binary = (np.asarray(probability) >= threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None, 0.0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[largest, cv2.CC_STAT_AREA])
    area_ratio = area / max(1, binary.size)
    if area_ratio < minimum_area_ratio:
        return None, float(area_ratio)
    x = int(stats[largest, cv2.CC_STAT_LEFT])
    y = int(stats[largest, cv2.CC_STAT_TOP])
    width = int(stats[largest, cv2.CC_STAT_WIDTH])
    height = int(stats[largest, cv2.CC_STAT_HEIGHT])
    return [x, y, x + width, y + height], float(area_ratio)


def _scale_region(region: list[int], resolution: int, image_size: tuple[int, int]) -> list[int]:
    width, height = image_size
    return [
        max(0, min(width, round(region[0] * width / resolution))),
        max(0, min(height, round(region[1] * height / resolution))),
        max(0, min(width, round(region[2] * width / resolution))),
        max(0, min(height, round(region[3] * height / resolution))),
    ]


def _unavailable(name: str, reason: str, status: str = "unavailable") -> Dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "label": None,
        "score": None,
        "synthetic_probability": None,
        "manipulation_probability": None,
        "task": "manipulation",
        "model_version": name,
        "calibration_id": None,
        "suspicious_regions": [],
        "details": {"reason": reason},
    }
