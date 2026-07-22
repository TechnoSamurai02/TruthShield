from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

from PIL import Image


def run_community_forensics(
    image: Image.Image,
    model_id: str,
    official_repo_path: str | None,
) -> Dict[str, Any]:
    name = f"community-forensics::{model_id}"
    if not official_repo_path:
        return _unavailable(name, "Set COMMUNITY_FORENSICS_REPO_PATH to a reviewed checkout of the official MIT repository.")
    repo = Path(official_repo_path).expanduser().resolve()
    required = [repo / "models.py", repo / "LICENSE"]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return _unavailable(name, f"The configured Community Forensics checkout is missing: {', '.join(missing)}")
    try:
        license_text = (repo / "LICENSE").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError as exc:
        return _unavailable(name, f"Could not read the specialist license: {exc}")
    if "mit license" not in license_text:
        return _unavailable(name, "The configured checkout does not contain the expected MIT license text.")
    try:
        model, torch = _load_official_model(str(repo), model_id)
        values = _official_test_preprocess(image.convert("RGB"), torch=torch)
        with torch.inference_mode():
            probability = float(torch.sigmoid(model(values.to("cpu"))).reshape(-1)[0].item())
    except ImportError as exc:
        return _unavailable(name, f"Install timm and the official Community Forensics dependencies: {exc}")
    except Exception as exc:
        return _unavailable(name, str(exc)[:300], status="error")
    return {
        "name": name,
        "status": "completed",
        "label": "fake" if probability >= 0.5 else "real",
        "score": round(probability if probability >= 0.5 else 1.0 - probability, 4),
        "synthetic_probability": round(probability, 4),
        "manipulation_probability": None,
        "task": "generation",
        "model_version": model_id,
        "calibration_id": "bootstrap-conservative-v4",
        "details": {
            "model_provider": "community_forensics_official_adapter",
            "official_repo_path": str(repo),
            "input_size": 224,
            "license": "MIT",
            "note": "Official model output; the shared v4 calibration and abstention policy still determines the verdict.",
        },
    }


@functools.lru_cache(maxsize=2)
def _load_official_model(repo_path: str, model_id: str) -> tuple[Any, Any]:
    try:
        import torch
    except Exception as exc:
        raise ImportError("torch is unavailable") from exc
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    models = _load_module("_truthshield_commfor_models", Path(repo_path) / "models.py")
    model = models.ViTClassifier.from_pretrained(model_id, device="cpu")
    model = model.to("cpu").eval()
    return model, torch


def _official_test_preprocess(image: Image.Image, *, torch: Any) -> Any:
    width, height = image.size
    scale = 256.0 / max(1, min(width, height))
    resized = image.resize(
        (max(224, round(width * scale)), max(224, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    left = max(0, (resized.width - 224) // 2)
    top = max(0, (resized.height - 224) // 2)
    cropped = resized.crop((left, top, left + 224, top + 224))
    import numpy as np

    array = np.asarray(cropped, dtype=np.float32) / 255.0
    values = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=values.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=values.dtype).view(1, 3, 1, 1)
    return (values - mean) / std


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _unavailable(name: str, reason: str, status: str = "unavailable") -> Dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "label": None,
        "score": None,
        "synthetic_probability": None,
        "manipulation_probability": None,
        "task": "generation",
        "model_version": name.split("::", 1)[-1],
        "calibration_id": None,
        "details": {"reason": reason},
    }
