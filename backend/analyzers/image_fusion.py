from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any, Dict, Iterable


FUSION_NAME = "truthshield-image-generation-fusion-v4"
EXPECTED_FEATURES = ["community_forensics_logit", "truthshield_comparison_logit"]


def run_image_generation_fusion(
    detectors: Iterable[Dict[str, Any]],
    artifact_path: str,
) -> Dict[str, Any]:
    detector_list = list(detectors)
    community = _probability_for(detector_list, "community")
    comparison = _probability_for(detector_list, "comparison")
    if community is None or comparison is None:
        return _unavailable(
            "Both Community Forensics and the TruthShield comparison model must complete before fusion."
        )
    try:
        bundle = _load_fusion_bundle(str(Path(artifact_path).expanduser().resolve()))
        features = [_logit(community), _logit(comparison)]
        model = bundle["model"]
        probabilities = model.predict_proba([features])[0]
        classes = list(getattr(model, "classes_", range(len(probabilities))))
        positive_index = classes.index(1) if 1 in classes else len(probabilities) - 1
        probability = max(0.0, min(1.0, float(probabilities[positive_index])))
    except FileNotFoundError as exc:
        return _unavailable(str(exc))
    except ImportError as exc:
        return _unavailable(f"Install joblib and scikit-learn to enable image fusion: {exc}")
    except Exception as exc:
        return _unavailable(str(exc)[:300], status="error")
    return {
        "name": FUSION_NAME,
        "status": "completed",
        "label": "synthetic" if probability >= 0.5 else "authentic",
        "score": round(probability if probability >= 0.5 else 1.0 - probability, 4),
        "synthetic_probability": round(probability, 6),
        "manipulation_probability": None,
        "task": "generation",
        "model_version": FUSION_NAME,
        "calibration_id": None,
        "details": {
            "model_provider": "truthshield_regularized_fusion",
            "feature_names": list(bundle["feature_names"]),
            "input_scores": {
                "community_forensics": round(community, 6),
                "truthshield_comparison": round(comparison, 6),
            },
            "input_model_versions": bundle.get("input_model_versions", {}),
            "fit_split": bundle.get("fit_split"),
            "calibration_status": bundle.get("calibration_status"),
            "note": "The separate calibrated media policy applies thresholds and abstention to this fused score.",
        },
    }


@functools.lru_cache(maxsize=2)
def _load_fusion_bundle(artifact_path: str) -> dict[str, Any]:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise ImportError("joblib is unavailable") from exc
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image fusion artifact not found: {path}")
    value = joblib.load(path)
    if not isinstance(value, dict):
        raise ValueError("The image fusion artifact is not a dictionary bundle.")
    feature_names = value.get("feature_names")
    if feature_names != EXPECTED_FEATURES:
        raise ValueError(f"Unexpected image fusion features: {feature_names}")
    model = value.get("model")
    if model is None or not callable(getattr(model, "predict_proba", None)):
        raise ValueError("The image fusion bundle has no probability model.")
    return value


def _probability_for(detectors: list[Dict[str, Any]], role: str) -> float | None:
    for detector in detectors:
        if detector.get("status") != "completed" or str(detector.get("task") or "generation") != "generation":
            continue
        name = str(detector.get("name") or "").lower()
        model_version = str(detector.get("model_version") or "").lower()
        if role == "community" and not name.startswith("community-forensics::"):
            continue
        if role == "comparison" and not (
            "truthshield-image-detector" in name
            or "image-comparison-v4" in name
            or "truthshield-image-detector" in model_version
            or "image-comparison-v4" in model_version
        ):
            continue
        value = detector.get("synthetic_probability")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return max(0.0, min(1.0, float(value)))
    return None


def _logit(value: float) -> float:
    clipped = max(1e-6, min(1.0 - 1e-6, float(value)))
    return math.log(clipped / (1.0 - clipped))


def _unavailable(reason: str, status: str = "unavailable") -> Dict[str, Any]:
    return {
        "name": FUSION_NAME,
        "status": status,
        "label": None,
        "score": None,
        "synthetic_probability": None,
        "manipulation_probability": None,
        "task": "generation",
        "model_version": FUSION_NAME,
        "calibration_id": None,
        "details": {"reason": reason},
    }
