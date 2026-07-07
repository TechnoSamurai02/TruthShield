from __future__ import annotations

import functools
from typing import Any, Dict, List

from PIL import Image

from analyzers.config import get_settings


SYNTHETIC_LABEL_MARKERS = (
    "fake",
    "ai generated",
    "ai-generated",
    "generated",
    "synthetic",
    "artificial",
    "deepfake",
    "sdxl",
    "diffusion",
)
REAL_LABEL_MARKERS = (
    "real",
    "authentic",
    "natural",
    "camera",
    "human",
    "not ai",
    "not_ai",
    "non ai",
    "non-ai",
    "non_ai",
    "human made",
    "photograph",
)
FILENAME_SYNTHETIC_MARKERS = ("chatgpt", "dall", "dalle", "midjourney", "stable-diffusion", "sdxl", "generated", "ai")


def run_image_detectors(
    image: Image.Image,
    filename: str,
    metadata_present: bool,
    technical_details: Dict[str, Any],
) -> List[Dict[str, Any]]:
    settings = get_settings()
    results = [
        _heuristic_synthetic_detector(
            filename=filename,
            metadata_present=metadata_present,
            technical_details=technical_details,
        )
    ]
    if not settings.enable_local_ai_models:
        results.append(
            {
                "name": "huggingface_image_detector",
                "status": "skipped",
                "label": None,
                "score": None,
                "synthetic_probability": None,
                "details": {"reason": "ENABLE_LOCAL_AI_MODELS is false."},
            }
        )
        return results

    for model_id in settings.ai_image_detector_models:
        results.append(_run_huggingface_detector(image, model_id))
    return results


def highest_synthetic_probability(detectors: List[Dict[str, Any]]) -> float | None:
    probabilities = [
        float(detector["synthetic_probability"])
        for detector in detectors
        if isinstance(detector.get("synthetic_probability"), (int, float))
    ]
    if not probabilities:
        return None
    return max(0.0, min(1.0, max(probabilities)))


def combined_synthetic_probability(detectors: List[Dict[str, Any]]) -> float | None:
    weighted_probabilities = []
    for detector in detectors:
        probability = detector.get("synthetic_probability")
        if not isinstance(probability, (int, float)):
            continue
        weight = 0.65 if detector.get("name") == "local_heuristic_synthetic_likelihood" else 1.0
        weighted_probabilities.append((float(probability), weight))
    if not weighted_probabilities:
        return None
    if len(weighted_probabilities) == 1:
        return max(0.0, min(1.0, weighted_probabilities[0][0]))

    weighted_average = sum(probability * weight for probability, weight in weighted_probabilities) / sum(
        weight for _, weight in weighted_probabilities
    )
    peak = max(probability for probability, _ in weighted_probabilities)
    if peak >= 0.85 and weighted_average >= 0.55:
        combined = peak * 0.55 + weighted_average * 0.45
    else:
        combined = peak * 0.25 + weighted_average * 0.75
    return max(0.0, min(1.0, combined))


def completed_model_count(detectors: List[Dict[str, Any]]) -> int:
    return sum(
        1
        for detector in detectors
        if detector.get("status") == "completed" and detector.get("name") != "local_heuristic_synthetic_likelihood"
    )


def _heuristic_synthetic_detector(filename: str, metadata_present: bool, technical_details: Dict[str, Any]) -> Dict[str, Any]:
    probability = 0.14
    reasons: List[str] = []
    lowered_filename = filename.lower()
    forensic = technical_details.get("forensic_analysis")
    forensic = forensic if isinstance(forensic, dict) else {}
    caption_overlay = forensic.get("caption_overlay")
    caption_like = isinstance(caption_overlay, dict) and bool(caption_overlay.get("is_likely"))
    forensic_synthetic = _float_from(forensic, "synthetic_artifact_probability")
    forensic_manipulation = _float_from(forensic, "manipulation_probability")

    if any(marker in lowered_filename for marker in FILENAME_SYNTHETIC_MARKERS):
        probability += 0.40
        reasons.append("The filename contains wording commonly used by AI generation tools.")
    if not metadata_present:
        probability += 0.05 if caption_like else 0.10
        if caption_like:
            reasons.append("No readable camera metadata was found, but the detected caption/edit can explain stripped metadata.")
        else:
            reasons.append("No readable camera metadata was found.")
    detected_format = str(technical_details.get("detected_format") or "").upper()
    if detected_format in {"PNG", "WEBP"} and not metadata_present:
        probability += 0.03 if caption_like else 0.08
        reasons.append("The file format and metadata pattern are common for generated or exported images.")
    entropy = _float_detail(technical_details, "entropy")
    if entropy is not None and (entropy < 3.5 or entropy > 7.8):
        probability += 0.10
        reasons.append("Image entropy is outside the typical range.")
    blur = _float_detail(technical_details, "blur_laplacian_variance")
    if blur is not None and blur < 35:
        probability += 0.12
        reasons.append("The image has possible over-smoothing or heavy blur.")
    compression = technical_details.get("compression_consistency")
    if isinstance(compression, dict) and compression.get("is_inconsistent"):
        probability += 0.02 if caption_like else 0.08
        if caption_like:
            reasons.append("Compression consistency is uneven, but an added caption or graphic overlay can cause this.")
        else:
            reasons.append("Compression or texture consistency is uneven.")
    if forensic_synthetic is not None:
        if forensic_synthetic >= 0.60:
            probability += min(0.26, (forensic_synthetic - 0.50) * 0.70)
            reasons.append("Pixel-level forensic checks found synthetic-image artifact signals.")
        elif forensic_synthetic <= 0.28:
            probability -= 0.07
            reasons.append("Pixel-level forensic checks did not find strong AI-generation artifacts.")
    if forensic_manipulation is not None and forensic_manipulation >= 0.58 and not caption_like:
        probability += 0.06
        reasons.append("Pixel-level forensic checks found possible editing or compositing artifacts.")
    if caption_like:
        probability -= 0.04
        reasons.append("A caption or graphic overlay was detected, which is an edit/context clue rather than direct AI-generation evidence.")
    if not reasons:
        reasons.append("No strong local synthetic markers were found.")

    probability = max(0.02, min(0.98, probability))
    label = "likely_ai_generated" if probability >= 0.65 else "uncertain" if probability >= 0.35 else "likely_camera_or_natural"
    return {
        "name": "local_heuristic_synthetic_likelihood",
        "status": "completed",
        "label": label,
        "score": round(probability, 3),
        "synthetic_probability": round(probability, 3),
        "details": {"reasons": reasons, "model_type": "deterministic_free_fallback"},
    }


def _run_huggingface_detector(image: Image.Image, model_id: str) -> Dict[str, Any]:
    try:
        classifier = _load_pipeline(model_id)
        outputs = classifier(image.convert("RGB"))
    except ImportError:
        return {
            "name": model_id,
            "status": "unavailable",
            "label": None,
            "score": None,
            "synthetic_probability": None,
            "details": {"reason": "Install transformers and torch to enable this free local model."},
        }
    except Exception as exc:
        return {
            "name": model_id,
            "status": "error",
            "label": None,
            "score": None,
            "synthetic_probability": None,
            "details": {"reason": str(exc)[:300]},
        }

    normalized = _normalize_outputs(outputs)
    synthetic_probability = _synthetic_probability(normalized)
    top = normalized[0] if normalized else {"label": "unknown", "score": 0.0}
    return {
        "name": model_id,
        "status": "completed",
        "label": str(top["label"]),
        "score": round(float(top["score"]), 4),
        "synthetic_probability": round(synthetic_probability, 4),
        "details": {
            "model_provider": "huggingface_local",
            "top_labels": normalized[:5],
            "note": "Model output is an evidence signal, not proof of authenticity or manipulation.",
        },
    }


@functools.lru_cache(maxsize=4)
def _load_pipeline(model_id: str) -> Any:
    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:
        raise ImportError("transformers is not installed") from exc
    return pipeline("image-classification", model=model_id)


def _normalize_outputs(outputs: Any) -> List[Dict[str, Any]]:
    if isinstance(outputs, dict):
        outputs = [outputs]
    if not isinstance(outputs, list):
        return []
    normalized = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "label": str(item.get("label", "unknown")),
                "score": float(item.get("score") or 0.0),
            }
        )
    return sorted(normalized, key=lambda item: item["score"], reverse=True)


def _synthetic_probability(outputs: List[Dict[str, Any]]) -> float:
    synthetic = 0.0
    real = 0.0
    unknown = 0.0
    for item in outputs:
        label = _normalized_label(item["label"])
        score = float(item["score"])
        if any(marker in label for marker in REAL_LABEL_MARKERS):
            real += score
        elif any(marker in label for marker in SYNTHETIC_LABEL_MARKERS):
            synthetic += score
        else:
            unknown += score
    if synthetic == 0.0 and real == 0.0:
        return 0.5 if unknown > 0 else 0.0
    return max(0.0, min(1.0, synthetic / max(1e-6, synthetic + real)))


def _float_detail(details: Dict[str, Any], key: str) -> float | None:
    value = details.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _float_from(details: Dict[str, Any], key: str) -> float | None:
    value = details.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalized_label(label: str) -> str:
    return (
        str(label)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace(".", " ")
    )
