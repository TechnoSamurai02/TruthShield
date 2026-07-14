from __future__ import annotations

import functools
import math
from typing import Any, Dict, List, Sequence

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
def run_image_detectors(
    image: Image.Image,
    filename: str,
    metadata_present: bool,
    technical_details: Dict[str, Any],
    model_ids: Sequence[str] | None = None,
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

    configured_models = list(model_ids) if model_ids is not None else settings.ai_image_detector_models
    for model_id in configured_models:
        results.append(_run_huggingface_detector(image, model_id))
    return results


def run_tiled_image_detectors(
    image: Image.Image,
    model_ids: Sequence[str],
    tile_size: int = 448,
    overlap: float = 0.15,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """Run every configured model over tiles that collectively cover every source pixel."""
    settings = get_settings()
    if not settings.enable_local_ai_models or not model_ids:
        return []

    rgb = image.convert("RGB")
    tiles, boxes = _covering_tiles(rgb, tile_size=max(224, tile_size), overlap=overlap)
    results: List[Dict[str, Any]] = []
    for model_id in model_ids:
        results.append(_run_huggingface_detector_batch(tiles, boxes, model_id, batch_size=batch_size))
    return results


def reuse_full_frame_predictions_as_single_tile(
    detectors: Sequence[Dict[str, Any]],
    model_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    """Represent a small full frame as its one covering tile without a duplicate model call."""
    by_name = {str(detector.get("name") or ""): detector for detector in detectors}
    reused = []
    for model_id in model_ids:
        source = by_name.get(str(model_id))
        if not source or source.get("status") != "completed":
            continue
        reused.append(
            {
                "name": f"{model_id}:tiled_pixel_scan",
                "status": "completed",
                "label": source.get("label"),
                "score": source.get("score"),
                "synthetic_probability": source.get("synthetic_probability"),
                "details": {
                    "model_provider": "huggingface_local",
                    "tile_count": 1,
                    "source_pixel_coverage": 1.0,
                    "reused_full_frame_prediction": True,
                    "note": "The frame fits inside one tile, so its full-frame prediction is also the single-tile prediction.",
                },
            }
        )
    return reused


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
    learned_probabilities = []
    truthshield_probabilities = []
    fallback_probabilities = []
    for detector in detectors:
        probability = detector.get("synthetic_probability")
        if not isinstance(probability, (int, float)) or not math.isfinite(float(probability)):
            continue
        name = str(detector.get("name") or "")
        if name == "local_heuristic_synthetic_likelihood":
            fallback_probabilities.append((float(probability), 1.0))
            continue
        elif name.endswith(":tiled_pixel_scan"):
            if (detector.get("details") or {}).get("reused_full_frame_prediction"):
                continue
            weight = 0.75
        else:
            weight = 1.0
        if detector.get("status") == "completed":
            learned_probabilities.append((float(probability), weight))
            if "truthshield-image-detector" in name.lower():
                truthshield_probabilities.append((float(probability), weight))

    # Hand-written image statistics are only a fallback. Averaging them into a
    # completed learned model can erase a correct high-confidence prediction
    # (modern generated images often have perfectly ordinary entropy, sharpness,
    # and compression). This was the main cause of contradictory app reports.
    weighted_probabilities = truthshield_probabilities or learned_probabilities or fallback_probabilities
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
        if detector.get("status") == "completed"
        and detector.get("name") != "local_heuristic_synthetic_likelihood"
        and isinstance(detector.get("synthetic_probability"), (int, float))
        and math.isfinite(float(detector["synthetic_probability"]))
    )


def _heuristic_synthetic_detector(filename: str, metadata_present: bool, technical_details: Dict[str, Any]) -> Dict[str, Any]:
    probability = 0.14
    reasons: List[str] = []
    forensic = technical_details.get("forensic_analysis")
    forensic = forensic if isinstance(forensic, dict) else {}
    caption_overlay = forensic.get("caption_overlay")
    caption_like = isinstance(caption_overlay, dict) and bool(caption_overlay.get("is_likely"))
    forensic_synthetic = _float_from(forensic, "synthetic_artifact_probability")
    forensic_manipulation = _float_from(forensic, "manipulation_probability")

    # Filenames, PNG/WEBP formats, and missing metadata are deliberately not
    # scored. They are easy to change and are common for genuine downloads,
    # screenshots, conversions, and social-media images.
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
    label = (
        "elevated_supporting_signal"
        if probability >= 0.65
        else "uncertain_supporting_signal"
        if probability >= 0.35
        else "lower_supporting_signal"
    )
    return {
        "name": "local_heuristic_synthetic_likelihood",
        "status": "completed",
        "label": label,
        "score": round(probability, 3),
        "synthetic_probability": round(probability, 3),
        "details": {
            "reasons": reasons,
            "model_type": "deterministic_supporting_heuristic",
            "note": "This fallback cannot independently produce an AI verdict.",
        },
    }


def _run_huggingface_detector(image: Image.Image, model_id: str) -> Dict[str, Any]:
    try:
        classifier = _load_pipeline(model_id)
        prepared = _prepare_classifier_image(classifier, image)
        try:
            outputs = classifier(prepared, top_k=None)
        except TypeError:
            outputs = classifier(prepared)
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
    if synthetic_probability is None:
        return {
            "name": model_id,
            "status": "unsupported_labels" if normalized else "error",
            "label": str(top["label"]) if normalized else None,
            "score": round(float(top["score"]), 4) if normalized else None,
            "synthetic_probability": None,
            "details": {
                "model_provider": "huggingface_local",
                "top_labels": normalized[:5],
                "reason": (
                    "The model labels could not be mapped unambiguously to synthetic and authentic classes."
                    if normalized
                    else "The model returned no usable predictions."
                ),
            },
        }
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


def _run_huggingface_detector_batch(
    images: Sequence[Image.Image],
    boxes: Sequence[tuple[int, int, int, int]],
    model_id: str,
    batch_size: int,
) -> Dict[str, Any]:
    prepared_images = list(images)
    try:
        classifier = _load_pipeline(model_id)
        prepared_images = [_prepare_classifier_image(classifier, image) for image in images]
        outputs = classifier(prepared_images, top_k=None, batch_size=max(1, batch_size))
    except TypeError:
        try:
            outputs = classifier(prepared_images, batch_size=max(1, batch_size))
        except Exception as exc:
            return _model_error(model_id, exc, suffix=":tiled_pixel_scan")
    except ImportError:
        return {
            "name": f"{model_id}:tiled_pixel_scan",
            "status": "unavailable",
            "label": None,
            "score": None,
            "synthetic_probability": None,
            "details": {"reason": "Install transformers and torch to enable tiled model analysis."},
        }
    except Exception as exc:
        return _model_error(model_id, exc, suffix=":tiled_pixel_scan")

    if outputs and isinstance(outputs, list) and isinstance(outputs[0], dict):
        outputs = [outputs]
    normalized_tiles = [_normalize_outputs(output) for output in outputs] if isinstance(outputs, list) else []
    probabilities = [_synthetic_probability(output) for output in normalized_tiles if output]
    probabilities = [probability for probability in probabilities if probability is not None]
    if not probabilities:
        return {
            "name": f"{model_id}:tiled_pixel_scan",
            "status": "error",
            "label": None,
            "score": None,
            "synthetic_probability": None,
            "details": {"reason": "The model returned no usable tile predictions."},
        }

    ordered = sorted(probabilities)
    mean_probability = sum(ordered) / len(ordered)
    p90 = _percentile(ordered, 0.90)
    p95 = _percentile(ordered, 0.95)
    suspicious_ratio = sum(probability >= 0.65 for probability in ordered) / len(ordered)
    aggregate = mean_probability * 0.55 + p90 * 0.30 + suspicious_ratio * 0.15
    aggregate = max(0.0, min(1.0, aggregate))
    highest_indices = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)[:5]
    return {
        "name": f"{model_id}:tiled_pixel_scan",
        "status": "completed",
        "label": "tiled_regions_likely_synthetic" if aggregate >= 0.65 else "tiled_regions_uncertain" if aggregate >= 0.35 else "tiled_regions_lower_signal",
        "score": round(aggregate, 4),
        "synthetic_probability": round(aggregate, 4),
        "details": {
            "model_provider": "huggingface_local",
            "tile_count": len(probabilities),
            "source_pixel_coverage": 1.0,
            "mean_probability": round(mean_probability, 4),
            "p90_probability": round(p90, 4),
            "p95_probability": round(p95, 4),
            "maximum_probability": round(max(ordered), 4),
            "suspicious_tile_ratio": round(suspicious_ratio, 4),
            "highest_risk_tiles": [
                {"box": list(boxes[index]), "synthetic_probability": round(probabilities[index], 4)}
                for index in highest_indices
                if index < len(boxes)
            ],
            "note": "Tiles cover the full frame. Each tile is resized to the model input size, so this is supporting evidence rather than literal native-resolution classification.",
        },
    }


def _model_error(model_id: str, exc: Exception, suffix: str = "") -> Dict[str, Any]:
    return {
        "name": f"{model_id}{suffix}",
        "status": "error",
        "label": None,
        "score": None,
        "synthetic_probability": None,
        "details": {"reason": str(exc)[:300]},
    }


def _covering_tiles(
    image: Image.Image,
    tile_size: int,
    overlap: float,
) -> tuple[List[Image.Image], List[tuple[int, int, int, int]]]:
    width, height = image.size
    tile_width = min(tile_size, width)
    tile_height = min(tile_size, height)
    step_x = max(1, int(tile_width * (1.0 - max(0.0, min(0.45, overlap)))))
    step_y = max(1, int(tile_height * (1.0 - max(0.0, min(0.45, overlap)))))
    x_starts = _tile_starts(width, tile_width, step_x)
    y_starts = _tile_starts(height, tile_height, step_y)
    boxes = [(x, y, x + tile_width, y + tile_height) for y in y_starts for x in x_starts]
    return [image.crop(box) for box in boxes], boxes


def _tile_starts(length: int, tile_length: int, step: int) -> List[int]:
    if length <= tile_length:
        return [0]
    starts = list(range(0, length - tile_length + 1, step))
    final_start = length - tile_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _percentile(ordered_values: Sequence[float], quantile: float) -> float:
    if not ordered_values:
        return 0.0
    position = max(0.0, min(1.0, quantile)) * (len(ordered_values) - 1)
    lower = int(position)
    upper = min(len(ordered_values) - 1, lower + 1)
    fraction = position - lower
    return float(ordered_values[lower] * (1.0 - fraction) + ordered_values[upper] * fraction)


@functools.lru_cache(maxsize=4)
def _load_pipeline(model_id: str) -> Any:
    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:
        raise ImportError("transformers is not installed") from exc
    return pipeline("image-classification", model=model_id)


def _prepare_classifier_image(classifier: Any, image: Image.Image) -> Image.Image:
    """Reproduce a local model's declared training-frame encoding when required."""
    config = getattr(getattr(classifier, "model", None), "config", None)
    encoding = str(getattr(config, "truthshield_training_frame_encoding", "") or "")
    rgb = image.convert("RGB")
    if encoding == "opencv_jpeg_95":
        try:
            import cv2
            import numpy as np

            bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            success, encoded = cv2.imencode(
                ".jpg",
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            if success:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded is not None:
                    rgb = Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
        except Exception:
            pass

    max_dimension = int(getattr(config, "truthshield_preprocess_max_dimension", 0) or 0)
    width, height = rgb.size
    longest = max(width, height)
    if max_dimension > 0 and longest > max_dimension:
        scale = max_dimension / float(longest)
        rgb = rgb.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.Resampling.LANCZOS,
        )
    return rgb


def _normalize_outputs(outputs: Any) -> List[Dict[str, Any]]:
    if isinstance(outputs, dict):
        outputs = [outputs]
    if not isinstance(outputs, list):
        return []
    normalized = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score) or score < 0.0:
            continue
        normalized.append({"label": str(item.get("label", "unknown")), "score": score})
    return sorted(normalized, key=lambda item: item["score"], reverse=True)


def _synthetic_probability(outputs: List[Dict[str, Any]]) -> float | None:
    synthetic = 0.0
    real = 0.0
    unknown = 0.0
    for item in outputs:
        label = _normalized_label(item["label"])
        score = float(item["score"])
        if not math.isfinite(score) or score < 0.0:
            continue
        if any(marker in label for marker in REAL_LABEL_MARKERS):
            real += score
        elif any(marker in label for marker in SYNTHETIC_LABEL_MARKERS):
            synthetic += score
        else:
            unknown += score
    if synthetic == 0.0 and real == 0.0:
        return None
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
