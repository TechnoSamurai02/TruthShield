from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageStat, UnidentifiedImageError

from analyzers.enhanced import enhance_image_result
from analyzers.image_forensics import analyze_image_forensics
from analyzers.metadata import extract_exif_metadata, has_camera_make_or_model, metadata_score
from analyzers.scoring import (
    DISCLAIMER,
    IMAGE_VIDEO_RECOMMENDATIONS,
    clamp_score,
    get_risk_level,
    summarize_result,
    unique_messages,
)


EXTENSION_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
}


def analyze_image_bytes(data: bytes, filename: str = "upload") -> Dict[str, Any]:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except UnidentifiedImageError as exc:
        raise ValueError("The image could not be opened safely.") from exc

    return _analyze_pil_image(
        image=image,
        filename=filename,
        include_metadata=True,
        content_bytes=data,
        content_label="image",
    )


def analyze_frame_array(frame_bgr: np.ndarray, frame_label: str = "video frame") -> Dict[str, Any]:
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("The video frame was empty.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    return _analyze_pil_image(
        image=image,
        filename=frame_label,
        include_metadata=False,
        content_bytes=None,
        content_label="video frame",
    )


def _analyze_pil_image(
    image: Image.Image,
    filename: str,
    include_metadata: bool,
    content_bytes: bytes | None,
    content_label: str,
) -> Dict[str, Any]:
    score = 70
    warnings: List[str] = []
    positives: List[str] = []

    image_format = (image.format or "").upper()
    width, height = image.size
    mode = image.mode
    extension = Path(filename).suffix.lower()

    exif: Dict[str, Any] = {}
    if include_metadata:
        exif = extract_exif_metadata(image)
        if exif:
            score += 8
            positives.append("Readable EXIF metadata is present.")
            if has_camera_make_or_model(exif):
                score += 8
                positives.append("Camera make or model metadata is present.")
            else:
                warnings.append("Metadata exists, but no camera make or model was found.")
        else:
            score -= 10
            warnings.append("No readable EXIF metadata was found.")
            if image_format in {"JPEG", "WEBP"}:
                score -= 8
                warnings.append("Metadata appears stripped or unavailable for a format that often carries it.")

        expected_format = EXTENSION_FORMATS.get(extension)
        if expected_format and image_format and expected_format != image_format:
            score -= 10
            warnings.append("File extension does not match the detected image format.")
        elif expected_format:
            positives.append("File extension matches the detected image format.")

    if _has_reasonable_dimensions(width, height):
        score += 4
        positives.append("Image dimensions are within a reasonable range.")
    else:
        score -= 5
        warnings.append("Image dimensions or aspect ratio are unusual.")

    entropy = _calculate_entropy(image)
    if 3.5 <= entropy <= 7.8:
        score += 5
        positives.append("Image entropy falls in a typical range.")
    else:
        score -= 8
        warnings.append("Image entropy is unusually low or high, which can indicate synthetic or degraded content.")

    blur_score = _laplacian_variance(image)
    if blur_score < 35:
        score -= 8
        warnings.append("Possible over-smoothing or heavy blur was detected.")

    forensics = analyze_image_forensics(image, content_bytes=content_bytes, filename=filename)
    forensic_warnings = list(forensics.get("warnings", []))
    forensic_positives = list(forensics.get("positive_signals", []))
    caption_like = bool((forensics.get("caption_overlay") or {}).get("is_likely"))

    compression = _compression_consistency_score(image)
    if compression["is_inconsistent"]:
        score -= 2 if caption_like else 7
        if caption_like:
            warnings.append("Compression varies across the image, likely influenced by an added caption or graphic overlay.")
        else:
            warnings.append("Compression or texture patterns vary sharply across the image.")
    else:
        score += 5
        positives.append("No major compression inconsistency was detected.")

    color_stats = _color_statistics(image)
    forensic_score = float(forensics.get("score", 50.0))
    synthetic_artifact_probability = float(forensics.get("synthetic_artifact_probability", 0.5))
    if synthetic_artifact_probability >= 0.62:
        score -= 12
    elif synthetic_artifact_probability <= 0.28:
        score += 6
    if forensic_score >= 75:
        score += 5
    elif forensic_score < 45:
        score -= 8
    warnings.extend(forensic_warnings)
    positives.extend(forensic_positives)

    visual_consistency = _visual_consistency_score(entropy, blur_score, width, height)
    final_score = clamp_score(score)
    risk_level, verdict = get_risk_level(final_score)

    evidence = {
        "metadata_score": metadata_score(exif) if include_metadata else 55.0,
        "visual_consistency_score": visual_consistency,
        "compression_score": compression["score"],
        "pixel_forensic_score": round(forensic_score, 2),
        "ai_artifact_score": round(synthetic_artifact_probability * 100.0, 2),
        "source_score": 50.0,
        "overall_risk_score": float(100 - final_score),
    }

    technical_details: Dict[str, Any] = {
        "filename": filename,
        "detected_format": image_format or "UNKNOWN",
        "mode": mode,
        "width": width,
        "height": height,
        "entropy": round(entropy, 3),
        "blur_laplacian_variance": round(blur_score, 3),
        "compression_consistency": compression,
        "forensic_analysis": forensics,
        "color_statistics": color_stats,
        "metadata_fields_found": sorted(exif.keys())[:20],
        "heuristic_note": "These signals are educational heuristics and are not proof of authenticity or manipulation.",
    }
    if content_bytes is not None:
        technical_details["file_size_bytes"] = len(content_bytes)

    result = {
        "content_type": "image" if content_label == "image" else "video_frame",
        "truth_score": final_score,
        "risk_level": risk_level,
        "verdict": verdict,
        "summary": summarize_result(content_label, final_score, warnings, positives),
        "warnings": unique_messages(warnings),
        "positive_signals": unique_messages(positives),
        "recommendations": IMAGE_VIDEO_RECOMMENDATIONS,
        "evidence": evidence,
        "technical_details": technical_details,
        "disclaimer": DISCLAIMER,
    }
    return enhance_image_result(
        result=result,
        image=image,
        filename=filename,
        content_bytes=content_bytes,
        content_label=content_label,
    )


def _has_reasonable_dimensions(width: int, height: int) -> bool:
    if width < 300 or height < 300:
        return False
    if width > 12000 or height > 12000:
        return False
    ratio = max(width, height) / max(1, min(width, height))
    return ratio <= 4.5


def _calculate_entropy(image: Image.Image) -> float:
    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    pixels = float(sum(histogram))
    entropy = 0.0
    for count in histogram:
        if count:
            probability = count / pixels
            entropy -= probability * math.log2(probability)
    return entropy


def _laplacian_variance(image: Image.Image) -> float:
    grayscale = np.array(image.convert("L"))
    return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())


def _compression_consistency_score(image: Image.Image) -> Dict[str, Any]:
    grayscale = np.array(image.convert("L"), dtype=np.float32)
    height, width = grayscale.shape
    block = max(32, min(width, height) // 6)
    block_scores: List[float] = []

    for y in range(0, height - block + 1, block):
        for x in range(0, width - block + 1, block):
            patch = grayscale[y : y + block, x : x + block]
            if patch.size == 0:
                continue
            blur = cv2.GaussianBlur(patch, (3, 3), 0)
            residual = patch - blur
            block_scores.append(float(np.std(residual)))

    if len(block_scores) < 4:
        return {"score": 60.0, "is_inconsistent": False, "block_noise_cv": 0.0}

    mean_noise = float(np.mean(block_scores))
    std_noise = float(np.std(block_scores))
    coefficient = std_noise / max(mean_noise, 1e-6)
    score = float(max(20.0, min(95.0, 95.0 - coefficient * 85.0)))
    return {
        "score": round(score, 2),
        "is_inconsistent": coefficient > 0.7,
        "block_noise_cv": round(coefficient, 3),
    }


def _visual_consistency_score(entropy: float, blur_score: float, width: int, height: int) -> float:
    score = 70.0
    if 3.5 <= entropy <= 7.8:
        score += 12.0
    else:
        score -= 18.0
    if blur_score < 35:
        score -= 18.0
    elif blur_score > 80:
        score += 8.0
    if _has_reasonable_dimensions(width, height):
        score += 8.0
    else:
        score -= 12.0
    return round(max(0.0, min(100.0, score)), 2)


def _color_statistics(image: Image.Image) -> Dict[str, Any]:
    rgb = image.convert("RGB")
    stat = ImageStat.Stat(rgb)
    channel_names = ["red", "green", "blue"]
    return {
        channel: {
            "mean": round(stat.mean[index], 2),
            "stddev": round(stat.stddev[index], 2),
        }
        for index, channel in enumerate(channel_names)
    }
