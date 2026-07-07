from __future__ import annotations

import io
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


def analyze_image_forensics(
    image: Image.Image,
    content_bytes: bytes | None = None,
    filename: str = "upload",
) -> Dict[str, Any]:
    rgb_image = image.convert("RGB")
    rgb = np.asarray(rgb_image, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    caption = _caption_overlay_stats(gray)
    noise = _noise_residual_stats(gray)
    frequency = _frequency_stats(gray)
    blockiness = _jpeg_blockiness(gray)
    ela = _ela_stats(rgb_image)
    duplicate = _duplicate_patch_stats(gray)
    channel = _channel_stats(rgb)
    clipping = _clipping_stats(rgb)

    synthetic_probability = _synthetic_probability(
        caption=caption,
        noise=noise,
        frequency=frequency,
        blockiness=blockiness,
        ela=ela,
        duplicate=duplicate,
        clipping=clipping,
    )
    manipulation_probability = _manipulation_probability(
        caption=caption,
        blockiness=blockiness,
        ela=ela,
        duplicate=duplicate,
    )
    forensic_score = 100.0 - (synthetic_probability * 72.0 + manipulation_probability * 28.0)
    forensic_score = max(5.0, min(98.0, forensic_score))

    warnings, positives = _forensic_messages(
        synthetic_probability=synthetic_probability,
        manipulation_probability=manipulation_probability,
        caption=caption,
        noise=noise,
        frequency=frequency,
        blockiness=blockiness,
        ela=ela,
        duplicate=duplicate,
    )

    return {
        "score": round(forensic_score, 2),
        "synthetic_artifact_probability": round(synthetic_probability, 3),
        "manipulation_probability": round(manipulation_probability, 3),
        "caption_overlay": caption,
        "noise_residual": noise,
        "frequency_spectrum": frequency,
        "jpeg_blockiness": blockiness,
        "error_level_analysis": ela,
        "duplicate_patch_analysis": duplicate,
        "channel_statistics": channel,
        "clipping": clipping,
        "warnings": warnings,
        "positive_signals": positives,
        "notes": [
            "Pixel-level checks are forensic clues, not proof.",
            "Captions, screenshots, crops, and social-media recompression can mimic manipulation artifacts.",
        ],
        "filename": filename,
        "bytes_available": content_bytes is not None,
    }


def _caption_overlay_stats(gray: np.ndarray) -> Dict[str, Any]:
    height, width = gray.shape
    image_area = float(height * width)
    edge_map = cv2.Canny(gray, 80, 180)
    dark_pixels = gray < 48
    results: List[Dict[str, Any]] = []

    bands = {
        "top": (0, int(height * 0.38)),
        "bottom": (int(height * 0.62), height),
    }
    for name, (start_y, end_y) in bands.items():
        if end_y <= start_y:
            continue
        band_dark = dark_pixels[start_y:end_y].astype(np.uint8) * 255
        band_edges = edge_map[start_y:end_y]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        merged = cv2.dilate(band_dark, kernel, iterations=1)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
        candidates = []
        for index in range(1, component_count):
            x, y, component_width, component_height, area = stats[index]
            if area < image_area * 0.00035 or area > image_area * 0.08:
                continue
            if component_width < width * 0.012 or component_height < height * 0.025:
                continue
            aspect_ratio = component_width / max(1, component_height)
            if 0.08 <= aspect_ratio <= 14.0:
                candidates.append(
                    {
                        "x": int(x),
                        "y": int(y + start_y),
                        "width": int(component_width),
                        "height": int(component_height),
                        "area_ratio": round(float(area) / image_area, 5),
                    }
                )
        band_area = float((end_y - start_y) * width)
        edge_density = float(np.count_nonzero(band_edges)) / max(1.0, band_area)
        dark_density = float(np.count_nonzero(band_dark)) / max(1.0, band_area)
        results.append(
            {
                "band": name,
                "candidate_components": candidates[:8],
                "component_count": len(candidates),
                "edge_density": round(edge_density, 4),
                "dark_pixel_density": round(dark_density, 4),
            }
        )

    best = max(results, key=lambda item: (item["component_count"], item["edge_density"]), default={})
    best_candidates = best.get("candidate_components", []) if best else []
    large_caption_word = any(
        isinstance(candidate, dict)
        and candidate.get("width", 0) >= width * 0.12
        and candidate.get("height", 0) >= height * 0.06
        and candidate.get("area_ratio", 0.0) >= 0.008
        for candidate in best_candidates
    )
    likely = bool(
        best
        and (
            (
                best.get("component_count", 0) >= 2
                and (best.get("edge_density", 0.0) >= 0.035 or best.get("dark_pixel_density", 0.0) >= 0.018)
            )
            or (
                large_caption_word
                and best.get("edge_density", 0.0) >= 0.014
                and best.get("dark_pixel_density", 0.0) >= 0.014
            )
        )
    )
    confidence = 0.0
    if best:
        confidence = min(
            0.94,
            best.get("component_count", 0) * 0.16
            + best.get("edge_density", 0.0) * 4.0
            + best.get("dark_pixel_density", 0.0) * 3.0,
        )
    return {
        "is_likely": likely,
        "confidence": round(confidence if likely else min(confidence, 0.45), 2),
        "location": best.get("band") if likely else None,
        "bands": results,
        "explanation": (
            "Large high-contrast components in a top or bottom band look like added caption text."
            if likely
            else "No strong caption-like overlay pattern was isolated."
        ),
    }


def _noise_residual_stats(gray: np.ndarray) -> Dict[str, Any]:
    gray_f = gray.astype(np.float32)
    blur = cv2.GaussianBlur(gray_f, (5, 5), 0)
    residual = gray_f - blur
    edges = cv2.Canny(gray, 70, 170)
    flat_mask = edges == 0
    edge_mask = edges > 0

    block = max(32, min(gray.shape) // 8)
    block_stds: List[float] = []
    for y in range(0, gray.shape[0] - block + 1, block):
        for x in range(0, gray.shape[1] - block + 1, block):
            patch = residual[y : y + block, x : x + block]
            if patch.size:
                block_stds.append(float(np.std(patch)))

    mean_noise = float(np.mean(block_stds)) if block_stds else float(np.std(residual))
    std_noise = float(np.std(block_stds)) if block_stds else 0.0
    coefficient = std_noise / max(mean_noise, 1e-6)
    flat_noise = float(np.std(residual[flat_mask])) if np.any(flat_mask) else mean_noise
    edge_noise = float(np.std(residual[edge_mask])) if np.any(edge_mask) else mean_noise
    edge_to_flat = edge_noise / max(flat_noise, 1e-6)

    return {
        "mean_residual_noise": round(mean_noise, 3),
        "block_noise_cv": round(coefficient, 3),
        "flat_region_noise": round(flat_noise, 3),
        "edge_region_noise": round(edge_noise, 3),
        "edge_to_flat_noise_ratio": round(edge_to_flat, 3),
        "interpretation": _noise_interpretation(mean_noise, coefficient, edge_to_flat),
    }


def _frequency_stats(gray: np.ndarray) -> Dict[str, Any]:
    resized = _resize_for_analysis(gray, 512)
    centered = resized.astype(np.float32) - float(np.mean(resized))
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    magnitude = np.log1p(np.abs(spectrum))
    height, width = magnitude.shape
    yy, xx = np.indices((height, width))
    radius = np.sqrt((yy - height / 2.0) ** 2 + (xx - width / 2.0) ** 2)
    radius = radius / max(radius.max(), 1e-6)
    total = float(np.sum(magnitude)) or 1.0
    high_frequency_ratio = float(np.sum(magnitude[radius > 0.42]) / total)
    mid_frequency_ratio = float(np.sum(magnitude[(radius > 0.16) & (radius <= 0.42)]) / total)
    non_center = magnitude[radius > 0.08]
    median = float(np.median(non_center)) if non_center.size else 0.0
    p999 = float(np.percentile(non_center, 99.9)) if non_center.size else 0.0
    spike_ratio = p999 / max(median, 1e-6)
    return {
        "high_frequency_ratio": round(high_frequency_ratio, 4),
        "mid_frequency_ratio": round(mid_frequency_ratio, 4),
        "spectral_spike_ratio": round(spike_ratio, 3),
        "interpretation": _frequency_interpretation(high_frequency_ratio, spike_ratio),
    }


def _jpeg_blockiness(gray: np.ndarray) -> Dict[str, Any]:
    gray_f = gray.astype(np.float32)
    vertical_diff = np.abs(np.diff(gray_f, axis=1))
    horizontal_diff = np.abs(np.diff(gray_f, axis=0))

    vertical_indices = np.arange(vertical_diff.shape[1])
    horizontal_indices = np.arange(horizontal_diff.shape[0])
    vertical_boundary = vertical_indices % 8 == 7
    horizontal_boundary = horizontal_indices % 8 == 7

    boundary_values = []
    non_boundary_values = []
    if np.any(vertical_boundary):
        boundary_values.append(vertical_diff[:, vertical_boundary].ravel())
        non_boundary_values.append(vertical_diff[:, ~vertical_boundary].ravel())
    if np.any(horizontal_boundary):
        boundary_values.append(horizontal_diff[horizontal_boundary, :].ravel())
        non_boundary_values.append(horizontal_diff[~horizontal_boundary, :].ravel())
    boundary = np.concatenate(boundary_values) if boundary_values else np.array([], dtype=np.float32)
    non_boundary = np.concatenate(non_boundary_values) if non_boundary_values else np.array([], dtype=np.float32)
    boundary_mean = float(np.mean(boundary)) if boundary.size else 0.0
    non_boundary_mean = float(np.mean(non_boundary)) if non_boundary.size else 0.0
    ratio = boundary_mean / max(non_boundary_mean, 1e-6)
    return {
        "boundary_difference_mean": round(boundary_mean, 3),
        "non_boundary_difference_mean": round(non_boundary_mean, 3),
        "blockiness_ratio": round(ratio, 3),
        "interpretation": _blockiness_interpretation(ratio),
    }


def _ela_stats(image: Image.Image) -> Dict[str, Any]:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=False)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    original = np.asarray(image, dtype=np.int16)
    recompressed_arr = np.asarray(recompressed, dtype=np.int16)
    diff = np.mean(np.abs(original - recompressed_arr), axis=2)
    mean = float(np.mean(diff))
    std = float(np.std(diff))
    p95 = float(np.percentile(diff, 95))
    threshold = mean + std * 2.2
    hot_ratio = float(np.count_nonzero(diff > threshold)) / max(1.0, float(diff.size))
    return {
        "mean_error": round(mean, 3),
        "p95_error": round(p95, 3),
        "localized_hot_area_ratio": round(hot_ratio, 4),
        "interpretation": _ela_interpretation(hot_ratio, p95),
    }


def _duplicate_patch_stats(gray: np.ndarray) -> Dict[str, Any]:
    resized = _resize_for_analysis(gray, 640)
    block = 32
    hashes: Dict[str, List[Tuple[int, int]]] = {}
    textured_blocks = 0
    for y in range(0, resized.shape[0] - block + 1, block):
        for x in range(0, resized.shape[1] - block + 1, block):
            patch = resized[y : y + block, x : x + block]
            if float(np.std(patch)) < 8.0:
                continue
            textured_blocks += 1
            small = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA)
            bits = small >= float(np.mean(small))
            patch_hash = f"{int(''.join('1' if bit else '0' for bit in bits.flatten()), 2):016x}"
            hashes.setdefault(patch_hash, []).append((x, y))

    duplicate_groups = 0
    duplicate_blocks = 0
    for coords in hashes.values():
        if len(coords) < 2:
            continue
        if _has_far_apart_pair(coords, block * 2):
            duplicate_groups += 1
            duplicate_blocks += len(coords)
    ratio = duplicate_blocks / max(1, textured_blocks)
    return {
        "textured_blocks": textured_blocks,
        "duplicate_groups": duplicate_groups,
        "duplicate_patch_ratio": round(float(ratio), 4),
        "interpretation": (
            "Repeated textured patches may indicate cloning, tiling, or generated texture repetition."
            if ratio > 0.08
            else "No strong repeated textured-patch pattern was found."
        ),
    }


def _channel_stats(rgb: np.ndarray) -> Dict[str, Any]:
    reshaped = rgb.reshape(-1, 3).astype(np.float32)
    correlations = {}
    pairs = {"red_green": (0, 1), "red_blue": (0, 2), "green_blue": (1, 2)}
    for name, (first, second) in pairs.items():
        if float(np.std(reshaped[:, first])) < 1e-6 or float(np.std(reshaped[:, second])) < 1e-6:
            correlations[name] = 0.0
        else:
            correlations[name] = round(float(np.corrcoef(reshaped[:, first], reshaped[:, second])[0, 1]), 4)
    return {"channel_correlations": correlations}


def _clipping_stats(rgb: np.ndarray) -> Dict[str, Any]:
    dark_ratio = float(np.count_nonzero(np.all(rgb <= 5, axis=2))) / max(1.0, float(rgb.shape[0] * rgb.shape[1]))
    bright_ratio = float(np.count_nonzero(np.all(rgb >= 250, axis=2))) / max(1.0, float(rgb.shape[0] * rgb.shape[1]))
    return {
        "near_black_pixel_ratio": round(dark_ratio, 4),
        "near_white_pixel_ratio": round(bright_ratio, 4),
    }


def _synthetic_probability(
    caption: Dict[str, Any],
    noise: Dict[str, Any],
    frequency: Dict[str, Any],
    blockiness: Dict[str, Any],
    ela: Dict[str, Any],
    duplicate: Dict[str, Any],
    clipping: Dict[str, Any],
) -> float:
    probability = 0.16
    caption_likely = bool(caption.get("is_likely"))
    mean_noise = float(noise.get("mean_residual_noise", 0.0))
    noise_cv = float(noise.get("block_noise_cv", 0.0))
    edge_to_flat = float(noise.get("edge_to_flat_noise_ratio", 0.0))
    high_frequency = float(frequency.get("high_frequency_ratio", 0.0))
    spike_ratio = float(frequency.get("spectral_spike_ratio", 0.0))
    blockiness_ratio = float(blockiness.get("blockiness_ratio", 1.0))
    hot_ratio = float(ela.get("localized_hot_area_ratio", 0.0))
    duplicate_ratio = float(duplicate.get("duplicate_patch_ratio", 0.0))
    bright_ratio = float(clipping.get("near_white_pixel_ratio", 0.0))

    if mean_noise < 1.2:
        probability += 0.16
    elif mean_noise > 2.5:
        probability -= 0.04
    if noise_cv < 0.16 and mean_noise < 2.4:
        probability += 0.08
    if edge_to_flat > 5.5 and not caption_likely:
        probability += 0.08
    if high_frequency < 0.06:
        probability += 0.08
    elif high_frequency > 0.11:
        probability -= 0.04
    if spike_ratio > 7.5 and not caption_likely:
        probability += 0.09
    if blockiness_ratio > 1.42:
        probability += 0.04
    if hot_ratio > 0.12 and not caption_likely:
        probability += 0.06
    if duplicate_ratio > 0.08:
        probability += 0.12
    if bright_ratio > 0.08 and not caption_likely:
        probability += 0.03
    if caption_likely:
        probability -= 0.07

    return max(0.03, min(0.92, probability))


def _manipulation_probability(
    caption: Dict[str, Any],
    blockiness: Dict[str, Any],
    ela: Dict[str, Any],
    duplicate: Dict[str, Any],
) -> float:
    probability = 0.12
    if caption.get("is_likely"):
        probability += 0.30
    if float(ela.get("localized_hot_area_ratio", 0.0)) > 0.12:
        probability += 0.14
    if float(blockiness.get("blockiness_ratio", 1.0)) > 1.45:
        probability += 0.05
    if float(duplicate.get("duplicate_patch_ratio", 0.0)) > 0.08:
        probability += 0.18
    return max(0.03, min(0.9, probability))


def _forensic_messages(
    synthetic_probability: float,
    manipulation_probability: float,
    caption: Dict[str, Any],
    noise: Dict[str, Any],
    frequency: Dict[str, Any],
    blockiness: Dict[str, Any],
    ela: Dict[str, Any],
    duplicate: Dict[str, Any],
) -> tuple[List[str], List[str]]:
    warnings: List[str] = []
    positives: List[str] = []
    if caption.get("is_likely"):
        warnings.append("A caption or graphic overlay was detected; treat this as an edited/meme image, not as raw camera output.")
        positives.append("The caption-like overlay explains some compression and edge artifacts without proving AI generation.")
    if synthetic_probability >= 0.62:
        warnings.append("Pixel-level forensic checks found several synthetic-image artifact signals.")
    elif synthetic_probability <= 0.28:
        positives.append("Pixel-level forensic checks did not find strong AI-generation artifacts.")
    if manipulation_probability >= 0.58 and not caption.get("is_likely"):
        warnings.append("Pixel-level forensic checks found possible editing or compositing artifacts.")
    if float(noise.get("mean_residual_noise", 0.0)) < 1.2:
        warnings.append("Fine-grain noise is unusually low, which can happen after generation, smoothing, or heavy recompression.")
    if float(frequency.get("spectral_spike_ratio", 0.0)) > 7.5 and not caption.get("is_likely"):
        warnings.append("Frequency analysis found unusually strong periodic artifacts.")
    if float(blockiness.get("blockiness_ratio", 1.0)) <= 1.35:
        positives.append("JPEG block-boundary artifacts are not unusually strong.")
    if float(ela.get("localized_hot_area_ratio", 0.0)) <= 0.12:
        positives.append("Error-level analysis did not find a large localized recompression hotspot.")
    if float(duplicate.get("duplicate_patch_ratio", 0.0)) > 0.08:
        warnings.append("Repeated textured patches were found, which can indicate cloning or generated texture repetition.")
    else:
        positives.append("No strong repeated textured-patch pattern was found.")
    return warnings, positives


def _noise_interpretation(mean_noise: float, coefficient: float, edge_to_flat: float) -> str:
    if mean_noise < 1.2:
        return "very smooth residual noise"
    if coefficient > 1.0 or edge_to_flat > 5.5:
        return "uneven residual noise"
    return "residual noise is within a normal broad range"


def _frequency_interpretation(high_frequency_ratio: float, spike_ratio: float) -> str:
    if high_frequency_ratio < 0.06:
        return "fine detail is low"
    if spike_ratio > 7.5:
        return "periodic frequency spikes are elevated"
    return "frequency distribution is within a normal broad range"


def _blockiness_interpretation(ratio: float) -> str:
    if ratio > 1.45:
        return "strong JPEG block-boundary pattern"
    if ratio < 0.7:
        return "weak or non-JPEG-like block-boundary pattern"
    return "JPEG block-boundary pattern is not unusually strong"


def _ela_interpretation(hot_ratio: float, p95: float) -> str:
    if hot_ratio > 0.18:
        return "localized recompression differences are elevated"
    if p95 < 1.0:
        return "error level is very low"
    return "no large localized recompression hotspot"


def _resize_for_analysis(gray: np.ndarray, max_side: int) -> np.ndarray:
    height, width = gray.shape
    longest = max(height, width)
    if longest <= max_side:
        return gray
    scale = max_side / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)


def _has_far_apart_pair(coords: List[Tuple[int, int]], minimum_distance: int) -> bool:
    for index, first in enumerate(coords):
        for second in coords[index + 1 :]:
            if abs(first[0] - second[0]) + abs(first[1] - second[1]) >= minimum_distance:
                return True
    return False
