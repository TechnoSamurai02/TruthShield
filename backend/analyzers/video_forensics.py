from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from analyzers.ai_detectors import combined_synthetic_probability


VIDEO_FEATURE_NAMES = (
    "frame_ai_probability_mean",
    "frame_ai_probability_std",
    "frame_ai_probability_p90",
    "frame_ai_probability_p95",
    "frame_ai_high_risk_ratio",
    "frame_ai_sustained_ratio",
    "pixel_forensic_probability_mean",
    "pixel_forensic_probability_p95",
    "frame_truth_score_mean",
    "frame_truth_score_std",
    "frame_truth_score_p10",
    "luma_delta_mean",
    "luma_delta_std",
    "luma_delta_p95",
    "noise_delta_mean",
    "edge_density_delta_mean",
    "flow_warp_error_mean",
    "flow_warp_error_p95",
    "flow_inconsistency_mean",
    "duplicate_frame_ratio",
    "scene_cut_ratio",
)


class VideoForensicsAccumulator:
    """Collect full-frame and consecutive-frame evidence without retaining video frames."""

    def __init__(self) -> None:
        self.frames_seen = 0
        self.native_pixels_examined = 0
        self.tile_count = 0
        self.tiled_pixels_examined = 0
        self.frame_ai_probabilities: List[float] = []
        self.pixel_forensic_probabilities: List[float] = []
        self.frame_truth_scores: List[float] = []
        self.luma_deltas: List[float] = []
        self.noise_deltas: List[float] = []
        self.edge_density_deltas: List[float] = []
        self.flow_warp_errors: List[float] = []
        self.flow_inconsistencies: List[float] = []
        self.scene_cuts = 0
        self.duplicate_frames = 0
        self._previous_gray: np.ndarray | None = None
        self._previous_histogram: np.ndarray | None = None
        self._previous_noise = 0.0
        self._previous_edge_density = 0.0

    def update(self, frame_bgr: np.ndarray, frame_result: Dict[str, Any]) -> Dict[str, float | bool | None]:
        height, width = frame_bgr.shape[:2]
        self.frames_seen += 1
        self.native_pixels_examined += int(width * height)

        detectors = frame_result.get("detectors") or []
        frame_probability = combined_synthetic_probability(detectors)
        if frame_probability is not None:
            self.frame_ai_probabilities.append(float(frame_probability))
        self.frame_truth_scores.append(float(frame_result.get("truth_score", 50.0)))

        technical = frame_result.get("technical_details") or {}
        forensic = technical.get("forensic_analysis") or {}
        pixel_probability = forensic.get("synthetic_artifact_probability")
        if isinstance(pixel_probability, (int, float)):
            self.pixel_forensic_probabilities.append(float(pixel_probability))

        tile_probability: float | None = None
        completed_tile_scan = False
        for detector in detectors:
            if not str(detector.get("name") or "").endswith(":tiled_pixel_scan"):
                continue
            details = detector.get("details") or {}
            self.tile_count += int(details.get("tile_count") or 0)
            completed_tile_scan = completed_tile_scan or detector.get("status") == "completed"
            value = detector.get("synthetic_probability")
            if isinstance(value, (int, float)):
                tile_probability = max(tile_probability or 0.0, float(value))
        if completed_tile_scan:
            self.tiled_pixels_examined += width * height

        gray_native = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = _resize_for_temporal_checks(gray_native, max_width=384)
        edge_density = float(np.count_nonzero(cv2.Canny(gray, 70, 170))) / max(1.0, float(gray.size))
        residual = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
        noise_level = float(np.std(residual))
        histogram = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(histogram, histogram)

        luma_delta: float | None = None
        scene_cut = False
        duplicate = False
        if self._previous_gray is not None and self._previous_histogram is not None:
            luma_delta = float(np.mean(cv2.absdiff(gray, self._previous_gray))) / 255.0
            histogram_correlation = float(
                cv2.compareHist(self._previous_histogram, histogram, cv2.HISTCMP_CORREL)
            )
            scene_cut = bool(luma_delta >= 0.18 and histogram_correlation < 0.65)
            duplicate = bool(luma_delta <= 0.0015)
            self.luma_deltas.append(luma_delta)
            self.noise_deltas.append(
                abs(noise_level - self._previous_noise) / max(1.0, noise_level, self._previous_noise)
            )
            self.edge_density_deltas.append(abs(edge_density - self._previous_edge_density))
            if scene_cut:
                self.scene_cuts += 1
            elif duplicate:
                self.duplicate_frames += 1
            else:
                warp_error, flow_inconsistency = _optical_flow_consistency(self._previous_gray, gray)
                self.flow_warp_errors.append(warp_error)
                self.flow_inconsistencies.append(flow_inconsistency)

        self._previous_gray = gray
        self._previous_histogram = histogram
        self._previous_noise = noise_level
        self._previous_edge_density = edge_density
        return {
            "synthetic_probability": frame_probability,
            "tile_synthetic_probability": tile_probability,
            "luma_delta": luma_delta,
            "scene_cut": scene_cut,
            "duplicate_frame": duplicate,
        }

    def model_features(self) -> Dict[str, float]:
        frame_count = max(1, self.frames_seen)
        pair_count = max(1, self.frames_seen - 1)
        ai_values = self.frame_ai_probabilities
        truth_values = self.frame_truth_scores
        pixel_values = self.pixel_forensic_probabilities
        return {
            "frame_ai_probability_mean": _mean(ai_values, 0.5),
            "frame_ai_probability_std": _std(ai_values),
            "frame_ai_probability_p90": _percentile(ai_values, 0.90, 0.5),
            "frame_ai_probability_p95": _percentile(ai_values, 0.95, 0.5),
            "frame_ai_high_risk_ratio": _ratio_at_or_above(ai_values, 0.65),
            "frame_ai_sustained_ratio": _longest_run_ratio(ai_values, 0.65),
            "pixel_forensic_probability_mean": _mean(pixel_values, 0.5),
            "pixel_forensic_probability_p95": _percentile(pixel_values, 0.95, 0.5),
            "frame_truth_score_mean": _mean(truth_values, 50.0),
            "frame_truth_score_std": _std(truth_values),
            "frame_truth_score_p10": _percentile(truth_values, 0.10, 50.0),
            "luma_delta_mean": _mean(self.luma_deltas),
            "luma_delta_std": _std(self.luma_deltas),
            "luma_delta_p95": _percentile(self.luma_deltas, 0.95),
            "noise_delta_mean": _mean(self.noise_deltas),
            "edge_density_delta_mean": _mean(self.edge_density_deltas),
            "flow_warp_error_mean": _mean(self.flow_warp_errors),
            "flow_warp_error_p95": _percentile(self.flow_warp_errors, 0.95),
            "flow_inconsistency_mean": _mean(self.flow_inconsistencies),
            "duplicate_frame_ratio": self.duplicate_frames / pair_count,
            "scene_cut_ratio": self.scene_cuts / pair_count,
        }

    def summary(self) -> Dict[str, Any]:
        features = self.model_features()
        frame_aggregate = (
            features["frame_ai_probability_mean"] * 0.45
            + features["frame_ai_probability_p90"] * 0.30
            + features["frame_ai_high_risk_ratio"] * 0.15
            + features["frame_ai_sustained_ratio"] * 0.10
        )
        temporal_probability, reasons = _temporal_heuristic_probability(features, self.frames_seen)
        return {
            "frames_seen": self.frames_seen,
            "native_pixels_examined": self.native_pixels_examined,
            "tile_count": self.tile_count,
            "tiled_pixels_examined": self.tiled_pixels_examined,
            "frame_ai_probability": round(max(0.0, min(1.0, frame_aggregate)), 4),
            "temporal_synthetic_probability": round(temporal_probability, 4),
            "temporal_reasons": reasons,
            "features": {name: round(float(features[name]), 6) for name in VIDEO_FEATURE_NAMES},
        }

    def detector_results(self, temporal_model_path: str | None) -> List[Dict[str, Any]]:
        summary = self.summary()
        probability = float(summary["temporal_synthetic_probability"])
        results = [
            {
                "name": "temporal_pixel_consistency_heuristic",
                "status": "completed",
                "label": _probability_label(probability),
                "score": round(probability, 4),
                "synthetic_probability": round(probability, 4),
                "manipulation_probability": None,
                "task": "temporal",
                "model_version": "truthshield-temporal-heuristic-v4",
                "calibration_id": None,
                "details": {
                    "reasons": summary["temporal_reasons"],
                    "frames_compared": max(0, self.frames_seen - 1),
                    "model_type": "deterministic_temporal_forensics",
                },
            }
        ]
        if temporal_model_path:
            results.append(run_trained_video_detector(summary["features"], temporal_model_path))
        return results


def run_trained_video_detector(features: Dict[str, float], model_path: str) -> Dict[str, Any]:
    try:
        bundle = _load_video_model(str(Path(model_path).expanduser().resolve()))
        model = bundle["model"] if isinstance(bundle, dict) else bundle
        feature_names = list(bundle.get("feature_names", VIDEO_FEATURE_NAMES)) if isinstance(bundle, dict) else list(VIDEO_FEATURE_NAMES)
        row = [[float(features.get(name, 0.0)) for name in feature_names]]
        probabilities = model.predict_proba(row)[0]
        classes = [str(value).lower() for value in model.classes_]
        positive_index = next(
            (index for index, label in enumerate(classes) if label in {"1", "ai_generated", "synthetic", "fake"}),
            len(probabilities) - 1,
        )
        probability = float(probabilities[positive_index])
        threshold = float(bundle.get("threshold", 0.5)) if isinstance(bundle, dict) else 0.5
        metrics = bundle.get("metrics", {}) if isinstance(bundle, dict) else {}
        return {
            "name": "trained_truthshield_video_detector",
            "status": "completed",
            "label": "likely_ai_generated_video" if probability >= threshold else "lower_ai_video_signal",
            "score": round(probability, 4),
            "synthetic_probability": round(probability, 4),
            "manipulation_probability": None,
            "task": "generation",
            "model_version": str(bundle.get("model_version", Path(model_path).name)) if isinstance(bundle, dict) else Path(model_path).name,
            "calibration_id": str(bundle.get("calibration_id", "legacy-video-calibration")) if isinstance(bundle, dict) else "legacy-video-calibration",
            "details": {
                "model_path": model_path,
                "decision_threshold": threshold,
                "feature_count": len(feature_names),
                "held_out_metrics": metrics.get("test", metrics),
                "note": "Only load a video model file that you trained or trust; serialized model files can execute code when loaded.",
            },
        }
    except ImportError:
        reason = "Install scikit-learn and joblib to run the trained temporal video model."
    except Exception as exc:
        reason = str(exc)[:300]
    return {
        "name": "trained_truthshield_video_detector",
        "status": "unavailable",
        "label": None,
        "score": None,
        "synthetic_probability": None,
        "manipulation_probability": None,
        "task": "generation",
        "model_version": Path(model_path).name,
        "details": {"reason": reason},
    }


def trained_video_sampling_policy(model_path: str) -> str:
    try:
        bundle = _load_video_model(str(Path(model_path).expanduser().resolve()))
    except Exception:
        return "legacy_uniform16"
    if isinstance(bundle, dict):
        return str(bundle.get("sampling_policy") or "legacy_uniform16")
    return "legacy_uniform16"


@functools.lru_cache(maxsize=4)
def _load_video_model(model_path: str) -> Any:
    try:
        import joblib  # type: ignore
    except Exception as exc:
        raise ImportError("joblib is not installed") from exc
    return joblib.load(model_path)


def _temporal_heuristic_probability(features: Dict[str, float], frame_count: int) -> tuple[float, List[str]]:
    if frame_count < 3:
        return 0.35, ["Fewer than three decoded frames make temporal checks weak."]
    probability = 0.12
    reasons: List[str] = []
    if features["flow_warp_error_mean"] > 0.10:
        probability += 0.10
        reasons.append("Motion-compensated frame differences were elevated.")
    if features["flow_warp_error_p95"] > 0.18:
        probability += 0.10
        reasons.append("Some neighboring frames had unusually large motion-warp residuals.")
    if features["flow_inconsistency_mean"] > 1.35:
        probability += 0.08
        reasons.append("Optical-flow direction changed unevenly across neighboring regions.")
    if features["noise_delta_mean"] > 0.38:
        probability += 0.09
        reasons.append("Fine-grain residual noise changed sharply between frames.")
    if features["edge_density_delta_mean"] > 0.055:
        probability += 0.06
        reasons.append("Edge detail flickered more than expected between frames.")
    if features["duplicate_frame_ratio"] > 0.18:
        probability += 0.07
        reasons.append("An unusually large share of neighboring frames were duplicates.")
    if features["frame_ai_sustained_ratio"] >= 0.35:
        probability += 0.12
        reasons.append("Frame-level AI signals stayed high over a sustained run.")
    if not reasons:
        reasons.append("Temporal checks did not find a strong repeated synthetic-video pattern.")
    return max(0.03, min(0.90, probability)), reasons


def _optical_flow_consistency(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    backward_flow = cv2.calcOpticalFlowFarneback(
        current,
        previous,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    height, width = current.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    warped_previous = cv2.remap(
        previous,
        grid_x + backward_flow[..., 0],
        grid_y + backward_flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    warp_error = float(np.mean(cv2.absdiff(current, warped_previous))) / 255.0
    flow_dx = np.gradient(backward_flow[..., 0])
    flow_dy = np.gradient(backward_flow[..., 1])
    roughness = float(np.mean([np.mean(np.abs(value)) for value in (*flow_dx, *flow_dy)]))
    magnitude = float(np.mean(np.linalg.norm(backward_flow, axis=2)))
    inconsistency = roughness / max(0.05, magnitude)
    return warp_error, inconsistency


def _resize_for_temporal_checks(gray: np.ndarray, max_width: int) -> np.ndarray:
    height, width = gray.shape
    if width <= max_width:
        return gray
    scale = max_width / float(width)
    return cv2.resize(gray, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def _mean(values: List[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if values else default


def _std(values: List[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _percentile(values: List[float], percentile: float, default: float = 0.0) -> float:
    return float(np.quantile(values, percentile)) if values else default


def _ratio_at_or_above(values: List[float], threshold: float) -> float:
    return sum(value >= threshold for value in values) / max(1, len(values))


def _longest_run_ratio(values: List[float], threshold: float) -> float:
    longest = 0
    current = 0
    for value in values:
        if value >= threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest / max(1, len(values))


def _probability_label(probability: float) -> str:
    if probability >= 0.65:
        return "temporal_patterns_likely_synthetic"
    if probability <= 0.30:
        return "lower_temporal_synthetic_signal"
    return "temporal_patterns_uncertain"
