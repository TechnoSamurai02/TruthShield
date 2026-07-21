from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image

from analyzers.ai_detectors import (
    combined_synthetic_probability,
    reuse_full_frame_predictions_as_single_tile,
    run_tiled_image_detectors,
)
from analyzers.config import get_settings
from analyzers.enhanced import enhance_video_result
from analyzers.image_analyzer import analyze_frame_array
from analyzers.scoring import (
    DISCLAIMER,
    IMAGE_VIDEO_RECOMMENDATIONS,
    clamp_score,
    get_risk_level,
    summarize_result,
    unique_messages,
)
from analyzers.video_forensics import (
    VideoForensicsAccumulator,
    run_trained_video_detector,
    trained_video_sampling_policy,
)


SAMPLED_FRAME_COUNT = 8
TEMPORAL_MODEL_FRAME_COUNT = 16
MAX_RETURNED_SUSPICIOUS_FRAMES = 12
MAX_FRAME_SCORE_PREVIEW = 240


def analyze_video_path(path: str, filename: str = "upload") -> Dict[str, Any]:
    settings = get_settings()
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError("The video could not be opened safely.")

    try:
        reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_seconds = reported_frame_count / fps if reported_frame_count > 0 and fps > 0 else 0.0
        adaptive_scan: Dict[str, Any] = {}
        if settings.video_analysis_mode == "adaptive":
            adaptive_limit = min(
                settings.video_keyframe_max,
                settings.video_max_frames if settings.video_max_frames > 0 else settings.video_keyframe_max,
            )
            adaptive_scan = _scan_video_for_candidates(
                path,
                reported_frame_count,
                fps,
                max_frames=adaptive_limit,
                max_windows=settings.video_window_max,
            )
            sampled_positions = set(adaptive_scan.get("selected_positions") or [])
        elif settings.video_analysis_mode == "sampled":
            sampled_positions = set(_sampled_frame_positions(reported_frame_count))
        else:
            sampled_positions = None
        temporal_model_positions = set(
            _uniform_frame_positions(reported_frame_count, TEMPORAL_MODEL_FRAME_COUNT)
        )
        temporal_model_names = {str(model_id) for model_id in settings.ai_video_frame_detector_models}

        accumulator = VideoForensicsAccumulator()
        temporal_model_accumulator = VideoForensicsAccumulator()
        compact_frame_results: List[Dict[str, Any]] = []
        suspicious_candidates: List[Dict[str, Any]] = []
        frame_score_series: List[tuple[int, float]] = []
        warning_counts: Counter[str] = Counter()
        positive_counts: Counter[str] = Counter()
        decoded_frames = 0
        analyzed_frames = 0

        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            source_frame_index = decoded_frames
            decoded_frames += 1

            if not _should_analyze_frame(
                source_frame_index=source_frame_index,
                sampled_positions=sampled_positions,
                stride=settings.video_frame_stride,
            ):
                continue
            if settings.video_max_frames > 0 and analyzed_frames >= settings.video_max_frames:
                break

            # Keep the detector filename neutral so training labels and upload names cannot leak into predictions.
            frame_result = analyze_frame_array(
                frame,
                frame_label=f"video-frame-{source_frame_index + 1:08d}",
                detector_model_ids=settings.ai_video_frame_detector_models,
            )
            if settings.video_tile_analysis and settings.ai_video_frame_detector_models:
                if frame.shape[1] <= settings.video_tile_size and frame.shape[0] <= settings.video_tile_size:
                    tile_detectors = reuse_full_frame_predictions_as_single_tile(
                        frame_result.get("detectors", []),
                        settings.ai_video_frame_detector_models,
                    )
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tile_detectors = run_tiled_image_detectors(
                        Image.fromarray(frame_rgb),
                        settings.ai_video_frame_detector_models,
                        tile_size=settings.video_tile_size,
                        overlap=settings.video_tile_overlap,
                    )
                frame_result.setdefault("detectors", []).extend(tile_detectors)

            timestamp_seconds = source_frame_index / fps if fps > 0 else None
            temporal_frame = accumulator.update(frame, frame_result)
            if source_frame_index in temporal_model_positions:
                # The learned temporal model was validated on 16 uniformly spaced full-frame
                # neural signals. Keep exhaustive/tiled evidence in the primary accumulator, but
                # feed the learned classifier the same distribution it saw during training.
                model_detectors = [
                    detector
                    for detector in frame_result.get("detectors", [])
                    if str(detector.get("name") or "") in temporal_model_names
                ]
                temporal_model_accumulator.update(
                    frame,
                    {
                        "truth_score": 50.0,
                        "detectors": model_detectors,
                        "technical_details": {
                            "forensic_analysis": {"synthetic_artifact_probability": 0.5}
                        },
                    },
                )
            synthetic_probability = temporal_frame.get("synthetic_probability")
            manipulation_probability = max(
                [
                    float(detector["manipulation_probability"])
                    for detector in frame_result.get("detectors", [])
                    if isinstance(detector.get("manipulation_probability"), (int, float))
                ],
                default=None,
            )
            truth_score = float(frame_result["truth_score"])
            frame_score_series.append((source_frame_index + 1, truth_score))
            warning_counts.update(frame_result.get("warnings", []))
            positive_counts.update(frame_result.get("positive_signals", []))

            compact_frame_results.append(
                {
                    "truth_score": truth_score,
                    "warnings": frame_result.get("warnings", [])[:4],
                    "positive_signals": frame_result.get("positive_signals", [])[:4],
                    "detectors": [
                        {
                            "name": detector.get("name"),
                            "status": detector.get("status"),
                            "label": detector.get("label"),
                            "synthetic_probability": detector.get("synthetic_probability"),
                            "manipulation_probability": detector.get("manipulation_probability"),
                            "task": detector.get("task"),
                            "model_version": detector.get("model_version"),
                            "calibration_id": detector.get("calibration_id"),
                            "suspicious_regions": detector.get("suspicious_regions", []),
                        }
                        for detector in frame_result.get("detectors", [])
                    ],
                    "source_frame_index": source_frame_index,
                }
            )
            if truth_score < 50 or (
                isinstance(synthetic_probability, (int, float)) and synthetic_probability >= 0.65
            ) or (
                isinstance(manipulation_probability, (int, float)) and manipulation_probability >= 0.65
            ):
                suspicious_candidates.append(
                    {
                        "frame_index": source_frame_index + 1,
                        "source_frame_number": source_frame_index + 1,
                        "timestamp_seconds": round(timestamp_seconds, 4) if timestamp_seconds is not None else None,
                        "truth_score": int(round(truth_score)),
                        "synthetic_probability": (
                            round(float(synthetic_probability), 4)
                            if isinstance(synthetic_probability, (int, float))
                            else None
                        ),
                        "tile_synthetic_probability": temporal_frame.get("tile_synthetic_probability"),
                        "manipulation_probability": (
                            round(float(manipulation_probability), 4)
                            if isinstance(manipulation_probability, (int, float))
                            else None
                        ),
                        "warnings": frame_result.get("warnings", [])[:3],
                        "kind": (
                            "manipulation"
                            if isinstance(manipulation_probability, (int, float))
                            and float(manipulation_probability) > float(synthetic_probability or 0.0)
                            else "generation"
                        ),
                    }
                )
            analyzed_frames += 1

        if not compact_frame_results:
            raise ValueError("No readable frames were found in this video.")

        temporal_summary = accumulator.summary()
        temporal_model_summary = temporal_model_accumulator.summary()
        production_policy_features = dict(temporal_summary["features"])
        if adaptive_scan:
            production_policy_features.update(
                {
                    "second_order_motion_mean": float(adaptive_scan.get("second_order_motion_mean") or 0.0),
                    "second_order_motion_p95": float(adaptive_scan.get("second_order_motion_p95") or 0.0),
                }
            )
        video_detectors = accumulator.detector_results(None)
        if adaptive_scan:
            video_detectors.append(_second_order_temporal_detector(adaptive_scan))
        if settings.ai_video_temporal_model_path:
            trained_sampling_policy = trained_video_sampling_policy(settings.ai_video_temporal_model_path)
            video_detectors.append(
                run_trained_video_detector(
                    (
                        production_policy_features
                        if trained_sampling_policy == "adaptive_v4"
                        else temporal_model_summary["features"]
                    ),
                    settings.ai_video_temporal_model_path,
                )
            )
        else:
            trained_sampling_policy = "none"
        features = temporal_model_summary["features"]
        temporal_windows = _window_scores(
            adaptive_scan.get("temporal_windows", []) if adaptive_scan else [],
            compact_frame_results,
            fps,
        )
        frame_scores = [score for _, score in frame_score_series]
        average_frame_score = float(np.mean(frame_scores))
        suspicious_candidates.sort(
            key=lambda item: (
                float(item.get("synthetic_probability") or 0.0),
                100.0 - float(item["truth_score"]),
            ),
            reverse=True,
        )

        expected_frames = max(reported_frame_count, decoded_frames)
        coverage_percent = analyzed_frames / max(1, expected_frames) * 100.0
        exhaustive = bool(
            settings.video_analysis_mode == "exhaustive"
            and settings.video_frame_stride == 1
            and settings.video_max_frames == 0
            and analyzed_frames == decoded_frames
            and (reported_frame_count <= 0 or decoded_frames >= reported_frame_count)
        )
        warnings: List[str] = []
        positives: List[str] = []
        score = average_frame_score

        suspicious_ratio = len(suspicious_candidates) / max(1, analyzed_frames)
        if suspicious_ratio >= 0.20:
            warnings.append("A sustained share of analyzed frames showed stronger synthetic or forensic risk signals.")
            score -= min(14.0, 5.0 + suspicious_ratio * 15.0)
        elif suspicious_candidates:
            warnings.append("A small number of frames showed stronger risk signals and should be reviewed.")
            score -= min(5.0, suspicious_ratio * 20.0)
        else:
            positives.append("Analyzed frames did not show high-risk visual signals.")

        if exhaustive:
            positives.append("Every decoded frame was analyzed in exhaustive mode.")
        else:
            warnings.append(
                f"Video analysis covered {coverage_percent:.1f}% of reported or decoded frames; use exhaustive mode for full coverage."
            )
        if accumulator.tiled_pixels_examined >= accumulator.native_pixels_examined > 0:
            positives.append("Overlapping model tiles collectively covered every pixel of every analyzed frame.")
        elif settings.video_tile_analysis:
            warnings.append("Tiled model analysis was requested but did not complete across every analyzed frame.")

        if reported_frame_count <= 0 or fps <= 0:
            score -= 8
            warnings.append("Basic video metadata was missing or unreadable.")
        else:
            positives.append("Basic video metadata could be read.")
        if duration_seconds and duration_seconds < 1.5:
            score -= 5
            warnings.append("The video is very short, which limits confidence in temporal analysis.")
        if width < 360 or height < 240:
            score -= 8
            warnings.append("Video resolution is low, which limits fine visual checks.")
        elif width > 0 and height > 0:
            positives.append("Video resolution is sufficient for detailed frame analysis.")

        warnings.extend(_top_repeated_messages(warning_counts, limit=4))
        positives.extend(_top_repeated_messages(positive_counts, limit=4))
        final_score = clamp_score(score)
        risk_level, verdict = get_risk_level(final_score)
        metadata_score = 82.0 if reported_frame_count > 0 and fps > 0 and width > 0 and height > 0 else 35.0
        temporal_probability = float(temporal_summary["temporal_synthetic_probability"])

        result = {
            "content_type": "video",
            "truth_score": final_score,
            "risk_level": risk_level,
            "verdict": verdict,
            "summary": summarize_result("video", final_score, warnings, positives),
            "warnings": unique_messages(warnings),
            "positive_signals": unique_messages(positives),
            "recommendations": IMAGE_VIDEO_RECOMMENDATIONS,
            "evidence": {
                "frame_analysis_score": round(average_frame_score, 2),
                "metadata_score": metadata_score,
                "visual_consistency_score": round(float(np.mean(frame_scores)), 2),
                "temporal_consistency_score": round((1.0 - temporal_probability) * 100.0, 2),
                "frame_ai_generation_score": round(float(temporal_summary["frame_ai_probability"]) * 100.0, 2),
                "overall_risk_score": float(100 - final_score),
            },
            "frames_analyzed": analyzed_frames,
            "suspicious_frames": suspicious_candidates[:MAX_RETURNED_SUSPICIOUS_FRAMES],
            "detectors": video_detectors,
            "technical_details": {
                "filename": filename,
                "frame_count": reported_frame_count,
                "decoded_frame_count": decoded_frames,
                "fps": round(fps, 3),
                "duration_seconds": round(duration_seconds, 3),
                "width": width,
                "height": height,
                "analysis_coverage": {
                    "mode": settings.video_analysis_mode,
                    "exhaustive": exhaustive,
                    "frame_stride": settings.video_frame_stride,
                    "configured_max_frames": settings.video_max_frames,
                    "frames_analyzed": analyzed_frames,
                    "coverage_percent": round(coverage_percent, 3),
                    "native_pixels_examined": accumulator.native_pixels_examined,
                    "tile_analysis_enabled": settings.video_tile_analysis,
                    "tile_count": accumulator.tile_count,
                    "tiled_source_pixels_covered": min(
                        accumulator.native_pixels_examined,
                        accumulator.tiled_pixels_examined,
                    ),
                    "model_input_note": "Native-resolution forensic checks process full frames. Neural classifiers resize full frames and overlapping tiles to their learned input size.",
                    "selection_policy": (
                        "uniform coverage plus scene boundaries, anomaly peaks, and neighboring frames"
                        if settings.video_analysis_mode == "adaptive"
                        else settings.video_analysis_mode
                    ),
                    "complete_video_cheap_scan": bool(adaptive_scan),
                },
                "frame_score_summary": {
                    "mean": round(float(np.mean(frame_scores)), 3),
                    "standard_deviation": round(float(np.std(frame_scores)), 3),
                    "minimum": round(float(np.min(frame_scores)), 3),
                    "p10": round(float(np.quantile(frame_scores, 0.10)), 3),
                    "maximum": round(float(np.max(frame_scores)), 3),
                },
                "frame_scores_preview": _downsample_score_series(frame_score_series, MAX_FRAME_SCORE_PREVIEW),
                "temporal_forensics": temporal_summary,
                "adaptive_scan": adaptive_scan,
                "temporal_windows": temporal_windows,
                "video_model_features": features,
                "production_policy_features": production_policy_features,
                "trained_video_model_frame_count": temporal_model_summary["frames_seen"],
                "trained_video_model_sampling": (
                    "Adaptive v4 selected-frame distribution."
                    if trained_sampling_policy == "adaptive_v4"
                    else "Legacy model: up to 16 uniformly spaced full-frame signals."
                ),
                "heuristic_note": "Every-frame and pixel-level checks provide evidence, not proof. Natural motion, edits, animation, compression, and screen recordings can resemble AI artifacts.",
            },
            "disclaimer": DISCLAIMER,
        }
        return enhance_video_result(
            result,
            compact_frame_results,
            filename,
            video_detectors=video_detectors,
        )
    finally:
        capture.release()


def _should_analyze_frame(
    source_frame_index: int,
    sampled_positions: set[int] | None,
    stride: int,
) -> bool:
    if sampled_positions is not None:
        return source_frame_index in sampled_positions
    return source_frame_index % max(1, stride) == 0


def _sampled_frame_positions(frame_count: int) -> List[int]:
    if frame_count <= 0:
        return list(range(SAMPLED_FRAME_COUNT))
    count = min(SAMPLED_FRAME_COUNT, frame_count)
    return sorted({int(position) for position in np.linspace(0, frame_count - 1, count)})


def _uniform_frame_positions(frame_count: int, limit: int) -> List[int]:
    if frame_count <= 0:
        return list(range(max(1, limit)))
    count = min(max(1, limit), frame_count)
    return sorted({int(round(position)) for position in np.linspace(0, frame_count - 1, count)})


def _scan_video_for_candidates(
    path: str,
    reported_frame_count: int,
    fps: float,
    *,
    max_frames: int,
    max_windows: int,
) -> Dict[str, Any]:
    """Decode the complete video with cheap features before neural inference."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return {
            "status": "unavailable",
            "selected_positions": _uniform_frame_positions(reported_frame_count, max_frames),
            "temporal_windows": [],
        }
    anomaly_scores: List[tuple[int, float]] = []
    scene_boundaries: List[int] = []
    motion_values: List[float] = []
    previous_gray: np.ndarray | None = None
    previous_histogram: np.ndarray | None = None
    decoded = 0
    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.shape[1] > 192:
                scale = 192.0 / gray.shape[1]
                gray = cv2.resize(
                    gray,
                    (192, max(1, round(gray.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            histogram = cv2.calcHist([gray], [0], None, [32], [0, 256])
            cv2.normalize(histogram, histogram)
            if previous_gray is not None and previous_histogram is not None:
                motion = float(np.mean(cv2.absdiff(gray, previous_gray))) / 255.0
                correlation = float(cv2.compareHist(previous_histogram, histogram, cv2.HISTCMP_CORREL))
                edge = float(np.count_nonzero(cv2.Canny(gray, 70, 170))) / max(1.0, float(gray.size))
                anomaly = motion * 0.72 + max(0.0, 1.0 - correlation) * 0.23 + edge * 0.05
                anomaly_scores.append((decoded, anomaly))
                motion_values.append(motion)
                if motion >= 0.18 and correlation < 0.65:
                    scene_boundaries.append(decoded)
            previous_gray = gray
            previous_histogram = histogram
            decoded += 1
    finally:
        capture.release()

    actual_count = decoded or reported_frame_count
    selected = _select_adaptive_frame_positions(
        actual_count,
        anomaly_scores,
        scene_boundaries,
        fps=fps,
        limit=max_frames,
    )
    windows = _select_temporal_windows(
        actual_count,
        anomaly_scores,
        fps=fps,
        limit=max_windows,
    )
    second_order = [abs(motion_values[index] - motion_values[index - 1]) for index in range(1, len(motion_values))]
    return {
        "status": "completed",
        "decoded_frames": decoded,
        "selected_positions": selected,
        "scene_boundary_count": len(scene_boundaries),
        "anomaly_peak_count": min(len(anomaly_scores), max(1, max_frames // 3)),
        "temporal_windows": windows,
        "second_order_motion_mean": round(float(np.mean(second_order)), 6) if second_order else 0.0,
        "second_order_motion_p95": round(float(np.quantile(second_order, 0.95)), 6) if second_order else 0.0,
        "selection_note": "Complete-video cheap scan; neural inference is limited to diverse and suspicious frames.",
    }


def _select_adaptive_frame_positions(
    frame_count: int,
    anomaly_scores: List[tuple[int, float]],
    scene_boundaries: List[int],
    *,
    fps: float,
    limit: int,
) -> List[int]:
    if frame_count <= 0:
        return []
    limit = min(max(1, limit), frame_count)
    selected = set(_uniform_frame_positions(frame_count, min(limit, max(16, limit // 2))))
    selected.update(index for index in scene_boundaries if 0 <= index < frame_count)
    peak_limit = max(1, limit // 3)
    peaks = sorted(anomaly_scores, key=lambda item: item[1], reverse=True)[:peak_limit]
    neighbor = max(1, int(round(fps * 0.20))) if fps > 0 else 1
    for index, _ in peaks:
        selected.update({index, index - 1, index + 1, index - neighbor, index + neighbor})
    selected = {index for index in selected if 0 <= index < frame_count}
    if len(selected) <= limit:
        return sorted(selected)
    priorities = {index: score for index, score in anomaly_scores}
    required = {0, frame_count - 1}
    required.update(_uniform_frame_positions(frame_count, min(16, limit)))
    ordered_optional = sorted(
        selected - required,
        key=lambda index: (priorities.get(index, 0.0), -index),
        reverse=True,
    )
    return sorted([*required, *ordered_optional[: max(0, limit - len(required))]])[:limit]


def _select_temporal_windows(
    frame_count: int,
    anomaly_scores: List[tuple[int, float]],
    *,
    fps: float,
    limit: int,
) -> List[Dict[str, int | str]]:
    if frame_count <= 0:
        return []
    half_width = max(2, int(round(fps * 0.75))) if fps > 0 else max(2, frame_count // 40)
    peak_centers = [index for index, _ in sorted(anomaly_scores, key=lambda item: item[1], reverse=True)]
    uniform_centers = _uniform_frame_positions(frame_count, min(4, limit))
    windows: List[Dict[str, int | str]] = []
    for center in [*peak_centers, *uniform_centers]:
        start = max(0, center - half_width)
        end = min(frame_count - 1, center + half_width)
        if any(not (end < int(item["start_frame"]) or start > int(item["end_frame"])) for item in windows):
            continue
        windows.append(
            {
                "start_frame": start,
                "end_frame": end,
                "selection_reason": "anomaly_peak" if center in peak_centers else "uniform_coverage",
            }
        )
        if len(windows) >= limit:
            break
    return sorted(windows, key=lambda item: int(item["start_frame"]))


def _window_scores(
    windows: List[Dict[str, Any]],
    frame_results: List[Dict[str, Any]],
    fps: float,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for window in windows:
        start = int(window.get("start_frame", 0))
        end = int(window.get("end_frame", start))
        values = [
            combined_synthetic_probability(frame.get("detectors", []))
            for frame in frame_results
            if start <= int(frame.get("source_frame_index", -1)) <= end
        ]
        probabilities = [float(value) for value in values if value is not None]
        manipulation_values = [
            float(detector["manipulation_probability"])
            for frame in frame_results
            if start <= int(frame.get("source_frame_index", -1)) <= end
            for detector in (frame.get("detectors", []) if isinstance(frame.get("detectors"), list) else [])
            if detector.get("status") == "completed"
            and isinstance(detector.get("manipulation_probability"), (int, float))
        ]
        scored.append(
            {
                **window,
                "start_seconds": round(start / fps, 4) if fps > 0 else None,
                "end_seconds": round(end / fps, 4) if fps > 0 else None,
                "frames_analyzed": len(probabilities),
                "generation_score": round(sum(probabilities) / len(probabilities), 4) if probabilities else None,
                "manipulation_score": (
                    round(sum(manipulation_values) / len(manipulation_values), 4)
                    if manipulation_values
                    else None
                ),
            }
        )
    return scored


def _second_order_temporal_detector(scan: Dict[str, Any]) -> Dict[str, Any]:
    p95 = float(scan.get("second_order_motion_p95") or 0.0)
    return {
        "name": "second_order_temporal_screen",
        "status": "completed" if scan.get("status") == "completed" else "unavailable",
        "label": "elevated_second_order_change" if p95 >= 0.20 else "ordinary_second_order_change",
        "score": round(min(1.0, p95 * 3.0), 4),
        "synthetic_probability": None,
        "manipulation_probability": None,
        "task": "temporal",
        "model_version": "truthshield-second-order-screen-v4",
        "calibration_id": None,
        "details": {
            "second_order_motion_mean": scan.get("second_order_motion_mean"),
            "second_order_motion_p95": scan.get("second_order_motion_p95"),
            "decisive_capable": False,
            "note": "Cheap second-order motion evidence. The official optional D3 checkpoint is not claimed unless separately configured and licensed.",
        },
    }


def _top_repeated_messages(counts: Counter[str], limit: int) -> List[str]:
    return [message for message, _ in counts.most_common(limit)]


def _downsample_score_series(series: List[tuple[int, float]], limit: int) -> List[Dict[str, float | int]]:
    if len(series) <= limit:
        selected = series
    else:
        indices = sorted({int(index) for index in np.linspace(0, len(series) - 1, limit)})
        selected = [series[index] for index in indices]
    return [{"frame_index": index, "truth_score": round(score, 2)} for index, score in selected]
