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
from analyzers.video_forensics import VideoForensicsAccumulator, run_trained_video_detector


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
        sampled_positions = (
            set(_sampled_frame_positions(reported_frame_count))
            if settings.video_analysis_mode == "sampled"
            else None
        )
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
                        }
                        for detector in frame_result.get("detectors", [])
                    ],
                }
            )
            if truth_score < 50 or (
                isinstance(synthetic_probability, (int, float)) and synthetic_probability >= 0.65
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
                        "warnings": frame_result.get("warnings", [])[:3],
                    }
                )
            analyzed_frames += 1

        if not compact_frame_results:
            raise ValueError("No readable frames were found in this video.")

        temporal_summary = accumulator.summary()
        temporal_model_summary = temporal_model_accumulator.summary()
        video_detectors = accumulator.detector_results(None)
        if settings.ai_video_temporal_model_path:
            video_detectors.append(
                run_trained_video_detector(
                    temporal_model_summary["features"],
                    settings.ai_video_temporal_model_path,
                )
            )
        features = temporal_model_summary["features"]
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
                "video_model_features": features,
                "trained_video_model_frame_count": temporal_model_summary["frames_seen"],
                "trained_video_model_sampling": (
                    "Up to 16 uniformly spaced full-frame signals; exhaustive native and tiled "
                    "analysis remains in temporal_forensics and analysis_coverage."
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


def _top_repeated_messages(counts: Counter[str], limit: int) -> List[str]:
    return [message for message, _ in counts.most_common(limit)]


def _downsample_score_series(series: List[tuple[int, float]], limit: int) -> List[Dict[str, float | int]]:
    if len(series) <= limit:
        selected = series
    else:
        indices = sorted({int(index) for index in np.linspace(0, len(series) - 1, limit)})
        selected = [series[index] for index in indices]
    return [{"frame_index": index, "truth_score": round(score, 2)} for index, score in selected]
