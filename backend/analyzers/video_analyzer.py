from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

from analyzers.image_analyzer import analyze_frame_array
from analyzers.scoring import (
    DISCLAIMER,
    IMAGE_VIDEO_RECOMMENDATIONS,
    clamp_score,
    get_risk_level,
    summarize_result,
    unique_messages,
)


MAX_FRAMES_TO_ANALYZE = 8


def analyze_video_path(path: str, filename: str = "upload") -> Dict[str, Any]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError("The video could not be opened safely.")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_seconds = frame_count / fps if frame_count > 0 and fps > 0 else 0.0

        frame_positions = _frame_positions(frame_count)
        frame_results: List[Dict[str, Any]] = []

        for index, frame_position in enumerate(frame_positions):
            if frame_position is not None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_position))
            success, frame = capture.read()
            if not success or frame is None:
                continue
            result = analyze_frame_array(frame, frame_label=f"{filename}:frame-{index + 1}")
            result["frame_index"] = index + 1
            result["timestamp_seconds"] = (float(frame_position) / fps) if frame_position is not None and fps > 0 else None
            frame_results.append(result)

        if not frame_results:
            raise ValueError("No readable frames were found in this video.")

        warnings: List[str] = []
        positives: List[str] = []
        frame_scores = [result["truth_score"] for result in frame_results]
        average_frame_score = float(np.mean(frame_scores))
        score = average_frame_score

        suspicious_frames = [result for result in frame_results if result["truth_score"] < 50]
        if suspicious_frames:
            warnings.append("One or more sampled frames showed stronger risk signals.")
            score -= min(10, len(suspicious_frames) * 3)
        else:
            positives.append("Sampled frames did not show high-risk visual signals.")

        if frame_count <= 0 or fps <= 0:
            score -= 8
            warnings.append("Basic video metadata was missing or unreadable.")
        else:
            positives.append("Basic video metadata could be read.")

        if duration_seconds and duration_seconds < 1.5:
            score -= 5
            warnings.append("The video is very short, which limits confidence in the analysis.")

        if width < 360 or height < 240:
            score -= 8
            warnings.append("Video resolution is low, which limits visual consistency checks.")
        elif width > 0 and height > 0:
            positives.append("Video resolution is sufficient for basic frame analysis.")

        frame_warnings = [warning for result in frame_results for warning in result["warnings"]]
        frame_positives = [positive for result in frame_results for positive in result["positive_signals"]]
        warnings.extend(_top_repeated_messages(frame_warnings, limit=4))
        positives.extend(_top_repeated_messages(frame_positives, limit=4))

        final_score = clamp_score(score)
        risk_level, verdict = get_risk_level(final_score)
        avg_visual_score = float(np.mean([result["evidence"]["visual_consistency_score"] for result in frame_results]))
        metadata_score = 82.0 if frame_count > 0 and fps > 0 and width > 0 and height > 0 else 35.0

        return {
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
                "visual_consistency_score": round(avg_visual_score, 2),
                "overall_risk_score": float(100 - final_score),
            },
            "frames_analyzed": len(frame_results),
            "suspicious_frames": [
                {
                    "frame_index": result["frame_index"],
                    "timestamp_seconds": result["timestamp_seconds"],
                    "truth_score": result["truth_score"],
                    "warnings": result["warnings"][:3],
                }
                for result in suspicious_frames[:5]
            ],
            "technical_details": {
                "filename": filename,
                "frame_count": frame_count,
                "fps": round(fps, 3),
                "duration_seconds": round(duration_seconds, 3),
                "width": width,
                "height": height,
                "frame_scores": frame_scores,
                "heuristic_note": "Video scoring samples frames and averages heuristic image signals. It is not proof of manipulation.",
            },
            "disclaimer": DISCLAIMER,
        }
    finally:
        capture.release()


def _frame_positions(frame_count: int) -> List[int | None]:
    if frame_count <= 0:
        return [None] * MAX_FRAMES_TO_ANALYZE
    count = min(MAX_FRAMES_TO_ANALYZE, frame_count)
    return [int(position) for position in np.linspace(0, frame_count - 1, count)]


def _top_repeated_messages(messages: List[str], limit: int) -> List[str]:
    counts: Dict[str, int] = {}
    for message in messages:
        counts[message] = counts.get(message, 0) + 1
    repeated = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [message for message, _ in repeated[:limit]]
