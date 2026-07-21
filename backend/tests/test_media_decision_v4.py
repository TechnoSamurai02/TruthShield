from __future__ import annotations

import unittest
import os
import tempfile
from unittest.mock import patch

import cv2
import numpy as np

from analyzers.media_decision import assess_media_evidence
from analyzers.video_analyzer import _select_adaptive_frame_positions, _select_temporal_windows, analyze_video_path
from models.schemas import AnalysisResponse, MediaAssessment


def detector(
    *,
    generation: float | None = None,
    manipulation: float | None = None,
    task: str = "generation",
    regions: bool = False,
) -> dict:
    return {
        "name": f"test-{task}",
        "status": "completed",
        "synthetic_probability": generation,
        "manipulation_probability": manipulation,
        "task": task,
        "model_version": f"test-{task}-v1",
        "suspicious_regions": [{"box": [0, 0, 10, 10]}] if regions else [],
    }


class MediaDecisionV4Tests(unittest.TestCase):
    def test_generated_threshold_is_inclusive(self) -> None:
        assessment, _ = assess_media_evidence(
            [detector(generation=0.95)], {}, None, None, media_type="image"
        )
        self.assertEqual(assessment["verdict"], "likely_ai_generated")

    def test_authentic_requires_both_low_scores(self) -> None:
        assessment, _ = assess_media_evidence(
            [detector(generation=0.05, manipulation=0.04)], {}, None, None, media_type="image"
        )
        self.assertEqual(assessment["verdict"], "likely_authentic")
        self.assertEqual(assessment["generation_score"], 0.05)
        self.assertEqual(assessment["manipulation_score"], 0.04)

    def test_missing_manipulation_screen_prevents_authentic(self) -> None:
        assessment, _ = assess_media_evidence(
            [detector(generation=0.03)], {}, None, None, media_type="image"
        )
        self.assertEqual(assessment["verdict"], "inconclusive")

    def test_manipulation_needs_dedicated_localized_support(self) -> None:
        assessment, _ = assess_media_evidence(
            [detector(generation=0.10), detector(manipulation=0.98, task="manipulation", regions=True)],
            {},
            None,
            None,
            media_type="image",
        )
        self.assertEqual(assessment["verdict"], "likely_ai_manipulated")

    def test_high_editing_screen_without_specialist_abstains(self) -> None:
        assessment, _ = assess_media_evidence(
            [detector(generation=0.10, manipulation=0.98)], {}, None, None, media_type="image"
        )
        self.assertEqual(assessment["verdict"], "inconclusive")

    def test_transformation_instability_forces_abstention(self) -> None:
        assessment, debug = assess_media_evidence(
            [detector(generation=0.99)],
            {},
            None,
            None,
            media_type="image",
            transformation_instability=0.40,
        )
        self.assertEqual(assessment["verdict"], "inconclusive")
        self.assertEqual(debug["transformation_or_window_instability"], 0.4)

    def test_unavailable_models_are_neutral(self) -> None:
        assessment, _ = assess_media_evidence(
            [{"name": "missing", "status": "unavailable", "synthetic_probability": 1.0}],
            {},
            None,
            None,
            media_type="image",
        )
        self.assertEqual(assessment["verdict"], "inconclusive")
        self.assertIsNone(assessment["generation_score"])

    def test_schema_accepts_shared_assessment_for_video(self) -> None:
        assessment, _ = assess_media_evidence(
            [detector(generation=0.98)],
            {"analysis_coverage": {"frames_analyzed": 16}},
            None,
            None,
            media_type="video",
        )
        MediaAssessment(**assessment)

    def test_adaptive_selection_is_bounded_and_expands_peaks(self) -> None:
        positions = _select_adaptive_frame_positions(
            300,
            [(150, 1.0), (20, 0.8)],
            [100],
            fps=30.0,
            limit=64,
        )
        self.assertLessEqual(len(positions), 64)
        self.assertIn(0, positions)
        self.assertIn(299, positions)
        self.assertIn(150, positions)
        self.assertTrue(any(position in positions for position in (144, 149, 151, 156)))

    def test_temporal_windows_are_non_overlapping_and_bounded(self) -> None:
        windows = _select_temporal_windows(
            600,
            [(100, 1.0), (300, 0.9), (500, 0.8)],
            fps=30.0,
            limit=3,
        )
        self.assertLessEqual(len(windows), 3)
        for previous, current in zip(windows, windows[1:]):
            self.assertLess(int(previous["end_frame"]), int(current["start_frame"]))

    def test_adaptive_video_path_scans_all_frames_but_bounds_neural_frames(self) -> None:
        width, height, frame_count = 160, 120, 20
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as temporary:
            path = temporary.name
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(frame_count):
                value = 25 if index < 10 else 220
                writer.write(np.full((height, width, 3), value, dtype=np.uint8))
            writer.release()
            with patch.dict(
                os.environ,
                {
                    "VIDEO_ANALYSIS_MODE": "adaptive",
                    "VIDEO_KEYFRAME_MAX": "12",
                    "VIDEO_WINDOW_MAX": "4",
                    "VIDEO_TILE_ANALYSIS": "false",
                    "ENABLE_LOCAL_AI_MODELS": "false",
                    "BRAVE_SEARCH_API_KEY": "",
                    "GOOGLE_VISION_API_KEY": "",
                },
                clear=False,
            ):
                result = analyze_video_path(path, "adaptive-test.avi")
            scan = result["technical_details"]["adaptive_scan"]
            self.assertEqual(scan["decoded_frames"], frame_count)
            self.assertLessEqual(result["frames_analyzed"], 12)
            self.assertEqual(result["assessment"]["verdict"], "inconclusive")
        finally:
            writer.release()
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
