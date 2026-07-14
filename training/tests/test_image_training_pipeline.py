from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from training.check_image_robustness import _robustness_variants
from training.evaluate_image_detector import _likely_ai_threshold_metrics, _three_way_metrics


class ImageTrainingPipelineTests(unittest.TestCase):
    def test_robustness_variants_cover_common_upload_changes(self) -> None:
        image = Image.new("RGB", (1000, 800), color=(40, 90, 140))

        variants = list(_robustness_variants(image))
        names = {name for name, _ in variants}

        self.assertEqual(len(variants), 8)
        self.assertIn("original", names)
        self.assertIn("jpeg_quality_50", names)
        self.assertIn("resize_50pct", names)
        self.assertIn("center_crop_85pct", names)
        self.assertIn("social_resize_jpeg_70", names)
        self.assertTrue(all(variant.mode == "RGB" for _, variant in variants))

    def test_three_way_report_tracks_false_accusations_and_abstentions(self) -> None:
        truth = np.asarray([True, True, False, False, False])
        scores = np.asarray([0.96, 0.55, 0.92, 0.40, 0.04])

        report = _three_way_metrics(truth, scores, authentic_max=0.15, ai_min=0.90)

        self.assertEqual(report["counts"]["true_ai"], 1)
        self.assertEqual(report["counts"]["false_ai_alarm"], 1)
        self.assertEqual(report["counts"]["inconclusive"], 2)
        self.assertEqual(report["counts"]["true_authentic"], 1)

    def test_legacy_threshold_report_uses_frontend_seventy_percent_cutoff(self) -> None:
        truth = np.asarray([True, False, False])
        scores = np.asarray([0.95, 0.73, 0.20])

        report = _likely_ai_threshold_metrics(truth, scores, threshold=0.70)

        self.assertEqual(report["true_ai"], 1)
        self.assertEqual(report["false_ai_alarm"], 1)
        self.assertEqual(report["false_positive_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
