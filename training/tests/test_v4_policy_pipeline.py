from __future__ import annotations

import unittest

from training.calibrate_media_policy import calibrate
from training.evaluate_media_policy import evaluate
from training.media_manifest import validate_records


class V4PolicyPipelineTests(unittest.TestCase):
    def test_manifest_rejects_source_and_generator_leakage(self) -> None:
        common = {
            "path": "a.jpg",
            "media_type": "image",
            "class_label": "ai_generated",
            "source": "test",
            "license": "MIT",
            "generator_or_editor": "generator-a",
            "parent_media": None,
            "transformation": "none",
            "semantic_category": "photo",
        }
        report = validate_records(
            [
                {**common, "sha256": "a" * 64, "source_group": "shared", "split": "train"},
                {**common, "path": "b.jpg", "sha256": "b" * 64, "source_group": "shared", "split": "locked_test"},
            ]
        )
        self.assertFalse(report["valid"])
        self.assertTrue(any("source-group leakage" in error for error in report["errors"]))
        self.assertTrue(any("generator-family leakage" in error for error in report["errors"]))

    def test_calibrator_meets_limits_or_disables_outcome(self) -> None:
        rows = []
        for index in range(100):
            rows.append(
                {
                    "label": "authentic",
                    "generation_score": 0.02 + index * 0.0001,
                    "manipulation_score": 0.01 + index * 0.0001,
                }
            )
        for index in range(30):
            rows.append(
                {
                    "label": "generated",
                    "generation_score": 0.97 + index * 0.0001,
                    "manipulation_score": 0.03,
                }
            )
            rows.append(
                {
                    "label": "manipulated",
                    "generation_score": 0.05,
                    "manipulation_score": 0.97 + index * 0.0001,
                }
            )
        policy = calibrate(
            rows,
            media_type="image",
            minimum_precision=0.95,
            authentic_false_warning_limit=0.01,
            false_authentic_limit=0.02,
        )
        self.assertTrue(policy["generation"]["enabled"])
        self.assertTrue(policy["manipulation"]["enabled"])
        report = evaluate(rows, policy, "image", bootstrap_samples=20)
        self.assertTrue(report["promotion_gates"]["authentic_false_ai_rate_within_limit"])
        self.assertTrue(report["promotion_gates"]["synthetic_false_authentic_rate_within_limit"])


if __name__ == "__main__":
    unittest.main()
