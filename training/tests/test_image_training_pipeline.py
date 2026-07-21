from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from training.check_image_robustness import _robustness_variants
from training.evaluate_image_detector import _likely_ai_threshold_metrics, _three_way_metrics
from training.prepare_defactify_sample import V4_GENERATOR_SPLITS, _group_split, _target_split
from training.train_image_detector import _training_data_files


class ImageTrainingPipelineTests(unittest.TestCase):
    def test_v4_defactify_split_keeps_caption_and_generator_isolation(self) -> None:
        caption = "the same source caption"
        group_split = _group_split(caption)
        real_split = _target_split(
            source_split="train",
            raw_label=0,
            raw_generator=0,
            source_group_key=caption,
            split_policy="generator-heldout-v4",
        )
        retained_generators = {
            generator
            for generator in V4_GENERATOR_SPLITS
            if _target_split(
                source_split="train",
                raw_label=1,
                raw_generator=generator,
                source_group_key=caption,
                split_policy="generator-heldout-v4",
            )
            is not None
        }

        self.assertEqual(real_split, group_split)
        self.assertTrue(retained_generators)
        self.assertTrue(all(V4_GENERATOR_SPLITS[value] == group_split for value in retained_generators))

    def test_v4_training_loader_excludes_calibration_and_locked_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "tuning", "calibration", "locked_test"):
                folder = root / split / "ai_generated"
                folder.mkdir(parents=True)
                Image.new("RGB", (16, 16)).save(folder / f"{split}.png")
            data_files = _training_data_files(root)
        self.assertEqual(set(data_files), {"train", "validation"})
        self.assertIn("tuning.png", data_files["validation"][0])

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
