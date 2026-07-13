from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.prepare_aigvdbench_sample import _deduplicate_content_groups, _source_group
from training.prepare_video_frames import _uniform_positions
from training.train_video_detector import (
    _best_threshold,
    _candidate_models,
    _evaluate,
    _positive_probabilities,
)


class VideoTrainingPipelineTests(unittest.TestCase):
    def test_real_clip_coordinates_share_one_source_group(self) -> None:
        first = _source_group("folder/abc_DEF-12_4_0to150.mp4")
        second = _source_group("folder/abc_DEF-12_9_200to350.mp4")
        self.assertEqual(first, second)

    def test_frame_cap_covers_start_middle_and_end(self) -> None:
        positions = sorted(_uniform_positions(frame_count=101, frame_stride=1, max_frames=5) or [])
        self.assertEqual(len(positions), 5)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 100)
        self.assertTrue(any(45 <= position <= 55 for position in positions))

    def test_content_deduplication_protects_test_before_train(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_file = root / "train.mp4"
            test_file = root / "test.mp4"
            train_file.write_bytes(b"train")
            test_file.write_bytes(b"test")
            rows = [
                {
                    "split": "train",
                    "label": "ai_generated",
                    "source_group": "shared-content",
                    "archive_member": "shared-content_1_0to16.mp4",
                    "local_path": str(train_file),
                },
                {
                    "split": "test",
                    "label": "ai_generated",
                    "source_group": "shared-content",
                    "archive_member": "shared-content_9_0to16.mp4",
                    "local_path": str(test_file),
                },
            ]
            kept = _deduplicate_content_groups(rows)
            self.assertEqual([row["split"] for row in kept], ["test"])
            self.assertTrue(test_file.exists())
            self.assertFalse(train_file.exists())

    def test_temporal_candidate_models_produce_valid_probabilities(self) -> None:
        rng = np.random.default_rng(12)
        real = rng.normal(0.2, 0.06, size=(30, 6))
        ai = rng.normal(0.8, 0.06, size=(30, 6))
        features = np.vstack([real, ai])
        truth = np.asarray([0] * len(real) + [1] * len(ai), dtype=np.int64)
        order = rng.permutation(len(truth))
        features, truth = features[order], truth[order]
        args = argparse.Namespace(trees=100, max_depth=10, min_samples_leaf=2, seed=42)

        for name, model in _candidate_models(args):
            with self.subTest(model=name):
                model.fit(features, truth)
                probabilities = _positive_probabilities(model, features)
                threshold = _best_threshold(truth, probabilities)
                metrics = _evaluate(truth, probabilities, threshold)
                self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
                self.assertGreaterEqual(float(metrics["balanced_accuracy"]), 0.9)
                self.assertIn("roc_auc", metrics)


if __name__ == "__main__":
    unittest.main()
