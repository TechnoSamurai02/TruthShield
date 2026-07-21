from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from training.check_image_robustness import _robustness_variants
from training.evaluate_image_detector import _likely_ai_threshold_metrics, _three_way_metrics
from training.prepare_defactify_sample import V4_GENERATOR_SPLITS, _group_split, _target_split
from training.media_manifest import MediaRecord, read_manifest, sha256_file, validate_records, write_jsonl
from training.prepare_manipulation_pairs import main as prepare_manipulation_pairs
from training.prepare_diffusion_manipulation_pairs import (
    SPLIT_MODEL_SPECS,
    _random_inpainting_mask,
)
from training.evaluate_manipulation_localizer import _localized_support, _threshold_metrics
from training.train_image_detector import (
    _augment_image,
    _binary_manipulation_dataset,
    _training_data_files,
)
from training.train_manipulation_localizer import _balanced_sample_weights, _image_score


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

    def test_paired_manipulation_dataset_is_balanced_and_split_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "paired"
            records = []
            for index, split in enumerate(("train", "tuning", "calibration", "locked_test")):
                authentic_path = source / split / "real_camera" / "authentic.png"
                generated_path = source / split / "ai_generated" / "generated.png"
                authentic_path.parent.mkdir(parents=True, exist_ok=True)
                generated_path.parent.mkdir(parents=True, exist_ok=True)
                authentic = np.zeros((128, 160, 3), dtype=np.uint8)
                authentic[:, :, 0] = np.arange(160, dtype=np.uint8)
                authentic[:, :, 1] = 40 + index * 20
                Image.fromarray(authentic).save(authentic_path)
                Image.new("RGB", (160, 128), color=(160, 40 + index * 20, 90)).save(generated_path)
                for path, label, generator in (
                    (authentic_path, "real_camera", "authentic"),
                    (generated_path, "ai_generated", f"held-out-generator-{split}"),
                ):
                    records.append(
                        MediaRecord(
                            path=path.relative_to(source).as_posix(),
                            sha256=sha256_file(path),
                            media_type="image",
                            class_label=label,
                            source="unit-test-owned",
                            license="unit-test-owned",
                            generator_or_editor=generator,
                            parent_media=None,
                            transformation="none",
                            semantic_category="unit-test",
                            source_group=f"{split}-{label}",
                            split=split,
                        )
                    )
            write_jsonl(source / "manifest.v4.jsonl", records)
            with patch(
                "sys.argv",
                [
                    "prepare_manipulation_pairs.py",
                    "--source-dir",
                    str(source),
                    "--output-dir",
                    str(output),
                    "--clean-output",
                ],
            ):
                prepare_manipulation_pairs()

            prepared = list(read_manifest(output / "manifest.v4.jsonl"))
            report = validate_records(prepared)
            labels_by_split = {
                split: [row["class_label"] for row in prepared if row["split"] == split]
                for split in ("train", "tuning", "calibration", "locked_test")
            }
            localized = list(read_manifest(output / "localization.v4.jsonl"))

        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(len(prepared), 24)
        self.assertEqual(len(localized), 8)
        for labels in labels_by_split.values():
            self.assertEqual(labels.count("real_camera"), 2)
            self.assertEqual(labels.count("ai_generated"), 2)
            self.assertEqual(labels.count("ai_manipulated"), 2)

    def test_binary_manipulation_objective_balances_train_only(self) -> None:
        from datasets import ClassLabel, Dataset, DatasetDict, Features, Value, concatenate_datasets

        features = Features(
            {
                "item": Value("int64"),
                "label": ClassLabel(
                    names=["ai_generated", "ai_manipulated", "real_camera"]
                ),
            }
        )
        source = Dataset.from_dict(
            {"item": [0, 1, 2, 3], "label": [0, 0, 1, 2]},
            features=features,
        )
        converted = _binary_manipulation_dataset(
            DatasetDict({"train": source, "validation": source}),
            source_labels=["ai_generated", "ai_manipulated", "real_camera"],
            ClassLabel=ClassLabel,
            DatasetDict=DatasetDict,
            concatenate_datasets=concatenate_datasets,
            seed=42,
        )

        train_labels = list(converted["train"]["label"])
        validation_labels = list(converted["validation"]["label"])
        self.assertEqual(converted["train"].features["label"].names, ["unaltered_media", "ai_manipulated"])
        self.assertEqual(train_labels.count(0), train_labels.count(1))
        self.assertEqual(len(validation_labels), 4)
        self.assertEqual(validation_labels.count(1), 1)

    def test_manipulation_augmentation_does_not_crop_away_the_edit(self) -> None:
        image = Image.new("RGB", (100, 100), color=(30, 80, 120))
        with patch("training.train_image_detector.random.random", side_effect=[1, 0, 1, 1, 1, 1, 1]), patch(
            "training.train_image_detector.random.uniform", return_value=0.82
        ), patch("training.train_image_detector.random.randint", return_value=0):
            cropped = _augment_image(image, allow_random_crop=True)
        with patch("training.train_image_detector.random.random", side_effect=[1, 1, 1, 1, 1, 1]):
            preserved = _augment_image(image, allow_random_crop=False)

        self.assertEqual(cropped.size, (82, 82))
        self.assertEqual(preserved.size, (100, 100))

    def test_diffusion_editor_families_are_isolated_by_split(self) -> None:
        families = [spec["family"] for spec in SPLIT_MODEL_SPECS.values()]
        model_ids = [spec["model_id"] for spec in SPLIT_MODEL_SPECS.values()]

        self.assertEqual(len(families), 4)
        self.assertEqual(len(set(families)), 4)
        self.assertEqual(len(set(model_ids)), 4)
        self.assertTrue(all(spec["license"] for spec in SPLIT_MODEL_SPECS.values()))
        self.assertTrue(all(spec["license_url"].startswith("https://") for spec in SPLIT_MODEL_SPECS.values()))

    def test_diffusion_masks_are_nontrivial_and_deterministic(self) -> None:
        import random

        first = _random_inpainting_mask((512, 512), random.Random(71))
        second = _random_inpainting_mask((512, 512), random.Random(71))
        coverage = np.count_nonzero(np.asarray(first)) / (512 * 512)

        self.assertTrue(np.array_equal(np.asarray(first), np.asarray(second)))
        self.assertGreaterEqual(coverage, 0.045)
        self.assertLessEqual(coverage, 0.36)

    def test_localizer_score_uses_a_region_not_one_hot_pixel(self) -> None:
        probability = np.zeros((1, 100, 100), dtype=np.float32)
        probability[:, :10, :10] = 0.8
        probability[:, 50, 50] = 1.0

        score = _image_score(probability, top_fraction=0.01)

        self.assertGreater(score, 0.79)
        self.assertLess(score, 0.81)

    def test_localized_support_requires_a_meaningful_component(self) -> None:
        isolated = np.zeros((100, 100), dtype=np.float32)
        isolated[10, 10] = 0.99
        region = np.zeros((100, 100), dtype=np.float32)
        region[20:40, 30:60] = 0.9

        isolated_support, _, isolated_ratio = _localized_support(isolated, threshold=0.5)
        region_support, bounds, region_ratio = _localized_support(region, threshold=0.5)

        self.assertFalse(isolated_support)
        self.assertLess(isolated_ratio, 0.001)
        self.assertTrue(region_support)
        self.assertEqual(bounds, [30, 20, 60, 40])
        self.assertAlmostEqual(region_ratio, 0.06)

    def test_localizer_threshold_ignores_unstable_or_unlocalized_scores(self) -> None:
        rows = [
            {
                "label": "ai_manipulated",
                "manipulation_score": 0.95,
                "localized_or_persistent_support": True,
                "stable_across_views": True,
            },
            {
                "label": "real_camera",
                "manipulation_score": 0.99,
                "localized_or_persistent_support": True,
                "stable_across_views": False,
            },
            {
                "label": "ai_generated",
                "manipulation_score": 0.98,
                "localized_or_persistent_support": False,
                "stable_across_views": True,
            },
        ]

        metrics = _threshold_metrics(rows, 0.90)

        self.assertEqual(metrics["predicted_manipulated"], 1)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["authentic_false_warning_rate"], 0.0)
        self.assertEqual(metrics["generated_false_manipulation_rate"], 0.0)

    def test_localizer_balanced_weights_equalize_class_mass(self) -> None:
        labels = [0, 0, 0, 1]
        weights = _balanced_sample_weights(labels)

        self.assertAlmostEqual(sum(weight for label, weight in zip(labels, weights) if not label), 1.0)
        self.assertAlmostEqual(sum(weight for label, weight in zip(labels, weights) if label), 1.0)


if __name__ == "__main__":
    unittest.main()
