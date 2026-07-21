from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from analyzers.manipulation_localizer import (
    _largest_region,
    _load_localizer,
    _scale_region,
    run_manipulation_localizer,
)
from analyzers.model_registry import _artifact_status


class ManipulationLocalizerTests(unittest.TestCase):
    def tearDown(self) -> None:
        _load_localizer.cache_clear()

    def test_region_support_and_source_coordinate_mapping(self) -> None:
        probability = np.zeros((100, 100), dtype=np.float32)
        probability[20:50, 30:70] = 0.95

        region, area_ratio = _largest_region(
            probability,
            threshold=0.5,
            minimum_area_ratio=0.001,
        )

        self.assertEqual(region, [30, 20, 70, 50])
        self.assertAlmostEqual(area_ratio, 0.12)
        self.assertEqual(_scale_region(region, 100, (1000, 500)), [300, 100, 700, 250])

    def test_torchscript_artifact_runs_and_reports_localized_support(self) -> None:
        import torch

        class DummyLocalizer(torch.nn.Module):
            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                logits = torch.full_like(pixel_values[:, :1], -8.0)
                logits[:, :, 56:168, 56:168] = 8.0
                return logits

        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary)
            example = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            traced = torch.jit.trace(DummyLocalizer().eval(), example)
            torch.jit.save(traced, str(artifact / "model.ts"))
            (artifact / "preprocess.json").write_text(
                json.dumps(
                    {
                        "resolution": 224,
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                        "pixel_threshold": 0.5,
                        "model_version": "test-localizer",
                        "calibration_id": "test-calibration",
                    }
                ),
                encoding="utf-8",
            )
            result = run_manipulation_localizer(
                Image.new("RGB", (448, 280), color=(120, 130, 140)),
                str(artifact),
            )
            health = _artifact_status(str(artifact), "manipulation")

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["task"], "manipulation")
        self.assertGreater(result["manipulation_probability"], 0.99)
        self.assertTrue(result["suspicious_regions"])
        self.assertEqual(result["model_version"], "test-localizer")
        self.assertEqual(result["calibration_id"], "test-calibration")
        self.assertEqual(health["status"], "available_for_lazy_load")
        self.assertIn("model.ts", health["files_checked"])
        self.assertIn("preprocess.json", health["files_checked"])

    def test_missing_artifact_is_neutral(self) -> None:
        result = run_manipulation_localizer(
            Image.new("RGB", (300, 300)),
            "missing/manipulation-localizer",
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["manipulation_probability"])
        self.assertEqual(result["suspicious_regions"], [])


if __name__ == "__main__":
    unittest.main()
