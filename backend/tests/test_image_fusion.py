from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyzers.ai_detectors import combined_synthetic_probability
from analyzers.community_forensics import _official_test_preprocess
from analyzers.image_fusion import (
    EXPECTED_FEATURES,
    FUSION_NAME,
    _load_fusion_bundle,
    run_image_generation_fusion,
)


class ImageFusionTests(unittest.TestCase):
    def tearDown(self) -> None:
        _load_fusion_bundle.cache_clear()

    def test_serialized_fusion_pipeline_is_used_as_generation_score(self) -> None:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(C=0.25, random_state=42)),
            ]
        )
        model.fit(
            [[-5.0, -4.0], [-3.0, -2.0], [2.0, 3.0], [5.0, 4.0]],
            [0, 0, 1, 1],
        )
        detectors = [
            {
                "name": "community-forensics::OwensLab/commfor-model-224",
                "status": "completed",
                "task": "generation",
                "synthetic_probability": 0.95,
            },
            {
                "name": "/app/training/models/truthshield-image-detector-v4-comparison",
                "model_version": "image-comparison-v4-pilot",
                "status": "completed",
                "task": "generation",
                "synthetic_probability": 0.85,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fusion.joblib"
            joblib.dump(
                {
                    "model": model,
                    "feature_names": EXPECTED_FEATURES,
                    "input_model_versions": {"community_forensics": "test", "truthshield_comparison": "test"},
                    "fit_split": "tuning",
                    "calibration_status": "not_calibrated",
                },
                path,
            )
            result = run_image_generation_fusion(detectors, str(path))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["name"], FUSION_NAME)
        self.assertGreater(result["synthetic_probability"], 0.5)
        self.assertEqual(combined_synthetic_probability([*detectors, result]), result["synthetic_probability"])

    def test_fusion_abstains_when_an_input_specialist_is_missing(self) -> None:
        result = run_image_generation_fusion(
            [{
                "name": "community-forensics::OwensLab/commfor-model-224",
                "status": "completed",
                "task": "generation",
                "synthetic_probability": 0.9,
            }],
            "missing.joblib",
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["synthetic_probability"])

    def test_official_224_test_preprocessing_is_deterministic(self) -> None:
        import torch

        image = Image.fromarray(np.arange(320 * 480 * 3, dtype=np.uint32).reshape(320, 480, 3).astype(np.uint8))
        first = _official_test_preprocess(image, torch=torch)
        second = _official_test_preprocess(image, torch=torch)

        self.assertEqual(tuple(first.shape), (1, 3, 224, 224))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()
