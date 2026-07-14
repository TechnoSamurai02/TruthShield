from __future__ import annotations

import unittest

from PIL import Image

from training.check_image_robustness import _robustness_variants


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


if __name__ == "__main__":
    unittest.main()
