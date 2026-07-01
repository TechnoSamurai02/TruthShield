from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from analyzers.enhanced import enhance_image_result, enhance_video_result
from analyzers.image_analyzer import analyze_frame_array, analyze_image_bytes
from analyzers.provenance import verify_image_provenance
from analyzers.text_analyzer import analyze_text
from analyzers.web_research import research_text_claims
from models.schemas import AnalysisResponse, VideoAnalysisResponse


class EnhancedAnalysisTests(unittest.TestCase):
    def test_high_ai_probability_recalibrates_generated_image_lower(self) -> None:
        image = Image.new("RGB", (900, 600), color=(90, 140, 70))
        base_result = {
            "content_type": "image",
            "truth_score": 74,
            "risk_level": "Medium Trust",
            "verdict": "Needs light verification",
            "summary": "Baseline heuristic report.",
            "warnings": ["No readable EXIF metadata was found."],
            "positive_signals": ["Image dimensions are within a reasonable range."],
            "recommendations": ["Verify before sharing."],
            "evidence": {
                "metadata_score": 35.0,
                "visual_consistency_score": 98.0,
                "compression_score": 60.0,
                "source_score": 50.0,
                "overall_risk_score": 26.0,
            },
            "technical_details": {
                "filename": "ChatGPT Image.png",
                "detected_format": "PNG",
                "metadata_fields_found": [],
                "entropy": 6.2,
                "blur_laplacian_variance": 120.0,
                "compression_consistency": {"is_inconsistent": False},
            },
            "disclaimer": "Test disclaimer.",
        }
        detectors = [
            {
                "name": "mock_detector",
                "status": "completed",
                "label": "fake",
                "score": 0.94,
                "synthetic_probability": 0.94,
                "details": {},
            }
        ]
        provenance = {
            "status": "no_manifest",
            "score": 42.0,
            "summary": "No C2PA content credentials were found.",
            "details": {},
        }
        web = {
            "status": "no_results",
            "provider": "brave_search",
            "score": 38.0,
            "queries": ["world cup free kick"],
            "matches_found": 0,
            "summary": "No corroborating indexed web results were found.",
            "citations": [],
            "details": {},
        }

        with patch("analyzers.enhanced.run_image_detectors", return_value=detectors), patch(
            "analyzers.enhanced.verify_image_provenance", return_value=provenance
        ), patch("analyzers.enhanced.research_image_context", return_value=web):
            enhanced = enhance_image_result(base_result, image, "ChatGPT Image.png", b"image-bytes", "image")

        self.assertLess(enhanced["truth_score"], 50)
        self.assertIn("ai_generation_score", enhanced["evidence"])
        self.assertTrue(enhanced["custom_feedback"]["headline"])
        AnalysisResponse(**enhanced)

    def test_web_research_skips_without_brave_key(self) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
            result = research_text_claims("A test claim that should not call the network.")
        self.assertEqual(result["status"], "not_configured")
        self.assertEqual(result["provider"], "brave_search")
        self.assertEqual(result["score"], 50.0)
        self.assertEqual(result["details"]["source_match"]["status"], "not_checked")

    def test_provenance_fallback_when_tools_absent(self) -> None:
        with patch("analyzers.provenance._try_c2pa_python", return_value=None), patch(
            "analyzers.provenance.shutil.which", return_value=None
        ):
            result = verify_image_provenance(b"not-really-an-image", "sample.png")
        self.assertEqual(result["status"], "tool_unavailable")
        self.assertLess(result["score"], 50)

    def test_image_and_text_schema_compatibility(self) -> None:
        with patch.dict(os.environ, {"ENABLE_LOCAL_AI_MODELS": "false", "BRAVE_SEARCH_API_KEY": ""}, clear=False):
            image = Image.new("RGB", (512, 512), color=(120, 80, 160))
            import io

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_result = analyze_image_bytes(buffer.getvalue(), "sample.png")
            text_result = analyze_text("According to Reuters, a city council vote happened in 2026.")

        AnalysisResponse(**image_result)
        AnalysisResponse(**text_result)
        self.assertIn("custom_feedback", image_result)
        self.assertIn("web_research", text_result)
        fingerprint = image_result["technical_details"]["attachment_fingerprint"]
        self.assertEqual(len(fingerprint["sha256"]), 64)
        self.assertEqual(len(fingerprint["perceptual_hashes"]["phash"]), 16)
        self.assertEqual(image_result["web_research"]["details"]["source_match"]["status"], "not_checked")

    def test_video_frame_and_video_schema_compatibility(self) -> None:
        with patch.dict(os.environ, {"ENABLE_LOCAL_AI_MODELS": "false", "BRAVE_SEARCH_API_KEY": ""}, clear=False):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_result = analyze_frame_array(frame, "unit-test-frame")
            video_result = {
                "content_type": "video",
                "truth_score": 62,
                "risk_level": "Medium Trust",
                "verdict": "Needs light verification",
                "summary": "Video baseline.",
                "warnings": [],
                "positive_signals": ["Basic video metadata could be read."],
                "recommendations": ["Verify before sharing."],
                "evidence": {"frame_analysis_score": 62.0, "overall_risk_score": 38.0},
                "frames_analyzed": 1,
                "suspicious_frames": [],
                "technical_details": {"filename": "sample.mp4"},
                "disclaimer": "Test disclaimer.",
            }
            enhanced_video = enhance_video_result(video_result, [frame_result], "sample.mp4")

        self.assertEqual(frame_result["content_type"], "video_frame")
        VideoAnalysisResponse(**enhanced_video)


if __name__ == "__main__":
    unittest.main()
