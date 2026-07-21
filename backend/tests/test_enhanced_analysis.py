from __future__ import annotations

import io
import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from analyzers.enhanced import enhance_image_result, enhance_video_result
from analyzers.feedback import build_media_feedback
from analyzers.ai_detectors import (
    _covering_tiles,
    _normalize_outputs,
    _prepare_classifier_image,
    _synthetic_probability,
    combined_synthetic_probability,
    reuse_full_frame_predictions_as_single_tile,
    run_tiled_manipulation_detectors,
)
from analyzers.image_forensics import analyze_image_forensics
from analyzers.image_decision import assess_image_evidence
from analyzers.image_analyzer import analyze_frame_array, analyze_image_bytes
from analyzers.config import _configured_models
from analyzers.metadata import analyze_metadata_evidence
from analyzers.provenance import verify_image_provenance
from analyzers.text_analyzer import analyze_text
from analyzers.video_analyzer import _uniform_frame_positions, analyze_video_path
from analyzers.web_research import research_image_context, research_text_claims
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
                "score": 0.97,
                "synthetic_probability": 0.97,
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

        self.assertEqual(enhanced["truth_score"], 74)
        self.assertIn("ai_generation_score", enhanced["evidence"])
        self.assertTrue(enhanced["technical_details"]["ai_detector_summary"]["learned_model_available"])
        self.assertEqual(enhanced["assessment"]["verdict"], "likely_ai_generated")
        feedback = enhanced["custom_feedback"]
        self.assertEqual(feedback["headline"], "This image is likely AI-generated")
        self.assertIn("warning, not proof", feedback["plain_language_summary"])
        self.assertTrue(any("generation pipeline" in item for item in feedback["reasons_it_might_be_ai"]))
        self.assertIn("reasons_it_might_not_be_ai", feedback)
        self.assertTrue(feedback["uncertainty_note"])
        AnalysisResponse(**enhanced)

    def test_web_research_skips_without_brave_key(self) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}, clear=False):
            result = research_text_claims("A test claim that should not call the network.")
        self.assertEqual(result["status"], "not_configured")
        self.assertEqual(result["provider"], "brave_search")
        self.assertEqual(result["score"], 50.0)
        self.assertEqual(result["details"]["source_match"]["status"], "not_checked")

    def test_caption_overlay_is_not_treated_as_strong_ai_signal(self) -> None:
        rng = np.random.default_rng(7)
        base = np.zeros((420, 640, 3), dtype=np.uint8)
        base[:, :, 0] = np.linspace(120, 185, 640, dtype=np.uint8)
        base[:, :, 1] = np.linspace(135, 205, 420, dtype=np.uint8)[:, None]
        base[:, :, 2] = 150
        noise = rng.normal(0, 7, base.shape).astype(np.int16)
        base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.putText(base, "YES", (250, 115), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (0, 0, 0), 9, cv2.LINE_AA)
        cv2.putText(base, "YES", (250, 115), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (255, 255, 255), 4, cv2.LINE_AA)
        image = Image.fromarray(base, "RGB")

        forensics = analyze_image_forensics(image, filename="captioned-real-photo.jpg")

        self.assertTrue(forensics["caption_overlay"]["is_likely"])
        self.assertLess(forensics["synthetic_artifact_probability"], 0.50)

    def test_google_vision_web_detection_exact_match(self) -> None:
        google_payload = {
            "responses": [
                {
                    "webDetection": {
                        "fullMatchingImages": [{"url": "https://example.com/success-kid.jpg"}],
                        "pagesWithMatchingImages": [
                            {
                                "url": "https://example.com/original",
                                "pageTitle": "Original photo source",
                                "fullMatchingImages": [{"url": "https://example.com/success-kid.jpg"}],
                            }
                        ],
                        "bestGuessLabels": [{"label": "success kid"}],
                        "webEntities": [{"description": "Success Kid", "score": 0.92}],
                    }
                }
            ]
        }
        with patch.dict(os.environ, {"GOOGLE_VISION_API_KEY": "test-key", "BRAVE_SEARCH_API_KEY": ""}, clear=False), patch(
            "analyzers.web_research._google_vision_post",
            return_value={"status": "ok", "data": google_payload},
        ):
            result = research_image_context(
                "success-kid.jpg",
                attachment_fingerprint={"sha256": "abc"},
                content_bytes=b"image-bytes",
            )

        self.assertEqual(result["provider"], "google_vision_web_detection")
        self.assertEqual(result["details"]["source_match"]["status"], "exact_visual_match")
        self.assertGreater(result["matches_found"], 0)

    def test_google_visual_clues_feed_indexed_search_fallback(self) -> None:
        google_payload = {
            "responses": [
                {
                    "webDetection": {
                        "bestGuessLabels": [{"label": "Eiffel Tower at night"}],
                        "webEntities": [{"description": "Paris landmark", "score": 0.88}],
                    }
                }
            ]
        }
        brave_result = {
            "status": "no_results",
            "provider": "brave_search",
            "score": 50.0,
            "queries": [],
            "matches_found": 0,
            "summary": "No indexed matches.",
            "citations": [],
            "details": {},
        }
        with patch.dict(
            os.environ,
            {"GOOGLE_VISION_API_KEY": "test-key", "BRAVE_SEARCH_API_KEY": "brave-key"},
            clear=False,
        ), patch(
            "analyzers.web_research._google_vision_post",
            return_value={"status": "ok", "data": google_payload},
        ), patch("analyzers.web_research._run_research", return_value=brave_result) as indexed_search:
            result = research_image_context("IMG_0001.jpg", content_bytes=b"image-bytes")

        query = indexed_search.call_args.args[0][0]
        self.assertIn("eiffel", query)
        self.assertIn("paris", query)
        self.assertEqual(
            result["details"]["visual_query_clues"]["best_guess_labels"],
            ["Eiffel Tower at night"],
        )

    def test_provenance_fallback_when_tools_absent(self) -> None:
        with patch("analyzers.provenance._try_c2pa_python", return_value=None), patch(
            "analyzers.provenance.shutil.which", return_value=None
        ):
            result = verify_image_provenance(b"not-really-an-image", "sample.png")
        self.assertEqual(result["status"], "tool_unavailable")
        self.assertLess(result["score"], 50)

    def test_image_and_text_schema_compatibility(self) -> None:
        with patch.dict(
            os.environ,
            {"ENABLE_LOCAL_AI_MODELS": "false", "BRAVE_SEARCH_API_KEY": "", "GOOGLE_VISION_API_KEY": ""},
            clear=False,
        ):
            image = Image.new("RGB", (512, 512), color=(120, 80, 160))

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
        with patch.dict(
            os.environ,
            {"ENABLE_LOCAL_AI_MODELS": "false", "BRAVE_SEARCH_API_KEY": "", "GOOGLE_VISION_API_KEY": ""},
            clear=False,
        ):
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
        self.assertFalse(enhanced_video["technical_details"]["ai_detector_summary"]["learned_model_available"])
        self.assertEqual(enhanced_video["custom_feedback"]["headline"], "We cannot tell with enough confidence")
        self.assertIn("insufficient", enhanced_video["custom_feedback"]["plain_language_summary"])
        VideoAnalysisResponse(**enhanced_video)

    def test_exhaustive_video_mode_analyzes_every_decoded_frame(self) -> None:
        width, height, frame_count = 320, 240, 6
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as temp:
            path = temp.name
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 6.0, (width, height))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(frame_count):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:, :, 1] = 30 + index * 20
                cv2.circle(frame, (50 + index * 18, 120), 28, (240, 180, 40), -1)
                writer.write(frame)
        finally:
            writer.release()

        try:
            with patch.dict(
                os.environ,
                {
                    "ENABLE_LOCAL_AI_MODELS": "false",
                    "BRAVE_SEARCH_API_KEY": "",
                    "GOOGLE_VISION_API_KEY": "",
                    "VIDEO_ANALYSIS_MODE": "exhaustive",
                    "VIDEO_FRAME_STRIDE": "1",
                    "VIDEO_MAX_FRAMES": "0",
                    "VIDEO_TILE_ANALYSIS": "false",
                },
                clear=False,
            ):
                result = analyze_video_path(path, "unit-test.avi")
        finally:
            os.unlink(path)

        coverage = result["technical_details"]["analysis_coverage"]
        self.assertEqual(result["frames_analyzed"], frame_count)
        self.assertEqual(result["technical_details"]["decoded_frame_count"], frame_count)
        self.assertTrue(coverage["exhaustive"])
        self.assertEqual(coverage["coverage_percent"], 100.0)
        self.assertEqual(coverage["native_pixels_examined"], width * height * frame_count)
        self.assertIn("video_model_features", result["technical_details"])
        VideoAnalysisResponse(**result)

    def test_tiled_scan_boxes_cover_every_source_pixel(self) -> None:
        image = Image.new("RGB", (1000, 731), color=(30, 80, 120))
        _, boxes = _covering_tiles(image, tile_size=448, overlap=0.15)
        coverage = np.zeros((731, 1000), dtype=np.uint8)
        for left, top, right, bottom in boxes:
            coverage[top:bottom, left:right] = 1
        self.assertTrue(np.all(coverage == 1))

    def test_tiled_manipulation_scan_returns_localized_support(self) -> None:
        class Classifier:
            model = SimpleNamespace(config=SimpleNamespace())

            def __call__(self, images, **kwargs):
                return [
                    [
                        {"label": "ai_manipulated", "score": 0.92},
                        {"label": "real_camera", "score": 0.08},
                    ]
                    for _ in images
                ]

        image = Image.new("RGB", (1000, 731), color=(30, 80, 120))
        with patch("analyzers.ai_detectors._load_pipeline", return_value=Classifier()):
            results = run_tiled_manipulation_detectors(image, ["test-manipulation-model"])

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["task"], "manipulation")
        self.assertIsNone(result["synthetic_probability"])
        self.assertGreater(result["manipulation_probability"], 0.9)
        self.assertTrue(result["suspicious_regions"])
        self.assertIn("manipulation_score", result["suspicious_regions"][0])

    def test_declared_video_frame_preprocessing_is_applied(self) -> None:
        classifier = SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(
                    truthshield_training_frame_encoding="opencv_jpeg_95",
                    truthshield_preprocess_max_dimension=384,
                )
            )
        )
        prepared = _prepare_classifier_image(
            classifier,
            Image.new("RGB", (720, 480), color=(20, 90, 180)),
        )
        self.assertEqual(prepared.size, (384, 256))

    def test_temporal_model_positions_cover_video_start_and_end(self) -> None:
        positions = _uniform_frame_positions(frame_count=172, limit=16)
        self.assertEqual(len(positions), 16)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 171)

    def test_reused_single_tile_does_not_double_weight_the_model(self) -> None:
        detectors = [
            {
                "name": "local_heuristic_synthetic_likelihood",
                "status": "completed",
                "synthetic_probability": 0.2,
                "details": {},
            },
            {
                "name": "test-model",
                "status": "completed",
                "label": "ai_generated",
                "score": 0.8,
                "synthetic_probability": 0.8,
                "details": {},
            },
        ]
        before = combined_synthetic_probability(detectors)
        reused = reuse_full_frame_predictions_as_single_tile(detectors, ["test-model"])
        after = combined_synthetic_probability([*detectors, *reused])
        self.assertEqual(before, after)
        self.assertTrue(reused[0]["details"]["reused_full_frame_prediction"])

    def test_learned_detector_is_not_diluted_by_heuristic_fallback(self) -> None:
        detectors = [
            {
                "name": "local_heuristic_synthetic_likelihood",
                "status": "completed",
                "synthetic_probability": 0.08,
                "details": {},
            },
            {
                "name": "truthshield-image-detector-v2",
                "status": "completed",
                "synthetic_probability": 0.97,
                "details": {"model_provider": "huggingface_local"},
            },
        ]

        self.assertEqual(combined_synthetic_probability(detectors), 0.97)

    def test_packaged_detector_stays_ahead_of_stale_environment_models(self) -> None:
        with tempfile.TemporaryDirectory() as model_dir, patch.dict(
            os.environ,
            {"AI_IMAGE_DETECTOR_MODELS": "generic/older-detector"},
            clear=False,
        ):
            configured = _configured_models("AI_IMAGE_DETECTOR_MODELS", [model_dir])

        self.assertEqual(configured, [model_dir, "generic/older-detector"])

    def test_fallback_only_image_cannot_receive_a_reassuring_score(self) -> None:
        image = Image.new("RGB", (640, 480), color=(80, 120, 160))
        base_result = {
            "content_type": "image",
            "truth_score": 98,
            "risk_level": "High Trust",
            "verdict": "Likely trustworthy",
            "summary": "Baseline heuristic report.",
            "warnings": [],
            "positive_signals": ["Basic file checks passed."],
            "recommendations": ["Verify important claims."],
            "evidence": {
                "metadata_score": 98.0,
                "visual_consistency_score": 98.0,
                "compression_score": 98.0,
                "pixel_forensic_score": 98.0,
            },
            "technical_details": {"metadata_fields_found": []},
            "disclaimer": "Test disclaimer.",
        }
        detectors = [
            {
                "name": "local_heuristic_synthetic_likelihood",
                "status": "completed",
                "synthetic_probability": 0.05,
                "details": {},
            }
        ]

        with patch("analyzers.enhanced.run_image_detectors", return_value=detectors):
            enhanced = enhance_image_result(base_result, image, "upload.png", None, "image")

        self.assertEqual(enhanced["truth_score"], 98)
        self.assertFalse(enhanced["technical_details"]["ai_detector_summary"]["learned_model_available"])
        self.assertEqual(enhanced["assessment"]["verdict"], "inconclusive")
        self.assertNotIn("ai_generation_score", enhanced["evidence"])

    def test_legacy_seventy_percent_false_alarm_now_abstains(self) -> None:
        assessment, debug = assess_image_evidence(
            [
                {
                    "name": "truthshield-image-detector-v2",
                    "status": "completed",
                    "synthetic_probability": 0.7296,
                    "details": {"model_provider": "huggingface_local"},
                }
            ],
            {
                "metadata_analysis": analyze_metadata_evidence({}),
                "forensic_analysis": {"synthetic_artifact_probability": 0.32},
                "compression_consistency": {"score": 70.0, "is_inconsistent": False},
            },
            {"status": "no_manifest", "score": 42.0},
            {"status": "not_configured", "score": 50.0, "details": {"source_match": {"status": "not_checked"}}},
        )

        self.assertEqual(assessment["verdict"], "inconclusive")
        self.assertEqual(debug["decision_thresholds"]["likely_ai_detector_min"], 0.95)
        self.assertIsNone(debug["combined_calibrated_score"])

    def test_missing_metadata_and_detector_failure_are_inconclusive(self) -> None:
        assessment, _ = assess_image_evidence(
            [{"name": "truthshield-image-detector-v2", "status": "error", "synthetic_probability": None}],
            {
                "metadata_analysis": analyze_metadata_evidence({}),
                "forensic_analysis": {"synthetic_artifact_probability": 0.88},
                "compression_consistency": {"score": 20.0, "is_inconsistent": True},
            },
            {"status": "error", "score": 45.0},
            {"status": "error", "score": 50.0, "details": {"source_match": {"status": "not_checked"}}},
        )

        self.assertEqual(assessment["verdict"], "inconclusive")
        self.assertTrue(any("not evidence of AI generation" in item for item in assessment["limitations"]))

    def test_weak_heuristic_cannot_trigger_ai_without_learned_model(self) -> None:
        assessment, _ = assess_image_evidence(
            [
                {
                    "name": "local_heuristic_synthetic_likelihood",
                    "status": "completed",
                    "synthetic_probability": 0.98,
                }
            ],
            {
                "metadata_analysis": analyze_metadata_evidence({}),
                "forensic_analysis": {"synthetic_artifact_probability": 0.92},
                "compression_consistency": {"score": 15.0, "is_inconsistent": True},
            },
            None,
            None,
        )

        self.assertEqual(assessment["verdict"], "inconclusive")

    def test_conservative_thresholds_preserve_both_decisive_outcomes(self) -> None:
        common = {
            "metadata_analysis": analyze_metadata_evidence({}),
            "forensic_analysis": {"synthetic_artifact_probability": 0.30},
            "compression_consistency": {"score": 70.0, "is_inconsistent": False},
        }
        authentic, _ = assess_image_evidence(
            [{
                "name": "truthshield-image-detector-v2",
                "status": "completed",
                "synthetic_probability": 0.05,
                "manipulation_probability": 0.04,
            }],
            common,
            None,
            None,
        )
        synthetic, _ = assess_image_evidence(
            [{"name": "truthshield-image-detector-v2", "status": "completed", "synthetic_probability": 0.97}],
            common,
            None,
            None,
        )

        self.assertEqual(authentic["verdict"], "likely_authentic")
        self.assertEqual(synthetic["verdict"], "likely_ai_generated")

    def test_camera_metadata_conflict_abstains(self) -> None:
        assessment, _ = assess_image_evidence(
            [{"name": "truthshield-image-detector-v2", "status": "completed", "synthetic_probability": 0.97}],
            {
                "metadata_analysis": analyze_metadata_evidence({"Make": "Example Camera"}),
                "forensic_analysis": {"synthetic_artifact_probability": 0.30},
                "compression_consistency": {"score": 75.0, "is_inconsistent": False},
            },
            None,
            None,
        )

        self.assertEqual(assessment["verdict"], "inconclusive")

    def test_explicit_ai_software_metadata_is_positive_evidence(self) -> None:
        assessment, _ = assess_image_evidence(
            [{"name": "truthshield-image-detector-v2", "status": "error", "synthetic_probability": None}],
            {
                "metadata_analysis": analyze_metadata_evidence({"Software": "ComfyUI"}),
                "forensic_analysis": {"synthetic_artifact_probability": 0.40},
                "compression_consistency": {"score": 75.0, "is_inconsistent": False},
            },
            None,
            None,
        )

        self.assertEqual(assessment["verdict"], "inconclusive")
        self.assertTrue(any("ComfyUI" in item for item in assessment["evidence_raising_concern"]))
        feedback = build_media_feedback(assessment, [], "image")
        self.assertEqual(feedback["headline"], "We cannot tell with enough confidence")
        self.assertTrue(any("AI-generation software" in item for item in feedback["reasons_it_might_be_ai"]))

    def test_invalid_and_unknown_model_outputs_do_not_become_ai_scores(self) -> None:
        normalized = _normalize_outputs(
            [
                {"label": "ai_generated", "score": math.nan},
                {"label": "real_camera", "score": 0.8},
            ]
        )
        self.assertEqual(_synthetic_probability(normalized), 0.0)
        self.assertIsNone(_synthetic_probability([{"label": "class_0", "score": 1.0}]))


if __name__ == "__main__":
    unittest.main()
