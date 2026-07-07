from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image

from analyzers.ai_detectors import combined_synthetic_probability, completed_model_count, run_image_detectors
from analyzers.config import get_settings
from analyzers.feedback import build_custom_feedback
from analyzers.fingerprints import build_image_fingerprint
from analyzers.provenance import verify_image_provenance
from analyzers.scoring import clamp_score, get_risk_level, summarize_result, unique_messages
from analyzers.web_research import research_image_context, research_text_claims, research_video_context


def enhance_image_result(
    result: Dict[str, Any],
    image: Image.Image,
    filename: str,
    content_bytes: bytes | None,
    content_label: str,
) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.enable_enhanced_analysis:
        return _with_local_mode(result)

    technical = result.get("technical_details", {})
    attachment_fingerprint = build_image_fingerprint(image, content_bytes, filename) if content_bytes else None
    if attachment_fingerprint:
        technical["attachment_fingerprint"] = attachment_fingerprint
        result["technical_details"] = technical
    metadata_present = bool(technical.get("metadata_fields_found"))
    detectors = run_image_detectors(image, filename, metadata_present, technical)
    provenance = verify_image_provenance(content_bytes, filename) if content_bytes else None
    web_research = (
        research_image_context(filename, attachment_fingerprint=attachment_fingerprint, content_bytes=content_bytes)
        if content_bytes
        else None
    )

    evidence = dict(result.get("evidence", {}))
    detector_probability = combined_synthetic_probability(detectors)
    detector_truth = 50.0 if detector_probability is None else 100.0 - detector_probability * 100.0
    provenance_score = float(provenance["score"]) if provenance else 50.0
    web_score = float(web_research["score"]) if web_research else 50.0
    recalibrated_score = _weighted_score(
        {
            "metadata": evidence.get("metadata_score", 50.0),
            "visual": evidence.get("visual_consistency_score", result.get("truth_score", 50.0)),
            "compression": evidence.get("compression_score", 50.0),
            "forensic": evidence.get("pixel_forensic_score", 50.0),
            "detector": detector_truth,
            "provenance": provenance_score,
            "web": web_score,
        },
        {
            "metadata": 0.06,
            "visual": 0.08,
            "compression": 0.05,
            "forensic": 0.20,
            "detector": 0.36,
            "provenance": 0.10,
            "web": 0.15,
        },
    )

    warnings = list(result.get("warnings", []))
    positives = list(result.get("positive_signals", []))
    if detector_probability is not None and detector_probability >= 0.70:
        warnings.append("AI detector signals indicate a high likelihood of synthetic or generated imagery.")
    elif detector_probability is not None and detector_probability <= 0.30:
        positives.append("AI detector signals did not strongly indicate generated imagery.")
    if provenance and provenance["status"] in {"no_manifest", "tool_unavailable"}:
        warnings.append("No verifiable C2PA content credentials were found for this file.")
    elif provenance and provenance["status"] == "verified":
        positives.append("Verifiable C2PA content credentials were found.")
    if web_research and web_research["status"] == "no_results":
        warnings.append("Automated indexed web research did not find corroborating source leads.")
    elif web_research and web_research["matches_found"] > 0:
        positives.append("Automated indexed web research found possible source or context leads.")
    source_match = _source_match(web_research)
    if source_match.get("status") == "exact_hash_match":
        positives.append("An indexed result appears to match this file's exact fingerprint.")
    elif source_match.get("status") == "exact_visual_match":
        positives.append("Uploaded-image web detection found full visual matches online.")
    elif source_match.get("status") == "partial_visual_match":
        positives.append("Uploaded-image web detection found partial matches or related pages online.")
    elif source_match.get("status") == "visually_similar_match":
        positives.append("Uploaded-image web detection found visually similar images online.")
    elif source_match.get("status") == "possible_context_match":
        positives.append("Indexed search found possible online context, but not a pixel-level match.")
    elif source_match.get("status") == "not_found":
        warnings.append("No indexed source match was found for this attachment from the available search cues.")

    final_score = clamp_score(recalibrated_score)
    risk_level, verdict = get_risk_level(final_score)
    evidence.update(
        {
            "ai_generation_score": round(100.0 - detector_truth, 2),
            "provenance_score": round(provenance_score, 2),
            "web_corroboration_score": round(web_score, 2),
            "overall_risk_score": float(100 - final_score),
        }
    )
    result.update(
        {
            "truth_score": final_score,
            "risk_level": risk_level,
            "verdict": verdict,
            "summary": summarize_result(content_label, final_score, warnings, positives),
            "warnings": unique_messages(warnings),
            "positive_signals": unique_messages(positives),
            "evidence": evidence,
            "analysis_mode": _analysis_mode(web_research),
            "confidence": _confidence(detectors, provenance, web_research),
            "detectors": detectors,
            "provenance": provenance,
            "web_research": web_research,
            "citations": (web_research or {}).get("citations", []),
        }
    )
    result["custom_feedback"] = build_custom_feedback(
        content_label,
        final_score,
        result["warnings"],
        result["positive_signals"],
        detectors,
        provenance,
        web_research,
    )
    return result


def enhance_text_result(result: Dict[str, Any], text: str) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.enable_enhanced_analysis:
        return _with_local_mode(result)

    web_research = research_text_claims(text)
    warnings = list(result.get("warnings", []))
    positives = list(result.get("positive_signals", []))
    base_score = float(result.get("truth_score", 50))
    web_score = float(web_research["score"])
    if web_research["matches_found"] > 0:
        positives.append("Automated indexed web research found possible corroborating context.")
    elif web_research["status"] == "no_results":
        warnings.append("Automated indexed web research did not find corroborating results for the extracted claim.")

    if web_research["status"] in {"completed", "no_results"}:
        final_score = clamp_score(base_score * 0.62 + web_score * 0.38)
    else:
        final_score = clamp_score(base_score)
    risk_level, verdict = get_risk_level(final_score)
    evidence = dict(result.get("evidence", {}))
    evidence.update({"web_corroboration_score": round(web_score, 2), "overall_risk_score": float(100 - final_score)})
    result.update(
        {
            "truth_score": final_score,
            "risk_level": risk_level,
            "verdict": verdict,
            "summary": summarize_result("text post", final_score, warnings, positives),
            "warnings": unique_messages(warnings),
            "positive_signals": unique_messages(positives),
            "evidence": evidence,
            "analysis_mode": _analysis_mode(web_research),
            "confidence": _confidence([], None, web_research),
            "detectors": [],
            "provenance": None,
            "web_research": web_research,
            "citations": web_research.get("citations", []),
        }
    )
    result["custom_feedback"] = build_custom_feedback(
        "text post",
        final_score,
        result["warnings"],
        result["positive_signals"],
        [],
        None,
        web_research,
    )
    return result


def enhance_video_result(result: Dict[str, Any], frame_results: List[Dict[str, Any]], filename: str) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.enable_enhanced_analysis:
        return _with_local_mode(result)

    frame_probabilities = [
        combined_synthetic_probability(frame.get("detectors", []))
        for frame in frame_results
        if frame.get("detectors")
    ]
    frame_probabilities = [probability for probability in frame_probabilities if probability is not None]
    detector_probability = max(frame_probabilities) if frame_probabilities else None
    detectors = [
        {
            "name": "sampled_frame_ai_detector_summary",
            "status": "completed" if detector_probability is not None else "unavailable",
            "label": _probability_label(detector_probability),
            "score": detector_probability,
            "synthetic_probability": detector_probability,
            "details": {
                "frames_with_detector_signals": len(frame_probabilities),
                "note": "Video evidence is aggregated from sampled frames.",
            },
        }
    ]
    frame_notes = [
        warning
        for frame in frame_results[:3]
        for warning in frame.get("warnings", [])[:2]
    ]
    web_research = research_video_context(filename, frame_notes)
    base_score = float(result.get("truth_score", 50))
    detector_truth = 50.0 if detector_probability is None else 100.0 - detector_probability * 100.0
    web_score = float(web_research["score"])
    final_score = clamp_score(base_score * 0.58 + detector_truth * 0.28 + web_score * 0.14)
    warnings = list(result.get("warnings", []))
    positives = list(result.get("positive_signals", []))
    if detector_probability is not None and detector_probability >= 0.70:
        warnings.append("One or more sampled frames had high synthetic-image detector signals.")
    if web_research["matches_found"] > 0:
        positives.append("Automated indexed web research found possible video context leads.")
    elif web_research["status"] == "no_results":
        warnings.append("Automated indexed web research did not find corroborating context for this video.")

    risk_level, verdict = get_risk_level(final_score)
    evidence = dict(result.get("evidence", {}))
    evidence.update(
        {
            "sampled_frame_ai_generation_score": round(100.0 - detector_truth, 2),
            "web_corroboration_score": round(web_score, 2),
            "overall_risk_score": float(100 - final_score),
        }
    )
    result.update(
        {
            "truth_score": final_score,
            "risk_level": risk_level,
            "verdict": verdict,
            "summary": summarize_result("video", final_score, warnings, positives),
            "warnings": unique_messages(warnings),
            "positive_signals": unique_messages(positives),
            "evidence": evidence,
            "analysis_mode": _analysis_mode(web_research),
            "confidence": _confidence(detectors, None, web_research),
            "detectors": detectors,
            "provenance": None,
            "web_research": web_research,
            "citations": web_research.get("citations", []),
        }
    )
    result["custom_feedback"] = build_custom_feedback(
        "video",
        final_score,
        result["warnings"],
        result["positive_signals"],
        detectors,
        None,
        web_research,
    )
    return result


def _with_local_mode(result: Dict[str, Any]) -> Dict[str, Any]:
    result.setdefault("analysis_mode", "local_heuristic")
    result.setdefault("confidence", 0.55)
    result.setdefault("detectors", [])
    result.setdefault("provenance", None)
    result.setdefault("web_research", None)
    result.setdefault("citations", [])
    result.setdefault(
        "custom_feedback",
        {
            "headline": "Local heuristic analysis only",
            "explanation": "Enhanced analysis is disabled, so this report uses local heuristic signals only.",
            "evidence_notes": result.get("warnings", [])[:3],
            "next_steps": result.get("recommendations", [])[:3],
        },
    )
    return result


def _weighted_score(values: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    weight_total = 0.0
    for key, weight in weights.items():
        total += max(0.0, min(100.0, float(values.get(key, 50.0)))) * weight
        weight_total += weight
    if weight_total <= 0:
        return 50.0
    return total / weight_total


def _analysis_mode(web_research: Dict[str, Any] | None) -> str:
    if web_research and web_research.get("status") in {"completed", "no_results", "error"}:
        return "enhanced_free_hybrid"
    return "enhanced_free_local"


def _confidence(
    detectors: List[Dict[str, Any]],
    provenance: Dict[str, Any] | None,
    web_research: Dict[str, Any] | None,
) -> float:
    confidence = 0.45
    if completed_model_count(detectors) > 0:
        confidence += 0.18
    elif any(detector.get("status") == "completed" for detector in detectors):
        confidence += 0.08
    if provenance and provenance.get("status") in {"verified", "no_manifest"}:
        confidence += 0.10
    if web_research and web_research.get("status") in {"completed", "no_results"}:
        confidence += 0.15
    return round(max(0.2, min(0.9, confidence)), 2)


def _probability_label(probability: float | None) -> str | None:
    if probability is None:
        return None
    if probability >= 0.65:
        return "sampled_frames_likely_synthetic"
    if probability <= 0.30:
        return "sampled_frames_lower_synthetic_signal"
    return "sampled_frames_uncertain"


def _source_match(web_research: Dict[str, Any] | None) -> Dict[str, Any]:
    details = (web_research or {}).get("details")
    if not isinstance(details, dict):
        return {}
    source_match = details.get("source_match")
    return source_match if isinstance(source_match, dict) else {}
