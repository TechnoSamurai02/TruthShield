from __future__ import annotations

from typing import Any, Dict, List, Sequence

from PIL import Image

from analyzers.ai_detectors import combined_synthetic_probability, completed_model_count, run_image_detectors
from analyzers.config import get_settings
from analyzers.feedback import build_custom_feedback, build_image_feedback, build_video_feedback
from analyzers.fingerprints import build_image_fingerprint
from analyzers.image_decision import assess_image_evidence
from analyzers.provenance import verify_image_provenance
from analyzers.scoring import clamp_score, get_risk_level, summarize_result, unique_messages
from analyzers.web_research import research_image_context, research_text_claims, research_video_context


def enhance_image_result(
    result: Dict[str, Any],
    image: Image.Image,
    filename: str,
    content_bytes: bytes | None,
    content_label: str,
    detector_model_ids: Sequence[str] | None = None,
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
    detectors = run_image_detectors(image, filename, metadata_present, technical, model_ids=detector_model_ids)
    provenance = verify_image_provenance(content_bytes, filename) if content_bytes else None
    web_research = (
        research_image_context(filename, attachment_fingerprint=attachment_fingerprint, content_bytes=content_bytes)
        if content_bytes
        else None
    )

    evidence = dict(result.get("evidence", {}))
    detector_probability = combined_synthetic_probability(detectors)
    learned_model_count = completed_model_count(detectors)
    learned_model_available = learned_model_count > 0
    assessment, decision_debug = assess_image_evidence(detectors, technical, provenance, web_research)
    # Keep this numeric field only for backwards-compatible API clients. It is
    # now explicitly a local file-context score and does not determine the
    # real/uncertain/AI assessment.
    final_score = clamp_score(float(result.get("truth_score", 50.0)))
    provenance_score = float(provenance["score"]) if provenance and provenance.get("status") == "verified" else 50.0
    web_match_status = str(_source_match(web_research).get("status") or "")
    web_score = (
        float(web_research["score"])
        if web_research and web_match_status in {"exact_hash_match", "exact_visual_match", "partial_visual_match", "visually_similar_match"}
        else 50.0
    )

    warnings = list(result.get("warnings", []))
    positives = list(result.get("positive_signals", []))
    if assessment["verdict"] == "likely_ai_generated_or_manipulated":
        warnings.extend(assessment["evidence_raising_concern"])
        positives = [
            message
            for message in positives
            if not any(
                marker in message
                for marker in (
                    "Pixel-level forensic checks",
                    "JPEG block-boundary artifacts",
                    "Error-level analysis",
                    "repeated textured-patch pattern",
                )
            )
        ]
    else:
        # Low-reliability forensic heuristics remain visible in technical
        # diagnostics, but must not read like an AI accusation when the final
        # evidence assessment abstains or favors authenticity.
        speculative_markers = (
            "AI-generation artifacts",
            "synthetic-image artifact",
            "Fine-grain noise is unusually low",
            "periodic artifacts",
        )
        warnings = [message for message in warnings if not any(marker in message for marker in speculative_markers)]
    positives.extend(assessment["evidence_supporting_authenticity"])
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
    risk_level = assessment["label"]
    verdict = assessment["label"]
    evidence.update(
        {
            "provenance_score": round(provenance_score, 2),
            "web_corroboration_score": round(web_score, 2),
            "overall_risk_score": float(100 - final_score),
        }
    )
    if learned_model_available and detector_probability is not None:
        evidence["ai_generation_score"] = round(float(detector_probability) * 100.0, 2)
    technical["ai_detector_summary"] = {
        "learned_model_available": learned_model_available,
        "completed_learned_models": learned_model_count,
        "synthetic_probability": (
            round(detector_probability, 4)
            if learned_model_available and detector_probability is not None
            else None
        ),
        "fallback_heuristic_score": (
            round(detector_probability, 4)
            if not learned_model_available and detector_probability is not None
            else None
        ),
        "scoring_mode": "three_way_evidence_assessment",
        "decision": assessment["verdict"],
    }
    technical["decision_debug"] = decision_debug
    technical["legacy_context_score_note"] = (
        "The truth_score field is retained for API compatibility and is not used for the image authenticity verdict."
    )
    result.update(
        {
            "truth_score": final_score,
            "risk_level": risk_level,
            "verdict": verdict,
            "summary": assessment["reason"],
            "warnings": unique_messages(warnings),
            "positive_signals": unique_messages(positives),
            "evidence": evidence,
            "analysis_mode": _analysis_mode(web_research),
            "confidence": _confidence(detectors, provenance, web_research),
            "detectors": detectors,
            "provenance": provenance,
            "web_research": web_research,
            "citations": (web_research or {}).get("citations", []),
            "assessment": assessment,
        }
    )
    result["custom_feedback"] = build_image_feedback(assessment, result.get("recommendations", []))
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


def enhance_video_result(
    result: Dict[str, Any],
    frame_results: List[Dict[str, Any]],
    filename: str,
    video_detectors: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.enable_enhanced_analysis:
        return _with_local_mode(result)

    frame_probabilities = [
        combined_synthetic_probability(frame.get("detectors", []))
        for frame in frame_results
        if frame.get("detectors")
    ]
    frame_probabilities = [probability for probability in frame_probabilities if probability is not None]
    learned_frame_model_names = {
        str(detector.get("name") or "")
        for frame in frame_results
        for detector in (frame.get("detectors", []) if isinstance(frame.get("detectors"), list) else [])
        if detector.get("status") == "completed"
        and detector.get("name") != "local_heuristic_synthetic_likelihood"
        and isinstance(detector.get("synthetic_probability"), (int, float))
    }
    learned_frame_model_count = len(learned_frame_model_names)
    learned_frame_signal_count = sum(
        completed_model_count(frame.get("detectors", [])) > 0
        for frame in frame_results
        if isinstance(frame.get("detectors"), list)
    )
    technical = result.get("technical_details") or {}
    temporal_forensics = technical.get("temporal_forensics") or {}
    summarized_probability = temporal_forensics.get("frame_ai_probability")
    if not isinstance(summarized_probability, (int, float)):
        summarized_probability = _robust_frame_probability(frame_probabilities)
    frame_summary_detector = {
        "name": "all_frame_ai_detector_summary",
        "status": "completed" if summarized_probability is not None else "unavailable",
        "label": _probability_label(summarized_probability),
        "score": summarized_probability,
        "synthetic_probability": summarized_probability,
        "details": {
            "frames_with_detector_signals": len(frame_probabilities),
            "frames_with_learned_detector_signals": learned_frame_signal_count,
            "frames_analyzed": result.get("frames_analyzed", len(frame_results)),
            "learned_model_available": learned_frame_model_count > 0,
            "model_type": "learned_frame_aggregate" if learned_frame_model_count > 0 else "heuristic_frame_aggregate",
            "note": "Frame evidence uses a robust mean, upper percentile, high-risk ratio, and sustained-run ratio instead of trusting one isolated frame.",
        },
    }
    detectors = [frame_summary_detector, *(video_detectors or result.get("detectors", []))]
    trained_video_model_available = any(
        detector.get("name") == "trained_truthshield_video_detector"
        and detector.get("status") == "completed"
        and isinstance(detector.get("synthetic_probability"), (int, float))
        for detector in detectors
    )
    learned_model_available = learned_frame_model_count > 0 or trained_video_model_available
    weighted_probabilities: List[tuple[float, float]] = []
    if isinstance(summarized_probability, (int, float)):
        weighted_probabilities.append((float(summarized_probability), 1.25))
    for detector in detectors[1:]:
        probability = detector.get("synthetic_probability")
        if not isinstance(probability, (int, float)):
            continue
        weight = 1.35 if detector.get("name") == "trained_truthshield_video_detector" else 0.55
        weighted_probabilities.append((float(probability), weight))
    detector_probability = (
        sum(probability * weight for probability, weight in weighted_probabilities)
        / sum(weight for _, weight in weighted_probabilities)
        if weighted_probabilities
        else None
    )
    frame_notes = [
        warning
        for frame in frame_results[:3]
        for warning in frame.get("warnings", [])[:2]
    ]
    web_research = research_video_context(filename, frame_notes)
    base_score = float(result.get("truth_score", 50))
    detector_truth = 50.0 if detector_probability is None else 100.0 - detector_probability * 100.0
    web_score = float(web_research["score"])
    final_score = clamp_score(base_score * 0.50 + detector_truth * 0.38 + web_score * 0.12)
    warnings = list(result.get("warnings", []))
    positives = list(result.get("positive_signals", []))
    if detector_probability is not None and detector_probability >= 0.70:
        warnings.append("Frame and temporal detector signals indicate a high likelihood of synthetic video.")
    elif detector_probability is not None and detector_probability <= 0.30:
        positives.append("Frame and temporal detectors did not show strong synthetic-video signals.")
    if web_research["matches_found"] > 0:
        positives.append("Automated indexed web research found possible video context leads.")
    elif web_research["status"] == "no_results":
        warnings.append("Automated indexed web research did not find corroborating context for this video.")

    risk_level, verdict = get_risk_level(final_score)
    evidence = dict(result.get("evidence", {}))
    evidence.update(
        {
            "sampled_frame_ai_generation_score": round(100.0 - detector_truth, 2),
            "video_ai_generation_score": round(100.0 - detector_truth, 2),
            "web_corroboration_score": round(web_score, 2),
            "overall_risk_score": float(100 - final_score),
        }
    )
    technical["ai_detector_summary"] = {
        "learned_model_available": learned_model_available,
        "completed_learned_models": learned_frame_model_count + int(trained_video_model_available),
        "synthetic_probability": round(detector_probability, 4) if learned_model_available and detector_probability is not None else None,
        "fallback_heuristic_score": round(detector_probability, 4) if not learned_model_available and detector_probability is not None else None,
        "scoring_mode": "video_frame_and_temporal_evidence",
    }
    result.update(
        {
            "truth_score": final_score,
            "risk_level": risk_level,
            "verdict": verdict,
            "summary": summarize_result("video", final_score, warnings, positives),
            "warnings": unique_messages(warnings),
            "positive_signals": unique_messages(positives),
            "evidence": evidence,
            "technical_details": technical,
            "analysis_mode": _analysis_mode(web_research),
            "confidence": _confidence(
                detectors,
                None,
                web_research,
                learned_model_available=learned_model_available,
            ),
            "detectors": detectors,
            "provenance": None,
            "web_research": web_research,
            "citations": web_research.get("citations", []),
        }
    )
    result["custom_feedback"] = build_video_feedback(
        detector_probability,
        learned_model_available,
        result["warnings"],
        result["positive_signals"],
        result.get("recommendations", []),
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


def _robust_frame_probability(probabilities: List[float]) -> float | None:
    if not probabilities:
        return None
    ordered = sorted(float(value) for value in probabilities)
    mean = sum(ordered) / len(ordered)
    position = 0.90 * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    p90 = ordered[lower] * (1.0 - (position - lower)) + ordered[upper] * (position - lower)
    high_ratio = sum(value >= 0.65 for value in ordered) / len(ordered)
    return max(0.0, min(1.0, mean * 0.55 + p90 * 0.30 + high_ratio * 0.15))


def _analysis_mode(web_research: Dict[str, Any] | None) -> str:
    if web_research and web_research.get("status") in {"completed", "no_results", "error"}:
        return "enhanced_free_hybrid"
    return "enhanced_free_local"


def _confidence(
    detectors: List[Dict[str, Any]],
    provenance: Dict[str, Any] | None,
    web_research: Dict[str, Any] | None,
    learned_model_available: bool | None = None,
) -> float:
    confidence = 0.45
    has_learned_model = completed_model_count(detectors) > 0 if learned_model_available is None else learned_model_available
    if has_learned_model:
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
        return "analyzed_frames_likely_synthetic"
    if probability <= 0.30:
        return "analyzed_frames_lower_synthetic_signal"
    return "analyzed_frames_uncertain"


def _source_match(web_research: Dict[str, Any] | None) -> Dict[str, Any]:
    details = (web_research or {}).get("details")
    if not isinstance(details, dict):
        return {}
    source_match = details.get("source_match")
    return source_match if isinstance(source_match, dict) else {}
