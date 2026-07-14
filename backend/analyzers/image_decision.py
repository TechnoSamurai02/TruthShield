from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from analyzers.ai_detectors import combined_synthetic_probability, completed_model_count


# These are abstention thresholds for the packaged detector score, not claims
# that the score is a real-world probability. The wide middle band is
# deliberate: an accusation requires a much stronger model signal than the
# legacy frontend's 0.70 cutoff.
AUTHENTIC_DETECTOR_MAX = 0.15
AI_DETECTOR_MIN = 0.95
AUTHENTIC_WITH_CORROBORATION_MAX = 0.30


def assess_image_evidence(
    detectors: List[Dict[str, Any]],
    technical: Dict[str, Any],
    provenance: Dict[str, Any] | None,
    web_research: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    combined_score = _finite_probability(combined_synthetic_probability(detectors))
    learned_available = completed_model_count(detectors) > 0 and combined_score is not None
    detector_score = combined_score if learned_available else None
    metadata = _mapping(technical.get("metadata_analysis"))
    forensics = _mapping(technical.get("forensic_analysis"))
    compression = _mapping(technical.get("compression_consistency"))

    signals = [
        _detector_signal(detectors, detector_score, learned_available),
        _metadata_signal(metadata),
        _forensic_signal(forensics),
        _compression_signal(compression),
        _provenance_signal(provenance),
        _web_signal(web_research),
        {
            "source": "vision_model",
            "status": "not_used",
            "signal": "unavailable",
            "confidence": 0.0,
            "reliability": 0.0,
            "raw_score": None,
            "evidence": [],
            "limitations": [
                "No general-purpose vision-language model is used as an image-forensics authority in this pipeline."
            ],
        },
    ]

    outcome, confidence, reason = _decide(
        detector_score=detector_score,
        learned_available=learned_available,
        metadata=metadata,
        forensics=forensics,
        provenance=provenance,
    )
    supporting_authenticity = _evidence_for(signals, "authentic")
    concerns = _evidence_for(signals, "ai_generated_or_manipulated")
    limitations = _unique(
        limitation
        for signal in signals
        for limitation in signal.get("limitations", [])
        if isinstance(limitation, str)
    )

    assessment = {
        "verdict": outcome,
        "label": _outcome_label(outcome),
        "confidence": confidence,
        "detector_score": round(detector_score, 4) if detector_score is not None else None,
        "reason": reason,
        "evidence_supporting_authenticity": supporting_authenticity,
        "evidence_raising_concern": concerns,
        "limitations": limitations,
        "signals": signals,
    }
    debug = {
        "dedicated_detector_score": assessment["detector_score"],
        "metadata_evidence": metadata,
        "compression_analysis": compression,
        "frequency_domain_analysis": _mapping(forensics.get("frequency_spectrum")),
        "visual_artifact_analysis": {
            "synthetic_artifact_score": _finite_probability(forensics.get("synthetic_artifact_probability")),
            "manipulation_score": _finite_probability(forensics.get("manipulation_probability")),
            "caption_overlay": _mapping(forensics.get("caption_overlay")),
        },
        "web_similarity_evidence": _source_match(web_research),
        "vision_model_assessment": {
            "status": "not_used",
            "reason": "A general-purpose multimodal LLM does not participate in the image verdict.",
        },
        "individual_signal_confidence": [
            {
                "source": signal["source"],
                "status": signal["status"],
                "signal": signal["signal"],
                "confidence": signal["confidence"],
                "reliability": signal["reliability"],
            }
            for signal in signals
        ],
        "combined_calibrated_score": None,
        "combined_score_note": (
            "No arbitrary averaged probability is produced. The system applies explicit abstention rules to independently reported signals."
        ),
        "decision_thresholds": {
            "likely_authentic_detector_max": AUTHENTIC_DETECTOR_MAX,
            "likely_ai_detector_min": AI_DETECTOR_MIN,
            "authentic_with_camera_or_provenance_max": AUTHENTIC_WITH_CORROBORATION_MAX,
            "legacy_frontend_ai_threshold": 0.70,
        },
        "final_decision": outcome,
        "reason_for_decision": reason,
    }
    return assessment, debug


def _decide(
    detector_score: float | None,
    learned_available: bool,
    metadata: Dict[str, Any],
    forensics: Dict[str, Any],
    provenance: Dict[str, Any] | None,
) -> Tuple[str, str, str]:
    ai_metadata = bool(metadata.get("ai_software_marker"))
    camera_metadata = bool(metadata.get("camera_make_or_model_present"))
    verified_provenance = bool(provenance and provenance.get("status") == "verified")
    forensic_score = _finite_probability(forensics.get("synthetic_artifact_probability"))
    forensic_concern = forensic_score is not None and forensic_score >= 0.72

    if ai_metadata:
        if learned_available and detector_score is not None and detector_score <= AUTHENTIC_DETECTOR_MAX:
            return (
                "inconclusive",
                "moderate",
                "An explicit AI-software metadata tag conflicts with a strong detector signal away from AI generation.",
            )
        confidence = "high" if learned_available and detector_score is not None and detector_score >= AI_DETECTOR_MIN else "moderate"
        return (
            "likely_ai_generated_or_manipulated",
            confidence,
            "The file contains an explicit AI-generation software tag; this is positive evidence rather than an inference from appearance.",
        )

    if not learned_available or detector_score is None:
        return (
            "inconclusive",
            "low",
            "The dedicated detector did not return a valid, label-mapped score, so weaker signals cannot support a reliable accusation.",
        )

    if detector_score >= AI_DETECTOR_MIN:
        if (camera_metadata or verified_provenance) and not forensic_concern:
            return (
                "inconclusive",
                "moderate",
                "The strong dedicated-detector concern conflicts with camera metadata or verifiable provenance and lacks forensic corroboration.",
            )
        return (
            "likely_ai_generated_or_manipulated",
            "high" if forensic_concern else "moderate",
            "The dedicated detector exceeded the conservative AI threshold"
            + (" and a separate pixel-forensics check also raised concern." if forensic_concern else "."),
        )

    if detector_score <= AUTHENTIC_DETECTOR_MAX:
        if forensic_concern:
            return (
                "inconclusive",
                "moderate",
                "The dedicated detector strongly leans away from AI generation, but a separate low-reliability forensic check conflicts.",
            )
        return (
            "likely_authentic",
            "high" if detector_score <= 0.05 and (camera_metadata or verified_provenance) else "moderate",
            "The dedicated detector is below the conservative authentic threshold and no independent positive AI evidence was found.",
        )

    if detector_score <= AUTHENTIC_WITH_CORROBORATION_MAX and (camera_metadata or verified_provenance) and not forensic_concern:
        return (
            "likely_authentic",
            "moderate",
            "The detector leans away from AI generation and is supported by camera metadata or verifiable provenance.",
        )

    return (
        "inconclusive",
        "low",
        "The dedicated detector score falls inside the abstention band, where the available evidence is not strong enough for either conclusion.",
    )


def _detector_signal(
    detectors: List[Dict[str, Any]],
    score: float | None,
    learned_available: bool,
) -> Dict[str, Any]:
    completed_names = [
        str(detector.get("name") or "")
        for detector in detectors
        if detector.get("status") == "completed" and detector.get("name") != "local_heuristic_synthetic_likelihood"
    ]
    if not learned_available or score is None:
        return {
            "source": "dedicated_detector",
            "status": "unavailable",
            "signal": "unavailable",
            "confidence": 0.0,
            "reliability": 0.0,
            "raw_score": None,
            "evidence": [],
            "limitations": ["No valid dedicated-detector score was available."],
        }
    if score >= AI_DETECTOR_MIN:
        signal = "ai_generated_or_manipulated"
    elif score <= AUTHENTIC_DETECTOR_MAX:
        signal = "authentic"
    else:
        signal = "neutral"
    packaged = any("truthshield-image-detector" in name.lower() for name in completed_names)
    return {
        "source": "dedicated_detector",
        "status": "completed",
        "signal": signal,
        "confidence": round(abs(score - 0.5) * 2.0, 2),
        "reliability": 0.65 if packaged else 0.50,
        "raw_score": round(score, 4),
        "evidence": [f"The dedicated detector returned an AI-class score of {score:.3f}."],
        "limitations": [
            "This softmax score is a model output, not a calibrated real-world probability or proof."
        ],
    }


def _metadata_signal(metadata: Dict[str, Any]) -> Dict[str, Any]:
    if metadata.get("ai_software_marker"):
        marker = str(metadata["ai_software_marker"])
        return {
            "source": "metadata",
            "status": "completed",
            "signal": "ai_generated_or_manipulated",
            "confidence": 0.95,
            "reliability": 0.85,
            "raw_score": None,
            "evidence": [f"Metadata contains an explicit {marker} software tag."],
            "limitations": list(metadata.get("limitations") or []),
        }
    if metadata.get("camera_make_or_model_present"):
        return {
            "source": "metadata",
            "status": "completed",
            "signal": "authentic",
            "confidence": 0.65,
            "reliability": 0.55,
            "raw_score": None,
            "evidence": ["Camera make or model metadata is present."],
            "limitations": list(metadata.get("limitations") or []),
        }
    evidence = ["Readable metadata is present, but it does not identify a camera."] if metadata.get("metadata_present") else []
    return {
        "source": "metadata",
        "status": "completed",
        "signal": "neutral",
        "confidence": 0.0,
        "reliability": 0.30,
        "raw_score": None,
        "evidence": evidence,
        "limitations": list(metadata.get("limitations") or ["Metadata availability could not be assessed."]),
    }


def _forensic_signal(forensics: Dict[str, Any]) -> Dict[str, Any]:
    score = _finite_probability(forensics.get("synthetic_artifact_probability"))
    if score is None:
        return {
            "source": "pixel_forensics",
            "status": "unavailable",
            "signal": "unavailable",
            "confidence": 0.0,
            "reliability": 0.0,
            "raw_score": None,
            "evidence": [],
            "limitations": ["The handcrafted pixel-forensics checks did not return a valid score."],
        }
    if score >= 0.72:
        signal = "ai_generated_or_manipulated"
        evidence = ["Handcrafted pixel checks found an elevated combination of synthetic-artifact indicators."]
    elif score <= 0.22:
        signal = "authentic"
        evidence = ["Handcrafted pixel checks did not find a strong synthetic-artifact combination."]
    else:
        signal = "neutral"
        evidence = []
    return {
        "source": "pixel_forensics",
        "status": "completed",
        "signal": signal,
        "confidence": round(abs(score - 0.47) * 1.5, 2),
        "reliability": 0.25,
        "raw_score": round(score, 4),
        "evidence": evidence,
        "limitations": [
            "These handcrafted statistics are low-reliability supporting evidence and can react to blur, HDR, editing, or compression."
        ],
    }


def _compression_signal(compression: Dict[str, Any]) -> Dict[str, Any]:
    inconsistent = bool(compression.get("is_inconsistent"))
    return {
        "source": "compression_analysis",
        "status": "completed" if compression else "unavailable",
        "signal": "neutral",
        "confidence": 0.0,
        "reliability": 0.15 if compression else 0.0,
        "raw_score": _finite_number(compression.get("score")),
        "evidence": (["Compression consistency varies across image regions."] if inconsistent else []),
        "limitations": [
            "Compression variation can result from ordinary editing, screenshots, captions, downloads, or repeated saving and is not AI evidence by itself."
        ],
    }


def _provenance_signal(provenance: Dict[str, Any] | None) -> Dict[str, Any]:
    status = str((provenance or {}).get("status") or "unavailable")
    if status == "verified":
        return {
            "source": "provenance",
            "status": status,
            "signal": "authentic",
            "confidence": 0.65,
            "reliability": 0.60,
            "raw_score": _finite_number((provenance or {}).get("score")),
            "evidence": ["Verifiable C2PA content credentials were found and parsed."],
            "limitations": ["Credential presence supports provenance but does not by itself establish how every pixel was created."],
        }
    limitation = (
        "No C2PA credentials were found; most genuine photographs do not carry them, so absence is neutral."
        if status == "no_manifest"
        else "C2PA provenance could not be verified; unavailable provenance is neutral."
    )
    return {
        "source": "provenance",
        "status": status,
        "signal": "neutral" if status == "no_manifest" else "unavailable",
        "confidence": 0.0,
        "reliability": 0.50 if status == "no_manifest" else 0.0,
        "raw_score": _finite_number((provenance or {}).get("score")),
        "evidence": [],
        "limitations": [limitation],
    }


def _web_signal(web_research: Dict[str, Any] | None) -> Dict[str, Any]:
    status = str((web_research or {}).get("status") or "not_checked")
    source_match = _source_match(web_research)
    match_status = str(source_match.get("status") or "not_checked")
    if match_status in {"exact_hash_match", "exact_visual_match"}:
        signal = "authentic"
        confidence = 0.55
        reliability = 0.30
        evidence = ["Uploaded-image web matching found an exact-file or full visual match online."]
    elif match_status == "partial_visual_match":
        signal = "authentic"
        confidence = 0.35
        reliability = 0.20
        evidence = ["Uploaded-image web matching found a partial match or a page containing a related version."]
    else:
        signal = "neutral" if status in {"completed", "no_results"} else "unavailable"
        confidence = 0.0
        reliability = 0.15 if status in {"completed", "no_results"} else 0.0
        evidence = []
    limitations = [
        "A web match supplies real-world context but cannot prove authenticity; AI imagery can depict or reuse real places and images."
    ]
    if status in {"error", "not_configured", "quota_disabled", "quota_exceeded", "no_query", "not_checked"}:
        limitations.append("Web similarity evidence was unavailable or incomplete; this does not count against the image.")
    elif match_status == "not_found":
        limitations.append("No indexed match was found; non-discovery is not evidence of AI generation.")
    return {
        "source": "web_context",
        "status": status,
        "signal": signal,
        "confidence": confidence,
        "reliability": reliability,
        "raw_score": _finite_number((web_research or {}).get("score")),
        "evidence": evidence,
        "limitations": limitations,
    }


def _source_match(web_research: Dict[str, Any] | None) -> Dict[str, Any]:
    details = _mapping((web_research or {}).get("details"))
    return _mapping(details.get("source_match"))


def _evidence_for(signals: List[Dict[str, Any]], expected_signal: str) -> List[str]:
    return _unique(
        evidence
        for signal in signals
        if signal.get("signal") == expected_signal
        for evidence in signal.get("evidence", [])
        if isinstance(evidence, str)
    )


def _outcome_label(outcome: str) -> str:
    return {
        "likely_authentic": "Likely authentic photograph",
        "likely_ai_generated_or_manipulated": "Likely AI-generated or manipulated",
        "inconclusive": "Inconclusive / uncertain",
    }.get(outcome, "Inconclusive / uncertain")


def _finite_probability(value: Any) -> float | None:
    number = _finite_number(value)
    if number is None:
        return None
    return max(0.0, min(1.0, number))


def _finite_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unique(values: Any) -> List[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result
