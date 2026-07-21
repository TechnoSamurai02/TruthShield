from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from analyzers.ai_detectors import combined_synthetic_probability
from analyzers.config import get_settings


DEFAULT_POLICY: Dict[str, Any] = {
    "policy_version": "truthshield-media-policy-v4.0.0",
    "calibration_id": "bootstrap-conservative-v4",
    "generation": {"lower_threshold": 0.15, "upper_threshold": 0.95, "enabled": True},
    "manipulation": {"lower_threshold": 0.15, "upper_threshold": 0.95, "enabled": True},
    "stability": {"max_view_score_range": 0.18, "max_window_score_range": 0.30},
    "quality": {"minimum_short_side": 224, "minimum_video_frames": 8},
    "requirements": {
        "generation_for_decisive_verdict": True,
        "manipulation_for_authentic_verdict": True,
    },
}


def load_media_policy(path: str | None = None) -> Dict[str, Any]:
    configured = path if path is not None else get_settings().media_policy_path
    if configured:
        try:
            value = json.loads(Path(configured).read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return _deep_merge(DEFAULT_POLICY, value)
        except (OSError, ValueError, TypeError):
            pass
    return _deep_merge({}, DEFAULT_POLICY)


def assess_media_evidence(
    detectors: List[Dict[str, Any]],
    technical: Dict[str, Any],
    provenance: Dict[str, Any] | None,
    web_research: Dict[str, Any] | None,
    *,
    media_type: str,
    generation_score: float | None = None,
    manipulation_score: float | None = None,
    transformation_instability: float | None = None,
    localized_or_persistent_manipulation: bool = False,
    generation_specialist_available: bool | None = None,
    manipulation_specialist_available: bool | None = None,
    policy: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply the shared calibrated four-way policy without blending context scores.

    Web matches, metadata absence, compression, and legacy truth scores are reported
    as context only. They never change generation_score or manipulation_score.
    """
    active_policy = _deep_merge(DEFAULT_POLICY, policy or load_media_policy())
    generation_cfg = _mapping(active_policy.get("generation"))
    manipulation_cfg = _mapping(active_policy.get("manipulation"))
    stability_cfg = _mapping(active_policy.get("stability"))

    generation_values = _detector_probabilities(detectors, "synthetic_probability", task="generation")
    screening_values = _detector_probabilities(detectors, "manipulation_probability", task=None)
    dedicated_manipulation_values = _detector_probabilities(
        detectors, "manipulation_probability", task="manipulation"
    )
    generated = _probability(generation_score)
    if generated is None:
        generated = _probability(combined_synthetic_probability(detectors))
    manipulated = _probability(manipulation_score)
    if manipulated is None and dedicated_manipulation_values:
        # Full-frame classifiers can dilute a small edited region. The tiled
        # result is already a robust aggregate over every source pixel, so keep
        # the strongest dedicated score and let calibration control precision.
        manipulated = max(dedicated_manipulation_values)
    if manipulated is None and screening_values:
        manipulated = _mean(screening_values)

    if generation_specialist_available is None:
        generation_specialist_available = bool(generation_values)
    if manipulation_specialist_available is None:
        # A multiclass generation model may provide a useful negative editing
        # screen for authentic decisions, but only a task=manipulation model may
        # issue a positive manipulation verdict.
        manipulation_specialist_available = bool(screening_values)
    dedicated_manipulation_available = bool(dedicated_manipulation_values)

    model_disagreement = _range(generation_values)
    instability = _probability(transformation_instability)
    if instability is None:
        instability = _detail_probability(technical, "transformation_instability")
    instability_limit = float(
        stability_cfg.get("max_window_score_range" if media_type == "video" else "max_view_score_range", 0.18)
    )
    quality_ok, quality_limitations = _quality_status(media_type, technical, active_policy)

    metadata = _mapping(technical.get("metadata_analysis"))
    ai_marker = str(metadata.get("ai_software_marker") or "").strip()
    camera_marker = bool(metadata.get("camera_make_or_model_present"))
    verified_provenance = bool(provenance and provenance.get("status") == "verified")
    specialist_regions = any(
        detector.get("task") == "manipulation"
        and detector.get("status") == "completed"
        and bool(detector.get("suspicious_regions") or _mapping(detector.get("details")).get("suspicious_regions"))
        for detector in detectors
    )
    localized_support = localized_or_persistent_manipulation or specialist_regions

    conflict_reasons: List[str] = []
    if model_disagreement is not None and model_disagreement > 0.35:
        conflict_reasons.append(
            f"Generation specialists disagree substantially (score range {model_disagreement:.3f})."
        )
    if instability is not None and instability > instability_limit:
        conflict_reasons.append(
            f"The detector is unstable across controlled views or time windows (range {instability:.3f})."
        )
    if generated is not None and generated >= float(generation_cfg.get("upper_threshold", 0.95)):
        if camera_marker or verified_provenance:
            conflict_reasons.append(
                "A strong generation score conflicts with camera metadata or verifiable provenance."
            )

    verdict, confidence, reason = _decide(
        generated=generated,
        manipulated=manipulated,
        generation_available=bool(generation_specialist_available),
        manipulation_screen_available=bool(manipulation_specialist_available),
        dedicated_manipulation_available=dedicated_manipulation_available,
        localized_support=localized_support,
        ai_marker=ai_marker,
        quality_ok=quality_ok,
        conflict_reasons=conflict_reasons,
        generation_cfg=generation_cfg,
        manipulation_cfg=manipulation_cfg,
        requirements=_mapping(active_policy.get("requirements")),
    )

    signals = _signals(
        generated=generated,
        manipulated=manipulated,
        generation_available=bool(generation_specialist_available),
        manipulation_available=bool(manipulation_specialist_available),
        dedicated_manipulation_available=dedicated_manipulation_available,
        ai_marker=ai_marker,
        camera_marker=camera_marker,
        verified_provenance=verified_provenance,
        web_research=web_research,
        generation_cfg=generation_cfg,
        manipulation_cfg=manipulation_cfg,
    )
    authenticity_evidence = _signal_evidence(signals, "authentic")
    generation_evidence = _signal_evidence(signals, "ai_generated")
    manipulation_evidence = _signal_evidence(signals, "ai_manipulated")
    limitations = _unique(
        [
            *quality_limitations,
            *(
                []
                if generation_specialist_available
                else ["A required learned generation specialist did not return a usable score."]
            ),
            *(
                []
                if manipulation_specialist_available
                else ["A required manipulation specialist or editing screen did not return a usable score."]
            ),
            *[
                limitation
                for signal in signals
                for limitation in signal.get("limitations", [])
                if isinstance(limitation, str)
            ],
        ]
    )
    model_versions = _unique(
        str(detector.get("model_version") or detector.get("name") or "")
        for detector in detectors
        if detector.get("status") == "completed"
        and str(detector.get("task") or "generation") in {"generation", "manipulation", "temporal"}
    )
    assessment = {
        "verdict": verdict,
        "label": _verdict_label(verdict),
        "confidence": confidence,
        "generation_score": _rounded(generated),
        "manipulation_score": _rounded(manipulated),
        "decision_policy_version": str(active_policy.get("policy_version") or DEFAULT_POLICY["policy_version"]),
        "model_versions": model_versions,
        "detector_score": _rounded(generated),
        "reason": reason,
        "evidence_supporting_authenticity": authenticity_evidence,
        "evidence_supporting_generation": generation_evidence,
        "evidence_supporting_manipulation": manipulation_evidence,
        "evidence_conflicting": conflict_reasons,
        "evidence_raising_concern": _unique([*generation_evidence, *manipulation_evidence]),
        "limitations": limitations,
        "signals": signals,
    }
    debug = {
        "policy_version": assessment["decision_policy_version"],
        "calibration_id": active_policy.get("calibration_id"),
        "generation_score": assessment["generation_score"],
        "manipulation_score": assessment["manipulation_score"],
        "generation_specialist_available": bool(generation_specialist_available),
        "manipulation_screen_available": bool(manipulation_specialist_available),
        "dedicated_manipulation_specialist_available": dedicated_manipulation_available,
        "localized_or_persistent_manipulation": localized_support,
        "model_disagreement": _rounded(model_disagreement),
        "transformation_or_window_instability": _rounded(instability),
        "quality_sufficient": quality_ok,
        "decision_thresholds": {
            "generation_lower": generation_cfg.get("lower_threshold"),
            "generation_upper": generation_cfg.get("upper_threshold"),
            "manipulation_lower": manipulation_cfg.get("lower_threshold"),
            "manipulation_upper": manipulation_cfg.get("upper_threshold"),
            "instability_max": instability_limit,
            "likely_authentic_detector_max": generation_cfg.get("lower_threshold"),
            "likely_ai_detector_min": generation_cfg.get("upper_threshold"),
        },
        "combined_calibrated_score": None,
        "context_not_blended": ["truth_score", "web_research", "missing_metadata", "compression"],
        "final_decision": verdict,
        "reason_for_decision": reason,
    }
    return assessment, debug


def _decide(
    *,
    generated: float | None,
    manipulated: float | None,
    generation_available: bool,
    manipulation_screen_available: bool,
    dedicated_manipulation_available: bool,
    localized_support: bool,
    ai_marker: str,
    quality_ok: bool,
    conflict_reasons: List[str],
    generation_cfg: Dict[str, Any],
    manipulation_cfg: Dict[str, Any],
    requirements: Dict[str, Any],
) -> Tuple[str, str, str]:
    generation_lower = float(generation_cfg.get("lower_threshold", 0.15))
    generation_upper = float(generation_cfg.get("upper_threshold", 0.95))
    manipulation_lower = float(manipulation_cfg.get("lower_threshold", 0.15))
    manipulation_upper = float(manipulation_cfg.get("upper_threshold", 0.95))

    if conflict_reasons:
        return "inconclusive", "moderate", "The available specialist evidence conflicts or is unstable, so the policy abstains."
    if ai_marker and (not generation_available or generated is None or generated < generation_upper):
        return (
            "inconclusive",
            "moderate",
            f"The file names AI-generation software ({ai_marker}), but the calibrated generation score did not independently cross its required threshold.",
        )
    if not quality_ok:
        return "inconclusive", "low", "Media quality or coverage is insufficient for a decisive forensic result."
    if bool(requirements.get("generation_for_decisive_verdict", True)) and (
        not generation_available or generated is None
    ):
        return "inconclusive", "low", "A required learned generation specialist did not complete, so the policy will not guess."
    if not bool(generation_cfg.get("enabled", True)):
        return "inconclusive", "low", "The calibrated generation outcome is disabled because its false-warning limit was not met."

    if generated is not None and generated >= generation_upper:
        return (
            "likely_ai_generated",
            "high",
            "The policy generation score exceeds its conservative upper threshold and the evidence is stable."
            + (" An explicit AI-software tag provides separate supporting provenance." if ai_marker else ""),
        )

    if generated is not None and generated < generation_upper and manipulated is not None and manipulated >= manipulation_upper:
        if not bool(manipulation_cfg.get("enabled", True)):
            return "inconclusive", "low", "The calibrated manipulation outcome is disabled because its false-warning limit was not met."
        if not dedicated_manipulation_available:
            return "inconclusive", "moderate", "An editing screen raised concern, but no dedicated manipulation specialist completed."
        if not localized_support:
            return "inconclusive", "moderate", "The manipulation score is high but lacks localized or temporally persistent support."
        return (
            "likely_ai_manipulated",
            "high",
            "A dedicated manipulation specialist exceeded its policy threshold with localized or persistent support.",
        )

    if generated is not None and generated <= generation_lower:
        if bool(requirements.get("manipulation_for_authentic_verdict", True)) and (
            not manipulation_screen_available or manipulated is None
        ):
            return "inconclusive", "low", "Generation evidence is low, but the required manipulation check did not complete."
        if manipulated is not None and manipulated <= manipulation_lower:
            return (
                "likely_authentic",
                "moderate",
                "Both calibrated specialist scores are below their conservative lower thresholds and no positive AI evidence conflicts.",
            )

    return (
        "inconclusive",
        "low",
        "At least one specialist score is inside an abstention band or generated-versus-manipulated evidence is ambiguous.",
    )


def _signals(
    *,
    generated: float | None,
    manipulated: float | None,
    generation_available: bool,
    manipulation_available: bool,
    dedicated_manipulation_available: bool,
    ai_marker: str,
    camera_marker: bool,
    verified_provenance: bool,
    web_research: Dict[str, Any] | None,
    generation_cfg: Dict[str, Any],
    manipulation_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    generation_signal = "unavailable"
    if generation_available and generated is not None:
        if generated >= float(generation_cfg.get("upper_threshold", 0.95)):
            generation_signal = "ai_generated"
        elif generated <= float(generation_cfg.get("lower_threshold", 0.15)):
            generation_signal = "authentic"
        else:
            generation_signal = "neutral"
    manipulation_signal = "unavailable"
    if manipulation_available and manipulated is not None:
        if dedicated_manipulation_available and manipulated >= float(manipulation_cfg.get("upper_threshold", 0.95)):
            manipulation_signal = "ai_manipulated"
        elif manipulated <= float(manipulation_cfg.get("lower_threshold", 0.15)):
            manipulation_signal = "authentic"
        else:
            manipulation_signal = "neutral"
    metadata_signal = "ai_generated" if ai_marker else "authentic" if camera_marker else "neutral"
    provenance_signal = "authentic" if verified_provenance else "neutral"
    return [
        {
            "source": "generation_specialist",
            "status": "completed" if generation_available and generated is not None else "unavailable",
            "signal": generation_signal,
            "confidence": _score_confidence(generated),
            "reliability": 0.70 if generation_available else 0.0,
            "raw_score": _rounded(generated),
            "evidence": (
                [f"The generation pipeline returned {generated:.3f}."] if generated is not None else []
            ),
            "limitations": ["A classifier score is not a real-world probability or proof."],
        },
        {
            "source": "manipulation_specialist",
            "status": "completed" if manipulation_available and manipulated is not None else "unavailable",
            "signal": manipulation_signal,
            "confidence": _score_confidence(manipulated),
            "reliability": 0.70 if dedicated_manipulation_available else 0.40 if manipulation_available else 0.0,
            "raw_score": _rounded(manipulated),
            "evidence": (
                [f"The manipulation pipeline returned {manipulated:.3f}."] if manipulated is not None else []
            ),
            "limitations": (
                ["The available editing screen can rule out some edits but cannot positively identify AI manipulation."]
                if manipulation_available and not dedicated_manipulation_available
                else []
            ),
        },
        {
            "source": "metadata",
            "status": "completed",
            "signal": metadata_signal,
            "confidence": 0.95 if ai_marker else 0.55 if camera_marker else 0.0,
            "reliability": 0.85 if ai_marker else 0.45 if camera_marker else 0.20,
            "raw_score": None,
            "evidence": (
                [f"Metadata names AI-generation software: {ai_marker}."]
                if ai_marker
                else ["Camera make or model metadata is present."] if camera_marker else []
            ),
            "limitations": [
                "Metadata can be removed, copied, or changed.",
                "Missing or ordinary metadata is not evidence of AI generation.",
            ],
        },
        {
            "source": "provenance",
            "status": "verified" if verified_provenance else "neutral",
            "signal": provenance_signal,
            "confidence": 0.65 if verified_provenance else 0.0,
            "reliability": 0.60 if verified_provenance else 0.0,
            "raw_score": None,
            "evidence": ["Verifiable content credentials were found."] if verified_provenance else [],
            "limitations": ["The absence of content credentials is neutral."],
        },
        {
            "source": "web_context",
            "status": str((web_research or {}).get("status") or "not_checked"),
            "signal": "neutral",
            "confidence": 0.0,
            "reliability": 0.0,
            "raw_score": None,
            "evidence": [],
            "limitations": ["Web matches and non-matches are context only and are not blended into media-generation decisions."],
        },
    ]


def _quality_status(media_type: str, technical: Dict[str, Any], policy: Dict[str, Any]) -> Tuple[bool, List[str]]:
    quality = _mapping(policy.get("quality"))
    if media_type == "video":
        coverage = _mapping(technical.get("analysis_coverage"))
        frames = coverage.get("frames_analyzed", technical.get("frames_analyzed", 0))
        minimum = int(quality.get("minimum_video_frames", 8))
        if isinstance(frames, (int, float)) and int(frames) < minimum:
            return False, [f"Only {int(frames)} frames were analyzed; at least {minimum} are required."]
        return True, []
    width = technical.get("width")
    height = technical.get("height")
    minimum = int(quality.get("minimum_short_side", 224))
    if isinstance(width, (int, float)) and isinstance(height, (int, float)) and min(width, height) < minimum:
        return False, [f"The image short side is below the policy minimum of {minimum} pixels."]
    return True, []


def _detector_probabilities(detectors: Iterable[Dict[str, Any]], field: str, task: str | None) -> List[float]:
    values: List[float] = []
    for detector in detectors:
        if detector.get("status") != "completed":
            continue
        if detector.get("name") == "local_heuristic_synthetic_likelihood":
            continue
        detector_task = str(detector.get("task") or "generation")
        if task is not None and detector_task != task:
            continue
        value = _probability(detector.get(field))
        if value is not None:
            values.append(value)
    return values


def _signal_evidence(signals: Iterable[Dict[str, Any]], signal_name: str) -> List[str]:
    return _unique(
        evidence
        for signal in signals
        if signal.get("signal") == signal_name
        for evidence in signal.get("evidence", [])
        if isinstance(evidence, str)
    )


def _detail_probability(technical: Dict[str, Any], key: str) -> float | None:
    value = technical.get(key)
    if isinstance(value, dict):
        value = value.get("score_range")
    return _probability(value)


def _verdict_label(verdict: str) -> str:
    return {
        "likely_authentic": "Likely authentic",
        "likely_ai_generated": "Likely AI-generated",
        "likely_ai_manipulated": "Likely AI-edited/manipulated",
        "inconclusive": "Inconclusive",
    }.get(verdict, "Inconclusive")


def _score_confidence(value: float | None) -> float:
    return round(abs(value - 0.5) * 2.0, 3) if value is not None else 0.0


def _range(values: List[float]) -> float | None:
    return max(values) - min(values) if len(values) >= 2 else None


def _mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def _rounded(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _probability(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return max(0.0, min(1.0, number))


def _mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _unique(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result
