from __future__ import annotations

import re
from typing import Any, Dict, List

from analyzers.scoring import (
    DISCLAIMER,
    TEXT_RECOMMENDATIONS,
    clamp_score,
    get_risk_level,
    summarize_result,
    unique_messages,
)


SOURCE_PATTERNS = [
    r"https?://",
    r"\baccording to\b",
    r"\breported by\b",
    r"\bsource:\b",
    r"\bReuters\b",
    r"\bAssociated Press\b",
    r"\bAP News\b",
    r"\bBBC\b",
    r"\bNPR\b",
    r"\buniversity\b",
    r"\bpolicy group\b",
    r"\bofficial\b",
]

URGENCY_PHRASES = [
    "share now",
    "breaking",
    "they don't want you to know",
    "they dont want you to know",
    "before it gets deleted",
    "wake up",
    "urgent",
    "click this link now",
    "act now",
]

FEAR_WORDS = [
    "secretly",
    "cover-up",
    "cover up",
    "destroyed",
    "collapse",
    "invasion",
    "takeover",
    "deleted",
    "permanent suspension",
    "exposed",
]

CONSPIRACY_PHRASES = [
    "the government does not want you to know",
    "mainstream media won't report",
    "hidden truth",
    "secret plan",
    "they are hiding",
]

BALANCED_WORDS = ["may", "could", "according", "reported", "study", "researchers", "evidence", "over time"]
PUBLIC_FIGURES_OR_EVENTS = [
    "president",
    "government",
    "election",
    "war",
    "hospitals",
    "new york",
    "white house",
    "congress",
    "supreme court",
]


def analyze_text(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Text cannot be empty.")

    score = 75
    warnings: List[str] = []
    positives: List[str] = []

    lower = cleaned.lower()
    has_source = _matches_any(cleaned, SOURCE_PATTERNS)
    has_urgency = any(phrase in lower for phrase in URGENCY_PHRASES)
    fear_hits = [word for word in FEAR_WORDS if word in lower]
    conspiracy_hits = [phrase for phrase in CONSPIRACY_PHRASES if phrase in lower]
    exclamation_count = cleaned.count("!")
    all_caps_ratio = _all_caps_ratio(cleaned)
    has_verifiable_details = _has_dates_locations_or_details(cleaned)
    has_balanced_wording = any(word in lower for word in BALANCED_WORDS)
    has_public_claim = any(term in lower for term in PUBLIC_FIGURES_OR_EVENTS)

    if has_source:
        score += 5
        positives.append("The text mentions a source, link, or reporting context.")
    else:
        score -= 10
        warnings.append("No source, link, or reporting context was found.")

    if has_balanced_wording:
        score += 5
        positives.append("The wording includes some balanced or cautious language.")

    if has_verifiable_details:
        score += 5
        positives.append("The text includes dates, locations, or details that can be checked.")

    if not has_urgency and not fear_hits and exclamation_count <= 1:
        score += 5
        positives.append("The text avoids strong fear or urgency signals.")

    if all_caps_ratio > 0.18:
        score -= 8
        warnings.append("The text uses heavy all-caps emphasis.")

    if has_urgency:
        score -= 8
        warnings.append("The text uses urgent sharing or action language.")

    if _has_extreme_claim_without_evidence(lower, has_source):
        score -= 12
        warnings.append("The text makes a strong claim without clear evidence.")

    if fear_hits or conspiracy_hits:
        score -= 10
        warnings.append("The text uses fear-based or conspiracy-style language.")

    if exclamation_count >= 3:
        score -= 5
        warnings.append("The text uses excessive exclamation marks.")

    if has_public_claim and not has_source:
        score -= 8
        warnings.append("The text mentions public figures, institutions, or major events without evidence.")

    if len(cleaned.split()) < 10:
        score -= 5
        warnings.append("The claim is very short and lacks context.")

    language_risk = _language_risk_score(all_caps_ratio, exclamation_count, has_urgency, len(fear_hits), len(conspiracy_hits))
    claim_risk = _claim_risk_score(cleaned, has_source, has_public_claim)
    manipulation_risk = _manipulation_score(has_urgency, len(fear_hits), len(conspiracy_hits), exclamation_count)
    final_score = clamp_score(score)
    risk_level, verdict = get_risk_level(final_score)

    technical_details: Dict[str, Any] = {
        "character_count": len(cleaned),
        "word_count": len(cleaned.split()),
        "exclamation_count": exclamation_count,
        "all_caps_ratio": round(all_caps_ratio, 3),
        "source_detected": has_source,
        "urgency_detected": has_urgency,
        "fear_terms_found": fear_hits,
        "conspiracy_phrases_found": conspiracy_hits,
        "heuristic_note": "Text scoring uses explainable language heuristics and does not prove whether a claim is true or false.",
    }

    return {
        "content_type": "text",
        "truth_score": final_score,
        "risk_level": risk_level,
        "verdict": verdict,
        "summary": summarize_result("text post", final_score, warnings, positives),
        "warnings": unique_messages(warnings),
        "positive_signals": unique_messages(positives),
        "recommendations": TEXT_RECOMMENDATIONS,
        "evidence": {
            "source_score": 82.0 if has_source else 25.0,
            "language_risk_score": language_risk,
            "claim_risk_score": claim_risk,
            "manipulation_score": manipulation_risk,
            "overall_risk_score": float(100 - final_score),
        },
        "technical_details": technical_details,
        "disclaimer": DISCLAIMER,
    }


def _matches_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _all_caps_ratio(text: str) -> float:
    words = re.findall(r"\b[A-Za-z]{2,}\b", text)
    if not words:
        return 0.0
    caps = [word for word in words if word.isupper()]
    return len(caps) / len(words)


def _has_dates_locations_or_details(text: str) -> bool:
    date_pattern = r"\b(\d{1,2}/\d{1,2}/\d{2,4}|20\d{2}|19\d{2}|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b"
    location_pattern = r"\b(New York|California|Texas|Florida|United States|U\.S\.|UK|London|Washington|Europe|Asia)\b"
    named_detail_pattern = r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b"
    return bool(
        re.search(date_pattern, text)
        or re.search(location_pattern, text)
        or re.search(named_detail_pattern, text)
    )


def _has_extreme_claim_without_evidence(lower: str, has_source: bool) -> bool:
    extreme_markers = [
        "secretly taken over",
        "guaranteed",
        "everyone is lying",
        "will be deleted",
        "permanent suspension",
        "100%",
        "proof",
        "exposed",
    ]
    return not has_source and any(marker in lower for marker in extreme_markers)


def _language_risk_score(
    all_caps_ratio: float,
    exclamation_count: int,
    has_urgency: bool,
    fear_count: int,
    conspiracy_count: int,
) -> float:
    risk = 15.0
    risk += min(30.0, all_caps_ratio * 100)
    risk += min(20.0, exclamation_count * 4.0)
    risk += 18.0 if has_urgency else 0.0
    risk += min(18.0, fear_count * 6.0)
    risk += min(18.0, conspiracy_count * 9.0)
    return round(max(0.0, min(100.0, risk)), 2)


def _claim_risk_score(text: str, has_source: bool, has_public_claim: bool) -> float:
    risk = 25.0
    if len(text.split()) < 10:
        risk += 15.0
    if not has_source:
        risk += 22.0
    if has_public_claim:
        risk += 16.0
    return round(max(0.0, min(100.0, risk)), 2)


def _manipulation_score(has_urgency: bool, fear_count: int, conspiracy_count: int, exclamation_count: int) -> float:
    risk = 12.0
    risk += 24.0 if has_urgency else 0.0
    risk += min(24.0, fear_count * 8.0)
    risk += min(24.0, conspiracy_count * 12.0)
    risk += min(16.0, exclamation_count * 3.0)
    return round(max(0.0, min(100.0, risk)), 2)
