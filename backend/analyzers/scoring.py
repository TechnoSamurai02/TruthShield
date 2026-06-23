from __future__ import annotations

from typing import Iterable, List, Tuple


DISCLAIMER = (
    "TruthShield AI provides a risk-based analysis, not a final proof that content "
    "is real or fake. Use this tool as a first check and verify important claims "
    "with trusted sources."
)

IMAGE_VIDEO_RECOMMENDATIONS = [
    "Check whether the content appears on trusted news or official sources.",
    "Look for the original uploader or source.",
    "Do a reverse image search.",
    "Be cautious if the content triggers strong anger, fear, or urgency.",
    "Do not share until verified by another reliable source.",
]

TEXT_RECOMMENDATIONS = [
    "Search the claim using trusted sources.",
    "Check whether multiple reliable outlets report the same thing.",
    "Look for dates, names, and original evidence.",
    "Avoid sharing posts that rely only on fear or urgency.",
]


def clamp_score(score: float) -> int:
    return int(max(0, min(100, round(score))))


def get_risk_level(score: int) -> Tuple[str, str]:
    if score >= 80:
        return "High Trust", "Likely trustworthy"
    if score >= 60:
        return "Medium Trust", "Needs light verification"
    if score >= 40:
        return "Low Trust", "Suspicious / verify before sharing"
    return "High Risk", "Do not share until verified"


def unique_messages(messages: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for message in messages:
        cleaned = message.strip()
        if cleaned and cleaned not in seen:
            result.append(cleaned)
            seen.add(cleaned)
    return result


def summarize_result(content_label: str, score: int, warnings: List[str], positives: List[str]) -> str:
    if score >= 80:
        return (
            f"TruthShield AI found more reassuring signals than risk signals in this "
            f"{content_label}. This does not prove authenticity, but it appears lower risk."
        )
    if score >= 60:
        return (
            f"TruthShield AI found mixed signals in this {content_label}. It may be "
            "reasonable, but important claims should still be checked before sharing."
        )
    if score >= 40:
        return (
            f"TruthShield AI found several risk signals in this {content_label}. The "
            "content should be verified with reliable sources before sharing."
        )
    return (
        f"TruthShield AI found strong risk signals in this {content_label}. This result "
        "does not prove the content is fake, but it should not be shared until verified."
    )
