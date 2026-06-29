from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List

from analyzers.config import get_settings


def build_custom_feedback(
    content_label: str,
    score: int,
    warnings: List[str],
    positives: List[str],
    detectors: List[Dict[str, Any]],
    provenance: Dict[str, Any] | None,
    web_research: Dict[str, Any] | None,
) -> Dict[str, Any]:
    deterministic = _deterministic_feedback(
        content_label=content_label,
        score=score,
        warnings=warnings,
        positives=positives,
        detectors=detectors,
        provenance=provenance,
        web_research=web_research,
    )
    local_feedback = _try_local_reasoning_feedback(deterministic, content_label)
    return local_feedback or deterministic


def _deterministic_feedback(
    content_label: str,
    score: int,
    warnings: List[str],
    positives: List[str],
    detectors: List[Dict[str, Any]],
    provenance: Dict[str, Any] | None,
    web_research: Dict[str, Any] | None,
) -> Dict[str, Any]:
    detector_note = _detector_note(detectors)
    provenance_note = provenance["summary"] if provenance else "No provenance check was available."
    web_note = web_research["summary"] if web_research else "No web research was available."
    if score < 40:
        headline = f"High-risk {content_label} signals found"
    elif score < 60:
        headline = f"Several {content_label} signals need verification"
    elif score < 80:
        headline = f"Mixed {content_label} evidence"
    else:
        headline = f"Mostly reassuring {content_label} signals"

    strongest_risks = warnings[:3] or ["No major warning signal was isolated by the current checks."]
    strongest_positive = positives[:2] or ["No strong positive authenticity signal was available."]
    explanation = (
        f"The score is based on detector signals, provenance checks, indexed web research, and local forensic checks. "
        f"{detector_note} {provenance_note} {web_note}"
    )
    next_steps = [
        "Open the cited sources or run a manual reverse image search if the claim matters.",
        "Look for the earliest uploader, official source, or original context before sharing.",
    ]
    if web_research and web_research.get("status") == "not_configured":
        next_steps.insert(0, "Add a free Brave Search API key to enable automated indexed web research.")
    return {
        "headline": headline,
        "explanation": explanation,
        "evidence_notes": [*strongest_risks, *strongest_positive],
        "next_steps": next_steps[:4],
    }


def _detector_note(detectors: List[Dict[str, Any]]) -> str:
    probabilities = [
        float(detector["synthetic_probability"])
        for detector in detectors
        if isinstance(detector.get("synthetic_probability"), (int, float))
    ]
    if not probabilities:
        return "No AI detector probability was available."
    probability = max(probabilities)
    if probability >= 0.75:
        return f"The strongest AI detector signal estimates about {probability:.0%} synthetic likelihood."
    if probability >= 0.45:
        return f"The strongest AI detector signal is uncertain at about {probability:.0%} synthetic likelihood."
    return f"The strongest AI detector signal estimates about {probability:.0%} synthetic likelihood."


def _try_local_reasoning_feedback(base_feedback: Dict[str, Any], content_label: str) -> Dict[str, Any] | None:
    settings = get_settings()
    if not settings.local_reasoning_base_url:
        return None
    endpoint = f"{settings.local_reasoning_base_url}/v1/chat/completions"
    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write concise media verification feedback. Use only the provided evidence. "
                    "Do not claim certainty. Return JSON with headline, explanation, evidence_notes, next_steps."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"content_label": content_label, "base_feedback": base_feedback}),
            },
        ],
        "temperature": 0.2,
    }
    try:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            parsed = json.loads(response.read().decode("utf-8", errors="replace"))
        content = parsed["choices"][0]["message"]["content"]
        generated = json.loads(content)
        if isinstance(generated, dict) and "headline" in generated and "explanation" in generated:
            return {
                "headline": str(generated.get("headline") or base_feedback["headline"])[:160],
                "explanation": str(generated.get("explanation") or base_feedback["explanation"])[:900],
                "evidence_notes": _string_list(generated.get("evidence_notes"))[:5] or base_feedback["evidence_notes"],
                "next_steps": _string_list(generated.get("next_steps"))[:4] or base_feedback["next_steps"],
            }
    except Exception:
        return None
    return None


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]
