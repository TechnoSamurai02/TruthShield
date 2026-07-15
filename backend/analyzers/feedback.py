from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List

from analyzers.config import get_settings


def build_image_feedback(
    assessment: Dict[str, Any],
    recommendations: List[str],
) -> Dict[str, Any]:
    """Turn image evidence into cautious language for a general audience."""
    verdict = str(assessment.get("verdict") or "inconclusive")
    signals = assessment.get("signals") if isinstance(assessment.get("signals"), list) else []
    reasons_ai: List[str] = []
    reasons_not_ai: List[str] = []
    learned_detector_available = False

    for value in signals:
        if not isinstance(value, dict):
            continue
        source = str(value.get("source") or "")
        signal = str(value.get("signal") or "")
        status = str(value.get("status") or "")
        raw_score = value.get("raw_score")

        if source == "dedicated_detector":
            learned_detector_available = status == "completed" and isinstance(raw_score, (int, float))
            if signal == "ai_generated_or_manipulated":
                reasons_ai.append(
                    "The trained image model found a very strong match with patterns it learned from AI-generated images."
                )
            elif signal == "authentic":
                reasons_not_ai.append(
                    "The trained image model found very little resemblance to the AI-generated images it learned from."
                )
            elif learned_detector_available and float(raw_score) >= 0.5:
                reasons_ai.append(
                    "The image model noticed some AI-like patterns, but the signal was not strong enough to call the image AI-generated."
                )
            elif learned_detector_available:
                reasons_not_ai.append(
                    "The image model leaned away from AI generation, but not strongly enough to call the image authentic."
                )
        elif source == "metadata" and signal == "ai_generated_or_manipulated":
            reasons_ai.append("The file says it was saved by AI-generation software.")
        elif source == "metadata" and signal == "authentic":
            reasons_not_ai.append(
                "The file includes camera make or model information. This supports a camera origin, although metadata can be changed."
            )
        elif source == "pixel_forensics" and signal == "ai_generated_or_manipulated":
            reasons_ai.append(
                "A separate pixel check found unusual patterns sometimes seen in AI images. Editing and compression can cause similar patterns."
            )
        elif source == "provenance" and signal == "authentic":
            reasons_not_ai.append("The file includes verifiable content credentials that help trace where it came from.")
        elif source == "web_context" and signal == "authentic":
            reasons_not_ai.append(
                "A matching or closely related version was found online, which gives the image more real-world context."
            )

    if verdict == "likely_ai_generated_or_manipulated" and reasons_ai:
        headline = "This image may be AI-generated or altered"
        summary = (
            "The strongest available checks point toward AI generation or meaningful digital alteration. "
            "This is a warning, not proof."
        )
        uncertainty_note = (
            "A real photo that was heavily edited, compressed, or saved as a screenshot can sometimes look "
            "AI-made to a detector."
        )
    elif not learned_detector_available:
        summary = (
            "The trained image detector did not return a usable result. The remaining checks are not strong enough "
            "to decide whether this image is AI-generated."
        )
        headline = "We cannot reliably tell"
        uncertainty_note = (
            "Only weaker clues were available. Missing metadata, ordinary editing, compression, or no web match "
            "cannot prove that an image is AI-generated."
        )
    elif verdict == "likely_authentic":
        headline = "This image appears more likely to be a real photograph"
        summary = (
            "The strongest available checks lean away from AI generation, and no separate strong AI clue was found. "
            "This does not guarantee that the image is authentic or unedited."
        )
        uncertainty_note = (
            "New or unfamiliar AI tools can make images the model does not recognize. A likely-authentic result "
            "also cannot prove that the image's caption or story is true."
        )
    else:
        headline = "We cannot tell with enough confidence"
        summary = (
            "The available checks were mixed, missing, or not strong enough to support either an AI-generated "
            "or an authentic result."
        )
        uncertainty_note = (
            "Inconclusive does not mean a 50% chance of AI. It means the system is choosing not to guess from weak "
            "or conflicting clues."
        )

    next_steps = _plain_media_next_steps(recommendations)
    return {
        "headline": headline,
        "explanation": summary,
        "plain_language_summary": summary,
        "reasons_it_might_be_ai": _unique_strings(reasons_ai)[:4],
        "reasons_it_might_not_be_ai": _unique_strings(reasons_not_ai)[:4],
        "uncertainty_note": uncertainty_note,
        "evidence_notes": [*_unique_strings(reasons_ai), *_unique_strings(reasons_not_ai)][:6],
        "next_steps": next_steps,
    }


def build_video_feedback(
    detector_probability: float | None,
    learned_model_available: bool,
    warnings: List[str],
    positives: List[str],
    recommendations: List[str],
) -> Dict[str, Any]:
    """Explain the video estimate without presenting a model score as proof."""
    reasons_ai: List[str] = []
    reasons_not_ai: List[str] = []

    if not learned_model_available or detector_probability is None:
        headline = "No reliable AI verdict for this video"
        summary = (
            "A trained video or frame detector did not return a usable result. Basic file and motion checks ran, "
            "but they are not reliable enough to decide whether this video is AI-generated."
        )
        uncertainty_note = (
            "Fallback motion and pixel checks can react to animation, video editing, low quality, screen recording, "
            "or heavy compression. They should not be used as proof."
        )
    else:
        probability = max(0.0, min(1.0, float(detector_probability)))
        if probability >= 0.90:
            headline = "This video may be AI-generated"
            summary = (
                "The trained video checks found a very strong AI-like pattern across the analyzed frames and motion. "
                "This is a warning, not proof."
            )
            reasons_ai.append(
                "The trained model found a strong, repeated match with patterns learned from AI-generated videos."
            )
        elif probability <= 0.15:
            headline = "This video appears less likely to be AI-generated"
            summary = (
                "The trained video checks found a low AI signal across the analyzed frames and motion. "
                "This does not prove the video is real or unedited."
            )
            reasons_not_ai.append(
                "The trained model found very little resemblance to the AI-generated videos it learned from."
            )
        else:
            headline = "We cannot tell with enough confidence"
            summary = (
                "The trained video checks did not produce a strong enough signal for a reliable AI or non-AI result."
            )
            if probability >= 0.50:
                reasons_ai.append(
                    "Some analyzed frames or motion patterns looked AI-like, but the pattern was not strong enough for a reliable warning."
                )
            else:
                reasons_not_ai.append(
                    "The model leaned away from AI generation, but not strongly enough to call the video camera-recorded."
                )
        uncertainty_note = (
            "Video detectors can be confused by animation, filters, fast motion, editing, screen recordings, and "
            "compression. A result also cannot prove that the video's caption or story is true."
        )

    if any("sustained share" in warning.lower() for warning in warnings):
        reasons_ai.append("AI-like or unusual patterns appeared across several analyzed frames, not only one frame.")
    if any("did not show strong synthetic-video signals" in positive.lower() for positive in positives):
        reasons_not_ai.append("The frame and motion checks did not find a strong repeated AI pattern.")

    next_steps = _plain_media_next_steps(recommendations)
    return {
        "headline": headline,
        "explanation": summary,
        "plain_language_summary": summary,
        "reasons_it_might_be_ai": _unique_strings(reasons_ai)[:4],
        "reasons_it_might_not_be_ai": _unique_strings(reasons_not_ai)[:4],
        "uncertainty_note": uncertainty_note,
        "evidence_notes": [*_unique_strings(reasons_ai), *_unique_strings(reasons_not_ai)][:6],
        "next_steps": next_steps,
    }


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
    should_discuss_attachment_match = content_label != "text post"
    source_match_note = _source_match_note(web_research) if should_discuss_attachment_match else ""
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
        f"{detector_note} {provenance_note} {web_note} {source_match_note}"
    )
    next_steps = [
        "Open the cited sources or run a manual reverse image search if the claim matters.",
        "Look for the earliest uploader, official source, or original context before sharing.",
    ]
    if web_research and web_research.get("status") == "not_configured":
        next_steps.insert(0, "Add a free Brave Search API key to enable automated indexed web research.")
    if should_discuss_attachment_match and _source_match_status(web_research) in {
        "not_checked",
        "possible_context_match",
        "visually_similar_match",
    }:
        next_steps.insert(0, "Use a reverse-image provider such as Google Lens, TinEye, or Bing Visual Search for pixel-level web matching.")
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
        return f"The strongest detector returned an AI-class score of {probability:.0%}; this is not a real-world probability."
    if probability >= 0.45:
        return f"The strongest detector returned an uncertain AI-class score of {probability:.0%}; this is not proof."
    return f"The strongest detector returned an AI-class score of {probability:.0%}; this is not a real-world probability."


def _source_match_note(web_research: Dict[str, Any] | None) -> str:
    status = _source_match_status(web_research)
    if status == "exact_hash_match":
        return "The attachment search found a strong exact-file fingerprint lead."
    if status == "exact_visual_match":
        return "The uploaded-image search found full visual matches online."
    if status == "partial_visual_match":
        return "The uploaded-image search found partial matches or pages containing related versions online."
    if status == "visually_similar_match":
        return "The uploaded-image search found visually similar images, but not a confirmed copy."
    if status == "possible_context_match":
        return "The attachment search found possible context leads, but not a confirmed pixel-level match."
    if status == "not_found":
        return "The attachment search did not find an indexed source match from the available search cues."
    return "No pixel-level reverse-image provider was available for this scan."


def _source_match_status(web_research: Dict[str, Any] | None) -> str:
    details = (web_research or {}).get("details")
    if not isinstance(details, dict):
        return ""
    source_match = details.get("source_match")
    if not isinstance(source_match, dict):
        return ""
    return str(source_match.get("status") or "")


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


def _plain_media_next_steps(recommendations: List[str]) -> List[str]:
    defaults = [
        "If this content affects an important decision, do not rely on this result alone.",
        "Look for the original uploader and check whether a trusted source shows the same content in context.",
        "Use a reverse-image search on the image or on a clear video frame before sharing it.",
    ]
    return _unique_strings([*defaults, *recommendations])[:3]


def _unique_strings(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result
