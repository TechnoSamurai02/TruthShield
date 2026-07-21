"""Backward-compatible import wrapper for the shared v4 media policy."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from analyzers.media_decision import assess_media_evidence

# Deprecated imports retained for existing evaluation scripts. V4 loads these
# values from the versioned calibration artifact at runtime.
AUTHENTIC_DETECTOR_MAX = 0.15
AI_DETECTOR_MIN = 0.95


def assess_image_evidence(
    detectors: List[Dict[str, Any]],
    technical: Dict[str, Any],
    provenance: Dict[str, Any] | None,
    web_research: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return assess_media_evidence(
        detectors,
        technical,
        provenance,
        web_research,
        media_type="image",
    )
