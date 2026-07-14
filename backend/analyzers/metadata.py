from __future__ import annotations

from typing import Any, Dict

from PIL import ExifTags, Image


AI_SOFTWARE_MARKERS = {
    "automatic1111": "AUTOMATIC1111",
    "comfyui": "ComfyUI",
    "dall-e": "DALL-E",
    "dalle": "DALL-E",
    "firefly": "Adobe Firefly",
    "fooocus": "Fooocus",
    "invokeai": "InvokeAI",
    "midjourney": "Midjourney",
    "stable diffusion": "Stable Diffusion",
}
EDITING_SOFTWARE_MARKERS = {
    "adobe photoshop": "Adobe Photoshop",
    "adobe lightroom": "Adobe Lightroom",
    "affinity photo": "Affinity Photo",
    "gimp": "GIMP",
    "snapseed": "Snapseed",
}
CAPTURE_FIELD_NAMES = {
    "datetimeoriginal",
    "exposuretime",
    "fnumber",
    "focallength",
    "isospeedratings",
    "lensmake",
    "lensmodel",
    "make",
    "model",
}


def extract_exif_metadata(image: Image.Image) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    try:
        raw_exif = image.getexif()
    except Exception:
        return metadata

    if not raw_exif:
        return metadata

    for tag_id, value in raw_exif.items():
        tag = ExifTags.TAGS.get(tag_id, str(tag_id))
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="replace")
            except Exception:
                value = "<binary>"
        metadata[str(tag)] = value
    return metadata


def has_camera_make_or_model(exif: Dict[str, Any]) -> bool:
    keys = {key.lower() for key in exif.keys()}
    return "make" in keys or "model" in keys


def metadata_score(exif: Dict[str, Any]) -> float:
    if not exif:
        # Missing EXIF is common after downloads, screenshots, social-media
        # processing, and format conversion. It is absence of evidence, so the
        # neutral score is intentional.
        return 50.0
    if has_camera_make_or_model(exif):
        return 92.0
    return 72.0


def analyze_metadata_evidence(exif: Dict[str, Any]) -> Dict[str, Any]:
    """Return privacy-conscious metadata evidence without exposing raw EXIF values."""
    normalized = {str(key).lower(): str(value) for key, value in exif.items()}
    searchable = " ".join(normalized.values()).lower()
    ai_marker = next((label for marker, label in AI_SOFTWARE_MARKERS.items() if marker in searchable), None)
    editing_marker = next((label for marker, label in EDITING_SOFTWARE_MARKERS.items() if marker in searchable), None)
    capture_fields = sorted(
        original_key
        for original_key in exif
        if str(original_key).lower() in CAPTURE_FIELD_NAMES
    )
    camera_present = has_camera_make_or_model(exif)

    if ai_marker:
        status = "explicit_ai_software_tag"
    elif camera_present:
        status = "camera_metadata_present"
    elif exif:
        status = "metadata_present"
    else:
        status = "metadata_absent"

    return {
        "status": status,
        "metadata_present": bool(exif),
        "camera_make_or_model_present": camera_present,
        "capture_fields_present": capture_fields,
        "ai_software_marker": ai_marker,
        "editing_software_marker": editing_marker,
        "field_count": len(exif),
        "limitations": (
            ["Metadata is absent; this is common for downloaded, compressed, converted, or social-media images and is not evidence of AI generation."]
            if not exif
            else ["Metadata can be removed or altered and cannot prove pixel authenticity by itself."]
        ),
    }
