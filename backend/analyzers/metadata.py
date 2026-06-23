from __future__ import annotations

from typing import Any, Dict

from PIL import ExifTags, Image


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
        return 35.0
    if has_camera_make_or_model(exif):
        return 92.0
    return 72.0
