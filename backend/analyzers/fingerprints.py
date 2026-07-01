from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable

import cv2
import numpy as np
from PIL import Image


def build_image_fingerprint(image: Image.Image, content_bytes: bytes | None, filename: str) -> Dict[str, Any]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    fingerprint: Dict[str, Any] = {
        "kind": "image",
        "filename": filename,
        "dimensions": {"width": width, "height": height},
        "perceptual_hashes": {
            "average_hash": _average_hash(rgb),
            "difference_hash": _difference_hash(rgb),
            "phash": _perceptual_hash(rgb),
        },
        "notes": [
            "SHA-256 identifies this exact file byte-for-byte.",
            "Perceptual hashes help compare visually similar images, but require a database or reverse-search provider to match against.",
        ],
    }
    if content_bytes is not None:
        fingerprint["sha256"] = hashlib.sha256(content_bytes).hexdigest()
        fingerprint["file_size_bytes"] = len(content_bytes)
    return fingerprint


def _average_hash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float32)
    average = float(pixels.mean())
    return _bits_to_hex(pixels.flatten() >= average)


def _difference_hash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float32)
    differences = pixels[:, 1:] > pixels[:, :-1]
    return _bits_to_hex(differences.flatten())


def _perceptual_hash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float32)
    dct = cv2.dct(pixels)
    low_frequency = dct[:8, :8].flatten()
    median = float(np.median(low_frequency[1:])) if low_frequency.size > 1 else float(low_frequency.mean())
    return _bits_to_hex(low_frequency >= median)


def _bits_to_hex(bits: Iterable[bool]) -> str:
    bit_string = "".join("1" if bit else "0" for bit in bits)
    if not bit_string:
        return ""
    width = max(1, len(bit_string) // 4)
    return f"{int(bit_string, 2):0{width}x}"
