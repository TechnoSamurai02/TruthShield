from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

from fastapi import HTTPException, UploadFile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_VIDEO_BYTES = 80 * 1024 * 1024


def safe_filename(filename: str | None) -> str:
    if not filename:
        return "upload"
    return os.path.basename(filename)


def validate_extension(filename: str | None, allowed_extensions: Iterable[str]) -> str:
    name = safe_filename(filename)
    suffix = Path(name).suffix.lower()
    if suffix not in set(allowed_extensions):
        allowed = ", ".join(sorted(allowed_extensions))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")
    return suffix


async def read_upload_bytes(upload: UploadFile, max_bytes: int) -> bytes:
    data = bytearray()
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail="File is too large for this MVP analyzer.")
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file was empty.")
    return bytes(data)


async def save_upload_to_temp_file(upload: UploadFile, suffix: str, max_bytes: int) -> str:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = temp.name
    size = 0
    try:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(status_code=413, detail="File is too large for this MVP analyzer.")
            temp.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded file was empty.")
        return path
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    finally:
        temp.close()


def remove_temp_file(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass
