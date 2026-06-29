from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict


def verify_image_provenance(data: bytes | None, filename: str) -> Dict[str, Any]:
    if not data:
        return {
            "status": "unavailable",
            "score": 50.0,
            "summary": "No original file bytes were available for content credential checks.",
            "details": {},
        }

    c2pa_result = _try_c2pa_python(data)
    if c2pa_result is not None:
        return c2pa_result

    tool_path = shutil.which("c2patool")
    if not tool_path:
        return {
            "status": "tool_unavailable",
            "score": 45.0,
            "summary": "No C2PA verifier is installed, so content credentials could not be checked.",
            "details": {"checked_with": "c2pa-python or c2patool", "filename": filename},
        }

    suffix = Path(filename).suffix or ".image"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(data)
            temp_path = temp_file.name

        completed = subprocess.run(
            [tool_path, temp_path, "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=12,
        )
        output = completed.stdout.strip()
        parsed = _safe_json_loads(output)
        if completed.returncode == 0 and parsed:
            manifest_count = _count_manifest_like_items(parsed)
            if manifest_count > 0:
                return {
                    "status": "verified",
                    "score": 88.0,
                    "summary": "C2PA content credentials were found and parsed.",
                    "details": {
                        "checked_with": "c2patool",
                        "manifest_count": manifest_count,
                        "metadata_keys": sorted(parsed.keys())[:12] if isinstance(parsed, dict) else [],
                    },
                }
            return {
                "status": "no_manifest",
                "score": 42.0,
                "summary": "The file was checked, but no C2PA content credentials were found.",
                "details": {"checked_with": "c2patool"},
            }
        return {
            "status": "no_manifest",
            "score": 42.0,
            "summary": "The file did not expose verifiable C2PA content credentials.",
            "details": {
                "checked_with": "c2patool",
                "return_code": completed.returncode,
                "stderr": completed.stderr.strip()[:300],
            },
        }
    except Exception as exc:
        return {
            "status": "error",
            "score": 45.0,
            "summary": "The C2PA verification step could not complete safely.",
            "details": {"error": str(exc)[:300], "checked_with": "c2patool"},
        }
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _try_c2pa_python(data: bytes) -> Dict[str, Any] | None:
    try:
        import c2pa  # type: ignore
    except Exception:
        return None

    try:
        reader = getattr(c2pa, "Reader", None)
        if reader is None:
            return {
                "status": "tool_unavailable",
                "score": 45.0,
                "summary": "The c2pa-python package is present, but its reader API was not available.",
                "details": {"checked_with": "c2pa-python"},
            }
        result = reader.from_bytes(data) if hasattr(reader, "from_bytes") else None
        manifest = getattr(result, "manifest_store", None) or getattr(result, "manifest", None)
        if manifest:
            return {
                "status": "verified",
                "score": 88.0,
                "summary": "C2PA content credentials were found and parsed.",
                "details": {"checked_with": "c2pa-python"},
            }
        return {
            "status": "no_manifest",
            "score": 42.0,
            "summary": "The file was checked, but no C2PA content credentials were found.",
            "details": {"checked_with": "c2pa-python"},
        }
    except Exception:
        return None


def _safe_json_loads(raw: str) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _count_manifest_like_items(value: Any) -> int:
    if isinstance(value, dict):
        count = 0
        for key, nested in value.items():
            lowered = str(key).lower()
            if "manifest" in lowered or "claim" in lowered:
                count += 1
            count += _count_manifest_like_items(nested)
        return count
    if isinstance(value, list):
        return sum(_count_manifest_like_items(item) for item in value)
    return 0
