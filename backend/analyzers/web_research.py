from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from analyzers.config import EnhancedSettings, get_settings


BRAVE_WEB_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_IMAGE_ENDPOINT = "https://api.search.brave.com/res/v1/images/search"


def research_image_context(filename: str, visible_text: str = "") -> Dict[str, Any]:
    settings = get_settings()
    queries = _image_queries(filename, visible_text, settings.web_research_per_scan_limit)
    return _run_research(queries, settings, search_kind="image")


def research_text_claims(text: str) -> Dict[str, Any]:
    settings = get_settings()
    queries = _text_queries(text, settings.web_research_per_scan_limit)
    return _run_research(queries, settings, search_kind="web")


def research_video_context(filename: str, frame_notes: Iterable[str]) -> Dict[str, Any]:
    settings = get_settings()
    joined_notes = " ".join(note for note in frame_notes if note)
    queries = _image_queries(filename, joined_notes, settings.web_research_per_scan_limit)
    return _run_research(queries, settings, search_kind="web")


def _run_research(queries: List[str], settings: EnhancedSettings, search_kind: str) -> Dict[str, Any]:
    if not settings.brave_search_api_key:
        return {
            "status": "not_configured",
            "provider": "brave_search",
            "score": 50.0,
            "queries": queries,
            "matches_found": 0,
            "summary": "Web research was skipped because BRAVE_SEARCH_API_KEY is not configured.",
            "citations": [],
            "details": {"free_provider": True, "search_kind": search_kind},
        }
    if settings.web_research_per_scan_limit <= 0:
        return {
            "status": "quota_disabled",
            "provider": "brave_search",
            "score": 50.0,
            "queries": queries,
            "matches_found": 0,
            "summary": "Web research is disabled by the per-scan request limit.",
            "citations": [],
            "details": {"search_kind": search_kind},
        }
    if not queries:
        return {
            "status": "no_query",
            "provider": "brave_search",
            "score": 48.0,
            "queries": [],
            "matches_found": 0,
            "summary": "There was not enough searchable context to run automated web research.",
            "citations": [],
            "details": {"search_kind": search_kind},
        }

    quota = _QuotaCounter(settings.monthly_counter_path, settings.web_research_monthly_limit)
    citations: List[Dict[str, Any]] = []
    errors: List[str] = []
    consumed = 0

    for query in queries[: settings.web_research_per_scan_limit]:
        if not quota.can_consume():
            return {
                "status": "quota_exceeded",
                "provider": "brave_search",
                "score": 50.0,
                "queries": queries[:consumed],
                "matches_found": len(citations),
                "summary": "The free monthly web research limit has been reached.",
                "citations": citations,
                "details": {"monthly_limit": settings.web_research_monthly_limit, "errors": errors},
            }
        quota.consume()
        consumed += 1
        endpoint = BRAVE_IMAGE_ENDPOINT if search_kind == "image" else BRAVE_WEB_ENDPOINT
        response = _brave_get(endpoint, settings.brave_search_api_key, query)
        if response["status"] != "ok":
            errors.append(response["error"])
            continue
        citations.extend(_extract_citations(response["data"], search_kind))

    citations = _dedupe_citations(citations)[:5]
    if citations:
        return {
            "status": "completed",
            "provider": "brave_search",
            "score": 68.0,
            "queries": queries[:consumed],
            "matches_found": len(citations),
            "summary": "Free indexed web/image search found possible context or source leads.",
            "citations": citations,
            "details": {"free_provider": True, "errors": errors, "search_kind": search_kind},
        }
    if errors:
        return {
            "status": "error",
            "provider": "brave_search",
            "score": 50.0,
            "queries": queries[:consumed],
            "matches_found": 0,
            "summary": "Web research could not complete successfully.",
            "citations": [],
            "details": {"errors": errors[:3], "search_kind": search_kind},
        }
    return {
        "status": "no_results",
        "provider": "brave_search",
        "score": 38.0,
        "queries": queries[:consumed],
        "matches_found": 0,
        "summary": "No corroborating indexed web results were found from the generated search queries.",
        "citations": [],
        "details": {"search_kind": search_kind},
    }


def _brave_get(endpoint: str, api_key: str, query: str) -> Dict[str, Any]:
    params = urllib.parse.urlencode({"q": query, "count": 5, "safesearch": "moderate"})
    request = urllib.request.Request(
        f"{endpoint}?{params}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
            "User-Agent": "TruthShieldAI/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = response.read().decode("utf-8", errors="replace")
            return {"status": "ok", "data": json.loads(payload)}
    except urllib.error.HTTPError as exc:
        return {"status": "error", "error": f"Brave Search returned HTTP {exc.code}."}
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:220]}


def _extract_citations(data: Dict[str, Any], search_kind: str) -> List[Dict[str, Any]]:
    if search_kind == "image":
        results = data.get("results") or []
    else:
        results = (data.get("web") or {}).get("results") or []
    citations: List[Dict[str, Any]] = []
    for item in results:
        url = item.get("url") or item.get("properties", {}).get("url")
        title = item.get("title") or item.get("page_title") or "Search result"
        if not url:
            continue
        citations.append(
            {
                "title": _clean_text(str(title))[:140] or "Search result",
                "url": str(url),
                "source": _domain(str(url)),
                "snippet": _clean_text(str(item.get("description") or item.get("page_fetched") or ""))[:260] or None,
            }
        )
    return citations


def _image_queries(filename: str, visible_text: str, limit: int) -> List[str]:
    tokens = _meaningful_tokens(f"{Path(filename).stem} {visible_text}")
    if tokens:
        return [" ".join(tokens[:8])][: max(1, limit)]
    return ["uploaded image source verification"][: max(1, limit)]


def _text_queries(text: str, limit: int) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    candidates = [_clean_text(sentence) for sentence in sentences if len(sentence.split()) >= 4]
    if not candidates:
        candidates = [_clean_text(text)]
    return [candidate[:180] for candidate in candidates[: max(1, limit)] if candidate]


def _meaningful_tokens(text: str) -> List[str]:
    stop_words = {
        "image",
        "photo",
        "picture",
        "upload",
        "uploaded",
        "screenshot",
        "chatgpt",
        "generated",
        "copy",
        "scan",
        "truthshield",
    }
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", text)
    result: List[str] = []
    for word in words:
        lowered = word.lower()
        if lowered in stop_words or lowered in result:
            continue
        result.append(lowered)
    return result


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _domain(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc.removeprefix("www.")


def _dedupe_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for citation in citations:
        url = citation.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(citation)
    return unique


class _QuotaCounter:
    def __init__(self, path: Path, monthly_limit: int) -> None:
        self.path = path
        self.monthly_limit = monthly_limit
        self.month_key = datetime.now(timezone.utc).strftime("%Y-%m")

    def can_consume(self) -> bool:
        if self.monthly_limit <= 0:
            return False
        data = self._read()
        return int(data.get(self.month_key, 0)) < self.monthly_limit

    def consume(self) -> None:
        data = self._read()
        data[self.month_key] = int(data.get(self.month_key, 0)) + 1
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass

    def _read(self) -> Dict[str, int]:
        try:
            raw = self.path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(key): int(value) for key, value in parsed.items()}
        except Exception:
            pass
        return {}
