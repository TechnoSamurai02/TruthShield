from __future__ import annotations

import base64
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
GOOGLE_VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"


def research_image_context(
    filename: str,
    visible_text: str = "",
    attachment_fingerprint: Dict[str, Any] | None = None,
    content_bytes: bytes | None = None,
) -> Dict[str, Any]:
    settings = get_settings()
    queries = _image_queries(filename, visible_text, settings.web_research_per_scan_limit)
    if content_bytes and settings.google_vision_api_key:
        google_result = _run_google_vision_web_detection(content_bytes, settings, attachment_fingerprint)
        if google_result["status"] == "completed" or not settings.brave_search_api_key:
            return google_result
        google_details = google_result.get("details")
        google_details = google_details if isinstance(google_details, dict) else {}
        best_guess_labels = [
            str(label)
            for label in google_details.get("best_guess_labels", [])
            if isinstance(label, str) and label.strip()
        ]
        web_entities = google_details.get("web_entities")
        web_entities = web_entities if isinstance(web_entities, list) else []
        entity_descriptions = [
            str(entity.get("description"))
            for entity in web_entities
            if isinstance(entity, dict) and entity.get("description")
        ]
        visual_context = " ".join([*best_guess_labels, *entity_descriptions])
        queries = _image_queries(
            filename,
            f"{visible_text} {visual_context}",
            settings.web_research_per_scan_limit,
        )
        brave_result = _run_research(queries, settings, search_kind="image", attachment_fingerprint=attachment_fingerprint)
        brave_details = brave_result.setdefault("details", {})
        if isinstance(brave_details, dict):
            brave_details["uploaded_image_search"] = google_details
            brave_details["visual_query_clues"] = {
                "best_guess_labels": best_guess_labels,
                "web_entity_descriptions": entity_descriptions,
            }
        if brave_result["matches_found"] > 0:
            brave_result["summary"] = (
                "Uploaded-image search found no full/partial visual matches, but indexed image search found context leads from available visual or text clues."
            )
        return brave_result
    if content_bytes and not settings.google_vision_api_key and not settings.brave_search_api_key:
        return _image_research_not_configured(queries, attachment_fingerprint)
    return _run_research(queries, settings, search_kind="image", attachment_fingerprint=attachment_fingerprint)


def research_text_claims(text: str) -> Dict[str, Any]:
    settings = get_settings()
    queries = _text_queries(text, settings.web_research_per_scan_limit)
    return _run_research(queries, settings, search_kind="web")


def research_video_context(filename: str, frame_notes: Iterable[str]) -> Dict[str, Any]:
    settings = get_settings()
    joined_notes = " ".join(note for note in frame_notes if note)
    queries = _image_queries(filename, joined_notes, settings.web_research_per_scan_limit)
    return _run_research(queries, settings, search_kind="web")


def _image_research_not_configured(
    queries: List[str],
    attachment_fingerprint: Dict[str, Any] | None,
) -> Dict[str, Any]:
    return {
        "status": "not_configured",
        "provider": "google_vision_web_detection+brave_search",
        "score": 50.0,
        "queries": queries,
        "matches_found": 0,
        "summary": "Uploaded-image web matching was skipped because GOOGLE_VISION_API_KEY and BRAVE_SEARCH_API_KEY are not configured.",
        "citations": [],
        "details": _research_details(
            "image",
            attachment_fingerprint,
            source_match=_source_match(
                "not_checked",
                "No uploaded-image matching or indexed image-search provider is configured.",
            ),
            extra={"free_provider": False, "uploaded_image_matching": "not_configured"},
        ),
    }


def _run_google_vision_web_detection(
    content_bytes: bytes,
    settings: EnhancedSettings,
    attachment_fingerprint: Dict[str, Any] | None,
) -> Dict[str, Any]:
    response = _google_vision_post(settings.google_vision_api_key or "", content_bytes, settings.google_vision_max_results)
    if response["status"] != "ok":
        return {
            "status": "error",
            "provider": "google_vision_web_detection",
            "score": 50.0,
            "queries": ["uploaded image web detection"],
            "matches_found": 0,
            "summary": "Uploaded-image web matching could not complete successfully.",
            "citations": [],
            "details": _research_details(
                "image",
                attachment_fingerprint,
                source_match=_source_match("not_checked", "Google Vision Web Detection returned an error."),
                extra={"errors": [response["error"]], "requires_api_key": True},
            ),
        }

    parsed = _parse_google_web_detection(response["data"])
    source_match = parsed["source_match"]
    citations = parsed["citations"]
    status = "completed" if citations else "no_results"
    score = _google_source_score(source_match["status"])
    return {
        "status": status,
        "provider": "google_vision_web_detection",
        "score": score,
        "queries": ["uploaded image web detection"],
        "matches_found": len(citations),
        "summary": _google_web_summary(source_match),
        "citations": citations[:8],
        "details": _research_details(
            "image",
            attachment_fingerprint,
            source_match=source_match,
            extra={
                "uploaded_image_matching": "checked",
                "best_guess_labels": parsed["best_guess_labels"],
                "web_entities": parsed["web_entities"],
                "match_counts": parsed["match_counts"],
                "requires_api_key": True,
            },
        ),
    }


def _google_vision_post(api_key: str, content_bytes: bytes, max_results: int) -> Dict[str, Any]:
    encoded = base64.b64encode(content_bytes).decode("ascii")
    payload = {
        "requests": [
            {
                "image": {"content": encoded},
                "features": [{"type": "WEB_DETECTION", "maxResults": max_results}],
            }
        ]
    }
    request = urllib.request.Request(
        f"{GOOGLE_VISION_ENDPOINT}?key={urllib.parse.quote(api_key)}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "TruthShieldAI/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return {"status": "ok", "data": json.loads(raw)}
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            message = parsed.get("error", {}).get("message") or body
        except Exception:
            message = f"HTTP {exc.code}"
        return {"status": "error", "error": f"Google Vision returned HTTP {exc.code}: {str(message)[:180]}"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:220]}


def _parse_google_web_detection(data: Dict[str, Any]) -> Dict[str, Any]:
    response = ((data.get("responses") or [{}])[0]) if isinstance(data.get("responses"), list) else {}
    web = response.get("webDetection") if isinstance(response, dict) else {}
    web = web if isinstance(web, dict) else {}

    full_images = _google_image_urls(web.get("fullMatchingImages"))
    partial_images = _google_image_urls(web.get("partialMatchingImages"))
    similar_images = _google_image_urls(web.get("visuallySimilarImages"))
    pages = web.get("pagesWithMatchingImages") or []
    pages = pages if isinstance(pages, list) else []

    page_citations: List[Dict[str, Any]] = []
    full_page_count = 0
    partial_page_count = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        url = str(page.get("url") or "")
        if not url:
            continue
        full_page_matches = _google_image_urls(page.get("fullMatchingImages"))
        partial_page_matches = _google_image_urls(page.get("partialMatchingImages"))
        full_page_count += len(full_page_matches)
        partial_page_count += len(partial_page_matches)
        page_citations.append(
            {
                "title": _clean_text(str(page.get("pageTitle") or "Page with matching image"))[:140],
                "url": url,
                "source": _domain(url),
                "snippet": _google_page_snippet(full_page_matches, partial_page_matches),
            }
        )

    citations = [
        *_direct_image_citations(full_images, "Full matching image"),
        *_direct_image_citations(partial_images, "Partial matching image"),
        *page_citations,
        *_direct_image_citations(similar_images[:3], "Visually similar image"),
    ]
    citations = _dedupe_citations(citations)
    best_guess_labels = [
        str(item.get("label"))
        for item in (web.get("bestGuessLabels") or [])
        if isinstance(item, dict) and item.get("label")
    ][:5]
    web_entities = [
        {
            "description": str(item.get("description") or ""),
            "score": float(item.get("score") or 0.0),
        }
        for item in (web.get("webEntities") or [])
        if isinstance(item, dict) and item.get("description")
    ][:8]
    match_counts = {
        "full_matching_images": len(full_images),
        "partial_matching_images": len(partial_images),
        "pages_with_matching_images": len(page_citations),
        "page_full_matching_images": full_page_count,
        "page_partial_matching_images": partial_page_count,
        "visually_similar_images": len(similar_images),
    }
    source_match = _google_source_match(match_counts)
    return {
        "citations": citations,
        "source_match": source_match,
        "best_guess_labels": best_guess_labels,
        "web_entities": web_entities,
        "match_counts": match_counts,
    }


def _run_research(
    queries: List[str],
    settings: EnhancedSettings,
    search_kind: str,
    attachment_fingerprint: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not settings.brave_search_api_key:
        return {
            "status": "not_configured",
            "provider": "brave_search",
            "score": 50.0,
            "queries": queries,
            "matches_found": 0,
            "summary": "Web research was skipped because BRAVE_SEARCH_API_KEY is not configured.",
            "citations": [],
            "details": _research_details(
                search_kind,
                attachment_fingerprint,
                source_match=_source_match("not_checked", "No search provider is configured for attachment matching."),
                extra={"free_provider": True},
            ),
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
            "details": _research_details(
                search_kind,
                attachment_fingerprint,
                source_match=_source_match("not_checked", "The per-scan request limit disabled attachment matching."),
            ),
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
            "details": _research_details(
                search_kind,
                attachment_fingerprint,
                source_match=_source_match("not_checked", "There was not enough searchable text context to look for this attachment."),
            ),
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
                "details": _research_details(
                    search_kind,
                    attachment_fingerprint,
                    source_match=_source_match("not_checked", "The monthly search quota was reached before attachment matching could finish."),
                    extra={"monthly_limit": settings.web_research_monthly_limit, "errors": errors},
                ),
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
        source_match = _build_source_match(citations, attachment_fingerprint)
        score = 82.0 if source_match["status"] == "exact_hash_match" else 68.0
        return {
            "status": "completed",
            "provider": "brave_search",
            "score": score,
            "queries": queries[:consumed],
            "matches_found": len(citations),
            "summary": _web_summary_for_match(source_match),
            "citations": citations,
            "details": _research_details(
                search_kind,
                attachment_fingerprint,
                source_match=source_match,
                extra={"free_provider": True, "errors": errors},
            ),
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
            "details": _research_details(
                search_kind,
                attachment_fingerprint,
                source_match=_source_match("not_checked", "Search errors prevented attachment matching from completing."),
                extra={"errors": errors[:3]},
            ),
        }
    return {
        "status": "no_results",
        "provider": "brave_search",
        "score": 38.0,
        "queries": queries[:consumed],
        "matches_found": 0,
        "summary": "No corroborating indexed web results were found from the generated search queries.",
        "citations": [],
        "details": _research_details(
            search_kind,
            attachment_fingerprint,
            source_match=_source_match(
                "not_found",
                "No indexed source leads were found from the generated queries. This is not proof the attachment is original or real.",
            ),
        ),
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


def _google_image_urls(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    urls = []
    for item in value:
        if isinstance(item, dict) and item.get("url"):
            urls.append(str(item["url"]))
    return urls


def _direct_image_citations(urls: List[str], title: str) -> List[Dict[str, Any]]:
    citations = []
    for url in urls:
        citations.append(
            {
                "title": title,
                "url": url,
                "source": _domain(url),
                "snippet": None,
            }
        )
    return citations


def _google_page_snippet(full_matches: List[str], partial_matches: List[str]) -> str | None:
    if full_matches:
        return "Google Vision reported this page with a full matching image."
    if partial_matches:
        return "Google Vision reported this page with a partial matching image."
    return "Google Vision reported this page as related to the uploaded image."


def _google_source_match(match_counts: Dict[str, int]) -> Dict[str, Any]:
    full_count = match_counts.get("full_matching_images", 0) + match_counts.get("page_full_matching_images", 0)
    partial_count = match_counts.get("partial_matching_images", 0) + match_counts.get("page_partial_matching_images", 0)
    page_count = match_counts.get("pages_with_matching_images", 0)
    similar_count = match_counts.get("visually_similar_images", 0)
    if full_count > 0:
        return _source_match(
            "exact_visual_match",
            "Uploaded-image search found full visual matches for this image online.",
            confidence=0.9,
            matched_citations=full_count + page_count,
        )
    if partial_count > 0 or page_count > 0:
        return _source_match(
            "partial_visual_match",
            "Uploaded-image search found partial matches or pages containing related versions of this image.",
            confidence=0.72,
            matched_citations=partial_count + page_count,
        )
    if similar_count > 0:
        return _source_match(
            "visually_similar_match",
            "Uploaded-image search found visually similar images, but not a confirmed copy of the same image.",
            confidence=0.42,
            matched_citations=similar_count,
        )
    return _source_match(
        "not_found",
        "Uploaded-image search did not find full, partial, or visually similar indexed matches.",
    )


def _google_source_score(status: str) -> float:
    if status == "exact_visual_match":
        return 88.0
    if status == "partial_visual_match":
        return 78.0
    if status == "visually_similar_match":
        return 62.0
    if status == "not_found":
        return 40.0
    return 50.0


def _google_web_summary(source_match: Dict[str, Any]) -> str:
    status = source_match.get("status")
    if status == "exact_visual_match":
        return "Uploaded-image web detection found full visual matches online."
    if status == "partial_visual_match":
        return "Uploaded-image web detection found partial matches or related pages online."
    if status == "visually_similar_match":
        return "Uploaded-image web detection found visually similar images, but no confirmed copy."
    return "Uploaded-image web detection did not find indexed visual matches."


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


def _research_details(
    search_kind: str,
    attachment_fingerprint: Dict[str, Any] | None,
    source_match: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    details: Dict[str, Any] = {"search_kind": search_kind, "source_match": source_match}
    if attachment_fingerprint is not None:
        details["attachment_fingerprint"] = attachment_fingerprint
    if extra:
        details.update(extra)
    return details


def _build_source_match(citations: List[Dict[str, Any]], attachment_fingerprint: Dict[str, Any] | None) -> Dict[str, Any]:
    sha256 = str((attachment_fingerprint or {}).get("sha256") or "")
    if sha256 and _citation_contains(citations, sha256):
        return _source_match(
            "exact_hash_match",
            "An indexed result appears to contain the uploaded file's SHA-256 fingerprint. Treat this as a strong exact-file lead.",
            confidence=0.95,
            matched_citations=len(citations),
        )
    return _source_match(
        "possible_context_match",
        "Indexed search returned possible context or source leads from filename or text cues, but did not compare the uploaded pixels directly.",
        confidence=0.35,
        matched_citations=len(citations),
    )


def _source_match(
    status: str,
    explanation: str,
    confidence: float = 0.0,
    matched_citations: int = 0,
) -> Dict[str, Any]:
    return {
        "status": status,
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
        "matched_citations": matched_citations,
        "explanation": explanation,
    }


def _web_summary_for_match(source_match: Dict[str, Any]) -> str:
    if source_match.get("status") == "exact_hash_match":
        return "Indexed search found a strong exact-file source lead based on the uploaded file fingerprint."
    if source_match.get("status") == "exact_visual_match":
        return "Uploaded-image search found full visual matches online."
    if source_match.get("status") == "partial_visual_match":
        return "Uploaded-image search found partial visual matches or related pages online."
    if source_match.get("status") == "visually_similar_match":
        return "Uploaded-image search found visually similar images online."
    return "Free indexed web/image search found possible context or source leads."


def _citation_contains(citations: List[Dict[str, Any]], needle: str) -> bool:
    lowered_needle = needle.lower()
    for citation in citations:
        haystack = " ".join(
            str(citation.get(field) or "")
            for field in ("title", "url", "source", "snippet")
        ).lower()
        if lowered_needle in haystack:
            return True
    return False


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
