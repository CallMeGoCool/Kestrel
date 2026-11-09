"""
Kestrel Shorts - Search Client
==============================

A thin client for the shared search service (ResearchAssistant/search_service.py).
It does not search by itself and holds no API keys; the service does.

    from core import search
    outcome = search.search("antikythera mechanism", cache_dir=ctx.run_directory / "search")
    if outcome.status == "unavailable": ...     # service down and could not be started
    text, seen_urls = search.format_evidence(outcome.results)

Outcomes: success | no_results | failed | unavailable.
If the service is not running, the client starts it (config.SEARCH_AUTOSTART).
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import requests

from core import config

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 90   # the service tries providers one after another


@dataclass
class SearchOutcome:
    query: str
    status: str                      # success | no_results | failed | unavailable
    results: List[Dict[str, str]] = field(default_factory=list)   # {title, url, snippet}
    error: Optional[str] = None
    cached: bool = False
    searched_at: str = ""          # local time the search ran

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Service management
# ----------------------------------------------------------------------

def _service_url() -> str:
    return str(getattr(config, "SEARCH_SERVICE_URL", "http://127.0.0.1:8000/search"))


def _base_url() -> str:
    parts = urlsplit(_service_url())
    return f"{parts.scheme}://{parts.netloc}"


def is_service_up() -> bool:
    try:
        return requests.get(_base_url() + "/openapi.json", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def ensure_service(auto_start: Optional[bool] = None, wait_seconds: float = 25.0) -> bool:
    """Make sure the search service is reachable, starting it if allowed."""
    if is_service_up():
        return True
    if auto_start is None:
        auto_start = bool(getattr(config, "SEARCH_AUTOSTART", True))
    script = Path(getattr(config, "SEARCH_SERVICE_SCRIPT", ""))
    if not auto_start or not script.is_file():
        return False

    logger.info("Starting search service: %s", script)
    kwargs: Dict[str, Any] = {
        "cwd": str(script.parent),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        subprocess.Popen([sys.executable, str(script)], **kwargs)
    except OSError as exc:
        logger.warning("Could not start the search service: %s", exc)
        return False

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if is_service_up():
            return True
        time.sleep(0.5)
    return False


# ----------------------------------------------------------------------
# Searching
# ----------------------------------------------------------------------

def _normalize(payload: Any, limit: int) -> List[Dict[str, str]]:
    items = payload.get("results", []) if isinstance(payload, dict) else payload
    results: List[Dict[str, str]] = []
    seen: set = set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("link") or item.get("url") or item.get("href") or "").strip()
        snippet = str(item.get("snippet") or item.get("content") or item.get("description") or "").strip()
        title = str(item.get("title") or "").strip()
        if url and url not in seen and (title or snippet):
            seen.add(url)
            results.append({"title": title or url, "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def _cache_file(cache_dir: Path, query: str, limit: int) -> Path:
    key = hashlib.sha1(f"{query.casefold()}|{limit}".encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{key}.json"


def search(
    query: str,
    max_results: Optional[int] = None,
    *,
    cache_dir: Optional[Path | str] = None,
    auto_start: Optional[bool] = None,
) -> SearchOutcome:
    """
    Run one search. Never raises. With cache_dir, a repeated query in the
    same run is answered from disk, so retries do not repeat searches.
    """
    query = query.strip() if isinstance(query, str) else ""
    if not query:
        return SearchOutcome("", "failed", error="Search query cannot be empty.")

    limit = int(max_results or getattr(config, "SEARCH_RESULTS_PER_QUERY", 5))

    if cache_dir is not None:
        path = _cache_file(Path(cache_dir), query, limit)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return SearchOutcome(query, data["status"], data["results"], cached=True,
                                     searched_at=data.get("searched_at", ""))
            except (OSError, ValueError, KeyError):
                pass

    def post() -> requests.Response:
        return requests.post(
            _service_url(), json={"query": query, "count": limit}, timeout=REQUEST_TIMEOUT
        )

    try:
        try:
            response = post()
        except requests.ConnectionError:
            if not ensure_service(auto_start):
                return SearchOutcome(
                    query, "unavailable",
                    error=f"Search service not reachable at {_service_url()} and could not be started.",
                )
            response = post()
        response.raise_for_status()
        results = _normalize(response.json(), limit)
    except Exception as exc:
        logger.warning("Search failed for %r: %s", query, exc)
        return SearchOutcome(query, "failed", error=str(exc))

    outcome = SearchOutcome(query, "success" if results else "no_results", results,
                            searched_at=config.now_stamp()["iso"])

    if cache_dir is not None:
        try:
            path = _cache_file(Path(cache_dir), query, limit)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(outcome.to_dict(), ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not cache search results: %s", exc)
    return outcome


# ----------------------------------------------------------------------
# Evidence helpers
# ----------------------------------------------------------------------

def merge_results(
    existing: List[Dict[str, str]], incoming: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Append new results, skipping URLs already present."""
    merged = list(existing)
    seen = {r["url"] for r in merged}
    for item in incoming:
        if item["url"] not in seen:
            merged.append(item)
            seen.add(item["url"])
    return merged


def format_evidence(
    results: List[Dict[str, str]],
    per_source_chars: int = 700,
    total_chars: int = 6000,
) -> Tuple[str, List[str]]:
    """
    Number the sources and cap their size so later evidence is never cut
    off by earlier evidence. Returns (text, urls_included); only the URLs
    actually shown to the model should be recorded as its sources.
    """
    blocks: List[str] = []
    included: List[str] = []
    used = 0
    for item in results:
        domain = urlsplit(item["url"]).netloc.removeprefix("www.")
        snippet = item["snippet"][:per_source_chars].strip()
        block = f"[{len(included) + 1}] {item['title']} ({domain})\n{snippet}"
        if used + len(block) > total_chars:
            break
        blocks.append(block)
        included.append(item["url"])
        used += len(block) + 2
    return "\n\n".join(blocks), included


if __name__ == "__main__":
    print("Search service up:", is_service_up())
    outcome = search("antikythera mechanism")
    print(outcome.status, len(outcome.results), outcome.error or "")
    if outcome.results:
        print(outcome.results[0]["title"])