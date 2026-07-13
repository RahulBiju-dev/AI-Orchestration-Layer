"""
tools/search.py — DuckDuckGo web search implementation.

This module provides a tool for the agent to search the web using DuckDuckGo.
It scales result depth based on the requested difficulty level:
  - easy   → 5 results  (quick facts, well-known topics)
  - medium → 8 results  (general questions, moderate research)
  - hard   → 15 results (deep research, niche/complex queries)

When the agent needs richer detail than snippets, it can either call the
dedicated web_scrape tool with a URL, or call web_search with include_content
to scrape the top public results in the same response.
"""

import json
from urllib.parse import urlsplit

from tools.web_scraper import web_scrape

# Map difficulty labels to max result counts.
_DIFFICULTY_MAP: dict[str, int] = {
    "easy": 5,
    "medium": 8,
    "hard": 15,
}


_SCRAPE_DEPTH_MAP: dict[str, int] = {
    "easy": 2,
    "medium": 4,
    "hard": 7,
}


def _safe_json_loads(value: str) -> object:
    try:
        return json.loads(value)
    except Exception:
        return {"error": "tool returned invalid JSON"}


def web_search(
    query: str,
    difficulty: str = "medium",
    include_content: bool = False,
    max_pages: int | None = None,
    max_chars_per_page: int = 6000,
) -> str:
    """
    Execute a DuckDuckGo search with depth scaled by difficulty.

    This function queries DuckDuckGo for the provided search string. The number
    of results returned is controlled by the 'difficulty' parameter, allowing
    the agent to balance between quick, concise answers and deep research.

    Args:
        query (str): The search query string.
        difficulty (str): One of 'easy', 'medium', or 'hard'.
            Controls how many results are fetched (5, 8, or 15 respectively).
            Defaults to 'medium'.
        include_content (bool): When true, scrape readable text from the top
            public results so the agent can answer from page detail rather
            than snippets alone.
        max_pages (int | None): Optional cap for scraped pages. Defaults by
            difficulty: easy=1, medium=2, hard=3. Hard-capped at 5.
        max_chars_per_page (int): Per-page extracted text limit.

    Returns:
        str: A JSON-encoded string containing search result dictionaries, where
            each dictionary has a 'title', 'url', and 'snippet'. When
            include_content is true, top results also include extracted page
            content or a scrape_error.
            If an error occurs, it returns a JSON-encoded dictionary with an 'error' key.
    """
    query = str(query or "").strip()
    if not query:
        return json.dumps({"error": "query is required"}, separators=(",", ":"))
    if len(query) > 1000:
        return json.dumps({"error": "query exceeds the 1000-character limit"}, separators=(",", ":"))
    difficulty_name = str(difficulty or "medium").lower().strip()
    if difficulty_name not in _DIFFICULTY_MAP:
        return json.dumps({"error": "difficulty must be easy, medium, or hard"}, separators=(",", ":"))
    max_results = _DIFFICULTY_MAP[difficulty_name]

    try:
        from ddgs import DDGS
    except ImportError:
        return json.dumps({
            "error": "Web search is unavailable because the optional 'ddgs' package is not installed",
            "error_code": "missing_dependency",
            "dependency": "ddgs",
        }, separators=(",", ":"))

    try:
        # Initialize DuckDuckGo Search client
        with DDGS() as ddgs:
            # Perform text search and limit results
            raw_results = list(ddgs.text(query, max_results=max_results))

        # Extract only the title and body snippet from each result to save space.
        condensed = []
        seen_urls = set()
        skipped_results = 0
        for result in raw_results[:max_results]:
            if not isinstance(result, dict):
                skipped_results += 1
                continue
            url = str(result.get("href") or result.get("url") or "").strip()
            if len(url) > 4096 or any(ord(char) < 32 for char in url):
                skipped_results += 1
                continue
            try:
                parsed_url = urlsplit(url)
                port = parsed_url.port
            except ValueError:
                skipped_results += 1
                continue
            if (
                parsed_url.scheme.lower() not in {"http", "https"}
                or not parsed_url.hostname
                or parsed_url.username is not None
                or parsed_url.password is not None
                or (port is not None and not 1 <= port <= 65535)
            ):
                skipped_results += 1
                continue
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            condensed.append({
                "title": str(result.get("title") or "")[:500],
                "url": url,
                "snippet": str(result.get("body") or "")[:4_000],
            })

        if include_content:
            try:
                page_limit = int(max_pages) if max_pages is not None else _SCRAPE_DEPTH_MAP[difficulty_name]
            except (TypeError, ValueError):
                page_limit = _SCRAPE_DEPTH_MAP[difficulty_name]
            page_limit = max(1, min(5, page_limit))
            try:
                chars_limit = int(max_chars_per_page)
            except (TypeError, ValueError):
                chars_limit = 6000
            chars_limit = max(1000, min(20000, chars_limit))

            scraped_count = 0
            scrape_attempts = 0
            for item in condensed:
                if scrape_attempts >= page_limit:
                    break
                url = item.get("url", "")
                if not url:
                    continue
                scrape_attempts += 1
                scraped = _safe_json_loads(web_scrape(url=url, max_chars=chars_limit, include_links=False))
                if isinstance(scraped, dict) and scraped.get("error"):
                    item["scrape_error"] = scraped["error"]
                elif isinstance(scraped, dict):
                    item["content"] = {
                        "title": scraped.get("title") or item.get("title") or "",
                        "description": scraped.get("description") or "",
                        "headings": scraped.get("headings") or [],
                        "text": scraped.get("text") or "",
                        "truncated": bool(scraped.get("truncated")),
                        "content_type": scraped.get("content_type") or "",
                        "status_code": scraped.get("status_code"),
                    }
                    scraped_count += 1
                else:
                    item["scrape_error"] = "unexpected scrape result"

        # Return results as a compact JSON string
        if skipped_results:
            return json.dumps({
                "results": condensed,
                "skipped_invalid_results": skipped_results,
            }, ensure_ascii=False, separators=(",", ":"))
        return json.dumps(condensed, ensure_ascii=False, separators=(",", ":"))

    except Exception as exc:
        # Catch network or parsing errors and return them cleanly in JSON
        return json.dumps({"error": str(exc)}, separators=(",", ":"))
