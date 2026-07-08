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

from tools.web_scraper import web_scrape

# Map difficulty labels to max result counts.
_DIFFICULTY_MAP: dict[str, int] = {
    "easy": 5,
    "medium": 8,
    "hard": 15,
}


_SCRAPE_DEPTH_MAP: dict[str, int] = {
    "easy": 1,
    "medium": 2,
    "hard": 3,
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

        # Initialize DuckDuckGo Search client
        with DDGS() as ddgs:
            # Perform text search and limit results
            raw_results = list(ddgs.text(query, max_results=max_results))

        # Extract only the title and body snippet from each result to save space.
        condensed = []
        seen_urls = set()
        for result in raw_results:
            url = str(result.get("href") or result.get("url") or "").strip()
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            condensed.append({"title": str(result.get("title") or ""), "url": url, "snippet": str(result.get("body") or "")})

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
            for item in condensed:
                if scraped_count >= page_limit:
                    break
                url = item.get("url", "")
                if not url:
                    continue
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
        return json.dumps(condensed, ensure_ascii=False, separators=(",", ":"))

    except Exception as exc:
        # Catch network or parsing errors and return them cleanly in JSON
        return json.dumps({"error": str(exc)}, separators=(",", ":"))
