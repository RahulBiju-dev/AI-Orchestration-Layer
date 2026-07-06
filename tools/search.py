"""
tools/search.py — DuckDuckGo web search implementation.

This module provides a tool for the agent to search the web using DuckDuckGo.
It scales result depth based on the requested difficulty level:
  - easy   → 5 results  (quick facts, well-known topics)
  - medium → 8 results  (general questions, moderate research)
  - hard   → 15 results (deep research, niche/complex queries)
"""

import json

# Map difficulty labels to max result counts.
_DIFFICULTY_MAP: dict[str, int] = {
    "easy": 5,
    "medium": 8,
    "hard": 15,
}


def web_search(query: str, difficulty: str = "medium") -> str:
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

    Returns:
        str: A JSON-encoded string containing a list of dictionaries, where each
            dictionary has a 'title' and a 'snippet' representing a search result.
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

        # Extract only the title and body snippet from each result to save space
        condensed = []
        seen_urls = set()
        for result in raw_results:
            url = str(result.get("href") or result.get("url") or "").strip()
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            condensed.append({"title": str(result.get("title") or ""), "url": url, "snippet": str(result.get("body") or "")})

        # Return results as a compact JSON string
        return json.dumps(condensed, ensure_ascii=False, separators=(",", ":"))

    except Exception as exc:
        # Catch network or parsing errors and return them cleanly in JSON
        return json.dumps({"error": str(exc)}, separators=(",", ":"))
