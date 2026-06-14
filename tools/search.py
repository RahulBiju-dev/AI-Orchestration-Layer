"""
tools/search.py — DuckDuckGo web search implementation.

Scales result depth by difficulty:
  easy   → 3 results  (quick facts, well-known topics)
  medium → 6 results  (general questions, moderate research)
  hard   → 10 results (deep research, niche/complex queries)
"""

import json
from ddgs import DDGS

# Map difficulty labels to max result counts.
_DIFFICULTY_MAP: dict[str, int] = {
    "easy": 5,
    "medium": 8,
    "hard": 15,
}


def web_search(query: str, difficulty: str = "medium") -> str:
    """Execute a DuckDuckGo search with depth scaled by difficulty.

    Args:
        query:      The search query string.
        difficulty: One of 'easy', 'medium', or 'hard'.
                    Controls how many results are fetched (4 / 7 / 15).

    Returns:
        A JSON string containing a list of {title, snippet} dicts,
        or an error payload if the search fails.
    """
    max_results = _DIFFICULTY_MAP.get(difficulty.lower().strip(), 5)

    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))

        condensed = [
            {"title": r.get("title", ""), "snippet": r.get("body", "")}
            for r in raw_results
        ]

        return json.dumps(condensed, separators=(",", ":"))

    except Exception as exc:
        return json.dumps({"error": str(exc)}, separators=(",", ":"))
