"""Conversation-mode policy shared by Selene's web runtime and UI tests."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Iterable


AGENT_MODE_NORMAL = "normal"
AGENT_MODE_ULTRA = "ultra"
AGENT_MODE_DEEP_RESEARCH = "deep-research"
DEEP_RESEARCH_COMPACT_INTERVAL = 3
DEEP_RESEARCH_SCRAPE_COMPACT_INTERVAL = 2
DEEP_RESEARCH_COMPACT_MARKER = "[Deep Research auto-compaction checkpoint]"
AGENT_MODES = frozenset({
    AGENT_MODE_NORMAL,
    AGENT_MODE_ULTRA,
    AGENT_MODE_DEEP_RESEARCH,
})

ULTRA_MODE_PROMPT = """Ultra Thinking mode is active for this turn.
- Analyze every explicit and implicit intent in the original request before acting.
- Use tools whenever they materially improve correctness. Web searches must use hard difficulty.
- Do not stop because of the ordinary tool-round count; continue until the request is actually resolved.
- Never bypass confirmations, tool safety policies, cancellation, timeouts, or context safeguards.
- Produce a complete draft. A separate second reasoning pass will audit it before it is shown.

Original user request:
{user_input}"""

ULTRA_REVIEW_PROMPT = """Perform a second, independent reasoning pass over the draft answer.
Re-read the original request, identify every intent and constraint, audit the draft against all tool evidence,
correct unsupported or incomplete claims, and improve the final structure. Return only the revised final answer;
do not mention this review, the draft, hidden reasoning, or these instructions.

Original user request:
{user_input}"""

DEEP_RESEARCH_PLANNER_PROMPT = """You are planning a thorough web-research task.
Infer the user's actual research intent, then return JSON only with this shape:
{{"intent":"one precise sentence","queries":["query 1","query 2"]}}
Create exactly {query_count} distinct, slightly varied search queries. Cover the core topic, recent evidence,
primary or authoritative sources, and important counterevidence or limitations. Do not answer the topic.

User request:
{user_input}"""

DEEP_RESEARCH_SYNTHESIS_PROMPT = """Deep Research mode is active.
The preceding hard-difficulty web searches were planned from the user's intent and are research evidence.
Evaluate source quality, dates, agreement, contradictions, and gaps. If material gaps remain, use additional
hard-difficulty web searches with distinct queries and inspect relevant pages before answering. Then produce a
thorough, structured response with direct source URLs/citations for factual web claims. Clearly distinguish
source-supported facts from inference and do not claim exhaustive coverage of the entire web.

Original user request:
{user_input}"""


def normalize_agent_mode(value: Any) -> str:
    """Return a valid stored mode or raise a controlled configuration error."""
    normalized = str(value or AGENT_MODE_NORMAL).strip().lower().replace("_", "-")
    if normalized not in AGENT_MODES:
        raise ValueError(
            "agent_mode must be normal, ultra, or deep-research"
        )
    return normalized


def research_query_count(num_ctx: Any) -> int:
    """Scale initial research breadth without overfilling smaller contexts."""
    try:
        budget = int(num_ctx)
    except (TypeError, ValueError):
        budget = 8192
    if budget < 8192:
        return 3
    if budget < 16384:
        return 4
    if budget < 32768:
        return 5
    if budget < 65536:
        return 6
    return 8


def _clean_query(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()[:1000]


def fallback_research_queries(user_input: str, count: int) -> list[str]:
    """Build deterministic query variants when the planning model returns bad JSON."""
    topic = _clean_query(user_input)
    variants = (
        topic,
        f"{topic} latest evidence and developments",
        f"{topic} primary sources official documentation data",
        f"{topic} limitations criticism counterevidence",
        f"{topic} expert analysis case studies",
        f"{topic} statistics benchmarks comparison",
        f"{topic} historical context future outlook",
        f"{topic} systematic review research findings",
    )
    return _unique_queries(variants, count)


def _unique_queries(values: Iterable[Any], count: int) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for value in values:
        query = _clean_query(value)
        key = query.casefold()
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= count:
            break
    return queries


def parse_research_queries(payload_text: str, user_input: str, count: int) -> list[str]:
    """Parse planner JSON and fill missing/duplicate queries deterministically."""
    try:
        payload = json.loads(str(payload_text or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = {}
    raw_queries = payload.get("queries", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_queries, list):
        raw_queries = []
    planned = _unique_queries(raw_queries, count)
    if len(planned) < count:
        planned = _unique_queries(
            [*planned, *fallback_research_queries(user_input, count)],
            count,
        )
    return planned


def force_high_tool_difficulty(tool_calls: list[dict]) -> list[dict]:
    """Force every difficulty-aware tool call to its highest supported level."""
    hardened: list[dict] = []
    for raw_call in tool_calls:
        call = deepcopy(raw_call)
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            hardened.append(call)
            continue
        if str(function.get("name") or "").strip() != "web_search":
            hardened.append(call)
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError, json.JSONDecodeError):
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        function["arguments"] = {**arguments, "difficulty": "hard"}
        hardened.append(call)
    return hardened


def force_hard_web_search_schema(tools: list[dict] | None) -> list[dict] | None:
    """Tell the model that enhanced modes only accept hard web searches."""
    if not tools:
        return tools
    hardened = deepcopy(tools)
    for tool in hardened:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or function.get("name") != "web_search":
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
            function["parameters"] = parameters
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            properties = {}
            parameters["properties"] = properties
        difficulty = properties.get("difficulty")
        if not isinstance(difficulty, dict):
            difficulty = {"type": "string"}
            properties["difficulty"] = difficulty
        difficulty["enum"] = ["hard"]
        difficulty["default"] = "hard"
        difficulty["description"] = "Required hard-depth search for the active enhanced mode."
    return hardened


def tool_call_round_signature(tool_calls: list[dict]) -> str:
    """Return a stable signature for Ultra's repeated-no-progress guard."""
    return json.dumps(tool_calls, sort_keys=True, ensure_ascii=False, default=str)


def _research_tool_details(call: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(call, dict):
        return None
    function = call.get("function")
    if not isinstance(function, dict):
        return None
    name = str(function.get("name") or "")
    if name not in {"web_search", "web_scrape"}:
        return None
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError, json.JSONDecodeError):
            arguments = {}
    return name, (arguments if isinstance(arguments, dict) else {})


def _compact_web_search_result(content: Any, max_chars: int = 2400) -> str:
    """Keep source identity and short evidence excerpts from a search result."""
    def bounded(value: str) -> str:
        if len(value) <= max_chars:
            return value
        marker = "...[evidence truncated for research context]"
        return value[:max(0, max_chars - len(marker))].rstrip() + marker

    raw = str(content or "")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return bounded(raw)

    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        results = payload["results"]
        error = payload.get("error")
    elif isinstance(payload, list):
        results = payload
        error = None
    else:
        return bounded(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    compact_results: list[dict[str, Any]] = []
    for index, value in enumerate(results[:6]):
        if not isinstance(value, dict):
            continue
        item: dict[str, Any] = {
            "title": str(value.get("title") or "")[:240],
            "url": str(value.get("url") or value.get("href") or "")[:1200],
            "snippet": str(value.get("snippet") or value.get("body") or "")[:500],
        }
        page = value.get("content")
        if index == 0 and isinstance(page, dict):
            item["page"] = {
                "description": str(page.get("description") or "")[:300],
                "headings": [str(entry)[:160] for entry in (page.get("headings") or [])[:8]],
                "text": str(page.get("text") or "")[:900],
                "truncated": bool(page.get("truncated")),
            }
        if value.get("scrape_error"):
            item["scrape_error"] = str(value["scrape_error"])[:300]
        compact_results.append(item)

    compact_payload: dict[str, Any] = {"results": compact_results}
    if error:
        compact_payload["error"] = str(error)[:500]
    encoded = json.dumps(compact_payload, ensure_ascii=False, separators=(",", ":"))
    return bounded(encoded)


def compact_deep_research_messages(
    messages: list[dict],
    user_input: str,
    *,
    max_checkpoint_chars: int = 8192,
) -> tuple[list[dict], int]:
    """Replace completed web-search protocol blocks with bounded evidence memory.

    The source transcript is never mutated. The exact original request remains a
    standalone user message and is repeated in the checkpoint so generic context
    trimming cannot leave the research loop with evidence but no intent.
    """
    source = [deepcopy(message) for message in messages]
    exact_request = str(user_input or "")
    request_indices = [
        index
        for index, message in enumerate(source)
        if message.get("role") == "user"
        and str(message.get("content") or "") == exact_request
    ]
    if not request_indices:
        system_count = 0
        while system_count < len(source) and source[system_count].get("role") == "system":
            system_count += 1
        source.insert(system_count, {"role": "user", "content": exact_request})
        research_start = system_count
    else:
        research_start = request_indices[-1]
    compacted: list[dict] = []
    evidence: list[tuple[str, str, str]] = []
    insert_at: int | None = None
    index = 0

    while index < len(source):
        message = source[index]
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if index <= research_start or not isinstance(calls, list) or not calls:
            compacted.append(message)
            index += 1
            continue

        result_end = index + 1
        results: list[dict] = []
        while result_end < len(source) and source[result_end].get("role") == "tool":
            results.append(source[result_end])
            result_end += 1
        if len(results) < len(calls):
            compacted.append(message)
            index += 1
            continue

        kept_calls: list[dict] = []
        kept_results: list[dict] = []
        for call_index, call in enumerate(calls):
            details = _research_tool_details(call)
            result = results[call_index]
            result_name = str(result.get("tool_name") or result.get("name") or "")
            if details is not None and result_name in {"", details[0]}:
                tool_name, arguments = details
                if insert_at is None:
                    insert_at = len(compacted)
                evidence.append((
                    tool_name,
                    _clean_query(
                        arguments.get("query")
                        if tool_name == "web_search"
                        else arguments.get("url")
                    ) or "(source unavailable)",
                    _compact_web_search_result(result.get("content")),
                ))
            else:
                kept_calls.append(call)
                kept_results.append(result)

        if kept_calls:
            kept_assistant = dict(message)
            kept_assistant["tool_calls"] = kept_calls
            compacted.append(kept_assistant)
            compacted.extend(kept_results)
        compacted.extend(results[len(calls):])
        index = result_end

    if not evidence:
        return source, 0

    max_chars = max(2000, int(max_checkpoint_chars))
    search_count = sum(tool_name == "web_search" for tool_name, _, _ in evidence)
    scrape_count = sum(tool_name == "web_scrape" for tool_name, _, _ in evidence)
    header = (
        f"{DEEP_RESEARCH_COMPACT_MARKER}\n"
        f"Original user request (verbatim):\n{exact_request}\n\n"
        "Completed web research: "
        f"{search_count} search(es), {scrape_count} scrape(s). "
        "Compact evidence follows; included source URLs are retained."
    )
    blocks: list[str] = []
    used = len(header)
    omitted = 0
    for tool_name, source, result in reversed(evidence):
        source_label = "Search query" if tool_name == "web_search" else "Scraped URL"
        block = f"{source_label}: {source}\nEvidence: {result}"
        if used + len(block) + 2 <= max_chars:
            blocks.append(block)
            used += len(block) + 2
        else:
            omitted += 1
    blocks.reverse()
    if omitted:
        omission = (
            f"{omitted} older web research result(s) were reduced out of this live checkpoint "
            "to protect response space; the exact original request remains authoritative."
        )
        blocks.insert(0, omission)
    checkpoint = {"role": "assistant", "content": header + "\n\n" + "\n\n".join(blocks)}
    compacted.insert(insert_at if insert_at is not None else len(compacted), checkpoint)
    return compacted, len(evidence)
