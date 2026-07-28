"""Deterministic conversation compaction with explicit preservation rules."""

from __future__ import annotations

import json
import re
from typing import Any


MAX_MESSAGES = 10_000
MAX_CRITICAL_TERMS = 100
MAX_TERM_CHARS = 200
MAX_MESSAGE_ROLE_CHARS = 50
MAX_INPUT_CHARS = 20_000_000
MAX_SENTENCE_CHARS = 1_200
MIN_TARGET_TOKENS = 256
MAX_TARGET_TOKENS = 100_000


def _tokens(value: Any) -> int:
    text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    ascii_chars = sum(ord(character) < 128 for character in text)
    return ascii_chars // 4 + (len(text) - ascii_chars) + 1


def _sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        cleaned = re.sub(r"\s+", " ", part).strip()
        while cleaned:
            sentences.append(cleaned[:MAX_SENTENCE_CHARS])
            cleaned = cleaned[MAX_SENTENCE_CHARS:].lstrip()
    return sentences


def _recent_boundary(conversation: list[dict], preserve_count: int) -> int:
    """Do not split an assistant tool call from its contiguous tool results."""
    start = max(0, len(conversation) - preserve_count) if preserve_count else len(conversation)
    if start >= len(conversation) or conversation[start].get("role") != "tool":
        return start
    while start > 0 and conversation[start - 1].get("role") == "tool":
        start -= 1
    if start > 0 and conversation[start - 1].get("role") == "assistant":
        if conversation[start - 1].get("tool_calls"):
            start -= 1
    return start


def _message_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(content)


def _tool_call_summary(message: dict) -> str:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return ""
    compact: list[dict[str, Any]] = []
    for call in calls[:20]:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        compact.append({
            "name": str(function.get("name") or "")[:200],
            "arguments": function.get("arguments", {}),
        })
    return (
        "Tool calls: "
        + json.dumps(compact, ensure_ascii=False, default=str, separators=(",", ":"))
        if compact else ""
    )


def context_memory_optimizer(
    messages: list[dict],
    target_tokens: int = 4000,
    preserve_recent: int = 6,
    critical_terms: list[str] | None = None,
) -> str:
    """Compress messages while retaining recent turns, decisions, facts, and links."""
    if not isinstance(messages, list):
        return json.dumps({"error": "messages must be an array"})
    if len(messages) > MAX_MESSAGES:
        return json.dumps({"error": f"messages exceeds the {MAX_MESSAGES}-item limit"})
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return json.dumps({"error": f"messages[{index}] must be an object"})
        role = message.get("role")
        if not isinstance(role, str) or not role.strip() or len(role) > MAX_MESSAGE_ROLE_CHARS:
            return json.dumps({
                "error": (
                    f"messages[{index}].role must be a non-empty string of at most "
                    f"{MAX_MESSAGE_ROLE_CHARS} characters"
                )
            })
        if role != role.strip():
            return json.dumps({
                "error": f"messages[{index}].role cannot contain leading or trailing whitespace"
            })
    if isinstance(target_tokens, bool) or isinstance(preserve_recent, bool):
        return json.dumps({"error": "target_tokens and preserve_recent must be integers"})
    try:
        target = int(target_tokens)
        preserve_count = int(preserve_recent)
    except (TypeError, ValueError, OverflowError):
        return json.dumps({"error": "target_tokens and preserve_recent must be integers"})
    if not MIN_TARGET_TOKENS <= target <= MAX_TARGET_TOKENS:
        return json.dumps({
            "error": (
                f"target_tokens must be between {MIN_TARGET_TOKENS} and "
                f"{MAX_TARGET_TOKENS}"
            )
        })
    if not 0 <= preserve_count <= 50:
        return json.dumps({"error": "preserve_recent must be between 0 and 50"})
    if critical_terms is not None and not isinstance(critical_terms, list):
        return json.dumps({"error": "critical_terms must be an array"})
    if len(critical_terms or []) > MAX_CRITICAL_TERMS:
        return json.dumps({"error": f"critical_terms exceeds the {MAX_CRITICAL_TERMS}-item limit"})
    terms: list[str] = []
    for index, value in enumerate(critical_terms or []):
        if not isinstance(value, str):
            return json.dumps({"error": f"critical_terms[{index}] must be a string"})
        term = value.strip().casefold()
        if len(term) > MAX_TERM_CHARS:
            return json.dumps({
                "error": f"critical_terms[{index}] exceeds the {MAX_TERM_CHARS}-character limit"
            })
        if term and term not in terms:
            terms.append(term)
    try:
        serialized_input = json.dumps(
            messages, ensure_ascii=False, default=str, separators=(",", ":")
        )
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": f"messages could not be serialized safely: {exc}"})
    if len(serialized_input) > MAX_INPUT_CHARS:
        return json.dumps({
            "error": f"serialized messages exceed the {MAX_INPUT_CHARS}-character limit"
        })

    systems = [message for message in messages if message.get("role") == "system"]
    conversation = [message for message in messages if message.get("role") != "system"]
    recent_start = _recent_boundary(conversation, preserve_count)
    recent = conversation[recent_start:]
    older = conversation[:recent_start]

    seen: set[str] = set()
    candidates: list[tuple[int, int, str, str]] = []
    first_user_seen = False
    markers = (
        "decid", "must", "should", "constraint", "error", "result", "todo",
        "agreed", "because", "http", "file", "path", "source", "warning",
    )
    for message_index, message in enumerate(older):
        role = str(message.get("role", "unknown"))
        content = _message_text(message)
        tool_summary = _tool_call_summary(message)
        for sentence in _sentences("\n".join(value for value in (content, tool_summary) if value)):
            normalized = re.sub(r"\s+", " ", sentence).strip().casefold()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            score = 1
            lowered = sentence.casefold()
            score += sum(2 for marker in markers if marker in lowered)
            score += sum(4 for term in terms if term in lowered)
            if role in {"user", "tool"}:
                score += 1
            if role == "user" and not first_user_seen:
                score += 3
                first_user_seen = True
            if tool_summary and sentence in tool_summary:
                score += 3
            candidates.append((score, message_index, role, sentence))

    base_cost = _tokens(systems + recent)
    selected: list[tuple[int, int, str, str]] = []
    summary_cost = _tokens("[Compacted conversation memory]\n")
    for candidate in sorted(candidates, key=lambda item: (-item[0], item[1])):
        _, _, role, sentence = candidate
        line = f"- [{role}] {sentence}"
        line_cost = _tokens(line) + 1
        if base_cost + summary_cost + line_cost > target:
            continue
        selected.append(candidate)
        summary_cost += line_cost

    chronological = sorted(selected, key=lambda item: item[1])
    summary_lines = [f"- [{role}] {sentence}" for _, _, role, sentence in chronological]
    omitted_candidates = len(candidates) - len(selected)

    optimized = list(systems)
    if summary_lines:
        optimized.append({
            "role": "assistant",
            "content": "[Compacted conversation memory]\n" + "\n".join(summary_lines),
            "metadata": {
                "compacted": True,
                "source_messages": len(older),
                "selected_facts": len(summary_lines),
                "omitted_candidates": omitted_candidates,
            },
        })
    optimized.extend(recent)
    output_tokens = _tokens(optimized)
    return json.dumps({
        "messages": optimized,
        "stats": {
            "input_messages": len(messages),
            "output_messages": len(optimized),
            "estimated_input_tokens": _tokens(messages),
            "estimated_output_tokens": output_tokens,
            "target_tokens": target,
            "target_met": output_tokens <= target,
            "minimum_preserved_tokens": base_cost,
            "preserved_system_messages": len(systems),
            "preserved_recent_messages": len(recent),
            "source_messages_compacted": len(older),
            "selected_facts": len(summary_lines),
            "omitted_candidates": omitted_candidates,
            "deduplicated_sentences": len(seen),
        },
    }, ensure_ascii=False)
