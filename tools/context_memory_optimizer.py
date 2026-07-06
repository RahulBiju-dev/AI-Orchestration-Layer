"""Deterministic conversation compaction with explicit preservation rules."""

from __future__ import annotations

import json
import re
from typing import Any


def _tokens(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str)) // 4 + 1


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def context_memory_optimizer(
    messages: list[dict],
    target_tokens: int = 4000,
    preserve_recent: int = 6,
    critical_terms: list[str] | None = None,
) -> str:
    """Compress messages while retaining recent turns, decisions, facts, and links."""
    if not isinstance(messages, list):
        return json.dumps({"error": "messages must be an array"})
    target = max(256, min(int(target_tokens), 100000))
    preserve_count = max(0, min(int(preserve_recent), 50))
    terms = [term.casefold() for term in (critical_terms or []) if str(term).strip()]

    systems = [message for message in messages if message.get("role") == "system"]
    conversation = [message for message in messages if message.get("role") != "system"]
    recent = conversation[-preserve_count:] if preserve_count else []
    older = conversation[:-preserve_count] if preserve_count else conversation

    seen: set[str] = set()
    candidates: list[tuple[int, str, str]] = []
    markers = ("decid", "must", "should", "constraint", "error", "result", "todo", "agreed", "because", "http", "/")
    for message in older:
        role = str(message.get("role", "unknown"))
        for sentence in _sentences(str(message.get("content", ""))):
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
            candidates.append((score, role, sentence[:800]))

    candidates.sort(key=lambda item: -item[0])
    base_cost = _tokens(systems + recent)
    summary_lines: list[str] = []
    for _, role, sentence in candidates:
        line = f"- [{role}] {sentence}"
        if base_cost + _tokens("\n".join(summary_lines + [line])) > target:
            continue
        summary_lines.append(line)

    optimized = list(systems)
    if summary_lines:
        optimized.append({
            "role": "assistant",
            "content": "[Compacted conversation memory]\n" + "\n".join(summary_lines),
            "metadata": {"compacted": True, "source_messages": len(older)},
        })
    optimized.extend(recent)
    return json.dumps({
        "messages": optimized,
        "stats": {
            "input_messages": len(messages),
            "output_messages": len(optimized),
            "estimated_input_tokens": _tokens(messages),
            "estimated_output_tokens": _tokens(optimized),
            "deduplicated_sentences": len(seen),
        },
    }, ensure_ascii=False)
