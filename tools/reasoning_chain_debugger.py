"""Audit an explicit claim/evidence graph without exposing private chain-of-thought."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any


def reasoning_chain_debugger(
    conclusion: str,
    steps: list[dict],
    evidence: list[dict] | None = None,
) -> str:
    """Find unsupported claims, dependency gaps, cycles, and confidence problems."""
    evidence = evidence or []
    evidence_map = {str(item.get("id")): item for item in evidence if isinstance(item, dict) and item.get("id")}
    step_map: dict[str, dict] = {}
    issues: list[dict] = []
    evidence_usage: dict[str, int] = defaultdict(int)

    for index, step in enumerate(steps or []):
        if not isinstance(step, dict):
            issues.append({"severity": "error", "step": index, "issue": "Step must be an object"})
            continue
        step_id = str(step.get("id") or f"step-{index + 1}")
        if step_id in step_map:
            issues.append({"severity": "error", "step": step_id, "issue": "Duplicate step id"})
        step_map[step_id] = {**step, "id": step_id}

    dependencies: dict[str, list[str]] = defaultdict(list)
    for step_id, step in step_map.items():
        claim = str(step.get("claim", "")).strip()
        deps = [str(value) for value in step.get("depends_on", [])]
        refs = [str(value) for value in step.get("evidence_ids", [])]
        dependencies[step_id] = deps
        if not claim:
            issues.append({"severity": "error", "step": step_id, "issue": "Missing claim"})
        missing_deps = [dep for dep in deps if dep not in step_map]
        if missing_deps:
            issues.append({"severity": "error", "step": step_id, "issue": "Unknown dependencies", "values": missing_deps})
        missing_refs = [ref for ref in refs if ref not in evidence_map]
        if missing_refs:
            issues.append({"severity": "error", "step": step_id, "issue": "Unknown evidence references", "values": missing_refs})
        if not deps and not refs and not step.get("assumption"):
            issues.append({"severity": "warning", "step": step_id, "issue": "Unsupported root claim; add evidence or mark it as an assumption"})
        try:
            confidence = float(step.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
            issues.append({"severity": "warning", "step": step_id, "issue": "Invalid confidence; treated as 0.5"})
        if confidence > 0.8 and not refs and not deps:
            issues.append({"severity": "warning", "step": step_id, "issue": "High confidence is not supported by evidence or prior steps"})
        for ref in refs:
            evidence_usage[ref] += 1
            item = evidence_map.get(ref, {})
            try:
                quality = float(item.get("quality", 1.0))
            except (TypeError, ValueError):
                quality = 1.0
            if confidence > 0.8 and quality < 0.5:
                issues.append({"severity": "warning", "step": step_id, "issue": "Confidence substantially exceeds cited evidence quality", "evidence": ref})

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str, trail: list[str]) -> None:
        if step_id in visiting:
            issues.append({"severity": "error", "step": step_id, "issue": "Circular dependency", "path": trail + [step_id]})
            return
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency in dependencies[step_id]:
            if dependency in step_map:
                visit(dependency, trail + [step_id])
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in step_map:
        visit(step_id, [])

    conclusion_supported = any(str(step.get("claim", "")).strip().casefold() == conclusion.strip().casefold() for step in step_map.values())
    if conclusion and not conclusion_supported:
        issues.append({"severity": "warning", "step": None, "issue": "No step explicitly establishes the final conclusion"})
    for ref, count in evidence_usage.items():
        if count >= 3 and len(evidence_map) > 1:
            issues.append({"severity": "warning", "step": None, "issue": "A single evidence item carries several claims; check for undue weight", "evidence": ref, "usage_count": count})

    lines = ["flowchart TD"]
    for step_id, step in step_map.items():
        safe_claim = str(step.get("claim", "")).replace('"', "'").replace("\n", " ")[:100]
        lines.append(f'  {step_id.replace("-", "_")}["{step_id}: {safe_claim}"]')
        for dependency in dependencies[step_id]:
            if dependency in step_map:
                lines.append(f'  {dependency.replace("-", "_")} --> {step_id.replace("-", "_")}')

    return json.dumps({
        "conclusion": conclusion,
        "valid": not any(issue["severity"] == "error" for issue in issues),
        "issues": issues,
        "evidence_coverage": {
            "provided": len(evidence_map),
            "referenced": len({str(ref) for step in step_map.values() for ref in step.get("evidence_ids", [])}),
        },
        "mermaid": "\n".join(lines),
        "privacy_note": "This audits the explicit rationale supplied to the tool; it does not reveal hidden model chain-of-thought.",
    }, ensure_ascii=False)
