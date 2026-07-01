"""Build and inspect small semantic knowledge graphs without external services."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any


CAUSAL_RELATIONS = {"causes", "enables", "increases", "decreases", "prevents", "mitigates"}
NEGATIVE_RELATIONS = {"decreases", "prevents", "mitigates", "contradicts", "inhibits"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def knowledge_graph_builder(
    concepts: list[dict],
    relationships: list[dict],
    query: dict | None = None,
    max_depth: int = 4,
) -> str:
    """Validate a semantic graph and infer explainable multi-hop connections.

    Every inferred connection includes the exact relationship path that supports it;
    the function never invents edges from concept names alone.
    """
    if not isinstance(concepts, list) or not isinstance(relationships, list):
        return _json({"error": "concepts and relationships must be arrays"})
    if len(concepts) > 500 or len(relationships) > 3000:
        return _json({"error": "Graph exceeds the 500-concept/3000-relationship safety limit"})

    nodes: dict[str, dict] = {}
    errors: list[str] = []
    for index, concept in enumerate(concepts):
        if not isinstance(concept, dict):
            errors.append(f"concepts[{index}] must be an object")
            continue
        node_id = str(concept.get("id", "")).strip()
        if not node_id:
            errors.append(f"concepts[{index}] is missing id")
        elif node_id in nodes:
            errors.append(f"Duplicate concept id: {node_id}")
        else:
            nodes[node_id] = {
                "id": node_id,
                "label": str(concept.get("label") or node_id),
                "attributes": concept.get("attributes", {}),
            }

    edges: list[dict] = []
    adjacency: dict[str, list[dict]] = defaultdict(list)
    reverse: dict[str, list[dict]] = defaultdict(list)
    signatures: dict[tuple[str, str], set[str]] = defaultdict(set)
    for index, relation in enumerate(relationships):
        if not isinstance(relation, dict):
            errors.append(f"relationships[{index}] must be an object")
            continue
        source = str(relation.get("source", "")).strip()
        target = str(relation.get("target", "")).strip()
        relation_type = str(relation.get("type", "related_to")).strip().lower()
        if source not in nodes or target not in nodes:
            errors.append(f"relationships[{index}] references an unknown concept")
            continue
        try:
            weight = max(0.0, min(1.0, float(relation.get("weight", 1.0))))
        except (TypeError, ValueError):
            weight = 1.0
        edge = {
            "id": str(relation.get("id") or f"r{index + 1}"),
            "source": source,
            "target": target,
            "type": relation_type,
            "weight": weight,
            "evidence": relation.get("evidence", []),
        }
        edges.append(edge)
        adjacency[source].append(edge)
        reverse[target].append(edge)
        signatures[(source, target)].add(relation_type)

    if errors:
        return _json({"error": "Invalid graph", "details": errors})

    depth_limit = max(1, min(int(max_depth), 8))
    start = str((query or {}).get("source", "")).strip() or None
    goal = str((query or {}).get("target", "")).strip() or None
    allowed_types = set((query or {}).get("relation_types") or CAUSAL_RELATIONS)
    if start and start not in nodes:
        return _json({"error": f"Unknown query source: {start}"})
    if goal and goal not in nodes:
        return _json({"error": f"Unknown query target: {goal}"})

    paths: list[dict] = []
    origins = [start] if start else list(nodes)
    for origin in origins:
        queue = deque([(origin, [], {origin})])
        while queue and len(paths) < 250:
            current, path, visited = queue.popleft()
            if len(path) >= depth_limit:
                continue
            for edge in adjacency[current]:
                if edge["type"] not in allowed_types or edge["target"] in visited:
                    continue
                new_path = path + [edge]
                destination = edge["target"]
                if len(new_path) >= 2 and (goal is None or destination == goal):
                    sign = -1 if sum(e["type"] in NEGATIVE_RELATIONS for e in new_path) % 2 else 1
                    paths.append({
                        "source": origin,
                        "target": destination,
                        "inferred_effect": "negative" if sign < 0 else "positive",
                        "confidence": round(min(e["weight"] for e in new_path), 4),
                        "path": [{"edge": e["id"], "from": e["source"], "type": e["type"], "to": e["target"]} for e in new_path],
                    })
                queue.append((destination, new_path, visited | {destination}))

    contradictions = []
    for (source, target), types in signatures.items():
        positive = types - NEGATIVE_RELATIONS
        negative = types & NEGATIVE_RELATIONS
        if positive and negative:
            contradictions.append({"source": source, "target": target, "positive": sorted(positive), "negative": sorted(negative)})

    cycles: list[list[str]] = []
    active: list[str] = []
    completed: set[str] = set()

    def find_cycles(node_id: str) -> None:
        if node_id in active:
            cycle = active[active.index(node_id):] + [node_id]
            signature = min(tuple(cycle[index:-1] + cycle[:index] + [cycle[index]]) for index in range(len(cycle) - 1))
            if list(signature) not in cycles:
                cycles.append(list(signature))
            return
        if node_id in completed:
            return
        active.append(node_id)
        for outgoing in adjacency[node_id]:
            find_cycles(outgoing["target"])
        active.pop()
        completed.add(node_id)

    for node_id in nodes:
        find_cycles(node_id)

    centrality = sorted(
        ({"concept": node_id, "degree": len(adjacency[node_id]) + len(reverse[node_id])} for node_id in nodes),
        key=lambda item: (-item["degree"], item["concept"]),
    )
    return _json({
        "graph": {"concepts": list(nodes.values()), "relationships": edges},
        "analysis": {
            "inferred_paths": paths,
            "contradictions": contradictions,
            "potential_feedback_cycles": cycles,
            "central_concepts": centrality[:10],
        },
        "limits": {"max_depth": depth_limit, "path_limit": 250},
    })
