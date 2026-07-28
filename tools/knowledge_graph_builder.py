"""Build and inspect small semantic knowledge graphs without external services."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from typing import Any

from agent.cancellation import CancellationToken


POSITIVE_RELATIONS = {"causes", "enables", "increases", "supports", "reinforces"}
NEGATIVE_RELATIONS = {"decreases", "prevents", "mitigates", "contradicts", "inhibits"}
MAX_INFERRED_PATHS = 100
MAX_TRAVERSAL_STATES = 20_000
MAX_IDENTIFIER_CHARS = 200
MAX_LABEL_CHARS = 2_000
MAX_INPUT_JSON_CHARS = 500_000
MAX_GRAPH_ECHO_CHARS = 20_000
MAX_FEEDBACK_COMPONENTS = 100
MAX_FEEDBACK_OUTPUT_CHARS = 10_000
MAX_PATH_OUTPUT_CHARS = 30_000
MAX_DIRECT_MATCHES = 100
MAX_DIRECT_OUTPUT_CHARS = 10_000
MAX_CONTRADICTIONS = 100
MAX_CONTRADICTION_OUTPUT_CHARS = 10_000


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _serialized_size(value: Any) -> int:
    try:
        return len(json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(
            "Graph inputs must contain only finite JSON-compatible values"
        ) from exc


def _bounded_items(values: list[dict], max_chars: int) -> tuple[list[dict], int]:
    selected = []
    used = 2
    for value in values:
        size = _serialized_size(value) + (1 if selected else 0)
        if used + size > max_chars:
            break
        selected.append(value)
        used += size
    return selected, len(values) - len(selected)


def _representative_cycle(
    component: set[str],
    adjacency: dict[str, list[dict]],
) -> list[str] | None:
    """Return one edge-backed cycle for a strongly connected component."""
    start = min(component)
    for outgoing in adjacency[start]:
        neighbor = outgoing["target"]
        if neighbor not in component:
            continue
        if neighbor == start:
            return [start, start]
        queue = deque([(neighbor, [start, neighbor])])
        visited = {start, neighbor}
        while queue:
            current, path = queue.popleft()
            for edge in adjacency[current]:
                target = edge["target"]
                if target not in component:
                    continue
                if target == start:
                    return path + [start]
                if target not in visited:
                    visited.add(target)
                    queue.append((target, path + [target]))
    return None


def _feedback_analysis(
    nodes: dict[str, dict],
    adjacency: dict[str, list[dict]],
    cancellation_token: CancellationToken | None,
) -> tuple[list[list[str]], list[dict], bool]:
    """Find every cyclic region and one representative cycle per region."""
    index = 0
    indices: dict[str, int] = {}
    low_links: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[set[str]] = []

    def visit(node_id: str) -> None:
        nonlocal index
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        indices[node_id] = index
        low_links[node_id] = index
        index += 1
        stack.append(node_id)
        on_stack.add(node_id)
        for edge in adjacency[node_id]:
            target = edge["target"]
            if target not in indices:
                visit(target)
                low_links[node_id] = min(low_links[node_id], low_links[target])
            elif target in on_stack:
                low_links[node_id] = min(low_links[node_id], indices[target])
        if low_links[node_id] != indices[node_id]:
            return
        component: set[str] = set()
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.add(member)
            if member == node_id:
                break
        has_self_loop = (
            len(component) == 1
            and any(edge["target"] == node_id for edge in adjacency[node_id])
        )
        if len(component) > 1 or has_self_loop:
            components.append(component)

    for node_id in nodes:
        if node_id not in indices:
            visit(node_id)

    components.sort(key=lambda value: tuple(sorted(value)))
    truncated = len(components) > MAX_FEEDBACK_COMPONENTS
    selected = components[:MAX_FEEDBACK_COMPONENTS]
    cycles = []
    summaries = []
    for component in selected:
        cycle = _representative_cycle(component, adjacency)
        internal_edges = sum(
            1
            for source in component
            for edge in adjacency[source]
            if edge["target"] in component
        )
        summary = {
            "concepts": sorted(component),
            "concept_count": len(component),
            "relationship_count": internal_edges,
        }
        combined_size = _serialized_size({
            "cycle": cycle,
            "summary": summary,
        })
        current_size = _serialized_size({
            "cycles": cycles,
            "summaries": summaries,
        })
        if current_size + combined_size > MAX_FEEDBACK_OUTPUT_CHARS:
            truncated = True
            break
        if cycle is not None:
            cycles.append(cycle)
        summaries.append(summary)
    return cycles, summaries, truncated


def knowledge_graph_builder(
    concepts: list[dict],
    relationships: list[dict],
    query: dict | None = None,
    max_depth: int = 4,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """Validate a semantic graph and infer explainable multi-hop connections.

    Every inferred connection includes the exact relationship path that supports it;
    the function never invents edges from concept names alone.
    """
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    if not isinstance(concepts, list) or not isinstance(relationships, list):
        return _json({"error": "concepts and relationships must be arrays"})
    if query is not None and not isinstance(query, dict):
        return _json({"error": "query must be an object"})
    if len(concepts) > 500 or len(relationships) > 3000:
        return _json({"error": "Graph exceeds the 500-concept/3000-relationship safety limit"})
    if not concepts:
        return _json({"error": "concepts must contain at least one concept"})

    nodes: dict[str, dict] = {}
    errors: list[str] = []
    for index, concept in enumerate(concepts):
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        if not isinstance(concept, dict):
            errors.append(f"concepts[{index}] must be an object")
            continue
        unknown_fields = sorted(set(concept) - {"id", "label", "attributes"})
        if unknown_fields:
            errors.append(
                f"concepts[{index}] contains unsupported fields: {', '.join(unknown_fields)}"
            )
            continue
        raw_id = concept.get("id")
        if not isinstance(raw_id, str):
            errors.append(f"concepts[{index}].id must be a string")
            continue
        node_id = raw_id.strip()
        if not node_id:
            errors.append(f"concepts[{index}] is missing id")
        elif len(node_id) > MAX_IDENTIFIER_CHARS or any(ord(char) < 32 for char in node_id):
            errors.append(f"concepts[{index}] id is invalid or exceeds {MAX_IDENTIFIER_CHARS} characters")
        elif node_id in nodes:
            errors.append(f"Duplicate concept id: {node_id}")
        else:
            raw_label = concept.get("label", node_id)
            if raw_label is None:
                raw_label = node_id
            if not isinstance(raw_label, str):
                errors.append(f"concepts[{index}].label must be a string")
                continue
            label = raw_label.strip()
            if not label:
                errors.append(f"concepts[{index}].label cannot be empty")
                continue
            if len(label) > MAX_LABEL_CHARS:
                errors.append(
                    f"concepts[{index}].label exceeds {MAX_LABEL_CHARS} characters"
                )
                continue
            attributes = concept.get("attributes", {})
            if not isinstance(attributes, dict):
                errors.append(f"concepts[{index}].attributes must be an object")
                continue
            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "attributes": attributes,
            }

    edges: list[dict] = []
    edge_ids: set[str] = set()
    adjacency: dict[str, list[dict]] = defaultdict(list)
    reverse: dict[str, list[dict]] = defaultdict(list)
    signatures: dict[tuple[str, str], set[str]] = defaultdict(set)
    for index, relation in enumerate(relationships):
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        if not isinstance(relation, dict):
            errors.append(f"relationships[{index}] must be an object")
            continue
        unknown_fields = sorted(
            set(relation) - {"id", "source", "target", "type", "weight", "evidence"}
        )
        if unknown_fields:
            errors.append(
                f"relationships[{index}] contains unsupported fields: "
                + ", ".join(unknown_fields)
            )
            continue
        raw_source = relation.get("source")
        raw_target = relation.get("target")
        raw_type = relation.get("type", "related_to")
        raw_edge_id = relation.get("id")
        if not isinstance(raw_source, str) or not isinstance(raw_target, str):
            errors.append(f"relationships[{index}] source and target must be strings")
            continue
        if not isinstance(raw_type, str):
            errors.append(f"relationships[{index}] type must be a string")
            continue
        if raw_edge_id is not None and not isinstance(raw_edge_id, str):
            errors.append(f"relationships[{index}] id must be a string")
            continue
        source = raw_source.strip()
        target = raw_target.strip()
        relation_type = raw_type.strip().casefold()
        edge_id = (raw_edge_id or f"r{index + 1}").strip()
        if source not in nodes or target not in nodes:
            errors.append(f"relationships[{index}] references an unknown concept")
            continue
        if (
            not relation_type
            or len(relation_type) > MAX_IDENTIFIER_CHARS
            or any(ord(char) < 32 for char in relation_type)
        ):
            errors.append(
                f"relationships[{index}] type is invalid or exceeds "
                f"{MAX_IDENTIFIER_CHARS} characters"
            )
            continue
        if not edge_id:
            errors.append(f"relationships[{index}] id cannot be empty")
            continue
        if len(edge_id) > MAX_IDENTIFIER_CHARS or any(ord(char) < 32 for char in edge_id):
            errors.append(f"relationships[{index}] id is invalid or exceeds {MAX_IDENTIFIER_CHARS} characters")
            continue
        if edge_id in edge_ids:
            errors.append(f"Duplicate relationship id: {edge_id}")
            continue
        if isinstance(relation.get("weight", 1.0), bool):
            errors.append(f"relationships[{index}] weight must be numeric")
            continue
        try:
            weight = float(relation.get("weight", 1.0))
        except (TypeError, ValueError, OverflowError):
            errors.append(f"relationships[{index}] weight must be numeric")
            continue
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            errors.append(f"relationships[{index}] weight must be finite and between 0 and 1")
            continue
        edge_ids.add(edge_id)
        edge = {
            "id": edge_id,
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
    try:
        input_size = _serialized_size({
            "concepts": list(nodes.values()),
            "relationships": edges,
            "query": query,
        })
    except ValueError as exc:
        return _json({"error": str(exc)})
    if input_size > MAX_INPUT_JSON_CHARS:
        return _json({
            "error": f"Graph input exceeds the {MAX_INPUT_JSON_CHARS}-character serialized limit"
        })

    if not isinstance(max_depth, int) or isinstance(max_depth, bool):
        return _json({"error": "max_depth must be an integer"})
    if not 1 <= max_depth <= 8:
        return _json({"error": "max_depth must be between 1 and 8"})
    depth_limit = max_depth
    query_value = query or {}
    unknown_query_fields = sorted(
        set(query_value) - {"source", "target", "relation_types"}
    )
    if unknown_query_fields:
        return _json({
            "error": "query contains unsupported fields",
            "unknown": unknown_query_fields,
        })
    raw_start = query_value.get("source")
    raw_goal = query_value.get("target")
    if raw_start is not None and not isinstance(raw_start, str):
        return _json({"error": "query.source must be a string"})
    if raw_goal is not None and not isinstance(raw_goal, str):
        return _json({"error": "query.target must be a string"})
    start = raw_start.strip() if isinstance(raw_start, str) else None
    goal = raw_goal.strip() if isinstance(raw_goal, str) else None
    start = start or None
    goal = goal or None
    raw_types = (query or {}).get("relation_types")
    if raw_types is not None and not isinstance(raw_types, list):
        return _json({"error": "query.relation_types must be an array"})
    if raw_types is not None and len(raw_types) > 100:
        return _json({"error": "query.relation_types exceeds the 100-item limit"})
    if raw_types is not None:
        invalid_type_indexes = [
            index
            for index, value in enumerate(raw_types)
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value.strip()) > MAX_IDENTIFIER_CHARS
                or any(ord(character) < 32 for character in value)
            )
        ]
        if invalid_type_indexes:
            return _json({
                "error": "query.relation_types entries must be non-empty strings",
                "invalid_indexes": invalid_type_indexes,
            })
        allowed_types = {value.strip().casefold() for value in raw_types}
    else:
        # A semantic graph can contain dependency, ownership, composition, or
        # domain-specific edges. Defaulting only to a small causal vocabulary
        # silently hid valid paths, so an omitted filter now means all types.
        allowed_types = {edge["type"] for edge in edges}
    if raw_types is not None and not allowed_types:
        return _json({"error": "query.relation_types cannot be empty"})
    if start and start not in nodes:
        return _json({"error": f"Unknown query source: {start}"})
    if goal and goal not in nodes:
        return _json({"error": f"Unknown query target: {goal}"})

    direct_matches = []
    direct_matches_truncated = False
    if start or goal:
        matching_edges = [
            edge
            for edge in edges
            if edge["type"] in allowed_types
            and (start is None or edge["source"] == start)
            and (goal is None or edge["target"] == goal)
        ]
        all_direct_matches = [
            {
                "edge": edge["id"],
                "from": edge["source"],
                "type": edge["type"],
                "to": edge["target"],
                "weight": edge["weight"],
                "evidence": edge["evidence"],
            }
            for edge in matching_edges[:MAX_DIRECT_MATCHES]
        ]
        direct_matches, omitted_direct_matches = _bounded_items(
            all_direct_matches, MAX_DIRECT_OUTPUT_CHARS
        )
        direct_matches_truncated = (
            len(matching_edges) > MAX_DIRECT_MATCHES
            or omitted_direct_matches > 0
        )

    paths: list[dict] = []
    path_output_chars = 2
    path_output_truncated = False
    traversal_states = 0
    traversal_truncated = False
    origins = [start] if start else list(nodes)
    for origin_index, origin in enumerate(origins):
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        queue = deque([(origin, [], {origin})])
        while (
            queue
            and len(paths) < MAX_INFERRED_PATHS
            and traversal_states < MAX_TRAVERSAL_STATES
            and not path_output_truncated
        ):
            if cancellation_token and traversal_states % 64 == 0:
                cancellation_token.raise_if_cancelled()
            current, path, visited = queue.popleft()
            traversal_states += 1
            if len(path) >= depth_limit:
                continue
            for edge in adjacency[current]:
                if edge["type"] not in allowed_types or edge["target"] in visited:
                    continue
                new_path = path + [edge]
                destination = edge["target"]
                if len(new_path) >= 2 and (goal is None or destination == goal):
                    path_types = {value["type"] for value in new_path}
                    if path_types <= POSITIVE_RELATIONS | NEGATIVE_RELATIONS:
                        negative_count = sum(
                            value["type"] in NEGATIVE_RELATIONS
                            for value in new_path
                        )
                        inferred_effect = (
                            "negative" if negative_count % 2 else "positive"
                        )
                    else:
                        inferred_effect = "unspecified"
                    record = {
                        "source": origin,
                        "target": destination,
                        "inferred_effect": inferred_effect,
                        "confidence": round(min(e["weight"] for e in new_path), 4),
                        "path": [{"edge": e["id"], "from": e["source"], "type": e["type"], "to": e["target"]} for e in new_path],
                    }
                    record_size = _serialized_size(record) + (1 if paths else 0)
                    if path_output_chars + record_size > MAX_PATH_OUTPUT_CHARS:
                        path_output_truncated = True
                        traversal_truncated = True
                        break
                    paths.append(record)
                    path_output_chars += record_size
                queue.append((destination, new_path, visited | {destination}))
        if queue or origin_index + 1 < len(origins):
            if (
                traversal_states >= MAX_TRAVERSAL_STATES
                or len(paths) >= MAX_INFERRED_PATHS
                or path_output_truncated
            ):
                traversal_truncated = True
        if (
            len(paths) >= MAX_INFERRED_PATHS
            or traversal_states >= MAX_TRAVERSAL_STATES
            or path_output_truncated
        ):
            traversal_truncated = True
            break

    all_contradictions = []
    for (source, target), types in signatures.items():
        positive = types & POSITIVE_RELATIONS
        negative = types & NEGATIVE_RELATIONS
        if positive and negative:
            all_contradictions.append({
                "source": source,
                "target": target,
                "positive": sorted(positive),
                "negative": sorted(negative),
            })
    contradictions, omitted_contradictions = _bounded_items(
        all_contradictions[:MAX_CONTRADICTIONS],
        MAX_CONTRADICTION_OUTPUT_CHARS,
    )
    contradiction_truncated = (
        len(all_contradictions) > MAX_CONTRADICTIONS
        or omitted_contradictions > 0
    )

    cycles, feedback_components, feedback_truncated = _feedback_analysis(
        nodes, adjacency, cancellation_token
    )

    centrality = sorted(
        ({"concept": node_id, "degree": len(adjacency[node_id]) + len(reverse[node_id])} for node_id in nodes),
        key=lambda item: (-item["degree"], item["concept"]),
    )
    normalized_graph = {
        "concepts": list(nodes.values()),
        "relationships": edges,
    }
    if _serialized_size(normalized_graph) <= MAX_GRAPH_ECHO_CHARS:
        graph_result = {
            **normalized_graph,
            "concept_count": len(nodes),
            "relationship_count": len(edges),
            "echo_truncated": False,
        }
    else:
        concept_preview, omitted_concepts = _bounded_items(
            normalized_graph["concepts"], MAX_GRAPH_ECHO_CHARS // 3
        )
        relationship_preview, omitted_relationships = _bounded_items(
            normalized_graph["relationships"], MAX_GRAPH_ECHO_CHARS * 2 // 3
        )
        graph_result = {
            "concept_count": len(nodes),
            "relationship_count": len(edges),
            "concepts": concept_preview,
            "relationships": relationship_preview,
            "echo_truncated": True,
            "omitted_concepts": omitted_concepts,
            "omitted_relationships": omitted_relationships,
        }
    return _json({
        "graph": graph_result,
        "analysis": {
            "direct_matches": direct_matches,
            "direct_matches_truncated": direct_matches_truncated,
            "inferred_paths": paths,
            "traversal_truncated": traversal_truncated,
            "path_output_truncated": path_output_truncated,
            "contradictions": contradictions,
            "contradictions_truncated": contradiction_truncated,
            "potential_feedback_cycles": cycles,
            "feedback_components": feedback_components,
            "feedback_components_truncated": feedback_truncated,
            "central_concepts": centrality[:10],
            "confidence_method": "minimum edge weight along each inferred path",
        },
        "limits": {
            "max_depth": depth_limit,
            "path_limit": MAX_INFERRED_PATHS,
            "traversal_state_limit": MAX_TRAVERSAL_STATES,
            "traversal_states": traversal_states,
        },
    })
