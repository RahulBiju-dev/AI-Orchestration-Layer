"""
agent/tool_runner.py — shared tool-call execution helpers.

Tool calls emitted in the same model response are normally independent, but
some tools mutate local/external state or are intended as ordered preflights.
This module centralizes the conservative rules used by both the terminal CLI
and the web UI so safe read-only calls can run concurrently without changing
side-effect ordering.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from typing import Callable, Iterable

from tools.registry import TOOL_DISPATCH

MAX_PARALLEL_TOOL_WORKERS = 4

_PARALLEL_SAFE_TOOLS = {
    "get_current_datetime",
    "web_search",
    "web_scrape",
    "read_document",
    "read_file",
    "view_code",
    "describe_image",
    "vault_search",
    "list_vaults",
    "list_vault_aliases",
    "knowledge_graph_builder",
    "run_simulation",
    "context_memory_optimizer",
    "reasoning_chain_debugger",
}

_TEMPORAL_PRELIGHT_TOOL = "get_current_datetime"
_TEMPORAL_DEPENDENT_TOOLS = {
    "web_search",
    "web_scrape",
    "google_workspace",
    "api_orchestrator",
}


@dataclass(frozen=True)
class ToolCallSpec:
    index: int
    name: str
    arguments: dict
    raw: dict


@dataclass(frozen=True)
class ToolCallResult:
    spec: ToolCallSpec
    content: str

    def as_tool_message(self) -> dict:
        return {
            "role": "tool",
            "tool_name": self.spec.name,
            "name": self.spec.name,
            "content": self.content,
        }


def normalize_tool_calls(tool_calls: Iterable[dict]) -> list[ToolCallSpec]:
    specs: list[ToolCallSpec] = []
    for index, call in enumerate(tool_calls):
        function = call.get("function") if isinstance(call, dict) else None
        function = function if isinstance(function, dict) else {}
        arguments = function.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        specs.append(
            ToolCallSpec(
                index=index,
                name=str(function.get("name") or ""),
                arguments=arguments,
                raw=call,
            )
        )
    return specs


def _json_error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False, separators=(",", ":"))


def execute_tool_call(spec: ToolCallSpec) -> ToolCallResult:
    handler = TOOL_DISPATCH.get(spec.name)
    if handler is None:
        return ToolCallResult(spec, _json_error(f"Unknown tool '{spec.name}'"))
    try:
        result = handler(**spec.arguments)
    except Exception as exc:
        return ToolCallResult(spec, _json_error(f"Tool execution failed: {exc}"))
    if isinstance(result, str):
        return ToolCallResult(spec, result)
    return ToolCallResult(spec, json.dumps(result, ensure_ascii=False, separators=(",", ":")))


def is_parallel_safe(spec: ToolCallSpec) -> bool:
    if spec.name == "spreadsheet":
        return str(spec.arguments.get("action") or "").lower() in {"view", "read"}
    return spec.name in _PARALLEL_SAFE_TOOLS


def _has_temporal_preflight_dependency(specs: list[ToolCallSpec]) -> bool:
    names = {spec.name for spec in specs}
    return _TEMPORAL_PRELIGHT_TOOL in names and bool(names & _TEMPORAL_DEPENDENT_TOOLS)


def build_execution_batches(specs: list[ToolCallSpec]) -> list[tuple[bool, list[ToolCallSpec]]]:
    """Return ordered batches as (can_run_parallel, specs)."""
    batches: list[tuple[bool, list[ToolCallSpec]]] = []
    pending_parallel: list[ToolCallSpec] = []

    def flush_parallel() -> None:
        nonlocal pending_parallel
        if not pending_parallel:
            return
        can_parallel = len(pending_parallel) > 1 and not _has_temporal_preflight_dependency(pending_parallel)
        batches.append((can_parallel, pending_parallel))
        pending_parallel = []

    for spec in specs:
        if is_parallel_safe(spec):
            pending_parallel.append(spec)
            continue
        flush_parallel()
        batches.append((False, [spec]))
    flush_parallel()
    return batches


StartCallback = Callable[[ToolCallSpec], None]
EndCallback = Callable[[ToolCallResult], None]
ParallelCallback = Callable[[list[ToolCallSpec]], None]


def execute_tool_calls(
    tool_calls: Iterable[dict],
    *,
    on_start: StartCallback | None = None,
    on_end: EndCallback | None = None,
    on_parallel_batch: ParallelCallback | None = None,
) -> list[ToolCallResult]:
    specs = normalize_tool_calls(tool_calls)
    results: dict[int, ToolCallResult] = {}

    for can_parallel, batch in build_execution_batches(specs):
        if not can_parallel:
            for spec in batch:
                if on_start:
                    on_start(spec)
                result = execute_tool_call(spec)
                results[spec.index] = result
                if on_end:
                    on_end(result)
            continue

        if on_parallel_batch:
            on_parallel_batch(batch)
        for spec in batch:
            if on_start:
                on_start(spec)

        worker_count = min(MAX_PARALLEL_TOOL_WORKERS, len(batch))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures: dict[Future[ToolCallResult], ToolCallSpec] = {
                executor.submit(execute_tool_call, spec): spec for spec in batch
            }
            for future in as_completed(futures):
                result = future.result()
                results[result.spec.index] = result
                if on_end:
                    on_end(result)

    return [results[index] for index in sorted(results)]
