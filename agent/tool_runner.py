"""Shared, bounded and metadata-driven tool-call execution helpers."""

from __future__ import annotations

import atexit
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
import inspect
import json
import math
import threading
import time
from typing import Callable, ContextManager, Iterable

from agent.cancellation import CancellationToken, OperationCancelled
from agent.runtime_config import get_runtime_config
from tools.registry import (
    TOOL_DISPATCH,
    TOOL_SCHEMA_BY_NAME,
    ToolMetadata,
    get_tool_metadata,
)


_RUNTIME_CONFIG = get_runtime_config()
MAX_PARALLEL_TOOL_WORKERS = _RUNTIME_CONFIG.tool_workers
_HANDLER_WORKERS = _RUNTIME_CONFIG.tool_workers
_POLL_SECONDS = 0.05
_HANDLER_EXECUTOR = ThreadPoolExecutor(
    max_workers=_HANDLER_WORKERS,
    thread_name_prefix="selene-tool",
)
_SHUTDOWN_LOCK = threading.Lock()
_SHUTDOWN = False
_HEAVY_TOOL_SLOTS = threading.BoundedSemaphore(_RUNTIME_CONFIG.heavy_tool_concurrency)
_RUNNER_CONTEXT = threading.local()


class ToolResultStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ToolCallSpec:
    index: int
    name: str
    arguments: dict
    raw: dict
    argument_error: str | None = None


@dataclass(frozen=True)
class ToolCallResult:
    spec: ToolCallSpec
    content: str
    status: ToolResultStatus = ToolResultStatus.SUCCESS
    duplicate_of: int | None = None
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.status is ToolResultStatus.SUCCESS

    def as_tool_message(self) -> dict:
        return {
            "role": "tool",
            "tool_name": self.spec.name,
            "name": self.spec.name,
            "content": self.content,
        }


def _error_content(code: str, message: str, **details: object) -> str:
    payload: dict[str, object] = {
        "ok": False,
        "error": message,
        "error_code": code,
    }
    if details:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def normalize_tool_arguments(arguments: object) -> tuple[dict, str | None]:
    """Return parsed tool arguments and an optional validation error."""
    if arguments is None:
        return {}, None
    if isinstance(arguments, dict):
        return arguments, None
    if isinstance(arguments, str):
        raw = arguments.strip()
        if not raw:
            return {}, None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {}, f"Tool arguments are not valid JSON: {exc.msg}"
        if isinstance(parsed, dict):
            return parsed, None
        return {}, "Tool arguments must decode to a JSON object"
    return {}, "Tool arguments must be a JSON object"


def normalize_tool_calls(tool_calls: Iterable[dict]) -> list[ToolCallSpec]:
    specs: list[ToolCallSpec] = []
    for index, call in enumerate(tool_calls):
        function = call.get("function") if isinstance(call, dict) else None
        function = function if isinstance(function, dict) else {}
        arguments, argument_error = normalize_tool_arguments(function.get("arguments"))
        specs.append(
            ToolCallSpec(
                index=index,
                name=str(function.get("name") or "").strip(),
                arguments=arguments,
                raw=call if isinstance(call, dict) else {},
                argument_error=argument_error,
            )
        )
    return specs


def _matches_type(value: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _validate_schema_value(value: object, schema: dict, path: str, errors: list[str]) -> None:
    expected = schema.get("type")
    expected_types = [expected] if isinstance(expected, str) else expected if isinstance(expected, list) else []
    if expected_types and not any(_matches_type(value, item) for item in expected_types):
        errors.append(f"{path} must be {', '.join(expected_types)}")
        return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']}")
        return

    if isinstance(value, dict):
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for required in schema.get("required", []):
            if required not in value:
                errors.append(f"{path}.{required} is required")
        if schema.get("additionalProperties") is False:
            for key in value.keys() - properties.keys():
                errors.append(f"{path}.{key} is not allowed")
        for key, item in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, dict):
                _validate_schema_value(item, child_schema, f"{path}.{key}", errors)
    elif isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            errors.append(f"{path} must contain at least {minimum} item(s)")
        if isinstance(maximum, int) and len(value) > maximum:
            errors.append(f"{path} must contain at most {maximum} item(s)")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, f"{path}[{index}]", errors)
    elif isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            errors.append(f"{path} must be at least {minimum} character(s)")
        if isinstance(maximum, int) and len(value) > maximum:
            errors.append(f"{path} must be at most {maximum} character(s)")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            error = f"{path} must be finite"
            if error not in errors:
                errors.append(error)
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path} must be at least {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path} must be at most {maximum}")


def _validate_finite_numbers(
    value: object,
    path: str,
    errors: list[str],
    seen: set[int],
    depth: int = 0,
) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path} must be finite")
        return
    if not isinstance(value, (dict, list)):
        return
    if depth >= 100:
        errors.append(f"{path} exceeds the maximum nesting depth")
        return
    identity = id(value)
    if identity in seen:
        errors.append(f"{path} contains a recursive value")
        return
    seen.add(identity)
    try:
        if isinstance(value, dict):
            for key, child in value.items():
                _validate_finite_numbers(child, f"{path}.{key}", errors, seen, depth + 1)
        else:
            for index, child in enumerate(value):
                _validate_finite_numbers(child, f"{path}[{index}]", errors, seen, depth + 1)
    finally:
        seen.remove(identity)


def validate_tool_arguments(name: str, arguments: dict) -> list[str]:
    """Validate model-provided arguments against the registry's JSON schema subset."""
    schema = TOOL_SCHEMA_BY_NAME.get(name)
    if not isinstance(schema, dict):
        return []
    errors: list[str] = []
    _validate_finite_numbers(arguments, "arguments", errors, set())
    _validate_schema_value(arguments, schema, "arguments", errors)
    return errors


def _serialize_result(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _finalize_handler_result(
    spec: ToolCallSpec,
    metadata: ToolMetadata,
    result: object,
) -> ToolCallResult:
    """Serialize, classify, and bound a handler result without leaking failures."""
    try:
        serialized = _serialize_result(result)
    except (TypeError, ValueError, OverflowError) as exc:
        return ToolCallResult(
            spec,
            _error_content(
                "invalid_tool_result",
                f"Tool returned a result that could not be serialized: {exc}",
                exception_type=type(exc).__name__,
            ),
            ToolResultStatus.ERROR,
        )
    status = _handler_result_status(serialized)
    content, truncated = _bounded_content(serialized, metadata, status)
    return ToolCallResult(spec, content, status, truncated=truncated)


def _bounded_content(
    content: str,
    metadata: ToolMetadata,
    status: ToolResultStatus,
) -> tuple[str, bool]:
    # Metadata supplies a tool-specific ceiling; the active hardware profile
    # supplies a stricter context-aware ceiling so a nominally valid 50K web
    # result cannot consume an entire 4K follow-up prompt by itself.
    limit = min(metadata.max_output_chars, max(2_000, _RUNTIME_CONFIG.num_ctx))
    if len(content) <= limit:
        return content, False
    payload: dict[str, object] = {
        "ok": status is ToolResultStatus.SUCCESS,
        "truncated": True,
        "tool": metadata.name,
        "original_characters": len(content),
        "content_preview": "",
    }
    low, high = 0, len(content)
    bounded = ""
    while low <= high:
        midpoint = (low + high) // 2
        payload["content_preview"] = content[:midpoint]
        candidate = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(candidate) <= limit:
            bounded = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    if not bounded:
        # ToolMetadata enforces a minimum large enough for this fallback.
        payload["content_preview"] = ""
        bounded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return bounded, True


def _handler_result_status(content: str) -> ToolResultStatus:
    """Recognize the structured error convention used by existing tools."""
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        # A few older tools still return human-readable errors rather than
        # JSON. Do not report those as successful executions or allow a later
        # side effect to proceed as though the call succeeded.
        normalized = content.lstrip().casefold() if isinstance(content, str) else ""
        if normalized.startswith(("error:", "error ")):
            return ToolResultStatus.ERROR
        return ToolResultStatus.SUCCESS
    if isinstance(payload, dict) and (
        payload.get("ok") is False or bool(payload.get("error"))
    ):
        return ToolResultStatus.ERROR
    return ToolResultStatus.SUCCESS


ResourceGuard = Callable[[ToolMetadata, CancellationToken], ContextManager]
_RESOURCE_GUARD: ResourceGuard | None = None


def set_tool_resource_guard(guard: ResourceGuard | None) -> None:
    """Install a process-wide resource coordinator hook (for GPU-heavy tools)."""
    global _RESOURCE_GUARD
    _RESOURCE_GUARD = guard


def _default_model_resource_guard(
    metadata: ToolMetadata,
    cancellation_token: CancellationToken,
) -> ContextManager:
    if not metadata.gpu_heavy:
        return nullcontext()
    # Lazy import keeps registry/tool imports lightweight and avoids making an
    # optional Ollama client failure prevent ordinary tools from loading.
    from agent.ollama_runtime import OperationKind, get_ollama_coordinator

    kind = OperationKind.VISION if metadata.name == "describe_image" else OperationKind.EMBEDDING
    # Nested embedding/vision helpers run on this same handler thread and use
    # the identical owner, making coordinator acquisition re-entrant.
    owner = f"tool:{threading.get_ident()}"
    return get_ollama_coordinator().operation(
        kind,
        owner=owner,
        cancellation_token=cancellation_token,
        wait_timeout=metadata.default_timeout_seconds,
        operation_timeout=metadata.default_timeout_seconds,
    )


@contextmanager
def _heavy_tool_guard(metadata: ToolMetadata, cancellation_token: CancellationToken):
    """Bound CPU/GPU-heavy tools without serializing ordinary tool work."""
    acquired = False
    if metadata.cpu_heavy or metadata.gpu_heavy:
        while not acquired:
            cancellation_token.raise_if_cancelled()
            acquired = _HEAVY_TOOL_SLOTS.acquire(timeout=_POLL_SECONDS)
    try:
        yield
    finally:
        if acquired:
            _HEAVY_TOOL_SLOTS.release()


def _invoke_handler(
    handler: Callable,
    spec: ToolCallSpec,
    metadata: ToolMetadata,
    cancellation_token: CancellationToken,
) -> object:
    cancellation_token.raise_if_cancelled()
    coordinator_guard = (
        _RESOURCE_GUARD(metadata, cancellation_token)
        if _RESOURCE_GUARD
        else _default_model_resource_guard(metadata, cancellation_token)
    )
    previous_depth = int(getattr(_RUNNER_CONTEXT, "depth", 0))
    _RUNNER_CONTEXT.depth = previous_depth + 1
    try:
        with _heavy_tool_guard(metadata, cancellation_token):
            with coordinator_guard:
                cancellation_token.raise_if_cancelled()
                arguments = dict(spec.arguments)
                if metadata.supports_cancellation:
                    try:
                        parameters = inspect.signature(handler).parameters
                    except (TypeError, ValueError):
                        parameters = {}
                    if "cancellation_token" in parameters:
                        arguments["cancellation_token"] = cancellation_token
                return handler(**arguments)
    finally:
        _RUNNER_CONTEXT.depth = previous_depth


def execute_tool_call(
    spec: ToolCallSpec,
    *,
    cancellation_token: CancellationToken | None = None,
    timeout_seconds: float | None = None,
) -> ToolCallResult:
    """Execute one call with validation, cooperative cancellation, timeout and bounds."""
    owner_token = cancellation_token or CancellationToken()
    if owner_token.cancelled:
        return ToolCallResult(
            spec,
            _error_content("cancelled", owner_token.reason),
            ToolResultStatus.CANCELLED,
        )
    if spec.argument_error:
        return ToolCallResult(
            spec,
            _error_content("invalid_arguments", spec.argument_error),
            ToolResultStatus.ERROR,
        )
    if not spec.name:
        return ToolCallResult(
            spec,
            _error_content("unknown_tool", "Tool name is missing"),
            ToolResultStatus.ERROR,
        )
    handler = TOOL_DISPATCH.get(spec.name)
    metadata = get_tool_metadata(spec.name, spec.arguments)
    if handler is None or metadata is None:
        return ToolCallResult(
            spec,
            _error_content("unknown_tool", f"Unknown tool '{spec.name}'"),
            ToolResultStatus.ERROR,
        )
    validation_errors = validate_tool_arguments(spec.name, spec.arguments)
    if validation_errors:
        return ToolCallResult(
            spec,
            _error_content(
                "invalid_arguments",
                "Tool arguments failed schema validation",
                validation_errors=validation_errors,
            ),
            ToolResultStatus.ERROR,
        )

    try:
        timeout = (
            float(metadata.default_timeout_seconds)
            if timeout_seconds is None
            else float(timeout_seconds)
        )
    except (TypeError, ValueError, OverflowError):
        timeout = math.nan
    if not math.isfinite(timeout) or timeout <= 0:
        return ToolCallResult(
            spec,
            _error_content("invalid_timeout", "Tool timeout must be a positive finite number"),
            ToolResultStatus.ERROR,
        )

    # Registered tools such as routines may dispatch another registered tool.
    # Running that nested call through the same bounded worker pool can deadlock
    # when every worker is already inside a parent tool. Execute inline while
    # retaining validation/resource guards and cooperative timeout signalling.
    if int(getattr(_RUNNER_CONTEXT, "depth", 0)) > 0:
        timed_out = threading.Event()

        def request_timeout() -> None:
            timed_out.set()
            owner_token.cancel("Tool timeout expired")

        timer = threading.Timer(timeout, request_timeout)
        timer.daemon = True
        timer.start()
        try:
            result = _invoke_handler(handler, spec, metadata, owner_token)
        except OperationCancelled as exc:
            status = ToolResultStatus.TIMEOUT if timed_out.is_set() else ToolResultStatus.CANCELLED
            code = "timeout" if timed_out.is_set() else "cancelled"
            return ToolCallResult(spec, _error_content(code, str(exc)), status)
        except Exception as exc:
            return ToolCallResult(
                spec,
                _error_content(
                    "execution_failed",
                    f"Tool execution failed: {exc}",
                    exception_type=type(exc).__name__,
                ),
                ToolResultStatus.ERROR,
            )
        finally:
            timer.cancel()
        if timed_out.is_set():
            return ToolCallResult(
                spec,
                _error_content("timeout", f"Tool exceeded its {timeout:g}s timeout"),
                ToolResultStatus.TIMEOUT,
            )
        return _finalize_handler_result(spec, metadata, result)

    with _SHUTDOWN_LOCK:
        if _SHUTDOWN:
            return ToolCallResult(
                spec,
                _error_content("runner_shutdown", "Tool runner is shutting down"),
                ToolResultStatus.CANCELLED,
            )
        execution_token = CancellationToken()
        future = _HANDLER_EXECUTOR.submit(
            _invoke_handler,
            handler,
            spec,
            metadata,
            execution_token,
        )

    deadline = time.monotonic() + timeout
    try:
        while True:
            if owner_token.cancelled:
                execution_token.cancel(owner_token.reason)
                future.cancel()
                return ToolCallResult(
                    spec,
                    _error_content(
                        "cancelled",
                        owner_token.reason,
                        backend_may_still_be_finishing=not metadata.supports_cancellation,
                    ),
                    ToolResultStatus.CANCELLED,
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                execution_token.cancel("Tool timeout expired")
                future.cancel()
                return ToolCallResult(
                    spec,
                    _error_content(
                        "timeout",
                        f"Tool exceeded its {timeout:g}s timeout",
                        backend_may_still_be_finishing=not metadata.supports_cancellation,
                    ),
                    ToolResultStatus.TIMEOUT,
                )
            try:
                result = future.result(timeout=min(_POLL_SECONDS, remaining))
                break
            except FutureTimeout:
                continue
    except OperationCancelled as exc:
        return ToolCallResult(
            spec,
            _error_content("cancelled", str(exc)),
            ToolResultStatus.CANCELLED,
        )
    except Exception as exc:
        return ToolCallResult(
            spec,
            _error_content(
                "execution_failed",
                f"Tool execution failed: {exc}",
                exception_type=type(exc).__name__,
            ),
            ToolResultStatus.ERROR,
        )

    return _finalize_handler_result(spec, metadata, result)


def is_parallel_safe(spec: ToolCallSpec) -> bool:
    metadata = get_tool_metadata(spec.name, spec.arguments)
    return bool(metadata and metadata.parallel_safe)


def _has_temporal_preflight_dependency(specs: list[ToolCallSpec]) -> bool:
    has_preflight = any(spec.name == "get_current_datetime" for spec in specs)
    return has_preflight and any(
        bool((metadata := get_tool_metadata(spec.name, spec.arguments)) and metadata.requires_temporal_preflight)
        for spec in specs
    )


def build_execution_batches(specs: list[ToolCallSpec]) -> list[tuple[bool, list[ToolCallSpec]]]:
    """Return ordered batches as ``(can_run_parallel, specs)``."""
    if _has_temporal_preflight_dependency(specs):
        # A live datetime read is side-effect free and may move ahead of the
        # dependent calls. Result delivery still follows original indices.
        specs = [
            *(spec for spec in specs if spec.name == "get_current_datetime"),
            *(spec for spec in specs if spec.name != "get_current_datetime"),
        ]
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


def tool_call_key(spec: ToolCallSpec) -> str | None:
    if not spec.name:
        return None
    arguments = spec.arguments
    if spec.argument_error:
        arguments = {"_argument_error": spec.argument_error}
    try:
        encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        encoded = str(arguments)
    return f"{spec.name}:{encoded}"


def _not_executed_result(spec: ToolCallSpec, code: str, message: str) -> ToolCallResult:
    return ToolCallResult(
        spec,
        _error_content(code, message),
        ToolResultStatus.CANCELLED,
    )


StartCallback = Callable[[ToolCallSpec], None]
EndCallback = Callable[[ToolCallResult], None]
ParallelCallback = Callable[[list[ToolCallSpec]], None]


def execute_tool_calls(
    tool_calls: Iterable[dict],
    *,
    on_start: StartCallback | None = None,
    on_end: EndCallback | None = None,
    on_parallel_batch: ParallelCallback | None = None,
    cancellation_token: CancellationToken | None = None,
) -> list[ToolCallResult]:
    """Execute a model batch, replaying exact duplicates in deterministic order."""
    token = cancellation_token or CancellationToken()
    specs = normalize_tool_calls(tool_calls)
    results: dict[int, ToolCallResult] = {}
    unique_specs: list[ToolCallSpec] = []
    first_index_by_key: dict[str, int] = {}
    duplicate_source: dict[int, int] = {}
    blocked_reason: str | None = None

    for spec in specs:
        key = tool_call_key(spec)
        if key is not None and key in first_index_by_key:
            duplicate_source[spec.index] = first_index_by_key[key]
            continue
        if key is not None:
            first_index_by_key[key] = spec.index
        unique_specs.append(spec)

    for can_parallel, batch in build_execution_batches(unique_specs):
        if token.cancelled or blocked_reason:
            for spec in batch:
                if on_start:
                    on_start(spec)
                result = (
                    execute_tool_call(spec, cancellation_token=token)
                    if token.cancelled
                    else _not_executed_result(
                        spec,
                        "blocked_by_prior_call",
                        blocked_reason or "A prior tool call did not finish safely",
                    )
                )
                results[spec.index] = result
                if on_end:
                    on_end(result)
            continue
        if not can_parallel:
            for spec in batch:
                if on_start:
                    on_start(spec)
                result = execute_tool_call(spec, cancellation_token=token)
                results[spec.index] = result
                if on_end:
                    on_end(result)
                metadata = get_tool_metadata(spec.name, spec.arguments)
                if (
                    result.status is ToolResultStatus.TIMEOUT
                    and metadata is not None
                    and metadata.side_effecting
                ):
                    blocked_reason = (
                        f"A prior side-effecting tool ({spec.name}) timed out and may still be finishing"
                    )
                elif (
                    result.status is ToolResultStatus.ERROR
                    and metadata is not None
                    and metadata.side_effecting
                    and not metadata.idempotent
                ):
                    # Non-idempotent writes that raise leave an uncertain world
                    # state: do not chain additional side effects in this batch.
                    blocked_reason = (
                        f"A prior non-idempotent side-effecting tool ({spec.name}) failed "
                        "and may have partially applied changes"
                    )
            continue

        if on_parallel_batch:
            on_parallel_batch(batch)
        for spec in batch:
            if on_start:
                on_start(spec)

        worker_count = min(MAX_PARALLEL_TOOL_WORKERS, len(batch))
        completed: dict[int, ToolCallResult] = {}
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="selene-tool-batch") as executor:
            futures: dict[Future[ToolCallResult], ToolCallSpec] = {
                executor.submit(execute_tool_call, spec, cancellation_token=token): spec
                for spec in batch
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    # execute_tool_call is deliberately exception-safe, but keep
                    # one unexpected worker failure from discarding every other
                    # result in an otherwise independent parallel batch.
                    result = ToolCallResult(
                        spec,
                        _error_content(
                            "runner_failed",
                            f"Tool runner worker failed: {exc}",
                            exception_type=type(exc).__name__,
                        ),
                        ToolResultStatus.ERROR,
                    )
                completed[result.spec.index] = result
        for spec in sorted(batch, key=lambda value: value.index):
            result = completed[spec.index]
            results[spec.index] = result
            if on_end:
                on_end(result)

    spec_by_index = {spec.index: spec for spec in specs}
    for index, source_index in duplicate_source.items():
        source = results[source_index]
        results[index] = ToolCallResult(
            spec=spec_by_index[index],
            content=source.content,
            status=source.status,
            duplicate_of=source_index,
            truncated=source.truncated,
        )
        if on_end:
            on_end(results[index])

    return [results[index] for index in sorted(results)]


def shutdown_tool_runner(*, wait: bool = False) -> None:
    """Reject new calls and release executor resources owned by Selene."""
    global _SHUTDOWN
    with _SHUTDOWN_LOCK:
        if _SHUTDOWN:
            return
        _SHUTDOWN = True
        _HANDLER_EXECUTOR.shutdown(wait=wait, cancel_futures=True)


atexit.register(shutdown_tool_runner)
