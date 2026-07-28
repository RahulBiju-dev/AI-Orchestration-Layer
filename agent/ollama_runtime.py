"""Shared Ollama operation coordination and startup-safe helpers.

The coordinator bounds only Ollama/model work.  It deliberately does not
serialize ordinary filesystem, CPU, or network tools.  Cancellation is
cooperative: callers should use :meth:`OperationLease.checkpoint` between
stream chunks and pass :meth:`OperationLease.remaining_seconds` to the local
API client so an abandoned operation cannot retain a lease indefinitely.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar

from agent.cancellation import CancellationToken, OperationCancelled
from agent.runtime_config import RuntimeConfig, get_runtime_config


class OllamaRuntimeError(RuntimeError):
    """Base class for controlled local-Ollama failures."""


class OllamaUnavailableError(OllamaRuntimeError):
    """The client package or local Ollama service is unavailable."""


class OllamaModelMissingError(OllamaRuntimeError):
    """A requested local model is not installed."""


class OllamaRequestTimeout(OllamaRuntimeError):
    """An Ollama operation exceeded its bounded deadline."""


class OllamaMalformedResponse(OllamaRuntimeError):
    """Ollama returned a response that does not satisfy the API contract."""


class OllamaContextOverflow(OllamaRuntimeError):
    """A model request cannot fit inside its configured context window."""


class InvalidModelfileError(OllamaRuntimeError):
    """A Modelfile cannot be safely parsed for a local model build."""


class CoordinatorError(RuntimeError):
    """Base class for coordinator ownership/lifecycle errors."""


class OperationWaitTimeout(CoordinatorError):
    """An operation could not obtain a model lease in time."""


class OperationDeadlineExceeded(CoordinatorError):
    """An active operation exceeded its execution deadline."""


class CoordinatorShutdownError(CoordinatorError):
    """New work was submitted after coordinator shutdown began."""


class OperationOwnershipError(CoordinatorError):
    """Cancellation or re-entry did not match the operation owner."""


class OperationKind(str, Enum):
    BUILD = "build"
    CHAT = "chat"
    TITLE = "title"
    SUMMARY = "summary"
    EMBEDDING = "embedding"
    VISION = "vision"


# Compatibility export; the canonical exception/token are shared with web
# generation and the tool runner in ``agent.cancellation``.
OperationCancelledError = OperationCancelled


@dataclass(frozen=True)
class ActiveOperationSnapshot:
    operation_id: str
    kind: OperationKind
    owner: str
    model: str | None
    started_monotonic: float
    deadline_monotonic: float
    elapsed_seconds: float
    cancellation_requested: bool
    thread_id: int


@dataclass
class _ActiveOperation:
    operation_id: str
    kind: OperationKind
    owner: str
    model: str | None
    started_monotonic: float
    deadline_monotonic: float
    cancellation_token: CancellationToken
    thread_id: int


@dataclass
class _Waiter:
    operation_id: str
    kind: OperationKind
    owner: str
    model: str | None
    cancellation_token: CancellationToken


class OperationLease:
    """A release-safe lease for one root or nested Ollama operation."""

    def __init__(
        self,
        coordinator: "OllamaCoordinator",
        *,
        operation_id: str,
        kind: OperationKind,
        owner: str,
        model: str | None,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
        owns_root_lease: bool,
    ) -> None:
        self._coordinator = coordinator
        self.operation_id = operation_id
        self.kind = kind
        self.owner = owner
        self.model = model
        self.cancellation_token = cancellation_token
        self.deadline_monotonic = deadline_monotonic
        self.is_reentrant = not owns_root_lease
        self._owns_root_lease = owns_root_lease
        self._released = False
        self._context_marker: str | None = None
        self._entered_thread_id: int | None = None

    def __enter__(self) -> "OperationLease":
        if self._released:
            raise CoordinatorError("Cannot enter a released Ollama operation lease.")
        if self._context_marker is not None:
            raise CoordinatorError("Ollama operation lease was entered more than once.")
        self._context_marker = self._coordinator._push_context(self.operation_id, self.owner)
        self._entered_thread_id = threading.get_ident()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.release()
        return False

    @property
    def cancelled(self) -> bool:
        return self.cancellation_token.cancelled

    def checkpoint(self) -> None:
        self.cancellation_token.raise_if_cancelled()
        if time.monotonic() >= self.deadline_monotonic:
            self.cancellation_token.cancel(
                f"{self.kind.value} operation {self.operation_id} exceeded its execution deadline."
            )
            raise OperationDeadlineExceeded(self.cancellation_token.reason)

    def remaining_seconds(self, minimum: float = 0.1) -> float:
        """Return remaining API time, raising if cancellation/deadline won."""
        self.checkpoint()
        return max(float(minimum), self.deadline_monotonic - time.monotonic())

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        context_error: CoordinatorError | None = None
        if self._context_marker is not None:
            # A streaming iterator may be closed by a disconnect/shutdown
            # thread rather than the thread that first consumed it. Thread
            # locals cannot be popped remotely; releasing the root operation
            # still must happen so low-VRAM work cannot deadlock forever. The
            # acquiring thread clears the stale marker on its next lookup.
            if self._entered_thread_id == threading.get_ident():
                try:
                    self._coordinator._pop_context(self._context_marker)
                except CoordinatorError as exc:
                    context_error = exc
            elif not self._owns_root_lease:
                context_error = CoordinatorError(
                    "Nested Ollama leases must be released on their acquiring thread."
                )
            self._context_marker = None
        if self._owns_root_lease:
            self._coordinator._release(self.operation_id)
        if context_error is not None:
            raise context_error


_T = TypeVar("_T")
_GPU_HEAVY_TOOL_KINDS = {OperationKind.EMBEDDING, OperationKind.VISION}


class OllamaCoordinator:
    """FIFO, re-entrant coordinator for bounded local model operations."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._condition = threading.Condition(threading.RLock())
        self._active: dict[str, _ActiveOperation] = {}
        self._waiters: list[_Waiter] = []
        self._shutdown = False
        self._context = threading.local()

    @staticmethod
    def _coerce_kind(kind: OperationKind | str) -> OperationKind:
        if isinstance(kind, OperationKind):
            return kind
        try:
            return OperationKind(str(kind).strip().lower())
        except ValueError as exc:
            choices = ", ".join(item.value for item in OperationKind)
            raise CoordinatorError(f"Unknown Ollama operation kind {kind!r}; expected: {choices}") from exc

    @staticmethod
    def _bounded_timeout(value: float | int | None, default: float, label: str) -> float:
        candidate = default if value is None else value
        try:
            timeout = float(candidate)
        except (TypeError, ValueError) as exc:
            raise CoordinatorError(f"{label} must be a positive number of seconds") from exc
        if not 0 < timeout <= 86400:
            raise CoordinatorError(f"{label} must be greater than 0 and no more than 86400 seconds")
        return timeout

    def _context_stack(self) -> list[tuple[str, str, str]]:
        stack = getattr(self._context, "stack", None)
        if stack is None:
            stack = []
            self._context.stack = stack
        return stack

    def _push_context(self, operation_id: str, owner: str) -> str:
        marker = uuid.uuid4().hex
        self._context_stack().append((marker, operation_id, owner))
        return marker

    def _pop_context(self, marker: str) -> None:
        stack = self._context_stack()
        if not stack or stack[-1][0] != marker:
            raise CoordinatorError("Ollama operation leases must exit in nested order on the acquiring thread.")
        stack.pop()

    def _current_context(self) -> tuple[str, str] | None:
        stack = self._context_stack()
        if not stack:
            return None
        _, operation_id, owner = stack[-1]
        with self._condition:
            if operation_id not in self._active:
                stack.clear()
                return None
        return operation_id, owner

    def is_owned_by_current_context(self, owner: str | None = None) -> bool:
        current = self._current_context()
        return bool(current and (owner is None or current[1] == owner))

    def current_context_owner(self) -> str | None:
        """Return the active nested-operation owner on this thread, if any."""
        current = self._current_context()
        return current[1] if current is not None else None

    def _can_start(self, kind: OperationKind) -> bool:
        active = tuple(self._active.values())
        if kind is OperationKind.BUILD:
            return not active
        if any(item.kind is OperationKind.BUILD for item in active):
            return False
        if len(active) >= self.config.model_concurrency:
            return False

        if self.config.serialize_embeddings:
            if kind is OperationKind.EMBEDDING and active:
                return False
            if any(item.kind is OperationKind.EMBEDDING for item in active):
                return False
        if self.config.serialize_vision:
            if kind is OperationKind.VISION and active:
                return False
            if any(item.kind is OperationKind.VISION for item in active):
                return False

        if kind in _GPU_HEAVY_TOOL_KINDS:
            heavy_count = sum(item.kind in _GPU_HEAVY_TOOL_KINDS for item in active)
            if heavy_count >= self.config.heavy_tool_concurrency:
                return False
        return True

    def acquire(
        self,
        kind: OperationKind | str,
        *,
        owner: str,
        model: str | None = None,
        cancellation_token: CancellationToken | None = None,
        wait_timeout: float | None = None,
        operation_timeout: float | None = None,
    ) -> OperationLease:
        """Acquire a FIFO lease, or reuse the current owner's root lease."""
        operation_kind = self._coerce_kind(kind)
        owner_name = str(owner or "").strip()
        if not owner_name or len(owner_name) > 256:
            raise OperationOwnershipError("Ollama operations require a non-empty owner up to 256 characters.")
        model_name = str(model).strip() if model else None
        default_timeout = self.config.timeout_for(operation_kind)
        run_timeout = self._bounded_timeout(operation_timeout, default_timeout, "operation_timeout")

        current = self._current_context()
        if current is not None:
            root_id, root_owner = current
            if root_owner != owner_name:
                raise OperationOwnershipError(
                    f"Nested Ollama work belongs to {root_owner!r}, not {owner_name!r}."
                )
            with self._condition:
                root = self._active.get(root_id)
                if root is None:
                    raise CoordinatorError("The current Ollama root lease is no longer active.")
                nested_deadline = min(root.deadline_monotonic, time.monotonic() + run_timeout)
                return OperationLease(
                    self,
                    operation_id=root.operation_id,
                    kind=operation_kind,
                    owner=owner_name,
                    model=model_name,
                    cancellation_token=root.cancellation_token,
                    deadline_monotonic=nested_deadline,
                    owns_root_lease=False,
                )

        queue_timeout = self._bounded_timeout(wait_timeout, default_timeout, "wait_timeout")
        token = cancellation_token or CancellationToken()
        token.raise_if_cancelled()
        waiter = _Waiter(
            operation_id=uuid.uuid4().hex,
            kind=operation_kind,
            owner=owner_name,
            model=model_name,
            cancellation_token=token,
        )
        wait_deadline = time.monotonic() + queue_timeout

        with self._condition:
            if self._shutdown:
                raise CoordinatorShutdownError("The Ollama coordinator is shutting down.")
            self._waiters.append(waiter)
            try:
                while True:
                    if self._shutdown:
                        raise CoordinatorShutdownError("The Ollama coordinator is shutting down.")
                    token.raise_if_cancelled()
                    now = time.monotonic()
                    if now >= wait_deadline:
                        raise OperationWaitTimeout(
                            f"Timed out waiting for a {operation_kind.value} model lease owned by {owner_name!r}."
                        )
                    if self._waiters and self._waiters[0] is waiter and self._can_start(operation_kind):
                        self._waiters.pop(0)
                        started = time.monotonic()
                        active = _ActiveOperation(
                            operation_id=waiter.operation_id,
                            kind=operation_kind,
                            owner=owner_name,
                            model=model_name,
                            started_monotonic=started,
                            deadline_monotonic=started + run_timeout,
                            cancellation_token=token,
                            thread_id=threading.get_ident(),
                        )
                        self._active[active.operation_id] = active
                        self._condition.notify_all()
                        return OperationLease(
                            self,
                            operation_id=active.operation_id,
                            kind=active.kind,
                            owner=active.owner,
                            model=active.model,
                            cancellation_token=active.cancellation_token,
                            deadline_monotonic=active.deadline_monotonic,
                            owns_root_lease=True,
                        )
                    self._condition.wait(timeout=min(0.1, max(0.001, wait_deadline - now)))
            except BaseException:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                self._condition.notify_all()
                raise

    def operation(self, kind: OperationKind | str, **kwargs: Any) -> OperationLease:
        """Readable alias for ``acquire`` intended for ``with`` statements."""
        return self.acquire(kind, **kwargs)

    def run(
        self,
        kind: OperationKind | str,
        callback: Callable[[OperationLease], _T],
        **kwargs: Any,
    ) -> _T:
        with self.acquire(kind, **kwargs) as lease:
            lease.checkpoint()
            result = callback(lease)
            lease.checkpoint()
            return result

    def _release(self, operation_id: str) -> None:
        with self._condition:
            self._active.pop(operation_id, None)
            self._condition.notify_all()

    def active_operations(self) -> tuple[ActiveOperationSnapshot, ...]:
        now = time.monotonic()
        with self._condition:
            operations = sorted(self._active.values(), key=lambda item: item.started_monotonic)
            return tuple(
                ActiveOperationSnapshot(
                    operation_id=item.operation_id,
                    kind=item.kind,
                    owner=item.owner,
                    model=item.model,
                    started_monotonic=item.started_monotonic,
                    deadline_monotonic=item.deadline_monotonic,
                    elapsed_seconds=max(0.0, now - item.started_monotonic),
                    cancellation_requested=item.cancellation_token.cancelled,
                    thread_id=item.thread_id,
                )
                for item in operations
            )

    def cancel_operation(
        self,
        operation_id: str,
        *,
        requester_owner: str | None = None,
        reason: str = "Operation cancelled.",
    ) -> bool:
        with self._condition:
            active = self._active.get(operation_id)
            waiter = next((item for item in self._waiters if item.operation_id == operation_id), None)
            target = active or waiter
            if target is None:
                return False
            if requester_owner is not None and target.owner != requester_owner:
                raise OperationOwnershipError(
                    f"Operation {operation_id} belongs to {target.owner!r}, not {requester_owner!r}."
                )
            changed = target.cancellation_token.cancel(reason)
            self._condition.notify_all()
            return changed

    def cancel_owner(self, owner: str, reason: str = "Owner cancelled all model operations.") -> int:
        owner_name = str(owner or "").strip()
        cancelled = 0
        with self._condition:
            targets = [item for item in self._active.values() if item.owner == owner_name]
            targets.extend(item for item in self._waiters if item.owner == owner_name)
            seen_tokens: set[int] = set()
            for target in targets:
                token_id = id(target.cancellation_token)
                if token_id in seen_tokens:
                    continue
                seen_tokens.add(token_id)
                if target.cancellation_token.cancel(reason):
                    cancelled += 1
            self._condition.notify_all()
        return cancelled

    def shutdown(
        self,
        *,
        cancel_active: bool = True,
        wait: bool = False,
        timeout: float = 5.0,
    ) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            self._shutdown = True
            if cancel_active:
                for operation in self._active.values():
                    operation.cancellation_token.cancel("Selene is shutting down.")
                for waiter in self._waiters:
                    waiter.cancellation_token.cancel("Selene is shutting down.")
            self._condition.notify_all()
            while wait and self._active and time.monotonic() < deadline:
                self._condition.wait(timeout=min(0.1, deadline - time.monotonic()))
            return not self._active


_DEFAULT_COORDINATOR: OllamaCoordinator | None = None
_DEFAULT_COORDINATOR_LOCK = threading.Lock()


def get_ollama_coordinator(config: RuntimeConfig | None = None) -> OllamaCoordinator:
    """Return the process-wide coordinator, created lazily exactly once."""
    global _DEFAULT_COORDINATOR
    with _DEFAULT_COORDINATOR_LOCK:
        if _DEFAULT_COORDINATOR is None:
            _DEFAULT_COORDINATOR = OllamaCoordinator(config or get_runtime_config())
        return _DEFAULT_COORDINATOR


@dataclass(frozen=True)
class OllamaProbeStatus:
    cli_installed: bool
    api_available: bool
    executable: str | None
    reason: str
    model_available: bool | None = None


def _plain_response(response: Any) -> Any:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return response


def _safe_error_text(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text[:500] or type(exc).__name__


def _estimate_local_tokens(value: Any) -> int:
    """Conservative tokenizer-free estimate for final service-level guards."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    ascii_count = sum(ord(character) < 128 for character in text)
    return ascii_count // 4 + (len(text) - ascii_count) + 1


def _guard_chat_options(
    messages: Any,
    tools: Any,
    options: Mapping[str, Any],
) -> dict[str, Any]:
    """Cap output headroom and reject prompts that cannot fit at API time."""
    guarded = dict(options)
    try:
        num_ctx = int(guarded["num_ctx"])
        requested = int(guarded["num_predict"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OllamaRuntimeError("Ollama chat options require integer num_ctx and num_predict values.") from exc
    if num_ctx < 1024 or requested < 1:
        raise OllamaRuntimeError("Ollama chat context and output limits are invalid.")

    prompt_tokens = _estimate_local_tokens(messages or []) + _estimate_local_tokens(tools or [])
    # Image encodings are not represented by their file-path string. Reserve a
    # conservative fixed budget for each image so vision calls fail controlled
    # on very small manual contexts instead of silently spilling.
    image_count = 0
    for message in messages or []:
        if isinstance(message, Mapping):
            images = message.get("images")
            if isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
                image_count += len(images)
    prompt_tokens += image_count * 1024
    safety_margin = max(256, int(num_ctx * 0.08))
    available = num_ctx - prompt_tokens - safety_margin
    if available < 96:
        raise OllamaContextOverflow(
            "The model request cannot fit in the configured context after prompt, tool, image, and safety overhead."
        )
    guarded["num_predict"] = min(requested, available)
    return guarded


def _translate_ollama_error(exc: BaseException, action: str) -> OllamaRuntimeError:
    if isinstance(exc, OllamaRuntimeError):
        return exc
    if isinstance(exc, CoordinatorError):
        return OllamaRuntimeError(str(exc))
    status_code = getattr(exc, "status_code", None)
    if status_code == 404:
        return OllamaModelMissingError(f"Ollama model was not found while attempting to {action}.")
    class_name = type(exc).__name__.lower()
    detail = _safe_error_text(exc)
    if "timeout" in class_name or "timed out" in detail.lower():
        return OllamaRequestTimeout(f"Ollama timed out while attempting to {action}: {detail}")
    unavailable_markers = (
        "connection refused",
        "failed to connect",
        "connecterror",
        "name or service not known",
        "no connection could be made",
    )
    if isinstance(exc, (ConnectionError, ModuleNotFoundError)) or any(
        marker in detail.lower() for marker in unavailable_markers
    ):
        return OllamaUnavailableError(f"Local Ollama is unavailable while attempting to {action}: {detail}")
    return OllamaRuntimeError(f"Ollama failed while attempting to {action}: {detail}")


class OllamaService:
    """Coordinated wrapper around the official local Ollama Python client."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        coordinator: OllamaCoordinator | None = None,
        host: str | None = None,
        client_factory: Callable[[float], Any] | None = None,
    ) -> None:
        self.config = config or (coordinator.config if coordinator else get_runtime_config())
        self.coordinator = coordinator or get_ollama_coordinator(self.config)
        self.host = host
        self._client_factory = client_factory

    def _client(self, timeout: float) -> Any:
        if self._client_factory is not None:
            return self._client_factory(timeout)
        try:
            import ollama
        except ImportError as exc:
            raise OllamaUnavailableError("The Ollama Python client is not installed.") from exc
        kwargs: dict[str, Any] = {"timeout": timeout}
        if self.host:
            kwargs["host"] = self.host
        return ollama.Client(**kwargs)

    def probe(self, model: str | None = None, timeout: float = 3.0) -> OllamaProbeStatus:
        executable = shutil.which("ollama")
        try:
            client = self._client(max(0.2, min(float(timeout), 10.0)))
            client.list()
        except Exception as exc:
            error = _translate_ollama_error(exc, "probe the local API")
            return OllamaProbeStatus(
                cli_installed=executable is not None,
                api_available=False,
                executable=executable,
                reason=str(error),
                model_available=None,
            )
        model_available = None
        if model:
            try:
                response = client.show(str(model).strip())
                if response is None:
                    raise OllamaMalformedResponse(
                        "Ollama returned no model metadata during the availability probe."
                    )
                model_available = True
            except Exception as exc:
                error = _translate_ollama_error(exc, f"inspect model {model!r}")
                if isinstance(error, OllamaModelMissingError):
                    model_available = False
                else:
                    # Ollama may disappear between list and show. Report the
                    # latest API state instead of leaking a connection error
                    # through startup or the web settings endpoint.
                    return OllamaProbeStatus(
                        cli_installed=executable is not None,
                        api_available=False,
                        executable=executable,
                        reason=str(error),
                        model_available=None,
                    )
        reason = "Local Ollama API is available."
        if model and not model_available:
            reason = f"Local Ollama API is available, but model {model!r} is missing."
        return OllamaProbeStatus(
            cli_installed=executable is not None,
            api_available=True,
            executable=executable,
            reason=reason,
            model_available=model_available,
        )

    def model_exists(self, model: str, timeout: float = 5.0) -> bool:
        try:
            self.show_model(model, timeout=timeout)
            return True
        except OllamaModelMissingError:
            return False

    def show_model(self, model: str, timeout: float = 5.0) -> Any:
        model_name = str(model or "").strip()
        if not model_name:
            raise OllamaRuntimeError("A model name is required.")
        try:
            return self._client(max(0.2, min(float(timeout), 60.0))).show(model_name)
        except Exception as exc:
            raise _translate_ollama_error(exc, f"inspect model {model_name!r}") from exc

    def list_models(self, timeout: float = 5.0) -> tuple[str, ...]:
        try:
            response = _plain_response(self._client(max(0.2, min(float(timeout), 60.0))).list())
        except Exception as exc:
            raise _translate_ollama_error(exc, "list local models") from exc
        models = response.get("models") if isinstance(response, Mapping) else getattr(response, "models", None)
        if models is None:
            raise OllamaMalformedResponse("Ollama model-list response did not contain a models field.")
        names: list[str] = []
        for item in models:
            plain = _plain_response(item)
            if isinstance(plain, Mapping):
                name = plain.get("model") or plain.get("name")
            else:
                name = getattr(item, "model", None) or getattr(item, "name", None)
            if name:
                names.append(str(name))
        return tuple(names)

    def chat(
        self,
        *,
        kind: OperationKind | str,
        owner: str,
        cancellation_token: CancellationToken | None = None,
        wait_timeout: float | None = None,
        operation_timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        operation_kind = self.coordinator._coerce_kind(kind)
        if operation_kind is OperationKind.EMBEDDING:
            raise OllamaRuntimeError("Use OllamaService.embed for embedding operations.")
        default_model = self.config.vision_model if operation_kind is OperationKind.VISION else self.config.chat_model
        model = str(kwargs.pop("model", None) or default_model)
        stream = bool(kwargs.get("stream", False))
        options = self.config.ollama_options()
        options.update(dict(kwargs.pop("options", {}) or {}))
        options = _guard_chat_options(
            kwargs.get("messages"),
            kwargs.get("tools"),
            options,
        )
        kwargs.update({"model": model, "options": options})
        kwargs.setdefault("keep_alive", self.config.keep_alive)
        if stream:
            return self._chat_stream(
                operation_kind,
                owner,
                model,
                kwargs,
                cancellation_token,
                wait_timeout,
                operation_timeout,
            )
        return self._chat_once(
            operation_kind,
            owner,
            model,
            kwargs,
            cancellation_token,
            wait_timeout,
            operation_timeout,
        )

    def _chat_once(
        self,
        kind: OperationKind,
        owner: str,
        model: str,
        kwargs: Mapping[str, Any],
        token: CancellationToken | None,
        wait_timeout: float | None,
        operation_timeout: float | None,
    ) -> Any:
        try:
            with self.coordinator.operation(
                kind,
                owner=owner,
                model=model,
                cancellation_token=token,
                wait_timeout=wait_timeout,
                operation_timeout=operation_timeout,
            ) as lease:
                response = self._client(lease.remaining_seconds()).chat(**dict(kwargs))
                lease.checkpoint()
                return response
        except (OperationCancelled, CoordinatorError, OllamaRuntimeError):
            raise
        except Exception as exc:
            raise _translate_ollama_error(exc, f"run {kind.value} inference") from exc

    def _chat_stream(
        self,
        kind: OperationKind,
        owner: str,
        model: str,
        kwargs: Mapping[str, Any],
        token: CancellationToken | None,
        wait_timeout: float | None,
        operation_timeout: float | None,
    ) -> Iterator[Any]:
        iterator: Iterator[Any] | None = None
        try:
            with self.coordinator.operation(
                kind,
                owner=owner,
                model=model,
                cancellation_token=token,
                wait_timeout=wait_timeout,
                operation_timeout=operation_timeout,
            ) as lease:
                iterator = iter(self._client(lease.remaining_seconds()).chat(**dict(kwargs)))
                for chunk in iterator:
                    lease.checkpoint()
                    yield chunk
        except (GeneratorExit, OperationCancelled, CoordinatorError, OllamaRuntimeError):
            raise
        except Exception as exc:
            raise _translate_ollama_error(exc, f"stream {kind.value} inference") from exc
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

    def embed(
        self,
        input: str | Sequence[str],
        *,
        owner: str,
        model: str | None = None,
        cancellation_token: CancellationToken | None = None,
        wait_timeout: float | None = None,
        operation_timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        model_name = str(model or self.config.embedding_model)
        values = [input] if isinstance(input, str) else list(input)
        if not values:
            raise OllamaRuntimeError("At least one embedding input is required.")
        if len(values) > 128:
            raise OllamaRuntimeError("Embedding requests are limited to 128 inputs per coordinated batch.")
        limit = self.config.num_ctx - max(256, int(self.config.num_ctx * 0.08))
        for index, value in enumerate(values):
            if _estimate_local_tokens(str(value)) > limit:
                raise OllamaContextOverflow(
                    f"Embedding input {index} cannot fit in the configured {self.config.num_ctx}-token context."
                )
        embed_options = {
            "num_ctx": self.config.num_ctx,
            "num_batch": self.config.num_batch,
        }
        embed_options.update(dict(kwargs.pop("options", {}) or {}))
        try:
            with self.coordinator.operation(
                OperationKind.EMBEDDING,
                owner=owner,
                model=model_name,
                cancellation_token=cancellation_token,
                wait_timeout=wait_timeout,
                operation_timeout=operation_timeout,
            ) as lease:
                response = self._client(lease.remaining_seconds()).embed(
                    model=model_name,
                    input=input,
                    options=embed_options,
                    keep_alive=self.config.keep_alive,
                    **kwargs,
                )
                lease.checkpoint()
                return response
        except (OperationCancelled, CoordinatorError, OllamaRuntimeError):
            raise
        except Exception as exc:
            raise _translate_ollama_error(exc, "generate embeddings") from exc

    def install_model_staged(
        self,
        *,
        model: str,
        staging_model: str,
        base_model: str,
        system_prompt: str,
        parameters: Mapping[str, Any],
        owner: str = "startup:model-build",
        cancellation_token: CancellationToken | None = None,
        operation_timeout: float | None = None,
    ) -> Any:
        """Build and verify a staging model before replacing the live alias.

        Ollama publishes the target alias only after the staging manifest is
        complete and inspectable. A failed or interrupted build therefore
        leaves an existing working ``model`` untouched.
        """
        target = str(model or "").strip()
        staging = str(staging_model or "").strip()
        base = str(base_model or "").strip()
        if not target or not staging or not base or staging == target:
            raise OllamaRuntimeError("Target, distinct staging, and base model names are required.")

        iterator: Iterator[Any] | None = None
        client: Any | None = None
        staging_deleted = False
        try:
            with self.coordinator.operation(
                OperationKind.BUILD,
                owner=owner,
                model=staging,
                cancellation_token=cancellation_token,
                operation_timeout=operation_timeout,
            ) as lease:
                client = self._client(lease.remaining_seconds())
                iterator = iter(client.create(
                    model=staging,
                    from_=base,
                    system=system_prompt or None,
                    parameters=dict(parameters),
                    stream=True,
                ))
                for _progress in iterator:
                    lease.checkpoint()

                # Inspection is the build-success boundary. Never publish an
                # uninspectable staging manifest over a working target alias.
                staged_response = client.show(staging)
                if staged_response is None:
                    raise OllamaMalformedResponse("Ollama returned no staging-model metadata after creation.")
                lease.checkpoint()
                client.copy(source=staging, destination=target)
                installed_response = client.show(target)
                if installed_response is None:
                    raise OllamaMalformedResponse("Ollama returned no target-model metadata after installation.")
                lease.checkpoint()
                try:
                    client.delete(staging)
                    staging_deleted = True
                except Exception:
                    # The unique Selene-owned staging alias is harmless if the
                    # local API refuses cleanup; target publication succeeded.
                    pass
                return installed_response
        except (OperationCancelled, CoordinatorError, OllamaRuntimeError):
            raise
        except Exception as exc:
            raise _translate_ollama_error(exc, f"build model {target!r}") from exc
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
            if client is not None and not staging_deleted:
                try:
                    client.delete(staging)
                except Exception:
                    # This name is unique and Selene-owned. A later cleanup pass
                    # may remove it; never risk the live target during failure.
                    pass


def normalized_modelfile_text(content: str) -> str:
    """Normalize only line endings/trailing whitespace for stable hashing."""
    text = str(content).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip() + "\n"


def modelfile_sha256(path: str | os.PathLike[str]) -> str:
    try:
        content = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise InvalidModelfileError(f"Could not read Modelfile {path!s}: {type(exc).__name__}") from exc
    return hashlib.sha256(normalized_modelfile_text(content).encode("utf-8")).hexdigest()


def _parameter_value(raw_value: str) -> Any:
    value = raw_value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


@dataclass(frozen=True)
class ParsedModelfile:
    path: Path
    sha256: str
    base_model: str
    system_prompt: str
    parameters: Mapping[str, Any]


_SYSTEM_BLOCK_RE = re.compile(r'(?ms)^\s*SYSTEM\s+"""(.*?)"""\s*')


def parse_modelfile(path: str | os.PathLike[str]) -> ParsedModelfile:
    """Parse Selene's supported FROM/SYSTEM/PARAMETER Modelfile subset."""
    modelfile_path = Path(path)
    try:
        content = modelfile_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise InvalidModelfileError(
            f"Could not read Modelfile {modelfile_path}: {type(exc).__name__}"
        ) from exc
    normalized = normalized_modelfile_text(content)
    system_matches = list(_SYSTEM_BLOCK_RE.finditer(normalized))
    if len(system_matches) > 1 or (re.search(r"(?m)^\s*SYSTEM\b", normalized) and not system_matches):
        raise InvalidModelfileError("Modelfile must contain at most one complete triple-quoted SYSTEM block.")
    system_prompt = system_matches[0].group(1).strip() if system_matches else ""
    directives = _SYSTEM_BLOCK_RE.sub("\n", normalized)

    base_model = ""
    parameters: dict[str, Any] = {}
    for line_number, line in enumerate(directives.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            parts = shlex.split(stripped, posix=True)
        except ValueError as exc:
            raise InvalidModelfileError(f"Invalid quoting on Modelfile line {line_number}: {exc}") from exc
        directive = parts[0].upper()
        if directive == "FROM" and len(parts) == 2:
            if base_model:
                raise InvalidModelfileError("Modelfile contains more than one FROM directive.")
            base_model = parts[1]
        elif directive == "PARAMETER" and len(parts) >= 3:
            name = parts[1]
            value = _parameter_value(" ".join(parts[2:]))
            if name in parameters:
                existing = parameters[name]
                parameters[name] = [*existing, value] if isinstance(existing, list) else [existing, value]
            else:
                parameters[name] = value
        else:
            raise InvalidModelfileError(
                f"Unsupported or malformed {directive} directive on Modelfile line {line_number}."
            )
    if not base_model:
        raise InvalidModelfileError("Modelfile is missing a valid FROM directive.")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return ParsedModelfile(
        path=modelfile_path,
        sha256=digest,
        base_model=base_model,
        system_prompt=system_prompt,
        parameters=parameters,
    )


@dataclass(frozen=True)
class ModelBuildRecord:
    schema_version: int
    model: str
    base_model: str
    modelfile_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "base_model": self.base_model,
            "modelfile_sha256": self.modelfile_sha256,
        }


def model_build_record(model: str, modelfile_path: str | os.PathLike[str]) -> ModelBuildRecord:
    parsed = parse_modelfile(modelfile_path)
    return ModelBuildRecord(
        schema_version=1,
        model=str(model).strip(),
        base_model=parsed.base_model,
        modelfile_sha256=parsed.sha256,
    )


def stale_model_reason(
    metadata: Mapping[str, Any] | ModelBuildRecord | None,
    *,
    model: str,
    modelfile_path: str | os.PathLike[str],
) -> str | None:
    """Return a deterministic stale reason, or ``None`` when hashes match."""
    expected = model_build_record(model, modelfile_path)
    if metadata is None:
        return "No model-build metadata exists for the current Modelfile."
    values = metadata.as_dict() if isinstance(metadata, ModelBuildRecord) else dict(metadata)
    if values.get("schema_version") != expected.schema_version:
        return "Model-build metadata schema is missing or unsupported."
    if values.get("model") != expected.model:
        return "Model-build metadata belongs to a different model name."
    if values.get("base_model") != expected.base_model:
        return "The Modelfile base model changed since the last successful build."
    if values.get("modelfile_sha256") != expected.modelfile_sha256:
        return "The Modelfile changed since the last successful build."
    return None


def is_model_stale(
    metadata: Mapping[str, Any] | ModelBuildRecord | None,
    *,
    model: str,
    modelfile_path: str | os.PathLike[str],
) -> bool:
    return stale_model_reason(metadata, model=model, modelfile_path=modelfile_path) is not None
