"""Session-view and generation-ownership primitives for the web runtime."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import re
import threading
import time
import uuid

from agent.cancellation import CancellationToken


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_NEW_SESSION_ALIASES = frozenset({"", "Active Session", "New conversation"})
LEGACY_CLIENT_ID = "legacy-client"


def normalize_runtime_id(value: object, *, fallback: str | None = None) -> str:
    """Validate an opaque client/generation identifier used for ownership checks."""
    text = str(value or "").strip()
    if not text and fallback is not None:
        return fallback
    if not _IDENTIFIER.fullmatch(text):
        raise ValueError("Runtime identifiers must be 1-128 safe ASCII characters")
    return text


def generation_session_key(session_name: str, client_id: str) -> str:
    """Return a stable lock key; unsaved chats are isolated per browser tab."""
    if session_name in _NEW_SESSION_ALIASES:
        return f"unsaved:{client_id}"
    return f"saved:{session_name}"


class TerminalState(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class GenerationConflict(RuntimeError):
    """Raised when a session already owns an active generation."""


class GenerationOwnershipError(RuntimeError):
    """Raised when a client tries to cancel another client's generation."""


@dataclass
class GenerationLease:
    generation_id: str
    client_id: str
    session_key: str
    session_name: str
    token: CancellationToken
    started_at: float


@dataclass(frozen=True)
class GenerationRecord:
    generation_id: str
    client_id: str
    session_key: str
    session_name: str
    state: TerminalState
    started_at: float
    finished_at: float
    detail: str | None = None


class GenerationRegistry:
    """Own one active generation per session and retain bounded terminal records."""

    def __init__(self, *, terminal_history: int = 128) -> None:
        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)
        self._active_by_session: dict[str, GenerationLease] = {}
        self._active_by_id: dict[str, GenerationLease] = {}
        self._terminal: dict[str, GenerationRecord] = {}
        self._terminal_order: deque[str] = deque()
        self._terminal_history = max(1, int(terminal_history))

    def begin(
        self,
        session_name: str,
        client_id: str,
        generation_id: str | None = None,
    ) -> GenerationLease:
        client_id = normalize_runtime_id(client_id, fallback=LEGACY_CLIENT_ID)
        generation_id = normalize_runtime_id(generation_id, fallback=str(uuid.uuid4()))
        session_key = generation_session_key(session_name, client_id)
        with self._lock:
            existing = self._active_by_session.get(session_key)
            if existing is not None:
                raise GenerationConflict(
                    f"Session already has an active generation ({existing.generation_id})"
                )
            if generation_id in self._active_by_id or generation_id in self._terminal:
                raise GenerationConflict(f"Generation id is already in use ({generation_id})")
            lease = GenerationLease(
                generation_id=generation_id,
                client_id=client_id,
                session_key=session_key,
                session_name=session_name,
                token=CancellationToken(),
                started_at=time.monotonic(),
            )
            self._active_by_session[session_key] = lease
            self._active_by_id[generation_id] = lease
            return lease

    def cancel(
        self,
        generation_id: str,
        client_id: str,
        *,
        reason: str = "Cancelled by the requesting client",
    ) -> GenerationLease:
        generation_id = normalize_runtime_id(generation_id)
        client_id = normalize_runtime_id(client_id, fallback=LEGACY_CLIENT_ID)
        with self._lock:
            lease = self._active_by_id.get(generation_id)
            if lease is None:
                terminal = self._terminal.get(generation_id)
                if terminal and terminal.client_id != client_id:
                    raise GenerationOwnershipError("Generation belongs to another client")
                raise KeyError("Generation is not active")
            if lease.client_id != client_id:
                raise GenerationOwnershipError("Generation belongs to another client")
            lease.token.cancel(reason)
            return lease

    def rebind(self, lease: GenerationLease, session_name: str) -> GenerationLease:
        """Move an active lease from an unsaved-chat key to its persisted name.

        The first chat write assigns a real filename.  Rebinding closes the
        small window where another tab could otherwise start work against that
        file while the original generation was still keyed as ``unsaved``.
        """
        new_key = generation_session_key(session_name, lease.client_id)
        with self._lock:
            active = self._active_by_id.get(lease.generation_id)
            if active is not lease:
                raise GenerationOwnershipError("Generation lease is no longer active")
            existing = self._active_by_session.get(new_key)
            if existing is not None and existing is not lease:
                raise GenerationConflict(
                    f"Session already has an active generation ({existing.generation_id})"
                )
            if self._active_by_session.get(lease.session_key) is lease:
                self._active_by_session.pop(lease.session_key, None)
            lease.session_key = new_key
            lease.session_name = session_name
            self._active_by_session[new_key] = lease
            return lease

    def rebind_generation(
        self,
        generation_id: str,
        client_id: str,
        session_name: str,
    ) -> GenerationLease:
        """Rebind an active id after checking browser-tab ownership."""
        generation_id = normalize_runtime_id(generation_id)
        client_id = normalize_runtime_id(client_id, fallback=LEGACY_CLIENT_ID)
        with self._lock:
            lease = self._active_by_id.get(generation_id)
            if lease is None:
                raise KeyError("Generation is not active")
            if lease.client_id != client_id:
                raise GenerationOwnershipError("Generation belongs to another client")
            return self.rebind(lease, session_name)

    def finish(
        self,
        lease: GenerationLease,
        state: TerminalState,
        detail: str | None = None,
    ) -> GenerationRecord:
        """Atomically release a lease; repeated calls return its first terminal state."""
        with self._lock:
            existing = self._terminal.get(lease.generation_id)
            if existing is not None:
                return existing
            active = self._active_by_id.get(lease.generation_id)
            if active is not lease:
                raise GenerationOwnershipError("Generation lease is no longer active")
            self._active_by_id.pop(lease.generation_id, None)
            if self._active_by_session.get(lease.session_key) is lease:
                self._active_by_session.pop(lease.session_key, None)
            record = GenerationRecord(
                generation_id=lease.generation_id,
                client_id=lease.client_id,
                session_key=lease.session_key,
                session_name=lease.session_name,
                state=state,
                started_at=lease.started_at,
                finished_at=time.monotonic(),
                detail=detail,
            )
            self._terminal[lease.generation_id] = record
            self._terminal_order.append(lease.generation_id)
            while len(self._terminal_order) > self._terminal_history:
                expired = self._terminal_order.popleft()
                self._terminal.pop(expired, None)
            self._idle.notify_all()
            return record

    def get_terminal(self, generation_id: str) -> GenerationRecord | None:
        with self._lock:
            return self._terminal.get(generation_id)

    def active_operations(self) -> list[dict]:
        with self._lock:
            operations = [
                {
                    "generation_id": lease.generation_id,
                    "client_id": lease.client_id,
                    "session_name": lease.session_name,
                    "cancel_requested": lease.token.cancelled,
                }
                for lease in self._active_by_id.values()
            ]
        return sorted(operations, key=lambda item: item["generation_id"])

    def active_for_session(self, session_name: str, client_id: str) -> GenerationLease | None:
        """Return the active owner for a saved or client-owned unsaved session."""
        key = generation_session_key(session_name, client_id)
        with self._lock:
            return self._active_by_session.get(key)

    def wait_for_session_idle(
        self,
        session_name: str,
        client_id: str,
        timeout: float,
    ) -> bool:
        """Wait a bounded time for a saved or client-owned unsaved session to release."""
        key = generation_session_key(session_name, client_id)
        deadline = time.monotonic() + max(0.0, timeout)
        with self._idle:
            while key in self._active_by_session:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
            return True

    def cancel_all(self, reason: str = "Selene is shutting down") -> list[GenerationLease]:
        with self._lock:
            leases = list(self._active_by_id.values())
        for lease in leases:
            lease.token.cancel(reason)
        return leases


@dataclass(frozen=True)
class SessionView:
    active_session_name: str
    session: dict
    history: list[dict]


class ClientSessionStore:
    """Keep browser-tab selection/settings isolated while generations use snapshots."""

    def __init__(self, default_session: dict) -> None:
        self._lock = threading.RLock()
        self._default_session = deepcopy(default_session)
        self._views: dict[str, SessionView] = {}

    def set_default_session(self, session: dict) -> None:
        with self._lock:
            self._default_session = deepcopy(session)

    def snapshot(self, client_id: str) -> SessionView:
        client_id = normalize_runtime_id(client_id, fallback=LEGACY_CLIENT_ID)
        with self._lock:
            view = self._views.get(client_id)
            if view is None:
                view = SessionView("Active Session", deepcopy(self._default_session), [])
                self._views[client_id] = view
            return deepcopy(view)

    def update_settings(
        self,
        client_id: str,
        settings: dict,
        *,
        history: list[dict] | None = None,
        expected_history: list[dict] | None = None,
    ) -> SessionView:
        """Replace a tab's settings, optionally swapping its history too.

        ``history`` is applied only when ``expected_history`` still matches what
        the store holds. A generation that committed since the caller took its
        snapshot therefore wins, instead of being silently overwritten by a
        rewrite computed from stale messages.
        """
        client_id = normalize_runtime_id(client_id, fallback=LEGACY_CLIENT_ID)
        with self._lock:
            current = self._views.get(client_id)
            if current is None:
                current = SessionView("Active Session", deepcopy(self._default_session), [])
            replacement_history = current.history
            if history is not None and (
                expected_history is None or current.history == expected_history
            ):
                replacement_history = history
            replacement = SessionView(
                current.active_session_name,
                deepcopy(settings),
                deepcopy(replacement_history),
            )
            self._views[client_id] = replacement
        return deepcopy(replacement)

    def select(
        self,
        client_id: str,
        session_name: str,
        session: dict,
        history: list[dict],
    ) -> SessionView:
        client_id = normalize_runtime_id(client_id, fallback=LEGACY_CLIENT_ID)
        replacement = SessionView(session_name, deepcopy(session), deepcopy(history))
        with self._lock:
            self._views[client_id] = replacement
        return deepcopy(replacement)

    def new_session(self, client_id: str) -> SessionView:
        client_id = normalize_runtime_id(client_id, fallback=LEGACY_CLIENT_ID)
        with self._lock:
            current = self._views.get(client_id)
            settings = current.session if current is not None else self._default_session
            replacement = SessionView("Active Session", deepcopy(settings), [])
            self._views[client_id] = replacement
            return deepcopy(replacement)

    def commit_generation(
        self,
        client_id: str,
        expected_session_name: str,
        new_session_name: str,
        session: dict,
        history: list[dict],
        generation_start_session: dict | None = None,
    ) -> bool:
        """Publish a completed snapshot only if the tab still views its origin."""
        client_id = normalize_runtime_id(client_id, fallback=LEGACY_CLIENT_ID)
        with self._lock:
            current = self._views.get(client_id)
            if current is None:
                current = SessionView("Active Session", deepcopy(self._default_session), [])
            same_origin = current.active_session_name == expected_session_name
            if current.active_session_name in _NEW_SESSION_ALIASES and expected_session_name in _NEW_SESSION_ALIASES:
                same_origin = True
            if not same_origin:
                return False
            committed_session = session
            if generation_start_session is not None and current.session != generation_start_session:
                # Settings may be edited while inference is running. Preserve the
                # newer tab-owned settings instead of letting an old snapshot win.
                committed_session = current.session
            self._views[client_id] = SessionView(
                new_session_name,
                deepcopy(committed_session),
                deepcopy(history),
            )
            return True

    def remove_session(self, session_name: str) -> None:
        with self._lock:
            for client_id, view in list(self._views.items()):
                if view.active_session_name == session_name:
                    self._views[client_id] = SessionView(
                        "Active Session", deepcopy(view.session), []
                    )

    def rename_session(self, session_name: str, new_session_name: str) -> None:
        """Repoint every tab viewing *session_name* at its new filename.

        Unlike remove_session this keeps each view's session and history: only
        the identity changed, so a tab reading the conversation must keep
        reading it rather than being reset to a blank chat.
        """
        with self._lock:
            for client_id, view in list(self._views.items()):
                if view.active_session_name == session_name:
                    self._views[client_id] = SessionView(
                        new_session_name,
                        deepcopy(view.session),
                        deepcopy(view.history),
                    )
