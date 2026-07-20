#!/usr/bin/env python3
"""
agent/web.py — Web Server for the Selene AI Agent UI.

Serves static frontend files and exposes HTTP endpoints / SSE streaming for interaction.
"""

import http.server
import socketserver
import json
import os
import glob
import hmac
import sys
import time
import socket
import re
import signal
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

# Import agent configurations and helpers from core
from agent.core import (
    CONTEXT_TOOL_LOOP_RESERVE,
    ContextWindowError,
    MODEL_NAME,
    MAX_OUTPUT_CONTINUATION_ROUNDS,
    OUTPUT_CONTINUATION_PROMPT,
    _check_and_compact_history,
    _automatic_vault_index_tool_call,
    _is_progressing_vault_index_round,
    _new_vault_index_loop_state,
    _should_reexecute_turn_duplicate,
    _tool_loop_stop_message,
    _update_vault_index_loop_state,
    build_tool_continuation_prompt,
    effective_session_model_options,
    guarded_options_for_call,
    load_default_system_prompt,
    prepare_messages_for_model,
    tool_schemas_for_model,
    _tool_call_turn_key,
    _chunk_done_reason,
    _output_limit_reached,
)
from agent.tool_runner import (
    build_execution_batches,
    execute_tool_call,
    execute_tool_calls,
    normalize_tool_calls,
    shutdown_tool_runner,
)
from agent.cancellation import CancellationToken, OperationCancelled
from agent.ollama_runtime import OllamaService, OperationKind, get_ollama_coordinator
from agent.persistence import PersistenceError, atomic_write_json, read_json_preserved
from agent.platform_runtime import get_runtime_paths, open_url_native, resource_path
from agent.runtime_config import (
    RuntimeConfigurationError,
    RuntimeConfig,
    get_runtime_config,
)
from agent.web_runtime import (
    ClientSessionStore,
    GenerationConflict,
    GenerationOwnershipError,
    GenerationRegistry,
    LEGACY_CLIENT_ID,
    TerminalState,
    normalize_runtime_id,
)
from agent.modes import (
    AGENT_MODE_DEEP_RESEARCH,
    AGENT_MODE_NORMAL,
    AGENT_MODE_ULTRA,
    DEEP_RESEARCH_COMPACT_INTERVAL,
    DEEP_RESEARCH_SCRAPE_COMPACT_INTERVAL,
    DEEP_RESEARCH_PLANNER_PROMPT,
    DEEP_RESEARCH_SYNTHESIS_PROMPT,
    ULTRA_MODE_PROMPT,
    ULTRA_REVIEW_PROMPT,
    compact_deep_research_messages,
    force_hard_web_search_schema,
    force_high_tool_difficulty,
    normalize_agent_mode,
    parse_research_queries,
    research_query_count,
    tool_call_round_signature,
)
from tools.registry import TOOL_DISPATCH, TOOL_SCHEMAS, get_tool_metadata

# Setup directories
STATIC_DIR = str(resource_path("agent/static"))
_SESSIONS_DIR = str(get_runtime_paths().data_dir / "sessions")


def _session_from_runtime(runtime: RuntimeConfig) -> dict:
    return {
        # Explicit per-session overrides only. Effective values are resolved
        # from the selected profile for every model request.
        "options": {},
        "runtime_profile": runtime.requested_profile.value,
        "verbose": True,
        "wordwrap": True,
        "system": "",
        "history": True,
        "format": "",
        "think": True,
        "agent_mode": AGENT_MODE_NORMAL,
    }


_BASE_RUNTIME_CONFIG = get_runtime_config()

# Global Application State
GLOBAL_STATE = {
    "history": [],
    "session": _session_from_runtime(_BASE_RUNTIME_CONFIG),
    "active_session_name": "Active Session"
}
_GLOBAL_STATE_LOCK = threading.RLock()
_SESSION_LIFECYCLE_LOCK = threading.RLock()
_SESSION_LOCKS_GUARD = threading.Lock()
_SESSION_LOCKS: dict[str, threading.RLock] = {}
_SESSION_RENAMES: dict[str, str] = {}
CLIENT_SESSIONS = ClientSessionStore(GLOBAL_STATE["session"])
ACTIVE_GENERATIONS = GenerationRegistry()


def _session_lock(filename: str) -> threading.RLock:
    safe_name = os.path.basename(filename)
    with _SESSION_LOCKS_GUARD:
        return _SESSION_LOCKS.setdefault(safe_name, threading.RLock())


def _resolved_session_filename(filename: str) -> str:
    safe_name = os.path.basename(filename)
    with _SESSION_LOCKS_GUARD:
        seen: set[str] = set()
        while safe_name in _SESSION_RENAMES and safe_name not in seen:
            seen.add(safe_name)
            safe_name = _SESSION_RENAMES[safe_name]
    return safe_name


def _normalize_session_settings(session: dict, *, fallback: dict | None = None) -> dict:
    if not isinstance(session, dict):
        raise RuntimeConfigurationError("Session settings must be a JSON object")
    base = deepcopy(fallback or GLOBAL_STATE.get("session") or _session_from_runtime(_BASE_RUNTIME_CONFIG))
    merged = {
        **base,
        **deepcopy(session),
        "options": {
            **base.get("options", {}),
            **deepcopy(session.get("options", {})),
        },
    }
    options = merged.get("options", {})
    if not isinstance(options, dict):
        raise RuntimeConfigurationError("Session options must be a JSON object")
    allowed_options = set(_BASE_RUNTIME_CONFIG.ollama_options())
    unknown_options = set(options) - allowed_options
    if unknown_options:
        raise RuntimeConfigurationError(
            f"Unknown model option(s): {', '.join(sorted(unknown_options))}"
        )
    runtime = get_runtime_config({**merged, "options": options})
    merged["options"] = deepcopy(options)
    merged["runtime_profile"] = runtime.requested_profile.value
    try:
        merged["agent_mode"] = normalize_agent_mode(merged.get("agent_mode"))
    except ValueError as exc:
        raise RuntimeConfigurationError(str(exc)) from exc
    for field in ("verbose", "wordwrap", "history", "think"):
        if not isinstance(merged.get(field), bool):
            raise RuntimeConfigurationError(f"{field} must be true or false")
    if not isinstance(merged.get("system", ""), str):
        raise RuntimeConfigurationError("system must be text")
    if merged.get("format") not in ("", "json"):
        raise RuntimeConfigurationError("format must be empty or 'json'")
    return merged


def _active_system_prompt(session: dict | None = None) -> tuple[str, str]:
    """Return (default, active) system prompts used for model calls and UI meters."""
    default_prompt = load_default_system_prompt() or ""
    override = str((session or {}).get("system") or "").strip()
    return default_prompt, (override or default_prompt)


def _runtime_payload(session: dict) -> dict:
    runtime = get_runtime_config(session)
    paths = get_runtime_paths()
    default_system_prompt, active_system_prompt = _active_system_prompt(session)
    return {
        "requested_profile": runtime.requested_profile.value,
        "profile": runtime.profile.value,
        "selection_reason": runtime.selection_reason,
        "warnings": list(runtime.warnings),
        "effective_options": runtime.ollama_options(),
        "storage": paths.report(),
        # Frontend context meter needs the prompt actually sent to the model.
        # Session ``system`` is only an override; empty means Modelfile default.
        "default_system_prompt": default_system_prompt,
        "active_system_prompt": active_system_prompt,
    }


def _read_session_snapshot(filename: str) -> tuple[dict, list[dict]]:
    requested_name = os.path.basename(filename)
    if not requested_name.endswith(".json"):
        raise FileNotFoundError("Session file not found")
    while True:
        safe_name = _resolved_session_filename(requested_name)
        filepath = os.path.join(_SESSIONS_DIR, safe_name)
        with _session_lock(safe_name):
            if _resolved_session_filename(requested_name) != safe_name:
                continue
            data = read_json_preserved(filepath, expected_type=dict)
            break
    session = data.get("session", {})
    history = data.get("history", [])
    if (
        not isinstance(session, dict)
        or not isinstance(history, list)
        or any(not isinstance(message, dict) for message in history)
    ):
        raise ValueError("Saved session has an invalid structure")
    normalized = _normalize_session_settings(
        session,
        fallback=_session_from_runtime(_BASE_RUNTIME_CONFIG),
    )
    return deepcopy(normalized), deepcopy(history)

# ── Session Management Functions ──────────────────────────────────────

def list_saved_sessions() -> list[str]:
    """Return a sorted list of session filenames (newest first).
    
    Returns:
        list[str]: A list of saved session filenames.
    """
    if not os.path.isdir(_SESSIONS_DIR):
        return []
    files = glob.glob(os.path.join(_SESSIONS_DIR, "*.json"))
    files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.basename(f) for f in files]


def save_session(
    name: str,
    session_data: dict | None = None,
    history_data: list[dict] | None = None,
    client_id: str = LEGACY_CLIENT_ID,
) -> str:
    """Persist current state to a JSON file.
    
    Args:
        name (str): An optional name for the session.
        
    Returns:
        str: The filename of the saved session.
    """
    os.makedirs(_SESSIONS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    unique_suffix = uuid.uuid4().hex[:8]
    if name:
        # Sanitize name
        safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
        filename = f"{safe_name}_{timestamp}_{unique_suffix}.json"
    else:
        filename = f"session_{timestamp}_{unique_suffix}.json"
        
    filepath = os.path.join(_SESSIONS_DIR, filename)
    if session_data is None or history_data is None:
        with _GLOBAL_STATE_LOCK:
            session_data = deepcopy(GLOBAL_STATE["session"])
            history_data = deepcopy(GLOBAL_STATE["history"])
    save_session_snapshot(filename, session_data, history_data)
    CLIENT_SESSIONS.select(client_id, filename, session_data, history_data)
    if client_id == LEGACY_CLIENT_ID:
        with _GLOBAL_STATE_LOCK:
            GLOBAL_STATE["active_session_name"] = filename
    return filename


def autosave_session(client_id: str = LEGACY_CLIENT_ID) -> str | None:
    """Create or update the active chat without requiring a manual save."""
    view = CLIENT_SESSIONS.snapshot(client_id)
    if not view.history:
        return None

    filename = view.active_session_name
    filepath = os.path.join(_SESSIONS_DIR, _resolved_session_filename(filename))
    if filename in ("", "Active Session", "New conversation") or not os.path.isfile(filepath):
        # Use a temporary name until the first response is complete and Selene
        # can name the conversation from its actual subject.
        filename = save_session_snapshot("", view.session, view.history)
        committed = CLIENT_SESSIONS.commit_generation(
            client_id,
            view.active_session_name,
            filename,
            view.session,
            view.history,
        )
        return filename if committed else None

    saved_name = save_session_snapshot(filename, view.session, view.history)
    if saved_name != filename:
        CLIENT_SESSIONS.commit_generation(
            client_id,
            filename,
            saved_name,
            view.session,
            view.history,
        )
    return saved_name


def save_session_snapshot(
    filename: str,
    session_data: dict,
    history_data: list[dict],
    *,
    generation_start_session: dict | None = None,
) -> str:
    """Persist one conversation without depending on the globally selected chat."""
    os.makedirs(_SESSIONS_DIR, exist_ok=True)
    safe_filename = os.path.basename(filename or "")
    if safe_filename in ("", "Active Session", "New conversation") or not safe_filename.endswith(".json"):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        safe_filename = f"session_{timestamp}_{uuid.uuid4().hex[:8]}.json"

    requested_filename = safe_filename
    while True:
        safe_filename = _resolved_session_filename(requested_filename)
        with _session_lock(safe_filename):
            # A title rename may have completed while this writer waited for
            # the old filename lock. Retry against the new name instead of
            # recreating the temporary session file.
            if _resolved_session_filename(requested_filename) != safe_filename:
                continue
            filepath = os.path.join(_SESSIONS_DIR, safe_filename)
            persisted_session = deepcopy(session_data)
            if generation_start_session is not None and os.path.isfile(filepath):
                try:
                    current = read_json_preserved(filepath, expected_type=dict)
                    current_session = current.get("session")
                    if (
                        isinstance(current_session, dict)
                        and current_session != generation_start_session
                    ):
                        persisted_session = current_session
                except (OSError, ValueError):
                    # The atomic writer below will preserve failures. Malformed
                    # existing data is surfaced rather than silently replaced.
                    raise
            payload = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "model": MODEL_NAME,
                "session": persisted_session,
                "history": deepcopy(history_data),
            }
            atomic_write_json(filepath, payload)
            return safe_filename


# Temporary first-turn files look like:
#   session_YYYYMMDD_HHMMSS.json
#   session_YYYYMMDD_HHMMSS_ffffff.json
#   session_YYYYMMDD_HHMMSS_ffffff_<8 hex>.json  (current uniqueness suffix)
# After the first completed reply they are renamed to Title_Words_<stamp>.json.
_TEMPORARY_SESSION_RE = re.compile(
    r"^session_"
    r"(?P<stamp>\d{8}_\d{6}(?:_\d{1,6})?)"
    r"(?:_[0-9a-f]{8})?"
    r"\.json$",
    re.IGNORECASE,
)


def _normalize_agent_title(value: str) -> str:
    """Constrain model output to a safe, human-readable 2-3 word title."""
    first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    first_line = re.sub(r"^(?:conversation\s+)?title\s*:\s*", "", first_line, flags=re.I)
    words = re.findall(r"[^\W_]+(?:['’][^\W_]+)?", first_line, flags=re.UNICODE)[:3]
    if len(words) == 1:
        words.append("Discussion" if words[0].casefold() == "chat" else "Chat")
    if len(words) < 2:
        return "New Conversation"
    return " ".join(words)


def is_temporary_session_filename(filename: str | None) -> bool:
    """Return True when *filename* is still awaiting agent title renaming."""
    return bool(_TEMPORARY_SESSION_RE.fullmatch(os.path.basename(filename or "")))


def generate_conversation_title(
    history: list[dict],
    *,
    session_data: dict | None = None,
    cancellation_token: CancellationToken | None = None,
    owner: str | None = None,
) -> str:
    """Ask the local agent for a short semantic title, with a stable fallback."""
    first_user = next(
        (str(message.get("content", "")) for message in history if message.get("role") == "user"),
        "",
    )
    last_assistant = next(
        (str(message.get("content", "")) for message in reversed(history) if message.get("role") == "assistant"),
        "",
    )
    if not first_user and not last_assistant:
        return "New Conversation"

    title_messages = [
        {
            "role": "system",
            "content": "Return only a descriptive two- or three-word conversation title without punctuation.",
        },
        {
            "role": "user",
            "content": f"User topic:\n{first_user[:1200]}\n\nAssistant response:\n{last_assistant[:1200]}",
        },
    ]
    try:
        runtime = get_runtime_config(session_data)
        response = OllamaService(runtime).chat(
            kind=OperationKind.TITLE,
            owner=owner or f"title:{threading.get_ident()}:{time.monotonic_ns()}",
            cancellation_token=cancellation_token,
            operation_timeout=runtime.title_timeout_seconds,
            messages=title_messages,
            stream=False,
            think=False,
            options={"temperature": 0.2, "num_predict": 12},
        )
        message = getattr(response, "message", None)
        content = getattr(message, "content", "")
        if isinstance(response, dict):
            content = response.get("message", {}).get("content", content)
        return _normalize_agent_title(str(content or ""))
    except Exception:
        return "New Conversation"


def title_temporary_session(
    history: list[dict],
    filename: str | None = None,
    cancellation_token: CancellationToken | None = None,
    *,
    session_data: dict | None = None,
    owner: str | None = None,
    generation_id: str | None = None,
    client_id: str = LEGACY_CLIENT_ID,
) -> str | None:
    """Replace a first-turn temporary filename with an agent-generated title."""
    filename = os.path.basename(filename or GLOBAL_STATE.get("active_session_name", ""))
    match = _TEMPORARY_SESSION_RE.fullmatch(filename)
    if not match:
        return None
    stamp = match.group("stamp")

    old_path = os.path.join(_SESSIONS_DIR, filename)
    if not os.path.isfile(old_path):
        return None

    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    title = generate_conversation_title(
        history,
        session_data=session_data,
        cancellation_token=cancellation_token,
        owner=owner,
    )
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    safe_title = "_".join(_normalize_agent_title(title).split())
    with _session_lock(filename):
        if not os.path.isfile(old_path):
            return None
        target = f"{safe_title}_{stamp}.json"
        target_path = os.path.join(_SESSIONS_DIR, target)
        suffix = 2
        while os.path.exists(target_path) and target_path != old_path:
            target = f"{safe_title}_{stamp}_{suffix}.json"
            target_path = os.path.join(_SESSIONS_DIR, target)
            suffix += 1
        rebound = False
        if generation_id:
            # Reserve the new session identity before making the renamed file
            # visible, closing the cross-tab generation race.
            ACTIVE_GENERATIONS.rebind_generation(generation_id, client_id, target)
            rebound = True
        try:
            os.replace(old_path, target_path)
        except BaseException:
            if rebound:
                ACTIVE_GENERATIONS.rebind_generation(generation_id, client_id, filename)
            raise
        with _SESSION_LOCKS_GUARD:
            _SESSION_RENAMES[filename] = target
    return target


def load_session(filename: str, client_id: str = LEGACY_CLIENT_ID) -> None:
    """Load session from a JSON file.
    
    Args:
        filename (str): The filename of the session to load.
        
    Raises:
        FileNotFoundError: If the specified session file does not exist.
    """
    loaded_session, loaded_history = _read_session_snapshot(filename)
    # A saved session owns its model settings. Loading it must not inherit
    # explicit overrides from whichever conversation the tab viewed before.
    CLIENT_SESSIONS.select(client_id, filename, loaded_session, loaded_history)
    if client_id == LEGACY_CLIENT_ID:
        with _GLOBAL_STATE_LOCK:
            GLOBAL_STATE["history"] = deepcopy(loaded_history)
            GLOBAL_STATE["session"] = deepcopy(loaded_session)
            GLOBAL_STATE["active_session_name"] = filename


# ── Slash Command Handler ─────────────────────────────────────────────

_COMMANDS_HELP_MD = """
### Available Commands
* `/help` or `/?` — Show this help
* `/clear` — Reset conversation and system override
* `/save [name]` — Save this session
* `/load [name|index]` — Load a session (lists if no arg)
* `/profile [name]` — Show or set profile (`manual` · `auto` · `low-vram` · `balanced`)
* `/set profile <name>` — Same as `/profile <name>` (default is Modelfile / manual)
* `/set parameter <name> <val>` — Model option (e.g. `temperature 0.25`, `num_ctx 8192`)
* `/set system "…"|default` — Override or reset system prompt
* `/set history` / `/set nohistory` — Multi-turn context on/off
* `/set wordwrap` / `/set nowordwrap` — Line wrap on/off
* `/set format json` / `/set noformat` — Force JSON or free-form output
* `/set verbose` / `/set quiet` — Generation stats on/off
* `/set think` / `/set nothink` — Thinking stream on/off
* `/show parameters` · `/show system` · `/show model` · `/show profile` — Inspect session
* `/vault list` · `/vault search <q>` · `/vault add <path>` · … — Local knowledge tools
* `/quit` or `/exit` or `/q` — Exit
"""

def execute_command_web(
    cmd: str,
    session: dict,
    history: list[dict],
    client_id: str = LEGACY_CLIENT_ID,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """Execute a slash command from the web interface.
    
    Similar to the terminal's `_handle_command`, but formats output as markdown
    strings suitable for rendering in the web UI.
    
    Args:
        cmd (str): The slash command string.
        session (dict): The active session state dictionary.
        history (list[dict]): The conversation history list.
        
    Returns:
        str: The markdown-formatted response of the command execution.
    """
    import os
    import json
    import glob
    import shlex
    from datetime import datetime
    from tools.registry import TOOL_DISPATCH
    
    parts = cmd.strip().split(None, 1)
    base = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    
    if base in ("/quit", "/exit", "/q"):
        return "Session quit signal received."
        
    elif base in ("/help", "/?"):
        return _COMMANDS_HELP_MD
        
    elif base == "/clear":
        history.clear()
        session["system"] = ""
        return "✓ Conversation history and system prompt cleared."
        
    elif base == "/save":
        filename = save_session(rest, session, history, client_id)
        display_name = filename.replace(".json", "")
        return f"✓ Session saved as `{display_name}`"
        
    elif base == "/load":
        saved = list_saved_sessions()
        if not rest:
            if not saved:
                return "No saved sessions found."
            out = ["### Saved Sessions"]
            for i, fp in enumerate(saved, 1):
                name = os.path.basename(fp).replace(".json", "")
                out.append(f"{i}. `{name}`")
            out.append("\nUse `/load <number>` or `/load <name>` to restore.")
            return "\n".join(out)
            
        target_filename = None
        # Try as index first
        try:
            idx = int(rest)
            if 1 <= idx <= len(saved):
                target_filename = os.path.basename(saved[idx - 1])
        except ValueError:
            pass
            
        # Try substring match
        if not target_filename:
            matches = [fp for fp in saved if rest.lower() in os.path.basename(fp).lower()]
            if len(matches) == 1:
                target_filename = os.path.basename(matches[0])
            elif len(matches) > 1:
                out = [f"Multiple sessions match '{rest}':"]
                for fp in matches:
                    out.append(f"- `{os.path.basename(fp).replace('.json', '')}`")
                return "\n".join(out)
                
        if not target_filename:
            return f"No session found matching '{rest}'."
            
        loaded_session, loaded_history = _read_session_snapshot(target_filename)
        session.clear()
        session.update(loaded_session)
        history[:] = loaded_history
        return f"✓ Session loaded: `{target_filename.replace('.json', '')}`"
        
    elif base == "/profile":
        # First-class profile command (parity with CLI).
        if not rest.strip():
            runtime = get_runtime_config(session)
            lines = [
                f"**Current profile:** `{runtime.profile.value}`",
                runtime.selection_reason,
                "",
                "Profiles:",
                "- `manual` — Modelfile defaults (recommended)",
                "- `auto` — Pick low-vram or balanced from VRAM",
                "- `low-vram` — Conservative ~4 GiB settings",
                "- `balanced` — Higher ctx/batch for larger GPUs",
                "",
                "Usage: `/profile <name>` or `/set profile <name>`",
            ]
            return "\n".join(lines)
        profile = rest.strip().lower().replace("_", "-")
        if profile not in {"manual", "auto", "low-vram", "balanced"}:
            return (
                f"Unknown profile: `{profile}`\n"
                "Valid: `manual` · `auto` · `low-vram` · `balanced`"
            )
        candidate = deepcopy(session)
        candidate["runtime_profile"] = profile
        try:
            normalized = _normalize_session_settings(candidate, fallback=session)
        except RuntimeConfigurationError as exc:
            return f"Invalid runtime profile: {exc}"
        session.clear()
        session.update(normalized)
        runtime = get_runtime_config(session)
        lines = [
            f"✓ Runtime profile = `{runtime.profile.value}`",
            runtime.selection_reason,
            (
                f"ctx {runtime.num_ctx} · out {runtime.num_predict} · "
                f"batch {runtime.num_batch} · temp {runtime.temperature}"
            ),
        ]
        lines.extend(f"⚠ {warning}" for warning in runtime.warnings)
        return "\n".join(lines)

    elif base == "/set":
        if not rest:
            return (
                "Usage: `/set <subcommand> [args]`\n"
                "Subcommands: `profile` · `parameter` · `system` · `history` · "
                "`format` · `verbose` · `think` · …\n"
                "Tip: type `/help` or try `/profile`"
            )
        subparts = rest.split(None, 1)
        sub = subparts[0].lower()
        args = subparts[1].strip() if len(subparts) > 1 else ""
        
        if sub == "verbose":
            session["verbose"] = True
            return "✓ Verbose mode enabled — stats shown after each response."
        elif sub == "quiet":
            session["verbose"] = False
            return "✓ Quiet mode enabled."
        elif sub == "wordwrap":
            session["wordwrap"] = True
            return "✓ Word wrapping enabled."
        elif sub == "nowordwrap":
            session["wordwrap"] = False
            return "✓ Word wrapping disabled."
        elif sub == "history":
            session["history"] = True
            return "✓ Conversation history enabled."
        elif sub == "nohistory":
            session["history"] = False
            return "✓ History disabled — each turn is standalone."
        elif sub == "think":
            session["think"] = True
            return "✓ Thinking/reasoning enabled."
        elif sub == "nothink":
            session["think"] = False
            return "✓ Thinking disabled — model will respond directly."
        elif sub == "format":
            if args.lower() == "json":
                session["format"] = "json"
                return "✓ JSON output mode enabled."
            else:
                return f"Unsupported format: `{args}` (supported: json)"
        elif sub == "noformat":
            session["format"] = ""
            return "✓ Output formatting reset to default."
        elif sub == "system":
            if args.startswith('"') and args.endswith('"'):
                args = args[1:-1]
            elif args.startswith("'") and args.endswith("'"):
                args = args[1:-1]
            session["system"] = args
            return f"✓ System prompt set to: {args}" if args else "✓ System prompt reset to default."
        elif sub == "parameter":
            _ALL_PARAMS = {
                "temperature": float,
                "top_p": float,
                "top_k": int,
                "num_predict": int,
                "num_ctx": int,
                "num_batch": int,
                "repeat_penalty": float,
            }
            subparts = args.split(None, 1)
            if len(subparts) != 2:
                return (
                    "Usage: `/set parameter <name> <value>`\n"
                    f"Common: {', '.join(sorted(_ALL_PARAMS.keys()))}"
                )
            name, raw_val = subparts[0].lower(), subparts[1]
            if name not in _ALL_PARAMS:
                return (
                    f"Unknown parameter: `{name}`\n"
                    f"Available: {', '.join(sorted(_ALL_PARAMS.keys()))}"
                )
            try:
                val = _ALL_PARAMS[name](raw_val)
                candidate = deepcopy(session)
                candidate.setdefault("options", {})[name] = val
                normalized = _normalize_session_settings(candidate, fallback=session)
                session.clear()
                session.update(normalized)
                return f"✓ `{name}` = `{val}`"
            except RuntimeConfigurationError as exc:
                return f"Invalid value for {name}: {exc}"
            except ValueError:
                expected = _ALL_PARAMS[name].__name__
                return f"Invalid value for {name}: expected {expected}, got '{raw_val}'"
        elif sub == "profile":
            if not args.strip():
                return (
                    "Usage: `/set profile <name>`\n"
                    "Profiles: `manual` · `auto` · `low-vram` · `balanced`\n"
                    "Tip: `/profile` lists the current profile"
                )
            profile = args.lower().replace("_", "-")
            if profile not in {"manual", "auto", "low-vram", "balanced"}:
                return (
                    f"Unknown profile: `{profile}`\n"
                    "Valid: `manual` · `auto` · `low-vram` · `balanced`"
                )
            candidate = deepcopy(session)
            candidate["runtime_profile"] = profile
            try:
                normalized = _normalize_session_settings(candidate, fallback=session)
            except RuntimeConfigurationError as exc:
                return f"Invalid runtime profile: {exc}"
            session.clear()
            session.update(normalized)
            runtime = get_runtime_config(session)
            lines = [
                f"✓ Runtime profile = `{runtime.profile.value}`",
                runtime.selection_reason,
                (
                    f"ctx {runtime.num_ctx} · out {runtime.num_predict} · "
                    f"batch {runtime.num_batch} · temp {runtime.temperature}"
                ),
            ]
            lines.extend(f"⚠ {warning}" for warning in runtime.warnings)
            return "\n".join(lines)
        else:
            return f"Unknown /set subcommand: `{sub}`"
            
    elif base == "/show":
        if not rest:
            return "Usage: `/show <subcommand>` (parameters · system · model · profile)"
        sub = rest.lower()
        if sub == "parameters":
            runtime = get_runtime_config(session)
            out = [
                f"**Runtime profile:** `{runtime.profile.value}`",
                "**Effective session parameters:**",
            ]
            for k, v in runtime.ollama_options().items():
                out.append(f"- `{k}` = `{v}`")
            if session.get("options"):
                out.append("\nExplicit session overrides are active.")
            return "\n".join(out)
        if sub == "profile":
            runtime = get_runtime_config(session)
            return "\n".join(
                [
                    f"**Runtime profile:** `{runtime.profile.value}`",
                    runtime.selection_reason,
                    (
                        f"ctx {runtime.num_ctx} · out {runtime.num_predict} · "
                        f"batch {runtime.num_batch} · temp {runtime.temperature}"
                    ),
                ]
            )
        elif sub == "system":
            prompt = session.get("system", "")
            if prompt:
                return f"**System prompt:**\n{prompt}"
            else:
                return "No system prompt set (using Modelfile default)."
        elif sub == "model":
            return f"**Model:** `{MODEL_NAME}`"
        else:
            return f"Unknown /show subcommand: `{sub}` (try: parameters, system, model)"
            
    elif base == "/vault":
        try:
            parts = shlex.split(rest)
        except ValueError as exc:
            return f"Invalid /vault command: {exc}"
            
        if not parts or parts[0].lower() in ("help", "-h", "--help"):
            return """
### Vault Commands
* `/vault list` — List indexed vault collections
* `/vault aliases` — List registered vault aliases
* `/vault alias <name> <coll>` — Register a friendly alias for a collection
* `/vault rename <old> <new>` — Rename a vault collection
* `/vault add <path>` — Add a file or folder to the searchable vault
* `/vault status <path>` — Show resumable large-PDF progress
* `/vault read [--cursor n]` — Read all chunks in source order
* `/vault search <query>` — Search the indexed vault
* `/vault delete <source>` — Delete indexed chunks
"""
        sub = parts[0].lower()
        tokens = parts[1:]
        
        def extract_option(tkns, names, default=None):
            for name in names:
                if name in tkns:
                    idx = tkns.index(name)
                    if idx + 1 < len(tkns):
                        val = tkns[idx + 1]
                        tkns.pop(idx + 1)
                        tkns.pop(idx)
                        return val
            return default
            
        collection_option = extract_option(tokens, ("--collection", "-c"), None)
        collection_raw = collection_option or "vault"
        
        try:
            from tools.vault_indexer import resolve_vault_alias
            collection = resolve_vault_alias(collection_raw)
        except ImportError:
            collection = collection_raw
            
        def call_tool(tool_name, **kwargs):
            try:
                spec = normalize_tool_calls([{
                    "function": {"name": tool_name, "arguments": kwargs}
                }])[0]
                execution = execute_tool_call(
                    spec,
                    cancellation_token=cancellation_token,
                )
                return json.loads(execution.content)
            except OperationCancelled:
                raise
            except Exception as exc:
                return {"error": str(exc)}
                
        if sub in ("list", "ls"):
            data = call_tool("list_vaults")
            if "error" in data:
                return f"Vault list failed: {data['error']}"
            vaults = data.get("vaults", [])
            if not vaults:
                return "No indexed vault collections found."
            out = ["### Indexed Vaults"]
            for vault in vaults:
                name = vault.get("collection", "unknown")
                chunk_count = vault.get("indexed_chunks")
                count_text = f"{chunk_count} chunk{'s' if chunk_count != 1 else ''}" if isinstance(chunk_count, int) else "unknown chunks"
                out.append(f"- `{name}` ({count_text})")
            return "\n".join(out)
            
        elif sub == "aliases":
            data = call_tool("list_vault_aliases")
            if "error" in data:
                return f"Failed to list aliases: {data['error']}"
            aliases = data.get("aliases", [])
            if not aliases:
                return "No registered vault aliases found."
            out = ["### Vault Aliases"]
            if isinstance(aliases, dict):
                for name, coll in aliases.items():
                    out.append(f"- `{name}` → `{coll}`")
            else:
                for entry in aliases:
                    if isinstance(entry, dict):
                        out.append(
                            f"- `{entry.get('alias', '?')}` → `{entry.get('collection', '?')}`"
                        )
                    else:
                        out.append(f"- `{entry}`")
            return "\n".join(out)
            
        elif sub == "alias":
            if len(tokens) < 2:
                return "Usage: `/vault alias <name> <collection>`"
            alias_name, target_coll = tokens[0], tokens[1]
            data = call_tool("register_vault_alias", alias=alias_name, collection=target_coll)
            if "error" in data:
                return f"Failed to register alias: {data['error']}"
            collection = data.get("collection", target_coll)
            return f"✓ Alias `{alias_name}` registered to collection `{collection}`"
            
        elif sub == "rename":
            if len(tokens) < 2:
                return "Usage: `/vault rename <old> <new>`"
            old_name, new_name = tokens[0], tokens[1]
            data = call_tool("rename_vault", old_name=old_name, new_name=new_name)
            if "error" in data:
                return f"Failed to rename vault: {data['error']}"
            old_collection = data.get("old_collection", old_name)
            new_collection = data.get("new_collection", new_name)
            return f"✓ Vault collection `{old_collection}` renamed to `{new_collection}`"
            
        elif sub == "add":
            if not tokens:
                return "Usage: `/vault add <path>`"
            vision_mode = extract_option(tokens, ("--vision",), "auto") or "auto"
            max_pages_raw = extract_option(tokens, ("--max-pages",), "20") or "20"
            try:
                max_pages = int(max_pages_raw)
            except ValueError:
                return f"Invalid --max-pages value: `{max_pages_raw}`"
            path = " ".join(tokens)
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                return f"Path does not exist: `{path}`"
                
            if os.path.isfile(abs_path):
                vault_path = None
                file_path = abs_path
            else:
                vault_path = abs_path
                file_path = None
                
            index_args = {
                "file_path": file_path,
                "vision_mode": vision_mode,
                "max_pages": max_pages,
            }
            if vault_path is not None:
                index_args["vault_path"] = vault_path
            if collection_option:
                index_args["collection"] = collection
            data = call_tool("index_vault", **index_args)
            if "error" in data:
                return f"Vault indexing failed: {data['error']}"
                
            incomplete = data.get("incomplete_pdf_count", 0)
            state_label = (
                "indexing complete" if data.get("complete")
                else "checkpoint saved" if incomplete
                else "indexing incomplete"
            )
            out = [
                f"✓ Vault {state_label} "
                f"(collection: `{data.get('collection', collection)}`, chunks: {data.get('indexed_chunks', 0)})"
            ]
            for job in data.get("pdf_jobs", []):
                out.append(
                    f"- `{job.get('source')}`: pages {job.get('indexed_pages')}/{job.get('page_count')}, "
                    f"vision pages {job.get('vision_pages')}, next page {job.get('next_page')}"
                )
            return "\n".join(out)

        elif sub == "status":
            if not tokens:
                return "Usage: `/vault status <pdf-path> [--collection name]`"
            path = " ".join(tokens)
            status_args = {
                "file_path": path,
                "action": "status",
            }
            if collection_option:
                status_args["collection"] = collection
            data = call_tool("index_vault", **status_args)
            if data.get("error"):
                return f"Vault status failed: {data['error']}"
            jobs = data.get("jobs", [])
            if not jobs:
                return "No PDF checkpoint exists for that path and collection."
            return "\n".join(
                f"- `{job.get('source', os.path.basename(path))}`: "
                f"pages {job.get('indexed_pages', 0)}/{job.get('page_count', '?')}, "
                f"chunks {job.get('indexed_chunks', 0)}, "
                f"{'complete' if job.get('complete') else 'next page ' + str(job.get('next_page', 1))}"
                for job in jobs
            )

        elif sub == "read":
            cursor = extract_option(tokens, ("--cursor",), "0") or "0"
            source_filter = extract_option(tokens, ("--source", "-s"), None)
            data = call_tool(
                "vault_read", collection=collection, cursor=cursor, source=source_filter,
            )
            if data.get("error"):
                return f"Vault read failed: {data['error']}"
            suffix = f"\n\nNext cursor: `{data['next_cursor']}`" if data.get("next_cursor") is not None else ""
            return f"```text\n{data.get('content', '')}\n```{suffix}"
            
        elif sub == "search":
            if not tokens:
                return "Usage: `/vault search <query> [--top-k n] [--source path]`"
                
            top_k_str = extract_option(tokens, ("--top-k", "-k"), "4")
            source_filter = extract_option(tokens, ("--source", "-s"), None)
            
            try:
                top_k = int(top_k_str)
            except ValueError:
                top_k = 4
                
            query = " ".join(tokens)
            data = call_tool(
                "vault_search",
                query=query,
                collection=collection,
                top_k=top_k,
                source=source_filter,
            )
            if "error" in data:
                return f"Vault search failed: {data['error']}"
                
            results = data.get("results", [])
            if not results:
                return f"No results found for query `{query}` in vault `{collection}`."
                
            out = [f"### Vault Search Results for '{query}'"]
            for idx, res in enumerate(results, 1):
                if not isinstance(res, dict):
                    continue
                src = res.get("source") or res.get("source_path") or "unknown"
                score = res.get("score", 0.0)
                text = res.get("text") or res.get("document") or ""
                snippet = text[:260] + "..." if len(text) > 260 else text
                try:
                    score_text = f"{float(score):.3f}"
                except (TypeError, ValueError):
                    score_text = str(score)
                out.append(f"{idx}. **{src}** (score: {score_text})\n>{snippet}\n")
            return "\n".join(out)
            
        elif sub == "delete":
            delete_all = False
            if "--all" in tokens:
                delete_all = True
                tokens.remove("--all")
                
            if not delete_all and not tokens:
                return "Usage: `/vault delete <source> [--collection name]` or `/vault delete --all [--collection name]`"
                
            source = tokens[0] if tokens else None
            data = call_tool(
                "delete_vault_item",
                source=source,
                collection=collection,
                delete_collection=delete_all,
            )
            if "error" in data:
                return f"Vault delete failed: {data['error']}"
                
            if data.get("deleted_collection"):
                return f"✓ Vault collection deleted: `{collection}`"
            deleted_chunks = data.get("deleted_chunks", 0)
            return f"✓ Vault chunks deleted: {deleted_chunks} chunk(s) (collection: `{collection}`)"
            
        else:
            return f"Unknown /vault subcommand: `{sub}` (try: list, aliases, rename, add, search, delete)"
            
    else:
        return f"Unknown command: `{base}` (type `/help` for available commands)"


# ── Chat Stream Generator ─────────────────────────────────────────────

def _deep_research_plan_events(
    user_input: str,
    session_data: dict,
    runtime: RuntimeConfig,
    operation_owner: str,
    cancellation_token: CancellationToken,
):
    """Stream the intent-planning pass and return context-scaled search queries."""
    query_count = research_query_count(runtime.num_ctx)
    planning_messages = [{
        "role": "user",
        "content": DEEP_RESEARCH_PLANNER_PROMPT.format(
            query_count=query_count,
            user_input=user_input,
        ),
    }]
    options = dict(effective_session_model_options(session_data)[1])
    options["num_predict"] = min(512, int(options.get("num_predict", 512)))
    try:
        guarded_options = guarded_options_for_call(
            planning_messages,
            options,
            tools=None,
        )
        stream = OllamaService(runtime).chat(
            kind=OperationKind.CHAT,
            owner=f"{operation_owner}:research-plan",
            cancellation_token=cancellation_token,
            operation_timeout=runtime.chat_timeout_seconds,
            model=MODEL_NAME,
            messages=planning_messages,
            stream=True,
            think=True,
            format="json",
            options=guarded_options,
        )
    except Exception as exc:
        yield {
            "type": "status",
            "message": f"Research planning used fallback queries: {exc}",
            "color": "yellow",
        }
        return parse_research_queries("", user_input, query_count)

    content = ""
    thinking_open = False
    planning_error: Exception | None = None
    try:
        for chunk in stream:
            cancellation_token.raise_if_cancelled()
            msg = getattr(chunk, "message", None)
            if msg is None:
                continue
            thinking = getattr(msg, "thinking", None) or ""
            if thinking:
                if not thinking_open:
                    thinking_open = True
                    yield {"type": "thinking_start"}
                yield {"type": "thinking_chunk", "text": thinking}
            content += getattr(msg, "content", None) or ""
    except OperationCancelled:
        raise
    except Exception as exc:
        planning_error = exc
    finally:
        if hasattr(stream, "close"):
            try:
                stream.close()
            except Exception:
                pass
    if thinking_open:
        yield {"type": "thinking_end"}
    if planning_error is not None:
        yield {
            "type": "status",
            "message": f"Research planning used fallback queries: {planning_error}",
            "color": "yellow",
        }
        return parse_research_queries("", user_input, query_count)
    return parse_research_queries(content, user_input, query_count)


def _deep_research_search_events(
    queries: list[str],
    cancellation_token: CancellationToken,
):
    """Execute the planned hard-difficulty searches and return transcript messages."""
    calls = [
        {
            "function": {
                "name": "web_search",
                "arguments": {
                    "query": query,
                    "difficulty": "hard",
                    # Deep Research reads the top page for every query instead
                    # of relying only on result snippets. Context trimming
                    # truthfully bounds the combined evidence afterward.
                    "include_content": True,
                    "max_pages": 1,
                    "max_chars_per_page": 4000,
                },
            }
        }
        for query in queries
    ]
    yield {"type": "tool_calls_start", "calls": calls}
    specs = normalize_tool_calls(calls)
    for index, spec in enumerate(specs):
        yield {
            "type": "tool_start",
            "id": f"research-{index}",
            "name": spec.name,
            "arguments": spec.arguments,
        }

    results_by_raw_id = {
        id(result.spec.raw): result
        for result in execute_tool_calls(
            calls,
            cancellation_token=cancellation_token,
        )
    }
    tool_messages: list[dict] = []
    for index, spec in enumerate(specs):
        result = results_by_raw_id[id(spec.raw)]
        yield {
            "type": "tool_end",
            "id": f"research-{index}",
            "name": result.spec.name,
            "result": result.content,
        }
        tool_messages.append(result.as_tool_message())
    return calls, tool_messages


def _ultra_review_events(
    user_input: str,
    draft_content: str,
    base_messages: list[dict],
    session_data: dict,
    runtime: RuntimeConfig,
    operation_owner: str,
    cancellation_token: CancellationToken,
):
    """Run Ultra's independent second reasoning pass and stream only its answer."""
    review_messages = prepare_messages_for_model(
        [
            *base_messages,
            {"role": "assistant", "content": draft_content},
            {
                "role": "user",
                "content": ULTRA_REVIEW_PROMPT.format(user_input=user_input),
            },
        ],
        session_data,
        tools=None,
    )
    try:
        guarded_options = guarded_options_for_call(
            review_messages,
            effective_session_model_options(session_data)[1],
            tools=None,
        )
        kwargs = {
            "model": MODEL_NAME,
            "messages": review_messages,
            "stream": True,
            "think": True,
            "options": guarded_options,
        }
        if session_data.get("format"):
            kwargs["format"] = session_data["format"]
        stream = OllamaService(runtime).chat(
            kind=OperationKind.CHAT,
            owner=f"{operation_owner}:ultra-review",
            cancellation_token=cancellation_token,
            operation_timeout=runtime.chat_timeout_seconds,
            **kwargs,
        )
    except Exception as exc:
        yield {
            "type": "status",
            "message": f"Second review was unavailable; returning the complete first draft: {exc}",
            "color": "yellow",
        }
        return "", ""

    content = ""
    thinking = ""
    thinking_open = False
    try:
        for chunk in stream:
            cancellation_token.raise_if_cancelled()
            msg = getattr(chunk, "message", None)
            if msg is None:
                continue
            thinking_chunk = getattr(msg, "thinking", None) or ""
            if thinking_chunk:
                if not thinking_open:
                    thinking_open = True
                    yield {"type": "thinking_start"}
                thinking += thinking_chunk
                yield {"type": "thinking_chunk", "text": thinking_chunk}
                continue
            content_chunk = getattr(msg, "content", None) or ""
            if content_chunk:
                if thinking_open:
                    thinking_open = False
                    yield {"type": "thinking_end"}
                content += content_chunk
                yield {"type": "content_chunk", "text": content_chunk}
    finally:
        if hasattr(stream, "close"):
            try:
                stream.close()
            except Exception:
                pass
    if thinking_open:
        yield {"type": "thinking_end"}
    return content, thinking


def _generate_chat_events_impl(
    user_input: str,
    session_data: dict,
    history_data: list[dict],
    session_name: str | None = None,
    *,
    cancellation_token: CancellationToken | None = None,
    generation_id: str | None = None,
    publish_global: bool = True,
    client_id: str = LEGACY_CLIENT_ID,
):
    """Generator yielding dictionary objects representing the progress of agent generation.
    
    Supports tool execution and chained follow-up model runs. Yields Server-Sent Events (SSE)
    compatible dictionary objects for streaming status, thinking chunks, content chunks,
    and tool execution updates to the frontend.
    
    Args:
        user_input (str): The raw text submitted by the user.
        session_data (dict): The session configuration and state.
        history_data (list[dict]): The conversation history.
        session_name (str | None): Conversation file that owns this generation.
        
    Yields:
        dict: A dictionary representing an event in the generation process.
    """
    cancellation_token = cancellation_token or CancellationToken()
    cancellation_token.raise_if_cancelled()
    runtime = get_runtime_config(session_data)
    agent_mode = normalize_agent_mode(session_data.get("agent_mode"))
    generation_start_session = deepcopy(session_data)
    operation_owner = f"web:{generation_id or threading.get_ident()}"
    origin_name = session_name or GLOBAL_STATE.get("active_session_name", "Active Session")

    if user_input.startswith('/'):
        output = execute_command_web(
            user_input,
            session_data,
            history_data,
            client_id,
            cancellation_token,
        )
        yield {"type": "content_chunk", "content": output}
        yield {
            "type": "done",
            "state": "completed",
            "history": history_data,
            "active_session_name": origin_name,
            "saved_sessions": list_saved_sessions(),
        }
        return

    previous_name = origin_name
    origin_name = save_session_snapshot(
        origin_name,
        session_data,
        history_data,
        generation_start_session=generation_start_session,
    )
    cancellation_token.raise_if_cancelled()
    if publish_global:
        with _GLOBAL_STATE_LOCK:
            if GLOBAL_STATE.get("active_session_name") == previous_name:
                GLOBAL_STATE["active_session_name"] = origin_name
    yield {"type": "conversation_started", "session_name": origin_name}

    # Exact, persistently approved routine triggers are deterministic. Do this
    # before asking the model so a saved phrase cannot be overlooked by tool
    # selection, while the routine executor still enforces its safety policy.
    routine_handler = TOOL_DISPATCH.get("automated_routine_executor")
    if routine_handler:
        cancellation_token.raise_if_cancelled()
        try:
            preview_call = normalize_tool_calls([{
                "function": {
                    "name": "automated_routine_executor",
                    "arguments": {"action": "show", "trigger": user_input},
                }
            }])[0]
            preview_result = execute_tool_call(
                preview_call,
                cancellation_token=cancellation_token,
            )
            preview = json.loads(preview_result.content)
        except (TypeError, ValueError, json.JSONDecodeError):
            preview = {}
        cancellation_token.raise_if_cancelled()
        if preview.get("automatic_trigger") is True:
            routine_name = str(preview.get("name", "routine"))
            yield {"type": "tool_start", "name": "automated_routine_executor"}
            try:
                routine_call = normalize_tool_calls([{
                    "function": {
                        "name": "automated_routine_executor",
                        "arguments": {
                            "action": "run",
                            "trigger": user_input,
                            "dry_run": False,
                        },
                    }
                }])[0]
                routine_result = execute_tool_call(
                    routine_call,
                    cancellation_token=cancellation_token,
                )
                result = json.loads(routine_result.content)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                result = {"ok": False, "error": str(exc)}
            cancellation_token.raise_if_cancelled()
            yield {
                "type": "tool_end",
                "name": "automated_routine_executor",
                "result": json.dumps(result, ensure_ascii=False),
            }
            if result.get("ok") is True:
                output = f"✓ Routine **{routine_name}** completed."
                routine_state = "completed"
            else:
                detail = result.get("error") or "One or more routine actions failed."
                output = f"Routine **{routine_name}** failed: {detail}"
                routine_state = "failed"

            if session_data.get("history", True):
                history_data.append({"role": "user", "content": user_input})
                history_data.append({"role": "assistant", "content": output})
                origin_name = save_session_snapshot(
                    origin_name,
                    session_data,
                    history_data,
                    generation_start_session=generation_start_session,
                )
                viewing_origin = publish_global and GLOBAL_STATE.get("active_session_name") == origin_name
                titled_name = title_temporary_session(
                    history_data,
                    origin_name,
                    cancellation_token,
                    session_data=session_data,
                    owner=operation_owner,
                    generation_id=generation_id,
                    client_id=client_id,
                )
                if titled_name:
                    origin_name = titled_name
                save_session_snapshot(
                    origin_name,
                    session_data,
                    history_data,
                    generation_start_session=generation_start_session,
                )
                if viewing_origin:
                    with _GLOBAL_STATE_LOCK:
                        GLOBAL_STATE["active_session_name"] = origin_name
                        GLOBAL_STATE["session"] = deepcopy(session_data)
                        GLOBAL_STATE["history"] = deepcopy(history_data)
            yield {"type": "content_chunk", "content": output}
            yield {
                "type": "done",
                "state": routine_state,
                "history": history_data,
                "active_session_name": origin_name,
                "saved_sessions": list_saved_sessions(),
            }
            return
    # 1. Sync system prompt override
    default_system_prompt = load_default_system_prompt()
        
    active_system = session_data.get("system") or default_system_prompt
    if active_system:
        if not history_data or history_data[0].get("role") != "system" or history_data[0].get("content") != active_system:
            history_data[:] = [m for m in history_data if m.get("role") != "system"]
            history_data.insert(0, {"role": "system", "content": active_system})
    else:
        history_data[:] = [m for m in history_data if m.get("role") != "system"]
        
    # 2. Check for local file auto-indexing
    pre_tool_message = None
    try:
        cancellation_token.raise_if_cancelled()
        if os.path.exists(user_input) and os.path.isfile(user_input):
            size = os.path.getsize(user_input)
            ext = os.path.splitext(user_input)[1].lower()
            INDEX_THRESHOLD = 200_000
            if size > INDEX_THRESHOLD or ext in (".pdf", ".docx"):
                yield {"type": "status", "message": f"Large/binary file detected — indexing {user_input}...", "color": "yellow"}
                handler = TOOL_DISPATCH.get("index_vault")
                if handler:
                    index_call = normalize_tool_calls([{
                        "function": {
                            "name": "index_vault",
                            "arguments": {
                                "file_path": user_input,
                            },
                        }
                    }])[0]
                    index_result = execute_tool_call(
                        index_call,
                        cancellation_token=cancellation_token,
                    )
                    cancellation_token.raise_if_cancelled()
                    tool_content = index_result.content
                    tool_msg = {
                        "role": "tool",
                        "tool_name": "index_vault",
                        "name": "index_vault",
                        "content": tool_content,
                    }
                    if session_data.get("history", True):
                        history_data.append(tool_msg)
                    else:
                        pre_tool_message = tool_msg
                    try:
                        index_payload = json.loads(tool_content)
                    except (TypeError, json.JSONDecodeError):
                        index_payload = {}
                    if (
                        index_result.ok
                        and not index_payload.get("error")
                        and index_payload.get("continuation_required")
                    ):
                        job = (index_payload.get("pdf_jobs") or [{}])[0]
                        yield {
                            "type": "status",
                            "message": (
                                f"Index checkpoint saved — pages {job.get('indexed_pages', '?')}/"
                                f"{job.get('page_count', '?')}; resume required."
                            ),
                            "color": "yellow",
                        }
                    elif index_result.ok and index_payload.get("complete"):
                        yield {"type": "status", "message": "Indexing complete.", "color": "green"}
                    else:
                        yield {"type": "status", "message": "Indexing did not complete.", "color": "yellow"}
    except Exception as e:
        if isinstance(e, OperationCancelled):
            raise
        yield {"type": "status", "message": f"Indexing failed: {e}", "color": "red"}
        
    # 3. Build messages to send
    if session_data.get("history", True):
        history_data.append({"role": "user", "content": user_input})
        # Persist the prompt immediately so stopping a generation cannot lose it.
        save_session_snapshot(
            origin_name,
            session_data,
            history_data,
            generation_start_session=generation_start_session,
        )
        messages_to_send = prepare_messages_for_model(history_data, session_data, tools=TOOL_SCHEMAS)
    else:
        messages_to_send = []
        if history_data and history_data[0].get("role") == "system":
            messages_to_send.append(history_data[0])
        if pre_tool_message:
            messages_to_send.append(pre_tool_message)
        messages_to_send.append({"role": "user", "content": user_input})
        messages_to_send = prepare_messages_for_model(messages_to_send, session_data, tools=TOOL_SCHEMAS)

    initial_research_calls: list[dict] = []
    initial_research_results: list[dict] = []
    deep_research_search_count = 0
    deep_research_next_compaction = DEEP_RESEARCH_COMPACT_INTERVAL
    deep_research_scrape_count = 0
    deep_research_next_scrape_compaction = DEEP_RESEARCH_SCRAPE_COMPACT_INTERVAL
    deep_research_compacted_prefix: list[dict] | None = None
    deep_research_history_checkpoint_index = 0
    deep_research_checkpoint_chars = max(3000, min(16000, int(runtime.num_ctx)))
    if agent_mode == AGENT_MODE_ULTRA:
        messages_to_send = prepare_messages_for_model(
            [
                *messages_to_send,
                {
                    "role": "user",
                    "content": ULTRA_MODE_PROMPT.format(user_input=user_input),
                },
            ],
            session_data,
            tools=TOOL_SCHEMAS,
        )
    elif agent_mode == AGENT_MODE_DEEP_RESEARCH:
        yield {
            "type": "status",
            "message": "Deep Research is planning a multi-query research pass…",
            "color": "blue",
            "activity_mode": AGENT_MODE_DEEP_RESEARCH,
        }
        queries = yield from _deep_research_plan_events(
            user_input,
            session_data,
            runtime,
            operation_owner,
            cancellation_token,
        )
        cancellation_token.raise_if_cancelled()
        yield {
            "type": "status",
            "message": f"Deep Research is running {len(queries)} complementary web searches…",
            "color": "blue",
            "activity_mode": AGENT_MODE_DEEP_RESEARCH,
        }
        initial_research_calls, initial_research_results = yield from _deep_research_search_events(
            queries,
            cancellation_token,
        )
        research_assistant = {
            "role": "assistant",
            "content": "",
            "tool_calls": initial_research_calls,
        }
        if session_data.get("history", True):
            history_data.append(research_assistant)
            history_data.extend(initial_research_results)
            save_session_snapshot(
                origin_name,
                session_data,
                history_data,
                generation_start_session=generation_start_session,
            )
            research_base = list(history_data)
        else:
            research_base = [
                *messages_to_send,
                research_assistant,
                *initial_research_results,
            ]
        deep_research_search_count = len(initial_research_results)
        if deep_research_search_count >= deep_research_next_compaction:
            deep_research_compacted_prefix, _ = compact_deep_research_messages(
                research_base,
                user_input,
                max_checkpoint_chars=deep_research_checkpoint_chars,
            )
            research_base = deep_research_compacted_prefix
            if session_data.get("history", True):
                deep_research_history_checkpoint_index = len(history_data)
            while deep_research_next_compaction <= deep_research_search_count:
                deep_research_next_compaction += DEEP_RESEARCH_COMPACT_INTERVAL
        messages_to_send = prepare_messages_for_model(
            [
                *research_base,
                {
                    "role": "user",
                    "content": (
                        build_tool_continuation_prompt(user_input)
                        + "\n\n"
                        + DEEP_RESEARCH_SYNTHESIS_PROMPT.format(
                            user_input=user_input,
                        )
                    ),
                },
            ],
            session_data,
            tools=TOOL_SCHEMAS,
            extra_reserved_tokens=CONTEXT_TOOL_LOOP_RESERVE,
        )
        
    # 4. Stream response loop (supports tool execution and model chain-calling)
    executed_tool_calls: dict[str, dict] = {}
    for call, result in zip(initial_research_calls, initial_research_results):
        turn_key = _tool_call_turn_key(call)
        if turn_key:
            executed_tool_calls[turn_key] = dict(result)
    vault_index_loop_state = _new_vault_index_loop_state()
    tool_rounds = 1 if initial_research_calls else 0
    unbounded_last_tool_signature = ""
    unbounded_repeated_tool_rounds = 0
    output_continuation_rounds = 0
    output_continuation_base: list[dict] | None = None
    accumulated_content = ""
    accumulated_thinking = ""
    continuing_output = False
    while True:
        cancellation_token.raise_if_cancelled()
        suppress_vault_completion_claim = bool(
            vault_index_loop_state.get("expected_arguments")
        )
        runtime_tools = None if continuing_output else tool_schemas_for_model(
            messages_to_send, session_data, TOOL_SCHEMAS
        )
        if agent_mode in {AGENT_MODE_ULTRA, AGENT_MODE_DEEP_RESEARCH}:
            runtime_tools = force_hard_web_search_schema(runtime_tools)
        try:
            guarded_options = guarded_options_for_call(
                messages_to_send,
                effective_session_model_options(session_data)[1],
                runtime_tools,
                extra_reserved_tokens=(
                    CONTEXT_TOOL_LOOP_RESERVE if tool_rounds and not continuing_output else 0
                ),
            )
        except ContextWindowError as exc:
            message = f"Context window guard stopped this response before generation: {exc}"
            yield {"type": "status", "message": message, "color": "yellow"}
            assistant_msg = {"role": "assistant", "content": message}
            if session_data.get("history", True):
                history_data.append(assistant_msg)
                save_session_snapshot(
                    origin_name,
                    session_data,
                    history_data,
                    generation_start_session=generation_start_session,
                )
            yield {"type": "content_chunk", "content": message}
            yield {
                "type": "done",
                "state": "failed",
                "error": message,
                "history": history_data,
                "active_session_name": origin_name,
                "saved_sessions": list_saved_sessions(),
            }
            break

        kwargs = {
            "model": MODEL_NAME,
            "messages": messages_to_send,
            "stream": True,
            "think": (
                False
                if continuing_output
                else (
                    True
                    if agent_mode in {AGENT_MODE_ULTRA, AGENT_MODE_DEEP_RESEARCH}
                    else session_data.get("think", True)
                )
            ),
        }
        if session_data.get("format"):
            kwargs["format"] = session_data["format"]
        if runtime_tools:
            kwargs["tools"] = runtime_tools
        if guarded_options:
            kwargs["options"] = guarded_options
            
        try:
            stream = OllamaService(runtime).chat(
                kind=OperationKind.CHAT,
                owner=operation_owner,
                cancellation_token=cancellation_token,
                operation_timeout=runtime.chat_timeout_seconds,
                **kwargs,
            )
        except Exception as e:
            message = f"Ollama Chat error: {e}"
            yield {"type": "status", "message": message, "color": "red"}
            yield {
                "type": "done",
                "state": "failed",
                "error": message,
                "history": history_data,
                "active_session_name": origin_name,
                "saved_sessions": list_saved_sessions(),
            }
            break
            
        thinking_buf = ""
        content_buf = ""
        tool_calls = []
        in_thinking = False
        thinking_started = False
        prompt_tokens = 0
        eval_tokens = 0
        done_reason = ""
        
        try:
            for chunk in stream:
                cancellation_token.raise_if_cancelled()
                done_reason = _chunk_done_reason(chunk) or done_reason
                msg = chunk.message
            
                if getattr(chunk, "prompt_eval_count", None):
                    prompt_tokens = chunk.prompt_eval_count
                if getattr(chunk, "eval_count", None):
                    eval_tokens = chunk.eval_count
            
                # Intercept tool calls
                if getattr(msg, "tool_calls", None):
                    tool_calls = [
                        {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]
                    break
                
                # Stream thinking text
                thinking_chunk = getattr(msg, "thinking", None) or ""
                if thinking_chunk:
                    if not thinking_started:
                        thinking_started = True
                        in_thinking = True
                        yield {"type": "thinking_start"}
                    thinking_buf += thinking_chunk
                    yield {"type": "thinking_chunk", "text": thinking_chunk}
                    continue
                
                # Stream final content text
                content_chunk = getattr(msg, "content", None) or ""
                if content_chunk:
                    if in_thinking:
                        in_thinking = False
                        yield {"type": "thinking_end"}
                    content_buf += content_chunk
                    if (
                        not suppress_vault_completion_claim
                        and agent_mode != AGENT_MODE_ULTRA
                    ):
                        yield {"type": "content_chunk", "text": content_chunk}
        finally:
            if hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception:
                    pass
                
        if in_thinking:
            yield {"type": "thinking_end"}

        if (
            tool_calls
            and agent_mode in {AGENT_MODE_ULTRA, AGENT_MODE_DEEP_RESEARCH}
        ):
            tool_calls = force_high_tool_difficulty(tool_calls)

        if (
            not tool_calls
            and content_buf
            and _output_limit_reached(
                done_reason,
                eval_tokens,
                int(guarded_options.get("num_predict", 0)),
            )
            and output_continuation_rounds < MAX_OUTPUT_CONTINUATION_ROUNDS
        ):
            accumulated_content += content_buf
            accumulated_thinking += thinking_buf
            output_continuation_rounds += 1
            if output_continuation_base is None:
                output_continuation_base = list(messages_to_send)
            reminder = {
                "role": "user",
                "content": OUTPUT_CONTINUATION_PROMPT.format(user_input=user_input),
            }
            messages_to_send = prepare_messages_for_model(
                [
                    *output_continuation_base,
                    {"role": "assistant", "content": accumulated_content},
                    reminder,
                ],
                session_data,
                tools=None,
            )
            continuing_output = True
            continue

        content_buf = accumulated_content + content_buf
        thinking_buf = accumulated_thinking + thinking_buf
        if (
            content_buf
            and _output_limit_reached(
                done_reason,
                eval_tokens,
                int(guarded_options.get("num_predict", 0)),
            )
            and output_continuation_rounds >= MAX_OUTPUT_CONTINUATION_ROUNDS
        ):
            notice = (
                "\n\n[Selene paused after the safe automatic continuation limit. "
                "Ask to continue if you need more.]"
            )
            content_buf += notice
            yield {"type": "content_chunk", "text": notice}
            
        # Send token usage if available
        if prompt_tokens or eval_tokens:
            yield {
                "type": "token_usage", 
                "total": prompt_tokens + eval_tokens,
                "budget": int(guarded_options.get("num_ctx", runtime.num_ctx))
            }
            
        # Compile assistant message
        assistant_msg = {"role": "assistant", "content": content_buf}
        if thinking_buf:
            assistant_msg["thinking"] = thinking_buf
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
            
        if session_data.get("history", True):
            history_data.append(assistant_msg)

        # The checkpoint is authoritative. If the model emits a final answer
        # while a verified continuation remains, execute that exact continuation
        # instead of accepting a premature completion claim.
        if not tool_calls and vault_index_loop_state.get("expected_arguments"):
            automatic_call = _automatic_vault_index_tool_call(vault_index_loop_state)
            if automatic_call is not None:
                if (
                    session_data.get("history", True)
                    and history_data
                    and history_data[-1] is assistant_msg
                ):
                    history_data.pop()
                tool_calls = [automatic_call]
                assistant_msg = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                }
                content_buf = ""
                thinking_buf = ""
                if session_data.get("history", True):
                    history_data.append(assistant_msg)

        if not tool_calls and vault_index_loop_state.get("blocked_reason"):
            cancellation_token.raise_if_cancelled()
            message = str(vault_index_loop_state["blocked_reason"])
            if (
                session_data.get("history", True)
                and history_data
                and history_data[-1] is assistant_msg
            ):
                history_data.pop()
                history_data.append({"role": "assistant", "content": message})
            yield {"type": "status", "message": message, "color": "yellow"}
            yield {"type": "content_chunk", "text": message}
            save_session_snapshot(
                origin_name,
                session_data,
                history_data,
                generation_start_session=generation_start_session,
            )
            yield {
                "type": "done",
                "state": "failed",
                "error": message,
                "history": history_data,
                "active_session_name": origin_name,
                "saved_sessions": list_saved_sessions(),
            }
            break

        if not tool_calls and agent_mode == AGENT_MODE_ULTRA and content_buf:
            cancellation_token.raise_if_cancelled()
            yield {
                "type": "status",
                "message": "Running Ultra's second independent review…",
                "color": "blue",
                "activity_mode": AGENT_MODE_ULTRA,
            }
            reviewed_content, reviewed_thinking = yield from _ultra_review_events(
                user_input,
                content_buf,
                output_continuation_base or messages_to_send,
                session_data,
                runtime,
                operation_owner,
                cancellation_token,
            )
            if reviewed_content:
                content_buf = reviewed_content
                assistant_msg["content"] = reviewed_content
                if reviewed_thinking:
                    thinking_buf = "\n\n".join(
                        part for part in (thinking_buf, reviewed_thinking) if part
                    )
                    assistant_msg["thinking"] = thinking_buf
            else:
                # The first draft was intentionally withheld while the review
                # ran. Never finish the turn without showing the usable draft.
                yield {"type": "content_chunk", "text": content_buf}
            
        # If there are no tool calls, this turn is completed
        if not tool_calls:
            cancellation_token.raise_if_cancelled()
            viewing_origin = publish_global and GLOBAL_STATE.get("active_session_name") == origin_name
            titled_name = title_temporary_session(
                history_data,
                origin_name,
                cancellation_token,
                session_data=session_data,
                owner=operation_owner,
                generation_id=generation_id,
                client_id=client_id,
            )
            if titled_name:
                origin_name = titled_name
            save_session_snapshot(
                origin_name,
                session_data,
                history_data,
                generation_start_session=generation_start_session,
            )
            if viewing_origin:
                with _GLOBAL_STATE_LOCK:
                    GLOBAL_STATE["active_session_name"] = origin_name
                    GLOBAL_STATE["session"] = deepcopy(session_data)
                    GLOBAL_STATE["history"] = deepcopy(history_data)
            yield {
                "type": "done",
                "state": "completed",
                "history": history_data,
                "active_session_name": origin_name,
                "saved_sessions": list_saved_sessions(),
            }
            break

        if agent_mode in {AGENT_MODE_ULTRA, AGENT_MODE_DEEP_RESEARCH}:
            signature = tool_call_round_signature(tool_calls)
            progressing_vault = _is_progressing_vault_index_round(
                tool_calls,
                vault_index_loop_state,
            )
            if signature == unbounded_last_tool_signature and not progressing_vault:
                unbounded_repeated_tool_rounds += 1
            else:
                unbounded_repeated_tool_rounds = 0
            unbounded_last_tool_signature = signature
            mode_label = (
                "Ultra Thinking"
                if agent_mode == AGENT_MODE_ULTRA
                else "Deep Research"
            )
            message = (
                f"{mode_label} stopped a repeated no-progress tool loop. "
                "The ordinary tool-round limit was suspended, but the model requested "
                "the same tool batch three times without new evidence."
                if unbounded_repeated_tool_rounds >= 2
                else None
            )
        else:
            message = _tool_loop_stop_message(
                tool_rounds,
                tool_calls,
                vault_index_loop_state,
            )
        if message:
            yield {"type": "status", "message": message, "color": "yellow"}
            yield {"type": "content_chunk", "text": message}
            if session_data.get("history", True):
                # The just-appended model message contains tool_calls that were
                # deliberately not executed. Never persist an assistant call
                # without matching tool-result messages; it would make the
                # next Ollama transcript structurally invalid.
                if history_data and history_data[-1] is assistant_msg:
                    history_data.pop()
                terminal_content = "\n\n".join(
                    part for part in (content_buf.strip(), message) if part
                )
                history_data.append({"role": "assistant", "content": terminal_content})
            save_session_snapshot(
                origin_name,
                session_data,
                history_data,
                generation_start_session=generation_start_session,
            )
            yield {
                "type": "done",
                "state": "failed",
                "error": message,
                "history": history_data,
                "active_session_name": origin_name,
                "saved_sessions": list_saved_sessions(),
            }
            break
        tool_rounds += 1
            
        # Execute tool calls
        yield {"type": "tool_calls_start", "calls": tool_calls}

        tool_results_by_index = {}
        calls_to_execute = []
        original_index_by_call_id = {}
        pending_key_by_index = {}
        pending_index_by_key = {}
        duplicate_source_by_index = {}
        for index, call in enumerate(tool_calls):
            turn_key = _tool_call_turn_key(call)
            if turn_key and turn_key in executed_tool_calls:
                cached = dict(executed_tool_calls[turn_key])
                if not _should_reexecute_turn_duplicate(call, cached):
                    tool_results_by_index[index] = cached
                    yield {
                        "type": "tool_end",
                        "id": index,
                        "name": cached.get("tool_name") or cached.get("name") or "tool",
                        "result": cached.get("content", ""),
                    }
                    continue
            if turn_key and turn_key in pending_index_by_key:
                duplicate_source_by_index[index] = pending_index_by_key[turn_key]
                continue
            if turn_key:
                pending_key_by_index[index] = turn_key
                pending_index_by_key[turn_key] = index
            original_index_by_call_id[id(call)] = index
            calls_to_execute.append(call)

        if (
            calls_to_execute
            and agent_mode in {AGENT_MODE_ULTRA, AGENT_MODE_DEEP_RESEARCH}
        ):
            # Enforce again at the execution boundary. Mutating the existing
            # dictionaries preserves object identity used by event/result
            # correlation while guaranteeing the tool runner sees hard depth.
            hardened_calls = force_high_tool_difficulty(calls_to_execute)
            for original, hardened in zip(calls_to_execute, hardened_calls):
                original.clear()
                original.update(hardened)

        execution_specs = normalize_tool_calls(calls_to_execute)
        for can_parallel, batch in build_execution_batches(execution_specs):
            if can_parallel:
                yield {
                    "type": "tool_parallel_start",
                    "count": len(batch),
                    "names": [spec.name for spec in batch],
                }
            for spec in batch:
                original_index = original_index_by_call_id[id(spec.raw)]
                yield {
                    "type": "tool_start",
                    "id": original_index,
                    "name": spec.name,
                    "arguments": spec.arguments,
                }

        # The shared executor owns timeout uncertainty, side-effect ordering,
        # deterministic result order, cancellation, and resource guards.
        for result in execute_tool_calls(
            calls_to_execute,
            cancellation_token=cancellation_token,
        ):
            original_index = original_index_by_call_id[id(result.spec.raw)]
            yield {
                "type": "tool_end",
                "id": original_index,
                "name": result.spec.name,
                "result": result.content,
            }
            tool_message = result.as_tool_message()
            tool_results_by_index[original_index] = tool_message
            if original_index in pending_key_by_index:
                executed_tool_calls[pending_key_by_index[original_index]] = dict(tool_message)

        for index, source_index in sorted(duplicate_source_by_index.items()):
            cached = dict(tool_results_by_index[source_index])
            tool_results_by_index[index] = cached
            yield {
                "type": "tool_end",
                "id": index,
                "name": cached.get("tool_name") or cached.get("name") or "tool",
                "result": cached.get("content", ""),
            }

        tool_results = [tool_results_by_index[index] for index in sorted(tool_results_by_index)]
        _update_vault_index_loop_state(
            vault_index_loop_state,
            tool_calls,
            tool_results,
        )
            
        if session_data.get("history", True):
            history_data.extend(tool_results)
        else:
            messages_to_send.append(assistant_msg)
            messages_to_send.extend(tool_results)

        research_compaction_due = False
        if agent_mode == AGENT_MODE_DEEP_RESEARCH:
            deep_research_search_count += sum(
                spec.name == "web_search" for spec in execution_specs
            )
            deep_research_scrape_count += sum(
                spec.name == "web_scrape" for spec in execution_specs
            )
            research_compaction_due = (
                deep_research_search_count >= deep_research_next_compaction
                or deep_research_scrape_count >= deep_research_next_scrape_compaction
            )
            scrape_compaction_due = (
                deep_research_scrape_count >= deep_research_next_scrape_compaction
            )
            research_compaction_due = search_compaction_due or scrape_compaction_due
            if research_compaction_due:
                compaction_source = (
                    history_data
                    if session_data.get("history", True)
                    else messages_to_send
                )
                deep_research_compacted_prefix, _ = compact_deep_research_messages(
                    compaction_source,
                    user_input,
                    max_checkpoint_chars=deep_research_checkpoint_chars,
                )
                if session_data.get("history", True):
                    deep_research_history_checkpoint_index = len(history_data)
                else:
                    messages_to_send = list(deep_research_compacted_prefix)
                while deep_research_next_compaction <= deep_research_search_count:
                    deep_research_next_compaction += DEEP_RESEARCH_COMPACT_INTERVAL
                while deep_research_next_scrape_compaction <= deep_research_scrape_count:
                    deep_research_next_scrape_compaction += DEEP_RESEARCH_SCRAPE_COMPACT_INTERVAL

        if vault_index_loop_state.get("blocked_reason"):
            message = str(vault_index_loop_state["blocked_reason"])
            if session_data.get("history", True):
                history_data.append({"role": "assistant", "content": message})
            yield {"type": "status", "message": message, "color": "yellow"}
            yield {"type": "content_chunk", "text": message}
            save_session_snapshot(
                origin_name,
                session_data,
                history_data,
                generation_start_session=generation_start_session,
            )
            yield {
                "type": "done",
                "state": "failed",
                "error": message,
                "history": history_data,
                "active_session_name": origin_name,
                "saved_sessions": list_saved_sessions(),
            }
            break

        continuation_prompt = build_tool_continuation_prompt(
            user_input,
            vault_index_loop_state,
        )
        if agent_mode == AGENT_MODE_ULTRA:
            continuation_prompt += "\n\n" + ULTRA_MODE_PROMPT.format(
                user_input=user_input,
            )
        elif agent_mode == AGENT_MODE_DEEP_RESEARCH:
            continuation_prompt += "\n\n" + DEEP_RESEARCH_SYNTHESIS_PROMPT.format(
                user_input=user_input,
            )

        if session_data.get("history", True):
            reminder = {
                "role": "user",
                "content": continuation_prompt,
            }
            research_history = history_data
            if (
                agent_mode == AGENT_MODE_DEEP_RESEARCH
                and deep_research_compacted_prefix is not None
            ):
                research_history = [
                    *deep_research_compacted_prefix,
                    *history_data[deep_research_history_checkpoint_index:],
                ]
            messages_to_send = prepare_messages_for_model(
                [*research_history, reminder],
                session_data,
                tools=TOOL_SCHEMAS,
                extra_reserved_tokens=CONTEXT_TOOL_LOOP_RESERVE,
            )
        else:
            messages_to_send.append({
                "role": "user",
                "content": continuation_prompt,
            })
            messages_to_send = prepare_messages_for_model(
                messages_to_send,
                session_data,
                tools=TOOL_SCHEMAS,
                extra_reserved_tokens=CONTEXT_TOOL_LOOP_RESERVE,
            )

    if session_data.get("history", True):
        _check_and_compact_history(history_data, session_data)


def generate_chat_events(
    user_input: str,
    session_data: dict,
    history_data: list[dict],
    session_name: str | None = None,
    *,
    cancellation_token: CancellationToken | None = None,
    generation_id: str | None = None,
    publish_global: bool = True,
    client_id: str = LEGACY_CLIENT_ID,
):
    """Yield one and only one terminal ``done`` event for a connected stream."""
    token = cancellation_token or CancellationToken()
    implementation = _generate_chat_events_impl(
        user_input,
        session_data,
        history_data,
        session_name,
        cancellation_token=token,
        generation_id=generation_id,
        publish_global=publish_global,
        client_id=client_id,
    )
    terminal_payload: dict | None = None
    active_name = session_name or "Active Session"
    terminal_state = TerminalState.FAILED
    terminal_detail: str | None = None
    try:
        for event in implementation:
            token.raise_if_cancelled()
            if event.get("type") == "conversation_started":
                active_name = str(event.get("session_name") or active_name)
            if event.get("type") == "done":
                terminal_payload = dict(event)
                active_name = str(event.get("active_session_name") or active_name)
                continue
            yield event
        if token.cancelled:
            terminal_state = TerminalState.CANCELLED
            terminal_detail = token.reason
        elif terminal_payload is not None:
            try:
                terminal_state = TerminalState(terminal_payload.get("state", "completed"))
            except ValueError:
                terminal_state = TerminalState.FAILED
            terminal_detail = terminal_payload.get("error")
        else:
            terminal_detail = "Generation ended without a completion result"
    except OperationCancelled as exc:
        terminal_state = TerminalState.CANCELLED
        terminal_detail = str(exc)
    except GeneratorExit:
        token.cancel("Client disconnected")
        raise
    except Exception as exc:
        terminal_state = TerminalState.FAILED
        terminal_detail = f"Generation failed: {exc}"
        yield {"type": "status", "message": terminal_detail, "color": "red"}
    finally:
        try:
            implementation.close()
        except Exception:
            pass

    payload = terminal_payload or {
        "history": history_data,
        "active_session_name": active_name,
        "saved_sessions": list_saved_sessions(),
    }
    payload.update({
        "type": "done",
        "state": terminal_state.value,
        "generation_id": generation_id,
        # The command path can change profiles and model options without a
        # separate settings request. Keep the web UI synchronized with the
        # effective runtime that will own the next turn.
        "settings": deepcopy(session_data),
        "runtime": _runtime_payload(session_data),
    })
    if terminal_detail:
        payload["error"] = terminal_detail
    yield payload


# ── HTTP Handler ──────────────────────────────────────────────────────

class AgentHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """Custom HTTP request handler for the agent's web interface.
    
    Handles serving static assets, managing session state via REST API,
    and providing an SSE endpoint for streaming chat generation.
    """
    
    def log_message(self, format, *args):
        """Mute standard output logs to keep the server output clean."""
        pass

    def _client_id(self, body: dict | None = None, query: dict | None = None) -> str:
        value = body.get("client_id") if body else None
        if not value and query:
            value = (query.get("client_id") or [None])[0]
        if not value:
            value = self.headers.get("X-Selene-Client-ID")
        return normalize_runtime_id(value, fallback=LEGACY_CLIENT_ID)

    def _origin_allowed(self) -> bool:
        """Reject cross-site browser writes to the loopback control API."""
        origin = str(self.headers.get("Origin") or "").strip()
        if not origin:
            # Native clients and same-origin requests may omit Origin.
            return True
        configured = str(os.environ.get("ALLOWED_ORIGIN") or "").rstrip("/")
        if configured and origin.rstrip("/") == configured:
            return True
        try:
            parsed = urlsplit(origin)
        except ValueError:
            return False
        request_host = str(self.headers.get("Host") or "").casefold()
        return (
            parsed.scheme in {"http", "https"}
            and bool(request_host)
            and parsed.netloc.casefold() == request_host
        )

    @staticmethod
    def _publish_legacy_view(client_id: str) -> None:
        if client_id != LEGACY_CLIENT_ID:
            return
        view = CLIENT_SESSIONS.snapshot(client_id)
        with _GLOBAL_STATE_LOCK:
            GLOBAL_STATE["session"] = deepcopy(view.session)
            GLOBAL_STATE["history"] = deepcopy(view.history)
            GLOBAL_STATE["active_session_name"] = view.active_session_name

    def send_json_response(self, status_code: int, data: dict):
        """Helper to send a JSON-encoded HTTP response.
        
        Args:
            status_code (int): The HTTP status code to return.
            data (dict): The dictionary to serialize to JSON.
        """
        payload = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')

        allowed_origin = os.environ.get('ALLOWED_ORIGIN')
        if allowed_origin:
            self.send_header('Access-Control-Allow-Origin', allowed_origin)

        self.end_headers()
        self.wfile.write(payload)

    def read_json_body(self) -> dict:
        """Helper to read and parse the JSON body of an HTTP POST request.
        
        Returns:
            dict: The parsed JSON body.
        """
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length <= 0:
            return {}
        if content_length > 2 * 1024 * 1024:
            raise ValueError("JSON request body exceeds the 2 MiB limit")
        body = self.rfile.read(content_length)
        value = json.loads(body.decode('utf-8'))
        if not isinstance(value, dict):
            raise ValueError("JSON request body must be an object")
        return value

    def serve_static_file(self, filename: str, content_type: str):
        """Serve a static file from the STATIC_DIR.
        
        Args:
            filename (str): The name of the file to serve.
            content_type (str): The MIME type of the file.
        """
        filepath = os.path.join(STATIC_DIR, filename)
        if not os.path.isfile(filepath):
            self.send_error(404, "File Not Found")
            return
            
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Pragma', 'no-cache')

            allowed_origin = os.environ.get('ALLOWED_ORIGIN')
            if allowed_origin:
                self.send_header('Access-Control-Allow-Origin', allowed_origin)

            self.end_headers()
            self.wfile.write(content)
        except OSError:
            self.send_error(500, "Internal Server Error")

    def do_GET(self):
        """Handle incoming HTTP GET requests for static files and settings."""
        parsed = urlsplit(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        # 1. Routing for Home and Assets
        if path == '/' or path == '/index.html':
            self.serve_static_file('index.html', 'text/html')
            return
        elif path == '/style.css':
            self.serve_static_file('style.css', 'text/css')
            return
        elif path == '/app.js':
            self.serve_static_file('app.js', 'application/javascript')
            return
            
        # 2. Routing for Settings/State load
        elif path == '/api/settings':
            try:
                client_id = self._client_id(query=query)
            except ValueError as exc:
                self.send_json_response(400, {"status": "error", "error": str(exc)})
                return
            saved = list_saved_sessions()
            view = CLIENT_SESSIONS.snapshot(client_id)
            probe = OllamaService(get_runtime_config(view.session)).probe(
                model=MODEL_NAME,
                timeout=3,
            )
            response_data = {
                "settings": view.session,
                "history": view.history,
                "saved_sessions": saved,
                "active_session_name": view.active_session_name,
                "model_name": MODEL_NAME,
                "ollama_status": "Online" if probe.api_available else "Offline",
                "ollama_reason": probe.reason,
                "runtime": _runtime_payload(view.session),
            }
            self.send_json_response(200, response_data)
            return

        elif path == '/api/generations':
            try:
                client_id = self._client_id(query=query)
            except ValueError as exc:
                self.send_json_response(400, {"status": "error", "error": str(exc)})
                return
            operations = [
                operation
                for operation in ACTIVE_GENERATIONS.active_operations()
                if operation["client_id"] == client_id
            ]
            self.send_json_response(200, {"active_operations": operations})
            return
            
        elif path == '/favicon.ico' or path == '/favicon.png':
            self.serve_static_file('favicon.png', 'image/png')
            return
            
        elif path == '/avatar.png':
            self.serve_static_file('avatar.png', 'image/jpeg')
            return
            
        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        if not self._origin_allowed():
            self.send_error(403, "Cross-origin API access is not allowed")
            return
        self.send_response(200)

        allowed_origin = os.environ.get('ALLOWED_ORIGIN')
        if allowed_origin:
            self.send_header('Access-Control-Allow-Origin', allowed_origin)

        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Selene-Client-ID')
        self.end_headers()

    def do_POST(self):
        """Handle incoming HTTP POST requests for API endpoints (chat, save/load/clear session)."""
        if not self._origin_allowed():
            self.send_json_response(403, {
                "status": "error",
                "error": "Cross-origin API access is not allowed",
            })
            return
        path = urlsplit(self.path).path

        if path == '/api/shutdown':
            expected_owner = str(os.environ.get('SELENE_BACKEND_OWNER') or '')
            provided_owner = str(self.headers.get('X-Selene-Backend-Owner') or '')
            if not expected_owner or not hmac.compare_digest(provided_owner, expected_owner):
                self.send_json_response(403, {
                    "status": "error",
                    "error": "Backend shutdown ownership could not be verified",
                })
                return
            ACTIVE_GENERATIONS.cancel_all("Electron requested backend shutdown")
            self.send_json_response(202, {"status": "shutting-down"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        if path == '/api/settings':
            try:
                body = self.read_json_body()
                client_id = self._client_id(body)
                body.pop("client_id", None)
                current = CLIENT_SESSIONS.snapshot(client_id)
                settings = _normalize_session_settings(body, fallback=current.session)
                CLIENT_SESSIONS.update_settings(client_id, settings)
                autosave_session(client_id)
                self._publish_legacy_view(client_id)
                self.send_json_response(200, {
                    "status": "success",
                    "settings": settings,
                    "runtime": _runtime_payload(settings),
                })
            except Exception as exc:
                self.send_json_response(400, {"status": "error", "error": str(exc)})
            return

        if path == '/api/cancel-generation':
            try:
                body = self.read_json_body()
                client_id = self._client_id(body)
                generation_id = normalize_runtime_id(body.get("generation_id"))
                lease = ACTIVE_GENERATIONS.cancel(
                    generation_id,
                    client_id,
                    reason="Cancelled by the requesting browser tab",
                )
                get_ollama_coordinator().cancel_owner(
                    f"web:{lease.generation_id}",
                    reason="Cancelled by the requesting browser tab",
                )
                self.send_json_response(202, {
                    "status": "cancelling",
                    "generation_id": generation_id,
                })
            except GenerationOwnershipError as exc:
                self.send_json_response(403, {"status": "error", "error": str(exc)})
            except (KeyError, ValueError) as exc:
                self.send_json_response(404, {"status": "error", "error": str(exc)})
            return

        if path == '/api/chat':
            lease = None
            generator = None
            terminal_state = TerminalState.FAILED
            terminal_detail = "Generation did not start"
            active_name = "Active Session"
            try:
                body = self.read_json_body()
                client_id = self._client_id(body)
                user_input = str(body.get("message", "")).strip()
                if not user_input:
                    raise ValueError("Message cannot be empty")
                # Session identity creation, generation ownership, and delete
                # use one short lifecycle lock. This closes both the blank-chat
                # duplicate race and the delete-then-recreate TOCTOU window.
                with _SESSION_LIFECYCLE_LOCK:
                    requested_name = os.path.basename(str(body.get("session_name", "")))
                    view = CLIENT_SESSIONS.snapshot(client_id)
                    aliases = {"", "Active Session", "New conversation"}
                    if requested_name == view.active_session_name or requested_name in aliases:
                        generation_session_name = view.active_session_name
                        generation_session = deepcopy(view.session)
                        generation_history = deepcopy(view.history)
                    else:
                        generation_session, generation_history = _read_session_snapshot(requested_name)
                        generation_session_name = requested_name
                        CLIENT_SESSIONS.select(
                            client_id,
                            requested_name,
                            generation_session,
                            generation_history,
                        )

                    # Give an unsaved conversation a stable disk/session
                    # identity before acquiring ownership.
                    if generation_session_name in aliases:
                        old_name = generation_session_name
                        generation_session_name = save_session_snapshot(
                            generation_session_name,
                            generation_session,
                            generation_history,
                        )
                        CLIENT_SESSIONS.commit_generation(
                            client_id,
                            old_name,
                            generation_session_name,
                            generation_session,
                            generation_history,
                        )
                    active_name = generation_session_name
                    generation_start_settings = deepcopy(generation_session)
                    lease = ACTIVE_GENERATIONS.begin(
                        generation_session_name,
                        client_id,
                        body.get("generation_id"),
                    )
            except GenerationConflict as exc:
                self.send_json_response(409, {"status": "error", "error": str(exc)})
                return
            except Exception as exc:
                self.send_json_response(400, {"status": "error", "error": str(exc)})
                return

            try:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'close')
                self.send_header('X-Accel-Buffering', 'no')
                allowed_origin = os.environ.get('ALLOWED_ORIGIN')
                if allowed_origin:
                    self.send_header('Access-Control-Allow-Origin', allowed_origin)
                self.end_headers()

                generator = generate_chat_events(
                    user_input,
                    generation_session,
                    generation_history,
                    generation_session_name,
                    cancellation_token=lease.token,
                    generation_id=lease.generation_id,
                    publish_global=client_id == LEGACY_CLIENT_ID,
                    client_id=client_id,
                )
                for event in generator:
                    if event.get("type") == "done":
                        active_name = str(event.get("active_session_name") or active_name)
                        try:
                            terminal_state = TerminalState(event.get("state", "completed"))
                        except ValueError:
                            terminal_state = TerminalState.FAILED
                        terminal_detail = event.get("error")
                    data_line = f"data: {json.dumps(event)}\n\n"
                    self.wfile.write(data_line.encode('utf-8'))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                lease.token.cancel("Client disconnected")
                terminal_state = TerminalState.CANCELLED
                terminal_detail = "Client disconnected"
            except Exception as exc:
                terminal_state = TerminalState.CANCELLED if lease.token.cancelled else TerminalState.FAILED
                terminal_detail = lease.token.reason if lease.token.cancelled else str(exc)
            finally:
                if generator is not None:
                    try:
                        generator.close()
                    except Exception:
                        pass
                CLIENT_SESSIONS.commit_generation(
                    client_id,
                    generation_session_name,
                    active_name,
                    generation_session,
                    generation_history,
                    generation_start_settings,
                )
                self._publish_legacy_view(client_id)
                ACTIVE_GENERATIONS.finish(lease, terminal_state, terminal_detail)
                self.close_connection = True
            return

        try:
            body = self.read_json_body() if self.headers.get('Content-Length') else {}
            client_id = self._client_id(body)
        except Exception as exc:
            self.send_json_response(400, {"status": "error", "error": str(exc)})
            return

        if path == '/api/save-session':
            try:
                view = CLIENT_SESSIONS.snapshot(client_id)
                filename = save_session(
                    str(body.get("name", "")).strip(),
                    view.session,
                    view.history,
                    client_id,
                )
                self._publish_legacy_view(client_id)
                self.send_json_response(200, {"status": "success", "filename": filename})
            except Exception as exc:
                self.send_json_response(500, {"status": "error", "error": str(exc)})
            return

        if path == '/api/load-session':
            try:
                load_session(str(body.get("name", "")).strip(), client_id)
                self._publish_legacy_view(client_id)
                view = CLIENT_SESSIONS.snapshot(client_id)
                self.send_json_response(200, {
                    "status": "success",
                    "settings": view.session,
                    "history": view.history,
                    "active_session_name": view.active_session_name,
                })
            except Exception as exc:
                self.send_json_response(404, {"status": "error", "error": str(exc)})
            return

        if path == '/api/new-session':
            autosave_session(client_id)
            view = CLIENT_SESSIONS.new_session(client_id)
            self._publish_legacy_view(client_id)
            self.send_json_response(200, {
                "status": "success",
                "settings": view.session,
                "saved_sessions": list_saved_sessions(),
                "active_session_name": view.active_session_name,
            })
            return

        if path == '/api/delete-session':
            name = os.path.basename(str(body.get("name", "")))
            try:
                delete_error = None
                with _SESSION_LIFECYCLE_LOCK:
                    if not name or name not in list_saved_sessions():
                        delete_error = (404, "Session not found")
                    elif not ACTIVE_GENERATIONS.wait_for_session_idle(name, client_id, 1.0):
                        delete_error = (
                            409,
                            "Session still has an active generation; cancel it before deletion",
                        )
                    else:
                        with _session_lock(name):
                            os.remove(os.path.join(_SESSIONS_DIR, name))
                        CLIENT_SESSIONS.remove_session(name)
                if delete_error is not None:
                    status, error = delete_error
                    self.send_json_response(status, {"status": "error", "error": error})
                    return
                self._publish_legacy_view(client_id)
                view = CLIENT_SESSIONS.snapshot(client_id)
                self.send_json_response(200, {
                    "status": "success",
                    "saved_sessions": list_saved_sessions(),
                    "active_session_name": view.active_session_name,
                })
            except OSError as exc:
                self.send_json_response(500, {"status": "error", "error": str(exc)})
            return

        if path == '/api/clear-session':
            view = CLIENT_SESSIONS.snapshot(client_id)
            settings = deepcopy(view.session)
            settings["system"] = ""
            cleared = CLIENT_SESSIONS.select(client_id, "Active Session", settings, [])
            self._publish_legacy_view(client_id)
            self.send_json_response(200, {
                "status": "success",
                "settings": cleared.session,
                "active_session_name": cleared.active_session_name,
            })
            return

        self.send_error(404, "Not Found")


# ── Threaded HTTP Server ──────────────────────────────────────────────

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """A threading version of the standard HTTPServer to handle concurrent requests."""
    daemon_threads = True
    block_on_close = False


def find_free_port() -> int:
    """Find and return an available port on the local system."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_web_server():
    """Starts the multi-threaded web server and opens the browser.
    
    Attempts to bind to port 5005. If unavailable, falls back to a random free port.
    It will also launch the default web browser automatically.
    """
    # Keep the user's latest UI/model preferences without reopening its chat.
    # Every application launch starts with a blank conversation; previous
    # conversations remain autosaved and can still be opened from the sidebar.
    saved_sessions = list_saved_sessions()
    if saved_sessions:
        try:
            saved_session, _saved_history = _read_session_snapshot(saved_sessions[0])
            GLOBAL_STATE["session"] = _normalize_session_settings(
                saved_session,
                fallback=GLOBAL_STATE["session"],
            )
        except PersistenceError as exc:
            print(f"Preserved malformed session settings: {exc}")
        except (OSError, ValueError, RuntimeConfigurationError):
            pass
    GLOBAL_STATE["history"] = []
    GLOBAL_STATE["active_session_name"] = "Active Session"
    CLIENT_SESSIONS.set_default_session(GLOBAL_STATE["session"])
    CLIENT_SESSIONS.select(
        LEGACY_CLIENT_ID,
        "Active Session",
        GLOBAL_STATE["session"],
        [],
    )

    # Attempt to bind to default port 5005 first, then fall back to random port
    host = '127.0.0.1'
    if '--public' in sys.argv:
        host = '0.0.0.0'
        
    try:
        server = ThreadingHTTPServer((host, 5005), AgentHTTPRequestHandler)
    except OSError:
        # Binding directly to port 0 avoids a find-then-bind race.
        server = ThreadingHTTPServer((host, 0), AgentHTTPRequestHandler)
    port = int(server.server_address[1])
        
    url = f"http://127.0.0.1:{port}"
    print(f"\n🚀 Starting Selene Web Interface at {url}")
    print(f"ELECTRON_PORT:{port}", flush=True)
    
    if host == '0.0.0.0':
        local_ip = "127.0.0.1"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass
        print(f"📡 Accessible across your local network at: http://{local_ip}:{port}")
        
    if "--no-browser" not in sys.argv:
        print(f"Opening default web browser...\n")
        
        def open_browser():
            time.sleep(0.5)
            result = open_url_native(url)
            if not result.ok:
                print(f"Could not open the browser automatically: {result.error}")
            
        threading.Thread(target=open_browser, daemon=True).start()
    
    previous_handlers: dict[int, object] = {}

    def request_shutdown(signum, _frame):
        print("\nStopping web server...")
        ACTIVE_GENERATIONS.cancel_all("Selene is shutting down")
        # HTTPServer.shutdown must run outside the serve_forever thread.
        threading.Thread(target=server.shutdown, daemon=True).start()

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                previous_handlers[signum] = signal.signal(signum, request_shutdown)
            except (OSError, ValueError):
                pass
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        ACTIVE_GENERATIONS.cancel_all("Selene web server stopped")
        server.server_close()
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                pass
        get_ollama_coordinator().shutdown(cancel_active=True, wait=True, timeout=5)
        shutdown_tool_runner(wait=False)
