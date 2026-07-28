"""Store, preview, and safely execute reusable local workflow macros."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path

from agent.cancellation import CancellationToken, OperationCancelled
from agent.persistence import PersistenceError, atomic_write_json, read_json_preserved
from agent.platform_runtime import (
    get_runtime_paths,
    open_url_native,
    path_is_within,
    spawn_detached,
    terminate_process_tree,
    validate_http_url,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = get_runtime_paths().data_dir
STORE_PATH = DATA_DIR / "routines.json"
LEGACY_STORE_PATH = PROJECT_ROOT / ".selene" / "routines.json"
MAX_ACTIONS = 50
MAX_TRIGGERS = 25
MAX_ROUTINES = 200
MAX_STORE_BYTES = 2 * 1024 * 1024
MAX_ROUTINE_NAME_CHARS = 200
MAX_COMMAND_ARGUMENTS = 100
MAX_ARGUMENT_CHARS = 4096
MAX_TOOL_ARGUMENT_JSON_CHARS = 32_000
MAX_ROUTINE_JSON_CHARS = 256_000
MAX_COMMAND_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_CAPTURE_TAIL_BYTES = 12_000
AUTOMATIC_ACTION_TYPES = {"open_app", "delay", "tool"}
AUTOMATIC_TOOL_NAMES = {"open_app", "launch_apps"}
CONFIRMATION_TOOL_NAMES = {*AUTOMATIC_TOOL_NAMES, "open_terminal_at_path"}
ALLOWED_ROUTINE_COMMANDS = {"python", "pytest", "git", "ls", "cat", "echo", "grep", "node", "npm"}
_STORE_LOCK = threading.RLock()


def _active_store_path() -> Path:
    """Choose one routine store without silently copying or moving legacy data."""
    if STORE_PATH.exists() or not LEGACY_STORE_PATH.is_file():
        return STORE_PATH
    return LEGACY_STORE_PATH


def _load() -> dict[str, dict]:
    store = _active_store_path()
    try:
        if store.stat().st_size > MAX_STORE_BYTES:
            raise PersistenceError(
                f"Routine state at '{store}' exceeds the {MAX_STORE_BYTES}-byte safety limit "
                "and was preserved without modification"
            )
        routines = read_json_preserved(store, expected_type=dict)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise PersistenceError(
            f"Routine state at '{store}' could not be read and was preserved: {exc}"
        ) from exc
    if len(routines) > MAX_ROUTINES:
        raise PersistenceError(
            f"Routine state at '{store}' contains more than {MAX_ROUTINES} routines "
            "and was preserved without modification"
        )
    invalid = [
        name
        for name, value in routines.items()
        if not isinstance(name, str)
        or not isinstance(value, dict)
        or not isinstance(value.get("triggers", []), list)
        or not isinstance(value.get("actions", []), list)
        or any(not isinstance(item, dict) for item in value.get("actions", []))
    ]
    if invalid:
        raise PersistenceError(
            f"Routine state at '{store}' contains invalid records and was preserved without modification"
        )
    if store == LEGACY_STORE_PATH and not STORE_PATH.exists():
        try:
            atomic_write_json(STORE_PATH, routines, private=True)
        except (OSError, TypeError, ValueError):
            # Reading and executing a valid legacy store is still preferable to
            # making every routine unavailable because migration is blocked.
            pass
    return routines


def _save(routines: dict[str, dict]) -> None:
    atomic_write_json(STORE_PATH, routines, private=True)


def _resolve(routines: dict[str, dict], name: str | None, trigger: str | None) -> tuple[str | None, dict | None]:
    if name and name in routines:
        return name, routines[name]
    normalized = (trigger or "").strip().casefold()
    exact_matches = []
    phrase_matches = []
    for routine_name, routine in routines.items():
        values = [routine_name, *routine.get("triggers", [])]
        normalized_values = [str(value).strip().casefold() for value in values if str(value).strip()]
        if normalized in normalized_values:
            exact_matches.append((routine_name, routine))
        elif any(value in normalized for value in normalized_values):
            phrase_matches.append((routine_name, routine))
    if len(exact_matches) == 1:
        return exact_matches[0]
    return phrase_matches[0] if not exact_matches and len(phrase_matches) == 1 else (None, None)


def _normalized_triggers(routine: dict, legacy_trigger: str | None = None) -> list[str]:
    """Normalize and deduplicate trigger phrases while accepting the old argument."""
    raw_triggers = routine.get("triggers", [])
    if not isinstance(raw_triggers, list):
        return []
    values = [*raw_triggers, *([legacy_trigger] if legacy_trigger else [])]
    triggers = []
    seen = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        cleaned = value.strip()
        normalized = cleaned.casefold()
        if normalized not in seen:
            seen.add(normalized)
            triggers.append(cleaned)
    return triggers


def _validate(routine: dict) -> list[str]:
    errors = []
    try:
        serialized = json.dumps(
            routine,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError):
        serialized = ""
        errors.append("routine must contain only finite JSON-compatible values")
    if serialized and len(serialized) > MAX_ROUTINE_JSON_CHARS:
        errors.append(
            f"routine exceeds the {MAX_ROUTINE_JSON_CHARS}-character serialized limit"
        )
    description = routine.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append("routine.description must clearly describe the routine and cannot be empty")
    elif len(description.strip()) > 500:
        errors.append("routine.description may contain at most 500 characters")
    raw_triggers = routine.get("triggers")
    if not isinstance(raw_triggers, list) or not raw_triggers:
        errors.append("routine.triggers must be a non-empty array of user phrases")
    elif len(raw_triggers) > MAX_TRIGGERS:
        errors.append(f"A routine may contain at most {MAX_TRIGGERS} triggers")
    else:
        normalized_triggers = set()
        for index, value in enumerate(raw_triggers):
            if not isinstance(value, str) or not value.strip():
                errors.append(f"triggers[{index}] must be a non-empty string")
            elif len(value.strip()) > 200:
                errors.append(f"triggers[{index}] may contain at most 200 characters")
            else:
                normalized = value.strip().casefold()
                if normalized in normalized_triggers:
                    errors.append(f"triggers[{index}] duplicates another trigger")
                normalized_triggers.add(normalized)
    actions = routine.get("actions", [])
    if not isinstance(actions, list) or not actions:
        errors.append("routine.actions must be a non-empty array")
        return errors
    if len(actions) > MAX_ACTIONS:
        errors.append(f"A routine may contain at most {MAX_ACTIONS} actions")
    for index, item in enumerate(actions):
        if not isinstance(item, dict) or item.get("type") not in {"command", "open_app", "open_url", "delay", "tool"}:
            errors.append(f"actions[{index}] has an unsupported type")
        elif item.get("type") == "command":
            if not isinstance(item.get("argv"), list) or not item.get("argv"):
                errors.append(f"actions[{index}].argv must be a non-empty argument array; shell strings are not accepted")
            else:
                argv = item["argv"]
                if len(argv) > MAX_COMMAND_ARGUMENTS:
                    errors.append(f"actions[{index}].argv may contain at most {MAX_COMMAND_ARGUMENTS} arguments")
                invalid_arg = any(
                    not isinstance(value, str) or not value or len(value) > MAX_ARGUMENT_CHARS or "\0" in value
                    for value in argv
                )
                if invalid_arg:
                    errors.append(f"actions[{index}].argv entries must be non-empty strings of at most {MAX_ARGUMENT_CHARS} characters")
                executable = str(argv[0])
                if executable not in ALLOWED_ROUTINE_COMMANDS:
                    errors.append(f"actions[{index}].argv[0] must be an allowed command ({', '.join(sorted(ALLOWED_ROUTINE_COMMANDS))}); found '{executable}'")
                try:
                    timeout = float(item.get("timeout", 60))
                    if not math.isfinite(timeout) or not 1 <= timeout <= 600:
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    errors.append(f"actions[{index}].timeout must be between 1 and 600 seconds")
                cwd = item.get("cwd", ".")
                if not isinstance(cwd, str) or not cwd.strip() or len(cwd) > MAX_ARGUMENT_CHARS or "\0" in cwd:
                    errors.append(f"actions[{index}].cwd must be a valid project-relative path")
                else:
                    try:
                        requested_cwd = (PROJECT_ROOT / cwd).resolve()
                    except (OSError, RuntimeError):
                        requested_cwd = None
                    if requested_cwd is None or not path_is_within(requested_cwd, PROJECT_ROOT):
                        errors.append(f"actions[{index}].cwd must stay inside the project workspace")
        elif item.get("type") == "open_app":
            app_name = item.get("app_name")
            if (
                not isinstance(app_name, str)
                or not app_name.strip()
                or len(app_name) > 128
                or any(ord(character) < 32 for character in app_name)
            ):
                errors.append(f"actions[{index}].app_name must be an installed application display name")
        elif item.get("type") == "tool":
            tool_name = item.get("tool_name")
            if (
                not isinstance(tool_name, str)
                or not tool_name.strip()
                or len(tool_name) > 200
                or any(ord(character) < 32 for character in tool_name)
            ):
                errors.append(f"actions[{index}].tool_name must be a registered tool name")
            elif tool_name == "automated_routine_executor":
                errors.append(f"actions[{index}] cannot recursively call automated_routine_executor")
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict):
                errors.append(f"actions[{index}].arguments must be an object")
            else:
                try:
                    serialized_arguments = json.dumps(
                        arguments,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                except (TypeError, ValueError, RecursionError):
                    errors.append(
                        f"actions[{index}].arguments must contain only finite JSON-compatible values"
                    )
                else:
                    if len(serialized_arguments) > MAX_TOOL_ARGUMENT_JSON_CHARS:
                        errors.append(
                            f"actions[{index}].arguments exceeds the "
                            f"{MAX_TOOL_ARGUMENT_JSON_CHARS}-character limit"
                        )
        elif item.get("type") == "open_url":
            try:
                validate_http_url(item.get("url"))
            except ValueError as exc:
                errors.append(f"actions[{index}].url is invalid: {exc}")
        elif item.get("type") == "delay":
            try:
                seconds = float(item.get("seconds", 1))
                if not math.isfinite(seconds) or not 0 <= seconds <= 30:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                errors.append(f"actions[{index}].seconds must be between 0 and 30")
        if isinstance(item, dict) and "continue_on_error" in item and not isinstance(item["continue_on_error"], bool):
            errors.append(f"actions[{index}].continue_on_error must be boolean")
    if routine.get("allow_automatic") is True:
        unsafe = sorted({
            str(item.get("type"))
            for item in actions
            if isinstance(item, dict) and (
                item.get("type") not in AUTOMATIC_ACTION_TYPES
                or (item.get("type") == "tool" and item.get("tool_name") not in AUTOMATIC_TOOL_NAMES)
            )
        })
        if unsafe:
            errors.append(
                "allow_automatic is limited to app-launch tools and delay actions; found: "
                + ", ".join(unsafe)
            )
    return errors


def _stored_routine_errors(routine: dict) -> list[str]:
    """Validate persisted data again before previewing or executing it."""
    candidate = {
        "description": routine.get("description"),
        "triggers": routine.get("triggers"),
        "actions": routine.get("actions"),
    }
    if routine.get("automatic_approved") is True:
        candidate["allow_automatic"] = True
    return _validate(candidate)


def _trigger_conflicts(
    routines: dict[str, dict],
    candidate_name: str,
    triggers: list[str],
) -> dict[str, list[str]]:
    """Find exact trigger collisions that would make deterministic dispatch ambiguous."""
    requested = {value.strip().casefold(): value for value in triggers}
    conflicts: dict[str, list[str]] = {}
    for existing_name, existing in routines.items():
        if existing_name.casefold() == candidate_name.casefold():
            continue
        overlap = []
        for value in existing.get("triggers", []):
            if isinstance(value, str) and value.strip().casefold() in requested:
                overlap.append(requested[value.strip().casefold()])
        if overlap:
            conflicts[existing_name] = sorted(set(overlap), key=str.casefold)
    return conflicts


def _trigger_matches(routine: dict, trigger: str | None) -> bool:
    normalized = (trigger or "").strip().casefold()
    return bool(normalized) and any(
        normalized == str(value).strip().casefold()
        for value in routine.get("triggers", [])
    )


def _is_safe_automatic_routine(routine: dict) -> bool:
    """Recheck stored data before bypassing per-run confirmation."""
    actions = routine.get("actions")
    if not isinstance(actions, list) or not actions:
        return False
    for item in actions:
        if not isinstance(item, dict) or item.get("type") not in AUTOMATIC_ACTION_TYPES:
            return False
        if item.get("type") == "open_app":
            has_name = isinstance(item.get("app_name"), str) and bool(item["app_name"].strip())
            legacy_argv = item.get("argv")
            has_legacy_name = isinstance(legacy_argv, list) and bool(legacy_argv)
            if not has_name and not has_legacy_name:
                return False
        if item.get("type") == "tool":
            if item.get("tool_name") not in AUTOMATIC_TOOL_NAMES:
                return False
            if not isinstance(item.get("arguments", {}), dict):
                return False
    return True


def _run_registered_tool(
    tool_name: str,
    arguments: dict,
    action_type: str = "tool",
    cancellation_token: CancellationToken | None = None,
) -> dict:
    """Call tools through the shared registry used by normal agent tool calls."""
    if tool_name == "automated_routine_executor":
        raise ValueError("A routine cannot recursively invoke itself")

    # Imported lazily because registry.py imports this module while constructing
    # the shared dispatch map.
    from agent.tool_runner import execute_tool_call, normalize_tool_calls
    from tools.registry import TOOL_DISPATCH, TOOL_SCHEMA_BY_NAME

    handler = TOOL_DISPATCH.get(tool_name)
    if handler is None:
        raise ValueError(f"Unknown registered tool: {tool_name}")

    call_arguments = dict(arguments)
    # A confirmed manual routine run is approval for every previewed action.
    # Persistently automatic routines are separately restricted to the two
    # app-launch tools, so this cannot silently approve arbitrary tool effects.
    target_schema = TOOL_SCHEMA_BY_NAME.get(tool_name, {})
    confirmed_schema = target_schema.get("properties", {}).get("confirmed", {})
    if (
        tool_name in CONFIRMATION_TOOL_NAMES
        or confirmed_schema.get("type") == "boolean"
    ):
        call_arguments["confirmed"] = True
    spec = normalize_tool_calls([{
        "function": {"name": tool_name, "arguments": call_arguments}
    }])[0]
    execution = execute_tool_call(spec, cancellation_token=cancellation_token)
    raw_result = execution.content
    if isinstance(raw_result, str):
        try:
            result = json.loads(raw_result)
        except json.JSONDecodeError:
            result = {"output": raw_result}
    else:
        result = raw_result

    failed = not execution.ok or (isinstance(result, dict) and (
        "error" in result or result.get("success") is False or result.get("ok") is False
    ))
    return {
        "type": action_type,
        "ok": not failed,
        "tool_name": tool_name,
        "status": execution.status.value,
        "result": result,
    }


def _run_action(
    item: dict,
    cancellation_token: CancellationToken | None = None,
) -> dict:
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    action_type = item["type"]
    if action_type == "delay":
        seconds = max(0.0, min(float(item.get("seconds", 1)), 30.0))
        if cancellation_token and cancellation_token.wait(seconds):
            cancellation_token.raise_if_cancelled()
        elif not cancellation_token:
            time.sleep(seconds)
        return {"type": action_type, "ok": True, "seconds": seconds}
    if action_type == "open_url":
        launch = open_url_native(item["url"])
        return {"type": action_type, **launch.as_dict(), "url": item["url"]}
    if action_type == "open_app":
        app_name = item.get("app_name")
        if not app_name and isinstance(item.get("argv"), list) and item["argv"]:
            app_name = str(item["argv"][0])
        if not app_name:
            raise ValueError("open_app requires app_name; command arguments are not permitted")
        result = _run_registered_tool(
            "open_app",
            {"app_name": str(app_name)},
            action_type,
            cancellation_token,
        )
        result["app_name"] = app_name
        return result
    if action_type == "tool":
        return _run_registered_tool(
            item["tool_name"],
            item.get("arguments", {}),
            cancellation_token=cancellation_token,
        )
    argv = [str(value) for value in item["argv"]]
    if not argv:
        raise ValueError("argv cannot be empty")
    if argv[0] not in ALLOWED_ROUTINE_COMMANDS:
        raise ValueError(f"Command '{argv[0]}' is not permitted")
    requested_cwd = (PROJECT_ROOT / str(item.get("cwd", "."))).resolve()
    if not path_is_within(requested_cwd, PROJECT_ROOT):
        raise ValueError("Command cwd must stay inside the project workspace")
    timeout = max(1.0, min(float(item.get("timeout", 60)), 600.0))
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        handle = spawn_detached(
            argv,
            cwd=requested_cwd,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        deadline = time.monotonic() + timeout
        timed_out = False
        output_limit_exceeded = False
        termination_confirmed = False
        while handle.poll() is None:
            if cancellation_token and cancellation_token.wait(0.05):
                if not terminate_process_tree(handle):
                    raise RuntimeError(
                        "Cancellation was requested, but termination of the owned process tree could not be confirmed"
                    )
                cancellation_token.raise_if_cancelled()
            if time.monotonic() >= deadline:
                timed_out = True
                termination_confirmed = terminate_process_tree(handle)
                break
            output_bytes = os.fstat(stdout_file.fileno()).st_size + os.fstat(
                stderr_file.fileno()
            ).st_size
            if output_bytes > MAX_COMMAND_OUTPUT_BYTES:
                output_limit_exceeded = True
                termination_confirmed = terminate_process_tree(handle)
                break
            if not cancellation_token:
                time.sleep(0.05)
        returncode = handle.process.poll()
        stdout_size = os.fstat(stdout_file.fileno()).st_size
        stderr_size = os.fstat(stderr_file.fileno()).st_size
        if (
            not output_limit_exceeded
            and stdout_size + stderr_size > MAX_COMMAND_OUTPUT_BYTES
        ):
            output_limit_exceeded = True
            termination_confirmed = (
                True if returncode is not None else terminate_process_tree(handle)
            )
            returncode = handle.process.poll()

        def output_tail(stream) -> str:
            stream.flush()
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - MAX_CAPTURE_TAIL_BYTES))
            return stream.read().decode("utf-8", errors="replace")

        result = {
            "type": action_type,
            "ok": not timed_out and not output_limit_exceeded and returncode == 0,
            "argv": argv,
            "returncode": returncode,
            "stdout": output_tail(stdout_file),
            "stderr": output_tail(stderr_file),
            "stdout_truncated": stdout_size > MAX_CAPTURE_TAIL_BYTES,
            "stderr_truncated": stderr_size > MAX_CAPTURE_TAIL_BYTES,
        }
        if timed_out:
            if termination_confirmed:
                result["error"] = (
                    f"Command exceeded its {timeout:g}s timeout and its owned process tree was stopped"
                )
            else:
                result["error"] = (
                    f"Command exceeded its {timeout:g}s timeout; termination of its owned process tree "
                    "could not be confirmed"
                )
        elif output_limit_exceeded:
            if termination_confirmed:
                result["error"] = (
                    f"Command output exceeded the {MAX_COMMAND_OUTPUT_BYTES}-byte limit "
                    "and its owned process tree was stopped"
                )
            else:
                result["error"] = (
                    f"Command output exceeded the {MAX_COMMAND_OUTPUT_BYTES}-byte limit; "
                    "termination of its owned process tree could not be confirmed"
                )
        return result


def automated_routine_executor(
    action: str,
    name: str | None = None,
    routine: dict | None = None,
    trigger: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    confirmed: bool = False,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """Manage workflow macros with per-run or narrowly scoped persistent approval."""
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    action = str(action or "").strip().casefold()
    for field_name, field_value in (
        ("dry_run", dry_run),
        ("overwrite", overwrite),
        ("confirmed", confirmed),
    ):
        if not isinstance(field_value, bool):
            return json.dumps({"error": f"{field_name} must be boolean"})
    try:
        with _STORE_LOCK:
            routines = _load()
    except PersistenceError as exc:
        return json.dumps({
            "error": str(exc),
            "store": str(_active_store_path()),
            "preserved": True,
        }, ensure_ascii=False)
    if action == "list":
        items = [{"name": key, "description": value.get("description", ""), "triggers": value.get("triggers", []), "action_count": len(value.get("actions", []))} for key, value in sorted(routines.items())]
        return json.dumps({"routines": items, "store": str(_active_store_path())}, ensure_ascii=False)
    if action == "define":
        if not name or not routine:
            return json.dumps({"error": "name and routine are required for define"})
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > MAX_ROUTINE_NAME_CHARS or any(ord(char) < 32 for char in name):
            return json.dumps({"error": f"name must contain 1-{MAX_ROUTINE_NAME_CHARS} printable characters"})
        name = name.strip()
        candidate = dict(routine)
        candidate["description"] = str(candidate.get("description", "")).strip()
        candidate["triggers"] = _normalized_triggers(candidate, trigger)
        errors = _validate(candidate)
        if not errors:
            from agent.tool_runner import validate_tool_arguments
            from tools.registry import TOOL_DISPATCH

            for index, item in enumerate(candidate["actions"]):
                if item.get("type") != "tool":
                    continue
                tool_name = item.get("tool_name")
                if tool_name not in TOOL_DISPATCH:
                    errors.append(
                        f"actions[{index}].tool_name is not registered: {tool_name}"
                    )
                    continue
                argument_errors = validate_tool_arguments(
                    tool_name, item.get("arguments", {})
                )
                errors.extend(
                    f"actions[{index}].arguments: {error}"
                    for error in argument_errors
                )
        if errors:
            return json.dumps({"error": "Invalid routine", "details": errors})
        wants_automatic = candidate.get("allow_automatic") is True
        if wants_automatic and not confirmed:
            return json.dumps({
                "error": "Persistent automatic execution requires confirmed=true after the user approves the preview.",
                "routine": candidate,
            }, ensure_ascii=False)
        try:
            with _STORE_LOCK:
                routines = _load()
                matching_name = next(
                    (
                        existing_name
                        for existing_name in routines
                        if existing_name.casefold() == name.casefold()
                    ),
                    None,
                )
                if matching_name is not None:
                    if not overwrite:
                        return json.dumps({
                            "error": (
                                f"Routine '{matching_name}' already exists. Set overwrite=true "
                                "and confirmed=true only after the user approves replacement."
                            ),
                            "existing": matching_name,
                            "preserved": True,
                        }, ensure_ascii=False)
                    if confirmed is not True:
                        return json.dumps({
                            "error": "Replacing a routine requires confirmed=true",
                            "existing": matching_name,
                            "preserved": True,
                        }, ensure_ascii=False)
                    name = matching_name
                conflicts = _trigger_conflicts(
                    routines, name, candidate["triggers"]
                )
                if conflicts:
                    return json.dumps({
                        "error": "Routine triggers must be unique across saved routines",
                        "conflicts": conflicts,
                        "preserved": True,
                    }, ensure_ascii=False)
                if matching_name is None and len(routines) >= MAX_ROUTINES:
                    return json.dumps({
                        "error": f"At most {MAX_ROUTINES} routines may be stored",
                        "preserved": True,
                    })
                routines[name] = {
                    "description": candidate["description"],
                    "triggers": candidate["triggers"],
                    "actions": candidate["actions"],
                    "automatic_approved": wants_automatic and confirmed is True,
                }
                _save(routines)
        except (PersistenceError, OSError, TypeError, ValueError) as exc:
            return json.dumps({"error": str(exc), "store": str(_active_store_path()), "preserved": True})
        return json.dumps({
            "ok": True,
            "defined": name,
            "description": candidate["description"],
            "triggers": candidate["triggers"],
            "action_count": len(candidate["actions"]),
            "automatic_approved": wants_automatic and confirmed is True,
            "store": str(_active_store_path()),
        })
    if action == "delete":
        if not confirmed:
            return json.dumps({"error": "Deleting a routine requires confirmed=true"})
        try:
            with _STORE_LOCK:
                routines = _load()
                if not name or name not in routines:
                    return json.dumps({"error": "Routine not found"})
                del routines[name]
                _save(routines)
        except (PersistenceError, OSError, TypeError, ValueError) as exc:
            return json.dumps({"error": str(exc), "store": str(_active_store_path()), "preserved": True})
        return json.dumps({"ok": True, "deleted": name})
    if action not in {"show", "run"}:
        return json.dumps({"error": "action must be list, define, show, run, or delete"})
    resolved_name, selected = _resolve(routines, name, trigger)
    if not selected:
        return json.dumps({"error": "No unique routine matched", "name": name, "trigger": trigger})
    stored_errors = _stored_routine_errors(selected)
    if stored_errors:
        return json.dumps({
            "error": "Stored routine is invalid and was not executed",
            "name": resolved_name,
            "details": stored_errors,
            "store": str(_active_store_path()),
            "preserved": True,
        }, ensure_ascii=False)
    automatic_trigger = (
        selected.get("automatic_approved") is True
        and _trigger_matches(selected, trigger)
        and _is_safe_automatic_routine(selected)
    )
    # ``show`` is always a preview. ``run`` executes unless the caller
    # explicitly asks for a dry run; requiring dry_run=false as well as
    # action="run" made approved routine calls silently do nothing.
    if action == "show" or dry_run is True:
        requirement = (
            "This exact trigger is persistently approved; call run with the trigger."
            if automatic_trigger
            else "Call run with confirmed=true after user approval."
        )
        return json.dumps({
            "name": resolved_name,
            "routine": selected,
            "dry_run": True,
            "automatic_trigger": automatic_trigger,
            "execution_required": requirement,
        }, ensure_ascii=False)
    if not confirmed and not automatic_trigger:
        return json.dumps({"error": "Routine execution requires confirmed=true after the user reviews the preview"})
    results = []
    for index, item in enumerate(selected["actions"]):
        try:
            result = _run_action(item, cancellation_token)
        except OperationCancelled:
            raise
        except Exception as exc:
            result = {"type": item.get("type"), "ok": False, "error": str(exc)}
        results.append({"index": index, **result})
        if not result.get("ok") and item.get("continue_on_error") is not True:
            break
    ok = len(results) == len(selected["actions"]) and all(item.get("ok") for item in results)
    return json.dumps({
        "ok": ok,
        "name": resolved_name,
        "actions_planned": len(selected["actions"]),
        "actions_executed": len(results),
        "stopped_early": len(results) < len(selected["actions"]),
        "results": results,
    }, ensure_ascii=False)
