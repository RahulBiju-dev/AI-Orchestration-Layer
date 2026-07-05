"""Store, preview, and safely execute reusable local workflow macros."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(
    os.path.abspath(os.path.expanduser(os.environ.get("SELENE_DATA_DIR", "~/.selene-agent")))
)
STORE_PATH = DATA_DIR / "routines.json"
LEGACY_STORE_PATH = PROJECT_ROOT / ".selene" / "routines.json"
MAX_ACTIONS = 50
AUTOMATIC_ACTION_TYPES = {"open_app", "delay", "tool"}
AUTOMATIC_TOOL_NAMES = {"open_app", "launch_apps"}


def _load() -> dict[str, dict]:
    store = STORE_PATH
    # Older releases kept routines inside the source checkout. Import that file
    # once so upgrades preserve existing routines while all future writes use
    # the per-user application data directory.
    if not store.exists() and LEGACY_STORE_PATH.is_file():
        try:
            value = json.loads(LEGACY_STORE_PATH.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                _save(value)
                return value
        except (OSError, json.JSONDecodeError):
            pass
    try:
        value = json.loads(store.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def _save(routines: dict[str, dict]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="routines-", suffix=".json", dir=STORE_PATH.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(routines, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, STORE_PATH)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resolve(routines: dict[str, dict], name: str | None, trigger: str | None) -> tuple[str | None, dict | None]:
    if name and name in routines:
        return name, routines[name]
    normalized = (trigger or "").strip().casefold()
    matches = []
    for routine_name, routine in routines.items():
        values = [routine_name, *routine.get("triggers", [])]
        if any(normalized == str(value).strip().casefold() for value in values):
            matches.append((routine_name, routine))
    return matches[0] if len(matches) == 1 else (None, None)


def _validate(routine: dict) -> list[str]:
    errors = []
    actions = routine.get("actions", [])
    if not isinstance(actions, list) or not actions:
        return ["routine.actions must be a non-empty array"]
    if len(actions) > MAX_ACTIONS:
        errors.append(f"A routine may contain at most {MAX_ACTIONS} actions")
    for index, item in enumerate(actions):
        if not isinstance(item, dict) or item.get("type") not in {"command", "open_app", "open_url", "delay", "tool"}:
            errors.append(f"actions[{index}] has an unsupported type")
        elif item.get("type") == "command" and not isinstance(item.get("argv"), list):
            errors.append(f"actions[{index}].argv must be an argument array; shell strings are not accepted")
        elif item.get("type") == "open_app" and not isinstance(item.get("app_name"), str):
            errors.append(f"actions[{index}].app_name must be an installed application display name")
        elif item.get("type") == "tool":
            if not isinstance(item.get("tool_name"), str) or not item["tool_name"].strip():
                errors.append(f"actions[{index}].tool_name must be a registered tool name")
            elif item["tool_name"] == "automated_routine_executor":
                errors.append(f"actions[{index}] cannot recursively call automated_routine_executor")
            if not isinstance(item.get("arguments", {}), dict):
                errors.append(f"actions[{index}].arguments must be an object")
        elif item.get("type") == "open_url" and not str(item.get("url", "")).startswith(("http://", "https://")):
            errors.append(f"actions[{index}].url must use http or https")
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


def _run_registered_tool(tool_name: str, arguments: dict, action_type: str = "tool") -> dict:
    """Call tools through the shared registry used by normal agent tool calls."""
    if tool_name == "automated_routine_executor":
        raise ValueError("A routine cannot recursively invoke itself")

    # Imported lazily because registry.py imports this module while constructing
    # the shared dispatch map.
    from tools.registry import TOOL_DISPATCH

    handler = TOOL_DISPATCH.get(tool_name)
    if handler is None:
        raise ValueError(f"Unknown registered tool: {tool_name}")

    call_arguments = dict(arguments)
    if tool_name in AUTOMATIC_TOOL_NAMES:
        call_arguments["confirmed"] = True
    raw_result = handler(**call_arguments)
    if isinstance(raw_result, str):
        try:
            result = json.loads(raw_result)
        except json.JSONDecodeError:
            result = {"output": raw_result}
    else:
        result = raw_result

    failed = isinstance(result, dict) and (
        "error" in result or result.get("success") is False or result.get("ok") is False
    )
    return {
        "type": action_type,
        "ok": not failed,
        "tool_name": tool_name,
        "result": result,
    }


def _run_action(item: dict) -> dict:
    action_type = item["type"]
    if action_type == "delay":
        seconds = max(0.0, min(float(item.get("seconds", 1)), 30.0))
        time.sleep(seconds)
        return {"type": action_type, "ok": True, "seconds": seconds}
    if action_type == "open_url":
        return {"type": action_type, "ok": bool(webbrowser.open(str(item["url"]), new=2)), "url": item["url"]}
    if action_type == "open_app":
        app_name = item.get("app_name")
        if not app_name and isinstance(item.get("argv"), list) and item["argv"]:
            app_name = str(item["argv"][0])
        if not app_name:
            raise ValueError("open_app requires app_name; command arguments are not permitted")
        result = _run_registered_tool("open_app", {"app_name": str(app_name)}, action_type)
        result["app_name"] = app_name
        return result
    if action_type == "tool":
        return _run_registered_tool(item["tool_name"], item.get("arguments", {}))
    argv = [str(value) for value in item["argv"]]
    if not argv:
        raise ValueError("argv cannot be empty")
    requested_cwd = (PROJECT_ROOT / str(item.get("cwd", "."))).resolve()
    if requested_cwd != PROJECT_ROOT and PROJECT_ROOT not in requested_cwd.parents:
        raise ValueError("Command cwd must stay inside the project workspace")
    timeout = max(1.0, min(float(item.get("timeout", 60)), 600.0))
    completed = subprocess.run(argv, cwd=requested_cwd, capture_output=True, text=True, timeout=timeout, shell=False)
    return {"type": action_type, "ok": completed.returncode == 0, "argv": argv, "returncode": completed.returncode, "stdout": completed.stdout[-12000:], "stderr": completed.stderr[-12000:]}


def automated_routine_executor(
    action: str,
    name: str | None = None,
    routine: dict | None = None,
    trigger: str | None = None,
    dry_run: bool = False,
    confirmed: bool = False,
) -> str:
    """Manage workflow macros with per-run or narrowly scoped persistent approval."""
    routines = _load()
    if action == "list":
        items = [{"name": key, "description": value.get("description", ""), "triggers": value.get("triggers", []), "action_count": len(value.get("actions", []))} for key, value in sorted(routines.items())]
        return json.dumps({"routines": items}, ensure_ascii=False)
    if action == "define":
        if not name or not routine:
            return json.dumps({"error": "name and routine are required for define"})
        errors = _validate(routine)
        if errors:
            return json.dumps({"error": "Invalid routine", "details": errors})
        wants_automatic = routine.get("allow_automatic") is True
        if wants_automatic and not confirmed:
            return json.dumps({
                "error": "Persistent automatic execution requires confirmed=true after the user approves the preview.",
                "routine": routine,
            }, ensure_ascii=False)
        triggers = [str(value) for value in routine.get("triggers", [])]
        # ``trigger`` is a first-class tool argument and is the form models most
        # naturally use when defining a routine.  Older code silently discarded
        # it, leaving saved routines with no usable trigger phrases.
        if trigger and not any(
            trigger.strip().casefold() == value.strip().casefold()
            for value in triggers
        ):
            triggers.append(trigger.strip())
        routines[name] = {
            "description": str(routine.get("description", "")),
            "triggers": triggers,
            "actions": routine["actions"],
            "automatic_approved": wants_automatic and confirmed is True,
        }
        _save(routines)
        return json.dumps({
            "ok": True,
            "defined": name,
            "action_count": len(routine["actions"]),
            "automatic_approved": wants_automatic and confirmed is True,
            "store": str(STORE_PATH),
        })
    if action == "delete":
        if not confirmed:
            return json.dumps({"error": "Deleting a routine requires confirmed=true"})
        if not name or name not in routines:
            return json.dumps({"error": "Routine not found"})
        del routines[name]
        _save(routines)
        return json.dumps({"ok": True, "deleted": name})
    if action not in {"show", "run"}:
        return json.dumps({"error": "action must be list, define, show, run, or delete"})
    resolved_name, selected = _resolve(routines, name, trigger)
    if not selected:
        return json.dumps({"error": "No unique routine matched", "name": name, "trigger": trigger})
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
            result = _run_action(item)
        except Exception as exc:
            result = {"type": item.get("type"), "ok": False, "error": str(exc)}
        results.append({"index": index, **result})
        if not result.get("ok") and item.get("continue_on_error") is not True:
            break
    ok = len(results) == len(selected["actions"]) and all(item.get("ok") for item in results)
    return json.dumps({"ok": ok, "name": resolved_name, "results": results}, ensure_ascii=False)
