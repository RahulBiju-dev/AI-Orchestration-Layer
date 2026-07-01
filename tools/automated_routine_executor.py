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
STORE_PATH = PROJECT_ROOT / ".selene" / "routines.json"
MAX_ACTIONS = 50


def _load() -> dict[str, dict]:
    try:
        value = json.loads(STORE_PATH.read_text(encoding="utf-8"))
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
        if not isinstance(item, dict) or item.get("type") not in {"command", "open_app", "open_url", "delay"}:
            errors.append(f"actions[{index}] has an unsupported type")
        elif item.get("type") in {"command", "open_app"} and not isinstance(item.get("argv"), list):
            errors.append(f"actions[{index}].argv must be an argument array; shell strings are not accepted")
        elif item.get("type") == "open_url" and not str(item.get("url", "")).startswith(("http://", "https://")):
            errors.append(f"actions[{index}].url must use http or https")
    return errors


def _run_action(item: dict) -> dict:
    action_type = item["type"]
    if action_type == "delay":
        seconds = max(0.0, min(float(item.get("seconds", 1)), 30.0))
        time.sleep(seconds)
        return {"type": action_type, "ok": True, "seconds": seconds}
    if action_type == "open_url":
        return {"type": action_type, "ok": bool(webbrowser.open(str(item["url"]), new=2)), "url": item["url"]}
    argv = [str(value) for value in item["argv"]]
    if not argv:
        raise ValueError("argv cannot be empty")
    if action_type == "open_app":
        subprocess.Popen(argv, cwd=PROJECT_ROOT, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"type": action_type, "ok": True, "argv": argv}
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
    dry_run: bool = True,
    confirmed: bool = False,
) -> str:
    """Manage workflow macros; execution requires dry_run=false and confirmed=true."""
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
        routines[name] = {"description": str(routine.get("description", "")), "triggers": [str(value) for value in routine.get("triggers", [])], "actions": routine["actions"]}
        _save(routines)
        return json.dumps({"ok": True, "defined": name, "action_count": len(routine["actions"]), "store": str(STORE_PATH)})
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
    if action == "show" or dry_run:
        return json.dumps({"name": resolved_name, "routine": selected, "dry_run": True, "execution_required": "Call run with dry_run=false and confirmed=true after user approval"}, ensure_ascii=False)
    if not confirmed:
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
