"""
tools/app_launcher.py — Cross-platform application launcher.

Allows the agent to search for and launch local GUI and terminal applications
on Linux, macOS, and Windows in a detached, non-blocking background process.
"""

import os
import re
import sys
import shlex
import shutil
import subprocess
import json
import threading
import time
from pathlib import Path


_APP_CACHE: tuple[float, list[dict]] = (0.0, [])
_APP_CACHE_LOCK = threading.Lock()
_APP_CACHE_SECONDS = 30.0
MAX_APPS_PER_REQUEST = 10

# App launching is intentionally not a general process-execution primitive. These
# programs can turn an innocent-looking app request into arbitrary command access.
_BLOCKED_EXECUTABLES = {
    "bash", "cmd", "cmd.exe", "command", "dash", "fish", "gnome-terminal",
    "konsole", "osascript", "powershell", "powershell.exe", "pwsh", "sh",
    "terminal", "terminal.app", "wscript", "wscript.exe", "xterm", "zsh",
}
_SAFE_APP_NAME = re.compile(r"^[\w .+&'()@#-]+$", flags=re.UNICODE)


# ── Linux Desktop Entry Parser ────────────────────────────────────────

def _parse_desktop_file(file_path: str) -> dict | None:
    """
    Parse a Linux .desktop file and extract core fields.
    
    This helper reads the contents of a .desktop file, identifies the
    '[Desktop Entry]' section, and extracts key/value pairs that define
    how the application should be launched or displayed.
    
    Args:
        file_path (str): The absolute path to the .desktop file.
        
    Returns:
        dict | None: A dictionary containing extracted keys ('Name', 'Exec', 
            'Terminal', 'NoDisplay', 'Type') if successful, or None if parsing fails
            or the file is invalid.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return None

    in_entry = False
    entry_data = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            if line == "[Desktop Entry]":
                in_entry = True
            else:
                in_entry = False
            continue

        if in_entry and "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            # Only store standard keys we care about
            if key in (
                "Name", "GenericName", "Keywords", "Exec", "Terminal",
                "NoDisplay", "Hidden", "TryExec", "Type",
            ):
                entry_data[key] = val

    if entry_data and "Name" in entry_data and "Exec" in entry_data:
        return entry_data
    return None


def _get_installed_apps() -> list[dict]:
    """
    Scan standard XDG paths to build a list of installed desktop applications.
    
    This function looks through standard system directories where .desktop files
    are typically installed (e.g., /usr/share/applications, ~/.local/share/applications,
    and flatpak directories) to compile a registry of available GUI applications.
    
    Returns:
        list[dict]: A list of application dictionaries, each containing 'filename',
            'filepath', 'name', 'exec', and 'terminal' properties. Applications
            marked as 'NoDisplay' or not of Type 'Application' are excluded.
    """
    global _APP_CACHE
    with _APP_CACHE_LOCK:
        cached_at, cached_apps = _APP_CACHE
        if cached_apps and time.monotonic() - cached_at < _APP_CACHE_SECONDS:
            return [dict(app) for app in cached_apps]

    xdg_dirs = [part for part in os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(os.pathsep) if part]
    # User entries precede system entries because desktop files with the same
    # ID are user-overridable under the XDG specification.
    app_dirs = [os.path.expanduser("~/.local/share/applications")]
    app_dirs.extend(os.path.join(d, "applications") for d in xdg_dirs)
    # Add flatpak applications export directory if not already included
    flatpak_dir = "/var/lib/flatpak/exports/share/applications"
    if flatpak_dir not in app_dirs:
        app_dirs.append(flatpak_dir)

    apps = []
    seen_files = set()
    for dir_path in app_dirs:
        if not os.path.isdir(dir_path):
            continue
        try:
            for file_name in os.listdir(dir_path):
                if not file_name.endswith(".desktop"):
                    continue
                file_path = os.path.join(dir_path, file_name)
                # Avoid duplicates (user overrides system applications)
                if file_name in seen_files:
                    continue
                seen_files.add(file_name)

                entry = _parse_desktop_file(file_path)
                if entry:
                    # Skip applications marked NoDisplay
                    is_hidden = entry.get("NoDisplay", "false").lower() in ("true", "1") or entry.get("Hidden", "false").lower() in ("true", "1")
                    if is_hidden:
                        continue
                    
                    app_type = entry.get("Type", "Application")
                    if app_type != "Application":
                        continue
                    try_exec = entry.get("TryExec", "").strip()
                    if try_exec and not (os.path.isfile(os.path.expanduser(try_exec)) or shutil.which(try_exec)):
                        continue

                    apps.append({
                        "filename": file_name,
                        "filepath": file_path,
                        "name": entry.get("Name", ""),
                        "generic_name": entry.get("GenericName", ""),
                        "keywords": [
                            keyword.strip()
                            for keyword in entry.get("Keywords", "").split(";")
                            if keyword.strip()
                        ],
                        "exec": entry.get("Exec", ""),
                        "terminal": entry.get("Terminal", "false").lower() in ("true", "1")
                    })
        except Exception:
            pass
    with _APP_CACHE_LOCK:
        _APP_CACHE = (time.monotonic(), [dict(app) for app in apps])
    return apps


def _clean_exec_line(exec_str: str) -> list[str]:
    """
    Remove standard %f, %u, etc. placeholders from an Exec command and return shell tokens.
    
    Linux desktop files often include placeholders (like %f or %U) which are meant to be
    replaced by file paths or URIs by the desktop environment. This function strips
    them out so the application can be launched without arguments safely.
    
    Args:
        exec_str (str): The raw 'Exec' line from a .desktop file.
        
    Returns:
        list[str]: A list of clean command-line arguments ready for subprocess execution.
    """
    # Replace literal percent first, then strip standard desktop field codes.
    sentinel = "\0PERCENT\0"
    cleaned = exec_str.replace("%%", sentinel)
    cleaned = re.sub(r"%[fFuUdDnNickvm]", "", cleaned).replace(sentinel, "%")
    try:
        cmd_parts = shlex.split(cleaned)
    except Exception:
        cmd_parts = cleaned.split()
    return [part for part in cmd_parts if part]


def _find_terminal_emulator() -> str | None:
    """Find a supported terminal emulator on Linux to launch CLI-only apps."""
    for term in ("gnome-terminal", "konsole", "xfce4-terminal", "xterm", "alacritty", "kitty"):
        if shutil.which(term):
            return term
    return None


def _normalize_app_name(value: str) -> str:
    """Normalize a human-facing app name for comparisons (``VS Code`` -> ``vscode``)."""
    return "".join(character for character in value.casefold() if character.isalnum())


def _validate_app_name(app_name: object) -> str:
    """Accept a display name, never a path, URL, or command line."""
    value = str(app_name or "").strip()
    if not value:
        raise ValueError("No application name provided.")
    if len(value) > 128 or any(ord(char) < 32 for char in value):
        raise ValueError("Application name is invalid.")
    if value in {".", ".."}:
        raise ValueError("Use an application display name, not a filesystem location.")
    if not _SAFE_APP_NAME.fullmatch(value):
        raise ValueError("Use an application name only; paths, URLs, and command syntax are not accepted.")
    if "/" in value or "\\" in value or value.casefold().endswith((".desktop", ".lnk")):
        raise ValueError("Use the application's display name, not a file path or shortcut path.")
    if value.casefold() in _BLOCKED_EXECUTABLES:
        raise ValueError("Shells and terminal applications cannot be launched by this tool.")
    return value


def _is_safe_desktop_entry(app: dict) -> bool:
    """Reject desktop entries that would expose an interactive command runner."""
    command = _clean_exec_line(app.get("exec", ""))
    if not command:
        return False
    executable = os.path.basename(command[0]).casefold()
    return not app.get("terminal", False) and executable not in _BLOCKED_EXECUTABLES


def _find_windows_shortcut(query: str) -> tuple[Path | None, list[str]]:
    """Resolve a display name to an installed Start Menu shortcut."""
    roots = []
    for environment_name, suffix in (
        ("APPDATA", ("Microsoft", "Windows", "Start Menu", "Programs")),
        ("PROGRAMDATA", ("Microsoft", "Windows", "Start Menu", "Programs")),
    ):
        base = os.environ.get(environment_name)
        if base:
            roots.append(Path(base).joinpath(*suffix))

    normalized = _normalize_app_name(query)
    matches = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            shortcuts = root.rglob("*.lnk")
            for shortcut in shortcuts:
                if normalized in _name_aliases(shortcut.stem):
                    matches.append(shortcut)
        except OSError:
            continue
    unique = {str(item).casefold(): item for item in matches}
    values = list(unique.values())
    if len(values) == 1:
        return values[0], []
    return None, sorted({item.stem for item in values})[:10]


def _name_aliases(value: str) -> set[str]:
    """Build compact aliases for names such as ``Visual Studio Code`` -> ``vscode``."""
    words = re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE)
    if not words:
        return set()

    aliases = {_normalize_app_name(value)}
    if len(words) > 1:
        aliases.add("".join(word[0] for word in words))
        # People commonly abbreviate a product prefix but retain its final word:
        # Visual Studio Code -> VS Code, Android Studio -> A Studio.
        for split_at in range(1, len(words)):
            aliases.add("".join(word[0] for word in words[:split_at]) + "".join(words[split_at:]))
    return {alias for alias in aliases if alias}


def _app_aliases(app: dict) -> set[str]:
    """Return names by which a Linux desktop entry can reasonably be requested."""
    filename = app.get("filename", "")
    if filename.casefold().endswith(".desktop"):
        filename = filename[:-8]

    aliases = _name_aliases(app.get("name", ""))
    aliases.update(_name_aliases(app.get("generic_name", "")))
    aliases.update(_normalize_app_name(keyword) for keyword in app.get("keywords", []))
    aliases.add(_normalize_app_name(filename))

    command = _clean_exec_line(app.get("exec", ""))
    if command:
        aliases.add(_normalize_app_name(os.path.basename(command[0])))
    return {alias for alias in aliases if alias}


def _find_matching_app(query: str, apps: list[dict]) -> tuple[dict | None, list[str]]:
    """
    Search for a matching desktop app based on query.
    Returns (matched_app_dict, list_of_suggested_names).
    """
    query_clean = query.strip().lower()
    if not query_clean:
        return None, []
    query_normalized = _normalize_app_name(query)

    exact_matches = []
    alias_matches = []
    name_substring_matches = []
    filename_substring_matches = []
    exec_substring_matches = []

    for app in apps:
        name = app["name"].lower()
        filename_no_ext = app["filename"].lower()
        if filename_no_ext.endswith(".desktop"):
            filename_no_ext = filename_no_ext[:-8]
        exec_cmd = app["exec"].lower()

        # 1. Exact name or exact desktop filename match
        if query_clean == name or query_clean == filename_no_ext:
            exact_matches.append(app)
            continue

        # Desktop metadata is a better source of aliases than a hard-coded
        # package-name table. It covers distro packages, Flatpaks, and locally
        # installed apps, including common abbreviations such as VS Code.
        if query_normalized in _app_aliases(app):
            alias_matches.append(app)
            continue

        # 2. Substring match of display Name
        if query_clean in name:
            name_substring_matches.append(app)
            continue

        # 3. Substring match of filename
        if query_clean in filename_no_ext:
            filename_substring_matches.append(app)
            continue

        # 4. Substring match of Exec binary/command
        if query_clean in exec_cmd:
            exec_substring_matches.append(app)
            continue

    # Return matches in order of priority
    if exact_matches:
        return exact_matches[0], []

    if alias_matches:
        if len(alias_matches) == 1:
            return alias_matches[0], []
        return None, sorted({app["name"] for app in alias_matches})[:10]

    if name_substring_matches:
        if len(name_substring_matches) == 1:
            return name_substring_matches[0], []
        else:
            return None, sorted({app["name"] for app in name_substring_matches})[:10]

    if filename_substring_matches:
        if len(filename_substring_matches) == 1:
            return filename_substring_matches[0], []
        else:
            return None, sorted({app["name"] for app in filename_substring_matches})[:10]

    if exec_substring_matches:
        if len(exec_substring_matches) == 1:
            return exec_substring_matches[0], []
        else:
            return None, sorted({app["name"] for app in exec_substring_matches})[:10]

    return None, []


# ── Public tool function ──────────────────────────────────────────────

def open_app(app_name: str, confirmed: bool = False) -> str:
    """
    Open an application on the user's computer.

    Args:
        app_name (str): The name of the application to open (e.g., 'chrome', 'VS Code', 'spotify').

    Returns:
        str: A JSON string containing execution status or error.
    """
    if confirmed is not True:
        return json.dumps({
            "error": "Launching an application requires explicit user approval.",
            "required": "Call again with confirmed=true only when the user explicitly asked to open the app.",
        })
    try:
        app_name = _validate_app_name(app_name)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    platform = sys.platform

    # ── macOS ─────────────────────────────────────────────────────────
    if platform == "darwin":
        try:
            check = subprocess.run(
                ["open", "-Ra", app_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            if check.returncode != 0:
                return json.dumps({"error": f"Could not find an installed application named '{app_name}'."})
            # We use `open -a` to start applications on macOS
            subprocess.Popen(
                ["open", "-a", app_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            return json.dumps({
                "success": True,
                "message": f"Sent launch request for application '{app_name}' via open -a."
            })
        except Exception as e:
            return json.dumps({"error": f"Failed to launch '{app_name}' on macOS: {str(e)}"})

    # ── Windows ───────────────────────────────────────────────────────
    elif platform == "win32":
        try:
            shortcut, suggestions = _find_windows_shortcut(app_name)
            if not shortcut:
                response = {"error": f"Could not find an installed Start Menu application named '{app_name}'."}
                if suggestions:
                    response["suggestions"] = suggestions
                return json.dumps(response)
            # Only a shortcut discovered beneath a Start Menu directory reaches
            # ShellExecute; user input itself is never treated as a path.
            os.startfile(str(shortcut))  # type: ignore[attr-defined]
            return json.dumps({
                "success": True,
                "message": f"Sent launch request for application '{shortcut.stem}'."
            })
        except Exception as e:
            return json.dumps({"error": f"Failed to launch '{app_name}' on Windows: {str(e)}"})

    # ── Linux (and others) ────────────────────────────────────────────
    else:
        apps = _get_installed_apps()
        matched, suggestions = _find_matching_app(app_name, apps)

        if not matched:
            if suggestions:
                return json.dumps({
                    "error": f"Application '{app_name}' not found. Did you mean one of these?",
                    "suggestions": suggestions
                })
            else:
                return json.dumps({
                    "error": f"Could not find an installed desktop application named '{app_name}'."
                })

        if not _is_safe_desktop_entry(matched):
            return json.dumps({
                "error": f"'{matched['name']}' is a terminal or command-runner entry and is blocked by app-launch safety policy."
            })

        # Launching the desktop entry
        # Prefer gtk-launch if available
        gtk_launch_bin = shutil.which("gtk-launch")
        if gtk_launch_bin:
            try:
                subprocess.Popen(
                    [gtk_launch_bin, matched["filename"]],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True
                )
                return json.dumps({
                    "success": True,
                    "message": f"Opened application '{matched['name']}' ({matched['filename']}) via gtk-launch."
                })
            except Exception:
                # Fall through to manual exec if gtk-launch fails
                pass

        # Manual execution fallback
        cmd_parts = _clean_exec_line(matched["exec"])
        if not cmd_parts:
            return json.dumps({"error": f"Application '{matched['name']}' has an empty or invalid Exec entry."})

        try:
            subprocess.Popen(
                cmd_parts,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True
            )
            return json.dumps({
                "success": True,
                "message": f"Launched application '{matched['name']}'."
            })
        except Exception as e:
            return json.dumps({
                "error": f"Failed to manually execute application '{matched['name']}': {str(e)}"
            })


def launch_apps(app_names: list[str], confirmed: bool = False) -> str:
    """Launch a small, user-approved set of desktop apps by display name."""
    if confirmed is not True:
        return json.dumps({
            "error": "Launching applications requires explicit user approval.",
            "required": "Set confirmed=true only for an explicit user launch request or an approved automatic app routine.",
        })
    if not isinstance(app_names, list) or not app_names:
        return json.dumps({"error": "app_names must be a non-empty array."})
    if len(app_names) > MAX_APPS_PER_REQUEST:
        return json.dumps({"error": f"At most {MAX_APPS_PER_REQUEST} applications may be launched at once."})

    results = []
    seen = set()
    for requested_name in app_names:
        normalized = _normalize_app_name(str(requested_name))
        if normalized in seen:
            continue
        seen.add(normalized)
        result = json.loads(open_app(requested_name, confirmed=True))
        results.append({"app_name": str(requested_name), **result})

    return json.dumps({
        "ok": bool(results) and all(item.get("success") is True for item in results),
        "results": results,
    }, ensure_ascii=False)
