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


# ── Linux Desktop Entry Parser ────────────────────────────────────────

def _parse_desktop_file(file_path: str) -> dict | None:
    """Parse a Linux .desktop file and extract core fields."""
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
            if key in ("Name", "Exec", "Terminal", "NoDisplay", "Type"):
                entry_data[key] = val

    if entry_data and "Name" in entry_data and "Exec" in entry_data:
        return entry_data
    return None


def _get_installed_apps() -> list[dict]:
    """Scan standard XDG paths to build a list of installed desktop applications."""
    xdg_dirs = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":")
    app_dirs = [os.path.join(d, "applications") for d in xdg_dirs]
    # Add user local applications
    app_dirs.append(os.path.expanduser("~/.local/share/applications"))
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
                    is_nodisplay = entry.get("NoDisplay", "false").lower() in ("true", "1")
                    if is_nodisplay:
                        continue
                    
                    app_type = entry.get("Type", "Application")
                    if app_type != "Application":
                        continue

                    apps.append({
                        "filename": file_name,
                        "filepath": file_path,
                        "name": entry.get("Name", ""),
                        "exec": entry.get("Exec", ""),
                        "terminal": entry.get("Terminal", "false").lower() in ("true", "1")
                    })
        except Exception:
            pass
    return apps


def _clean_exec_line(exec_str: str) -> list[str]:
    """Remove standard %f, %u, etc. placeholders from Exec command and return tokens."""
    # Remove standard desktop placeholders
    cleaned = re.sub(r'%\s*[fFuUdDnNvVmMkic]', '', exec_str)
    try:
        cmd_parts = shlex.split(cleaned)
    except Exception:
        cmd_parts = cleaned.split()
    return [p for p in cmd_parts if p and not p.startswith("%")]


def _find_terminal_emulator() -> str | None:
    """Find a supported terminal emulator on Linux to launch CLI-only apps."""
    for term in ("gnome-terminal", "konsole", "xfce4-terminal", "xterm", "alacritty", "kitty"):
        if shutil.which(term):
            return term
    return None


def _find_matching_app(query: str, apps: list[dict]) -> tuple[dict | None, list[str]]:
    """
    Search for a matching desktop app based on query.
    Returns (matched_app_dict, list_of_suggested_names).
    """
    query_clean = query.strip().lower()
    if not query_clean:
        return None, []

    exact_matches = []
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

    if name_substring_matches:
        if len(name_substring_matches) == 1:
            return name_substring_matches[0], []
        else:
            return None, sorted([app["name"] for app in name_substring_matches])

    if filename_substring_matches:
        if len(filename_substring_matches) == 1:
            return filename_substring_matches[0], []
        else:
            return None, sorted([app["name"] for app in filename_substring_matches])

    if exec_substring_matches:
        if len(exec_substring_matches) == 1:
            return exec_substring_matches[0], []
        else:
            return None, sorted([app["name"] for app in exec_substring_matches])

    return None, []


# ── Public tool function ──────────────────────────────────────────────

def open_app(app_name: str) -> str:
    """
    Open an application on the user's computer.

    Args:
        app_name (str): The name of the application to open (e.g., 'chrome', 'VS Code', 'spotify').

    Returns:
        str: A JSON string containing execution status or error.
    """
    if not app_name:
        return json.dumps({"error": "No application name provided."})

    platform = sys.platform

    # ── macOS ─────────────────────────────────────────────────────────
    if platform == "darwin":
        try:
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
            # Run start command in background via shell
            cmd = f'start /b "" "{app_name}"'
            subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            return json.dumps({
                "success": True,
                "message": f"Sent launch request for application '{app_name}' via start command."
            })
        except Exception as e:
            return json.dumps({"error": f"Failed to launch '{app_name}' on Windows: {str(e)}"})

    # ── Linux (and others) ────────────────────────────────────────────
    else:
        apps = _get_installed_apps()
        matched, suggestions = _find_matching_app(app_name, apps)

        if not matched:
            # If no .desktop files matched, fall back to checking PATH directly
            binary_path = shutil.which(app_name)
            if binary_path:
                try:
                    subprocess.Popen(
                        [binary_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True
                    )
                    return json.dumps({
                        "success": True,
                        "message": f"Launched binary '{app_name}' from PATH: {binary_path}"
                    })
                except Exception as e:
                    return json.dumps({"error": f"Failed to launch binary '{app_name}' from PATH: {str(e)}"})

            if suggestions:
                return json.dumps({
                    "error": f"Application '{app_name}' not found. Did you mean one of these?",
                    "suggestions": suggestions
                })
            else:
                return json.dumps({
                    "error": f"Could not find application or binary named '{app_name}' on this system."
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

        # Handle Terminal=true apps by wrapping with terminal emulator if possible
        if matched["terminal"]:
            terminal = _find_terminal_emulator()
            if terminal:
                if terminal == "gnome-terminal":
                    cmd_parts = ["gnome-terminal", "--"] + cmd_parts
                elif terminal == "kitty":
                    cmd_parts = ["kitty"] + cmd_parts
                else:
                    cmd_parts = [terminal, "-e"] + cmd_parts
            else:
                # Proceed but note to user
                pass

        try:
            subprocess.Popen(
                cmd_parts,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True
            )
            term_note = " in terminal wrapper" if matched["terminal"] else ""
            return json.dumps({
                "success": True,
                "message": f"Launched application '{matched['name']}' directly{term_note} using command: {' '.join(cmd_parts)}"
            })
        except Exception as e:
            return json.dumps({
                "error": f"Failed to manually execute application '{matched['name']}': {str(e)}"
            })
