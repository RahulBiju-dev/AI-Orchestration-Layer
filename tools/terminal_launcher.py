"""Safely open a terminal window at an existing directory."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_directory(path: object) -> Path:
    """Return a normalized existing directory without interpreting shell syntax."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("A directory path is required.")
    value = path.strip()
    if len(value) > 4096 or "\0" in value:
        raise ValueError("The directory path is invalid.")

    directory = Path(value).expanduser()
    if not directory.is_absolute():
        directory = Path.cwd() / directory
    try:
        directory = directory.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError(f"Directory does not exist: {value}") from None
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {value}")
    return directory


def _linux_terminal_command(directory: Path) -> tuple[list[str], str] | None:
    """Select an installed terminal and its native working-directory option."""
    path = str(directory)
    candidates = (
        ("ptyxis", ["--new-window", "--working-directory", path]),
        ("gnome-terminal", ["--working-directory", path]),
        ("kgx", ["--working-directory", path]),
        ("konsole", ["--workdir", path]),
        ("xfce4-terminal", ["--working-directory", path]),
        ("kitty", ["--directory", path]),
        ("alacritty", ["--working-directory", path]),
        # xterm inherits the child process working directory. No command is
        # passed to it, so this fallback cannot become shell execution.
        ("xterm", []),
    )
    for name, arguments in candidates:
        executable = shutil.which(name)
        if executable:
            return [executable, *arguments], name
    return None


def open_terminal_at_path(path: str, confirmed: bool = False) -> str:
    """Open a terminal at ``path`` after an explicit user or routine approval."""
    if confirmed is not True:
        return json.dumps({
            "error": "Opening a terminal requires explicit user approval.",
            "required": "Call again with confirmed=true only when the user requested it.",
        })

    try:
        directory = _resolve_directory(path)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    try:
        if sys.platform == "darwin":
            command = ["/usr/bin/open", "-a", "Terminal", str(directory)]
            terminal_name = "Terminal"
        elif sys.platform == "win32":
            executable = shutil.which("wt.exe") or shutil.which("wt")
            if not executable:
                return json.dumps({
                    "error": "Windows Terminal is required to open a terminal at a directory safely."
                })
            command = [executable, "-d", str(directory)]
            terminal_name = "Windows Terminal"
        else:
            selected = _linux_terminal_command(directory)
            if selected is None:
                return json.dumps({"error": "No supported terminal emulator is installed."})
            command, terminal_name = selected

        subprocess.Popen(
            command,
            cwd=directory,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return json.dumps({
            "success": True,
            "terminal": terminal_name,
            "path": str(directory),
            "message": f"Opened {terminal_name} at '{directory}'.",
        }, ensure_ascii=False)
    except OSError as exc:
        return json.dumps({
            "error": f"Failed to open a terminal at '{directory}': {exc}"
        }, ensure_ascii=False)
