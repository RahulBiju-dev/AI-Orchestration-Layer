"""Small, dependency-free loader for Selene's server-side ``.env`` file."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import MutableMapping


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Bookkeeping for :func:`refresh_dotenv`. Only the default (process
# environment) load records state; callers that pass their own mapping are
# isolated test fixtures and must not disturb the live process.
_STATE_LOCK = threading.Lock()
_LOADED_PATH: Path | None = None
_LOADED_STAMP: tuple[int, int] | None = None
_LOADED_KEYS: dict[str, str] = {}


def _candidates(
    path: str | os.PathLike[str] | None,
    destination: MutableMapping[str, str],
) -> list[Path]:
    if path is not None:
        return [Path(path).expanduser()]
    explicit = str(destination.get("SELENE_ENV_FILE") or "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ])
    return candidates


def _parse(target: Path) -> dict[str, str]:
    """Return the KEY=VALUE pairs in ``target``, last assignment winning."""
    values: dict[str, str] = {}
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not _ENV_NAME.fullmatch(name):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[name] = value
    return values


def _stamp(target: Path) -> tuple[int, int] | None:
    try:
        info = target.stat()
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def load_dotenv(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> Path | None:
    """Load simple KEY=VALUE entries without replacing process environment.

    This intentionally supports only the subset needed by this project: blank
    lines, comments, optional ``export``, and single/double quoted values.  It
    never logs values because this file commonly contains provider secrets.
    """
    global _LOADED_PATH, _LOADED_STAMP, _LOADED_KEYS

    destination = os.environ if environ is None else environ
    target = next(
        (candidate for candidate in _candidates(path, destination) if candidate.is_file()),
        None,
    )
    if target is None:
        return None

    values = _parse(target)
    applied: dict[str, str] = {}
    for name, value in values.items():
        destination.setdefault(name, value)
        # Only entries this file actually supplied may later be rewritten or
        # withdrawn by refresh_dotenv(); a real export always outranks the file.
        if destination.get(name) == value:
            applied[name] = value

    if environ is None:
        with _STATE_LOCK:
            _LOADED_PATH = target
            _LOADED_STAMP = _stamp(target)
            _LOADED_KEYS = applied
    return target


def reset_dotenv_cache() -> None:
    """Forget which ``.env`` was loaded and what it supplied.

    The next :func:`refresh_dotenv` then re-discovers the file from scratch
    instead of trying to withdraw keys sourced from a mapping that no longer
    describes this process.  Used by tests that swap the environment.
    """
    global _LOADED_PATH, _LOADED_STAMP, _LOADED_KEYS
    with _STATE_LOCK:
        _LOADED_PATH = None
        _LOADED_STAMP = None
        _LOADED_KEYS = {}


def refresh_dotenv(*, environ: MutableMapping[str, str] | None = None) -> bool:
    """Re-apply the ``.env`` file when it changed on disk since the last load.

    Editing provider configuration is a normal thing to do while Selene is
    running, so the registry re-reads the file instead of freezing whatever
    existed at import time.  Values this loader previously supplied are
    overwritten with the new contents and withdrawn when their line is
    deleted; anything exported into the real environment is left untouched.

    Returns ``True`` when the environment was updated.
    """
    global _LOADED_PATH, _LOADED_STAMP, _LOADED_KEYS

    destination = os.environ if environ is None else environ
    with _STATE_LOCK:
        previous_path = _LOADED_PATH
        previous_stamp = _LOADED_STAMP
        previous_keys = dict(_LOADED_KEYS)

    target = next(
        (candidate for candidate in _candidates(None, destination) if candidate.is_file()),
        None,
    )
    if target is None:
        if not previous_keys:
            return False
        # The file was deleted: withdraw what it had supplied.
        for name in previous_keys:
            destination.pop(name, None)
        with _STATE_LOCK:
            _LOADED_PATH = None
            _LOADED_STAMP = None
            _LOADED_KEYS = {}
        return True

    stamp = _stamp(target)
    if target == previous_path and stamp is not None and stamp == previous_stamp:
        return False

    try:
        values = _parse(target)
    except OSError:
        return False

    applied: dict[str, str] = {}
    for name, value in values.items():
        if name in previous_keys or name not in destination:
            destination[name] = value
        if destination.get(name) == value:
            applied[name] = value
    for name in previous_keys:
        if name not in values:
            destination.pop(name, None)

    with _STATE_LOCK:
        _LOADED_PATH = target
        _LOADED_STAMP = stamp
        _LOADED_KEYS = applied
    return True
