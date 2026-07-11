"""
agent/terminal.py — Terminal helpers and lightweight LaTeX math renderer

Contains ANSI helpers, a spinner, and a compact LaTeX-to-terminal
renderer used by the streaming output in `agent.core`.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from os.path import commonprefix
from typing import Sequence
from rich.console import Console

# Shared console (write to stderr so Live and spinner use the same stream)
_console = Console(stderr=True)


@dataclass(frozen=True)
class TerminalTheme:
    """Frontier-lab palette: cool ice cyan, soft violet, precise mono chrome."""

    # Named colors stay portable across terminal emulators without truecolor.
    accent: str = "bright_cyan"
    accent_soft: str = "cyan"
    accent_2: str = "bright_magenta"
    ink: str = "bright_white"
    success: str = "green"
    warning: str = "yellow"
    danger: str = "red"
    muted: str = "dim"
    border: str = "cyan"
    thinking: str = "bright_magenta"
    surface: str = "grey23"
    prompt_name: str = "bold bright_cyan"
    prompt_glyph: str = "dim cyan"
    prompt_mark: str = "bold bright_magenta"
    rule: str = "dim cyan"
    label: str = "bold cyan"
    meta: str = "dim"


THEME = TerminalTheme()

# Geometric glyphs used consistently across the CLI chrome.
GLYPH_MARK = "◈"
GLYPH_PROMPT = "›"
GLYPH_SECTION = "◆"
GLYPH_TOOL = "▸"
GLYPH_OK = "✓"
GLYPH_WARN = "!"
GLYPH_ERR = "×"
GLYPH_RUN = "◌"


_ANSI_ESCAPE_RE = re.compile(
    r"""
    \x1b
    (?:
        \[[0-?]*[ -/]*[@-~]      # CSI sequences: arrows, mouse wheel, bracketed paste
      | \][^\x07]*(?:\x07|\x1b\\) # OSC sequences
      | [@-Z\\-_]                 # 2-byte escapes
    )
    """,
    re.VERBOSE,
)
_CONTROL_INPUT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def flush_terminal_input() -> None:
    """Drop queued terminal bytes before showing the next prompt.

    Scrolling in an alternate screen, arrow keys, bracketed paste markers, or
    impatient key presses can leave escape bytes in the terminal input queue.
    Flushing before each prompt prevents those bytes from becoming user text.
    """
    if not sys.stdin.isatty():
        return
    try:
        import termios

        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        # Windows and some pseudo terminals do not expose tcflush. Sanitization
        # still protects the prompt after the line is read.
        return


def sanitize_terminal_input(text: str) -> str:
    """Remove terminal-control garbage while preserving intentional text."""
    if not text:
        return ""

    text = _ANSI_ESCAPE_RE.sub("", text)

    chars: list[str] = []
    for char in text:
        if char in ("\b", "\x7f"):
            if chars:
                chars.pop()
            continue
        chars.append(char)

    cleaned = "".join(chars).replace("\r", "\n")
    cleaned = _CONTROL_INPUT_RE.sub("", cleaned)
    return cleaned.strip()


# Visible prompt chrome. The colored form is printed by Rich; the plain form is
# documentation for the fixed width. Chrome is never part of the editable buffer.
PROMPT_MARKUP = (
    f"[{THEME.prompt_glyph}]{GLYPH_MARK}[/] "
    f"[{THEME.prompt_name}]selene[/] "
    f"[{THEME.prompt_mark}]{GLYPH_PROMPT}[/] "
)
PROMPT_PLAIN = f"{GLYPH_MARK} selene {GLYPH_PROMPT} "


@dataclass
class _SlashCompletionState:
    """Track deterministic Tab completion for slash commands."""

    commands: tuple[str, ...]
    matches: tuple[str, ...] = ()
    index: int = -1
    last_value: str = ""

    def complete(self, value: str, preferred: str | None = None) -> str:
        if not value.startswith("/"):
            self.reset()
            return value

        if preferred and preferred.casefold().startswith(value.casefold()):
            self.last_value = preferred
            return preferred

        if value != self.last_value:
            folded = value.casefold()
            matches = tuple(
                command for command in self.commands
                if command.casefold().startswith(folded)
            )
            # At an exact parent command, the next useful completion is its
            # argument/subcommand boundary rather than the same text again.
            children = tuple(
                command for command in matches
                if command.casefold().startswith(f"{folded} ")
            )
            self.matches = children or matches
            self.index = -1

        if not self.matches:
            self.last_value = value
            return value

        shared = commonprefix(self.matches)
        if self.index < 0 and len(shared) > len(value):
            completed = shared
        else:
            self.index = (self.index + 1) % len(self.matches)
            completed = self.matches[self.index]
        self.last_value = completed
        return completed

    def reset(self) -> None:
        self.matches = ()
        self.index = -1
        self.last_value = ""


@dataclass
class _SlashMenuState:
    """Filtered, keyboard-selectable slash-command menu state."""

    commands: tuple[str, ...]
    max_visible: int = 7
    matches: tuple[str, ...] = ()
    selected: int = 0
    query: str = ""

    def update(self, value: str) -> None:
        normalized = str(value or "")
        if not normalized.startswith("/") or "\n" in normalized:
            self.reset()
            return
        matches = tuple(
            command for command in self.commands
            if command.casefold().startswith(normalized.casefold())
        )
        if normalized != self.query or matches != self.matches:
            previous = self.selected_command()
            self.query = normalized
            self.matches = matches
            self.selected = matches.index(previous) if previous in matches else 0

    def move(self, delta: int) -> None:
        if self.matches:
            self.selected = (self.selected + int(delta)) % len(self.matches)

    def selected_command(self) -> str | None:
        if not self.matches:
            return None
        return self.matches[self.selected % len(self.matches)]

    def visible_matches(self) -> tuple[tuple[int, str], ...]:
        if not self.matches:
            return ()
        size = max(1, int(self.max_visible))
        start = min(
            max(0, self.selected - size + 1),
            max(0, len(self.matches) - size),
        )
        return tuple(enumerate(self.matches[start:start + size], start=start))

    def reset(self) -> None:
        self.matches = ()
        self.selected = 0
        self.query = ""


def _slash_menu_lines(state: _SlashMenuState) -> tuple[str, ...]:
    """Build compact ANSI menu rows; empty state leaves no prompt chrome."""
    visible = state.visible_matches()
    if not visible:
        return ()
    lines = []
    for index, command in visible:
        if index == state.selected:
            lines.append(f"  \x1b[1;36m{GLYPH_PROMPT} {command}\x1b[0m")
        else:
            lines.append(f"    \x1b[2m{command}\x1b[0m")
    lines.append("  \x1b[2m↑↓ choose · Tab autofill\x1b[0m")
    return tuple(lines)


def _print_prompt_chrome() -> None:
    """Render the colored prompt without placing it in the editable input buffer."""
    _console.print(PROMPT_MARKUP, markup=True, end="", highlight=False)


def _read_line_with_fixed_prompt(completions: Sequence[str] = ()) -> str:
    """Read one line while keeping ``selene $`` outside the editable buffer.

    Rich's ``Console.input`` prints a styled prompt then calls bare ``input()``.
    That leaves the chrome on the same visual line without giving the line editor
    a real prompt, so backspace can walk into ``selene $``.  Here the chrome is
    painted once, then a character-level editor owns only the user text.
    """
    _print_prompt_chrome()
    if sys.stdin.isatty():
        try:
            return _read_line_protected_tty(completions=completions)
        except (ImportError, OSError, termios_error_type()) as exc:
            # Fall through when raw mode is unavailable.
            del exc

    # Non-interactive or raw-mode failure: never put chrome into the buffer.
    try:
        return sys.stdin.readline()
    except EOFError:
        return ""


def termios_error_type() -> type[BaseException]:
    """Return the local termios error type, or a harmless stand-in."""
    try:
        import termios

        return termios.error
    except ImportError:
        return OSError


def _tty_out() -> "object":
    """Stream used for prompt chrome and the protected line editor (same TTY)."""
    return getattr(_console, "file", None) or sys.stderr


def _read_line_protected_tty(completions: Sequence[str] = ()) -> str:
    """Character-level line editor that refuses to erase the fixed prompt."""
    import os

    out = _tty_out()
    buffer: list[str] = []
    cursor = 0
    commands = tuple(dict.fromkeys(completions))
    completion_state = _SlashCompletionState(commands)
    menu_state = _SlashMenuState(commands)
    menu_rows = 0

    def write(text: str) -> None:
        out.write(text)
        out.flush()

    def restore_prompt_cursor(rows: int) -> None:
        """Return from menu rows using relative movement that survives scrolling."""
        write("\r")
        if rows > 0:
            write(f"\x1b[{rows}A")
        column = len(PROMPT_PLAIN) + cursor
        if column > 0:
            write(f"\x1b[{column}C")

    def clear_menu() -> None:
        nonlocal menu_rows
        if menu_rows <= 0:
            return
        write("\x1b[?25l")
        for _ in range(menu_rows):
            write("\r\n\x1b[2K")
        restore_prompt_cursor(menu_rows)
        write("\x1b[?25h")
        menu_rows = 0

    def refresh_menu() -> None:
        nonlocal menu_rows
        value = "".join(buffer) if cursor == len(buffer) else ""
        menu_state.update(value)
        lines = _slash_menu_lines(menu_state)
        rows_to_clear = max(menu_rows, len(lines))
        if rows_to_clear <= 0:
            return
        write("\x1b[?25l")
        for index in range(rows_to_clear):
            write("\r\n\x1b[2K")
            if index < len(lines):
                write(lines[index])
        restore_prompt_cursor(rows_to_clear)
        write("\x1b[?25h")
        menu_rows = len(lines)

    def redraw_from_cursor() -> None:
        # Clear from cursor to end of line, rewrite tail, restore cursor.
        tail = "".join(buffer[cursor:])
        write("\x1b[K" + tail)
        if tail:
            write(f"\x1b[{len(tail)}D")

    def insert(text: str) -> None:
        nonlocal cursor
        for char in text:
            if char in ("\n", "\r") or ord(char) < 32:
                continue
            buffer.insert(cursor, char)
            write(char)
            cursor += 1
            redraw_from_cursor()
        refresh_menu()

    def backspace() -> None:
        nonlocal cursor
        if cursor <= 0:
            return
        cursor -= 1
        del buffer[cursor]
        write("\b")
        redraw_from_cursor()
        refresh_menu()

    def delete_forward() -> None:
        if cursor >= len(buffer):
            return
        del buffer[cursor]
        redraw_from_cursor()
        refresh_menu()

    def move_left() -> None:
        nonlocal cursor
        if cursor <= 0:
            return
        cursor -= 1
        write("\b")
        refresh_menu()

    def move_right() -> None:
        nonlocal cursor
        if cursor >= len(buffer):
            return
        write(buffer[cursor])
        cursor += 1
        refresh_menu()

    def move_home() -> None:
        nonlocal cursor
        if cursor <= 0:
            return
        write(f"\x1b[{cursor}D")
        cursor = 0
        refresh_menu()

    def move_end() -> None:
        nonlocal cursor
        remaining = len(buffer) - cursor
        if remaining <= 0:
            return
        write("".join(buffer[cursor:]))
        cursor = len(buffer)
        refresh_menu()

    def replace_buffer(text: str) -> None:
        nonlocal cursor
        if cursor:
            write(f"\x1b[{cursor}D")
        write("\x1b[K")
        buffer[:] = list(text)
        cursor = len(buffer)
        write(text)
        refresh_menu()

    def move_menu(delta: int) -> bool:
        if not menu_state.matches:
            return False
        menu_state.move(delta)
        refresh_menu()
        return True

    def complete() -> None:
        if cursor != len(buffer):
            return
        current = "".join(buffer)
        completed = completion_state.complete(
            current,
            preferred=menu_state.selected_command(),
        )
        if completed != current:
            replace_buffer(completed)

    helpers = dict(
        write=write,
        insert=insert,
        backspace=backspace,
        delete_forward=delete_forward,
        move_left=move_left,
        move_right=move_right,
        move_home=move_home,
        move_end=move_end,
        move_menu=move_menu,
        clear_menu=clear_menu,
        complete=complete,
    )
    if os.name == "nt":
        return _read_line_protected_windows(buffer, **helpers)
    return _read_line_protected_posix(buffer, **helpers)


def _read_line_protected_posix(
    buffer: list[str],
    *,
    write,
    insert,
    backspace,
    delete_forward,
    move_left,
    move_right,
    move_home,
    move_end,
    move_menu,
    clear_menu,
    complete,
) -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if not ch:
                clear_menu()
                write("\r\n")
                raise EOFError
            if ch in ("\r", "\n"):
                clear_menu()
                write("\r\n")
                return "".join(buffer)
            if ch == "\x03":  # Ctrl+C
                clear_menu()
                write("\r\n")
                raise KeyboardInterrupt
            if ch == "\x04":  # Ctrl+D
                if not buffer:
                    clear_menu()
                    write("\r\n")
                    raise EOFError
                continue
            if ch in ("\x7f", "\b"):
                backspace()
                continue
            if ch == "\x01":  # Ctrl+A home
                move_home()
                continue
            if ch == "\x05":  # Ctrl+E end
                move_end()
                continue
            if ch == "\t":
                complete()
                continue
            if ch == "\x1b":
                # Escape sequence (arrows, delete, home/end)
                seq = sys.stdin.read(1)
                if seq == "[":
                    rest = sys.stdin.read(1)
                    if rest == "D":
                        move_left()
                    elif rest == "C":
                        move_right()
                    elif rest == "A":
                        move_menu(-1)
                    elif rest == "B":
                        move_menu(1)
                    elif rest == "H":
                        move_home()
                    elif rest == "F":
                        move_end()
                    elif rest == "3":
                        tilde = sys.stdin.read(1)
                        if tilde == "~":
                            delete_forward()
                    elif rest in ("1", "7"):
                        tilde = sys.stdin.read(1)
                        if tilde == "~":
                            move_home()
                    elif rest in ("4", "8"):
                        tilde = sys.stdin.read(1)
                        if tilde == "~":
                            move_end()
                continue
            if ch == "\x15":  # Ctrl+U clear buffer only (not the chrome)
                while buffer:
                    backspace()
                continue
            if ord(ch) >= 32:
                insert(ch)
    finally:
        clear_menu()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_line_protected_windows(
    buffer: list[str],
    *,
    write,
    insert,
    backspace,
    delete_forward,
    move_left,
    move_right,
    move_home,
    move_end,
    move_menu,
    clear_menu,
    complete,
) -> str:
    import msvcrt

    while True:
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            clear_menu()
            write("\r\n")
            return "".join(buffer)
        if ch == "\x03":
            clear_menu()
            write("\r\n")
            raise KeyboardInterrupt
        if ch in ("\x08", "\x7f"):
            backspace()
            continue
        if ch == "\t":
            complete()
            continue
        if ch in ("\x00", "\xe0"):
            code = msvcrt.getwch()
            if code == "K":  # left
                move_left()
            elif code == "M":  # right
                move_right()
            elif code == "H":  # up
                move_menu(-1)
            elif code == "P":  # down
                move_menu(1)
            elif code == "G":  # home
                move_home()
            elif code == "O":  # end
                move_end()
            elif code == "S":  # delete
                delete_forward()
            continue
        if ord(ch) >= 32:
            insert(ch)


def read_user_input(completions: Sequence[str] = ()) -> str:
    """Read one prompt line; ``selene $`` is painted but never editable."""
    flush_terminal_input()
    try:
        raw = _read_line_with_fixed_prompt(completions=completions)
    except EOFError:
        return ""
    return sanitize_terminal_input(raw)


def _section_rule(label: str, *, style: str | None = None, glyph: str = GLYPH_SECTION) -> None:
    """Print a frontier-lab section rule with a left-aligned label."""
    from rich.rule import Rule

    color = style or THEME.rule
    title = f"[{THEME.label}]{glyph} {label}[/]"
    _console.print(Rule(title, style=color, align="left", characters="─"))


def assistant_stream_panel(text: str):
    """Build the live assistant renderable."""
    from rich.markdown import Markdown
    from rich.panel import Panel

    body = Markdown(_render_terminal_markdown(text or " "))
    return Panel(
        body,
        border_style=THEME.border,
        padding=(1, 2),
        title=f"[bold {THEME.accent}]{GLYPH_SECTION} response[/]",
        title_align="left",
        subtitle=f"[{THEME.meta}]selene[/]",
        subtitle_align="right",
    )


def print_assistant_message(text: str) -> None:
    """Print the final assistant response in the persistent scrollback."""
    if not text:
        return
    _console.print(assistant_stream_panel(text))
    _console.print()


def print_thinking_header() -> None:
    _section_rule("thinking", style=THEME.thinking, glyph=GLYPH_MARK)


def print_thinking_footer(label: str | None = None) -> None:
    from rich.rule import Rule

    if label:
        _console.print(
            Rule(
                f"[{THEME.thinking}]{GLYPH_MARK} {label}[/]",
                style=THEME.thinking,
                align="left",
                characters="─",
            )
        )
    else:
        _console.print(Rule(style=THEME.thinking, characters="─"))
    _console.print()


def thinking_stream_style() -> str:
    """Style used for streamed chain-of-thought tokens."""
    return f"dim {THEME.thinking}"


def _print_status(icon: str, message: str, color: str = "cyan") -> None:
    """Print a formatted status line to stderr so it doesn't mix with piped output.

    Legacy callers may still pass emoji icons; new code should prefer
    ``print_lab_status`` / ``print_tool_event``.
    """
    _console.print(f"[{color}]{icon}[/]  {message}")


def print_lab_status(
    message: str,
    *,
    kind: str = "info",
    detail: str | None = None,
) -> None:
    """Print a refined status line (success / warn / error / info / run)."""
    styles = {
        "info": (GLYPH_MARK, THEME.accent_soft),
        "run": (GLYPH_RUN, THEME.warning),
        "ok": (GLYPH_OK, THEME.success),
        "warn": (GLYPH_WARN, THEME.warning),
        "error": (GLYPH_ERR, THEME.danger),
        "tool": (GLYPH_TOOL, THEME.accent),
    }
    glyph, color = styles.get(kind, styles["info"])
    if detail:
        _console.print(
            f"  [{color}]{glyph}[/]  {message}  [{THEME.meta}]{detail}[/]"
        )
    else:
        _console.print(f"  [{color}]{glyph}[/]  {message}")


def print_tool_event(
    name: str,
    *,
    phase: str = "run",
    detail: str | None = None,
    message: str | None = None,
) -> None:
    """Print a tool lifecycle line: run / ok / error / parallel."""
    if phase == "parallel":
        print_lab_status(
            message or f"parallel tools × {detail or '?'}",
            kind="info",
        )
        return
    if phase == "error":
        print_lab_status(
            f"[{THEME.label}]{name}[/]  {message or 'failed'}",
            kind="error",
            detail=detail,
        )
        return
    if phase == "ok":
        print_lab_status(
            f"[{THEME.label}]{name}[/]  {message or 'complete'}",
            kind="ok",
            detail=detail,
        )
        return
    print_lab_status(
        f"[{THEME.label}]{name}[/]  {message or 'running'}",
        kind="run",
        detail=detail,
    )


def _welcome_art_lines(width: int) -> list[tuple[str, str]]:
    """Return ``(text, style)`` rows for the Selene cover mark."""
    if width < 64:
        return [
            ("      .o888.", THEME.accent),
            ("    .888P\"", THEME.accent_soft),
            ("   d88P\"", THEME.accent_soft),
            ("   888", THEME.accent),
            ("   Y88b.", THEME.accent_soft),
            ("    \"Y888.", THEME.accent),
            ("      `\"\"'", THEME.muted),
            ("", THEME.accent),
            ("  S E L E N E", f"bold {THEME.accent}"),
            ("  local agent runtime", THEME.meta),
        ]

    return [
        ("              .o8888888o.", THEME.accent),
        ("           .d888888888888b.", THEME.accent),
        ("         .d888888P\"'", THEME.accent_soft),
        ("        d88888P\"", THEME.accent_soft),
        ("       d8888P\"", THEME.accent_soft),
        ("       88888\"", THEME.accent),
        ("       Y8888b.", THEME.accent_soft),
        ("        Y88888b.", THEME.accent_soft),
        ("         \"Y888888bo.", THEME.accent),
        ("           \"Y8888888b.", THEME.accent),
        ("             `\"Y888P\"'", THEME.muted),
    ]


def _welcome_meta_pairs(context: dict | None = None) -> list[tuple[str, str]]:
    """Build key/value pairs for the splash status strip."""
    pairs: list[tuple[str, str]] = []
    ctx = dict(context or {})
    if not ctx:
        try:
            from agent.runtime_config import get_runtime_config

            runtime = get_runtime_config()
            ctx = {
                "profile": runtime.profile.value,
                "model": runtime.chat_model,
                "num_ctx": str(runtime.num_ctx),
                "num_predict": str(runtime.num_predict),
            }
        except Exception:
            ctx = {}
        try:
            from agent.platform_runtime import platform_family

            ctx.setdefault("platform", platform_family())
        except Exception:
            pass

    mapping = (
        ("profile", "profile"),
        ("model", "model"),
        ("num_ctx", "ctx"),
        ("num_predict", "out"),
        ("platform", "host"),
    )
    for key, label in mapping:
        value = ctx.get(key)
        if value is not None and str(value).strip():
            pairs.append((label, str(value)))
    return pairs


def print_welcome_header(context: dict | None = None) -> None:
    """Print a framed lab-style splash: crescent mark, product identity, runtime strip."""
    from rich.align import Align
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    width = max(48, min(_console.size.width, 92))
    art_lines = _welcome_art_lines(width)
    art = Text(justify="left")
    for index, (line, style) in enumerate(art_lines):
        if index:
            art.append("\n")
        art.append(line, style=style)

    brand = Text()
    brand.append("SELENE", style=f"bold {THEME.accent}")
    brand.append("\n")
    brand.append("local agent runtime", style=THEME.meta)
    brand.append("\n")
    brand.append("tools  ·  vault  ·  ollama", style=f"dim {THEME.accent_2}")

    if width >= 64:
        header = Table.grid(padding=(0, 3))
        header.add_column(justify="left", no_wrap=True)
        header.add_column(justify="left", ratio=1)
        header.add_row(art, Align.left(brand, vertical="middle"))
        body_renderable: object = header
    else:
        body_renderable = Group(Align.center(art), Text(""), Align.center(brand))

    meta = _welcome_meta_pairs(context)
    footer = Text(justify="left")
    if meta:
        for index, (label, value) in enumerate(meta):
            if index:
                footer.append("   ", style=THEME.meta)
            footer.append(f"{label} ", style=THEME.meta)
            footer.append(value, style=THEME.ink)

    panel = Panel(
        Group(body_renderable, Text(""), footer) if meta else body_renderable,
        border_style=THEME.border,
        padding=(1, 2),
        title=f"[bold {THEME.accent}]{GLYPH_MARK} selene[/]",
        title_align="left",
        subtitle=f"[{THEME.meta}]frontier local · type /help[/]",
        subtitle_align="right",
        width=min(width, _console.size.width),
    )
    _console.print()
    _console.print(Align.center(panel))
    _console.print()
    _console.print(
        f"  [{THEME.meta}]{GLYPH_PROMPT} local inference · /help for commands · "
        f"empty line ignored[/]"
    )
    _console.print()


class _Spinner:
    """Animated spinner wrapper that uses rich.status.Status."""

    def __init__(self, message: str = "reasoning", color: str | None = None) -> None:
        self._message = message
        self._color = color or THEME.thinking
        self._status = _console.status(
            self._render_message(),
            spinner="dots12",
            spinner_style=f"{self._color}",
        )
        import threading

        self._stop_event = threading.Event()
        self._thread = type("MockThread", (), {"is_alive": lambda: False, "join": lambda: None})()

    def _render_message(self) -> str:
        return (
            f"[{self._color}]{GLYPH_RUN}[/] "
            f"[{THEME.label}]{self._message}[/]"
            f"[{THEME.meta}] …[/]"
        )

    def start(self) -> "_Spinner":
        self._stop_event.clear()
        self._status.start()
        return self

    def update(self, message: str) -> None:
        self._message = message
        self._status.update(self._render_message())

    def stop(self) -> None:
        self._stop_event.set()
        self._status.stop()


# Small helpers for rendering common LaTeX math to terminal-friendly text
_UNICODE_FRACTIONS = {
    ("1", "2"): "½",
    ("1", "3"): "⅓",
    ("2", "3"): "⅔",
    ("1", "4"): "¼",
    ("3", "4"): "¾",
    ("1", "5"): "⅕",
    ("2", "5"): "⅖",
    ("3", "5"): "⅗",
    ("4", "5"): "⅘",
    ("1", "6"): "⅙",
    ("5", "6"): "⅚",
    ("1", "8"): "⅛",
    ("3", "8"): "⅜",
    ("5", "8"): "⅝",
    ("7", "8"): "⅞",
}

_LATEX_SYMBOLS = {
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\epsilon": "ε",
    r"\theta": "θ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\pi": "π",
    r"\sigma": "σ",
    r"\phi": "φ",
    r"\omega": "ω",
    r"\partial": "∂",
    r"\nabla": "∇",
    r"\sum": "∑",
    r"\prod": "∏",
    r"\int": "∫",
    r"\iint": "∬",
    r"\iiint": "∭",
    r"\oint": "∮",
    r"\times": "×",
    r"\cdot": "·",
    r"\div": "÷",
    r"\pm": "±",
    r"\mp": "∓",
    r"\leq": "≤",
    r"\geq": "≥",
    r"\lt": "<",
    r"\gt": ">",
    r"\neq": "≠",
    r"\approx": "≈",
    r"\equiv": "≡",
    r"\sim": "∼",
    r"\simeq": "≃",
    r"\propto": "∝",
    r"\infty": "∞",
    r"\to": "→",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
    r"\leftrightarrow": "↔",
    r"\mapsto": "↦",
    r"\uparrow": "↑",
    r"\downarrow": "↓",
    r"\updownarrow": "↕",
    r"\implies": "⇒",
    r"\iff": "⇔",
    r"\Rightarrow": "⇒",
    r"\Leftarrow": "⇐",
    r"\Leftrightarrow": "⇔",
    r"\Uparrow": "⇑",
    r"\Downarrow": "⇓",
    r"\Updownarrow": "⇕",
    r"\nearrow": "↗",
    r"\searrow": "↘",
    r"\swarrow": "↙",
    r"\nwarrow": "↖",
    r"\longleftarrow": "⟵",
    r"\longrightarrow": "⟶",
    r"\longleftrightarrow": "⟷",
    r"\Longleftarrow": "⟸",
    r"\Longrightarrow": "⟹",
    r"\Longleftrightarrow": "⟺",
    r"\longmapsto": "⟼",
    r"\hookleftarrow": "↩",
    r"\hookrightarrow": "↪",
    r"\leftharpoonup": "↼",
    r"\leftharpoondown": "↽",
    r"\rightharpoonup": "⇀",
    r"\rightharpoondown": "⇁",
    r"\rightleftharpoons": "⇌",
    r"\rightsquigarrow": "⇝",
    r"\circlearrowleft": "↺",
    r"\circlearrowright": "↻",
    r"\curvearrowleft": "↶",
    r"\curvearrowright": "↷",
    r"\leftleftarrows": "⇇",
    r"\rightrightarrows": "⇉",
    r"\upuparrows": "⇈",
    r"\downdownarrows": "⇊",
    r"\rightleftarrows": "⇄",
    r"\leftrightarrows": "⇆",
    r"\Lleftarrow": "⇚",
    r"\Rrightarrow": "⇛",
    r"\twoheadleftarrow": "↞",
    r"\twoheadrightarrow": "↠",
    r"\leftarrowtail": "↢",
    r"\rightarrowtail": "↣",
    r"\forall": "∀",
    r"\exists": "∃",
    r"\emptyset": "∅",
    r"\in": "∈",
    r"\notin": "∉",
    r"\subseteq": "⊆",
    r"\subset": "⊂",
    r"\supseteq": "⊇",
    r"\supset": "⊃",
    r"\cup": "∪",
    r"\cap": "∩",
    r"\setminus": "∖",
    r"\sin": "sin",
    r"\cos": "cos",
    r"\tan": "tan",
    r"\cot": "cot",
    r"\sec": "sec",
    r"\csc": "csc",
    r"\arcsin": "arcsin",
    r"\arccos": "arccos",
    r"\arctan": "arctan",
    r"\arccot": "arccot",
    r"\arcsec": "arcsec",
    r"\arccsc": "arccsc",
    r"\asin": "arcsin",
    r"\acos": "arccos",
    r"\atan": "arctan",
    r"\sinh": "sinh",
    r"\cosh": "cosh",
    r"\tanh": "tanh",
    r"\coth": "coth",
    r"\arcsinh": "arcsinh",
    r"\arccosh": "arccosh",
    r"\arctanh": "arctanh",
    r"\asinh": "arcsinh",
    r"\acosh": "arccosh",
    r"\atanh": "arctanh",
    r"\exp": "exp",
    r"\log": "log",
    r"\ln": "ln",
    r"\lg": "lg",
    r"\lim": "lim",
    r"\max": "max",
    r"\min": "min",
    r"\deg": "deg",
    r"\mathbb": "",
    r"\mathrm": "",
    r"\mathbf": "",
    r"\mathcal": "",
    r"\text": "",
    r"\textA": "",
    r"\textB": "",
    r"\textbf": "",
    r"\textit": "",
    r"\textsf": "",
    r"\texttt": "",
    r"\hline": "─",
    r"\array": "",
    r"\endarray": "",
    r"\wedge": "∧",
    r"\vee": "∨",
    r"\oplus": "⊕",
    r"\otimes": "⊗",
    r"\ll": "≪",
    r"\gg": "≫",
    r"\lnot": "¬",
    r"\neg": "¬",
    r"\land": "∧",
    r"\lor": "∨",
    r"\bin": "bin",
    r"\hex": "hex",
    r"\dots": "…",
    r"\cdots": "⋯",
    r"\vdots": "⋮",
    r"\ddots": "⋱",
    r"\Alpha": "A",
    r"\Beta": "B",
    r"\Gamma": "Γ",
    r"\Delta": "Δ",
    r"\Epsilon": "E",
    r"\Zeta": "Z",
    r"\Eta": "H",
    r"\Theta": "Θ",
    r"\Iota": "I",
    r"\Kappa": "K",
    r"\Lambda": "Λ",
    r"\Mu": "M",
    r"\Nu": "N",
    r"\Xi": "Ξ",
    r"\Omicron": "O",
    r"\Pi": "Π",
    r"\Rho": "P",
    r"\Sigma": "Σ",
    r"\Tau": "T",
    r"\Upsilon": "Υ",
    r"\Phi": "Φ",
    r"\Chi": "X",
    r"\Psi": "Ψ",
    r"\Omega": "Ω",
    r"\zeta": "ζ",
    r"\eta": "η",
    r"\iota": "ι",
    r"\kappa": "κ",
    r"\nu": "ν",
    r"\xi": "ξ",
    r"\rho": "ρ",
    r"\tau": "τ",
    r"\upsilon": "υ",
    r"\chi": "χ",
    r"\psi": "ψ",
    r"\degree": "°",
    r"\heartsuit": "♥",
    r"\diamondsuit": "♦",
    r"\clubsuit": "♣",
    r"\spadesuit": "♠",
    r"\checkmark": "✓",
    r"\bullet": "•",
    r"\star": "★",
    r"\ast": "*",
    r"\angle": "∠",
    r"\perp": "⊥",
    r"\parallel": "∥",
    r"\hbar": "ℏ",
    r"\ell": "ℓ",
    r"\square": "□",
    r"\triangle": "△",
    r"\diamond": "◇",
    r"\circ": "○",
    r"\quad": " ",
    r"\qquad": " ",
    r"\left": "",
    r"\right": "",
}

# Keep the CLI's symbol coverage aligned with the browser renderer. The base
# table above includes terminal-specific function names and arrows; these are
# the additional Greek variants, operators, relations, sets, and symbols used
# by the webview's Unicode pass.
_LATEX_SYMBOLS.update({
    r"\varepsilon": "ϵ",
    r"\vartheta": "ϑ",
    r"\varpi": "ϖ",
    r"\varrho": "ϱ",
    r"\varsigma": "ς",
    r"\varphi": "ϕ",
    r"\omicron": "ο",
    r"\coprod": "∐",
    r"\ominus": "⊖",
    r"\oslash": "⊘",
    r"\odot": "⊙",
    r"\bigoplus": "⨁",
    r"\bigotimes": "⨂",
    r"\bigodot": "⨀",
    r"\dagger": "†",
    r"\ddagger": "‡",
    r"\ne": "≠",
    r"\cong": "≅",
    r"\le": "≤",
    r"\ge": "≥",
    r"\prec": "≺",
    r"\succ": "≻",
    r"\preceq": "⪯",
    r"\succeq": "⪰",
    r"\nparallel": "∦",
    r"\mid": "∣",
    r"\asymp": "≍",
    r"\doteq": "≐",
    r"\models": "⊨",
    r"\nexists": "∄",
    r"\therefore": "∴",
    r"\because": "∵",
    r"\top": "⊤",
    r"\bot": "⊥",
    r"\varnothing": "∅",
    r"\ni": "∋",
    r"\notni": "∌",
    r"\nsubseteq": "⊈",
    r"\nsupseteq": "⊉",
    r"\uplus": "⊎",
    r"\bigcup": "⋃",
    r"\bigcap": "⋂",
    r"\sqsubset": "⊏",
    r"\sqsupset": "⊐",
    r"\sqsubseteq": "⊑",
    r"\sqsupseteq": "⊒",
    r"\sqcup": "⊔",
    r"\sqcap": "⊓",
    r"\gets": "←",
    r"\measuredangle": "∡",
    r"\lozenge": "◊",
    r"\aleph": "ℵ",
    r"\beth": "ℶ",
    r"\gimel": "ℷ",
    r"\Re": "ℜ",
    r"\Im": "ℑ",
    r"\wp": "℘",
    r"\prime": "′",
    r"\backprime": "‵",
    r"\copyright": "©",
    r"\registered": "®",
    r"\pounds": "£",
    r"\euro": "€",
    r"\yen": "¥",
})

# Pre-sorted symbol keys (longest first) to avoid re-sorting on every call
_SORTED_LATEX_KEYS = sorted(_LATEX_SYMBOLS.keys(), key=lambda s: -len(s))

_LATEX_SPACING = {
    r"\,": " ",
    r"\!": "",
    r"\;": " ",
    r"\:": " ",
}

# Unicode superscript / subscript mapping for common characters
_SUP_MAP = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "n": "ⁿ",
    "i": "ⁱ",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "A": "ᴬ",
    "B": "ᴮ",
    "C": "ᶜ",
    "D": "ᴰ",
    "E": "ᴱ",
    "F": "ᶠ",
    "x": "ˣ",
    "h": "ʰ",
}

_SUB_MAP = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
}


def _to_superscript(text: str) -> str:
    """Convert normal text characters to their Unicode superscript equivalents.
    
    Args:
        text (str): The string to convert.
        
    Returns:
        str: The converted superscript string.
    """
    out = []
    for ch in text:
        out.append(_SUP_MAP.get(ch, ch))
    return "".join(out)


def _to_subscript(text: str) -> str:
    """Convert normal text characters to their Unicode subscript equivalents.
    
    Args:
        text (str): The string to convert.
        
    Returns:
        str: The converted subscript string.
    """
    out = []
    for ch in text:
        out.append(_SUB_MAP.get(ch, ch))
    return "".join(out)


def _extract_braced(text: str, start: int) -> tuple[str, int] | None:
    """Extract a substring enclosed in matching curly braces.
    
    Args:
        text (str): The full text containing the braces.
        start (int): The index where the opening brace '{' is located.
        
    Returns:
        tuple[str, int] | None: A tuple containing the extracted inner string and 
                                the index immediately following the closing brace.
                                Returns None if parsing fails.
    """
    if start >= len(text) or text[start] != "{":
        return None

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def _render_latex_math(expr: str) -> str:
    """Convert a LaTeX math expression into terminal-friendly Unicode text.
    
    This function handles common LaTeX structures such as fractions, square roots,
    symbols, superscripts, and subscripts, rendering them cleanly without needing
    a full LaTeX engine.
    
    Args:
        expr (str): The raw LaTeX expression.
        
    Returns:
        str: The Unicode-rendered representation of the expression.
    """
    expr = expr.strip()
    if not expr:
        return ""

    expr = expr.replace(r"\displaystyle", "")
    expr = expr.replace(r"\textstyle", "")

    # Handle \frac{a}{b}
    def replace_frac(text: str) -> str:
        output = []
        index = 0
        while True:
            pos = text.find(r"\frac", index)
            if pos == -1:
                output.append(text[index:])
                break
            
            output.append(text[index:pos])
            numerator = _extract_braced(text, pos + 5)
            if numerator is not None:
                denominator = _extract_braced(text, numerator[1])
                if denominator is not None:
                    rendered_num = _render_latex_math(numerator[0])
                    rendered_den = _render_latex_math(denominator[0])
                    if rendered_num.isdigit() and rendered_den.isdigit():
                        output.append(_UNICODE_FRACTIONS.get((rendered_num, rendered_den), f"{rendered_num}/{rendered_den}"))
                    else:
                        output.append(f"({rendered_num})/({rendered_den})")
                    index = denominator[1]
                    continue
            
            # If extraction failed, just append the string and move on
            output.append(r"\frac")
            index = pos + 5
            
        return "".join(output)

    # Handle \sqrt{...}
    def replace_sqrt(text: str) -> str:
        output = []
        index = 0
        while True:
            pos = text.find(r"\sqrt", index)
            if pos == -1:
                output.append(text[index:])
                break
            
            output.append(text[index:pos])
            radicand = _extract_braced(text, pos + 5)
            if radicand is not None:
                rendered = _render_latex_math(radicand[0])
                output.append(f"√({rendered})")
                index = radicand[1]
                continue
                
            output.append(r"\sqrt")
            index = pos + 5
            
        return "".join(output)

    expr = replace_frac(expr)
    expr = replace_sqrt(expr)

    # Replace common LaTeX symbols using pre-sorted keys
    for latex in _SORTED_LATEX_KEYS:
        expr = expr.replace(latex, _LATEX_SYMBOLS[latex])

    for latex, replacement in _LATEX_SPACING.items():
        expr = expr.replace(latex, replacement)

    # Superscript/subscript handling: ^{...}, _{...}, ^x, _x
    def replace_scripts(text: str) -> str:
        # caret superscript
        def sup_repl(m: re.Match[str]) -> str:
            token = m.group(1)
            if token.startswith("{") and token.endswith("}"):
                inner = token[1:-1]
            else:
                inner = token
            inner_rendered = _render_latex_math(inner)
            mapped = _to_superscript(inner_rendered)
            if mapped != inner_rendered:
                return mapped
            return f"^{inner_rendered}"

        # subscript
        def sub_repl(m: re.Match[str]) -> str:
            token = m.group(1)
            if token.startswith("{") and token.endswith("}"):
                inner = token[1:-1]
            else:
                inner = token
            inner_rendered = _render_latex_math(inner)
            mapped = _to_subscript(inner_rendered)
            if mapped != inner_rendered:
                return mapped
            return f"_{inner_rendered}"

        text = _RE_SUP.sub(sup_repl, text)
        text = _RE_SUB.sub(sub_repl, text)
        return text

    expr = replace_scripts(expr)

    # Strip remaining braces and collapse whitespace
    expr = expr.replace("{", "").replace("}", "")
    expr = _RE_COLLAPSE_WS.sub(" ", expr)
    return expr.strip()


# Pre-compiled regex patterns for markdown rendering
_RE_BLOCK_LATEX = re.compile(r"\$\$(.+?)\$\$", flags=re.DOTALL)
_RE_INLINE_LATEX = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$")
_RE_SUP = re.compile(r"\^(\{.*?\}|.)")
_RE_SUB = re.compile(r"_(\{.*?\}|.)")
_RE_COLLAPSE_WS = re.compile(r"\s+")
_RE_MARKDOWN_CODE = re.compile(r"(```[\s\S]*?```|`[^`\n]*`)")
_RE_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_RE_TASK_ITEM = re.compile(r"(?m)^(\s*[-*+]\s+)\[([ xX])\]\s+")


def _render_bare_latex_symbols(text: str) -> str:
    """Convert known commands outside math delimiters without changing prose."""
    text = _RE_LATEX_COMMAND.sub(
        lambda match: _LATEX_SYMBOLS.get(match.group(0), match.group(0)),
        text,
    )
    for latex, replacement in _LATEX_SPACING.items():
        text = text.replace(latex, replacement)
    return text


def _render_terminal_markdown(text: str) -> str:
    """Pre-process markdown text to render LaTeX blocks before passing to rich.Markdown.
    
    Finds inline ($...$) and block ($$...$$) LaTeX expressions and replaces them
    with their Unicode equivalents.
    
    Args:
        text (str): The raw markdown string containing potential LaTeX.
        
    Returns:
        str: The processed markdown string with LaTeX rendered to Unicode.
    """
    def replace_block(match: re.Match[str]) -> str:
        rendered = _render_latex_math(match.group(1))
        return f"\n{rendered}\n"

    def replace_inline(match: re.Match[str]) -> str:
        return _render_latex_math(match.group(1))

    def render_prose(segment: str) -> str:
        segment = _RE_BLOCK_LATEX.sub(replace_block, segment)
        segment = _RE_INLINE_LATEX.sub(replace_inline, segment)
        segment = _render_bare_latex_symbols(segment)
        segment = _RE_TASK_ITEM.sub(
            lambda match: f"{match.group(1)}{'☑' if match.group(2).lower() == 'x' else '☐'} ",
            segment,
        )
        return segment.replace(r"\$", "$")

    # Code examples must remain literal, including their backslashes. Capturing
    # delimiters keeps fenced and inline code in the split result unchanged.
    parts = _RE_MARKDOWN_CODE.split(text)
    return "".join(part if index % 2 else render_prose(part) for index, part in enumerate(parts))
