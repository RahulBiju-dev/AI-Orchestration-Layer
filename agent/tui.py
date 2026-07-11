"""
agent/tui.py — Full-screen Selene TUI (Claude Code / Grok Build style).

Layout:
  • top status bar with model / profile meta
  • scrollable chat transcript (user + assistant + tools + status)
  • fixed bottom composer with slash-command palette

Generation runs on a worker thread; UI updates are marshalled onto the
Textual message loop via call_from_thread.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Callable, Sequence

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from agent.terminal import (
    GLYPH_DOT,
    GLYPH_ERR,
    GLYPH_MARK,
    GLYPH_OK,
    GLYPH_PROMPT,
    GLYPH_RUN,
    GLYPH_SECTION,
    GLYPH_TOOL,
    GLYPH_WARN,
    _render_terminal_markdown,
    set_display_sink,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\].*?(?:\x07|\x1b\\)")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def _estimate_tokens(text: str) -> int:
    """Fast tokenizer-free estimate (~4 ASCII chars / token; non-ASCII ≈ 1 each).

    Matches the heuristic used by the agent context budget so the thinking
    counter stays consistent with the rest of Selene.
    """
    value = str(text or "")
    if not value:
        return 0
    ascii_chars = sum(1 for character in value if ord(character) < 128)
    non_ascii_chars = len(value) - ascii_chars
    return ascii_chars // 4 + non_ascii_chars + 1


# ── Display sink (bridges core print_* → TUI) ─────────────────────────


class TuiDisplaySink:
    """Thread-safe sink that forwards chrome events into the running app."""

    is_tui = True

    def __init__(self, app: "SeleneTui") -> None:
        self._app = app

    def _call(self, method: str, *args, **kwargs) -> None:
        callback = getattr(self._app, method, None)
        if callback is None:
            return
        try:
            self._app.call_from_thread(callback, *args, **kwargs)
        except Exception:
            # App may already be shutting down.
            pass

    def lab_status(self, message: str, *, kind: str = "info", detail: str | None = None) -> None:
        # Transient run states use the activity line; permanent events stay in chat.
        if kind == "run":
            self.activity_start(message if not detail else f"{message} · {detail}")
            return
        self._call("ui_status", message, kind, detail)

    def apply_theme(self, name: str) -> None:
        self._call("ui_apply_theme", name)

    def activity_start(self, label: str = "Thinking") -> None:
        self._call("ui_activity_start", label)

    def activity_update(self, label: str) -> None:
        self._call("ui_activity_update", label)

    def activity_stop(self) -> None:
        self._call("ui_activity_stop")

    def thinking_header(self) -> None:
        self._call("ui_thinking_start")

    def thinking_delta(self, text: str) -> None:
        self._call("ui_thinking_delta", text)

    def thinking_footer(self, label: str | None = None) -> None:
        self._call("ui_thinking_end", label)

    def content_stream(self, text: str) -> None:
        self._call("ui_content_stream", text)

    def content_final(self, text: str) -> None:
        self._call("ui_content_final", text)

    def generation_stats(
        self,
        *,
        elapsed: float,
        total_tokens: int,
        tokens_per_sec: float,
    ) -> None:
        self._call("ui_stats", elapsed, total_tokens, tokens_per_sec)

    def command_help(
        self,
        entries: Sequence[tuple[str, str]],
        *,
        title: str = "commands",
        subtitle: str | None = None,
    ) -> None:
        self._call("ui_help", list(entries), title, subtitle)

    def console_line(self, text: str) -> None:
        cleaned = _strip_ansi(text).rstrip()
        if cleaned:
            self._call("ui_console_line", cleaned)


class _CaptureFile:
    """File-like object that feeds Rich Console output into the TUI log."""

    def __init__(self, sink: TuiDisplaySink) -> None:
        self._sink = sink
        self._buf = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._sink.console_line(line)
        return len(data)

    def flush(self) -> None:
        if self._buf.strip():
            self._sink.console_line(self._buf)
        self._buf = ""

    def isatty(self) -> bool:
        return False


# ── Textual application ───────────────────────────────────────────────


def _import_textual():
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, Vertical, VerticalScroll
        from textual.widgets import Input, Static
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The Selene TUI requires the 'textual' package. "
            "Install it with: pip install 'textual>=1.0.0'"
        ) from exc
    return {
        "App": App,
        "ComposeResult": ComposeResult,
        "Binding": Binding,
        "Horizontal": Horizontal,
        "Vertical": Vertical,
        "VerticalScroll": VerticalScroll,
        "Input": Input,
        "Static": Static,
    }


def _filter_slash_commands(
    query: str,
    commands: Sequence[str],
    descriptions: dict[str, str],
    *,
    limit: int = 14,
) -> list[tuple[str, str]]:
    """Rank slash matches: prefix first, then substring / word hits."""
    raw = (query or "").strip()
    if not raw.startswith("/"):
        return []
    folded = raw.casefold()
    # Bare "/" shows the full catalog (capped).
    if folded == "/":
        return [(cmd, descriptions.get(cmd, "")) for cmd in commands[:limit]]

    prefix: list[tuple[str, str]] = []
    contains: list[tuple[str, str]] = []
    for cmd in commands:
        cfold = cmd.casefold()
        if cfold.startswith(folded):
            prefix.append((cmd, descriptions.get(cmd, "")))
        elif folded.lstrip("/") and folded.lstrip("/") in cfold:
            contains.append((cmd, descriptions.get(cmd, "")))
    ranked = prefix + [item for item in contains if item not in prefix]
    return ranked[:limit]


def build_app_class():
    """Construct SeleneTui after Textual is importable (lazy for optional dep)."""
    t = _import_textual()
    App = t["App"]
    ComposeResult = t["ComposeResult"]
    Binding = t["Binding"]
    Horizontal = t["Horizontal"]
    Vertical = t["Vertical"]
    VerticalScroll = t["VerticalScroll"]
    Input = t["Input"]
    Static = t["Static"]

    # Colors come from Textual CSS variables ($background, $selene-*, …)
    # driven by agent.tui_themes. Hardcoded greys remain only as Rich fallbacks
    # until a theme is applied on mount.

    class ChatView(VerticalScroll):
        """Scrollable transcript region."""

        can_focus = False
        DEFAULT_CSS = """
        ChatView {
            height: 1fr;
            padding: 0 1;
            scrollbar-gutter: stable;
            background: transparent;
            scrollbar-background: $background;
            scrollbar-color: $primary 30%;
            scrollbar-color-hover: $primary;
        }
        """

    class MessageBlock(Static):
        """One visual block in the transcript."""

        DEFAULT_CSS = """
        MessageBlock {
            width: 100%;
            margin: 0 0 1 0;
            padding: 0 1;
            color: $secondary;
        }
        /* —— Important: user prompts & model responses —— */
        MessageBlock.user {
            color: $foreground;
            background: $panel;
            border-left: heavy $accent;
            padding: 1 2;
            margin: 0 0 1 0;
        }
        MessageBlock.assistant {
            color: $foreground;
            background: $panel;
            border-left: heavy $accent;
            padding: 1 2;
            margin: 0 0 1 0;
        }
        /* —— Secondary: slash / status / tools / thinking —— */
        MessageBlock.command {
            color: $secondary;
            background: transparent;
            border-left: solid $primary 20%;
            padding: 0 1 0 2;
            margin: 0 0 0 0;
            text-style: dim;
        }
        MessageBlock.thinking {
            color: $primary 40%;
            border-left: solid $primary 20%;
            padding-left: 2;
            margin-bottom: 0;
            text-style: dim italic;
        }
        MessageBlock.activity {
            color: $primary 40%;
            padding-left: 2;
            margin: 0 0 0 0;
            height: 1;
            text-style: dim;
        }
        MessageBlock.status {
            color: $primary 40%;
            padding-left: 2;
            margin-bottom: 0;
            text-style: dim;
        }
        MessageBlock.tool {
            color: $secondary;
            padding-left: 2;
            margin-bottom: 0;
            text-style: dim;
        }
        MessageBlock.error {
            color: $error;
            border-left: wide $error;
            padding-left: 1;
        }
        MessageBlock.system {
            color: $primary 40%;
            padding-left: 2;
            margin-bottom: 0;
            text-style: dim;
        }
        """

    class ThinkingFold(Static):
        """Collapsed thinking summary with optional full-text dropdown.

        After the model finishes thinking, this stays on screen (above the
        response) so the user can expand and read the chain-of-thought while
        generation continues or after it completes.
        """

        can_focus = True
        DEFAULT_CSS = """
        ThinkingFold {
            width: 100%;
            color: $primary 40%;
            border-left: solid $primary 20%;
            padding: 0 1 0 2;
            margin: 0 0 1 0;
            background: transparent;
        }
        ThinkingFold:hover {
            background: $surface;
            color: $secondary;
        }
        ThinkingFold:focus {
            background: $surface;
            border-left: solid $primary;
            color: $secondary;
        }
        ThinkingFold.-expanded {
            color: $secondary;
            max-height: 20;
            overflow-y: auto;
            background: $boost;
            padding: 1 2;
        }
        """

        BINDINGS = [
            Binding("enter", "toggle", "Expand/collapse", show=False),
            Binding("space", "toggle", "Expand/collapse", show=False),
        ]

        def __init__(
            self,
            full_text: str,
            tokens: int,
            *,
            title: str = "thinking",
            interrupted: bool = False,
            **kwargs,
        ) -> None:
            self._full_text = str(full_text or "")
            self._tokens = max(0, int(tokens or 0))
            self._title = (title or "thinking").strip() or "thinking"
            self._interrupted = bool(interrupted)
            self._expanded = False
            super().__init__(self._render_view(), **kwargs)

        def _token_label(self) -> str:
            if self._tokens <= 0:
                return ""
            unit = "token" if self._tokens == 1 else "tokens"
            return f"  ·  ~{self._tokens} {unit}"

        def _header(self) -> Text:
            chevron = "▾" if self._expanded else "▸"
            line = Text()
            if self._interrupted:
                line.append(f"{chevron}  ", style="#8a8a60")
                line.append(f"{GLYPH_WARN} {self._title}", style="#8a8a60")
            else:
                line.append(f"{chevron}  ", style="#6b6b6b")
                line.append(f"{GLYPH_OK} {self._title}", style="#6a8a6a")
            if self._tokens:
                line.append(self._token_label(), style="#555555")
            if self._full_text.strip():
                hint = "click to collapse" if self._expanded else "click to expand"
                line.append(f"  ·  {hint}", style="#555555")
            else:
                line.append("  ·  (empty)", style="#555555")
            return line

        def _render_view(self) -> object:
            header = self._header()
            if not self._expanded or not self._full_text.strip():
                return header
            body = Text(self._full_text.rstrip(), style="#6b6b6b")
            return Group(header, Text(""), body)

        def action_toggle(self) -> None:
            if not self._full_text.strip():
                return
            self._expanded = not self._expanded
            self.set_class(self._expanded, "-expanded")
            self.update(self._render_view())

        def on_click(self, event) -> None:  # noqa: ANN001
            event.stop()
            self.action_toggle()

    class SlashPalette(Static):
        """Command palette above the composer — ranked list with descriptions."""

        DEFAULT_CSS = """
        SlashPalette {
            display: none;
            height: auto;
            max-height: 16;
            padding: 1 1;
            background: $boost;
            color: $secondary;
            border: round $primary 20%;
            margin: 0 1 1 1;
        }
        SlashPalette.-visible {
            display: block;
        }
        """

        def _palette(self) -> dict[str, str]:
            from agent.tui_themes import DEFAULT_THEME, rich_palette

            app = self.app
            return getattr(app, "_selene_palette", None) or rich_palette(DEFAULT_THEME)

        def show_matches(
            self,
            matches: list[tuple[str, str]],
            selected: int,
            *,
            query: str = "/",
            total: int | None = None,
        ) -> None:
            if not matches:
                self.remove_class("-visible")
                self.update("")
                return

            # Build with rich.Text (not markup strings). Descriptions often
            # contain [brackets] like "/load [name|index]", which would corrupt
            # Rich markup and make selection backgrounds bleed onto later rows.
            pal = self._palette()
            muted = pal["muted"]
            faint = pal["faint"]
            select_fg = pal["select_fg"]
            select_bg = pal["select_bg"]

            cmd_width = min(28, max(len(cmd) for cmd, _ in matches))
            total = total if total is not None else len(matches)
            count = f"{len(matches)}" + (f"/{total}" if total > len(matches) else "")

            body = Text()
            body.append("commands  ", style=f"bold {muted}")
            body.append(f"{count}  ·  filter {query or '/'}", style=faint)
            body.append("\n")
            body.append("─" * min(56, cmd_width + 28), style=faint)
            body.append("\n")

            for index, (command, description) in enumerate(matches):
                padded = command.ljust(cmd_width)
                desc = (description or "").strip()
                if len(desc) > 42:
                    desc = desc[:41] + "…"
                if index == selected:
                    body.append(
                        f" ▐ {padded}  {desc} ",
                        style=f"bold {select_fg} on {select_bg}",
                    )
                else:
                    body.append(f"   {padded}", style=muted)
                    body.append(f"  {desc}", style=faint)
                body.append("\n")

            body.append(
                "↑↓ / ^N ^P  move  ·  tab complete  ·  enter run  ·  esc dismiss",
                style=faint,
            )
            self.update(body)
            self.add_class("-visible")

        def hide_palette(self) -> None:
            self.remove_class("-visible")
            self.update("")

    class Composer(Vertical):
        """Boxed prompt line + outside shortcut strip with accented keys."""

        DEFAULT_CSS = """
        Composer {
            height: auto;
            dock: bottom;
            background: $background;
            padding: 0 1 1 1;
            margin: 0 0 0 0;
        }
        #input-shell {
            height: 3;
            border: round $primary 30%;
            background: $background;
            padding: 0 1;
            margin: 0 0 0 0;
            /* Top-align so the '>' shares the Input's text row (Input draws y=0). */
            align: left top;
        }
        #input-shell:focus-within {
            border: round $primary;
        }
        #prompt-glyph {
            width: 2;
            height: 1;
            min-width: 2;
            color: $foreground;
            content-align: left middle;
            background: transparent;
            text-style: bold;
            padding: 0 0;
            margin: 0 0;
        }
        #prompt-input {
            width: 1fr;
            height: 3;
            background: transparent;
            color: $foreground;
            border: none;
            padding: 0 0;
            margin: 0 0;
        }
        #prompt-input:focus {
            background: transparent;
            border: none;
        }
        #prompt-input > .input--placeholder {
            color: $primary 40%;
        }
        #composer-meta {
            width: auto;
            height: 3;
            color: $primary 40%;
            content-align: right middle;
            padding: 0 0 0 1;
            text-style: dim;
            background: transparent;
        }
        #composer-footer {
            height: 1;
            padding: 0 1;
            background: transparent;
            align: left middle;
        }
        #composer-hint {
            width: 1fr;
            height: 1;
            color: $primary 40%;
            background: transparent;
        }
        #composer-context {
            width: auto;
            height: 1;
            color: $primary 50%;
            content-align: right middle;
            text-style: dim;
            background: transparent;
            padding: 0 0 0 1;
        }
        """

        def __init__(self, meta_text: str = "", **kwargs) -> None:
            super().__init__(**kwargs)
            self._meta_text = meta_text

        def _shortcut_strip(self) -> Text:
            """Outside-the-box key strip: accented keys, dim actions, ^ for Ctrl."""
            from agent.tui_themes import DEFAULT_THEME, rich_palette

            pal = getattr(self.app, "_selene_palette", None) or rich_palette(DEFAULT_THEME)
            faint = pal["faint"]
            soft = pal["text_soft"]
            items = (
                ("↵", "send"),
                ("/", "palette"),
                ("⇥", "complete"),
                ("^C", "stop"),
                ("^C^C", "quit"),
                ("F1", "help"),
            )
            line = Text()
            for index, (key, action) in enumerate(items):
                if index:
                    line.append("  |  ", style=faint)
                line.append(key, style=f"bold {soft}")
                line.append(":", style=faint)
                line.append(action, style=faint)
            return line

        def compose(self) -> ComposeResult:
            with Horizontal(id="input-shell"):
                # ASCII '>' on the same row as typed text (Input renders on y=0).
                yield Static(">", id="prompt-glyph", markup=False, shrink=False)
                yield Input(placeholder="", id="prompt-input")
                if self._meta_text:
                    yield Static(self._meta_text, id="composer-meta")
            # Shortcuts left, context usage right — outside the chatbox shell.
            with Horizontal(id="composer-footer"):
                yield Static(self._shortcut_strip(), id="composer-hint")
                yield Static("0 / 0", id="composer-context")

        def on_mount(self) -> None:
            # Keep the leading prompt arrow on the input text row.
            try:
                glyph = self.query_one("#prompt-glyph", Static)
                glyph.update(">")
            except Exception:
                pass
            # Rebuild shortcut colors after the app theme is applied.
            try:
                self.query_one("#composer-hint", Static).update(self._shortcut_strip())
            except Exception:
                pass
            try:
                app = self.app
                if hasattr(app, "refresh_context_usage"):
                    app.refresh_context_usage()
            except Exception:
                pass

    class StatusBar(Static):
        DEFAULT_CSS = """
        StatusBar {
            dock: top;
            height: 1;
            background: $background;
            color: $primary 40%;
            padding: 0 2;
            text-style: dim;
        }
        """

    class SeleneTui(App[None]):
        """Full-screen Selene agent interface."""

        TITLE = "Selene"
        CSS = """
        Screen {
            background: $background;
            color: $foreground;
        }
        #body {
            height: 1fr;
        }
        #welcome {
            margin: 1 2;
            color: $secondary;
            text-align: center;
        }
        """

        BINDINGS = [
            Binding("ctrl+c", "interrupt_or_quit", "Stop / quit", show=True, priority=True),
            Binding("ctrl+d", "quit_app", "Quit", show=False, priority=True),
            Binding("ctrl+l", "clear_chat", "Clear chat", show=True),
            Binding("ctrl+k", "clear_input", "Clear input", show=True),
            Binding("ctrl+slash", "open_commands", "Commands", show=True),
            Binding("f1", "show_help", "Help", show=True),
            Binding("ctrl+j", "submit_input", "Send", show=False, priority=True),
            Binding("escape", "blur_or_clear", "Esc", show=False, priority=True),
        ]

        # Second Ctrl+C within this window exits after a stop (or idle arm).
        _QUIT_ARM_SECONDS = 2.0

        def __init__(
            self,
            *,
            session: dict,
            history: list[dict],
            default_system_prompt: str | None,
            process_turn: Callable[..., None],
            handle_command: Callable[..., bool | None],
            slash_completions: Sequence[str],
            slash_descriptions: dict[str, str],
            status_meta: dict[str, str] | None = None,
        ) -> None:
            super().__init__()
            self.session = session
            self.history = history
            self.default_system_prompt = default_system_prompt
            self._process_turn = process_turn
            self._handle_command = handle_command
            self.slash_completions = tuple(slash_completions)
            self.slash_descriptions = dict(slash_descriptions)
            self.status_meta = dict(status_meta or {})
            self._busy = False
            self._busy_lock = threading.Lock()
            self._quit_armed_until = 0.0
            self._selene_palette = None  # filled on mount via ui_apply_theme
            self._stream_widget: MessageBlock | None = None
            self._thinking_widget: MessageBlock | None = None
            self._thinking_buf = ""
            self._thinking_tokens = 0
            self._activity_widget: MessageBlock | None = None

            self._activity_label = "Thinking"
            self._activity_frame = 0
            self._activity_timer = None
            self._activity_phase = "idle"  # idle | waiting | thinking
            self._slash_matches: list[tuple[str, str]] = []
            self._slash_selected = 0
            self._sink: TuiDisplaySink | None = None
            self._saved_console = None
            self._capture_file = None
            # Braille spinner frames — light, terminal-native motion.
            self._spinner_frames = tuple("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

        def compose(self) -> ComposeResult:
            meta = self._format_meta()
            yield StatusBar(f"{GLYPH_MARK}  selene  {GLYPH_DOT}  {meta}")
            with Vertical(id="body"):
                yield ChatView(id="chat")
                yield SlashPalette(id="slash-palette")
                yield Composer(meta_text=self._composer_meta())

        def on_mount(self) -> None:
            from agent.tui_themes import DEFAULT_THEME, normalize_theme_name, register_all_themes

            self._sink = TuiDisplaySink(self)
            set_display_sink(self._sink)
            # Redirect classic Rich prints into the transcript.
            import agent.terminal as terminal_mod

            self._saved_console = terminal_mod._console
            self._capture_file = _CaptureFile(self._sink)
            terminal_mod._console = Console(
                file=self._capture_file,
                force_terminal=False,
                no_color=True,
                width=100,
                soft_wrap=True,
            )

            register_all_themes(self)
            initial_theme = normalize_theme_name(
                (self.session or {}).get("tui_theme") or DEFAULT_THEME
            )
            self._apply_selene_theme(initial_theme, announce=False)
            # Keep the chatbox prompt arrow visible after theme CSS settles.
            try:
                self.query_one("#prompt-glyph", Static).update(">")
            except Exception:
                pass

            chat = self.query_one("#chat", ChatView)
            welcome = self._welcome_renderable()
            chat.mount(Static(welcome, id="welcome"))
            self.query_one("#prompt-input", Input).focus()
            self.refresh_context_usage()

        def on_unmount(self) -> None:
            set_display_sink(None)
            if self._saved_console is not None:
                import agent.terminal as terminal_mod

                terminal_mod._console = self._saved_console

        def _format_meta(self) -> str:
            parts = []
            for key in ("profile", "model", "ctx", "host"):
                value = self.status_meta.get(key)
                if value:
                    parts.append(f"{key} {value}")
            return f" {GLYPH_DOT} ".join(parts) if parts else "local agent runtime"

        def _composer_meta(self) -> str:
            """Short right-side chip inside the input shell (model · profile)."""
            bits = []
            model = self.status_meta.get("model")
            profile = self.status_meta.get("profile")
            if model:
                bits.append(str(model))
            if profile:
                bits.append(str(profile))
            return f" {GLYPH_DOT} ".join(bits)

        def _context_budget(self) -> int:
            """Effective num_ctx for the active session."""
            try:
                from agent.core import effective_session_model_options

                runtime, _ = effective_session_model_options(self.session or {})
                return max(1, int(runtime.num_ctx))
            except Exception:
                try:
                    return max(1, int(self.status_meta.get("ctx") or 8192))
                except Exception:
                    return 8192

        def _estimate_context_used(self) -> int:
            """Match web UI / core heuristics: history + draft input tokens."""
            from agent.core import _estimate_message_tokens, _estimate_messages_tokens

            history = list(self.history or [])
            used = 0
            try:
                used = int(_estimate_messages_tokens(history))
            except Exception:
                used = 0

            # Draft text in the composer (mirrors web estimatedContextTokens).
            try:
                draft = self.query_one("#prompt-input", Input).value or ""
            except Exception:
                draft = ""
            if draft.strip():
                try:
                    used += int(
                        _estimate_message_tokens({"role": "user", "content": draft})
                    )
                except Exception:
                    used += max(1, len(draft) // 4)

            return max(0, used)

        def refresh_context_usage(self) -> None:
            """Update bottom-right ``used / budget`` context label (no bar)."""
            try:
                label = self.query_one("#composer-context", Static)
            except Exception:
                return
            used = self._estimate_context_used()
            budget = self._context_budget()
            from agent.tui_themes import DEFAULT_THEME, rich_palette

            pal = self._selene_palette or rich_palette(DEFAULT_THEME)
            pct = (used / budget) if budget else 0.0
            if pct >= 0.90:
                color = pal["error"]
            elif pct >= 0.75:
                color = pal["warning"]
            else:
                color = pal["faint"]
            text = Text()
            text.append(f"{used} / {budget}", style=color)
            label.update(text)

        def _pal(self) -> dict[str, str]:
            from agent.tui_themes import DEFAULT_THEME, rich_palette

            return self._selene_palette or rich_palette(DEFAULT_THEME)

        def _apply_selene_theme(self, name: str, *, announce: bool = True) -> None:
            from agent.tui_themes import (
                normalize_theme_name,
                rich_palette,
                textual_theme_name,
                theme_label,
            )

            key = normalize_theme_name(name)
            self._selene_palette = rich_palette(key)
            try:
                # Textual registers place themes as Title Case (Oslo, Tokyo, …).
                self.theme = textual_theme_name(key)
            except Exception:
                pass
            if isinstance(self.session, dict):
                self.session["tui_theme"] = key
            # Refresh shortcut strip colors for the active palette.
            try:
                composer = self.query_one(Composer)
                hint = self.query_one("#composer-hint", Static)
                hint.update(composer._shortcut_strip())
            except Exception:
                pass
            if announce:
                self.ui_status(
                    f"Theme · {textual_theme_name(key)}",
                    kind="ok",
                    detail=theme_label(key),
                )

        def ui_apply_theme(self, name: str) -> None:
            self._apply_selene_theme(name, announce=True)
            self.refresh_context_usage()

        def watch_theme(self, theme: str) -> None:
            """Keep Selene palette/session in sync when Ctrl+P picks a place theme."""
            if not theme:
                return
            from agent.tui_themes import (
                normalize_theme_name,
                place_theme_display_names,
                rich_palette,
                textual_theme_name,
            )

            # Ignore non-place names (should be purged, but be defensive).
            if theme not in place_theme_display_names():
                return
            key = normalize_theme_name(theme)
            # Avoid re-entry loops when we set theme ourselves.
            if textual_theme_name(key) != theme:
                return
            self._selene_palette = rich_palette(key)
            if isinstance(self.session, dict):
                self.session["tui_theme"] = key
            try:
                self.query_one("#prompt-glyph", Static).update(">")
            except Exception:
                pass
            try:
                composer = self.query_one(Composer)
                hint = self.query_one("#composer-hint", Static)
                hint.update(composer._shortcut_strip())
            except Exception:
                pass
            try:
                self.refresh_context_usage()
            except Exception:
                pass

        def _welcome_renderable(self):
            pal = self._pal()
            brand = Text()
            brand.append(f"{GLYPH_MARK}  ", style=f"bold {pal['text_soft']}")
            brand.append("SELENE\n", style=f"bold {pal['text']}")
            brand.append("local agent runtime\n", style=pal["muted"])
            brand.append(
                f"tools {GLYPH_DOT} vault {GLYPH_DOT} ollama\n\n",
                style=pal["faint"],
            )
            brand.append("Type a message below", style=pal["muted"])
            brand.append("  ·  ", style=pal["faint"])
            brand.append("/", style=f"bold {pal['text_soft']}")
            brand.append(" commands  ·  ", style=pal["muted"])
            brand.append("/theme", style=f"bold {pal['text_soft']}")
            brand.append(" colors  ·  ", style=pal["muted"])
            brand.append("F1", style=f"bold {pal['text_soft']}")
            brand.append(" help", style=pal["muted"])
            from agent.tui_themes import DEFAULT_THEME, theme_label

            theme_name = DEFAULT_THEME
            if isinstance(self.session, dict):
                theme_name = str(self.session.get("tui_theme") or DEFAULT_THEME)

            return Panel(
                brand,
                border_style=pal["border"],
                title=f"[bold {pal['text_soft']}]{GLYPH_MARK}[/] [bold {pal['text']}]selene[/]",
                subtitle=f"[dim]{theme_label(theme_name)}[/]",
                padding=(1, 2),
            )

        # ── Transcript helpers ────────────────────────────────────────

        def _chat(self) -> ChatView:
            return self.query_one("#chat", ChatView)

        def _mount_block(self, renderable, *classes: str) -> MessageBlock:
            block = MessageBlock(renderable)
            for cls in classes:
                block.add_class(cls)
            chat = self._chat()
            chat.mount(block)
            chat.scroll_end(animate=False)
            return block

        def ui_add_user(self, text: str) -> None:
            # Slash commands are chrome; real prompts are primary content.
            if str(text).startswith("/"):
                line = Text()
                line.append(f"{GLYPH_PROMPT} ", style="#555555")
                line.append("command  ", style="bold #6b6b6b")
                line.append(text, style="#7a7a7a")
                self._mount_block(line, "command")
                return
            header = Text()
            header.append(f"{GLYPH_MARK} ", style="bold #f2f2f2")
            header.append("you\n", style="bold #ffffff")
            header.append(text, style="#f2f2f2")
            self._mount_block(header, "user")

        # ── Activity animation (pre-thinking / in-progress) ───────────

        def _activity_renderable(self) -> Text:
            frame = self._spinner_frames[
                self._activity_frame % len(self._spinner_frames)
            ]
            label = (self._activity_label or "Thinking").strip() or "Thinking"
            # Soft pulse dots after the label.
            dots = "." * (1 + (self._activity_frame % 3))
            line = Text()
            line.append(f"{frame}  ", style="#6b6b6b")
            line.append(label, style="#6b6b6b")
            line.append(dots.ljust(3), style="#555555")
            return line

        def _thinking_renderable(self) -> Text:
            frame = self._spinner_frames[
                self._activity_frame % len(self._spinner_frames)
            ]
            body = Text()
            body.append(f"{frame}  ", style="#6b6b6b")
            body.append("thinking", style="#6b6b6b")
            if self._thinking_tokens:
                label = "token" if self._thinking_tokens == 1 else "tokens"
                body.append(
                    f"  ·  ~{self._thinking_tokens} {label}",
                    style="#555555",
                )
            else:
                body.append("  ·  collecting", style="#555555")
            preview = self._thinking_buf.strip().replace("\n", " ")
            if preview:
                if len(preview) > 72:
                    preview = "…" + preview[-72:]
                body.append("\n  ", style="")
                body.append(preview, style="#555555")
            return body

        def _ensure_activity_widget(self) -> "MessageBlock":
            if self._activity_widget is not None:
                return self._activity_widget
            block = self._mount_block(self._activity_renderable(), "activity")
            self._activity_widget = block
            return block

        def _remove_activity_widget(self) -> None:
            if self._activity_widget is None:
                return
            try:
                self._activity_widget.remove()
            except Exception:
                pass
            self._activity_widget = None

        def _tick_activity(self) -> None:
            if self._activity_phase in {"idle", "paused"}:
                return
            self._activity_frame = (self._activity_frame + 1) % len(self._spinner_frames)
            try:
                if self._activity_phase == "thinking" and self._thinking_widget is not None:
                    self._thinking_widget.update(self._thinking_renderable())
                elif self._activity_widget is not None:
                    self._activity_widget.update(self._activity_renderable())
            except Exception:
                pass

        def _start_activity_timer(self) -> None:
            if self._activity_timer is not None:
                return
            try:
                self._activity_timer = self.set_interval(0.08, self._tick_activity)
            except Exception:
                self._activity_timer = None

        def _stop_activity_timer(self) -> None:
            timer = self._activity_timer
            self._activity_timer = None
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass

        def ui_activity_start(self, label: str = "Thinking") -> None:
            """Show a single animated waiting line (reused, never stacked)."""
            if self._activity_phase == "thinking":
                # Thinking block owns the slot — leave it alone.
                return
            self._activity_label = (label or "Thinking").strip() or "Thinking"
            self._activity_phase = "waiting"
            if self._activity_widget is None:
                self._activity_frame = 0
            widget = self._ensure_activity_widget()
            try:
                widget.remove_class("thinking")
            except Exception:
                pass
            widget.add_class("activity")
            widget.update(self._activity_renderable())
            self._start_activity_timer()
            self._chat().scroll_end(animate=False)

        def ui_activity_update(self, label: str) -> None:
            if self._activity_phase in {"idle", "paused"}:
                self.ui_activity_start(label)
                return
            if self._activity_phase == "thinking":
                return
            self._activity_label = (label or self._activity_label).strip()
            if self._activity_widget is not None:
                self._activity_widget.update(self._activity_renderable())

        def ui_activity_stop(self) -> None:
            """Pause/remove the waiting spinner.

            While a turn is busy we *keep* the widget so thinking can promote it
            (core stops the spinner immediately before the thinking header).
            """
            if self._activity_phase == "thinking":
                return
            self._stop_activity_timer()
            if self._busy and self._activity_widget is not None:
                # Freeze on the last frame until thinking/content claims it.
                self._activity_phase = "paused"
                return
            self._activity_phase = "idle"
            self._remove_activity_widget()

        def ui_status(self, message: str, kind: str = "info", detail: str | None = None) -> None:
            # While generating, keep chat clean — fold run-like noise into activity.
            if self._busy and kind in {"run", "info"}:
                label = message if not detail else f"{message} · {detail}"
                # Skip pure chrome / duplicate thinking notices.
                low = message.casefold()
                if any(skip in low for skip in ("thinking", "reasoning", "…", "...")):
                    self.ui_activity_start(message.split("·")[0].strip() or "Thinking")
                    return
                if kind == "run":
                    self.ui_activity_start(label)
                    return

            glyphs = {
                "info": (GLYPH_MARK, "#6b6b6b"),
                "run": (GLYPH_RUN, "#7a7a60"),
                "ok": (GLYPH_OK, "#6a8a6a"),
                "warn": (GLYPH_WARN, "#8a8a60"),
                "error": (GLYPH_ERR, "#c08080"),
                "tool": (GLYPH_TOOL, "#6b6b6b"),
            }
            glyph, color = glyphs.get(kind, glyphs["info"])
            line = Text()
            line.append(f"{glyph}  ", style=color)
            line.append(message, style=color if kind == "error" else "#6b6b6b")
            if detail:
                line.append(f"  {detail}", style="#555555")
            cls = "error" if kind == "error" else "status"
            self._mount_block(line, cls)

        def ui_console_line(self, text: str) -> None:
            # Suppress Rich chrome noise, especially while a turn is active.
            stripped = text.strip()
            if not stripped:
                return
            if set(stripped) <= set("─━╭╮╯╰│|+-═║╔╗╝╚▀▄█ "):
                return
            if self._busy:
                # Drop duplicate spinner/status echoes captured from Rich.
                low = stripped.casefold()
                if any(
                    token in low
                    for token in (
                        "thinking",
                        "reasoning",
                        "selene",
                        "response",
                        "frontier",
                    )
                ):
                    return
                if len(stripped) < 3:
                    return
            self._mount_block(Text(stripped, style="#555555"), "system")

        def ui_thinking_start(self) -> None:
            """Promote the activity line into a compact thinking block."""
            self._thinking_buf = ""
            self._thinking_tokens = 0
            self._activity_label = "thinking"
            self._activity_phase = "thinking"

            body = self._thinking_renderable()

            if self._activity_widget is not None:
                self._thinking_widget = self._activity_widget
                try:
                    self._thinking_widget.remove_class("activity")
                except Exception:
                    pass
                self._thinking_widget.add_class("thinking")
                self._thinking_widget.update(body)
                self._activity_widget = None
            elif self._thinking_widget is None:
                self._thinking_widget = self._mount_block(body, "thinking")
            else:
                self._thinking_widget.update(body)

            self._start_activity_timer()
            self._chat().scroll_end(animate=False)

        def ui_thinking_delta(self, text: str) -> None:
            if not text:
                return
            if self._thinking_widget is None:
                self.ui_thinking_start()
            self._thinking_buf += text
            self._thinking_tokens = _estimate_tokens(self._thinking_buf)
            if self._thinking_widget is not None:
                self._thinking_widget.update(self._thinking_renderable())
                # Throttle scroll to avoid jitter every stream chunk.
                if self._thinking_tokens % 32 < max(1, _estimate_tokens(text)):
                    self._chat().scroll_end(animate=False)

        def ui_thinking_end(self, label: str | None = None) -> None:
            """Finish live thinking and replace it with a collapsible fold."""
            self._stop_activity_timer()
            self._activity_phase = "idle"
            self._activity_widget = None

            full_text = self._thinking_buf
            tokens = self._thinking_tokens
            title = (label or "thinking").strip() or "thinking"
            interrupted = label == "interrupted"
            old = self._thinking_widget
            self._thinking_widget = None
            self._thinking_buf = ""
            self._thinking_tokens = 0

            fold = ThinkingFold(
                full_text,
                tokens,
                title=title if not interrupted else "thinking interrupted",
                interrupted=interrupted,
            )

            chat = self._chat()
            try:
                if old is not None and old.parent is not None:
                    # Keep position above the (upcoming) response stream.
                    chat.mount(fold, after=old)
                    old.remove()
                else:
                    chat.mount(fold)
            except Exception:
                try:
                    chat.mount(fold)
                except Exception:
                    return
            chat.scroll_end(animate=False)

        def _clear_waiting_activity(self) -> None:
            """Drop a paused/waiting spinner once real output starts."""
            if self._activity_phase in {"waiting", "paused"}:
                self._stop_activity_timer()
                self._activity_phase = "idle"
                self._remove_activity_widget()

        def ui_content_stream(self, text: str) -> None:
            self._clear_waiting_activity()
            rendered = Markdown(_render_terminal_markdown(text or " "))
            title = Text()
            title.append(f"{GLYPH_SECTION} ", style="bold #ffffff")
            title.append("response\n", style="bold #f2f2f2")
            panel = Group(title, rendered)
            if self._stream_widget is None:
                self._stream_widget = self._mount_block(panel, "assistant")
            else:
                self._stream_widget.update(panel)
            self._chat().scroll_end(animate=False)

        def ui_content_final(self, text: str) -> None:
            self._clear_waiting_activity()
            # Ensure thinking row is collapsed if footer was skipped.
            if self._thinking_widget is not None:
                self.ui_thinking_end()
            rendered = Markdown(_render_terminal_markdown(text or " "))
            title = Text()
            title.append(f"{GLYPH_SECTION} ", style="bold #ffffff")
            title.append("response\n", style="bold #f2f2f2")
            panel = Group(title, rendered)
            if self._stream_widget is None:
                self._mount_block(panel, "assistant")
            else:
                self._stream_widget.update(panel)
            self._stream_widget = None
            self._chat().scroll_end(animate=False)

        def ui_stats(self, elapsed: float, total_tokens: int, tokens_per_sec: float) -> None:
            line = (
                f"{GLYPH_DOT}  {elapsed:.1f}s  {GLYPH_DOT}  "
                f"~{total_tokens} tokens  {GLYPH_DOT}  ~{tokens_per_sec:.1f} tok/s"
            )
            self._mount_block(Text(line, style="#555555"), "status")

        def ui_help(
            self,
            entries: list[tuple[str, str]],
            title: str,
            subtitle: str | None,
        ) -> None:
            body = Text()
            body.append(f"{GLYPH_SECTION} ", style="#6b6b6b")
            body.append(f"{title}\n", style="bold #7a7a7a")
            if subtitle:
                body.append(f"{subtitle}\n\n", style="#555555")
            else:
                body.append("\n")
            for command, description in entries:
                body.append(f"  {command}", style="#7a7a7a")
                body.append(f"  {description}\n", style="#555555")
            self._mount_block(body, "system")

        # ── Slash palette ─────────────────────────────────────────────

        def _slash_query(self) -> str:
            try:
                return self.query_one("#prompt-input", Input).value or ""
            except Exception:
                return ""

        def _refresh_slash_view(self) -> None:
            palette = self.query_one("#slash-palette", SlashPalette)
            if not self._slash_matches:
                palette.hide_palette()
                return
            self._slash_selected = max(
                0, min(self._slash_selected, len(self._slash_matches) - 1)
            )
            palette.show_matches(
                self._slash_matches,
                self._slash_selected,
                query=self._slash_query() or "/",
                total=len(self.slash_completions),
            )

        def _update_slash_palette(self, value: str, *, reset_selection: bool = True) -> None:
            palette = self.query_one("#slash-palette", SlashPalette)
            if not value.startswith("/") or "\n" in value:
                self._slash_matches = []
                palette.hide_palette()
                return
            matches = _filter_slash_commands(
                value,
                self.slash_completions,
                self.slash_descriptions,
                limit=14,
            )
            self._slash_matches = matches
            if not self._slash_matches:
                palette.hide_palette()
                return
            if reset_selection:
                self._slash_selected = 0
            self._refresh_slash_view()

        def _move_slash(self, delta: int) -> None:
            if not self._slash_matches:
                return
            self._slash_selected = (self._slash_selected + int(delta)) % len(
                self._slash_matches
            )
            self._refresh_slash_view()

        def _accept_slash_selection(self, *, run: bool = False) -> None:
            """Fill the input with the highlighted command; optionally submit."""
            if not self._slash_matches:
                return
            preferred = self._slash_matches[self._slash_selected][0]
            inp = self.query_one("#prompt-input", Input)
            inp.value = preferred
            inp.cursor_position = len(preferred)
            self._update_slash_palette(preferred, reset_selection=False)
            if run:
                self._submit_from_input()

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id != "prompt-input":
                return
            self._update_slash_palette(event.value, reset_selection=True)
            self.refresh_context_usage()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id != "prompt-input":
                return
            self._submit_from_input()

        def _submit_from_input(self) -> None:
            inp = self.query_one("#prompt-input", Input)
            value = (inp.value or "").strip()
            # Accept highlighted slash command when the buffer is a prefix/filter.
            if (
                value.startswith("/")
                and self._slash_matches
                and 0 <= self._slash_selected < len(self._slash_matches)
            ):
                preferred = self._slash_matches[self._slash_selected][0]
                # Run the highlight if the typed text is an incomplete filter of it,
                # or if the user only typed "/".
                if (
                    preferred.casefold().startswith(value.casefold())
                    or value == "/"
                    or value.casefold() in preferred.casefold()
                ):
                    value = preferred
            inp.value = ""
            self.query_one("#slash-palette", SlashPalette).hide_palette()
            self._slash_matches = []
            if value:
                self._submit(value)

        def on_key(self, event) -> None:
            # Intercept palette navigation while the input is focused.
            try:
                focused = self.focused
            except Exception:
                focused = None
            if focused is None or getattr(focused, "id", None) != "prompt-input":
                return

            key = event.key
            # Normalize a few aliases Textual may emit.
            if key in ("ctrl+n", "down"):
                if self._slash_matches or (self._slash_query().startswith("/")):
                    if not self._slash_matches:
                        self._update_slash_palette(self._slash_query())
                    if self._slash_matches:
                        event.prevent_default()
                        event.stop()
                        self._move_slash(1)
                return
            if key in ("ctrl+p", "up"):
                if self._slash_matches or (self._slash_query().startswith("/")):
                    if not self._slash_matches:
                        self._update_slash_palette(self._slash_query())
                    if self._slash_matches:
                        event.prevent_default()
                        event.stop()
                        self._move_slash(-1)
                return
            if key in ("pageup",):
                if self._slash_matches:
                    event.prevent_default()
                    event.stop()
                    self._move_slash(-5)
                return
            if key in ("pagedown",):
                if self._slash_matches:
                    event.prevent_default()
                    event.stop()
                    self._move_slash(5)
                return
            if key in ("home",) and self._slash_matches:
                event.prevent_default()
                event.stop()
                self._slash_selected = 0
                self._refresh_slash_view()
                return
            if key in ("end",) and self._slash_matches:
                event.prevent_default()
                event.stop()
                self._slash_selected = len(self._slash_matches) - 1
                self._refresh_slash_view()
                return
            if key == "tab":
                inp = self.query_one("#prompt-input", Input)
                value = inp.value or ""
                if value.startswith("/") or self._slash_matches:
                    event.prevent_default()
                    event.stop()
                    if self._slash_matches:
                        self._accept_slash_selection(run=False)
                    else:
                        self._tab_complete(inp)
                return
            if key in ("right",) and self._slash_matches:
                # Right arrow accepts the highlight when the caret is at EOL.
                inp = self.query_one("#prompt-input", Input)
                if inp.cursor_position >= len(inp.value or ""):
                    event.prevent_default()
                    event.stop()
                    self._accept_slash_selection(run=False)

        def _tab_complete(self, inp: Input) -> None:
            value = inp.value or ""
            if not value.startswith("/"):
                value = "/" + value.lstrip()
            matches = _filter_slash_commands(
                value,
                self.slash_completions,
                self.slash_descriptions,
                limit=20,
            )
            if not matches:
                return
            cmds = [cmd for cmd, _ in matches]
            # Prefer children when the typed token is an exact parent.
            folded = value.casefold()
            children = [cmd for cmd in cmds if cmd.casefold().startswith(f"{folded} ")]
            pool = children or cmds
            if len(pool) == 1:
                completed = pool[0]
            else:
                from os.path import commonprefix

                shared = commonprefix(pool)
                completed = shared if len(shared) > len(value) else pool[0]
            inp.value = completed
            inp.cursor_position = len(completed)
            self._update_slash_palette(completed, reset_selection=True)

        # ── Turn execution ────────────────────────────────────────────

        def _submit(self, user_input: str) -> None:
            with self._busy_lock:
                if self._busy:
                    self.ui_status("Still generating — wait for the current turn", kind="warn")
                    return
                self._busy = True

            self.ui_add_user(user_input)
            self._stream_widget = None
            self._thinking_widget = None
            self._thinking_buf = ""
            self._thinking_tokens = 0
            # Fresh activity line for this turn (Thinking animation).
            self._stop_activity_timer()
            self._remove_activity_widget()
            self._activity_phase = "idle"
            self.query_one("#prompt-input", Input).disabled = True

            def work() -> None:
                try:
                    if user_input.startswith("/"):
                        result = self._handle_command(
                            user_input, self.session, self.history
                        )
                        if result is None:
                            self.call_from_thread(self.exit)
                            return
                        base = user_input.strip().split(None, 1)[0].lower()
                        if base == "/clear":
                            # Reset UI after core clears history; show one confirmation.
                            self.call_from_thread(self._clear_conversation_ui)
                    else:
                        self._process_turn(
                            user_input,
                            self.session,
                            self.history,
                            self.default_system_prompt,
                        )
                except Exception as exc:
                    self.call_from_thread(
                        self.ui_status, f"Turn failed · {exc}", "error", None
                    )
                finally:
                    self.call_from_thread(self._turn_finished)

            threading.Thread(target=work, name="selene-turn", daemon=True).start()

        def _turn_finished(self) -> None:
            # Collapse any leftover spinner / open thinking row.
            self._stop_activity_timer()
            if self._thinking_widget is not None:
                self.ui_thinking_end()
            self._clear_waiting_activity()
            with self._busy_lock:
                self._busy = False
            inp = self.query_one("#prompt-input", Input)
            inp.disabled = False
            inp.focus()
            # Flush any remaining capture buffer.
            if self._capture_file is not None:
                self._capture_file.flush()
            self.refresh_context_usage()

        def _reset_transcript(self) -> None:
            """Clear the chat view after /clear (history already wiped by handler).

            Textual's ``remove_children`` / ``remove`` are asynchronous. Remounting
            a fixed ``id="welcome"`` immediately races the prune and raises
            ``DuplicateIds``. Keep/update the welcome widget and only prune the
            rest of the transcript.
            """
            self._stop_activity_timer()
            self._activity_phase = "idle"
            self._activity_widget = None
            self._thinking_widget = None
            self._thinking_buf = ""
            self._thinking_tokens = 0
            self._stream_widget = None

            chat = self._chat()
            welcome = None
            stale: list = []
            for child in list(chat.children):
                if getattr(child, "id", None) == "welcome":
                    welcome = child
                else:
                    stale.append(child)
            for child in stale:
                try:
                    child.remove()
                except Exception:
                    pass

            renderable = self._welcome_renderable()
            if welcome is not None:
                try:
                    welcome.update(renderable)
                    return
                except Exception:
                    try:
                        welcome.remove()
                    except Exception:
                        pass

            # No welcome yet (or update failed) — mount without racing a twin id.
            try:
                existing = chat.query("#welcome")
                if existing:
                    existing.first().update(renderable)
                    return
            except Exception:
                pass
            chat.mount(Static(renderable, id="welcome"))

        def _clear_conversation_ui(self) -> None:
            """Reset transcript and confirm (used by /clear and Ctrl+L)."""
            self._reset_transcript()
            self.ui_status("Conversation cleared", kind="ok")
            self.refresh_context_usage()

        def action_quit_app(self) -> None:
            self.exit()

        def action_interrupt_or_quit(self) -> None:
            """Ctrl+C: stop generation first; press again to exit.

            - While a turn is running: request cooperative stream cancel.
            - Second Ctrl+C within a short window (or when idle after arming): quit.
            """
            now = time.monotonic()
            with self._busy_lock:
                busy = self._busy

            if busy:
                try:
                    from agent.core import request_generation_interrupt

                    request_generation_interrupt()
                except Exception:
                    pass
                self._quit_armed_until = now + self._QUIT_ARM_SECONDS
                self.ui_status(
                    "Generation stopping · press Ctrl+C again to quit",
                    kind="warn",
                )
                return

            if now <= float(self._quit_armed_until or 0):
                self.exit()
                return

            self._quit_armed_until = now + self._QUIT_ARM_SECONDS
            self.ui_status("Press Ctrl+C again to quit", kind="info")

        def action_clear_chat(self) -> None:
            if self._busy:
                return
            self.history.clear()
            self.session["system"] = ""
            self._clear_conversation_ui()

        def action_clear_input(self) -> None:
            """Ctrl+K — clear the composer (and dismiss the palette)."""
            inp = self.query_one("#prompt-input", Input)
            inp.value = ""
            inp.focus()
            self._slash_matches = []
            self.query_one("#slash-palette", SlashPalette).hide_palette()

        def action_open_commands(self) -> None:
            """Ctrl+/ — open the slash command palette."""
            inp = self.query_one("#prompt-input", Input)
            if self._busy:
                return
            if not (inp.value or "").startswith("/"):
                inp.value = "/"
            inp.focus()
            inp.cursor_position = len(inp.value or "")
            self._update_slash_palette(inp.value or "/", reset_selection=True)

        def action_show_help(self) -> None:
            """F1 — show command help in the transcript."""
            if self._busy:
                return
            entries = [
                (cmd, self.slash_descriptions.get(cmd, ""))
                for cmd in self.slash_completions
                if cmd not in ("/?", "/q", "/exit")
            ]
            # De-dupe while preserving order.
            seen: set[str] = set()
            unique: list[tuple[str, str]] = []
            for cmd, desc in entries:
                if cmd in seen:
                    continue
                seen.add(cmd)
                unique.append((cmd, desc))
            self.ui_help(
                unique,
                "commands",
                "ctrl+/ palette  ·  tab complete  ·  enter run",
            )
            shortcuts = Text()
            shortcuts.append(f"{GLYPH_SECTION} ", style="#6b6b6b")
            shortcuts.append("shortcuts\n", style="bold #7a7a7a")
            for key, desc in (
                ("Enter / Ctrl+J", "Send message"),
                ("Ctrl+C", "Stop generation"),
                ("Ctrl+C twice", "Quit application"),
                ("Ctrl+/", "Open command palette"),
                ("Tab / →", "Autofill highlighted command"),
                ("↑↓  Ctrl+N/P", "Move palette selection"),
                ("PgUp / PgDn", "Jump palette selection"),
                ("Esc", "Dismiss palette / clear input"),
                ("Ctrl+K", "Clear input"),
                ("Ctrl+L", "Clear conversation"),
                ("F1", "Show this help"),
            ):
                shortcuts.append(f"  {key:<16}", style="#7a7a7a")
                shortcuts.append(f"  {desc}\n", style="#555555")
            self._mount_block(shortcuts, "system")

        def action_submit_input(self) -> None:
            """Ctrl+J — send the current composer text."""
            if self._busy:
                return
            self._submit_from_input()

        def action_blur_or_clear(self) -> None:
            """Esc — dismiss palette first, then clear the input."""
            palette = self.query_one("#slash-palette", SlashPalette)
            if self._slash_matches:
                self._slash_matches = []
                palette.hide_palette()
                return
            inp = self.query_one("#prompt-input", Input)
            if inp.value:
                inp.value = ""
                return
            # Nothing to clear — leave focus on the composer.

    return SeleneTui


def run_tui(
    *,
    session: dict,
    history: list[dict],
    default_system_prompt: str | None,
    process_turn: Callable[..., None],
    handle_command: Callable[..., bool | None],
    slash_completions: Sequence[str],
    slash_descriptions: dict[str, str],
    status_meta: dict[str, str] | None = None,
) -> None:
    """Launch the full-screen Selene TUI (blocks until exit)."""
    app_cls = build_app_class()
    app = app_cls(
        session=session,
        history=history,
        default_system_prompt=default_system_prompt,
        process_turn=process_turn,
        handle_command=handle_command,
        slash_completions=slash_completions,
        slash_descriptions=slash_descriptions,
        status_meta=status_meta,
    )
    app.run()
