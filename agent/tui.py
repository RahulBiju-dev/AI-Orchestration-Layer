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

    def toggle_speech(self, action: str = "toggle") -> None:
        """Bridge ``/speech`` from the core command handler into the TUI."""
        self._call("ui_toggle_speech", action)

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
                "↑↓ / ^N ^P  move  ·  tab complete  ·  enter run  ·  esc close",
                style=faint,
            )
            self.update(body)
            self.add_class("-visible")

        def hide_palette(self) -> None:
            self.remove_class("-visible")
            self.update("")

    class SessionsMenu(Static):
        """Saved-conversation picker above the composer (Ctrl+O)."""

        DEFAULT_CSS = """
        SessionsMenu {
            display: none;
            height: auto;
            max-height: 18;
            padding: 1 1;
            background: $boost;
            color: $secondary;
            border: round $primary 20%;
            margin: 0 1 1 1;
        }
        SessionsMenu.-visible {
            display: block;
        }
        """

        NEW_KEY = "__new__"

        def _palette(self) -> dict[str, str]:
            from agent.tui_themes import DEFAULT_THEME, rich_palette

            app = self.app
            return getattr(app, "_selene_palette", None) or rich_palette(DEFAULT_THEME)

        def show_rows(
            self,
            rows: list[tuple[str, str, str]],
            selected: int,
            *,
            query: str = "",
            total: int | None = None,
        ) -> None:
            """Render ``(key, title, detail)`` rows; key ``__new__`` is New Conversation."""
            if not rows:
                self.remove_class("-visible")
                self.update("")
                return

            pal = self._palette()
            muted = pal["muted"]
            faint = pal["faint"]
            select_fg = pal["select_fg"]
            select_bg = pal["select_bg"]
            soft = pal["text_soft"]

            title_width = min(40, max(len(title) for _, title, _ in rows))
            total = total if total is not None else len(rows)
            count = f"{len(rows)}" + (f"/{total}" if total > len(rows) else "")

            body = Text()
            body.append("conversations  ", style=f"bold {muted}")
            body.append(f"{count}", style=faint)
            if query:
                body.append(f"  ·  filter {query}", style=faint)
            body.append("\n")
            body.append("─" * min(60, title_width + 28), style=faint)
            body.append("\n")

            for index, (key, title, detail) in enumerate(rows):
                padded = title.ljust(title_width)
                desc = (detail or "").strip()
                if len(desc) > 36:
                    desc = desc[:35] + "…"
                if index == selected:
                    body.append(
                        f" ▐ {padded}  {desc} ",
                        style=f"bold {select_fg} on {select_bg}",
                    )
                else:
                    style = soft if key == self.NEW_KEY else muted
                    body.append(f"   {padded}", style=style)
                    body.append(f"  {desc}", style=faint)
                body.append("\n")

            body.append(
                "↑↓ / ^N ^P  move  ·  enter open  ·  esc close",
                style=faint,
            )
            self.update(body)
            self.add_class("-visible")

        def hide_menu(self) -> None:
            self.remove_class("-visible")
            self.update("")

    class PromptQueuePanel(Static):
        """Compact numbered previews of prompts waiting above the composer."""

        DEFAULT_CSS = """
        PromptQueuePanel {
            display: none;
            height: auto;
            max-height: 5;
            padding: 0 1;
            background: $boost;
            color: $secondary;
            border: round $primary 20%;
            /* Flush against the composer so the panel bottom meets the chatbox top. */
            margin: 0 1 0 1;
        }
        PromptQueuePanel.-visible {
            display: block;
        }
        """

        # Soft ellipsis character used when truncating long one-line previews.
        _ELLIPSIS = "…"
        _PREVIEW_MAX = 56

        def _palette(self) -> dict[str, str]:
            from agent.tui_themes import DEFAULT_THEME, rich_palette

            app = self.app
            return getattr(app, "_selene_palette", None) or rich_palette(DEFAULT_THEME)

        @classmethod
        def preview_line(cls, text: str, *, max_len: int | None = None) -> str:
            """Collapse whitespace and truncate to a single preview line."""
            flat = " ".join(str(text or "").split()).strip()
            limit = max_len if max_len is not None else cls._PREVIEW_MAX
            if limit < 1:
                return ""
            if len(flat) <= limit:
                return flat
            # Leave room for the ellipsis character.
            keep = max(1, limit - 1)
            return flat[:keep].rstrip() + cls._ELLIPSIS

        def show_queue(self, items: list[str]) -> None:
            if not items:
                self.hide_queue()
                return

            pal = self._palette()
            muted = pal["muted"]
            faint = pal["faint"]
            soft = pal["text_soft"]

            body = Text()
            for index, raw in enumerate(items, start=1):
                if index > 1:
                    body.append("\n")
                preview = self.preview_line(raw)
                body.append(f" {index}. ", style=f"bold {soft}")
                body.append(preview or "·", style=muted if preview else faint)
            self.update(body)
            self.add_class("-visible")

        def hide_queue(self) -> None:
            self.remove_class("-visible")
            self.update("")

    class SpeechMenu(Vertical):
        """Centered voice popup: animated mic + editable transcript.

        Visual states:
          • idle      — soft breathing mic, quiet wave dots
          • recording — solid pulse + VU-style wave bars
          • finishing/error — stable state with concise in-popup detail
        """

        DEFAULT_CSS = """
        SpeechMenu {
            display: none;
            dock: top;
            layer: overlay;
            width: 100%;
            height: 100%;
            align: center middle;
            background: $background 82%;
        }
        SpeechMenu.-visible {
            display: block;
        }
        #speech-card {
            width: 76;
            max-width: 96%;
            height: auto;
            background: $boost;
            border: round $primary;
            padding: 1 2 1 2;
        }
        SpeechMenu.-recording #speech-card {
            border: round $error;
        }
        #speech-title {
            height: 1;
            width: 100%;
            color: $primary 50%;
            text-style: dim;
            background: transparent;
            margin: 0 0 1 0;
        }
        #speech-row {
            height: 3;
            width: 100%;
            align: left middle;
        }
        #speech-mic {
            width: 3;
            height: 3;
            min-width: 3;
            content-align: center middle;
            color: $primary;
            text-style: bold;
            background: transparent;
            padding: 0;
        }
        SpeechMenu.-recording #speech-mic {
            color: $error;
            text-style: bold;
        }
        #speech-wave {
            width: 10;
            height: 3;
            min-width: 10;
            content-align: center middle;
            color: $primary 45%;
            text-style: dim;
            background: transparent;
            padding: 0 1 0 0;
        }
        SpeechMenu.-recording #speech-wave {
            color: $error;
            text-style: none;
        }
        #speech-input {
            width: 1fr;
            height: 3;
            background: transparent;
            color: $foreground;
            border: none;
            padding: 0;
            margin: 0;
        }
        #speech-input:focus {
            background: transparent;
            border: none;
        }
        #speech-input > .input--placeholder {
            color: $primary 40%;
        }
        #speech-hint {
            height: 1;
            width: 100%;
            color: $primary 40%;
            text-style: dim;
            background: transparent;
            padding: 0;
            margin: 1 0 0 0;
        }
        """

        # Idle: soft “breathing” mic. Recording: hard pulse.
        _IDLE_MIC = ("○", "◔", "◑", "◕", "◉", "◕", "◑", "◔")
        _REC_MIC = ("●", "◉", "◎", "◉")
        # Idle dots vs VU meter when live.
        _IDLE_WAVE = ("· · · ·", " · · · ", "· · · ·", " · · · ")
        _REC_WAVE = (
            "▁▂▃▄▅▄▃▂",
            "▂▃▄▅▆▅▄▃",
            "▃▄▅▆▇▆▅▄",
            "▄▅▆▇█▇▆▅",
            "▅▆▇█▇▆▅▄",
            "▆▇█▇▆▅▄▃",
            "▇█▇▆▅▄▃▂",
            "█▇▆▅▄▃▂▁",
            "▇▆▅▄▃▂▁▂",
            "▆▅▄▃▂▁▂▃",
            "▅▄▃▂▁▂▃▄",
            "▄▃▂▁▂▃▄▅",
        )

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self._anim_frame = 0
            self._anim_timer = None
            self._recording = False

        def compose(self) -> ComposeResult:
            with Vertical(id="speech-card"):
                yield Static("voice", id="speech-title", markup=False)
                with Horizontal(id="speech-row"):
                    yield Static(self._IDLE_MIC[0], id="speech-mic", markup=False)
                    yield Static(self._IDLE_WAVE[0], id="speech-wave", markup=False)
                    yield Input(
                        placeholder="Speak or type…",
                        id="speech-input",
                    )
                yield Static(
                    "enter start  ·  enter again send  ·  esc close",
                    id="speech-hint",
                    markup=False,
                )

        def on_mount(self) -> None:
            # Keep the timer alive; _tick_anim no-ops while hidden.
            self._anim_timer = self.set_interval(0.11, self._tick_anim)

        def on_unmount(self) -> None:
            timer = self._anim_timer
            self._anim_timer = None
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass

        def _tick_anim(self) -> None:
            if not self.has_class("-visible"):
                return
            self._anim_frame = (self._anim_frame + 1) % 64
            recording = self._recording
            mic_frames = self._REC_MIC if recording else self._IDLE_MIC
            wave_frames = self._REC_WAVE if recording else self._IDLE_WAVE
            try:
                self.query_one("#speech-mic", Static).update(
                    mic_frames[self._anim_frame % len(mic_frames)]
                )
            except Exception:
                pass
            try:
                self.query_one("#speech-wave", Static).update(
                    wave_frames[self._anim_frame % len(wave_frames)]
                )
            except Exception:
                pass

        def show_menu(self) -> None:
            self._recording = False
            self._anim_frame = 0
            self.remove_class("-recording")
            try:
                self.query_one("#speech-mic", Static).update(self._IDLE_MIC[0])
                self.query_one("#speech-wave", Static).update(self._IDLE_WAVE[0])
            except Exception:
                pass
            try:
                self.query_one("#speech-title", Static).update("voice")
            except Exception:
                pass
            try:
                self.query_one("#speech-hint", Static).update(
                    "enter start  ·  enter again send  ·  esc close"
                )
            except Exception:
                pass
            self.add_class("-visible")

        def hide_menu(self) -> None:
            self.remove_class("-visible")
            self.remove_class("-recording")
            self._recording = False
            self._anim_frame = 0
            try:
                self.query_one("#speech-mic", Static).update(self._IDLE_MIC[0])
                self.query_one("#speech-wave", Static).update(self._IDLE_WAVE[0])
            except Exception:
                pass
            try:
                self.query_one("#speech-input", Input).value = ""
            except Exception:
                pass

        def set_recording(self, active: bool) -> None:
            active = bool(active)
            self._recording = active
            self.set_class(active, "-recording")
            try:
                title = "voice  ·  listening" if active else "voice"
                self.query_one("#speech-title", Static).update(title)
            except Exception:
                pass
            try:
                if active:
                    hint = "enter send  ·  esc cancel"
                else:
                    hint = "enter start  ·  enter again send  ·  esc close"
                self.query_one("#speech-hint", Static).update(hint)
            except Exception:
                pass
            # Snap animation frame so the state change is immediate.
            self._tick_anim()

        def set_finishing(self) -> None:
            self.set_recording(False)
            try:
                self.query_one("#speech-title", Static).update("voice  ·  finishing")
                self.query_one("#speech-hint", Static).update(
                    "transcribing final phrase  ·  esc cancel"
                )
            except Exception:
                pass

        def set_error(self, message: str) -> None:
            self.set_recording(False)
            detail = " ".join(str(message or "Voice input unavailable").split())
            if len(detail) > 72:
                detail = detail[:69].rstrip() + "…"
            try:
                self.query_one("#speech-title", Static).update("voice  ·  unavailable")
                self.query_one("#speech-hint", Static).update(detail)
            except Exception:
                pass

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
                ("^O", "chats"),
                ("^S", "speech"),
                ("⇥", "complete"),
                ("^C", "stop"),
                ("^C^C", "quit"),
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
            layers: base overlay;
        }
        #body {
            height: 1fr;
            layer: base;
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
            Binding("ctrl+o", "open_sessions", "Conversations", show=True, priority=True),
            Binding("ctrl+s", "toggle_speech", "Speech", show=True, priority=True),
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
            # Prompts typed while a turn is running (FIFO, capped).
            self._prompt_queue: list[str] = []
            self._PROMPT_QUEUE_MAX = 3
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
            self._slash_open = False  # command palette visible (Ctrl+/ or typed "/")
            # Conversations menu (Ctrl+O): rows are (key, title, detail).
            self._sessions_open = False
            self._session_rows: list[tuple[str, str, str]] = []
            self._session_selected = 0
            self._session_catalog_total = 0
            self._voice = None  # lazy VoiceInputController
            self._voice_active = False
            self._speech_open = False  # centered speech popup visible
            self._speech_armed = False  # True after first Enter (start); second Enter sends
            self._speech_pending_submit = False
            self._voice_error_message = ""
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
                yield SessionsMenu(id="sessions-menu")
                yield PromptQueuePanel(id="prompt-queue")
                yield Composer(meta_text=self._composer_meta())
            yield SpeechMenu(id="speech-menu")

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
            self._stop_voice(silent=True, abort=True)
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

        def _active_system_prompt_for_meter(self) -> str:
            """System prompt counted by the visible context meter."""
            override = str((self.session or {}).get("system") or "").strip()
            if override:
                return override
            for message in self.history or []:
                if isinstance(message, dict) and message.get("role") == "system":
                    return str(message.get("content") or "").strip()
            return str(self.default_system_prompt or "").strip()

        def _estimate_context_used(self) -> int:
            """Match web UI / core heuristics: history + draft input tokens."""
            from agent.core import _estimate_message_tokens, _estimate_messages_tokens

            history = [
                message
                for message in list(self.history or [])
                if not (isinstance(message, dict) and message.get("role") == "system")
            ]
            used = 0
            try:
                used = int(_estimate_messages_tokens(history))
            except Exception:
                used = 0
            system_prompt = self._active_system_prompt_for_meter()
            if system_prompt:
                try:
                    used += int(_estimate_message_tokens({
                        "role": "system",
                        "content": system_prompt,
                    }))
                except Exception:
                    used += max(1, len(system_prompt) // 4)

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
            brand.append(" colors", style=pal["muted"])
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

        def _thinking_line_width(self) -> int:
            """Usable character width for a full thinking preview line."""
            width = 0
            try:
                if self._thinking_widget is not None:
                    width = int(self._thinking_widget.size.width or 0)
            except Exception:
                width = 0
            if width <= 0:
                try:
                    width = int(self.query_one("#chat").size.width or 0)
                except Exception:
                    width = 0
            if width <= 0:
                try:
                    width = int(self.size.width or 0)
                except Exception:
                    width = 80
            # MessageBlock.thinking has left padding/border; keep a small gutter.
            return max(24, width - 4)

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
                # Fill the full chat width (was hard-capped at 72 chars ≈ half line).
                budget = self._thinking_line_width()
                if len(preview) > budget:
                    preview = "…" + preview[-(budget - 1) :]
                body.append("\n", style="")
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

        def _slash_palette_visible(self) -> bool:
            if self._slash_open or self._slash_matches:
                return True
            try:
                return self.query_one("#slash-palette", SlashPalette).has_class("-visible")
            except Exception:
                return False

        def _dismiss_slash_palette(self, *, clear_slash_draft: bool = False) -> bool:
            """Hide the command palette if open. Returns True when something closed."""
            closed = bool(self._slash_open or self._slash_matches)
            self._slash_open = False
            self._slash_matches = []
            self._slash_selected = 0
            try:
                palette = self.query_one("#slash-palette", SlashPalette)
                if palette.has_class("-visible"):
                    closed = True
                palette.hide_palette()
            except Exception:
                pass
            if clear_slash_draft and closed:
                try:
                    inp = self.query_one("#prompt-input", Input)
                    value = inp.value or ""
                    # Clear command drafts opened via Ctrl+/ or typed filters.
                    if value.startswith("/"):
                        inp.value = ""
                except Exception:
                    pass
            return closed

        def _refresh_slash_view(self) -> None:
            palette = self.query_one("#slash-palette", SlashPalette)
            if not self._slash_matches:
                self._slash_open = False
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
            self._slash_open = True

        def _update_slash_palette(self, value: str, *, reset_selection: bool = True) -> None:
            if self._sessions_open:
                # Conversations menu owns the overlay while open.
                return
            palette = self.query_one("#slash-palette", SlashPalette)
            if not value.startswith("/") or "\n" in value:
                self._slash_matches = []
                self._slash_open = False
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
                self._slash_open = False
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

        # ── Conversations menu ────────────────────────────────────────

        def _dismiss_sessions_menu(self) -> bool:
            """Hide the conversations menu if open. Returns True when closed."""
            closed = bool(self._sessions_open or self._session_rows)
            self._sessions_open = False
            self._session_rows = []
            self._session_selected = 0
            self._session_catalog_total = 0
            try:
                menu = self.query_one("#sessions-menu", SessionsMenu)
                if menu.has_class("-visible"):
                    closed = True
                menu.hide_menu()
            except Exception:
                pass
            return closed

        def _session_filter_query(self) -> str:
            try:
                return (self.query_one("#prompt-input", Input).value or "").strip()
            except Exception:
                return ""

        def _build_session_rows(self, query: str = "") -> list[tuple[str, str, str]]:
            """Top row is always New Conversation; then saved sessions (filterable)."""
            from agent.core import list_session_catalog

            catalog = list_session_catalog(limit=80)
            self._session_catalog_total = len(catalog) + 1  # include New Conversation
            needle = (query or "").strip().casefold()
            rows: list[tuple[str, str, str]] = [
                (SessionsMenu.NEW_KEY, "New Conversation", "start fresh"),
            ]
            for entry in catalog:
                title = str(entry.get("title") or "")
                detail = str(entry.get("detail") or "")
                path = str(entry.get("path") or "")
                if needle and needle not in title.casefold() and needle not in detail.casefold():
                    continue
                rows.append((path, title, detail))
            # Keep New Conversation even when filtering — unless the filter
            # clearly targets a saved name and user wants only matches? Spec says
            # top option always New Conversation.
            if needle and "new conversation".find(needle) < 0 and needle not in "new":
                # Still keep it first always.
                pass
            return rows

        def _refresh_sessions_view(self) -> None:
            menu = self.query_one("#sessions-menu", SessionsMenu)
            if not self._sessions_open or not self._session_rows:
                menu.hide_menu()
                return
            self._session_selected = max(
                0, min(self._session_selected, len(self._session_rows) - 1)
            )
            menu.show_rows(
                self._session_rows,
                self._session_selected,
                query=self._session_filter_query(),
                total=self._session_catalog_total,
            )

        def _open_sessions_menu(self, *, reset_selection: bool = True) -> None:
            self._dismiss_speech_menu()
            self._dismiss_slash_palette()
            self._sessions_open = True
            query = self._session_filter_query()
            # Opening with a slash filter leftover is confusing — clear leading '/'.
            if query.startswith("/"):
                try:
                    inp = self.query_one("#prompt-input", Input)
                    inp.value = ""
                    query = ""
                except Exception:
                    query = ""
            self._session_rows = self._build_session_rows(query)
            if reset_selection:
                self._session_selected = 0
            self._refresh_sessions_view()

        def _update_sessions_filter(self, value: str, *, reset_selection: bool = True) -> None:
            if not self._sessions_open:
                return
            self._session_rows = self._build_session_rows(value)
            if not self._session_rows:
                # Always at least New Conversation.
                self._session_rows = [
                    (SessionsMenu.NEW_KEY, "New Conversation", "start fresh"),
                ]
            if reset_selection:
                self._session_selected = 0
            self._refresh_sessions_view()

        def _move_session(self, delta: int) -> None:
            if not self._sessions_open or not self._session_rows:
                return
            self._session_selected = (self._session_selected + int(delta)) % len(
                self._session_rows
            )
            self._refresh_sessions_view()

        def _history_message_text(self, message: dict) -> str:
            content = message.get("content")
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text")
                        if text:
                            parts.append(str(text))
                    elif part:
                        parts.append(str(part))
                return "\n".join(parts)
            return str(content)

        def _rebuild_transcript_from_history(self) -> None:
            """Replay loaded history into the chat view (user + assistant only)."""
            self._reset_transcript()
            for message in list(self.history or []):
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "")
                text = self._history_message_text(message).strip()
                if not text:
                    continue
                if role == "user":
                    header = Text()
                    header.append(f"{GLYPH_MARK} ", style="bold #f2f2f2")
                    header.append("you\n", style="bold #ffffff")
                    header.append(text, style="#f2f2f2")
                    self._mount_block(header, "user")
                elif role == "assistant":
                    rendered = Markdown(_render_terminal_markdown(text or " "))
                    title = Text()
                    title.append(f"{GLYPH_SECTION} ", style="bold #ffffff")
                    title.append("response\n", style="bold #f2f2f2")
                    self._mount_block(Group(title, rendered), "assistant")
            self.refresh_context_usage()

        def _accept_session_selection(self) -> None:
            if not self._sessions_open or not self._session_rows:
                return
            if self._busy:
                self.ui_status("Still generating — wait for the current turn", kind="warn")
                return
            key, title, _detail = self._session_rows[self._session_selected]
            # Clear filter text from the composer.
            try:
                inp = self.query_one("#prompt-input", Input)
                inp.value = ""
            except Exception:
                pass
            self._dismiss_sessions_menu()

            if key == SessionsMenu.NEW_KEY:
                from agent.core import start_new_conversation

                start_new_conversation(self.session, self.history)
                self._reset_transcript()
                self.ui_status("New conversation", kind="ok")
                self.refresh_context_usage()
                return

            from agent.core import apply_saved_session_file
            from agent.runtime_config import RuntimeConfigurationError

            try:
                display_name, msg_count, warnings = apply_saved_session_file(
                    key, self.session, self.history
                )
            except RuntimeConfigurationError as exc:
                self.ui_status(f"Load failed · {exc}", kind="error")
                return
            except Exception as exc:
                self.ui_status(f"Load failed · {exc}", kind="error")
                return

            # Theme may have changed with the saved session.
            try:
                theme = str(self.session.get("tui_theme") or "oslo")
                self._apply_selene_theme(theme, announce=False)
            except Exception:
                pass

            self._rebuild_transcript_from_history()
            label = "message" if msg_count == 1 else "messages"
            self.ui_status(
                f"Loaded · {display_name}",
                kind="ok",
                detail=f"{msg_count} user {label}",
            )
            for warning in warnings:
                self.ui_status(str(warning), kind="warn")

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id == "speech-input":
                # Keep the recognizer base in sync so live phrases append to edits.
                voice = self._voice
                if voice is not None and (
                    self._voice_active or getattr(voice, "active", False)
                ):
                    try:
                        voice.set_base_text(event.value or "")
                    except Exception:
                        pass
                return
            if event.input.id != "prompt-input":
                return
            if self._sessions_open:
                self._update_sessions_filter(event.value, reset_selection=True)
                self.refresh_context_usage()
                return
            self._update_slash_palette(event.value, reset_selection=True)
            self.refresh_context_usage()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id == "speech-input":
                self._speech_on_enter()
                return
            if event.input.id != "prompt-input":
                return
            if self._sessions_open:
                self._accept_session_selection()
                return
            self._submit_from_input()

        def _submit_from_input(self) -> None:
            if self._speech_open:
                self._speech_on_enter()
                return
            if self._sessions_open:
                self._accept_session_selection()
                return
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
            self._dismiss_slash_palette()
            if value:
                self._submit(value)

        def on_key(self, event) -> None:
            key = event.key

            # Esc must close menus even if focus left the composer (e.g. after
            # opening the palette with Ctrl+/). Handle before the focus gate.
            if key == "escape":
                if (
                    self._speech_open
                    or self._sessions_open
                    or self._slash_palette_visible()
                ):
                    event.prevent_default()
                    event.stop()
                    self.action_blur_or_clear()
                    return

            # Intercept palette / conversations navigation while the input is focused.
            try:
                focused = self.focused
            except Exception:
                focused = None
            focused_id = getattr(focused, "id", None) if focused is not None else None
            if focused_id == "speech-input":
                # Speech popup owns Enter/typing; no slash/session navigation.
                return
            if focused is None or focused_id != "prompt-input":
                return

            # Conversations menu navigation takes priority while open.
            if self._sessions_open:
                if key in ("ctrl+n", "down"):
                    event.prevent_default()
                    event.stop()
                    self._move_session(1)
                    return
                if key in ("ctrl+p", "up"):
                    event.prevent_default()
                    event.stop()
                    self._move_session(-1)
                    return
                if key in ("pageup",):
                    event.prevent_default()
                    event.stop()
                    self._move_session(-5)
                    return
                if key in ("pagedown",):
                    event.prevent_default()
                    event.stop()
                    self._move_session(5)
                    return
                if key in ("home",):
                    event.prevent_default()
                    event.stop()
                    self._session_selected = 0
                    self._refresh_sessions_view()
                    return
                if key in ("end",):
                    event.prevent_default()
                    event.stop()
                    self._session_selected = max(0, len(self._session_rows) - 1)
                    self._refresh_sessions_view()
                    return
                if key == "tab":
                    # Tab does not complete sessions; Enter opens.
                    event.prevent_default()
                    event.stop()
                    return
                return

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

        def _queue_depth(self) -> int:
            with self._busy_lock:
                return len(self._prompt_queue)

        def _queue_snapshot(self) -> list[str]:
            with self._busy_lock:
                return list(self._prompt_queue)

        def _refresh_queue_ui(self) -> None:
            """Render numbered one-line previews above the composer."""
            try:
                panel = self.query_one("#prompt-queue", PromptQueuePanel)
            except Exception:
                return
            items = self._queue_snapshot()
            if not items:
                panel.hide_queue()
                return
            panel.show_queue(items)

        def _submit(self, user_input: str) -> None:
            stripped = str(user_input or "").strip()
            if not stripped:
                return
            base = stripped.split(None, 1)[0].lower() if stripped else ""
            # Voice toggle must stay on the UI thread (no generation lock).
            if base == "/speech":
                rest = stripped.split(None, 1)[1] if len(stripped.split(None, 1)) > 1 else ""
                self.ui_add_user(user_input)
                self.ui_toggle_speech(rest or "toggle")
                return

            with self._busy_lock:
                if self._busy:
                    # Keep the composer open: queue for after the current turn.
                    if len(self._prompt_queue) >= self._PROMPT_QUEUE_MAX:
                        full = True
                    else:
                        self._prompt_queue.append(stripped)
                        full = False
                    depth = len(self._prompt_queue)
                else:
                    self._busy = True
                    full = False
                    depth = -1

            if depth >= 0:
                # Was busy — queued or rejected.
                self._refresh_queue_ui()
                if full:
                    self.ui_status(
                        f"Prompt queue full ({self._PROMPT_QUEUE_MAX})",
                        kind="warn",
                    )
                return

            # Match web: stop mic when a turn starts.
            self._stop_voice(silent=True, abort=True)

            self.ui_add_user(user_input)
            self._stream_widget = None
            self._thinking_widget = None
            self._thinking_buf = ""
            self._thinking_tokens = 0
            # Fresh activity line for this turn (Thinking animation).
            self._stop_activity_timer()
            self._remove_activity_widget()
            self._activity_phase = "idle"
            # Composer stays enabled so the user can type / queue the next prompt.
            try:
                inp = self.query_one("#prompt-input", Input)
                inp.disabled = False
            except Exception:
                pass

            def work() -> None:
                try:
                    if user_input.startswith("/"):
                        result = self._handle_command(
                            user_input, self.session, self.history
                        )
                        if result is None:
                            self.call_from_thread(self.exit)
                            return
                        cmd_base = user_input.strip().split(None, 1)[0].lower()
                        if cmd_base == "/clear":
                            # Reset UI after core clears history; show one confirmation.
                            self.call_from_thread(self._clear_conversation_ui)
                        elif cmd_base == "/load" and len(user_input.strip().split(None, 1)) > 1:
                            # Rebuild transcript after a successful /load <target>.
                            self.call_from_thread(self._rebuild_transcript_from_history)
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
            next_prompt: str | None = None
            with self._busy_lock:
                self._busy = False
                if self._prompt_queue:
                    next_prompt = self._prompt_queue.pop(0)
            try:
                inp = self.query_one("#prompt-input", Input)
                inp.disabled = False
                if next_prompt is None:
                    inp.focus()
            except Exception:
                pass
            # Flush any remaining capture buffer.
            if self._capture_file is not None:
                self._capture_file.flush()
            self.refresh_context_usage()
            self._refresh_queue_ui()
            # Serve the next queued prompt immediately (FIFO).
            if next_prompt is not None:
                self._submit(next_prompt)

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
            """Ctrl+K — clear the composer (and dismiss menus)."""
            if self._dismiss_speech_menu():
                return
            inp = self.query_one("#prompt-input", Input)
            inp.value = ""
            inp.focus()
            self._dismiss_slash_palette()
            self._dismiss_sessions_menu()

        def action_open_commands(self) -> None:
            """Ctrl+/ — open (or toggle closed) the slash command palette."""
            inp = self.query_one("#prompt-input", Input)
            if self._busy:
                return
            self._dismiss_speech_menu()
            self._dismiss_sessions_menu()
            # Second Ctrl+/ closes the palette cleanly (same as Esc).
            if self._slash_palette_visible():
                self._dismiss_slash_palette(clear_slash_draft=True)
                try:
                    inp.focus()
                except Exception:
                    pass
                return
            if not (inp.value or "").startswith("/"):
                inp.value = "/"
            inp.focus()
            inp.cursor_position = len(inp.value or "")
            self._update_slash_palette(inp.value or "/", reset_selection=True)
            self._slash_open = bool(self._slash_matches)

        def action_open_sessions(self) -> None:
            """Ctrl+O — open the saved conversations menu."""
            if self._busy:
                return
            try:
                inp = self.query_one("#prompt-input", Input)
                inp.focus()
            except Exception:
                pass
            # Toggle: second Ctrl+O closes gracefully.
            if self._sessions_open:
                self._dismiss_sessions_menu()
                return
            self._open_sessions_menu(reset_selection=True)

        def action_show_help(self) -> None:
            """Show command help in the transcript."""
            if self._busy:
                return
            self._dismiss_sessions_menu()
            self._dismiss_slash_palette()
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
                "ctrl+/ palette  ·  ctrl+o chats  ·  ctrl+s speech  ·  tab complete",
            )
            shortcuts = Text()
            shortcuts.append(f"{GLYPH_SECTION} ", style="#6b6b6b")
            shortcuts.append("shortcuts\n", style="bold #7a7a7a")
            for key, desc in (
                ("Enter / Ctrl+J", "Send message (queues up to 3 while generating)"),
                ("Ctrl+C", "Stop generation"),
                ("Ctrl+C twice", "Quit application"),
                ("Ctrl+/", "Open command palette"),
                ("Ctrl+O", "Open conversations menu"),
                ("Ctrl+S", "Open speech menu (/speech)"),
                ("Tab / →", "Autofill highlighted command"),
                ("↑↓  Ctrl+N/P", "Move menu selection"),
                ("PgUp / PgDn", "Jump menu selection"),
                ("Esc", "Dismiss menu / speech / clear input"),
                ("Ctrl+K", "Clear input + dismiss menus"),
                ("Ctrl+L", "Clear conversation"),
            ):
                shortcuts.append(f"  {key:<16}", style="#7a7a7a")
                shortcuts.append(f"  {desc}\n", style="#555555")
            self._mount_block(shortcuts, "system")

        def action_submit_input(self) -> None:
            """Ctrl+J — send the current composer text (or queue while generating)."""
            self._submit_from_input()

        # ── Voice input (centered speech popup) ───────────────────────

        def _ensure_voice(self):
            if self._voice is not None:
                return self._voice
            from agent.speech_input import VoiceInputController

            def _on_transcript(text: str) -> None:
                try:
                    self.call_from_thread(self._voice_apply_transcript, text)
                except Exception:
                    pass

            def _on_active(active: bool) -> None:
                try:
                    self.call_from_thread(self._voice_set_active_ui, bool(active))
                except Exception:
                    pass

            def _on_error(_code: str, message: str) -> None:
                try:
                    self.call_from_thread(self._voice_on_error, message)
                except Exception:
                    pass

            def _on_finished() -> None:
                try:
                    self.call_from_thread(self._speech_finish_submit)
                except Exception:
                    pass

            self._voice = VoiceInputController(
                on_transcript=_on_transcript,
                on_active=_on_active,
                on_error=_on_error,
                on_finished=_on_finished,
            )
            return self._voice

        def _voice_on_error(self, message: str) -> None:
            """Keep audio failures contained and visible inside the popup."""
            self._voice_active = False
            self._voice_error_message = str(message or "Voice input unavailable")
            if not self._speech_open:
                return
            try:
                self.query_one("#speech-menu", SpeechMenu).set_error(
                    self._voice_error_message
                )
            except Exception:
                pass
            # Armed stays True so a second Enter still sends typed text.
            try:
                self.query_one("#speech-input", Input).focus()
            except Exception:
                pass
            try:
                self.refresh()
            except Exception:
                pass

        def _open_speech_menu(self, *, start_recording: bool = False) -> None:
            """Show the centered mic + transcript popup."""
            if self._busy:
                return
            self._dismiss_sessions_menu()
            self._dismiss_slash_palette(clear_slash_draft=True)
            try:
                menu = self.query_one("#speech-menu", SpeechMenu)
            except Exception:
                return

            # Prefill from the main composer when it holds a normal draft.
            draft = ""
            try:
                main = self.query_one("#prompt-input", Input)
                draft = main.value or ""
                if draft.strip().startswith("/"):
                    draft = ""
            except Exception:
                draft = ""

            try:
                speech = self.query_one("#speech-input", Input)
                speech.value = draft
                speech.cursor_position = len(speech.value or "")
            except Exception:
                pass

            menu.show_menu()
            self._speech_open = True
            self._speech_armed = False
            self._speech_pending_submit = False
            self._voice_error_message = ""
            try:
                self.query_one("#speech-input", Input).focus()
            except Exception:
                pass
            try:
                self.refresh()
            except Exception:
                pass

            if start_recording:
                self._speech_armed = True
                self._speech_start_recording()

        def _dismiss_speech_menu(self, *, stop_voice: bool = True) -> bool:
            """Hide the speech popup. Returns True when it was open."""
            closed = bool(self._speech_open)
            try:
                menu = self.query_one("#speech-menu", SpeechMenu)
                if menu.has_class("-visible"):
                    closed = True
                menu.hide_menu()
            except Exception:
                pass
            self._speech_open = False
            self._speech_armed = False
            self._speech_pending_submit = False
            self._voice_error_message = ""
            if stop_voice:
                self._stop_voice(silent=True, abort=True)
            else:
                self._voice_active = False
                try:
                    self.query_one("#speech-menu", SpeechMenu).set_recording(False)
                except Exception:
                    pass
            if closed:
                try:
                    self.query_one("#prompt-input", Input).focus()
                except Exception:
                    pass
                try:
                    self.refresh()
                except Exception:
                    pass
            return closed

        def _speech_start_recording(self) -> None:
            """Begin capture into the speech popup textbox."""
            if self._busy or not self._speech_open:
                return
            voice = self._ensure_voice()
            if voice.active or self._voice_active:
                return
            try:
                base = self.query_one("#speech-input", Input).value or ""
            except Exception:
                base = ""
            self._voice_error_message = ""
            # Optimistic UI — flip to recording animation immediately.
            try:
                self.query_one("#speech-menu", SpeechMenu).set_recording(True)
            except Exception:
                pass
            try:
                started = bool(voice.start(base_text=base))
            except Exception:
                started = False
            if not started:
                # A worker-side capability error updates the popup asynchronously.
                self._voice_active = False
                try:
                    from agent.speech_input import speech_capability

                    capability = speech_capability()
                    if not capability.available:
                        self._voice_on_error(capability.detail)
                    else:
                        self.query_one("#speech-menu", SpeechMenu).set_recording(False)
                except Exception:
                    pass
                try:
                    self.refresh()
                except Exception:
                    pass

        def _speech_on_enter(self) -> None:
            """Enter: start recording once; Enter again stops and sends the prompt."""
            if not self._speech_open or self._busy:
                return
            if self._speech_pending_submit:
                return
            if not self._speech_armed:
                # First Enter — arm and begin listening.
                self._speech_armed = True
                self._speech_start_recording()
                return

            # Second Enter requests a clean stop. Keep the popup alive until
            # the in-flight phrase has been recognized so the last words are
            # not lost merely because Enter was pressed quickly.
            voice = self._voice
            if voice is not None and (voice.active or self._voice_active):
                self._speech_pending_submit = True
                self._stop_voice(silent=False, abort=False)
                try:
                    self.query_one("#speech-menu", SpeechMenu).set_finishing()
                except Exception:
                    pass
                return
            self._speech_finish_submit(force=True)

        def _speech_finish_submit(self, *, force: bool = False) -> None:
            """Submit after clean speech shutdown has delivered its final text."""
            if not self._speech_open or (not force and not self._speech_pending_submit):
                return
            try:
                text = (self.query_one("#speech-input", Input).value or "").strip()
            except Exception:
                text = ""
            self._speech_pending_submit = False
            if text:
                self._dismiss_speech_menu(stop_voice=False)
                self._submit(text)
                return
            # Nothing was recognized. Keep the typed fallback available and
            # let Enter retry instead of closing an apparently broken popup.
            self._speech_armed = False
            try:
                menu = self.query_one("#speech-menu", SpeechMenu)
                if self._voice_error_message:
                    menu.set_error(self._voice_error_message)
                else:
                    menu.set_recording(False)
                self.query_one("#speech-input", Input).focus()
            except Exception:
                pass

        def _voice_apply_transcript(self, text: str) -> None:
            """Put live speech text into the speech popup textbox."""
            if not self._speech_open:
                return
            try:
                inp = self.query_one("#speech-input", Input)
            except Exception:
                return
            inp.value = str(text or "")
            try:
                inp.cursor_position = len(inp.value or "")
            except Exception:
                pass

        def _voice_set_active_ui(self, active: bool) -> None:
            was = self._voice_active
            self._voice_active = bool(active)
            if not self._speech_open:
                return
            try:
                menu = self.query_one("#speech-menu", SpeechMenu)
                if not active and self._voice_error_message:
                    menu.set_error(self._voice_error_message)
                elif not active and self._speech_pending_submit:
                    menu.set_finishing()
                else:
                    menu.set_recording(active)
            except Exception:
                pass
            # If the mic stopped unexpectedly, repaint so any residual driver
            # noise under the alternate screen is overwritten.
            if was and not active:
                try:
                    self.refresh()
                except Exception:
                    pass

        def _stop_voice(self, *, silent: bool = False, abort: bool = False) -> None:
            voice = self._voice
            if voice is None:
                self._voice_active = False
                try:
                    if self._speech_open:
                        self.query_one("#speech-menu", SpeechMenu).set_recording(False)
                except Exception:
                    pass
                return
            try:
                # Aborts are expected; clean stops still report recognition errors.
                voice.stop(abort=abort, silent=silent)
            except Exception:
                pass
            self._voice_active = False
            try:
                if self._speech_open:
                    self.query_one("#speech-menu", SpeechMenu).set_recording(False)
            except Exception:
                pass

        def ui_toggle_speech(self, action: str = "toggle") -> None:
            """Handle /speech or Ctrl+S — same centered speech menu.

            Enter starts recording; Enter again finalizes and sends. Errors stay
            contained inside the speech popup instead of corrupting TUI output.
            """
            action = str(action or "toggle").strip().lower() or "toggle"
            if action in {"status"}:
                # Capability check stays silent in the TUI.
                return

            if self._busy and action not in {"stop", "off"}:
                return

            if action in {"stop", "off"}:
                # Stop listening but keep the popup open so the draft remains editable.
                self._stop_voice(silent=True, abort=False)
                return

            if action in {"start", "on"}:
                if not self._speech_open:
                    self._open_speech_menu(start_recording=True)
                else:
                    self._speech_armed = True
                    self._speech_start_recording()
                return

            # toggle (default) — open/close the speech popup
            if self._speech_open:
                self._dismiss_speech_menu()
                return
            self._open_speech_menu(start_recording=False)

        def action_toggle_speech(self) -> None:
            """Ctrl+S — open or close the centered speech menu."""
            self.ui_toggle_speech("toggle")

        def action_blur_or_clear(self) -> None:
            """Esc — dismiss speech, then chats, then palette, then input.

            One Esc always exits the open menu (including palette opened via
            Ctrl+/). A further Esc clears leftover composer text.
            """
            if self._dismiss_speech_menu():
                return
            if self._dismiss_sessions_menu():
                try:
                    self.query_one("#prompt-input", Input).focus()
                except Exception:
                    pass
                return
            if self._dismiss_slash_palette(clear_slash_draft=True):
                try:
                    self.query_one("#prompt-input", Input).focus()
                except Exception:
                    pass
                return
            try:
                inp = self.query_one("#prompt-input", Input)
            except Exception:
                return
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
