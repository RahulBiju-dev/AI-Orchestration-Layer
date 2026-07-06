"""
agent/terminal.py — Terminal helpers and lightweight LaTeX math renderer

Contains ANSI helpers, a spinner, and a compact LaTeX-to-terminal
renderer used by the streaming output in `agent.core`.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from rich.console import Console

# Shared console (write to stderr so Live and spinner use the same stream)
_console = Console(stderr=True)


@dataclass(frozen=True)
class TerminalTheme:
    """Small central palette for the CLI chrome."""

    accent: str = "cyan"
    accent_2: str = "bright_magenta"
    success: str = "green"
    warning: str = "yellow"
    danger: str = "red"
    muted: str = "dim"
    border: str = "cyan"


THEME = TerminalTheme()


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


def read_user_input() -> str:
    """Read one prompt line using the shared console and terminal hygiene."""
    flush_terminal_input()
    raw = _console.input("[bold cyan]selene[/] [dim]$[/] ")
    return sanitize_terminal_input(raw)


def assistant_stream_panel(text: str):
    """Build the live assistant renderable."""
    from rich.markdown import Markdown
    from rich.panel import Panel

    body = Markdown(_render_terminal_markdown(text or " "))
    return Panel(
        body,
        border_style=THEME.border,
        padding=(1, 2),
        title="[bold cyan]assistant[/]",
        title_align="left",
    )


def print_assistant_message(text: str) -> None:
    """Print the final assistant response in the persistent scrollback."""
    if not text:
        return
    _console.print(assistant_stream_panel(text))
    _console.print()


def print_thinking_header() -> None:
    from rich.rule import Rule

    _console.print(Rule("[dim magenta]thinking[/]", style="dim magenta"))


def print_thinking_footer(label: str | None = None) -> None:
    from rich.rule import Rule

    text = f"[dim magenta]{label}[/]" if label else ""
    _console.print(Rule(text, style="dim magenta"))
    _console.print()


def _print_status(icon: str, message: str, color: str = "cyan") -> None:
    """Print a formatted status line to stderr so it doesn't mix with piped output.
    
    Args:
        icon (str): The emoji or icon to display at the beginning of the line.
        message (str): The status message to print.
        color (str, optional): The rich color to use for the output. Defaults to "cyan".
    """
    _console.print(f"[{color} bold]{icon}  {message}[/]")


def print_welcome_header() -> None:
    """Print the terminal app header."""
    from rich.text import Text

    logo_lines = [
        r"*          .                  .",
        r"      _..._",
        r"   .::::   `.",
        r"  :::::      :       ____  _____ _     _____ _   _ _____",
        r"  `::::.   .'       / ___|| ____| |   | ____| \ | | ____|",
        r"     `''''`         \___ \|  _| | |   |  _| |  \| |  _|",
        r".                    ___) | |___| |___| |___| |\  | |___",
        r"     *              |____/|_____|_____|_____|_| \_|_____|",
        "",
        r"                 N I G H T   M O D E   A G E N T",
    ]
    width = min(_console.size.width, 100)
    block_width = max(len(line) for line in logo_lines)
    left_margin = max((width - block_width) // 2, 0)
    _console.print()
    for index, line in enumerate(logo_lines):
        style = "dim cyan" if index == len(logo_lines) - 1 else "cyan"
        padded_line = f"{' ' * left_margin}{line}"
        _console.print(
            Text(padded_line, style=style),
            markup=False,
            highlight=False,
            no_wrap=True,
            overflow="crop",
        )
    _console.print()


class _Spinner:
    """Animated spinner wrapper that uses rich.status.Status.
    
    This class provides a simple API to start, update, and stop a terminal spinner.
    It handles its own state and threading compatibility variables.
    """

    def __init__(self, message: str = "Thinking", color: str = "magenta") -> None:
        self._message = message
        
        self._color = color
        self._status = _console.status(f"[{self._color} bold]{self._message}…[/]", spinner="dots", spinner_style=f"{self._color} bold")
        
        # Keep threading variables for backwards compatibility with core.py's interrupt checking
        import threading
        self._stop_event = threading.Event()
        self._thread = type('MockThread', (), {'is_alive': lambda: False, 'join': lambda: None})()

    def start(self) -> "_Spinner":
        """Start the animated spinner in the terminal.
        
        Returns:
            _Spinner: The current spinner instance for method chaining.
        """
        self._stop_event.clear()
        self._status.start()
        return self

    def update(self, message: str) -> None:
        """Update the spinner's message dynamically while it is running.
        
        Args:
            message (str): The new message to display.
        """
        self._message = message
        self._status.update(f"[{self._color} bold]{self._message}…[/]")

    def stop(self) -> None:
        """Stop the animated spinner and mark the stop event as set."""
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
