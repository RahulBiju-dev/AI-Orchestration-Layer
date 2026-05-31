"""
agent/terminal.py — Terminal helpers and lightweight LaTeX math renderer

Contains ANSI helpers, a spinner, and a compact LaTeX-to-terminal
renderer used by the streaming output in `agent.core`.
"""
from __future__ import annotations

import re
import sys
from rich.console import Console

# Shared console (write to stderr so Live and spinner use the same stream)
_console = Console(stderr=True)


def _print_status(icon: str, message: str, color: str = "cyan") -> None:
    """Print a formatted status line to stderr so it doesn't mix with piped output."""
    _console.print(f"[{color} bold]{icon}  {message}[/]")


def print_welcome_header() -> None:
    """Prints a stylish welcome header with an ASCII art logo."""
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    
    # ASCII Art logo
    logo_art_raw = r"""
                              /
                   __       //
                   -\= \=\ //
                 --=_\=---//=--
               -_==/  \/ //\/--
                ==/   /O   O\==--
   _ _ _ _     /_/    \  ]  /--
  /\ ( (- \    /       ] ] ]==-
 (\ _\_\_\-\__/     \  (,_,)--
(\_/                 \     \-
\/      /       (   ( \  ] /)
/      (         \   \_ \./ )
(       \         \      )  \
(       /\_ _ _ _ /---/ /\_  \
 \     / \     / ____/ /   \  \
  (   /   )   / /  /__ )   (  )
  (  )   / __/ '---`       / /
  \  /   \ \             _/ /
  ] ]     )_\_         /__\/
  /_\     ]___\
 (___)
""".strip("\n")
    logo_art = Text(logo_art_raw, style="cyan")
    
    title_text = Text("\n\n\n\n\n\nAI CLI Agent\n", style="bold white", justify="center")
    title_text.append("Type ", style="dim")
    title_text.append("/help", style="bold green")
    title_text.append(" to see available commands.", style="dim")
    
    columns = Columns([logo_art, title_text], expand=True, align="center")
    _console.print(Panel(columns, border_style="cyan", padding=(1, 2)))


class _Spinner:
    """Animated spinner wrapper that uses rich.status.Status."""

    def __init__(self, message: str = "Thinking", color: str = "magenta") -> None:
        self._message = message
        self._color = color
        self._status = _console.status(f"[{self._color} bold]{self._message}…[/]", spinner="dots", spinner_style=f"{self._color} bold")
        
        # Keep threading variables for backwards compatibility with core.py's interrupt checking
        import threading
        self._stop_event = threading.Event()
        self._thread = type('MockThread', (), {'is_alive': lambda: False, 'join': lambda: None})()

    def start(self) -> "_Spinner":
        self._stop_event.clear()
        self._status.start()
        return self

    def update(self, message: str) -> None:
        self._message = message
        self._status.update(f"[{self._color} bold]{self._message}…[/]")

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
    out = []
    for ch in text:
        out.append(_SUP_MAP.get(ch, ch))
    return "".join(out)


def _to_subscript(text: str) -> str:
    out = []
    for ch in text:
        out.append(_SUB_MAP.get(ch, ch))
    return "".join(out)


def _extract_braced(text: str, start: int) -> tuple[str, int] | None:
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


def _render_terminal_markdown(text: str) -> str:
    def replace_block(match: re.Match[str]) -> str:
        rendered = _render_latex_math(match.group(1))
        return f"\n{rendered}\n"

    def replace_inline(match: re.Match[str]) -> str:
        return _render_latex_math(match.group(1))

    text = _RE_BLOCK_LATEX.sub(replace_block, text)
    text = _RE_INLINE_LATEX.sub(replace_inline, text)
    return text.replace(r"\$", "$")
