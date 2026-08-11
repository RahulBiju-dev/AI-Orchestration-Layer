"""TUI color themes for the Selene full-screen agent interface.

Themes are named after places. Each palette is meant to evoke that place's
vibe (Oslo monochrome, Tokyo hollow neon-blue, Rome royal gold, Amazon forest,
Cairo sand, etc.). ``oslo`` is the default and is always listed first.

Themes drive Textual CSS variables plus a small Rich palette for markup-free
Text spans (slash palette, thinking chrome, etc.). Registered Textual theme
names use Title Case place names so Ctrl+P → Themes matches the slash catalog.

Each place also carries a ``selene-user`` hue. Prompts are painted with it and
responses keep the theme accent, so the two transcript roles never read as the
same block. Block tints (``selene-user-bg`` / ``selene-assistant-bg``) are
derived from that pair, which keeps every theme in step without hand-tuning a
second set of near-identical greys.
"""
from __future__ import annotations

from typing import Any

# Order matters: first entry is the default / catalog head.
THEME_ORDER: tuple[str, ...] = (
    "oslo",
    "tokyo",
    "rome",
    "amazon",
    "cairo",
    "kyoto",
    "bergen",
    "marrakech",
    "shanghai",
    "reykjavik",
    "venice",
    "seoul",
    "santorini",
    "havana",
)

DEFAULT_THEME = "oslo"

# Canonical key → Title Case name shown in Ctrl+P theme picker.
def _display_name(key: str) -> str:
    if key == DEFAULT_THEME:
        return "Oslo (default)"
    return key[:1].upper() + key[1:] if key else key


# Each theme: Textual Theme kwargs + extended palette for Rich Text styling.
# ``label`` is the short vibe blurb next to the place name in menus.
_THEME_DEFS: dict[str, dict[str, Any]] = {
    "oslo": {
        "label": "Oslo (default) — monochrome grey & white",
        "dark": True,
        "primary": "#cfcfcf",
        "secondary": "#7a7a7a",
        "accent": "#e8e8e8",
        "foreground": "#f2f2f2",
        "background": "#0b0b0b",
        "surface": "#131313",
        "panel": "#1a1a1a",
        "warning": "#c8c8a0",
        "error": "#c08080",
        "success": "#6a8a6a",
        "boost": "#161616",
        "variables": {
            "selene-muted": "#7a7a7a",
            "selene-faint": "#555555",
            "selene-text-soft": "#d8d8d8",
            "selene-border": "#2e2e2e",
            "selene-border-soft": "#242424",
            "selene-border-focus": "#9a9a9a",
            "selene-elevated": "#161616",
            "selene-select-fg": "#0a0a0a",
            "selene-select-bg": "#e8e8e8",
            "selene-content": "#1a1a1a",
            "selene-user": "#8f8f8f",
        },
    },
    "tokyo": {
        "label": "Tokyo — futuristic hollow blue",
        "dark": True,
        "primary": "#5ec8ff",
        "secondary": "#3d5a80",
        "accent": "#7dcfff",
        "foreground": "#d0e8ff",
        "background": "#070b14",
        "surface": "#0c1220",
        "panel": "#111a2c",
        "warning": "#e0af68",
        "error": "#f7768e",
        "success": "#7fd4a8",
        "boost": "#152038",
        "variables": {
            "selene-muted": "#4a6a90",
            "selene-faint": "#2a3d5c",
            "selene-text-soft": "#a8c8e8",
            "selene-border": "#1e3048",
            "selene-border-soft": "#152038",
            "selene-border-focus": "#5ec8ff",
            "selene-elevated": "#152038",
            "selene-select-fg": "#070b14",
            "selene-select-bg": "#5ec8ff",
            "selene-content": "#111a2c",
            "selene-user": "#9d7cd8",
        },
    },
    "rome": {
        "label": "Rome — royal gold & marble",
        "dark": True,
        "primary": "#d4af37",
        "secondary": "#8b7355",
        "accent": "#f0d78c",
        "foreground": "#f5efe0",
        "background": "#120e18",
        "surface": "#1a1424",
        "panel": "#241c30",
        "warning": "#e8c060",
        "error": "#c06070",
        "success": "#8aaa70",
        "boost": "#2a2038",
        "variables": {
            "selene-muted": "#8b7355",
            "selene-faint": "#5c4a60",
            "selene-text-soft": "#e0d4b8",
            "selene-border": "#3a3048",
            "selene-border-soft": "#2a2038",
            "selene-border-focus": "#d4af37",
            "selene-elevated": "#2a2038",
            "selene-select-fg": "#120e18",
            "selene-select-bg": "#d4af37",
            "selene-content": "#241c30",
            "selene-user": "#9a8ac4",
        },
    },
    "amazon": {
        "label": "Amazon — deep rainforest",
        "dark": True,
        "primary": "#6dbf6d",
        "secondary": "#4a7a4a",
        "accent": "#a8d4a0",
        "foreground": "#e4f0e4",
        "background": "#0a120c",
        "surface": "#101a12",
        "panel": "#162418",
        "warning": "#d4c070",
        "error": "#c07070",
        "success": "#6dbf6d",
        "boost": "#1a2e1c",
        "variables": {
            "selene-muted": "#5a7a5a",
            "selene-faint": "#3a4a3a",
            "selene-text-soft": "#b8d0b8",
            "selene-border": "#243828",
            "selene-border-soft": "#1a2e1c",
            "selene-border-focus": "#6dbf6d",
            "selene-elevated": "#1a2e1c",
            "selene-select-fg": "#0a120c",
            "selene-select-bg": "#6dbf6d",
            "selene-content": "#162418",
            "selene-user": "#d0b878",
        },
    },
    "cairo": {
        "label": "Cairo — sand & desert brown",
        "dark": True,
        "primary": "#c4a574",
        "secondary": "#8a7050",
        "accent": "#e0c898",
        "foreground": "#f5ead8",
        "background": "#14100c",
        "surface": "#1c1610",
        "panel": "#261e16",
        "warning": "#e0af68",
        "error": "#c07060",
        "success": "#8aaa60",
        "boost": "#2e2418",
        "variables": {
            "selene-muted": "#8a7050",
            "selene-faint": "#5c4a38",
            "selene-text-soft": "#d4c4a8",
            "selene-border": "#3a3020",
            "selene-border-soft": "#2e2418",
            "selene-border-focus": "#c4a574",
            "selene-elevated": "#2e2418",
            "selene-select-fg": "#14100c",
            "selene-select-bg": "#c4a574",
            "selene-content": "#261e16",
            "selene-user": "#8fb0b8",
        },
    },
    "kyoto": {
        "label": "Kyoto — soft sakura dusk",
        "dark": True,
        "primary": "#d4a0c0",
        "secondary": "#7a6a80",
        "accent": "#ebbcba",
        "foreground": "#f0e8f0",
        "background": "#141018",
        "surface": "#1c1620",
        "panel": "#241e2c",
        "warning": "#f0c480",
        "error": "#e07090",
        "success": "#90c0b0",
        "boost": "#2a2234",
        "variables": {
            "selene-muted": "#7a6a80",
            "selene-faint": "#524858",
            "selene-text-soft": "#c8b8c8",
            "selene-border": "#3a3048",
            "selene-border-soft": "#2a2234",
            "selene-border-focus": "#d4a0c0",
            "selene-elevated": "#2a2234",
            "selene-select-fg": "#141018",
            "selene-select-bg": "#d4a0c0",
            "selene-content": "#241e2c",
            "selene-user": "#9db4d4",
        },
    },
    "bergen": {
        "label": "Bergen — fjord frost",
        "dark": True,
        "primary": "#88c0d0",
        "secondary": "#5a7080",
        "accent": "#a3d4e0",
        "foreground": "#e8eef4",
        "background": "#121820",
        "surface": "#1a222c",
        "panel": "#222c38",
        "warning": "#ebcb8b",
        "error": "#bf7078",
        "success": "#8fbc8f",
        "boost": "#283440",
        "variables": {
            "selene-muted": "#6a8090",
            "selene-faint": "#445060",
            "selene-text-soft": "#c0d0d8",
            "selene-border": "#304050",
            "selene-border-soft": "#283440",
            "selene-border-focus": "#88c0d0",
            "selene-elevated": "#283440",
            "selene-select-fg": "#121820",
            "selene-select-bg": "#88c0d0",
            "selene-content": "#222c38",
            "selene-user": "#d0b090",
        },
    },
    "marrakech": {
        "label": "Marrakech — terracotta sunset",
        "dark": True,
        "primary": "#e08860",
        "secondary": "#8a6050",
        "accent": "#f0b090",
        "foreground": "#f8ece4",
        "background": "#140e0c",
        "surface": "#1c1410",
        "panel": "#261c16",
        "warning": "#e0af68",
        "error": "#d06060",
        "success": "#90a860",
        "boost": "#2e2018",
        "variables": {
            "selene-muted": "#8a6050",
            "selene-faint": "#5c4038",
            "selene-text-soft": "#e0c8b8",
            "selene-border": "#3a2820",
            "selene-border-soft": "#2e2018",
            "selene-border-focus": "#e08860",
            "selene-elevated": "#2e2018",
            "selene-select-fg": "#140e0c",
            "selene-select-bg": "#e08860",
            "selene-content": "#261c16",
            "selene-user": "#8fc0a8",
        },
    },
    "shanghai": {
        "label": "Shanghai — neon night market",
        "dark": True,
        "primary": "#e060c0",
        "secondary": "#7060a0",
        "accent": "#60d0f0",
        "foreground": "#f0e8f8",
        "background": "#0a0814",
        "surface": "#12101c",
        "panel": "#1a1628",
        "warning": "#f0c060",
        "error": "#f06080",
        "success": "#50d0a0",
        "boost": "#221c34",
        "variables": {
            "selene-muted": "#7060a0",
            "selene-faint": "#484060",
            "selene-text-soft": "#c8b8e0",
            "selene-border": "#302848",
            "selene-border-soft": "#221c34",
            "selene-border-focus": "#e060c0",
            "selene-elevated": "#221c34",
            "selene-select-fg": "#0a0814",
            "selene-select-bg": "#e060c0",
            "selene-content": "#1a1628",
            "selene-user": "#e060c0",
        },
    },
    "reykjavik": {
        "label": "Reykjavik — aurora ice",
        "dark": True,
        "primary": "#60e0b8",
        "secondary": "#508080",
        "accent": "#a080e0",
        "foreground": "#e0f4f0",
        "background": "#080e12",
        "surface": "#0e161c",
        "panel": "#141e26",
        "warning": "#e0c070",
        "error": "#e07080",
        "success": "#60e0b8",
        "boost": "#1a2830",
        "variables": {
            "selene-muted": "#508080",
            "selene-faint": "#385058",
            "selene-text-soft": "#b0d8d0",
            "selene-border": "#243840",
            "selene-border-soft": "#1a2830",
            "selene-border-focus": "#60e0b8",
            "selene-elevated": "#1a2830",
            "selene-select-fg": "#080e12",
            "selene-select-bg": "#60e0b8",
            "selene-content": "#141e26",
            "selene-user": "#60e0b8",
        },
    },
    "venice": {
        "label": "Venice — lagoon teal & rose",
        "dark": True,
        "primary": "#50b0b8",
        "secondary": "#608088",
        "accent": "#e0a0a8",
        "foreground": "#e8f0f0",
        "background": "#0c1214",
        "surface": "#141c1e",
        "panel": "#1c2628",
        "warning": "#e0c080",
        "error": "#d07070",
        "success": "#70b890",
        "boost": "#243032",
        "variables": {
            "selene-muted": "#608088",
            "selene-faint": "#405058",
            "selene-text-soft": "#b8d0d0",
            "selene-border": "#2c3c40",
            "selene-border-soft": "#243032",
            "selene-border-focus": "#50b0b8",
            "selene-elevated": "#243032",
            "selene-select-fg": "#0c1214",
            "selene-select-bg": "#50b0b8",
            "selene-content": "#1c2628",
            "selene-user": "#50b0b8",
        },
    },
    "seoul": {
        "label": "Seoul — electric violet night",
        "dark": True,
        "primary": "#a070f0",
        "secondary": "#6860a0",
        "accent": "#c0a0ff",
        "foreground": "#ece8f8",
        "background": "#0c0a14",
        "surface": "#14101e",
        "panel": "#1c182a",
        "warning": "#e0b070",
        "error": "#f07090",
        "success": "#70d0b0",
        "boost": "#242036",
        "variables": {
            "selene-muted": "#6860a0",
            "selene-faint": "#484060",
            "selene-text-soft": "#c0b8e0",
            "selene-border": "#302848",
            "selene-border-soft": "#242036",
            "selene-border-focus": "#a070f0",
            "selene-elevated": "#242036",
            "selene-select-fg": "#0c0a14",
            "selene-select-bg": "#a070f0",
            "selene-content": "#1c182a",
            "selene-user": "#70d0b0",
        },
    },
    "santorini": {
        "label": "Santorini — aegean blue & white",
        "dark": True,
        "primary": "#60b0e8",
        "secondary": "#6088a8",
        "accent": "#e8f0f8",
        "foreground": "#f0f6fa",
        "background": "#0a1018",
        "surface": "#121a24",
        "panel": "#1a2430",
        "warning": "#e0c070",
        "error": "#e08080",
        "success": "#70c0a0",
        "boost": "#223040",
        "variables": {
            "selene-muted": "#6088a8",
            "selene-faint": "#405868",
            "selene-text-soft": "#c0d8e8",
            "selene-border": "#2a3c50",
            "selene-border-soft": "#223040",
            "selene-border-focus": "#60b0e8",
            "selene-elevated": "#223040",
            "selene-select-fg": "#0a1018",
            "selene-select-bg": "#60b0e8",
            "selene-content": "#1a2430",
            "selene-user": "#60b0e8",
        },
    },
    "havana": {
        "label": "Havana — tropical coral & mint",
        "dark": True,
        "primary": "#e07070",
        "secondary": "#887060",
        "accent": "#70d0b0",
        "foreground": "#f8ece8",
        "background": "#120c0c",
        "surface": "#1a1212",
        "panel": "#241a18",
        "warning": "#e0b060",
        "error": "#e05050",
        "success": "#70d0b0",
        "boost": "#2c201c",
        "variables": {
            "selene-muted": "#887060",
            "selene-faint": "#584840",
            "selene-text-soft": "#e0c8c0",
            "selene-border": "#3c2c28",
            "selene-border-soft": "#2c201c",
            "selene-border-focus": "#e07070",
            "selene-elevated": "#2c201c",
            "selene-select-fg": "#120c0c",
            "selene-select-bg": "#e07070",
            "selene-content": "#241a18",
            "selene-user": "#e07070",
        },
    },
}

# Old catalog keys + friendly words → current place themes.
_THEME_ALIASES: dict[str, str] = {
    # Previous catalog (removed / renamed)
    "default": "oslo",
    "midnight": "tokyo",
    "nord": "bergen",
    "rose": "kyoto",
    "ember": "marrakech",
    "forest": "amazon",
    "slate": "oslo",
    "paper": "cairo",
    # Friendly words
    "grey": "oslo",
    "gray": "oslo",
    "white": "oslo",
    "dark": "oslo",
    "mono": "oslo",
    "monochrome": "oslo",
    "blue": "tokyo",
    "neon": "tokyo",
    "tokyonight": "tokyo",
    "tokyo-night": "tokyo",
    "royal": "rome",
    "gold": "rome",
    "forest-green": "amazon",
    "sand": "cairo",
    "desert": "cairo",
    "sakura": "kyoto",
    "rosepine": "kyoto",
    "rose-pine": "kyoto",
    "frost": "bergen",
    "fjord": "bergen",
    "warm": "marrakech",
    "sunset": "marrakech",
    "terracotta": "marrakech",
    "light": "cairo",
    # New places
    "neon": "shanghai",
    "cyber": "shanghai",
    "aurora": "reykjavik",
    "ice": "reykjavik",
    "lagoon": "venice",
    "violet": "seoul",
    "purple": "seoul",
    "aegean": "santorini",
    "tropical": "havana",
    "coral": "havana",
    # Ctrl+P / labels with default marker
    "oslo-(default)": "oslo",
    "oslo(default)": "oslo",
}


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    raw = str(value or "").strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(character * 2 for character in raw)
    if len(raw) != 6:
        return (0, 0, 0)
    try:
        return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))
    except ValueError:
        return (0, 0, 0)


def _mix(base: str, tint: str, amount: float) -> str:
    """Blend ``tint`` into ``base`` by ``amount`` (0..1) and return ``#rrggbb``."""
    ratio = min(1.0, max(0.0, float(amount)))
    base_rgb = _hex_to_rgb(base)
    tint_rgb = _hex_to_rgb(tint)
    blended = tuple(
        int(round(base_channel + (tint_channel - base_channel) * ratio))
        for base_channel, tint_channel in zip(base_rgb, tint_rgb)
    )
    return "#{:02x}{:02x}{:02x}".format(*blended)


def _role_variables(data: dict[str, Any]) -> dict[str, str]:
    """Transcript role colors (prompt vs response) derived from a theme.

    ``selene-user`` is authored per place; everything else is blended from it so
    a new theme only has to pick one extra hue and still gets matching block
    tints, bars, and labels.
    """
    variables = dict(data.get("variables") or {})
    content = str(variables.get("selene-content") or data["panel"])
    user = str(variables.get("selene-user") or data["secondary"])
    assistant = str(data["accent"])
    return {
        "selene-user": user,
        "selene-user-soft": _mix(user, str(data["foreground"]), 0.35),
        "selene-user-bg": _mix(content, user, 0.16),
        "selene-assistant": assistant,
        "selene-assistant-bg": _mix(content, assistant, 0.05),
    }


def theme_names() -> tuple[str, ...]:
    return THEME_ORDER


def theme_label(name: str) -> str:
    data = _THEME_DEFS.get(normalize_theme_name(name), _THEME_DEFS[DEFAULT_THEME])
    return str(data.get("label") or name)


def normalize_theme_name(name: str | None) -> str:
    """Return a canonical lowercase place-theme key (always valid)."""
    raw = str(name or DEFAULT_THEME).strip()
    # Ctrl+P may hand back "Oslo (default)".
    cleaned = raw
    for token in ("(default)", "(Default)", "(DEFAULT)"):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.strip()
    key = cleaned.lower().replace("_", "-").replace(" ", "-")
    if key in _THEME_DEFS:
        return key
    if key in _THEME_ALIASES:
        return _THEME_ALIASES[key]
    # Title Case place names from Ctrl+P ("Tokyo" → tokyo)
    lower = cleaned.lower()
    if lower in _THEME_DEFS:
        return lower
    return DEFAULT_THEME


def is_valid_theme(name: str | None) -> bool:
    raw = str(name or "").strip()
    if not raw:
        return False
    # Accept anything that normalizes to a known place (incl. "Oslo (default)").
    key = normalize_theme_name(raw)
    # Reject garbage that silently fell back to default unless it *is* the default.
    cleaned = raw
    for token in ("(default)", "(Default)", "(DEFAULT)"):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.strip().lower().replace("_", "-").replace(" ", "-")
    if cleaned in _THEME_DEFS or cleaned in _THEME_ALIASES:
        return True
    if cleaned == DEFAULT_THEME or cleaned in ("oslo", "default", "grey", "gray"):
        return True
    # Known title-case place only
    return cleaned in _THEME_DEFS


def theme_catalog() -> tuple[tuple[str, str], ...]:
    """``(name, label)`` pairs in display order (default place first)."""
    return tuple((name, theme_label(name)) for name in THEME_ORDER)


def textual_theme_name(name: str | None = None) -> str:
    """Title Case place name used when registering / selecting Textual themes."""
    return _display_name(normalize_theme_name(name))


def rich_palette(name: str | None = None) -> dict[str, str]:
    """Colors for Rich ``Text`` spans (slash menu, thinking chrome, etc.)."""
    key = normalize_theme_name(name)
    data = _THEME_DEFS[key]
    variables = dict(data.get("variables") or {})
    roles = _role_variables(data)
    return {
        "bg": str(data["background"]),
        "surface": str(data["surface"]),
        "elevated": str(variables.get("selene-elevated", data.get("boost", data["surface"]))),
        "content": str(variables.get("selene-content", data["panel"])),
        "border": str(variables.get("selene-border", "#333333")),
        "border_soft": str(variables.get("selene-border-soft", "#2a2a2a")),
        "border_focus": str(variables.get("selene-border-focus", data["primary"])),
        "text": str(data["foreground"]),
        "text_soft": str(variables.get("selene-text-soft", data["foreground"])),
        "muted": str(variables.get("selene-muted", data["secondary"])),
        "faint": str(variables.get("selene-faint", data["secondary"])),
        "accent": str(data["accent"]),
        "primary": str(data["primary"]),
        "select_fg": str(variables.get("selene-select-fg", data["background"])),
        "select_bg": str(variables.get("selene-select-bg", data["primary"])),
        "warning": str(data["warning"]),
        "error": str(data["error"]),
        "success": str(data["success"]),
        # Transcript roles — prompts and responses never share a color.
        "user": roles["selene-user"],
        "user_soft": roles["selene-user-soft"],
        "user_bg": roles["selene-user-bg"],
        "assistant": roles["selene-assistant"],
        "assistant_bg": roles["selene-assistant-bg"],
        "dark": "1" if data.get("dark", True) else "0",
    }


def build_textual_theme(name: str | None = None) -> Any:
    """Construct a Textual ``Theme`` for ``App.register_theme``.

    The registered ``Theme.name`` is the Title Case place name so the
    Ctrl+P → Themes list shows ``Oslo``, ``Tokyo``, etc.
    """
    from textual.theme import Theme

    key = normalize_theme_name(name)
    data = _THEME_DEFS[key]
    variables = {**dict(data.get("variables") or {}), **_role_variables(data)}
    return Theme(
        name=_display_name(key),
        primary=str(data["primary"]),
        secondary=str(data["secondary"]),
        accent=str(data["accent"]),
        foreground=str(data["foreground"]),
        background=str(data["background"]),
        surface=str(data["surface"]),
        panel=str(data["panel"]),
        warning=str(data["warning"]),
        error=str(data["error"]),
        success=str(data["success"]),
        boost=str(data.get("boost") or data["surface"]),
        dark=bool(data.get("dark", True)),
        variables=variables,
    )


def place_theme_display_names() -> frozenset[str]:
    """Title Case place names as registered with Textual (incl. default marker)."""
    return frozenset(_display_name(name) for name in THEME_ORDER)


def register_all_themes(app: Any) -> None:
    """Register only Selene place themes; drop Textual stock themes from the picker.

    Textual pre-registers catppuccin, nord, dracula, etc. Those do not follow
    our place-naming schedule, so they are unregistered after our themes are
    installed. Ctrl+P → Themes then lists only places (Oslo, Tokyo, …).
    """
    place_display = place_theme_display_names()
    for name in THEME_ORDER:
        try:
            app.register_theme(build_textual_theme(name))
        except Exception:
            # Older Textual / duplicate registration — ignore.
            pass

    # Leave the stock active theme before unregistering it.
    try:
        app.theme = _display_name(DEFAULT_THEME)
    except Exception:
        pass

    try:
        registered = list(getattr(app, "available_themes", {}) or {})
    except Exception:
        registered = []
    for theme_name in registered:
        if theme_name in place_display:
            continue
        try:
            app.unregister_theme(theme_name)
        except Exception:
            pass


def theme_specs_for_slash() -> tuple[tuple[str, str], ...]:
    """Slash palette entries: ``(/theme name, description)`` with default first.

    Oslo's label already includes ``(default)`` next to the place name.
    """
    rows: list[tuple[str, str]] = [("/theme", "TUI color theme  ·  /theme <place>")]
    for name, label in theme_catalog():
        rows.append((f"/theme {name}", label))
    return tuple(rows)
