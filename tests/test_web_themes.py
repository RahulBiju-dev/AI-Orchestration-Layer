from __future__ import annotations

import re
import unittest
from pathlib import Path

from agent.tui_themes import rich_palette, theme_names


ROOT = Path(__file__).resolve().parents[1]
STYLE = (ROOT / "agent" / "static" / "style.css").read_text(encoding="utf-8")
APP = (ROOT / "agent" / "static" / "app.js").read_text(encoding="utf-8")
HTML = (ROOT / "agent" / "static" / "index.html").read_text(encoding="utf-8")


def _theme_block(name: str) -> str:
    selector = r":root" if name == "oslo" else rf':root\[data-theme="{re.escape(name)}"\]'
    match = re.search(rf"{selector}\s*\{{(?P<body>.*?)\n\}}", STYLE, re.DOTALL)
    if not match:
        raise AssertionError(f"Missing CSS block for {name}")
    return match.group("body")


def _custom_properties(block: str) -> dict[str, str]:
    return {
        name: value.strip().lower()
        for name, value in re.findall(r"--([\w-]+)\s*:\s*([^;]+);", block)
    }


class WebThemeTests(unittest.TestCase):
    def test_startup_profile_chooser_defaults_to_manual_and_is_wired(self) -> None:
        self.assertRegex(
            APP,
            re.compile(
                r'const DEFAULT_SETTINGS\s*=\s*\{.*?runtime_profile:\s*"manual"',
                re.DOTALL,
            ),
        )
        self.assertIn('id="profile-dialog"', HTML)
        self.assertIn('id="profile-choice"', HTML)
        self.assertIn('id="profile-apply"', HTML)
        self.assertIn('id="setting-profile"', HTML)
        self.assertIn("function showStartupProfileDialog()", APP)
        self.assertIn("function applyStartupProfile()", APP)
        self.assertIn("state.settings.runtime_profile = selected", APP)
        self.assertRegex(STYLE, r"\.profile-backdrop\[hidden\]\s*\{\s*display:\s*none;")

    def test_every_tui_place_theme_is_available_on_web(self) -> None:
        for name in theme_names():
            with self.subTest(theme=name):
                _theme_block(name)
                self.assertIn(f'id: "{name}"', APP)

    def test_web_theme_tokens_match_tui_palettes(self) -> None:
        token_map = {
            "bg": "bg",
            "surface": "surface",
            "panel": "content",
            "elevated": "elevated",
            "primary": "primary",
            "line": "border_soft",
            "line-strong": "border",
            "line-focus": "border_focus",
            "text": "text",
            "text-soft": "text_soft",
            "muted": "muted",
            "faint": "faint",
            "accent": "accent",
            "danger": "error",
            "warning": "warning",
            "success": "success",
            "select-fg": "select_fg",
        }
        for name in theme_names():
            css = _custom_properties(_theme_block(name))
            tui = rich_palette(name)
            for css_name, tui_name in token_map.items():
                with self.subTest(theme=name, token=css_name):
                    self.assertEqual(css.get(css_name), tui[tui_name].lower())

    def test_theme_overrides_are_color_tokens_only(self) -> None:
        for name in theme_names()[1:]:
            with self.subTest(theme=name):
                block = _theme_block(name)
                declarations = [line.strip() for line in block.splitlines() if line.strip()]
                self.assertTrue(declarations)
                self.assertTrue(all(line.startswith("--") for line in declarations))

    def test_theme_previews_use_exact_tui_palette_colors(self) -> None:
        for name in theme_names():
            match = re.search(
                rf'\{{ id: "{re.escape(name)}".*?background: "(?P<background>#[0-9a-f]+)", '
                rf'surface: "(?P<surface>#[0-9a-f]+)", primary: "(?P<primary>#[0-9a-f]+)", '
                rf'accent: "(?P<accent>#[0-9a-f]+)" \}}',
                APP,
                re.IGNORECASE,
            )
            with self.subTest(theme=name):
                self.assertIsNotNone(match)
                preview = {key: value.lower() for key, value in match.groupdict().items()}
                tui = rich_palette(name)
                self.assertEqual(preview["background"], tui["bg"].lower())
                self.assertEqual(preview["surface"], tui["surface"].lower())
                self.assertEqual(preview["primary"], tui["primary"].lower())
                self.assertEqual(preview["accent"], tui["accent"].lower())

    def test_theme_command_and_centered_picker_are_wired(self) -> None:
        self.assertIn('{ command: "/theme"', APP)
        self.assertIn('id="theme-btn"', HTML)
        self.assertIn('id="theme-dialog"', HTML)
        self.assertIn('id="theme-options"', HTML)
        self.assertIn('class="theme-backdrop"', HTML)
        self.assertRegex(STYLE, r"\.theme-backdrop\[hidden\]\s*\{\s*display:\s*none;")
        self.assertRegex(STYLE, r"\.theme-backdrop\.open\s*\{[^}]*pointer-events:\s*auto;")

    def test_theme_picker_has_grid_navigation_and_focus_trapping(self) -> None:
        self.assertIn("function themeGridColumns()", APP)
        self.assertIn("function cycleThemeDialogFocus(event)", APP)
        self.assertIn("function handleThemeDialogKeydown(event)", APP)
        for key in ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "Tab"):
            with self.subTest(key=key):
                self.assertIn(key, APP)


if __name__ == "__main__":
    unittest.main()
