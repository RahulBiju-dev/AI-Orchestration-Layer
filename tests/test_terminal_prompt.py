"""CLI prompt chrome and cover-art tests."""

from __future__ import annotations

import unittest
import sys
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent import terminal


class PromptChromeTests(unittest.TestCase):
    def test_prompt_plain_matches_visible_chrome(self):
        self.assertEqual(terminal.PROMPT_PLAIN, f"{terminal.GLYPH_MARK} selene {terminal.GLYPH_PROMPT} ")
        self.assertIn("selene", terminal.PROMPT_MARKUP)
        self.assertIn(terminal.GLYPH_MARK, terminal.PROMPT_MARKUP)
        self.assertIn(terminal.GLYPH_PROMPT, terminal.PROMPT_MARKUP)

    def test_backspace_never_removes_fixed_prompt_from_buffer(self):
        """The editable buffer is user text only; chrome is not part of it."""
        buffer: list[str] = list("hi")
        cursor = len(buffer)

        def backspace() -> None:
            nonlocal cursor
            if cursor <= 0:
                return
            cursor -= 1
            del buffer[cursor]

        backspace()
        backspace()
        backspace()
        backspace()
        self.assertEqual(buffer, [])
        self.assertEqual(cursor, 0)
        self.assertTrue(terminal.PROMPT_PLAIN.startswith(terminal.GLYPH_MARK))

    def test_read_user_input_does_not_use_rich_bare_input(self):
        """Regression: Rich Console.input leaves chrome erasable."""
        with patch.object(terminal, "_read_line_with_fixed_prompt", return_value="hello"):
            with patch.object(terminal, "flush_terminal_input"):
                value = terminal.read_user_input()
        self.assertEqual(value, "hello")


class SlashCompletionTests(unittest.TestCase):
    COMMANDS = (
        "/help",
        "/save",
        "/set parameter",
        "/set profile",
        "/show model",
        "/vault list",
    )

    def test_unique_prefix_autofills_command(self):
        state = terminal._SlashCompletionState(self.COMMANDS)
        self.assertEqual(state.complete("/he"), "/help")

    def test_parent_command_autofills_subcommand_boundary(self):
        state = terminal._SlashCompletionState(self.COMMANDS)
        self.assertEqual(state.complete("/vault"), "/vault list")

    def test_ambiguous_prefix_cycles_matches(self):
        state = terminal._SlashCompletionState(self.COMMANDS)
        self.assertEqual(state.complete("/s"), "/save")
        self.assertEqual(state.complete("/save"), "/set parameter")

    def test_menu_selection_controls_autofill(self):
        state = terminal._SlashCompletionState(self.COMMANDS)
        self.assertEqual(
            state.complete("/s", preferred="/show model"),
            "/show model",
        )

    def test_non_command_text_is_unchanged(self):
        state = terminal._SlashCompletionState(self.COMMANDS)
        self.assertEqual(state.complete("hello"), "hello")


class SlashMenuTests(unittest.TestCase):
    COMMANDS = (
        "/help",
        "/save",
        "/set parameter",
        "/set profile",
        "/show model",
        "/vault list",
    )

    def test_slash_opens_menu_and_text_filters_it(self):
        state = terminal._SlashMenuState(self.COMMANDS)
        state.update("/")
        self.assertEqual(state.matches, self.COMMANDS)

        state.update("/set")
        self.assertEqual(state.matches, ("/set parameter", "/set profile"))
        self.assertEqual(state.selected_command(), "/set parameter")

        state.update("hello")
        self.assertEqual(state.matches, ())

    def test_arrow_selection_wraps_and_visible_window_tracks_it(self):
        state = terminal._SlashMenuState(self.COMMANDS, max_visible=3)
        state.update("/")
        state.move(-1)
        self.assertEqual(state.selected_command(), "/vault list")
        visible = state.visible_matches()
        self.assertEqual(len(visible), 3)
        self.assertEqual(visible[-1][1], "/vault list")

    def test_menu_renderer_is_bounded_and_includes_keyboard_hint(self):
        state = terminal._SlashMenuState(self.COMMANDS, max_visible=2)
        state.update("/")
        lines = terminal._slash_menu_lines(state)
        self.assertEqual(len(lines), 3)
        self.assertIn("/help", lines[0])
        self.assertIn("Tab autofill", lines[-1])

    def test_posix_down_arrow_then_tab_autofills_highlighted_command(self):
        class FakeTTY(StringIO):
            def fileno(self):
                return 99

            def isatty(self):
                return True

        source = FakeTTY("/s\x1b[B\t\r")
        output = StringIO()
        with (
            patch.object(terminal.sys, "stdin", source),
            patch.object(
                terminal,
                "_console",
                terminal.Console(file=output, width=80, force_terminal=True),
            ),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr"),
            patch("tty.setraw"),
        ):
            value = terminal._read_line_protected_tty(self.COMMANDS)

        self.assertEqual(value, "/set parameter")
        self.assertIn("Tab autofill", output.getvalue())

    def test_windows_arrow_key_is_routed_to_menu_selection(self):
        keys = iter(("\xe0", "P", "\t", "\r"))
        fake_msvcrt = SimpleNamespace(getwch=lambda: next(keys))
        move_menu = Mock(return_value=True)
        clear_menu = Mock()
        with patch.dict(sys.modules, {"msvcrt": fake_msvcrt}):
            value = terminal._read_line_protected_windows(
                [],
                write=Mock(),
                insert=Mock(),
                backspace=Mock(),
                delete_forward=Mock(),
                move_left=Mock(),
                move_right=Mock(),
                move_home=Mock(),
                move_end=Mock(),
                move_menu=move_menu,
                clear_menu=clear_menu,
                complete=Mock(),
            )

        self.assertEqual(value, "")
        move_menu.assert_called_once_with(1)
        clear_menu.assert_called_once_with()


class CoverArtTests(unittest.TestCase):
    def test_wide_art_contains_moon_and_name(self):
        lines = terminal._welcome_art_lines(80)
        text = "\n".join(row[0] for row in lines)
        self.assertIn("888", text)

    def test_narrow_art_fits_and_names_product(self):
        lines = terminal._welcome_art_lines(40)
        text = "\n".join(row[0] for row in lines)
        self.assertIn("S E L E N E", text)
        self.assertTrue(all(len(row[0]) <= 40 for row in lines))

    def test_print_welcome_header_renders_without_error(self):
        sink = StringIO()
        with patch.object(terminal, "_console", terminal.Console(file=sink, width=80, force_terminal=True)):
            terminal.print_welcome_header(
                {
                    "profile": "low-vram",
                    "model": "selene",
                    "num_ctx": "4096",
                    "num_predict": "768",
                    "platform": "linux",
                }
            )
        output = sink.getvalue()
        self.assertIn("SELENE", output.replace(" ", ""))
        self.assertIn("low-vram", output)
        self.assertIn("selene", output.casefold())

    def test_assistant_panel_uses_lab_title(self):
        panel = terminal.assistant_stream_panel("hello")
        self.assertIn("response", str(panel.title))

    def test_tool_event_formats_phases(self):
        sink = StringIO()
        with patch.object(terminal, "_console", terminal.Console(file=sink, width=80, force_terminal=True)):
            terminal.print_tool_event("web_search", phase="run", detail="latest news")
            terminal.print_tool_event("web_search", phase="ok")
            terminal.print_tool_event("web_search", phase="error", detail="timeout")
            terminal.print_tool_event("batch", phase="parallel", detail="3")
        output = sink.getvalue()
        self.assertIn("web_search", output)
        self.assertIn("complete", output)


if __name__ == "__main__":
    unittest.main()
