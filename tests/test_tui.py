"""Tests for the full-screen Selene TUI and display-sink routing."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agent import terminal
from agent.core import _should_use_tui, process_user_turn
from agent.tui import _filter_slash_commands


class DisplaySinkRoutingTests(unittest.TestCase):
    def tearDown(self) -> None:
        terminal.set_display_sink(None)

    def test_print_helpers_route_to_tui_sink(self):
        calls: list[tuple] = []

        class FakeSink:
            is_tui = True

            def lab_status(self, message, *, kind="info", detail=None):
                calls.append(("status", message, kind, detail))

            def activity_start(self, label="Thinking"):
                calls.append(("activity_start", label))

            def activity_update(self, label):
                calls.append(("activity_update", label))

            def activity_stop(self):
                calls.append(("activity_stop",))

            def thinking_header(self):
                calls.append(("thinking_header",))

            def thinking_delta(self, text):
                calls.append(("thinking_delta", text))

            def thinking_footer(self, label=None):
                calls.append(("thinking_footer", label))

            def content_stream(self, text):
                calls.append(("content_stream", text))

            def content_final(self, text):
                calls.append(("content_final", text))

            def generation_stats(self, *, elapsed, total_tokens, tokens_per_sec):
                calls.append(("stats", elapsed, total_tokens, tokens_per_sec))

            def command_help(self, entries, *, title="commands", subtitle=None):
                calls.append(("help", title, list(entries)))

        sink = FakeSink()
        terminal.set_display_sink(sink)
        self.assertTrue(terminal.display_is_tui())

        terminal.print_ok("saved", detail="demo")
        # Spinner path in TUI should hit activity_* not permanent status spam.
        spinner = terminal._Spinner("Thinking")
        spinner.start()
        spinner.update("Still thinking")
        spinner.stop()
        terminal.print_thinking_header()
        terminal.print_thinking_delta("step")
        terminal.print_thinking_footer()
        terminal.print_content_stream("hello")
        terminal.print_assistant_message("hello")
        terminal.print_generation_stats(elapsed=1.0, total_tokens=2, tokens_per_sec=2.0)
        terminal.print_command_help([("/help", "Show help")], title="commands")

        kinds = [item[0] for item in calls]
        self.assertIn("status", kinds)
        self.assertIn("activity_start", kinds)
        self.assertIn("activity_update", kinds)
        self.assertIn("activity_stop", kinds)
        self.assertIn("thinking_header", kinds)
        self.assertIn("thinking_delta", kinds)
        self.assertIn("content_stream", kinds)
        self.assertIn("content_final", kinds)
        self.assertIn("help", kinds)


class TuiSelectionTests(unittest.TestCase):
    def test_classic_flag_disables_tui(self):
        with patch("sys.argv", ["main.py", "--cli", "--classic"]):
            with patch("sys.stdin") as stdin, patch("sys.stdout") as stdout:
                stdin.isatty.return_value = True
                stdout.isatty.return_value = True
                self.assertFalse(_should_use_tui())

    def test_non_tty_disables_tui(self):
        with patch("sys.argv", ["main.py", "--cli"]):
            with patch("sys.stdin") as stdin, patch("sys.stdout") as stdout:
                stdin.isatty.return_value = False
                stdout.isatty.return_value = True
                self.assertFalse(_should_use_tui())


class TuiAppSmokeTests(unittest.TestCase):
    def test_build_app_class_and_compose(self):
        try:
            import textual  # noqa: F401
        except ImportError:
            self.skipTest("textual not installed")

        from agent.tui import build_app_class

        AppCls = build_app_class()
        app = AppCls(
            session={"history": True, "system": "", "options": {}, "verbose": False,
                     "wordwrap": True, "format": "", "think": True, "runtime_profile": "auto"},
            history=[],
            default_system_prompt="sys",
            process_turn=lambda *a, **k: None,
            handle_command=lambda *a, **k: True,
            slash_completions=("/help", "/quit"),
            slash_descriptions={"/help": "Show help", "/quit": "Exit"},
            status_meta={"profile": "low-vram", "model": "selene"},
        )
        # Headless compose smoke — does not open a real alternate screen.
        async def _run():
            async with app.run_test() as pilot:
                self.assertIsNotNone(app.query_one("#prompt-input"))
                self.assertIsNotNone(app.query_one("#chat"))
                self.assertIsNotNone(app.query_one("#slash-palette"))
                await pilot.click("#prompt-input")
                await pilot.press("/")
                # Palette updates on Input.Changed; give the app a tick.
                await pilot.pause()
                palette = app.query_one("#slash-palette")
                self.assertTrue(palette.has_class("-visible"))

        import asyncio

        asyncio.run(_run())


class ProcessTurnImportTests(unittest.TestCase):
    def test_process_user_turn_is_callable(self):
        self.assertTrue(callable(process_user_turn))


class SlashFilterTests(unittest.TestCase):
    COMMANDS = (
        "/help",
        "/clear",
        "/set parameter",
        "/set profile",
        "/vault list",
        "/vault search",
        "/quit",
    )
    DESCRIPTIONS = {
        "/help": "Show help",
        "/clear": "Clear chat",
        "/set parameter": "Set parameter",
        "/set profile": "Select profile",
        "/vault list": "List vaults",
        "/vault search": "Search vault",
        "/quit": "Exit",
    }

    def test_bare_slash_lists_catalog(self):
        matches = _filter_slash_commands("/", self.COMMANDS, self.DESCRIPTIONS, limit=5)
        self.assertEqual(len(matches), 5)
        self.assertEqual(matches[0][0], "/help")

    def test_prefix_ranks_before_substring(self):
        matches = _filter_slash_commands("/set", self.COMMANDS, self.DESCRIPTIONS)
        self.assertTrue(all(m[0].startswith("/set") for m in matches[:2]))
        self.assertEqual(matches[0][0], "/set parameter")

    def test_substring_matches_inner_token(self):
        matches = _filter_slash_commands("/vault", self.COMMANDS, self.DESCRIPTIONS)
        cmds = [m[0] for m in matches]
        self.assertIn("/vault list", cmds)
        self.assertIn("/vault search", cmds)


if __name__ == "__main__":
    unittest.main()
