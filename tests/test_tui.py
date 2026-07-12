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

    def test_clear_command_does_not_duplicate_welcome_id(self):
        try:
            import textual  # noqa: F401
        except ImportError:
            self.skipTest("textual not installed")

        import asyncio

        from agent.tui import build_app_class

        AppCls = build_app_class()
        history = [{"role": "user", "content": "hello"}]
        session = {
            "history": True,
            "system": "sys",
            "options": {},
            "verbose": False,
            "wordwrap": True,
            "format": "",
            "think": True,
            "runtime_profile": "manual",
        }
        app = AppCls(
            session=session,
            history=history,
            default_system_prompt="sys",
            process_turn=lambda *a, **k: None,
            handle_command=lambda *a, **k: True,
            slash_completions=("/clear", "/help"),
            slash_descriptions={"/clear": "Clear", "/help": "Help"},
            status_meta={"profile": "manual", "model": "selene"},
        )

        async def _run():
            async with app.run_test() as pilot:
                app.ui_add_user("hello")
                app.ui_status("noise", kind="info")
                await pilot.pause()
                # Must not raise DuplicateIds / NoMatches.
                app._clear_conversation_ui()
                await pilot.pause()
                welcomes = list(app.query("#welcome"))
                self.assertEqual(len(welcomes), 1)
                # Second clear still safe.
                app._clear_conversation_ui()
                await pilot.pause()
                self.assertEqual(len(list(app.query("#welcome"))), 1)

        asyncio.run(_run())

    def test_ctrl_c_stops_then_quits_on_second_press(self):
        try:
            import textual  # noqa: F401
        except ImportError:
            self.skipTest("textual not installed")

        import asyncio
        import time

        from agent import core as core_mod
        from agent.tui import build_app_class

        AppCls = build_app_class()
        app = AppCls(
            session={
                "history": True,
                "system": "",
                "options": {},
                "verbose": False,
                "wordwrap": True,
                "format": "",
                "think": True,
                "runtime_profile": "manual",
            },
            history=[],
            default_system_prompt="sys",
            process_turn=lambda *a, **k: None,
            handle_command=lambda *a, **k: True,
            slash_completions=("/help",),
            slash_descriptions={"/help": "Help"},
            status_meta={"profile": "manual"},
        )

        async def _run():
            async with app.run_test() as pilot:
                core_mod._interrupted = False
                app._busy = True
                app.action_interrupt_or_quit()
                self.assertTrue(core_mod.generation_interrupt_requested())
                self.assertGreater(app._quit_armed_until, time.monotonic())

                app._busy = False
                # Second press while armed should exit the app.
                app.action_interrupt_or_quit()
                await pilot.pause()
                self.assertTrue(app.return_code is not None or not app.is_running)

        asyncio.run(_run())
        core_mod._interrupted = False

    def test_thinking_fold_is_collapsible_after_stream(self):
        try:
            import textual  # noqa: F401
        except ImportError:
            self.skipTest("textual not installed")

        import asyncio

        from agent.tui import build_app_class

        AppCls = build_app_class()
        app = AppCls(
            session={"history": True, "system": "", "options": {}, "verbose": False,
                     "wordwrap": True, "format": "", "think": True, "runtime_profile": "manual"},
            history=[],
            default_system_prompt="sys",
            process_turn=lambda *a, **k: None,
            handle_command=lambda *a, **k: True,
            slash_completions=("/help",),
            slash_descriptions={"/help": "Show help"},
            status_meta={"profile": "manual", "model": "selene"},
        )

        async def _run():
            async with app.run_test() as pilot:
                app.ui_thinking_start()
                app.ui_thinking_delta("Check the vault, then answer carefully.")
                app.ui_thinking_end()
                await pilot.pause()
                folds = list(app.query("ThinkingFold"))
                self.assertEqual(len(folds), 1)
                fold = folds[0]
                self.assertFalse(fold._expanded)
                self.assertIn("vault", fold._full_text)
                fold.action_toggle()
                self.assertTrue(fold._expanded)
                self.assertTrue(fold.has_class("-expanded"))
                fold.action_toggle()
                self.assertFalse(fold._expanded)

        asyncio.run(_run())


class ProcessTurnImportTests(unittest.TestCase):
    def test_process_user_turn_is_callable(self):
        self.assertTrue(callable(process_user_turn))

    def test_request_generation_interrupt_sets_flag(self):
        from agent import core as core_mod

        core_mod._interrupted = False
        self.assertFalse(core_mod.generation_interrupt_requested())
        core_mod.request_generation_interrupt()
        self.assertTrue(core_mod.generation_interrupt_requested())
        core_mod._interrupted = False


class SlashPaletteRenderTests(unittest.TestCase):
    def test_bracketed_descriptions_do_not_break_selection_highlight(self):
        """Descriptions with [params] must not corrupt Rich markup styles."""
        try:
            import textual  # noqa: F401
        except ImportError:
            self.skipTest("textual not installed")

        import asyncio

        from agent.tui import build_app_class

        AppCls = build_app_class()
        app = AppCls(
            session={
                "history": True,
                "system": "",
                "options": {},
                "verbose": False,
                "wordwrap": True,
                "format": "",
                "think": True,
                "runtime_profile": "manual",
            },
            history=[],
            default_system_prompt="sys",
            process_turn=lambda *a, **k: None,
            handle_command=lambda *a, **k: True,
            slash_completions=("/help", "/save", "/load"),
            slash_descriptions={
                "/help": "Commands and usage",
                "/save": "Save session  ·  /save [name]",
                "/load": "Load session  ·  /load [name|index]",
            },
            status_meta={"profile": "manual"},
        )

        async def _run():
            async with app.run_test() as pilot:
                palette = app.query_one("#slash-palette")
                matches = [
                    ("/help", "Commands and usage"),
                    ("/save", "Save session  ·  /save [name]"),
                    ("/load", "Load session  ·  /load [name|index]"),
                ]
                # Select the middle row (same as the user screenshot case).
                palette.show_matches(matches, selected=1, query="/", total=3)
                await pilot.pause()
                # Content must be a Text object (not a markup string).
                from rich.text import Text as RichText

                renderable = palette.content
                self.assertIsInstance(renderable, RichText)
                plain = renderable.plain
                self.assertIn("/save", plain)
                self.assertIn("[name|index]", plain)
                # Background style spans should only cover the selected row.
                spans = list(renderable.spans)

                def _has_bg(style) -> bool:
                    if style is None:
                        return False
                    if isinstance(style, str):
                        return " on " in style
                    return getattr(style, "bgcolor", None) is not None

                bg_spans = [span for span in spans if _has_bg(span.style)]
                self.assertEqual(len(bg_spans), 1)
                selected_plain = plain[bg_spans[0].start : bg_spans[0].end]
                self.assertIn("/save", selected_plain)
                self.assertNotIn("/load", selected_plain)
                # Ensure bracketed desc of the *next* row is outside the bg span.
                load_at = plain.index("/load")
                self.assertGreaterEqual(load_at, bg_spans[0].end)

        asyncio.run(_run())


class ThemeCatalogTests(unittest.TestCase):
    def test_default_theme_is_first_place(self):
        from agent.tui_themes import (
            DEFAULT_THEME,
            theme_catalog,
            theme_names,
            normalize_theme_name,
            theme_specs_for_slash,
            textual_theme_name,
        )

        catalog = theme_catalog()
        self.assertEqual(len(theme_names()), 14)
        self.assertEqual(catalog[0][0], "oslo")
        self.assertEqual(DEFAULT_THEME, "oslo")
        self.assertIn("oslo", catalog[0][1].casefold())
        self.assertIn("(default)", catalog[0][1].casefold())
        self.assertIn("grey", catalog[0][1].casefold())
        places = {name for name, _ in catalog}
        for place in (
            "oslo", "tokyo", "rome", "amazon", "cairo", "kyoto", "bergen",
            "marrakech", "shanghai", "reykjavik", "venice", "seoul",
            "santorini", "havana",
        ):
            self.assertIn(place, places)
        specs = theme_specs_for_slash()
        names = [cmd for cmd, _ in specs]
        self.assertEqual(names[0], "/theme")
        self.assertEqual(names[1], "/theme oslo")
        self.assertIn("(default)", specs[1][1].casefold())
        self.assertEqual(normalize_theme_name("grey"), "oslo")
        self.assertEqual(normalize_theme_name("default"), "oslo")
        self.assertEqual(normalize_theme_name("Oslo (default)"), "oslo")
        self.assertEqual(normalize_theme_name("tokyo-night"), "tokyo")
        self.assertEqual(normalize_theme_name("nord"), "bergen")
        self.assertEqual(normalize_theme_name("Tokyo"), "tokyo")
        self.assertEqual(textual_theme_name("tokyo"), "Tokyo")
        self.assertEqual(textual_theme_name("oslo"), "Oslo (default)")

    def test_theme_command_applies_in_tui(self):
        try:
            import textual  # noqa: F401
        except ImportError:
            self.skipTest("textual not installed")

        import asyncio

        from agent.tui import build_app_class

        AppCls = build_app_class()
        session = {
            "history": True,
            "system": "",
            "options": {},
            "verbose": True,
            "wordwrap": True,
            "format": "",
            "think": True,
            "runtime_profile": "manual",
            "tui_theme": "oslo",
        }
        app = AppCls(
            session=session,
            history=[],
            default_system_prompt="sys",
            process_turn=lambda *a, **k: None,
            handle_command=lambda *a, **k: True,
            slash_completions=("/theme", "/theme tokyo"),
            slash_descriptions={"/theme": "Theme", "/theme tokyo": "Tokyo"},
            status_meta={"profile": "manual", "model": "selene"},
        )

        async def _run():
            async with app.run_test() as pilot:
                await pilot.pause()
                self.assertEqual(app.theme, "Oslo (default)")
                # Ctrl+P theme list must be place names only (no catppuccin/nord/…).
                from agent.tui_themes import place_theme_display_names

                available = set(app.available_themes.keys())
                self.assertEqual(available, set(place_theme_display_names()))
                self.assertNotIn("nord", available)
                self.assertNotIn("dracula", available)
                self.assertNotIn("catppuccin-mocha", available)
                app.ui_apply_theme("tokyo")
                await pilot.pause()
                self.assertEqual(app.theme, "Tokyo")
                self.assertEqual(session.get("tui_theme"), "tokyo")
                self.assertIsNotNone(app._selene_palette)
                # Glyph is ASCII '>' on the same row as Input text (y=0).
                glyph = app.query_one("#prompt-glyph")
                content = getattr(glyph, "content", None) or getattr(glyph, "_content", ">")
                plain = content.plain if hasattr(content, "plain") else str(content)
                self.assertIn(">", plain)
                cache = getattr(glyph, "_render_cache", None)
                if cache is not None and getattr(cache, "lines", None):
                    first = "".join(seg.text for seg in cache.lines[0])
                    self.assertIn(">", first)

        asyncio.run(_run())


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


class CommandPaletteEscapeTests(unittest.TestCase):
    def test_escape_closes_palette_opened_via_shortcut(self):
        try:
            import textual  # noqa: F401
        except ImportError:
            self.skipTest("textual not installed")

        import asyncio

        from agent.tui import build_app_class

        AppCls = build_app_class()
        session = {
            "history": True,
            "system": "",
            "options": {},
            "verbose": True,
            "wordwrap": True,
            "format": "",
            "think": True,
            "runtime_profile": "manual",
            "tui_theme": "oslo",
        }
        app = AppCls(
            session=session,
            history=[],
            default_system_prompt="sys",
            process_turn=lambda *a, **k: None,
            handle_command=lambda *a, **k: True,
            slash_completions=("/help", "/save", "/theme"),
            slash_descriptions={
                "/help": "Help",
                "/save": "Save",
                "/theme": "Theme",
            },
            status_meta={"profile": "manual", "model": "selene"},
        )

        async def _run():
            async with app.run_test() as pilot:
                await pilot.pause()
                # Open via the same shortcut path as Ctrl+/.
                app.action_open_commands()
                await pilot.pause()
                palette = app.query_one("#slash-palette")
                self.assertTrue(app._slash_open or app._slash_matches)
                self.assertTrue(palette.has_class("-visible"))
                self.assertEqual(app.query_one("#prompt-input").value, "/")

                # Escape must close palette and clear the slash draft.
                await pilot.press("escape")
                await pilot.pause()
                self.assertFalse(app._slash_open)
                self.assertFalse(app._slash_matches)
                self.assertFalse(palette.has_class("-visible"))
                self.assertEqual(app.query_one("#prompt-input").value, "")

                # Re-open, then toggle closed with the shortcut itself.
                app.action_open_commands()
                await pilot.pause()
                self.assertTrue(palette.has_class("-visible"))
                app.action_open_commands()
                await pilot.pause()
                self.assertFalse(palette.has_class("-visible"))
                self.assertEqual(app.query_one("#prompt-input").value, "")

        asyncio.run(_run())


class SessionsMenuTests(unittest.TestCase):
    def test_list_session_catalog_and_new_conversation_helpers(self):
        import json
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from agent.core import (
            apply_saved_session_file,
            list_session_catalog,
            start_new_conversation,
        )

        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            payload = {
                "saved_at": "2026-07-12T00:00:00+00:00",
                "model": "selene",
                "session": {
                    "options": {},
                    "verbose": True,
                    "wordwrap": True,
                    "system": "",
                    "history": True,
                    "format": "",
                    "think": True,
                    "runtime_profile": "manual",
                    "tui_theme": "tokyo",
                },
                "history": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                ],
            }
            path = sessions_dir / "Demo_chat.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with patch("agent.core._SESSIONS_DIR", str(sessions_dir)), patch(
                "agent.core._LEGACY_SESSIONS_DIR", str(sessions_dir / "legacy-missing")
            ):
                catalog = list_session_catalog()
            self.assertEqual(len(catalog), 1)
            self.assertEqual(catalog[0]["title"], "Demo_chat")
            self.assertIn("1 msg", catalog[0]["detail"])

            session = {
                "options": {},
                "verbose": True,
                "wordwrap": True,
                "system": "old",
                "history": True,
                "format": "",
                "think": True,
                "runtime_profile": "manual",
                "tui_theme": "oslo",
            }
            history = [{"role": "user", "content": "stale"}]
            name, count, warnings = apply_saved_session_file(str(path), session, history)
            self.assertEqual(name, "Demo_chat")
            self.assertEqual(count, 1)
            self.assertEqual(session.get("tui_theme"), "tokyo")
            self.assertEqual(len(history), 3)

            start_new_conversation(session, history)
            self.assertEqual(history, [])
            self.assertEqual(session.get("system"), "")

    def test_sessions_menu_opens_with_new_conversation_and_escape_closes(self):
        try:
            import textual  # noqa: F401
        except ImportError:
            self.skipTest("textual not installed")

        import asyncio
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from agent.tui import build_app_class

        AppCls = build_app_class()
        session = {
            "history": True,
            "system": "",
            "options": {},
            "verbose": True,
            "wordwrap": True,
            "format": "",
            "think": True,
            "runtime_profile": "manual",
            "tui_theme": "oslo",
        }
        app = AppCls(
            session=session,
            history=[{"role": "user", "content": "keep me"}],
            default_system_prompt="sys",
            process_turn=lambda *a, **k: None,
            handle_command=lambda *a, **k: True,
            slash_completions=("/help", "/save"),
            slash_descriptions={"/help": "Help", "/save": "Save"},
            status_meta={"profile": "manual", "model": "selene"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            payload = {
                "saved_at": "2026-07-12T00:00:00+00:00",
                "model": "selene",
                "session": {
                    "options": {},
                    "verbose": True,
                    "wordwrap": True,
                    "system": "",
                    "history": True,
                    "format": "",
                    "think": True,
                    "runtime_profile": "manual",
                    "tui_theme": "oslo",
                },
                "history": [
                    {"role": "user", "content": "old question"},
                    {"role": "assistant", "content": "old answer"},
                ],
            }
            path = sessions_dir / "Old_talk.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            async def _run():
                with patch("agent.core._SESSIONS_DIR", str(sessions_dir)), patch(
                    "agent.core._LEGACY_SESSIONS_DIR", str(sessions_dir / "nope")
                ):
                    async with app.run_test() as pilot:
                        await pilot.pause()
                        app.action_open_sessions()
                        await pilot.pause()
                        self.assertTrue(app._sessions_open)
                        self.assertGreaterEqual(len(app._session_rows), 2)
                        self.assertEqual(app._session_rows[0][0], "__new__")
                        self.assertEqual(app._session_rows[0][1], "New Conversation")
                        menu = app.query_one("#sessions-menu")
                        self.assertTrue(menu.has_class("-visible"))
                        plain = str(menu.renderable) if hasattr(menu, "renderable") else ""
                        content = getattr(menu, "content", None) or getattr(menu, "_content", "")
                        body = content.plain if hasattr(content, "plain") else str(content)
                        self.assertIn("New Conversation", body)
                        self.assertIn("Old_talk", body)

                        # Escape closes conversations menu without quitting.
                        app.action_blur_or_clear()
                        await pilot.pause()
                        self.assertFalse(app._sessions_open)
                        self.assertFalse(menu.has_class("-visible"))

                        # Slash palette opens; escape closes it too.
                        app.action_open_commands()
                        await pilot.pause()
                        self.assertTrue(app._slash_matches)
                        palette = app.query_one("#slash-palette")
                        self.assertTrue(palette.has_class("-visible"))
                        app.action_blur_or_clear()
                        await pilot.pause()
                        self.assertFalse(app._slash_matches)
                        self.assertFalse(palette.has_class("-visible"))

                        # Open chats again and choose New Conversation.
                        app.history[:] = [{"role": "user", "content": "stale"}]
                        app.action_open_sessions()
                        await pilot.pause()
                        app._session_selected = 0
                        app._accept_session_selection()
                        await pilot.pause()
                        self.assertEqual(app.history, [])
                        self.assertFalse(app._sessions_open)

                        # Load the saved conversation via the menu.
                        app.action_open_sessions()
                        await pilot.pause()
                        # Find Old_talk row
                        idx = next(
                            i
                            for i, row in enumerate(app._session_rows)
                            if row[1] == "Old_talk"
                        )
                        app._session_selected = idx
                        app._accept_session_selection()
                        await pilot.pause()
                        self.assertEqual(
                            [m.get("role") for m in app.history],
                            ["user", "assistant"],
                        )

            asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
