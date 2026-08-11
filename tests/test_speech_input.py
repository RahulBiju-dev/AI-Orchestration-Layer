"""Tests for CLI/TUI voice input (Web UI mic parity)."""

from __future__ import annotations

import os
import sys
import threading
import types
import unittest
from unittest.mock import MagicMock, patch


class SpeechCapabilityTests(unittest.TestCase):
    def test_transcript_combines_like_web_ui(self):
        from agent.speech_input import VoiceInputController

        transcripts: list[str] = []
        voice = VoiceInputController(on_transcript=transcripts.append)
        voice.set_base_text("Hello")
        voice._emit_transcript("world")
        self.assertEqual(transcripts[-1], "Hello world")
        voice._emit_transcript("again")
        self.assertEqual(transcripts[-1], "Hello world again")
        self.assertEqual(voice.get_base_text(), "Hello world again")
        voice.set_base_text("")
        voice._emit_transcript("solo")
        self.assertEqual(transcripts[-1], "solo")

    def test_controller_start_reports_unsupported(self):
        from agent.speech_input import VoiceInputController, SpeechCapability

        errors: list[tuple[str, str]] = []
        voice = VoiceInputController(
            on_error=lambda code, msg: errors.append((code, msg)),
        )
        with patch(
            "agent.speech_input.speech_capability",
            return_value=SpeechCapability(False, "no mic stack"),
        ):
            started = voice.start(base_text="")
        self.assertFalse(started)
        self.assertEqual(errors[0][0], "unsupported")
        self.assertIn("no mic stack", errors[0][1])

    def test_audio_stderr_silence_restores_depth(self):
        """ALSA-style stderr silence must nest and fully restore."""
        from agent import speech_input
        from agent.speech_input import audio_stderr_silence

        self.assertEqual(speech_input._silence_depth, 0)
        with audio_stderr_silence():
            self.assertEqual(speech_input._silence_depth, 1)
            # C libraries write the OS fd directly (not sys.stderr).
            os.write(2, b"ALSA lib pcm: should not appear\n")
            with audio_stderr_silence():
                self.assertEqual(speech_input._silence_depth, 2)
                os.write(2, b"nested alsa noise\n")
            self.assertEqual(speech_input._silence_depth, 1)
        self.assertEqual(speech_input._silence_depth, 0)
        self.assertIsNone(speech_input._silence_saved_fd)

    def test_capability_cache_avoids_repeated_probes(self):
        from agent import speech_input

        speech_input.clear_speech_capability_cache()
        cached = speech_input.SpeechCapability(True, "cached")
        with patch.object(speech_input, "_capability_cache", cached):
            result = speech_input.speech_capability()
        self.assertIs(result, cached)

    def test_capability_check_does_not_open_microphone(self):
        from agent import speech_input

        fake_sr = types.SimpleNamespace(Microphone=MagicMock())
        speech_input.clear_speech_capability_cache()
        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            result = speech_input.speech_capability(force=True)
        self.assertTrue(result.available)
        fake_sr.Microphone.assert_not_called()

    def test_clean_stop_keeps_final_inflight_phrase(self):
        from agent.speech_input import VoiceInputController

        transcripts: list[str] = []
        finished: list[bool] = []
        voice = VoiceInputController(
            on_transcript=transcripts.append,
            on_finished=lambda: finished.append(True),
        )

        class FakeMicrophone:
            stream = None

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        class FakeRecognizer:
            def adjust_for_ambient_noise(self, *_args, **_kwargs):
                return None

            def listen(self, *_args, **_kwargs):
                voice.stop(silent=True)
                return object()

            def recognize_google(self, *_args, **_kwargs):
                return "final words"

        class WaitTimeoutError(Exception):
            pass

        class UnknownValueError(Exception):
            pass

        class RequestError(Exception):
            pass

        fake_sr = types.SimpleNamespace(
            Recognizer=FakeRecognizer,
            Microphone=FakeMicrophone,
            WaitTimeoutError=WaitTimeoutError,
            UnknownValueError=UnknownValueError,
            RequestError=RequestError,
        )
        stop_event = threading.Event()
        with voice._lock:
            voice._generation = 1
            voice._active = True
            voice._stop = stop_event
            voice._thread = threading.current_thread()
        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            voice._listen_loop(1, stop_event)

        self.assertEqual(transcripts, ["final words"])
        self.assertEqual(finished, [True])

    def test_capture_continues_while_previous_phrase_is_transcribed(self):
        from agent.speech_input import VoiceInputController

        second_capture_started = threading.Event()
        release_recognition = threading.Event()
        transcripts: list[str] = []
        finished = threading.Event()
        voice = VoiceInputController(
            on_transcript=transcripts.append,
            on_finished=finished.set,
        )

        class FakeMicrophone:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        class FakeRecognizer:
            created = 0

            def __init__(self):
                self.role = FakeRecognizer.created
                FakeRecognizer.created += 1
                self.listen_count = 0

            def adjust_for_ambient_noise(self, *_args, **_kwargs):
                return None

            def listen(self, *_args, **_kwargs):
                self.listen_count += 1
                if self.listen_count == 1:
                    return "first audio"
                second_capture_started.set()
                voice.stop(silent=True)
                return "second audio"

            def recognize_google(self, audio, **_kwargs):
                if audio == "first audio":
                    release_recognition.wait(2)
                    return "first"
                return "second"

        class WaitTimeoutError(Exception):
            pass

        class UnknownValueError(Exception):
            pass

        class RequestError(Exception):
            pass

        fake_sr = types.SimpleNamespace(
            Recognizer=FakeRecognizer,
            Microphone=FakeMicrophone,
            WaitTimeoutError=WaitTimeoutError,
            UnknownValueError=UnknownValueError,
            RequestError=RequestError,
        )
        stop_event = threading.Event()
        worker = threading.Thread(
            target=voice._listen_loop,
            args=(1, stop_event),
            daemon=True,
        )
        with voice._lock:
            voice._generation = 1
            voice._active = True
            voice._stop = stop_event
            voice._thread = worker
        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            worker.start()
            self.assertTrue(second_capture_started.wait(1))
            # Recognition of phrase one is still blocked, yet phrase two was
            # already captured by the independent microphone worker.
            self.assertEqual(transcripts, [])
            release_recognition.set()
            worker.join(3)

        self.assertFalse(worker.is_alive())
        self.assertTrue(finished.is_set())
        self.assertEqual(transcripts, ["first", "first second"])

    def test_abort_releases_capture_without_waiting_for_obsolete_network_call(self):
        from agent.speech_input import VoiceInputController

        recognition_started = threading.Event()
        release_recognition = threading.Event()
        stop_event = threading.Event()
        voice = VoiceInputController()

        class FakeMicrophone:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        class FakeRecognizer:
            def __init__(self):
                self.listen_count = 0

            def adjust_for_ambient_noise(self, *_args, **_kwargs):
                return None

            def listen(self, *_args, **_kwargs):
                self.listen_count += 1
                if self.listen_count == 1:
                    return "audio"
                stop_event.wait(1)
                return "discarded"

            def recognize_google(self, *_args, **_kwargs):
                recognition_started.set()
                release_recognition.wait(2)
                return "obsolete"

        class WaitTimeoutError(Exception):
            pass

        class UnknownValueError(Exception):
            pass

        class RequestError(Exception):
            pass

        fake_sr = types.SimpleNamespace(
            Recognizer=FakeRecognizer,
            Microphone=FakeMicrophone,
            WaitTimeoutError=WaitTimeoutError,
            UnknownValueError=UnknownValueError,
            RequestError=RequestError,
        )
        worker = threading.Thread(
            target=voice._listen_loop,
            args=(1, stop_event),
            daemon=True,
        )
        with voice._lock:
            voice._generation = 1
            voice._active = True
            voice._stop = stop_event
            voice._thread = worker
        with patch.dict(sys.modules, {"speech_recognition": fake_sr}):
            worker.start()
            self.assertTrue(recognition_started.wait(1))
            voice.stop(abort=True, silent=True)
            worker.join(1)
            self.assertFalse(worker.is_alive())
            release_recognition.set()

        self.assertEqual(voice.get_base_text(), "")

    def test_stalled_transcription_cannot_freeze_clean_stop(self):
        from agent import speech_input
        from agent.speech_input import VoiceInputController

        recognition_started = threading.Event()
        release_recognition = threading.Event()
        finished = threading.Event()
        stop_event = threading.Event()
        voice = VoiceInputController(on_finished=finished.set)

        class FakeMicrophone:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        class FakeRecognizer:
            def adjust_for_ambient_noise(self, *_args, **_kwargs):
                return None

            def listen(_recognizer, *_args, **_kwargs):
                # Long-lived fd redirection can freeze Textual repainting.
                self.assertEqual(speech_input._silence_depth, 0)
                voice.stop(silent=True)
                return "audio"

            def recognize_google(self, *_args, **_kwargs):
                recognition_started.set()
                release_recognition.wait(2)
                return "late"

        class WaitTimeoutError(Exception):
            pass

        class UnknownValueError(Exception):
            pass

        class RequestError(Exception):
            pass

        fake_sr = types.SimpleNamespace(
            Recognizer=FakeRecognizer,
            Microphone=FakeMicrophone,
            WaitTimeoutError=WaitTimeoutError,
            UnknownValueError=UnknownValueError,
            RequestError=RequestError,
        )
        worker = threading.Thread(
            target=voice._listen_loop,
            args=(1, stop_event),
            daemon=True,
        )
        with voice._lock:
            voice._generation = 1
            voice._active = True
            voice._stop = stop_event
            voice._thread = worker
        with patch.dict(sys.modules, {"speech_recognition": fake_sr}), patch.object(
            speech_input, "_RECOGNITION_DRAIN_TIMEOUT", 0.05
        ):
            worker.start()
            self.assertTrue(recognition_started.wait(1))
            worker.join(0.5)
            self.assertFalse(worker.is_alive())
            self.assertTrue(finished.is_set())
            self.assertEqual(speech_input._silence_depth, 0)
            release_recognition.set()

        self.assertEqual(voice.get_base_text(), "")

    def test_stop_is_non_blocking_and_invalidates_late_worker(self):
        from agent.speech_input import VoiceInputController

        active_states: list[bool] = []
        voice = VoiceInputController(on_active=active_states.append)
        thread = MagicMock()
        stream = MagicMock()
        with voice._lock:
            voice._active = True
            voice._generation = 4
            voice._thread = thread
            voice._microphone = MagicMock(stream=stream)

        voice.stop(abort=True, silent=True)

        self.assertFalse(voice.active)
        self.assertEqual(voice._generation, 5)
        self.assertEqual(active_states, [False])
        # PortAudio teardown must stay on its owning worker; cross-thread close
        # produces ALSA assertions and can destabilize the next recording.
        stream.close.assert_not_called()
        thread.join.assert_not_called()

        # Completion from generation 4 must not clear a newer recording.
        with voice._lock:
            voice._active = True
            voice._generation = 6
            replacement = MagicMock()
            voice._thread = replacement
        voice._finish_loop(4)
        self.assertTrue(voice.active)
        self.assertIs(voice._thread, replacement)


class SpeechCommandTests(unittest.TestCase):
    def test_handle_speech_forwards_to_tui_sink(self):
        from agent.core import _handle_speech
        from agent import terminal

        calls: list[str] = []

        class Sink:
            is_tui = True

            def toggle_speech(self, action="toggle"):
                calls.append(action)

        terminal.set_display_sink(Sink())
        try:
            _handle_speech("start")
            _handle_speech("")
            _handle_speech("stop")
            self.assertEqual(calls, ["start", "toggle", "stop"])
        finally:
            terminal.set_display_sink(None)

    def test_speech_in_slash_catalog(self):
        from agent.core import CLI_SLASH_COMPLETIONS, CLI_SLASH_DESCRIPTIONS

        self.assertIn("/speech", CLI_SLASH_COMPLETIONS)
        self.assertIn("/speech start", CLI_SLASH_COMPLETIONS)
        self.assertIn("/speech stop", CLI_SLASH_COMPLETIONS)
        self.assertIn("speech", CLI_SLASH_DESCRIPTIONS["/speech"].casefold())
        # Help text should mention the TUI shortcut (Ctrl+S).
        self.assertIn("ctrl+s", CLI_SLASH_DESCRIPTIONS["/speech"].casefold())


class SpeechTuiTests(unittest.TestCase):
    def test_speech_menu_enter_records_then_sends(self):
        try:
            import textual  # noqa: F401
        except ImportError:
            self.skipTest("textual not installed")

        import asyncio
        from agent.tui import build_app_class
        from agent.speech_input import SpeechCapability

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
        submitted: list[str] = []
        app = AppCls(
            session=session,
            history=[],
            default_system_prompt="sys",
            process_turn=lambda *a, **k: None,
            handle_command=lambda *a, **k: True,
            slash_completions=("/speech", "/help"),
            slash_descriptions={"/speech": "Voice", "/help": "Help"},
            status_meta={},
        )

        async def _run():
            async with app.run_test() as pilot:
                await pilot.pause()
                fake = MagicMock()
                fake.active = False
                fake.start = MagicMock(return_value=True)
                fake.stop = MagicMock()
                fake.set_base_text = MagicMock()

                with patch(
                    "agent.speech_input.speech_capability",
                    return_value=SpeechCapability(True, "ok"),
                ), patch.object(app, "_ensure_voice", return_value=fake), patch.object(
                    app, "_submit", side_effect=lambda text: submitted.append(text)
                ):
                    app._voice = fake
                    # Ctrl+S / action opens the centered speech popup (does not record yet).
                    app.action_toggle_speech()
                    await pilot.pause()
                    self.assertTrue(app._speech_open)
                    menu = app.query_one("#speech-menu")
                    self.assertTrue(menu.has_class("-visible"))
                    fake.start.assert_not_called()
                    self.assertFalse(menu.has_class("-recording"))

                    # First Enter starts recording + animation.
                    app._speech_on_enter()
                    await pilot.pause()
                    fake.start.assert_called()
                    fake.active = True
                    app._voice_active = True
                    app._voice_set_active_ui(True)
                    await pilot.pause()
                    self.assertTrue(menu.has_class("-recording"))

                    # Live transcript lands in the speech textbox (editable).
                    app._voice_apply_transcript("hello agent")
                    await pilot.pause()
                    self.assertEqual(
                        app.query_one("#speech-input").value, "hello agent"
                    )
                    # Main composer stays untouched.
                    self.assertEqual(app.query_one("#prompt-input").value, "")

                    # Second Enter stops recording and submits the prompt.
                    app._speech_on_enter()
                    await pilot.pause()
                    fake.stop.assert_called()
                    self.assertTrue(app._speech_pending_submit)
                    # The controller's worker invokes this only after its final
                    # in-flight phrase has been transcribed.
                    app._speech_finish_submit()
                    await pilot.pause()
                    self.assertEqual(submitted, ["hello agent"])
                    self.assertFalse(app._speech_open)

                    # Toggle again reopens; toggle closes without status noise.
                    app.action_toggle_speech()
                    await pilot.pause()
                    self.assertTrue(app._speech_open)
                    app.ui_toggle_speech("toggle")
                    await pilot.pause()
                    self.assertFalse(app._speech_open)

                    # /speech start routes into the same menu.
                    app.ui_toggle_speech("start")
                    await pilot.pause()
                    self.assertTrue(app._speech_open)
                    fake.start.assert_called()

        asyncio.run(_run())

    def test_speech_popup_states_and_multiline_transcript(self):
        try:
            import textual  # noqa: F401
        except ImportError:
            self.skipTest("textual not installed")

        import asyncio

        from agent.tui import build_app_class

        AppCls = build_app_class()
        app = AppCls(
            session={"history": True, "system": "", "options": {}, "verbose": False,
                     "wordwrap": True, "format": "", "think": True,
                     "runtime_profile": "manual"},
            history=[],
            default_system_prompt="sys",
            process_turn=lambda *a, **k: None,
            handle_command=lambda *a, **k: True,
            slash_completions=("/speech",),
            slash_descriptions={"/speech": "Voice"},
            status_meta={},
        )

        def card_text(widget) -> str:
            visual = widget.visual
            renderable = getattr(visual, "_renderable", visual)
            return str(getattr(renderable, "plain", renderable))

        async def _run():
            async with app.run_test(size=(96, 32)) as pilot:
                app.action_toggle_speech()
                await pilot.pause()
                menu = app.query_one("#speech-menu")
                title = app.query_one("#speech-title")
                hint = app.query_one("#speech-hint")

                # Idle: no recording chrome, an actionable hint, no timer.
                self.assertFalse(menu.has_class("-recording"))
                self.assertIn("ready", card_text(title))
                self.assertIn("start listening", card_text(hint))
                self.assertEqual(card_text(app.query_one("#speech-timer")), "")

                menu.set_recording(True)
                await pilot.pause()
                self.assertTrue(menu.has_class("-recording"))
                self.assertIn("listening", card_text(title))
                self.assertIn("stop & send", card_text(hint))
                self.assertRegex(card_text(app.query_one("#speech-timer")), r"\d\d:\d\d")

                menu.set_finishing()
                await pilot.pause()
                self.assertFalse(menu.has_class("-recording"))
                self.assertIn("transcribing", card_text(title))

                # Errors stay inside the card and typing keeps working.
                menu.set_error("No microphone was found. Connect an input device.")
                await pilot.pause()
                self.assertTrue(menu.has_class("-error"))
                self.assertIn("unavailable", card_text(title))
                self.assertIn("microphone", card_text(app.query_one("#speech-detail")))

                # A long transcript wraps instead of scrolling out of sight.
                speech_input = app.query_one("#speech-input")
                body = app.query_one("#speech-body")
                short_height = body.size.height
                speech_input.value = "review the scaling notes " * 12
                await pilot.pause()
                self.assertGreater(body.size.height, short_height)

        asyncio.run(_run())

    def test_ctrl_s_binding_registered(self):
        try:
            import textual  # noqa: F401
        except ImportError:
            self.skipTest("textual not installed")

        from agent.tui import build_app_class

        AppCls = build_app_class()
        keys = []
        for binding in AppCls.BINDINGS:
            key = getattr(binding, "key", None) or getattr(binding, "keys", None)
            keys.append(str(key))
        joined = " ".join(keys)
        self.assertIn("ctrl+s", joined)
        # Ctrl+M collides with Enter in terminals — must not be the speech binding.
        self.assertNotIn("ctrl+m", joined)


if __name__ == "__main__":
    unittest.main()
