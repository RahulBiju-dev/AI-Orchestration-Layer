import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path


agent_module = sys.modules.get("agent")
if agent_module is not None and not hasattr(agent_module, "__path__"):
    sys.modules.pop("agent", None)
    sys.modules.pop("agent.runtime_config", None)
    sys.modules.pop("agent.ollama_runtime", None)

from agent.ollama_runtime import (
    CancellationToken,
    InvalidModelfileError,
    OllamaCoordinator,
    OllamaContextOverflow,
    OllamaService,
    OperationCancelledError,
    OperationKind,
    OperationOwnershipError,
    is_model_stale,
    model_build_record,
    parse_modelfile,
    stale_model_reason,
)
from agent.runtime_config import HardwareInfo, resolve_runtime_config


def low_vram_config():
    return resolve_runtime_config(
        {"profile": "low-vram"},
        environ={},
        hardware=HardwareInfo(gpu_name="test", gpu_vram_mb=4096, reason="test"),
    )


def _close_stream(stream, failures):
    try:
        stream.close()
    except BaseException as exc:
        failures.append(exc)


class OllamaCoordinatorTests(unittest.TestCase):
    def test_low_vram_model_operations_are_serialized(self):
        coordinator = OllamaCoordinator(low_vram_config())
        first_acquired = threading.Event()
        release_first = threading.Event()
        second_acquired = threading.Event()
        failures = []

        def first():
            try:
                with coordinator.operation(OperationKind.CHAT, owner="session-a"):
                    first_acquired.set()
                    release_first.wait(2)
            except BaseException as exc:
                failures.append(exc)

        def second():
            try:
                with coordinator.operation(OperationKind.VISION, owner="session-b"):
                    second_acquired.set()
            except BaseException as exc:
                failures.append(exc)

        first_thread = threading.Thread(target=first)
        second_thread = threading.Thread(target=second)
        first_thread.start()
        self.assertTrue(first_acquired.wait(1))
        second_thread.start()
        self.assertFalse(second_acquired.wait(0.15))
        self.assertEqual(len(coordinator.active_operations()), 1)
        release_first.set()
        self.assertTrue(second_acquired.wait(1))
        first_thread.join(1)
        second_thread.join(1)
        self.assertEqual(failures, [])
        self.assertEqual(coordinator.active_operations(), ())

    def test_lease_is_reentrant_for_same_owner_and_context(self):
        coordinator = OllamaCoordinator(low_vram_config())

        with coordinator.operation(OperationKind.EMBEDDING, owner="index:one") as outer:
            self.assertTrue(coordinator.is_owned_by_current_context("index:one"))
            self.assertEqual(coordinator.current_context_owner(), "index:one")
            with coordinator.operation(OperationKind.EMBEDDING, owner="index:one") as inner:
                self.assertTrue(inner.is_reentrant)
                self.assertEqual(inner.operation_id, outer.operation_id)
                self.assertEqual(len(coordinator.active_operations()), 1)
            with self.assertRaises(OperationOwnershipError):
                coordinator.operation(OperationKind.VISION, owner="another-owner")

        self.assertFalse(coordinator.is_owned_by_current_context())
        self.assertIsNone(coordinator.current_context_owner())
        self.assertEqual(coordinator.active_operations(), ())

    def test_release_occurs_after_callback_failure(self):
        coordinator = OllamaCoordinator(low_vram_config())

        with self.assertRaisesRegex(RuntimeError, "boom"):
            coordinator.run(
                OperationKind.CHAT,
                lambda lease: (_ for _ in ()).throw(RuntimeError("boom")),
                owner="session-a",
            )

        self.assertEqual(coordinator.active_operations(), ())
        with coordinator.operation(OperationKind.CHAT, owner="session-b"):
            self.assertEqual(len(coordinator.active_operations()), 1)

    def test_waiting_operation_observes_cancellation(self):
        coordinator = OllamaCoordinator(low_vram_config())
        token = CancellationToken()
        finished = threading.Event()
        errors = []

        def waiting_operation():
            try:
                with coordinator.operation(
                    OperationKind.TITLE,
                    owner="session-b",
                    cancellation_token=token,
                ):
                    errors.append(AssertionError("cancelled waiter unexpectedly acquired"))
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        with coordinator.operation(OperationKind.CHAT, owner="session-a"):
            thread = threading.Thread(target=waiting_operation)
            thread.start()
            time.sleep(0.05)
            token.cancel("browser disconnected")
            self.assertTrue(finished.wait(1))
            thread.join(1)

        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], OperationCancelledError)

    def test_cancel_owner_marks_active_operation_and_reports_it(self):
        coordinator = OllamaCoordinator(low_vram_config())
        with coordinator.operation(OperationKind.CHAT, owner="session-a") as lease:
            snapshot = coordinator.active_operations()[0]
            self.assertEqual(snapshot.owner, "session-a")
            self.assertEqual(coordinator.cancel_owner("session-a", "stop requested"), 1)
            self.assertTrue(coordinator.active_operations()[0].cancellation_requested)
            with self.assertRaises(OperationCancelledError):
                lease.checkpoint()


class _StreamingClient:
    def chat(self, **kwargs):
        def chunks():
            yield {"message": {"content": "one"}}
            yield {"message": {"content": "two"}}

        return chunks()

    def list(self):
        return {"models": []}


class _UnavailableClient:
    def list(self):
        raise ConnectionError("connection refused")


class _BuildClient:
    def __init__(self, *, fail_create=False):
        self.fail_create = fail_create
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(("create", kwargs["model"]))
        if self.fail_create:
            raise RuntimeError("invalid Modelfile")
        return iter(({"status": "building"}, {"status": "success"}))

    def show(self, model):
        self.calls.append(("show", model))
        return {"model": model}

    def copy(self, source, destination):
        self.calls.append(("copy", source, destination))
        return {"status": "success"}

    def delete(self, model):
        self.calls.append(("delete", model))
        return {"status": "success"}


class OllamaServiceTests(unittest.TestCase):
    def test_oversized_chat_is_rejected_before_client_creation(self):
        config = low_vram_config()
        service = OllamaService(
            config,
            coordinator=OllamaCoordinator(config),
            client_factory=lambda timeout: self.fail("client should not be created"),
        )

        with self.assertRaises(OllamaContextOverflow):
            service.chat(
                kind=OperationKind.CHAT,
                owner="session-a",
                messages=[{"role": "system", "content": "x" * 30000}],
                stream=False,
            )

    def test_oversized_embedding_is_rejected_before_client_creation(self):
        config = low_vram_config()
        service = OllamaService(
            config,
            coordinator=OllamaCoordinator(config),
            client_factory=lambda timeout: self.fail("client should not be created"),
        )

        with self.assertRaises(OllamaContextOverflow):
            service.embed("漢" * 5000, owner="index-a")

    def test_closing_stream_releases_coordinator_lease(self):
        config = low_vram_config()
        coordinator = OllamaCoordinator(config)
        service = OllamaService(
            config,
            coordinator=coordinator,
            client_factory=lambda timeout: _StreamingClient(),
        )
        stream = service.chat(
            kind=OperationKind.CHAT,
            owner="session-a",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )

        self.assertEqual(next(stream)["message"]["content"], "one")
        self.assertEqual(len(coordinator.active_operations()), 1)
        stream.close()
        self.assertEqual(coordinator.active_operations(), ())

    def test_closing_stream_from_disconnect_thread_releases_coordinator_lease(self):
        config = low_vram_config()
        coordinator = OllamaCoordinator(config)
        service = OllamaService(
            config,
            coordinator=coordinator,
            client_factory=lambda timeout: _StreamingClient(),
        )
        stream = service.chat(
            kind=OperationKind.CHAT,
            owner="session-a",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        next(stream)
        failures = []

        thread = threading.Thread(
            target=lambda: _close_stream(stream, failures),
        )
        thread.start()
        thread.join(1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(coordinator.active_operations(), ())
        with coordinator.operation(OperationKind.CHAT, owner="session-b"):
            self.assertEqual(len(coordinator.active_operations()), 1)

    def test_probe_reports_unavailable_without_raising(self):
        service = OllamaService(
            low_vram_config(),
            coordinator=OllamaCoordinator(low_vram_config()),
            client_factory=lambda timeout: _UnavailableClient(),
        )

        status = service.probe()

        self.assertFalse(status.api_available)
        self.assertIn("unavailable", status.reason.lower())

    def test_probe_handles_ollama_disappearing_between_list_and_show(self):
        class RacyClient:
            def list(self):
                return {"models": []}

            def show(self, model):
                raise ConnectionError("connection refused")

        config = low_vram_config()
        service = OllamaService(
            config,
            coordinator=OllamaCoordinator(config),
            client_factory=lambda timeout: RacyClient(),
        )

        status = service.probe(model="selene")

        self.assertFalse(status.api_available)
        self.assertIsNone(status.model_available)
        self.assertIn("unavailable", status.reason.lower())

    def test_probe_does_not_swallow_keyboard_interrupt(self):
        class InterruptedClient:
            def list(self):
                raise KeyboardInterrupt()

        config = low_vram_config()
        service = OllamaService(
            config,
            coordinator=OllamaCoordinator(config),
            client_factory=lambda timeout: InterruptedClient(),
        )

        with self.assertRaises(KeyboardInterrupt):
            service.probe()

    def test_staged_install_publishes_only_after_staging_verification(self):
        client = _BuildClient()
        config = low_vram_config()
        service = OllamaService(
            config,
            coordinator=OllamaCoordinator(config),
            client_factory=lambda timeout: client,
        )

        service.install_model_staged(
            model="selene",
            staging_model="selene-build-abc",
            base_model="base:model",
            system_prompt="policy",
            parameters={"num_ctx": 4096},
        )

        self.assertLess(
            client.calls.index(("show", "selene-build-abc")),
            client.calls.index(("copy", "selene-build-abc", "selene")),
        )
        self.assertEqual(client.calls[-1], ("delete", "selene-build-abc"))

    def test_failed_staging_build_never_replaces_working_target(self):
        client = _BuildClient(fail_create=True)
        config = low_vram_config()
        service = OllamaService(
            config,
            coordinator=OllamaCoordinator(config),
            client_factory=lambda timeout: client,
        )

        with self.assertRaisesRegex(Exception, "invalid Modelfile"):
            service.install_model_staged(
                model="selene",
                staging_model="selene-build-abc",
                base_model="base:model",
                system_prompt="policy",
                parameters={"num_ctx": 4096},
            )

        self.assertFalse(any(call[0] == "copy" for call in client.calls))


class ModelfileMetadataTests(unittest.TestCase):
    def test_hash_is_stable_across_line_endings_and_detects_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Modelfile"
            path.write_bytes(b"FROM test:model\r\nPARAMETER num_ctx 4096\r\n")
            record = model_build_record("selene", path)
            self.assertFalse(is_model_stale(record, model="selene", modelfile_path=path))

            path.write_text("FROM test:model\nPARAMETER num_ctx 8192\n", encoding="utf-8")
            self.assertTrue(is_model_stale(record, model="selene", modelfile_path=path))
            self.assertIn("changed", stale_model_reason(record, model="selene", modelfile_path=path))

    def test_invalid_modelfile_fails_before_model_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Modelfile"
            path.write_text('SYSTEM """unterminated', encoding="utf-8")
            with self.assertRaises(InvalidModelfileError):
                parse_modelfile(path)


if __name__ == "__main__":
    unittest.main()
