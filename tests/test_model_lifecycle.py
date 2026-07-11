import tempfile
import unittest
from pathlib import Path

from agent.model_lifecycle import ensure_managed_model
from agent.ollama_runtime import (
    OllamaModelMissingError,
    OllamaProbeStatus,
    OllamaUnavailableError,
    parse_modelfile,
)
from agent.platform_runtime import resolve_runtime_paths
from agent.runtime_config import HardwareInfo, resolve_runtime_config


def _config(**overrides):
    # Default runtime settings follow the bundled Modelfile (manual profile).
    return resolve_runtime_config(
        overrides or None,
        environ={},
        hardware=HardwareInfo(gpu_vram_mb=4096, reason="test"),
    )


class _LifecycleService:
    def __init__(self, models=(), *, api_available=True):
        self.models = set(models)
        self.api_available = api_available
        self.installs = []

    def probe(self, timeout=3.0):
        return OllamaProbeStatus(
            cli_installed=True,
            api_available=self.api_available,
            executable="ollama",
            reason="test API unavailable" if not self.api_available else "available",
        )

    def model_exists(self, model, timeout=5.0):
        return model in self.models

    def install_model_staged(self, **kwargs):
        self.installs.append(kwargs)
        self.models.add(kwargs["model"])
        return {"model": kwargs["model"]}


class ModelLifecycleTests(unittest.TestCase):
    def test_bundled_modelfile_fallback_matches_modelfile_defaults(self):
        config = _config()
        parsed = parse_modelfile(Path(__file__).resolve().parents[1] / "Modelfile")

        self.assertEqual(parsed.base_model, config.base_model)
        runtime_options = config.ollama_options()
        # Compare only parameters declared in the Modelfile (not Selene-only knobs).
        for name, value in parsed.parameters.items():
            self.assertIn(name, runtime_options)
            self.assertEqual(runtime_options[name], value)

    def _fixture(self, directory):
        root = Path(directory)
        modelfile = root / "Modelfile"
        modelfile.write_text(
            'FROM gemma4:e4b\nSYSTEM """policy"""\nPARAMETER num_ctx 4096\n',
            encoding="utf-8",
        )
        paths = resolve_runtime_paths(
            platform_name="linux",
            environ={"SELENE_DATA_DIR": str(root / "data")},
            home=root,
            legacy_exists=False,
        )
        return modelfile, paths

    def test_missing_managed_model_is_built_through_staging_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            modelfile, paths = self._fixture(directory)
            service = _LifecycleService({"gemma4:e4b"})

            result = ensure_managed_model(
                config=_config(),
                service=service,
                modelfile_path=modelfile,
                runtime_paths=paths,
            )

            self.assertEqual(result.action, "built")
            self.assertEqual(len(service.installs), 1)
            self.assertNotEqual(service.installs[0]["staging_model"], "selene")
            self.assertTrue(result.metadata_path.is_file())

    def test_matching_build_record_avoids_unnecessary_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            modelfile, paths = self._fixture(directory)
            service = _LifecycleService({"gemma4:e4b"})
            ensure_managed_model(
                config=_config(), service=service, modelfile_path=modelfile, runtime_paths=paths
            )

            result = ensure_managed_model(
                config=_config(), service=service, modelfile_path=modelfile, runtime_paths=paths
            )

            self.assertEqual(result.action, "ready")
            self.assertEqual(len(service.installs), 1)

    def test_changed_modelfile_rebuilds_without_predeleting_target(self):
        with tempfile.TemporaryDirectory() as directory:
            modelfile, paths = self._fixture(directory)
            service = _LifecycleService({"gemma4:e4b"})
            ensure_managed_model(
                config=_config(), service=service, modelfile_path=modelfile, runtime_paths=paths
            )
            modelfile.write_text(
                'FROM gemma4:e4b\nSYSTEM """new policy"""\nPARAMETER num_ctx 4096\n',
                encoding="utf-8",
            )

            result = ensure_managed_model(
                config=_config(), service=service, modelfile_path=modelfile, runtime_paths=paths
            )

            self.assertEqual(result.action, "rebuilt")
            self.assertIn("changed", result.reason.lower())
            self.assertEqual(len(service.installs), 2)

    def test_unavailable_api_and_missing_base_model_are_controlled(self):
        with tempfile.TemporaryDirectory() as directory:
            modelfile, paths = self._fixture(directory)
            with self.assertRaises(OllamaUnavailableError):
                ensure_managed_model(
                    config=_config(),
                    service=_LifecycleService(api_available=False),
                    modelfile_path=modelfile,
                    runtime_paths=paths,
                )
            with self.assertRaises(OllamaModelMissingError):
                ensure_managed_model(
                    config=_config(),
                    service=_LifecycleService(),
                    modelfile_path=modelfile,
                    runtime_paths=paths,
                )

    def test_explicit_external_chat_model_is_verified_but_never_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            modelfile, paths = self._fixture(directory)
            service = _LifecycleService({"other:model"})

            result = ensure_managed_model(
                config=_config(chat_model="other:model"),
                service=service,
                modelfile_path=modelfile,
                runtime_paths=paths,
            )

            self.assertEqual(result.action, "verified-external")
            self.assertEqual(service.installs, [])


if __name__ == "__main__":
    unittest.main()
