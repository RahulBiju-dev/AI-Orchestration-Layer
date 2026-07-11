import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


agent_module = sys.modules.get("agent")
if agent_module is not None and not hasattr(agent_module, "__path__"):
    sys.modules.pop("agent", None)
    sys.modules.pop("agent.runtime_config", None)

from agent.runtime_config import (
    HardwareInfo,
    RuntimeConfigManager,
    RuntimeConfigurationError,
    RuntimeProfile,
    detect_hardware,
    load_user_config,
    resolve_runtime_config,
)


LOW_VRAM_HARDWARE = HardwareInfo(
    gpu_name="NVIDIA RTX 3050 Ti Laptop GPU",
    gpu_vram_mb=4096,
    system_memory_mb=16384,
    detector="test",
    reason="test fixture",
)


class RuntimeConfigTests(unittest.TestCase):
    def test_default_profile_uses_modelfile_parameters_without_auto_hardware_pick(self):
        """Fresh systems must not auto-select low-vram/balanced from VRAM."""
        config = resolve_runtime_config(environ={}, hardware=LOW_VRAM_HARDWARE)

        self.assertEqual(config.requested_profile, RuntimeProfile.MANUAL)
        self.assertEqual(config.profile, RuntimeProfile.MANUAL)
        # Bundled Modelfile PARAMETER values.
        self.assertEqual(config.num_ctx, 8192)
        self.assertEqual(config.num_predict, 768)
        self.assertEqual(config.num_batch, 128)
        self.assertEqual(config.temperature, 0.25)
        self.assertEqual(config.top_p, 0.85)
        self.assertEqual(config.top_k, 40)
        self.assertIn("Modelfile", config.selection_reason)
        self.assertNotIn("selected the conservative low-VRAM profile", config.selection_reason)

    def test_explicit_auto_profile_is_conservative_for_four_gib_vram(self):
        config = resolve_runtime_config(
            {"profile": "auto"},
            environ={},
            hardware=LOW_VRAM_HARDWARE,
        )

        self.assertEqual(config.requested_profile, RuntimeProfile.AUTO)
        self.assertEqual(config.profile, RuntimeProfile.LOW_VRAM)
        self.assertEqual(config.num_ctx, 4096)
        self.assertEqual(config.num_predict, 768)
        self.assertEqual(config.num_batch, 128)
        self.assertEqual(config.model_concurrency, 1)
        self.assertTrue(config.serialize_embeddings)
        self.assertTrue(config.serialize_vision)
        self.assertIn("4096 MiB", config.selection_reason)

    def test_override_precedence_is_session_then_environment_then_user(self):
        config = resolve_runtime_config(
            {"options": {"num_ctx": 4096, "temperature": 0.1}},
            user_config={"runtime": {"profile": "balanced", "num_ctx": 7168, "temperature": 0.3}},
            environ={"SELENE_NUM_CTX": "6144", "SELENE_TEMPERATURE": "0.2"},
            hardware=HardwareInfo(gpu_vram_mb=12288, reason="test"),
        )

        self.assertEqual(config.profile, RuntimeProfile.BALANCED)
        self.assertEqual(config.num_ctx, 4096)
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.sources["num_ctx"], "session override")

        without_session = resolve_runtime_config(
            user_config={"num_ctx": 7168},
            environ={"SELENE_NUM_CTX": "6144"},
            hardware=HardwareInfo(gpu_vram_mb=12288, reason="test"),
        )
        self.assertEqual(without_session.num_ctx, 6144)
        self.assertEqual(without_session.sources["num_ctx"], "environment")

    def test_model_names_and_build_timeout_are_centralized(self):
        config = resolve_runtime_config(
            {"chat_model": "session-chat"},
            user_config={"embedding_model": "user-embed"},
            environ={
                "SELENE_VISION_MODEL": "env-vision:latest",
                "SELENE_BUILD_TIMEOUT": "1200",
            },
            hardware=LOW_VRAM_HARDWARE,
        )

        self.assertEqual(config.chat_model, "session-chat")
        self.assertEqual(config.embedding_model, "user-embed")
        self.assertEqual(config.vision_model, "env-vision:latest")
        self.assertEqual(config.build_timeout_seconds, 1200.0)
        self.assertEqual(config.timeout_for("build"), 1200.0)
        self.assertEqual(config.ollama_options()["repeat_penalty"], 1.08)

    def test_invalid_values_and_impossible_output_budget_are_rejected(self):
        with self.assertRaises(RuntimeConfigurationError):
            resolve_runtime_config({"num_ctx": "many"}, environ={}, hardware=LOW_VRAM_HARDWARE)
        with self.assertRaises(RuntimeConfigurationError):
            resolve_runtime_config(
                {"num_ctx": 1024, "num_predict": 768},
                environ={},
                hardware=LOW_VRAM_HARDWARE,
            )
        with self.assertRaises(RuntimeConfigurationError):
            resolve_runtime_config({"top_p": 1.5}, environ={}, hardware=LOW_VRAM_HARDWARE)
        with self.assertRaises(RuntimeConfigurationError):
            resolve_runtime_config({"chat_model": "bad model"}, environ={}, hardware=LOW_VRAM_HARDWARE)
        with self.assertRaises(RuntimeConfigurationError):
            resolve_runtime_config({"num_ctx": 4096.9}, environ={}, hardware=LOW_VRAM_HARDWARE)

    def test_manual_settings_are_preserved_and_warn_when_aggressive(self):
        config = resolve_runtime_config(
            {
                "profile": "manual",
                "num_ctx": 8192,
                "num_batch": 1024,
                "model_concurrency": 2,
                "tool_workers": 4,
            },
            environ={},
            hardware=LOW_VRAM_HARDWARE,
        )

        self.assertEqual(config.profile, RuntimeProfile.MANUAL)
        self.assertEqual(config.num_ctx, 8192)
        self.assertTrue(any("context size" in warning for warning in config.warnings))
        self.assertTrue(any("model concurrency" in warning for warning in config.warnings))

    def test_malformed_user_config_is_preserved_with_controlled_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime_config.json"
            original = '{"num_ctx": '
            path.write_text(original, encoding="utf-8")

            result = load_user_config(path)

            self.assertEqual(result.values, {})
            self.assertTrue(result.warnings)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_manager_loads_actual_explicit_user_config_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime_config.json"
            path.write_text(json.dumps({"profile": "low-vram", "num_ctx": 3072}), encoding="utf-8")
            manager = RuntimeConfigManager()
            environment = {"SELENE_CONFIG_FILE": str(path)}
            with patch("agent.runtime_config.detect_hardware", return_value=LOW_VRAM_HARDWARE):
                config = manager.resolve(environ=environment)

            self.assertEqual(config.num_ctx, 3072)
            self.assertEqual(config.sources["num_ctx"], "user config")

    def test_hardware_detection_uses_smallest_visible_nvidia_gpu(self):
        result = Mock(
            returncode=0,
            stdout="RTX 4090, 24564\nRTX 3050 Ti Laptop GPU, 4096\n",
        )
        with patch("agent.runtime_config.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "agent.runtime_config.subprocess.run", return_value=result
        ) as run:
            detected = detect_hardware(timeout_seconds=0.5)

        self.assertEqual(detected.gpu_name, "RTX 3050 Ti Laptop GPU")
        self.assertEqual(detected.gpu_vram_mb, 4096)
        self.assertEqual(run.call_args.kwargs["timeout"], 0.5)
        self.assertNotIn("shell", run.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
