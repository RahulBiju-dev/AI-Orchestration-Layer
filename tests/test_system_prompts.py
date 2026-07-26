from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.model_providers import chat_with_model
from agent.system_prompts import (
    active_system_prompt_for_session,
    apply_active_system_prompt,
    default_system_prompt_for_model,
    extract_local_system_prompt,
    load_external_system_prompt,
)


def runtime():
    return SimpleNamespace(chat_model="selene", chat_timeout_seconds=30.0)


class SystemPromptPolicyTests(unittest.TestCase):
    def test_desktop_backend_bundle_includes_external_prompt_asset(self):
        root = Path(__file__).resolve().parents[1]
        spec = (root / "selene-backend.spec").read_text(encoding="utf-8")
        self.assertIn(
            "('agent/prompts/external_models.md', 'agent/prompts')",
            spec,
        )

    def test_desktop_backend_bundle_collects_dynamic_chromadb_modules(self):
        root = Path(__file__).resolve().parents[1]
        spec = (root / "selene-backend.spec").read_text(encoding="utf-8")
        self.assertIn("chromadb_hiddenimports = collect_all(", spec)
        self.assertIn("'chromadb',", spec)
        self.assertIn("not name.startswith('chromadb.test')", spec)
        self.assertIn("binaries=chromadb_binaries", spec)
        self.assertIn("*chromadb_datas", spec)
        self.assertIn("*chromadb_hiddenimports", spec)

    def test_local_and_external_models_have_distinct_owned_defaults(self):
        local = extract_local_system_prompt()
        external = load_external_system_prompt()

        self.assertIn("You are Selene, a precise local assistant", local)
        self.assertIn("You are Selene, a precise assistant", external)
        self.assertNotEqual(local, external)
        self.assertGreater(len(external), len(local) * 2)
        self.assertIn("## Reasoning discipline", external)
        self.assertIn("## Tool-loop control and recovery", external)
        self.assertIn("## Enhanced modes", external)

    def test_model_default_and_explicit_override_precedence(self):
        local = "LOCAL POLICY"
        self.assertEqual(
            default_system_prompt_for_model(
                "local:default",
                local_prompt=local,
            ),
            local,
        )
        self.assertEqual(
            default_system_prompt_for_model("gemini:gemini-2.5-flash"),
            load_external_system_prompt(),
        )
        self.assertEqual(
            active_system_prompt_for_session({
                "model_id": "gemini:gemini-2.5-flash",
                "system": "CONVERSATION OVERRIDE",
            }),
            "CONVERSATION OVERRIDE",
        )

    def test_anchored_system_copies_are_replaced_together(self):
        messages = [
            {"role": "system", "content": "old"},
            {"role": "user", "content": "question"},
            {"role": "system", "content": "old"},
            {"role": "user", "content": "current"},
        ]
        routed = apply_active_system_prompt(messages, "new")

        self.assertEqual(
            [
                message["content"]
                for message in routed
                if message["role"] == "system"
            ],
            ["new", "new"],
        )
        self.assertEqual(messages[0]["content"], "old")

    def test_local_execution_reasserts_modelfile_prompt_after_api_fallback(self):
        service = MagicMock()
        service.chat.return_value = {"message": {"content": "ok"}}
        factory = MagicMock(return_value=service)

        chat_with_model(
            {"model_id": "local:default", "system": ""},
            runtime(),
            ollama_service_factory=factory,
            messages=[
                {"role": "system", "content": load_external_system_prompt()},
                {"role": "user", "content": "continue"},
            ],
            stream=False,
        )

        sent = service.chat.call_args.kwargs["messages"]
        self.assertEqual(sent[0]["content"], extract_local_system_prompt())

    def test_external_execution_uses_large_prompt_without_history_prompt(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
        }
        response.close.return_value = None
        with patch(
            "agent.model_providers.requests.post",
            return_value=response,
        ) as post:
            chat_with_model(
                {"model_id": "openrouter:default", "system": ""},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "OPENROUTER_API_KEY": "secret",
                    "OPENROUTER_MODEL": "openrouter/free",
                },
                messages=[{"role": "user", "content": "hello"}],
                stream=False,
            )

        sent = post.call_args.kwargs["json"]["messages"]
        self.assertEqual(sent[0]["content"], load_external_system_prompt())

    def test_non_chat_operation_keeps_its_task_specific_system_prompt(self):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "Short Title"}}],
        }
        response.close.return_value = None
        with patch(
            "agent.model_providers.requests.post",
            return_value=response,
        ) as post:
            chat_with_model(
                {"model_id": "openrouter:default", "system": ""},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "OPENROUTER_API_KEY": "secret",
                    "OPENROUTER_MODEL": "openrouter/free",
                },
                kind=SimpleNamespace(value="title"),
                messages=[{
                    "role": "system",
                    "content": "Return only a short title.",
                }],
                stream=False,
            )

        sent = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("Return only a short title.", sent)
        self.assertNotIn("## Tool selection and execution", sent)


if __name__ == "__main__":
    unittest.main()
