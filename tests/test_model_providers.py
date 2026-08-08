from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from agent.environment import load_dotenv
from agent.model_providers import (
    ERROR_FALLBACK_MODEL_IDS,
    GEMINI_FREE_CHAT_MODELS,
    InvalidProviderKeyError,
    MalformedProviderResponse,
    MissingProviderConfiguration,
    ModelProviderError,
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    UnsupportedModelError,
    available_models,
    build_model_registry,
    chat_with_model,
    normalize_provider_text,
    repair_text_encoding,
    resolve_model,
    resolve_error_fallback,
)
from agent.runtime_config import RuntimeConfigurationError


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "agent" / "static" / "app.js").read_text(encoding="utf-8")
HTML = (ROOT / "agent" / "static" / "index.html").read_text(encoding="utf-8")
STYLE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted((ROOT / "agent" / "static" / "css").glob("*.css"))
)
WEB = (ROOT / "agent" / "web.py").read_text(encoding="utf-8")


def runtime():
    return SimpleNamespace(chat_model="selene", chat_timeout_seconds=30.0)


class FakeResponse:
    def __init__(self, payload=None, *, status=200, lines=None):
        self.status_code = status
        self._payload = payload
        self._lines = lines or []
        self.closed = False

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def iter_lines(self, decode_unicode=True):
        del decode_unicode
        yield from self._lines

    def close(self):
        self.closed = True


class EnvironmentTests(unittest.TestCase):
    def test_dotenv_loader_is_server_side_and_does_not_override_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text(
                "# provider config\nexport OPENROUTER_API_KEY='file-key'\nCUSTOM_LLM_BASE_URL=http://localhost:8000/v1\n",
                encoding="utf-8",
            )
            environment = {"OPENROUTER_API_KEY": "exported-key"}
            loaded = load_dotenv(path, environ=environment)

        self.assertEqual(loaded, path)
        self.assertEqual(environment["OPENROUTER_API_KEY"], "exported-key")
        self.assertEqual(environment["CUSTOM_LLM_BASE_URL"], "http://localhost:8000/v1")

    def test_env_example_lists_every_registered_free_gemini_chat_model(self):
        example = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        configured = next(line for line in example if line.startswith("GEMINI_MODELS="))
        self.assertEqual(
            tuple(configured.partition("=")[2].split(",")),
            GEMINI_FREE_CHAT_MODELS,
        )

    def test_env_example_exposes_nvidia_multi_model_configuration(self):
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("NVIDIA_API_KEY=", example)
        self.assertIn("NVIDIA_MODELS=", example)
        self.assertIn("NVIDIA_BASE_URL=", example)
        self.assertIn("NVIDIA_CONTEXT_WINDOW=", example)


class RegistryTests(unittest.TestCase):
    def test_error_fallback_chain_uses_configured_remote_tiers_then_local(self):
        environment = {
            "GEMINI_API_KEY": "google-secret",
            "GEMINI_MODELS": (
                "gemini-3.5-flash,gemini-3.5-flash-lite,gemma-4-31b-it"
            ),
            "NVIDIA_API_KEY": "nvidia-secret",
            "NVIDIA_MODELS": "nvidia/nemotron-3-ultra-550b-a55b",
        }
        self.assertEqual(
            ERROR_FALLBACK_MODEL_IDS,
            (
                "gemini:gemini-3.5-flash-lite",
                "nvidia:nvidia/nemotron-3-ultra-550b-a55b",
                "gemini:gemma-4-31b-it",
                "local:default",
            ),
        )

        attempted: list[str] = []
        current = "gemini:gemini-3.5-flash"
        for expected in ERROR_FALLBACK_MODEL_IDS:
            fallback = resolve_error_fallback(
                current,
                runtime(),
                environment,
                attempted_model_ids=attempted,
            )
            self.assertIsNotNone(fallback)
            self.assertEqual(fallback.id, expected)
            attempted.append(fallback.id)
            current = fallback.id

        self.assertIsNone(
            resolve_error_fallback(
                "local:default",
                runtime(),
                environment,
                attempted_model_ids=attempted,
            )
        )

    def test_error_fallback_chain_skips_unconfigured_remote_tiers(self):
        fallback = resolve_error_fallback(
            "gemini:gemini-3.5-flash",
            runtime(),
            {"GEMINI_API_KEY": "google-secret", "GEMINI_MODELS": "gemini-3.5-flash"},
        )
        self.assertEqual(fallback.id, "local:default")

    def test_only_configured_remote_models_are_available(self):
        environment = {
            "OPENROUTER_API_KEY": "secret",
            "OPENROUTER_MODEL": "openrouter/free",
            "GEMINI_API_KEY": "google-secret",
            "GEMINI_MODEL": "gemini-2.5-flash",
            "CUSTOM_LLM_BASE_URL": "http://localhost:8000/v1",
            "CUSTOM_LLM_MODEL": "local-custom",
        }
        models = available_models(runtime(), environment)
        ids = [model["id"] for model in models]
        self.assertEqual(
            ids,
            [
                "local:default",
                "openrouter:openrouter/free",
                "gemini:gemini-2.5-flash",
                "custom:default",
            ],
        )
        self.assertEqual(
            [model["display_name"] for model in models],
            ["Gemma 4 E4B", "openrouter/free", "gemini-2.5-flash", "local-custom"],
        )
        self.assertTrue(all("Selene" not in model["display_name"] for model in models))
        self.assertEqual(models[2]["context_window"], 1_048_576)
        self.assertNotIn("secret", json.dumps(models))
        self.assertNotIn("endpoint", models[1])

    def test_remote_models_without_explicit_model_identifiers_are_hidden(self):
        models = available_models(
            runtime(),
            {
                "OPENROUTER_API_KEY": "secret",
                "GEMINI_API_KEY": "google-secret",
            },
        )
        self.assertEqual([model["id"] for model in models], ["local:default"])

    def test_one_gemini_key_exposes_each_configured_free_model(self):
        models = available_models(
            runtime(),
            {
                "GEMINI_API_KEY": "google-secret",
                "GEMINI_MODELS": (
                    "gemini-3.5-flash,gemini-2.5-flash-lite,gemini-3.5-flash"
                ),
            },
        )
        self.assertEqual(
            [model["id"] for model in models],
            [
                "local:default",
                "gemini:gemini-3.5-flash",
                "gemini:gemini-2.5-flash-lite",
            ],
        )

    def test_one_openrouter_key_exposes_each_configured_free_model(self):
        models = available_models(
            runtime(),
            {
                "OPENROUTER_API_KEY": "secret",
                "OPENROUTER_MODELS": (
                    "openrouter/free,nvidia/nemotron-3-ultra-550b-a55b:free,openrouter/free"
                ),
            },
        )
        self.assertEqual(
            [model["id"] for model in models],
            [
                "local:default",
                "openrouter:openrouter/free",
                "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free",
            ],
        )
        self.assertTrue(all(
            model["provider"] == "OpenRouter" for model in models[1:]
        ))

    def test_one_nvidia_key_exposes_each_configured_model(self):
        models = available_models(
            runtime(),
            {
                "NVIDIA_API_KEY": "secret",
                "NVIDIA_MODELS": (
                    "nvidia/llama-3.1-nemotron-ultra-253b-v1,"
                    "meta/llama-3.3-70b-instruct,"
                    "nvidia/llama-3.1-nemotron-ultra-253b-v1"
                ),
            },
        )
        self.assertEqual(
            [model["id"] for model in models],
            [
                "local:default",
                "nvidia:nvidia/llama-3.1-nemotron-ultra-253b-v1",
                "nvidia:meta/llama-3.3-70b-instruct",
            ],
        )
        self.assertTrue(all(model["provider"] == "NVIDIA" for model in models[1:]))
        self.assertNotIn("secret", json.dumps(models))

    def test_nvidia_models_are_hidden_until_key_and_model_list_are_set(self):
        environment = {
            "NVIDIA_MODELS": "meta/llama-3.3-70b-instruct",
        }
        self.assertEqual(
            [model["id"] for model in available_models(runtime(), environment)],
            ["local:default"],
        )
        with self.assertRaises(MissingProviderConfiguration) as error:
            resolve_model(
                "nvidia:meta/llama-3.3-70b-instruct", runtime(), environment
            )
        self.assertIn("NVIDIA_API_KEY", str(error.exception))

    def test_legacy_openrouter_selection_uses_first_available_configured_model(self):
        model = resolve_model(
            "openrouter:default",
            runtime(),
            {
                "OPENROUTER_API_KEY": "secret",
                "OPENROUTER_MODELS": "vendor/paid,nvidia/nemotron:free",
            },
        )
        self.assertEqual(model.id, "openrouter:nvidia/nemotron:free")

    def test_gemini_registry_uses_each_models_native_context_maximum(self):
        models = available_models(
            runtime(),
            {
                "GEMINI_API_KEY": "google-secret",
                "GEMINI_MODELS": "gemini-3.5-flash,gemma-4-31b-it",
            },
        )
        self.assertEqual(models[1]["context_window"], 1_048_576)
        self.assertEqual(models[2]["context_window"], 262_144)

    def test_remote_context_override_is_validated_and_client_safe(self):
        model = resolve_model(
            "custom:default",
            runtime(),
            {
                "CUSTOM_LLM_BASE_URL": "http://localhost:8000/v1",
                "CUSTOM_LLM_MODEL": "local-custom",
                "CUSTOM_LLM_CONTEXT_WINDOW": "65536",
            },
        )
        self.assertEqual(model.context_window, 65_536)
        with self.assertRaises(MissingProviderConfiguration):
            resolve_model(
                "custom:default",
                runtime(),
                {
                    "CUSTOM_LLM_BASE_URL": "http://localhost:8000/v1",
                    "CUSTOM_LLM_MODEL": "local-custom",
                    "CUSTOM_LLM_CONTEXT_WINDOW": "not-a-number",
                },
            )

    def test_non_free_gemini_model_is_not_available(self):
        environment = {
            "GEMINI_API_KEY": "google-secret",
            "GEMINI_MODELS": "gemini-3.1-pro-preview",
        }
        self.assertEqual(
            [model["id"] for model in available_models(runtime(), environment)],
            ["local:default"],
        )
        with self.assertRaises(UnsupportedModelError):
            resolve_model("gemini:gemini-3.1-pro-preview", runtime(), environment)

    def test_missing_key_and_unknown_model_have_safe_errors(self):
        with self.assertRaises(MissingProviderConfiguration) as missing:
            resolve_model("openrouter:default", runtime(), {})
        self.assertIn("OPENROUTER_API_KEY", str(missing.exception))
        with self.assertRaises(UnsupportedModelError):
            resolve_model("unregistered:model", runtime(), {})

    def test_registry_keeps_configuration_metadata_centralized(self):
        registry = build_model_registry(runtime(), {})
        self.assertEqual(registry["openrouter:default"].api_key_env, "OPENROUTER_API_KEY")
        self.assertEqual(registry["nvidia:default"].api_key_env, "NVIDIA_API_KEY")
        self.assertEqual(registry["gemini:default"].api_key_env, "GEMINI_API_KEY")
        self.assertEqual(registry["custom:default"].provider, "OpenAI-compatible / self-hosted")
        self.assertNotIn("openai:default", registry)
        self.assertNotIn("anthropic:default", registry)

    def test_openrouter_rejects_non_free_model_configuration(self):
        with self.assertRaises(UnsupportedModelError) as error:
            resolve_model(
                "openrouter:default",
                runtime(),
                {
                    "OPENROUTER_API_KEY": "secret",
                    "OPENROUTER_MODEL": "vendor/paid-model",
                },
            )
        self.assertIn(":free", str(error.exception))

    def test_web_session_validation_persists_a_configured_model(self):
        from agent import web

        with patch.dict(
            "os.environ",
            {"OPENROUTER_API_KEY": "secret", "OPENROUTER_MODEL": "openrouter/free"},
            clear=True,
        ):
            normalized = web._normalize_session_settings({"model_id": "openrouter:default"})
        self.assertEqual(normalized["model_id"], "openrouter:openrouter/free")

    def test_web_session_validation_uses_standard_error_for_stale_model(self):
        from agent import web

        with (
            patch.dict("os.environ", {}, clear=True),
            self.assertRaises(RuntimeConfigurationError) as error,
        ):
            web._normalize_session_settings({"model_id": "openrouter:default"})
        self.assertIn("OPENROUTER_API_KEY", str(error.exception))


class AdapterTests(unittest.TestCase):
    def test_nvidia_uses_openai_compatible_endpoint_with_selected_model(self):
        response = FakeResponse({
            "choices": [{"message": {"content": "NVIDIA answer."}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3},
        })
        with patch("agent.model_providers.requests.post", return_value=response) as post:
            result = chat_with_model(
                {"model_id": "nvidia:meta/llama-3.3-70b-instruct"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "NVIDIA_API_KEY": "nvidia-secret",
                    "NVIDIA_MODELS": "meta/llama-3.3-70b-instruct",
                },
                messages=[{"role": "user", "content": "hello"}],
                stream=False,
            )

        self.assertEqual(result.message.content, "NVIDIA answer.")
        self.assertEqual(
            post.call_args.args[0],
            "https://integrate.api.nvidia.com/v1/chat/completions",
        )
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"],
            "Bearer nvidia-secret",
        )
        self.assertEqual(
            post.call_args.kwargs["json"]["model"],
            "meta/llama-3.3-70b-instruct",
        )
        self.assertIn(
            "You are Selene",
            post.call_args.kwargs["json"]["messages"][0]["content"],
        )
        self.assertNotIn("chat_template_kwargs", post.call_args.kwargs["json"])

    def test_nemotron_uses_required_coding_content_flags(self):
        response = FakeResponse({
            "choices": [{"message": {"content": "NVIDIA answer."}}],
        })
        with patch("agent.model_providers.requests.post", return_value=response) as post:
            chat_with_model(
                {"model_id": "nvidia:nvidia/nemotron-3-ultra-550b-a55b"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "NVIDIA_API_KEY": "nvidia-secret",
                    "NVIDIA_MODELS": "nvidia/nemotron-3-ultra-550b-a55b",
                },
                messages=[{"role": "user", "content": "show code"}],
                stream=False,
                think=False,
            )

        self.assertEqual(
            post.call_args.kwargs["json"]["chat_template_kwargs"],
            {
                "enable_thinking": False,
                "force_nonempty_content": True,
            },
        )

    def test_nemotron_repairs_truncated_markdown_fence_non_streaming(self):
        response = FakeResponse({
            "choices": [{
                "message": {
                    "content": "down\n# Personal Operating Manual\n```",
                },
            }],
        })
        with patch("agent.model_providers.requests.post", return_value=response):
            result = chat_with_model(
                {"model_id": "nvidia:nvidia/nemotron-3-ultra-550b-a55b"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "NVIDIA_API_KEY": "nvidia-secret",
                    "NVIDIA_MODELS": "nvidia/nemotron-3-ultra-550b-a55b",
                },
                messages=[{"role": "user", "content": "put it in a code block"}],
                stream=False,
            )

        self.assertEqual(
            result.message.content,
            "```markdown\n# Personal Operating Manual\n```",
        )

    def test_nemotron_does_not_rewrite_down_without_a_fence_request(self):
        response = FakeResponse({
            "choices": [{
                "message": {
                    "content": "down\n# The hill and back again",
                },
            }],
        })
        with patch("agent.model_providers.requests.post", return_value=response):
            result = chat_with_model(
                {"model_id": "nvidia:nvidia/nemotron-3-ultra-550b-a55b"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "NVIDIA_API_KEY": "nvidia-secret",
                    "NVIDIA_MODELS": "nvidia/nemotron-3-ultra-550b-a55b",
                },
                messages=[{"role": "user", "content": "write a heading about walking"}],
                stream=False,
            )

        self.assertEqual(result.message.content, "down\n# The hill and back again")

    def test_nemotron_repairs_split_stream_markdown_fence_prefix(self):
        response = FakeResponse(lines=[
            'data: {"choices":[{"delta":{"content":"do"}}]}',
            'data: {"choices":[{"delta":{"content":"wn\\n# Personal Operating Manual\\n"}}]}',
            'data: {"choices":[{"delta":{"content":"```"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ])
        with patch("agent.model_providers.requests.post", return_value=response):
            chunks = list(chat_with_model(
                {"model_id": "nvidia:nvidia/nemotron-3-ultra-550b-a55b"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "NVIDIA_API_KEY": "nvidia-secret",
                    "NVIDIA_MODELS": "nvidia/nemotron-3-ultra-550b-a55b",
                },
                messages=[{"role": "user", "content": "put it in a code block"}],
                stream=True,
            ))

        self.assertEqual(
            "".join(chunk.message.content for chunk in chunks),
            "```markdown\n# Personal Operating Manual\n```",
        )

    def test_mojibake_is_repaired_without_changing_valid_unicode(self):
        self.assertEqual(repair_text_encoding("High (â‚¹1,00,000+)"), "High (₹1,00,000+)")
        self.assertEqual(repair_text_encoding("FranÃ§ais â€” résumé"), "Français — résumé")
        self.assertEqual(repair_text_encoding("₹ déjà correct"), "₹ déjà correct")

    def test_provider_sse_bytes_are_always_decoded_as_utf8(self):
        event = json.dumps(
            {"choices": [{"delta": {"content": "High (₹1,00,000+)"}}]},
            ensure_ascii=False,
        ).encode("utf-8")
        response = FakeResponse(lines=[b"data: " + event, b"data: [DONE]"])
        with patch("agent.model_providers.requests.post", return_value=response):
            chunks = list(chat_with_model(
                {"model_id": "openrouter:default"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "OPENROUTER_API_KEY": "secret",
                    "OPENROUTER_MODEL": "openrouter/free",
                },
                messages=[{"role": "user", "content": "salary"}],
                stream=True,
            ))
        self.assertEqual(chunks[0].message.content, "High (₹1,00,000+)")

    def test_structured_provider_content_is_normalized_to_display_text(self):
        content = [
            {"type": "text", "text": "Hello **world**"},
            {"type": "output_text", "text": "Second paragraph"},
        ]
        self.assertEqual(
            normalize_provider_text(content),
            "Hello **world**\nSecond paragraph",
        )

        response = FakeResponse({
            "choices": [{"message": {"content": content}}],
        })
        with patch("agent.model_providers.requests.post", return_value=response):
            result = chat_with_model(
                {"model_id": "openrouter:default"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "OPENROUTER_API_KEY": "secret",
                    "OPENROUTER_MODEL": "openrouter/free",
                },
                messages=[{"role": "user", "content": "hello"}],
                stream=False,
            )
        self.assertEqual(result.message.content, "Hello **world**\nSecond paragraph")

    def test_local_model_preserves_existing_ollama_service(self):
        service = MagicMock()
        service.chat.return_value = "local-result"
        result = chat_with_model(
            {"model_id": "local:default"},
            runtime(),
            ollama_service_factory=lambda _runtime: service,
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )
        self.assertEqual(result, "local-result")
        self.assertEqual(service.chat.call_args.kwargs["model"], "selene")

    def test_openai_compatible_response_is_normalized_and_keeps_selene_identity(self):
        response = FakeResponse({
            "choices": [{
                "message": {
                    "content": "",
                    "planning_content": "Search first, then compare the sources.",
                    "tool_calls": [{
                        "function": {"name": "web_search", "arguments": '{"query":"test"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        })
        with patch("agent.model_providers.requests.post", return_value=response) as post:
            result = chat_with_model(
                {"model_id": "openrouter:default"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "OPENROUTER_API_KEY": "top-secret",
                    "OPENROUTER_MODEL": "openrouter/free",
                },
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
                stream=False,
            )

        self.assertEqual(result.message.tool_calls[0].function.name, "web_search")
        self.assertEqual(result.message.tool_calls[0].function.arguments, {"query": "test"})
        self.assertEqual(
            result.message.thinking,
            "Search first, then compare the sources.",
        )
        self.assertEqual(result.prompt_eval_count, 10)
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer top-secret")
        sent_messages = post.call_args.kwargs["json"]["messages"]
        self.assertEqual(sent_messages[0]["role"], "system")
        self.assertIn("You are Selene", sent_messages[0]["content"])
        self.assertNotIn("top-secret", str(result))

    def test_external_model_preserves_the_active_system_prompt_under_selene_identity(self):
        response = FakeResponse({"choices": [{"message": {"content": "ok"}}]})
        with patch("agent.model_providers.requests.post", return_value=response) as post:
            chat_with_model(
                {
                    "model_id": "openrouter:default",
                    "system": "Use the active tool policy.",
                },
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "OPENROUTER_API_KEY": "secret",
                    "OPENROUTER_MODEL": "openrouter/free",
                },
                messages=[
                    {"role": "system", "content": "Use the active tool policy."},
                    {"role": "user", "content": "hello"},
                ],
                stream=False,
            )

        system_prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("You are Selene", system_prompt)
        self.assertIn("Use the active tool policy.", system_prompt)

    def test_openai_stream_accumulates_tool_arguments(self):
        lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"web_","arguments":"{\\"query\\":"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"search","arguments":"\\"test\\"}"}}]},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]
        response = FakeResponse(lines=lines)
        with patch("agent.model_providers.requests.post", return_value=response):
            chunks = list(chat_with_model(
                {"model_id": "openrouter:default"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "OPENROUTER_API_KEY": "secret",
                    "OPENROUTER_MODEL": "openrouter/free",
                },
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            ))
        call = chunks[-1].message.tool_calls[0].function
        self.assertEqual(call.name, "web_search")
        self.assertEqual(call.arguments, {"query": "test"})
        self.assertTrue(response.closed)

    def test_openai_stream_routes_provider_planning_into_thinking(self):
        response = FakeResponse(lines=[
            'data: {"choices":[{"delta":{"planning_content":"Inspect, then edit."}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ])
        with patch("agent.model_providers.requests.post", return_value=response):
            chunks = list(chat_with_model(
                {"model_id": "openrouter:default"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "OPENROUTER_API_KEY": "secret",
                    "OPENROUTER_MODEL": "openrouter/free",
                },
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            ))
        self.assertEqual(chunks[0].message.thinking, "Inspect, then edit.")

    def test_openrouter_stream_maps_inline_rate_limit_instead_of_empty_stream(self):
        response = FakeResponse(lines=[
            'data: {"error":{"code":429,"message":"Rate limit exceeded",'
            '"metadata":{"error_type":"rate_limit_exceeded"}}}',
            "data: [DONE]",
        ])
        with patch("agent.model_providers.requests.post", return_value=response):
            with self.assertRaises(ProviderRateLimitError) as error:
                list(chat_with_model(
                    {"model_id": "openrouter:default"},
                    runtime(),
                    ollama_service_factory=MagicMock(),
                    environ={
                        "OPENROUTER_API_KEY": "never-print-this",
                        "OPENROUTER_MODEL": "nvidia/nemotron-3-ultra-550b-a55b:free",
                    },
                    messages=[{"role": "user", "content": "hello"}],
                    stream=True,
                ))

        self.assertEqual(error.exception.code, "rate_limited")
        self.assertIn("free-capacity limit", str(error.exception))
        self.assertNotIn("never-print-this", str(error.exception))

    def test_openrouter_stream_maps_inline_provider_unavailability(self):
        response = FakeResponse(lines=[
            'data: {"error":{"code":502,"message":"Provider returned empty response",'
            '"metadata":{"error_type":"provider_unavailable"}},"choices":'
            '[{"delta":{"content":""},"finish_reason":"error"}]}',
            "data: [DONE]",
        ])
        with patch("agent.model_providers.requests.post", return_value=response):
            with self.assertRaises(ModelProviderError) as error:
                list(chat_with_model(
                    {"model_id": "openrouter:default"},
                    runtime(),
                    ollama_service_factory=MagicMock(),
                    environ={
                        "OPENROUTER_API_KEY": "secret",
                        "OPENROUTER_MODEL": "nvidia/nemotron-3-ultra-550b-a55b:free",
                    },
                    messages=[{"role": "user", "content": "hello"}],
                    stream=True,
                ))

        self.assertEqual(error.exception.code, "provider_unavailable")
        self.assertIn("no available capacity", str(error.exception))
        self.assertNotIn("empty response stream", str(error.exception))

    def test_openrouter_stream_exposes_reasoning_details(self):
        reasoning_details = [
            {"type": "reasoning.summary", "summary": "Check the constraints."},
            {"type": "reasoning.encrypted", "data": "opaque"},
        ]
        response = FakeResponse(lines=[
            'data: {"choices":[{"delta":{"reasoning_details":['
            '{"type":"reasoning.summary","summary":"Check the constraints."},'
            '{"type":"reasoning.encrypted","data":"opaque"}]}}]}',
            'data: {"choices":[{"delta":{"content":"Done."}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ])
        with patch("agent.model_providers.requests.post", return_value=response):
            chunks = list(chat_with_model(
                {"model_id": "openrouter:default"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "OPENROUTER_API_KEY": "secret",
                    "OPENROUTER_MODEL": "nvidia/nemotron-3-ultra-550b-a55b:free",
                },
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            ))

        self.assertEqual(chunks[0].message.thinking, "Check the constraints.")
        self.assertEqual(chunks[1].message.content, "Done.")
        self.assertEqual(
            chunks[-1].message.provider_metadata["reasoning_details"],
            reasoning_details,
        )

    def test_openrouter_replays_reasoning_details_on_follow_up(self):
        response = FakeResponse({
            "choices": [{"message": {"content": "continued"}}],
        })
        reasoning_details = [{
            "type": "reasoning.encrypted",
            "data": "opaque-provider-state",
            "id": "reasoning-1",
        }]
        with patch("agent.model_providers.requests.post", return_value=response) as post:
            chat_with_model(
                {"model_id": "openrouter:default"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "OPENROUTER_API_KEY": "secret",
                    "OPENROUTER_MODEL": "nvidia/nemotron-3-ultra-550b-a55b:free",
                },
                messages=[
                    {"role": "user", "content": "first"},
                    {
                        "role": "assistant",
                        "content": "first answer",
                        "provider_metadata": {"reasoning_details": reasoning_details},
                    },
                    {"role": "user", "content": "follow up"},
                ],
                stream=False,
            )

        sent_assistant = post.call_args.kwargs["json"]["messages"][2]
        self.assertEqual(sent_assistant["reasoning_details"], reasoning_details)
        self.assertIsNot(sent_assistant["reasoning_details"], reasoning_details)

    def test_custom_openai_endpoint_does_not_receive_openrouter_reasoning_metadata(self):
        response = FakeResponse({
            "choices": [{"message": {"content": "continued"}}],
        })
        with patch("agent.model_providers.requests.post", return_value=response) as post:
            chat_with_model(
                {"model_id": "custom:default"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "CUSTOM_LLM_BASE_URL": "http://localhost:8000/v1",
                    "CUSTOM_LLM_MODEL": "custom-model",
                },
                messages=[{
                    "role": "assistant",
                    "content": "first answer",
                    "provider_metadata": {
                        "reasoning_details": [{"type": "reasoning.text", "text": "private"}],
                    },
                }],
                stream=False,
            )

        sent_assistant = post.call_args.kwargs["json"]["messages"][1]
        self.assertNotIn("reasoning_details", sent_assistant)

    def test_openrouter_no_content_response_has_warmup_guidance(self):
        response = FakeResponse(lines=[
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ])
        with patch("agent.model_providers.requests.post", return_value=response):
            with self.assertRaises(MalformedProviderResponse) as error:
                list(chat_with_model(
                    {"model_id": "openrouter:default"},
                    runtime(),
                    ollama_service_factory=MagicMock(),
                    environ={
                        "OPENROUTER_API_KEY": "secret",
                        "OPENROUTER_MODEL": "nvidia/nemotron-3-ultra-550b-a55b:free",
                    },
                    messages=[{"role": "user", "content": "hello"}],
                    stream=True,
                ))

        self.assertEqual(error.exception.code, "empty_response")
        self.assertIn("warming up", str(error.exception))

    def test_gemini_response_is_normalized_and_keeps_selene_identity(self):
        response = FakeResponse({
            "candidates": [{
                "content": {"parts": [{
                    "thought": True,
                    "thoughtType": "PLAN",
                    "text": "Check current state, then use the tool.",
                }, {
                    "functionCall": {
                        "id": "gemini-call-1",
                        "name": "web_search",
                        "args": {"query": "test"},
                    },
                    "thoughtSignature": "opaque-signature",
                }]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 3},
        })
        with patch("agent.model_providers.requests.post", return_value=response) as post:
            result = chat_with_model(
                {
                    "model_id": "gemini:default",
                    "system": "Use the active tool policy.",
                },
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "GEMINI_API_KEY": "google-secret",
                    "GEMINI_MODEL": "gemini-2.5-flash",
                },
                messages=[
                    {"role": "system", "content": "Use the active tool policy."},
                    {"role": "user", "content": "hello"},
                ],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                }],
                stream=False,
            )

        call = result.message.tool_calls[0].function
        self.assertEqual(call.name, "web_search")
        self.assertEqual(call.arguments, {"query": "test"})
        self.assertEqual(result.message.tool_calls[0].id, "gemini-call-1")
        self.assertEqual(
            result.message.thinking,
            "Check current state, then use the tool.",
        )
        self.assertEqual(
            result.message.tool_calls[0].provider_metadata["thought_signature"],
            "opaque-signature",
        )
        self.assertEqual(result.prompt_eval_count, 8)
        self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], "google-secret")
        self.assertNotIn("google-secret", post.call_args.args[0])
        system = post.call_args.kwargs["json"]["systemInstruction"]["parts"][0]["text"]
        self.assertIn("You are Selene", system)
        self.assertIn("Use the active tool policy.", system)
        declaration = post.call_args.kwargs["json"]["tools"][0]["functionDeclarations"][0]
        self.assertIn("parametersJsonSchema", declaration)
        self.assertNotIn("parameters", declaration)
        self.assertIs(declaration["parametersJsonSchema"]["additionalProperties"], False)

    def test_gemini_replays_function_id_and_thought_signature(self):
        response = FakeResponse({
            "candidates": [{"content": {"parts": [{"text": "done"}]}}],
        })
        with patch("agent.model_providers.requests.post", return_value=response) as post:
            chat_with_model(
                {"model_id": "gemini:default"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "GEMINI_API_KEY": "google-secret",
                    "GEMINI_MODEL": "gemini-2.5-flash",
                },
                messages=[
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "gemini-call-1",
                            "provider_metadata": {
                                "thought_signature": "opaque-signature",
                            },
                            "function": {
                                "name": "web_search",
                                "arguments": {"query": "test"},
                            },
                        }],
                    },
                    {"role": "tool", "content": '{"result":"ok"}'},
                ],
                stream=False,
            )

        contents = post.call_args.kwargs["json"]["contents"]
        function_call_part = contents[0]["parts"][0]
        function_response = contents[1]["parts"][0]["functionResponse"]
        self.assertEqual(function_call_part["thoughtSignature"], "opaque-signature")
        self.assertEqual(function_call_part["functionCall"]["id"], "gemini-call-1")
        self.assertEqual(function_response["id"], "gemini-call-1")

    def test_gemini_stream_is_normalized(self):
        response = FakeResponse(lines=[
            "event: message",
            "id: provider-event-1",
            "data: not-json-from-a-proxy",
            'data: {"candidates":[{"content":{"parts":['
            '{"thought":true,"text":"Check the facts first."},'
            '{"text":"Hello"}]}}]}',
            "retry: 1000",
            'data: {"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],"usageMetadata":{"candidatesTokenCount":2}}',
        ])
        with patch("agent.model_providers.requests.post", return_value=response) as post:
            chunks = list(chat_with_model(
                {"model_id": "gemini:default"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "GEMINI_API_KEY": "google-secret",
                    "GEMINI_MODEL": "gemini-2.5-flash",
                },
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            ))
        self.assertEqual(chunks[0].message.content, "Hello")
        self.assertEqual(chunks[0].message.thinking, "Check the facts first.")
        self.assertEqual(chunks[-1].done_reason, "STOP")
        self.assertTrue(
            post.call_args.kwargs["json"]["generationConfig"]
            ["thinkingConfig"]["includeThoughts"]
        )
        self.assertTrue(response.closed)

    def test_gemini_nothink_disables_returned_thought_summaries(self):
        response = FakeResponse({
            "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
        })
        with patch("agent.model_providers.requests.post", return_value=response) as post:
            chat_with_model(
                {"model_id": "gemini:default"},
                runtime(),
                ollama_service_factory=MagicMock(),
                environ={
                    "GEMINI_API_KEY": "google-secret",
                    "GEMINI_MODEL": "gemini-3.5-flash",
                },
                messages=[{"role": "user", "content": "hello"}],
                stream=False,
                think=False,
            )

        self.assertFalse(
            post.call_args.kwargs["json"]["generationConfig"]
            ["thinkingConfig"]["includeThoughts"]
        )

    def test_unreadable_stream_without_any_valid_event_has_a_clear_error(self):
        response = FakeResponse(lines=["event: message", "data: definitely-not-json"])
        with patch("agent.model_providers.requests.post", return_value=response):
            with self.assertRaisesRegex(MalformedProviderResponse, "unreadable response stream"):
                list(chat_with_model(
                    {"model_id": "gemini:default"},
                    runtime(),
                    ollama_service_factory=MagicMock(),
                    environ={
                        "GEMINI_API_KEY": "google-secret",
                        "GEMINI_MODEL": "gemini-2.5-flash",
                    },
                    messages=[{"role": "user", "content": "hello"}],
                    stream=True,
                ))

    def test_gemini_invalid_key_error_is_safe(self):
        response = FakeResponse({
            "error": {
                "status": "INVALID_ARGUMENT",
                "details": [{"reason": "API_KEY_INVALID"}],
            },
        }, status=400)
        with patch("agent.model_providers.requests.post", return_value=response):
            with self.assertRaises(InvalidProviderKeyError) as error:
                chat_with_model(
                    {"model_id": "gemini:default"},
                    runtime(),
                    ollama_service_factory=MagicMock(),
                    environ={
                        "GEMINI_API_KEY": "never-print-google-secret",
                        "GEMINI_MODEL": "gemini-2.5-flash",
                    },
                    messages=[],
                    stream=False,
                )
        self.assertNotIn("never-print-google-secret", str(error.exception))

    def test_gemini_bad_request_is_not_mislabeled_as_an_invalid_key(self):
        response = FakeResponse({
            "error": {"status": "INVALID_ARGUMENT", "message": "Unknown request field"},
        }, status=400)
        with patch("agent.model_providers.requests.post", return_value=response):
            with self.assertRaisesRegex(ModelProviderError, "request format or options") as error:
                chat_with_model(
                    {"model_id": "gemini:default"},
                    runtime(),
                    ollama_service_factory=MagicMock(),
                    environ={
                        "GEMINI_API_KEY": "never-print-google-secret",
                        "GEMINI_MODEL": "gemini-2.5-flash",
                    },
                    messages=[],
                    stream=False,
                )
        self.assertNotIsInstance(error.exception, InvalidProviderKeyError)
        self.assertNotIn("never-print-google-secret", str(error.exception))

    def test_gemini_temporary_unavailability_suggests_another_model(self):
        response = FakeResponse({}, status=503)
        with patch("agent.model_providers.requests.post", return_value=response):
            with self.assertRaises(ModelProviderError) as error:
                chat_with_model(
                    {"model_id": "gemini:default"},
                    runtime(),
                    ollama_service_factory=MagicMock(),
                    environ={
                        "GEMINI_API_KEY": "google-secret",
                        "GEMINI_MODEL": "gemini-3.5-flash",
                    },
                    messages=[],
                    stream=False,
                )
        self.assertEqual(error.exception.code, "provider_unavailable")
        self.assertIn("select another model", str(error.exception))

    def test_provider_failures_are_mapped_without_response_secrets(self):
        cases = [
            (FakeResponse({}, status=401), InvalidProviderKeyError),
            (FakeResponse({}, status=429), ProviderRateLimitError),
            (FakeResponse({}, status=504), ProviderTimeoutError),
            (FakeResponse(ValueError("bad json")), MalformedProviderResponse),
            (FakeResponse({"unexpected": "shape"}), MalformedProviderResponse),
        ]
        for response, exception in cases:
            with self.subTest(exception=exception.__name__), patch(
                "agent.model_providers.requests.post", return_value=response
            ):
                with self.assertRaises(exception):
                    chat_with_model(
                        {"model_id": "openrouter:default"}, runtime(),
                        ollama_service_factory=MagicMock(),
                        environ={
                            "OPENROUTER_API_KEY": "never-print-this",
                            "OPENROUTER_MODEL": "openrouter/free",
                        },
                        messages=[], stream=False,
                    )

        for failure, exception in [
            (requests.Timeout("secret timeout"), ProviderTimeoutError),
            (requests.ConnectionError("secret network"), ProviderNetworkError),
        ]:
            with self.subTest(exception=exception.__name__), patch(
                "agent.model_providers.requests.post", side_effect=failure
            ):
                with self.assertRaises(exception) as caught:
                    chat_with_model(
                        {"model_id": "openrouter:default"}, runtime(),
                        ollama_service_factory=MagicMock(),
                        environ={
                            "OPENROUTER_API_KEY": "never-print-this",
                            "OPENROUTER_MODEL": "openrouter/free",
                        },
                        messages=[], stream=False,
                    )
                self.assertNotIn("never-print-this", str(caught.exception))


class ModelSelectorFrontendTests(unittest.TestCase):
    def test_web_composer_up_arrow_recalls_previous_user_prompts(self):
        self.assertIn("promptRecall: {", APP)
        self.assertIn("if (handlePromptHistoryKeydown(event)) return", APP)
        self.assertIn("function promptHistory()", APP)
        self.assertIn("message?.role === \"user\"", APP)
        self.assertIn("function handlePromptHistoryKeydown(event)", APP)
        self.assertIn('event.key === "ArrowUp"', APP)
        self.assertIn('event.key === "ArrowDown"', APP)
        self.assertIn("if (el.input.value) return false", APP)
        self.assertIn("el.input.setSelectionRange(cursor, cursor)", APP)

    def test_prompt_edit_and_response_copy_actions_cover_saved_and_live_messages(self):
        self.assertIn('appendMessageActions(row.stack, "edit")', APP)
        self.assertIn('appendMessageActions(row.stack, "copy")', APP)
        self.assertIn('appendMessageActions(state.stream.assistantStack, "copy")', APP)
        self.assertIn("function editPromptMessage(button)", APP)
        self.assertIn("function copyResponseMessage(button)", APP)
        self.assertIn("function writeClipboardText(text)", APP)
        self.assertIn("node.dataset.messageText = text", APP)
        self.assertIn(".message-actions", STYLE)
        self.assertIn(".message-action", STYLE)

    def test_api_response_rendering_normalizes_parts_and_falls_back_to_plain_text(self):
        self.assertIn("function displayText(value)", APP)
        self.assertIn("function repairTextEncoding(value)", APP)
        self.assertIn("WINDOWS_1252_BYTES", APP)
        self.assertIn("function renderResponseHTML(value)", APP)
        self.assertIn("function renderResponseInto(node, value)", APP)
        self.assertIn("return escapeHTML(text).replace", APP)
        self.assertNotIn(
            "state.stream.assistantBubble.innerHTML = renderMarkdown",
            APP,
        )
        self.assertIn("function scheduleStreamRender({ immediate = false } = {})", APP)
        self.assertIn("requestAnimationFrame(render)", APP)
        self.assertIn("scheduleStreamRender({ immediate: true });", APP)

    def test_sse_json_errors_are_isolated_from_rendering_errors(self):
        stream_reader = APP[
            APP.index("async function readEventStream"):
            APP.index("function handleStreamEvent")
        ]
        self.assertIn("event = JSON.parse", stream_reader)
        self.assertIn("handleStreamEvent(event, generation);", stream_reader)
        self.assertLess(
            stream_reader.index("} catch {"),
            stream_reader.index("handleStreamEvent(event, generation);"),
        )

    def test_model_selector_precedes_mode_and_uses_persisted_settings(self):
        self.assertLess(HTML.index('id="model-picker"'), HTML.index('id="mode-picker"'))
        self.assertIn('<span class="model-picker-label">Model</span>', HTML)
        self.assertIn('model_id: "local:default"', APP)
        self.assertIn("state.settings.model_id = modelId", APP)
        self.assertIn("function updateModelUI()", APP)
        self.assertIn("el.modelTrigger.disabled = state.isGenerating", APP)

    def test_model_selector_has_mobile_layout_rules(self):
        self.assertIn(".model-picker", STYLE)
        self.assertIn("#model-label", STYLE)
        self.assertIn("flex-wrap: wrap", STYLE)

    def test_provider_errors_render_inside_assistant_response_bubbles(self):
        self.assertIn("function appendStreamError(detail)", APP)
        self.assertIn('state.stream.assistantBubble.className = "bubble error"', APP)
        self.assertIn("if (event.error) state.stream.assistantBubble.classList.add", APP)
        self.assertIn(".bubble.error", STYLE)

    def test_model_fallback_immediately_updates_the_web_model_selector(self):
        self.assertIn('case "model_fallback"', APP)
        self.assertIn("state.settings.model_id = fallbackModel.id", APP)
        self.assertIn("updateModelUI();", APP)
        self.assertIn("continuing automatically", APP)

    def test_model_switch_reprimes_the_visible_transcript(self):
        self.assertIn("if (Array.isArray(data.history))", APP)
        self.assertIn("data.context_compaction?.compacted", APP)
        self.assertIn("historyShrank", APP)

    def test_compaction_marker_is_not_rendered_as_an_assistant_reply(self):
        self.assertIn("message.metadata?.compacted", APP)
        self.assertIn("function appendCompactionNotice(", APP)
        self.assertIn('role: "notice"', APP)
        self.assertIn(".message.compaction-notice", STYLE)

    def test_planning_metadata_is_redirected_without_a_plan_block(self):
        self.assertIn('case "planning_start"', APP)
        self.assertIn('{ ...event, type: "thinking_start" }', APP)
        self.assertIn('...(message.planning ? [{ type: "thinking"', APP)
        self.assertNotIn('detailBlock("Planning", "plan"', APP)
        self.assertNotIn("planningBlock:", APP)
        self.assertNotIn(".detail-block.planning", STYLE)

    def test_conversations_keep_independent_background_generations(self):
        self.assertIn("generations: new Map()", APP)
        self.assertIn("function generationForSession(", APP)
        self.assertIn("function renderActiveConversation()", APP)
        self.assertIn("generation.events", APP)
        self.assertNotIn(
            "if (state.isGenerating || state.generation) stopGeneration({ refresh: false });",
            APP,
        )
        self.assertIn(".session-running", STYLE)
        self.assertIn("X-Selene-Session-Name", WEB)

    def test_profile_popup_only_follows_switching_back_to_local(self):
        load_state = APP[APP.index("async function loadState()") : APP.index("function updateStartupProfileCopy()")]
        self.assertNotIn("showStartupProfileDialog()", load_state)
        self.assertIn('previousModel !== "local:default"', APP)
        self.assertIn('modelId === "local:default"', APP)


class ModelSlashCommandTests(unittest.TestCase):
    def test_tui_slash_catalog_exposes_model_commands(self):
        from agent.core import CLI_SLASH_COMPLETIONS

        self.assertIn("/model", CLI_SLASH_COMPLETIONS)
        self.assertIn("/set model", CLI_SLASH_COMPLETIONS)
        self.assertIn("/model local:default", CLI_SLASH_COMPLETIONS)

    def test_terminal_model_command_selects_configured_external_model(self):
        from agent import core

        environment = {
            "GEMINI_API_KEY": "google-secret",
            "GEMINI_MODELS": "gemini-3.1-flash-lite",
        }
        session = core._new_session_state()
        with (
            patch.dict("os.environ", environment, clear=True),
            patch.object(core, "_refresh_tui_runtime_meta"),
        ):
            result = core._handle_command(
                "/model gemini:gemini-3.1-flash-lite", session, []
            )

        self.assertTrue(result)
        self.assertEqual(session["model_id"], "gemini:gemini-3.1-flash-lite")

    def test_terminal_turn_routes_selected_model_with_native_context(self):
        from agent import core

        environment = {
            "GEMINI_API_KEY": "google-secret",
            "GEMINI_MODELS": "gemini-3.1-flash-lite",
        }
        session = core._new_session_state()
        session["model_id"] = "gemini:gemini-3.1-flash-lite"
        with (
            patch.dict("os.environ", environment, clear=True),
            patch.object(
                core,
                "_stream_complete_response",
                return_value={"role": "assistant", "content": "done"},
            ) as stream,
            patch.object(core, "_check_and_compact_history"),
        ):
            core.process_user_turn("hello", session, [], "system")

        request_session = stream.call_args.kwargs["session"]
        self.assertEqual(request_session["model_id"], "gemini:gemini-3.1-flash-lite")
        self.assertEqual(request_session["options"]["num_ctx"], 1_048_576)
        sent_messages = stream.call_args.kwargs["messages"]
        external_prompt = next(
            message["content"]
            for message in sent_messages
            if message["role"] == "system"
        )
        self.assertIn("## Reasoning discipline", external_prompt)
        self.assertIn("## Tool selection and execution", external_prompt)

    def test_web_model_command_updates_session(self):
        from agent import web
        from agent.runtime_config import get_runtime_config

        environment = {
            "GEMINI_API_KEY": "google-secret",
            "GEMINI_MODELS": "gemini-3.1-flash-lite",
        }
        with patch.dict("os.environ", environment, clear=True):
            session = web._session_from_runtime(get_runtime_config())
            response = web.execute_command_web(
                "/set model gemini:gemini-3.1-flash-lite", session, []
            )

        self.assertIn("gemini-3.1-flash-lite", response)
        self.assertEqual(session["model_id"], "gemini:gemini-3.1-flash-lite")

    def test_api_model_context_defaults_do_not_persist_into_local_settings(self):
        from agent import web
        from agent.runtime_config import get_runtime_config

        external = {
            "model_id": "gemini:gemini-3.5-flash",
            "options": {},
        }
        environment = {
            "GEMINI_API_KEY": "google-secret",
            "GEMINI_MODELS": "gemini-3.5-flash",
        }
        with patch.dict("os.environ", environment, clear=True):
            request_session = web._session_for_selected_model(
                external, get_runtime_config(external)
            )
        self.assertEqual(request_session["options"]["num_ctx"], 1_048_576)
        self.assertEqual(external["options"], {})

        local = {"model_id": "local:default", "options": {}}
        local_request = web._session_for_selected_model(local, get_runtime_config(local))
        self.assertEqual(local_request["options"], {})


class ErrorFallbackContextTests(unittest.TestCase):
    """A fallback continues the same conversation, so it must refit it."""

    def _long_history(self) -> list[dict]:
        history = [{"role": "system", "content": "policy"}]
        for index in range(6):
            history.extend([
                {"role": "user", "content": f"request {index} " + ("x" * 1500)},
                {"role": "assistant", "content": f"answer {index} " + ("y" * 1500)},
            ])
        return history

    def test_fallback_compacts_the_durable_history_for_the_new_model(self):
        from agent import web
        from agent.runtime_config import get_runtime_config

        session = {"model_id": "local:default", "options": {"num_ctx": 4096}, "history": True}
        history = self._long_history()
        before = len(history)

        _request, _runtime, model, compaction = web._activate_error_fallback(
            session,
            get_runtime_config(session),
            {"local:default"},
            history=history,
        )

        self.assertIsNotNone(compaction)
        self.assertNotEqual(model["id"], "local:default")
        self.assertTrue(compaction["compacted"])
        self.assertLess(len(history), before)

    def test_fallback_without_history_keeps_todays_behaviour(self):
        from agent import web
        from agent.runtime_config import get_runtime_config

        session = {"model_id": "local:default", "options": {}, "history": True}

        result = web._activate_error_fallback(
            session, get_runtime_config(session), {"local:default"}
        )

        self.assertEqual(len(result), 4)
        self.assertIsNone(result[3])

    def test_reprime_shrinks_a_prompt_budgeted_for_the_failed_model(self):
        from agent.core import (
            _estimate_messages_tokens,
            reprime_outgoing_messages_for_model,
        )

        # A transcript recorded against a million-token provider window.
        messages = self._long_history()
        oversized = _estimate_messages_tokens(messages)
        local_session = {
            "model_id": "local:default",
            "options": {"num_ctx": 4096},
            "history": True,
        }

        prepared = reprime_outgoing_messages_for_model(messages, local_session)

        self.assertLess(_estimate_messages_tokens(prepared), oversized)
        self.assertLessEqual(_estimate_messages_tokens(prepared), 4096)


if __name__ == "__main__":
    unittest.main()
