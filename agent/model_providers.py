"""Provider-agnostic chat model registry and response adapters.

The rest of Selene speaks one Ollama-like internal message/chunk contract.
Provider-specific authentication, URLs, payloads, streaming formats, and
errors are contained here so adding a provider does not touch the agent loop.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import quote, urlsplit

import requests

from agent.cancellation import CancellationToken
from agent.environment import load_dotenv, refresh_dotenv
from agent.runtime_config import RuntimeConfig
from agent.system_prompts import (
    active_system_prompt_for_session,
    apply_active_system_prompt,
)


LOCAL_MODEL_ID = "local:default"
ERROR_FALLBACK_MODEL = "gemini-3.5-flash-lite"
ERROR_FALLBACK_MODEL_ID = f"gemini:{ERROR_FALLBACK_MODEL}"
NVIDIA_ERROR_FALLBACK_MODEL_ID = "nvidia:nvidia/nemotron-3-ultra-550b-a55b"
GEMMA_ERROR_FALLBACK_MODEL_ID = "gemini:gemma-4-31b-it"
ERROR_FALLBACK_MODEL_IDS = (
    ERROR_FALLBACK_MODEL_ID,
    NVIDIA_ERROR_FALLBACK_MODEL_ID,
    GEMMA_ERROR_FALLBACK_MODEL_ID,
    LOCAL_MODEL_ID,
)
DEFAULT_CAPABILITIES = frozenset({"chat", "streaming", "tools", "thinking", "json"})
REMOTE_CAPABILITIES = frozenset({"chat", "streaming", "tools", "json"})
# Known-good free chat models, used for context-window and capability
# defaults and as the shipped .env.example list. This is metadata, not an
# allowlist: GEMINI_MODELS decides what the selector offers.
GEMINI_FREE_CHAT_MODELS = (
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
)
GEMINI_CONTEXT_WINDOWS = {
    model: 262_144 if model.startswith("gemma-4-") else 1_048_576
    for model in GEMINI_FREE_CHAT_MODELS
}

DEFAULT_REMOTE_CONTEXT_WINDOW = 131_072
MAX_REMOTE_CONTEXT_WINDOW = 1_048_576
NVIDIA_CONTEXT_WINDOWS: dict[str, int] = {
    "nvidia/nemotron-3-ultra-550b-a55b": 262_144,
}


def _gemini_context_window(model: str) -> int:
    """Context maximum for a configured Gemini model.

    Models outside the curated table inherit the family default rather than
    the conservative remote fallback, which would silently truncate history.
    """
    if model in GEMINI_CONTEXT_WINDOWS:
        return GEMINI_CONTEXT_WINDOWS[model]
    if model.startswith("gemma-"):
        return 262_144
    if model.startswith("gemini-"):
        return 1_048_576
    return DEFAULT_REMOTE_CONTEXT_WINDOW


LOCAL_MODEL_DISPLAY_NAME = "Gemma 4 E4B"
SELENE_IDENTITY = (
    "You are Selene, a precise assistant with calm, subtle warmth. Always identify "
    "and present yourself as Selene regardless of the underlying model endpoint."
)


class ModelProviderError(RuntimeError):
    """A provider failure already reduced to a safe user-facing message."""

    def __init__(self, message: str, *, code: str = "provider_error") -> None:
        super().__init__(message)
        self.code = code


class MissingProviderConfiguration(ModelProviderError):
    pass


class UnsupportedModelError(ModelProviderError):
    pass


class ProviderTimeoutError(ModelProviderError):
    pass


class ProviderRateLimitError(ModelProviderError):
    pass


class InvalidProviderKeyError(ModelProviderError):
    pass


class MalformedProviderResponse(ModelProviderError):
    pass


class ProviderNetworkError(ModelProviderError):
    pass


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    display_name: str
    provider: str
    model: str
    endpoint: str
    api_key_env: str | None
    context_window: int | None = None
    required_env: tuple[str, ...] = ()
    capabilities: frozenset[str] = DEFAULT_CAPABILITIES
    available: bool = False
    unavailable_reason: str | None = None
    unavailable_code: str | None = None

    def public_dict(self) -> dict[str, Any]:
        """Return client-safe metadata; endpoints and environment values stay server-side."""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "provider": self.provider,
            "context_window": self.context_window,
            "capabilities": sorted(self.capabilities),
            "available": self.available,
        }


@dataclass
class NormalizedFunction:
    name: str
    arguments: dict[str, Any]


@dataclass
class NormalizedToolCall:
    function: NormalizedFunction
    id: str = ""
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedMessage:
    content: str = ""
    thinking: str = ""
    tool_calls: list[NormalizedToolCall] = field(default_factory=list)
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedChunk:
    message: NormalizedMessage
    prompt_eval_count: int = 0
    eval_count: int = 0
    done_reason: str = ""


def _clean(value: object) -> str:
    return str(value or "").strip()


_MOJIBAKE_RUN = re.compile(r"(?:Ã.|Â.|â..|ð...|ï..)", re.DOTALL)


def repair_text_encoding(value: str) -> str:
    """Repair common UTF-8 bytes that were decoded as Windows-1252/Latin-1.

    The replacement is deliberately limited to recognizable mojibake byte
    runs. Correct Unicode and ordinary provider text are returned unchanged.
    """
    text = str(value or "")
    if not any(marker in text for marker in ("Ã", "Â", "â", "ð", "ï")):
        return text

    def repair_run(match: re.Match[str]) -> str:
        fragment = match.group(0)
        for encoding in ("cp1252", "latin-1"):
            try:
                return fragment.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        return fragment

    # Two passes also recover text that was accidentally transcoded twice.
    for _ in range(2):
        repaired = _MOJIBAKE_RUN.sub(repair_run, text)
        if repaired == text:
            break
        text = repaired
    return text


def normalize_provider_text(value: object) -> str:
    """Return displayable text from provider string or content-part payloads.

    OpenAI-compatible endpoints are allowed to return either a plain string or
    an array of typed content parts.  Keeping this conversion in the adapter
    prevents Python representations such as ``[{'type': 'text', ...}]`` from
    leaking into Web/TUI markdown renderers.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return repair_text_encoding(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        fragments = [normalize_provider_text(part) for part in value]
        return "\n".join(fragment for fragment in fragments if fragment)
    if isinstance(value, Mapping):
        for key in ("text", "content", "output_text", "value"):
            if key in value:
                return normalize_provider_text(value.get(key))
        # Unknown structured content is still useful diagnostic output, but it
        # should be stable JSON rather than a language-specific repr.
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _configured_model(
    environ: Mapping[str, str],
    *,
    id: str,
    display_name: str | None,
    provider: str,
    model_env: str,
    default_model: str,
    endpoint_env: str,
    default_endpoint: str,
    api_key_env: str | None,
    required_env: Sequence[str] = (),
    capabilities: frozenset[str] = REMOTE_CAPABILITIES,
    model_value: str | None = None,
    context_window: int | None = DEFAULT_REMOTE_CONTEXT_WINDOW,
    context_window_env: str | None = None,
) -> ModelDefinition:
    model = _clean(model_value) if model_value is not None else _clean(environ.get(model_env))
    model = model or default_model
    endpoint = _clean(environ.get(endpoint_env)) or default_endpoint
    required = tuple(required_env) or ((api_key_env,) if api_key_env else ())
    missing = [name for name in required if not _clean(environ.get(name))]
    reason = None
    unavailable_code = None
    configured_context = _clean(environ.get(context_window_env or ""))
    if configured_context:
        try:
            context_window = int(configured_context)
        except ValueError:
            context_window = None
        if context_window is None or not 1024 <= context_window <= MAX_REMOTE_CONTEXT_WINDOW:
            reason = (
                f"Set {context_window_env} to an integer between 1024 and "
                f"{MAX_REMOTE_CONTEXT_WINDOW}."
            )
            unavailable_code = "missing_configuration"
    if missing:
        reason = f"Set {', '.join(missing)} on the server to enable this model."
        unavailable_code = "missing_api_key" if api_key_env in missing else "missing_configuration"
    elif reason is not None:
        pass
    elif not model or len(model) > 255 or any(character.isspace() or ord(character) < 32 for character in model):
        reason = f"Set {model_env} to a valid provider model identifier."
        unavailable_code = "unsupported_model"
    else:
        parsed_endpoint = urlsplit(endpoint)
        if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
            reason = f"Set {endpoint_env} to a valid HTTP(S) provider endpoint."
            unavailable_code = "missing_configuration"
    return ModelDefinition(
        id=id,
        display_name=display_name or model,
        provider=provider,
        model=model,
        endpoint=endpoint.rstrip("/"),
        api_key_env=api_key_env,
        context_window=context_window,
        required_env=required,
        capabilities=capabilities,
        available=reason is None,
        unavailable_reason=reason,
        unavailable_code=unavailable_code,
    )


def _configured_gemini_models(environ: Mapping[str, str]) -> list[ModelDefinition]:
    configured = _clean(environ.get("GEMINI_MODELS"))
    source_env = "GEMINI_MODELS"
    if not configured:
        configured = _clean(environ.get("GEMINI_MODEL"))
        source_env = "GEMINI_MODEL"
    models = list(dict.fromkeys(
        model.strip() for model in configured.split(",") if model.strip()
    ))
    if not models:
        return [_configured_model(
            environ,
            id="gemini:default",
            display_name=None,
            provider="Google Gemini",
            model_env="GEMINI_MODELS",
            default_model="",
            endpoint_env="GEMINI_BASE_URL",
            default_endpoint="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GEMINI_API_KEY",
            required_env=("GEMINI_API_KEY", "GEMINI_MODELS"),
        )]
    return [
        _configured_model(
            environ,
            id=f"gemini:{model}",
            display_name=None,
            provider="Google Gemini",
            model_env=source_env,
            model_value=model,
            default_model="",
            endpoint_env="GEMINI_BASE_URL",
            default_endpoint="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GEMINI_API_KEY",
            required_env=("GEMINI_API_KEY",),
            capabilities=(
                DEFAULT_CAPABILITIES
                if model.startswith(("gemini-2.5-", "gemini-3"))
                else REMOTE_CAPABILITIES
            ),
            context_window=_gemini_context_window(model),
            context_window_env="GEMINI_CONTEXT_WINDOW",
        )
        for model in models
    ]


def _configured_openrouter_models(environ: Mapping[str, str]) -> list[ModelDefinition]:
    configured = _clean(environ.get("OPENROUTER_MODELS"))
    source_env = "OPENROUTER_MODELS"
    if not configured:
        configured = _clean(environ.get("OPENROUTER_MODEL"))
        source_env = "OPENROUTER_MODEL"
    models = list(dict.fromkeys(
        model.strip() for model in configured.split(",") if model.strip()
    ))
    if not models:
        return [_configured_model(
            environ,
            id="openrouter:default",
            display_name=None,
            provider="OpenRouter",
            model_env="OPENROUTER_MODELS",
            default_model="",
            endpoint_env="OPENROUTER_BASE_URL",
            default_endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key_env="OPENROUTER_API_KEY",
            required_env=("OPENROUTER_API_KEY", "OPENROUTER_MODELS"),
            context_window_env="OPENROUTER_CONTEXT_WINDOW",
        )]
    return [
        _configured_model(
            environ,
            id=f"openrouter:{model}",
            display_name=None,
            provider="OpenRouter",
            model_env=source_env,
            model_value=model,
            default_model="",
            endpoint_env="OPENROUTER_BASE_URL",
            default_endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key_env="OPENROUTER_API_KEY",
            required_env=("OPENROUTER_API_KEY",),
            context_window_env="OPENROUTER_CONTEXT_WINDOW",
        )
        for model in models
    ]


def _configured_nvidia_models(environ: Mapping[str, str]) -> list[ModelDefinition]:
    configured = _clean(environ.get("NVIDIA_MODELS"))
    models = list(dict.fromkeys(
        model.strip() for model in configured.split(",") if model.strip()
    ))
    if not models:
        return [_configured_model(
            environ,
            id="nvidia:default",
            display_name=None,
            provider="NVIDIA",
            model_env="NVIDIA_MODELS",
            default_model="",
            endpoint_env="NVIDIA_BASE_URL",
            default_endpoint="https://integrate.api.nvidia.com/v1/chat/completions",
            api_key_env="NVIDIA_API_KEY",
            required_env=("NVIDIA_API_KEY", "NVIDIA_MODELS"),
            context_window_env="NVIDIA_CONTEXT_WINDOW",
        )]
    return [
        _configured_model(
            environ,
            id=f"nvidia:{model}",
            display_name=None,
            provider="NVIDIA",
            model_env="NVIDIA_MODELS",
            model_value=model,
            default_model="",
            endpoint_env="NVIDIA_BASE_URL",
            default_endpoint="https://integrate.api.nvidia.com/v1/chat/completions",
            api_key_env="NVIDIA_API_KEY",
            required_env=("NVIDIA_API_KEY",),
            context_window=NVIDIA_CONTEXT_WINDOWS.get(
                model, DEFAULT_REMOTE_CONTEXT_WINDOW
            ),
            context_window_env="NVIDIA_CONTEXT_WINDOW",
        )
        for model in models
    ]


def build_model_registry(
    runtime: RuntimeConfig,
    environ: Mapping[str, str] | None = None,
) -> dict[str, ModelDefinition]:
    """Build model metadata from server-side configuration only."""
    if environ is None:
        # Provider lists live in .env, and operators edit them while Selene is
        # running. Re-read the file when it changed so the model menu reflects
        # the current configuration without a restart.
        refresh_dotenv()
        environment = os.environ
    else:
        environment = environ
    local_model = _clean(getattr(runtime, "chat_model", None)) or "selene"
    local = ModelDefinition(
        id=LOCAL_MODEL_ID,
        display_name=LOCAL_MODEL_DISPLAY_NAME,
        provider="Ollama (local)",
        model=local_model,
        endpoint=_clean(environment.get("OLLAMA_HOST")) or "http://127.0.0.1:11434",
        api_key_env=None,
        context_window=None,
        capabilities=DEFAULT_CAPABILITIES,
        available=True,
    )
    entries = [
        local,
        *_configured_openrouter_models(environment),
        *_configured_nvidia_models(environment),
        *_configured_gemini_models(environment),
        _configured_model(
            environment,
            id="custom:default",
            display_name=None,
            provider="OpenAI-compatible / self-hosted",
            model_env="CUSTOM_LLM_MODEL",
            default_model="custom-model",
            endpoint_env="CUSTOM_LLM_BASE_URL",
            default_endpoint="",
            api_key_env="CUSTOM_LLM_API_KEY",
            required_env=("CUSTOM_LLM_BASE_URL", "CUSTOM_LLM_MODEL"),
            context_window_env="CUSTOM_LLM_CONTEXT_WINDOW",
        ),
    ]
    return {entry.id: entry for entry in entries}


def available_models(runtime: RuntimeConfig, environ: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    return [
        model.public_dict()
        for model in build_model_registry(runtime, environ).values()
        if model.available
    ]


def resolve_model(
    model_id: object,
    runtime: RuntimeConfig,
    environ: Mapping[str, str] | None = None,
) -> ModelDefinition:
    requested = _clean(model_id) or LOCAL_MODEL_ID
    registry = build_model_registry(runtime, environ)
    if requested == "gemini:default" and requested not in registry:
        requested = next(
            (model.id for model in registry.values() if model.id.startswith("gemini:") and model.available),
            requested,
        )
    if requested == "openrouter:default" and requested not in registry:
        openrouter_models = [
            model for model in registry.values()
            if model.id.startswith("openrouter:")
        ]
        requested = next(
            (model.id for model in openrouter_models if model.available),
            openrouter_models[0].id if openrouter_models else requested,
        )
    model = registry.get(requested)
    if model is None:
        raise UnsupportedModelError(
            "The selected model is not registered. Choose another model in the Model menu.",
            code="unsupported_model",
        )
    if not model.available:
        if model.unavailable_code == "unsupported_model":
            raise UnsupportedModelError(
                model.unavailable_reason or "The selected model is not supported.",
                code="unsupported_model",
            )
        raise MissingProviderConfiguration(
            model.unavailable_reason or "The selected model is not configured on the server.",
            code=model.unavailable_code or "missing_configuration",
        )
    return model


def resolve_error_fallback(
    current_model_id: object,
    runtime: RuntimeConfig,
    environ: Mapping[str, str] | None = None,
    attempted_model_ids: Iterable[object] = (),
) -> ModelDefinition | None:
    """Return the next available model in the bounded automatic fallback chain."""
    attempted = {_clean(model_id) for model_id in attempted_model_ids}
    attempted.add(_clean(current_model_id))
    for fallback_id in ERROR_FALLBACK_MODEL_IDS:
        if fallback_id in attempted:
            continue
        try:
            return resolve_model(fallback_id, runtime, environ)
        except (MissingProviderConfiguration, UnsupportedModelError):
            continue
    return None


def session_for_model(
    session: Mapping[str, Any] | None,
    runtime: RuntimeConfig,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return request-only settings with the selected provider's context limit."""
    source = deepcopy(dict(session or {}))
    try:
        selected = resolve_model(source.get("model_id"), runtime, environ)
    except (MissingProviderConfiguration, UnsupportedModelError):
        selected = resolve_model(LOCAL_MODEL_ID, runtime, environ)
    if selected.id == LOCAL_MODEL_ID or not selected.context_window:
        return source

    options = deepcopy(source.get("options") or {})
    explicit_context = options.get("num_ctx")
    if explicit_context is None:
        options["num_ctx"] = selected.context_window
    else:
        options["num_ctx"] = min(int(explicit_context), selected.context_window)
    source["options"] = options
    return source


def normalize_model_id(
    model_id: object,
    runtime: RuntimeConfig,
    environ: Mapping[str, str] | None = None,
) -> str:
    return resolve_model(model_id, runtime, environ).id


def _json_arguments(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MalformedProviderResponse(
            "The provider returned malformed tool arguments.",
            code="malformed_response",
        ) from exc
    if not isinstance(parsed, dict):
        raise MalformedProviderResponse(
            "The provider returned tool arguments in an unsupported format.",
            code="malformed_response",
        )
    return parsed


def _provider_reasoning_text(message: Mapping[str, Any]) -> str:
    """Normalize provider reasoning and planning metadata into Thinking."""
    fragments: list[str] = []
    for key in ("reasoning_content", "reasoning", "planning_content", "planning", "plan"):
        value = message.get(key)
        if isinstance(value, str) and value:
            fragments.append(value)
        elif isinstance(value, (list, dict)) and value:
            fragments.append(json.dumps(value, ensure_ascii=False))
    details = message.get("reasoning_details")
    if isinstance(details, list):
        for detail in details:
            if not isinstance(detail, dict):
                continue
            detail_type = _clean(detail.get("type")).casefold()
            if "encrypted" in detail_type:
                continue
            value = detail.get("text") or detail.get("summary") or detail.get("content")
            if value:
                fragments.append(normalize_provider_text(value))
    return "\n".join(fragments)


# Reasoning models on OpenAI-compatible endpoints (NVIDIA NIM Nemotron,
# DeepSeek-R1, Qwen3) put their reasoning in the *content* stream wrapped in a
# tag instead of in a separate reasoning_content field, so _provider_reasoning_text
# above never sees it and the raw tags render in the answer body.
#
# Whitespace-tolerant on both ends: a stray "</think >" must still close the
# block, because failing to close swallows the entire answer into Thinking —
# a worse failure than the leak this repairs.
_INLINE_REASONING_OPEN = re.compile(r"^\s*<\s*(?:think|thinking|reasoning)\s*>", re.IGNORECASE)
_INLINE_REASONING_CLOSE = re.compile(r"<\s*/\s*(?:think|thinking|reasoning)\s*>", re.IGNORECASE)
# Longest opener plus room for the whitespace the pattern tolerates. Once the
# leading text is this long without matching, it never will.
_INLINE_REASONING_OPEN_MAX = 24
# A closer held back across a chunk boundary can be at most this long.
_INLINE_REASONING_CLOSE_MAX = 24


class _InlineReasoningSplitter:
    """Route tag-wrapped reasoning in a content stream to the Thinking channel.

    Only a tag that *opens* the response counts. Once the splitter decides the
    response is ordinary content it stops scanning for good, so an answer that
    merely discusses ``<think>`` — in prose or inside a code block — is passed
    through untouched.

    Handles tags split across streaming deltas (``<thi`` then ``nk>``) by
    holding back text that could still be part of a tag.
    """

    __slots__ = ("_state", "_buffer", "_capture")

    def __init__(self, capture: bool = True) -> None:
        self._state = "pending"
        self._buffer = ""
        self._capture = capture

    def feed(self, text: str) -> tuple[str, str]:
        """Consume a delta. Returns (content, thinking)."""
        if not text:
            return "", ""
        if self._state == "passthrough":
            return text, ""
        self._buffer += text
        return self._drain(final=False)

    def flush(self) -> tuple[str, str]:
        """Release whatever is still held at end of stream."""
        if self._state == "passthrough" or not self._buffer:
            self._buffer = ""
            return "", ""
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> tuple[str, str]:
        content = ""
        thinking = ""
        if self._state == "pending":
            match = _INLINE_REASONING_OPEN.match(self._buffer)
            if match:
                self._buffer = self._buffer[match.end():]
                self._state = "inside"
            elif final or len(self._buffer.lstrip()) >= _INLINE_REASONING_OPEN_MAX:
                # Not a reasoning response. Release everything and stop looking.
                content = self._buffer
                self._buffer = ""
                self._state = "passthrough"
                return content, thinking
            else:
                # Too early to tell; hold until more arrives.
                return "", ""

        if self._state == "inside":
            match = _INLINE_REASONING_CLOSE.search(self._buffer)
            if match:
                thinking = self._buffer[:match.start()]
                content = self._buffer[match.end():]
                self._buffer = ""
                self._state = "passthrough"
            elif final:
                # Truncated generation: there is reasoning but no answer.
                thinking = self._buffer
                self._buffer = ""
            else:
                # Hold back a tail that could be the start of a closing tag.
                cut = len(self._buffer)
                marker = self._buffer.rfind("<")
                if marker != -1 and len(self._buffer) - marker < _INLINE_REASONING_CLOSE_MAX:
                    cut = marker
                thinking = self._buffer[:cut]
                self._buffer = self._buffer[cut:]

        if not self._capture:
            thinking = ""
        return content, thinking


def _raise_for_provider_payload_error(payload: Mapping[str, Any], provider: str) -> None:
    """Map OpenAI-compatible in-body and HTTP-200 stream errors safely."""
    error = payload.get("error")
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    if not error and isinstance(choice, dict):
        error = choice.get("error")
    if not error and isinstance(choice, dict) and choice.get("finish_reason") == "error":
        raise ModelProviderError(
            f"{provider} stopped while generating the response. Try again or select another model.",
            code="provider_error",
        )
    if not error:
        return

    if isinstance(error, Mapping):
        metadata = error.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        raw_code = error.get("code") or error.get("status") or ""
        error_type = _clean(
            metadata.get("error_type")
            or error.get("error_type")
            or error.get("type")
        ).casefold()
        # The upstream message is used only for classification. It is never
        # returned because providers can echo request data in error messages.
        fingerprint = " ".join((
            error_type,
            _clean(raw_code).casefold(),
            _clean(error.get("message")).casefold(),
        ))
    else:
        raw_code = ""
        error_type = ""
        fingerprint = _clean(error).casefold()

    try:
        status = int(raw_code)
    except (TypeError, ValueError):
        status = 0

    if status == 401 or "authentication" in fingerprint or "invalid_api_key" in fingerprint:
        raise InvalidProviderKeyError(
            f"{provider} rejected its API key. Update the server environment and restart Selene.",
            code="invalid_api_key",
        )
    if status == 429 or "rate_limit" in fingerprint or "rate limited" in fingerprint:
        raise ProviderRateLimitError(
            f"{provider} has reached its current rate or free-capacity limit. "
            "Wait a moment and retry, or select another model.",
            code="rate_limited",
        )
    if status in {408, 504} or error_type == "timeout" or "timed out" in fingerprint:
        raise ProviderTimeoutError(
            f"{provider} timed out. Try again or select another model.",
            code="timeout",
        )
    if (
        status in {502, 503}
        or error_type in {"provider_overloaded", "provider_unavailable", "server"}
    ):
        raise ModelProviderError(
            f"{provider} has no available capacity for this model right now. "
            "Try again shortly or select another model.",
            code="provider_unavailable",
        )
    if error_type in {
        "context_length_exceeded",
        "max_tokens_exceeded",
        "token_limit_exceeded",
        "string_too_long",
    }:
        raise ModelProviderError(
            f"{provider} could not fit this conversation in the selected model's limits. "
            "Start a new conversation or reduce the requested context.",
            code="context_length",
        )
    if status == 402 or error_type == "payment_required":
        raise ModelProviderError(
            f"{provider} says this account cannot run the request under its current quota. "
            "Check the provider account limits or select another model.",
            code="quota_exhausted",
        )
    if status == 403 or error_type == "permission_denied":
        raise ModelProviderError(
            f"{provider} blocked this request or the API key lacks permission for the model.",
            code="access_denied",
        )
    if status in {400, 404, 422} or error_type in {"invalid_request", "invalid_prompt"}:
        raise UnsupportedModelError(
            f"{provider} rejected the selected model or request options. "
            "Check the configured model identifier or select another model.",
            code="unsupported_model",
        )
    raise ModelProviderError(
        f"{provider} stopped while generating the response. Try again or select another model.",
        code="provider_error",
    )


def _request_error(provider: str, exc: Exception) -> ModelProviderError:
    if isinstance(exc, requests.Timeout):
        return ProviderTimeoutError(
            f"{provider} timed out. Try again or select another model.",
            code="timeout",
        )
    return ProviderNetworkError(
        f"Could not reach {provider}. Check the server network connection and try again.",
        code="network_failure",
    )


def _raise_for_status(response: requests.Response, provider: str) -> None:
    status = response.status_code
    if 200 <= status < 300:
        return
    provider_status = ""
    provider_error_fingerprint = ""
    if provider == "Google Gemini":
        try:
            error_payload = response.json()
        except (TypeError, ValueError, json.JSONDecodeError):
            error_payload = {}
        if isinstance(error_payload, dict):
            error = error_payload.get("error") or {}
            if isinstance(error, dict):
                provider_status = _clean(error.get("status")).upper()
                try:
                    provider_error_fingerprint = json.dumps(error, default=str).upper()
                except (TypeError, ValueError):
                    provider_error_fingerprint = provider_status
    response.close()
    if provider == "Google Gemini" and (
        "API_KEY_INVALID" in provider_error_fingerprint
        or provider_status == "UNAUTHENTICATED"
    ):
        raise InvalidProviderKeyError(
            "Google Gemini rejected its API key. Update GEMINI_API_KEY and restart Selene.",
            code="invalid_api_key",
        )
    if status in {401, 403}:
        if provider == "Google Gemini":
            raise ModelProviderError(
                "Google Gemini denied this project access to the selected model. "
                "Check the key restrictions, project, region, and model access.",
                code="access_denied",
            )
        raise InvalidProviderKeyError(
            f"{provider} rejected its API key. Update the server environment and restart Selene.",
            code="invalid_api_key",
        )
    if status == 400 and provider == "Google Gemini":
        raise ModelProviderError(
            "Google Gemini rejected Selene's request format or options. "
            "Try another configured Gemini model; the API key itself was accepted.",
            code="invalid_request",
        )
    if status == 429:
        raise ProviderRateLimitError(
            f"{provider} is rate limiting requests. Wait a moment or select another model.",
            code="rate_limited",
        )
    if status in {408, 504}:
        raise ProviderTimeoutError(
            f"{provider} timed out. Try again or select another model.",
            code="timeout",
        )
    if status in {502, 503}:
        raise ModelProviderError(
            f"{provider} is temporarily unavailable. Try again or select another model.",
            code="provider_unavailable",
        )
    if status in {400, 404, 422}:
        raise UnsupportedModelError(
            f"{provider} could not use the configured model. Check the model identifier.",
            code="unsupported_model",
        )
    raise ModelProviderError(
        f"{provider} request failed (HTTP {status}). Try again or select another model.",
        code="provider_error",
    )


def _post(
    definition: ModelDefinition,
    *,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: float,
    stream: bool,
) -> requests.Response:
    bounded_timeout = max(1.0, float(timeout))
    try:
        response = requests.post(
            url,
            headers=dict(headers),
            json=dict(payload),
            timeout=(min(10.0, bounded_timeout), bounded_timeout),
            stream=stream,
        )
    except requests.RequestException as exc:
        raise _request_error(definition.provider, exc) from None
    _raise_for_status(response, definition.provider)
    return response


def _response_json(response: requests.Response, provider: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MalformedProviderResponse(
            f"{provider} returned a malformed response.",
            code="malformed_response",
        ) from exc
    finally:
        response.close()
    if not isinstance(payload, dict):
        raise MalformedProviderResponse(
            f"{provider} returned an unsupported response.",
            code="malformed_response",
        )
    return payload


def _tool_schema_openai(tools: Sequence[dict] | None) -> list[dict]:
    return [dict(tool) for tool in (tools or [])]


def _selene_messages(messages: Sequence[dict]) -> list[dict[str, Any]]:
    """Ensure every external endpoint receives Selene's stable identity."""
    normalized = [dict(message) for message in messages]
    for message in normalized:
        if message.get("role") != "system":
            continue
        content = normalize_provider_text(message.get("content"))
        if "you are selene" not in content.casefold():
            message["content"] = f"{SELENE_IDENTITY}\n\n{content}".strip()
        return normalized
    return [{"role": "system", "content": SELENE_IDENTITY}, *normalized]


def _openai_messages(
    messages: Sequence[dict],
    *,
    preserve_reasoning: bool = False,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    pending_ids: list[str] = []

    def add_reasoning(item: dict[str, Any], source: Mapping[str, Any]) -> None:
        if not preserve_reasoning:
            return
        metadata = source.get("provider_metadata")
        if not isinstance(metadata, Mapping):
            return
        details = metadata.get("reasoning_details")
        if isinstance(details, list) and all(isinstance(detail, dict) for detail in details):
            item["reasoning_details"] = deepcopy(details)

    for message_index, message in enumerate(messages):
        role = str(message.get("role") or "user")
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            pending_ids = []
            for call_index, call in enumerate(message.get("tool_calls") or []):
                function = call.get("function", call)
                call_id = str(call.get("id") or f"call_{message_index}_{call_index}")
                pending_ids.append(call_id)
                arguments = function.get("arguments", {})
                calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": str(function.get("name") or ""),
                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                    },
                })
            assistant = {
                "role": "assistant",
                "content": normalize_provider_text(message.get("content")),
                "tool_calls": calls,
            }
            add_reasoning(assistant, message)
            converted.append(assistant)
        elif role == "tool":
            call_id = pending_ids.pop(0) if pending_ids else f"call_{message_index}_0"
            converted.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": normalize_provider_text(message.get("content")),
            })
        elif role in {"system", "user", "assistant"}:
            item = {"role": role, "content": normalize_provider_text(message.get("content"))}
            if role == "assistant":
                add_reasoning(item, message)
            converted.append(item)
    return converted


def _remote_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    source = options or {}
    mapping = {
        "temperature": "temperature",
        "top_p": "top_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "seed": "seed",
        "num_predict": "max_tokens",
    }
    return {target: source[name] for name, target in mapping.items() if name in source}


def _openai_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def _openai_headers(definition: ModelDefinition, environ: Mapping[str, str]) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = _clean(environ.get(definition.api_key_env or ""))
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if definition.id.startswith("openrouter:"):
        site = _clean(environ.get("OPENROUTER_SITE_URL"))
        title = _clean(environ.get("OPENROUTER_APP_NAME")) or "Selene"
        if site:
            headers["HTTP-Referer"] = site
        headers["X-OpenRouter-Title"] = title
    return headers


def _openai_payload(definition: ModelDefinition, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": definition.model,
        "messages": _openai_messages(
            kwargs.get("messages") or [],
            preserve_reasoning=definition.id.startswith("openrouter:"),
        ),
        "stream": bool(kwargs.get("stream")),
        **_remote_options(kwargs.get("options")),
    }
    tools = _tool_schema_openai(kwargs.get("tools"))
    if tools:
        payload["tools"] = tools
    if kwargs.get("format") == "json":
        payload["response_format"] = {"type": "json_object"}
    if payload["stream"]:
        payload["stream_options"] = {"include_usage": True}
    if _is_nvidia_nemotron(definition):
        payload["chat_template_kwargs"] = {
            "enable_thinking": bool(kwargs.get("think", True)),
            "force_nonempty_content": True,
        }
    return payload


def _is_nvidia_nemotron(definition: ModelDefinition) -> bool:
    model = definition.model.casefold().removeprefix("nvidia/")
    return (
        definition.id.startswith("nvidia:")
        and model.startswith("nemotron-3-")
    )


def _requests_fenced_output(messages: Sequence[dict]) -> bool:
    for message in reversed(messages):
        if str(message.get("role") or "") != "user":
            continue
        request = normalize_provider_text(message.get("content")).casefold()
        return bool(re.search(
            r"\b(?:code\s*(?:block|fence)|fenced\s+(?:code|block|output))\b",
            request,
        ))
    return False


_TRUNCATED_MARKDOWN_FENCE_PREFIX = re.compile(
    r"^(markdown|arkdown|rkdown|kdown|down)(?=\r?\n(?:#{1,6}\s|[-*+]\s|>\s))"
)


def _repair_nemotron_markdown_prefix(
    content: str,
    definition: ModelDefinition,
    *,
    requested_fence: bool,
) -> str:
    """Repair the observed NIM loss of an opening Markdown fence fragment."""
    if (
        not content
        or not requested_fence
        or not _is_nvidia_nemotron(definition)
    ):
        return content
    match = _TRUNCATED_MARKDOWN_FENCE_PREFIX.match(content)
    if not match:
        return content
    return "```markdown" + content[match.end():]


def _openai_chunk(
    payload: Mapping[str, Any],
    definition: ModelDefinition,
    *,
    repair_markdown_fence: bool = False,
    capture_thinking: bool = True,
) -> NormalizedChunk:
    _raise_for_provider_payload_error(payload, definition.provider)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise MalformedProviderResponse(
            "The provider returned a response without a valid model choice.",
            code="malformed_response",
        )
    choice = choices[0]
    message = choice.get("message") or choice.get("delta")
    if not isinstance(message, dict):
        raise MalformedProviderResponse(
            "The provider returned a response without a valid model message.",
            code="malformed_response",
        )
    usage = payload.get("usage") or {}
    tool_calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        call_name = str(function.get("name") or "")
        if call_name in ("google_search", "googleSearch"):
            call_name = "web_search"
        tool_calls.append(NormalizedToolCall(NormalizedFunction(
            name=call_name,
            arguments=_json_arguments(function.get("arguments")),
        )))
    # Split before the fence repair: an unstripped "<think>" prefix would stop
    # _repair_nemotron_markdown_prefix from ever matching.
    splitter = _InlineReasoningSplitter(capture=capture_thinking)
    body, inline_thinking = splitter.feed(normalize_provider_text(message.get("content")))
    trailing_body, trailing_thinking = splitter.flush()
    body += trailing_body
    inline_thinking += trailing_thinking
    thinking = "\n".join(filter(None, (
        _provider_reasoning_text(message),
        inline_thinking,
    )))
    result = NormalizedChunk(
        message=NormalizedMessage(
            content=_repair_nemotron_markdown_prefix(
                body,
                definition,
                requested_fence=repair_markdown_fence,
            ),
            thinking=thinking,
            tool_calls=tool_calls,
            provider_metadata=(
                {"reasoning_details": deepcopy(message["reasoning_details"])}
                if isinstance(message.get("reasoning_details"), list)
                else {}
            ),
        ),
        prompt_eval_count=int(usage.get("prompt_tokens") or 0),
        eval_count=int(usage.get("completion_tokens") or 0),
        done_reason=str(choice.get("finish_reason") or ""),
    )
    if not (
        result.message.content
        or result.message.thinking
        or result.message.tool_calls
    ):
        raise MalformedProviderResponse(
            f"{definition.provider} completed the request without generating content. "
            "The endpoint may be warming up; try again shortly or select another model.",
            code="empty_response",
        )
    return result


def _iter_sse_json(response: requests.Response, provider: str) -> Iterator[dict[str, Any]]:
    pending_data: list[str] = []
    yielded_payload = False
    malformed_event = False

    def parse_event(value: str) -> dict[str, Any] | None:
        nonlocal malformed_event
        try:
            payload = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            malformed_event = True
            return None
        if isinstance(payload, dict):
            return payload
        malformed_event = True
        return None

    try:
        # Provider SSE is UTF-8 JSON. Do not use requests' inferred response
        # encoding here: text/event-stream endpoints often omit a charset and
        # UTF-8 symbols then become mojibake such as ``₹`` -> ``â‚¹``.
        for raw in response.iter_lines(decode_unicode=False):
            if isinstance(raw, bytes):
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    malformed_event = True
                    continue
            else:
                line = str(raw or "")
            line = line.rstrip("\r")
            if not line:
                if pending_data:
                    payload = parse_event("\n".join(pending_data))
                    pending_data.clear()
                    if payload is not None:
                        yielded_payload = True
                        yield payload
                continue
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if separator and field in {"event", "id", "retry"}:
                continue
            if separator and field == "data":
                value = value.lstrip()
                if value == "[DONE]":
                    pending_data.clear()
                    break
                # Providers normally send one JSON object per data line. If a
                # proxy splits JSON across data lines, retain it until it forms
                # a complete object. A bad event does not poison later events.
                candidate = "\n".join([*pending_data, value])
                payload = parse_event(candidate)
                if payload is None and pending_data:
                    payload = parse_event(value)
                    if payload is not None:
                        pending_data.clear()
                if payload is None:
                    pending_data.append(value)
                    continue
                pending_data.clear()
                yielded_payload = True
                yield payload
                continue
            stripped = line.strip()
            if stripped == "[DONE]":
                break
            # Some OpenAI-compatible servers use JSONL instead of strict SSE.
            if stripped.startswith("{"):
                payload = parse_event(stripped)
                if payload is not None:
                    yielded_payload = True
                    pending_data.clear()
                    yield payload
        if pending_data:
            payload = parse_event("\n".join(pending_data))
            if payload is not None:
                yielded_payload = True
                yield payload
        if malformed_event and not yielded_payload:
            raise MalformedProviderResponse(
                f"{provider} returned an unreadable response stream. Try the request again.",
                code="malformed_response",
            )
    except requests.RequestException as exc:
        raise _request_error(provider, exc) from None
    finally:
        response.close()


def _openai_stream(
    response: requests.Response,
    definition: ModelDefinition,
    *,
    repair_markdown_fence: bool = False,
    capture_thinking: bool = True,
) -> Iterator[NormalizedChunk]:
    pending: dict[int, dict[str, str]] = {}
    reasoning_details: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    done_reason = ""
    saw_choice = False
    saw_output = False
    content_started = False
    initial_content = ""
    splitter = _InlineReasoningSplitter(capture=capture_thinking)
    for payload in _iter_sse_json(response, definition.provider):
        _raise_for_provider_payload_error(payload, definition.provider)
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
        completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
        choices = payload.get("choices")
        if choices is None:
            continue
        if not isinstance(choices, list):
            raise MalformedProviderResponse(
                f"{definition.provider} returned a malformed response stream.",
                code="malformed_response",
            )
        if not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            raise MalformedProviderResponse(
                f"{definition.provider} returned a malformed response stream.",
                code="malformed_response",
            )
        saw_choice = True
        done_reason = str(choice.get("finish_reason") or done_reason)
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise MalformedProviderResponse(
                f"{definition.provider} returned a malformed response stream.",
                code="malformed_response",
            )
        details = delta.get("reasoning_details")
        if isinstance(details, list):
            reasoning_details.extend(
                deepcopy(detail) for detail in details if isinstance(detail, dict)
            )
        for raw_call in delta.get("tool_calls") or []:
            saw_output = True
            index = int(raw_call.get("index") or 0)
            current = pending.setdefault(index, {"name": "", "arguments": ""})
            function = raw_call.get("function") or {}
            current["name"] += str(function.get("name") or "")
            current["arguments"] += str(function.get("arguments") or "")
        # Pull any tag-wrapped reasoning out of the content stream before the
        # fence-repair buffer sees it, otherwise initial_content starts with
        # "<think>" and _repair_nemotron_markdown_prefix never matches.
        content, inline_thinking = splitter.feed(
            normalize_provider_text(delta.get("content"))
        )
        thinking = "\n".join(filter(None, (
            _provider_reasoning_text(delta),
            inline_thinking,
        )))
        if (
            content
            and repair_markdown_fence
            and _is_nvidia_nemotron(definition)
            and not content_started
        ):
            initial_content += content
            if "\n" in initial_content or len(initial_content) >= 64:
                content = _repair_nemotron_markdown_prefix(
                    initial_content,
                    definition,
                    requested_fence=repair_markdown_fence,
                )
                initial_content = ""
                content_started = True
            else:
                content = ""
        elif content:
            content_started = True
        if content or thinking:
            saw_output = True
            yield NormalizedChunk(message=NormalizedMessage(
                content=content,
                thinking=thinking,
            ))
    # Release whatever the splitter was holding back: an undecided prefix, or
    # reasoning from a generation that was cut off before it closed its tag.
    trailing_content, trailing_thinking = splitter.flush()
    if trailing_content:
        if repair_markdown_fence and _is_nvidia_nemotron(definition) and not content_started:
            initial_content += trailing_content
        else:
            content_started = True
            saw_output = True
            yield NormalizedChunk(message=NormalizedMessage(content=trailing_content))
    if trailing_thinking:
        saw_output = True
        yield NormalizedChunk(message=NormalizedMessage(thinking=trailing_thinking))
    if initial_content:
        saw_output = True
        yield NormalizedChunk(message=NormalizedMessage(
            content=_repair_nemotron_markdown_prefix(
                initial_content,
                definition,
                requested_fence=repair_markdown_fence,
            ),
        ))
    if not saw_choice:
        raise MalformedProviderResponse(
            f"{definition.provider} ended the request without returning a model response. "
            "The free endpoint may be warming up; try again shortly or select another model.",
            code="empty_response",
        )
    tool_calls = []
    for _, item in sorted(pending.items()):
        call_name = item["name"]
        if call_name in ("google_search", "googleSearch"):
            call_name = "web_search"
        tool_calls.append(NormalizedToolCall(NormalizedFunction(call_name, _json_arguments(item["arguments"]))))
    if not saw_output and not tool_calls:
        raise MalformedProviderResponse(
            f"{definition.provider} completed the request without generating content. "
            "The free endpoint may be warming up; try again shortly or select another model.",
            code="empty_response",
        )
    yield NormalizedChunk(
        message=NormalizedMessage(
            tool_calls=tool_calls,
            provider_metadata=(
                {"reasoning_details": reasoning_details}
                if reasoning_details
                else {}
            ),
        ),
        prompt_eval_count=prompt_tokens,
        eval_count=completion_tokens,
        done_reason=done_reason,
    )


def _gemini_contents(messages: Sequence[dict]) -> tuple[str, list[dict[str, Any]]]:
    systems: list[str] = []
    contents: list[dict[str, Any]] = []
    pending_calls: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        if role == "system":
            systems.append(normalize_provider_text(message.get("content")))
            continue
        if role == "assistant" and message.get("tool_calls"):
            parts: list[dict[str, Any]] = []
            if message.get("content"):
                parts.append({"text": normalize_provider_text(message.get("content"))})
            pending_calls = []
            for call in message.get("tool_calls") or []:
                function = call.get("function", call)
                name = str(function.get("name") or "")
                call_id = str(call.get("id") or "")
                pending_calls.append({"name": name, "id": call_id})
                function_call = {
                    "name": name,
                    "args": _json_arguments(function.get("arguments")),
                }
                if call_id:
                    function_call["id"] = call_id
                part: dict[str, Any] = {"functionCall": function_call}
                metadata = call.get("provider_metadata") or {}
                if isinstance(metadata, dict) and metadata.get("thought_signature"):
                    part["thoughtSignature"] = str(metadata["thought_signature"])
                parts.append(part)
            contents.append({"role": "model", "parts": parts})
        elif role == "tool":
            pending = pending_calls.pop(0) if pending_calls else {}
            name = str(
                pending.get("name")
                or message.get("name")
                or message.get("tool_name")
                or "tool"
            )
            raw = str(message.get("content") or "")
            try:
                response_data = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                response_data = {"result": raw}
            if not isinstance(response_data, dict):
                response_data = {"result": response_data}
            function_response = {
                "name": name,
                "response": response_data,
            }
            if pending.get("id"):
                function_response["id"] = pending["id"]
            contents.append({"role": "user", "parts": [{
                "functionResponse": function_response,
            }]})
        elif role in {"user", "assistant"}:
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": normalize_provider_text(message.get("content"))}],
            })
    merged: list[dict[str, Any]] = []
    for content in contents:
        if merged and merged[-1]["role"] == content["role"]:
            merged[-1]["parts"].extend(content["parts"])
        else:
            merged.append(content)
    return "\n\n".join(systems), merged


def _gemini_payload(definition: ModelDefinition, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    system, contents = _gemini_contents(kwargs.get("messages") or [])
    options = kwargs.get("options") or {}
    generation: dict[str, Any] = {}
    option_map = {
        "temperature": "temperature",
        "top_p": "topP",
        "top_k": "topK",
        "num_predict": "maxOutputTokens",
    }
    for source, target in option_map.items():
        if source in options:
            generation[target] = options[source]
    if definition.model.startswith(("gemini-2.5-", "gemini-3")):
        # Gemini returns thought summaries as parts marked with `thought: true`
        # only when explicitly requested. Keeping this tied to Selene's shared
        # `think` flag lets the Web UI, TUI, and classic terminal consume the
        # same normalized thinking channel.
        generation["thinkingConfig"] = {
            "includeThoughts": bool(kwargs.get("think", True)),
        }
    if kwargs.get("format") == "json":
        generation["responseMimeType"] = "application/json"
    payload: dict[str, Any] = {"contents": contents}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if generation:
        payload["generationConfig"] = generation
    declarations = []
    for tool in kwargs.get("tools") or []:
        function = tool.get("function") or {}
        declarations.append({
            "name": function.get("name"),
            "description": function.get("description", ""),
            "parametersJsonSchema": (
                function.get("parameters") or {"type": "object", "properties": {}}
            ),
        })
    if declarations:
        payload["tools"] = [{"functionDeclarations": declarations}]
    return payload


def _gemini_url(definition: ModelDefinition, *, streaming: bool) -> str:
    base = definition.endpoint.rstrip("/")
    if base.endswith("/models"):
        base = base[:-7]
    model = definition.model.removeprefix("models/")
    operation = "streamGenerateContent?alt=sse" if streaming else "generateContent"
    return f"{base}/models/{quote(model, safe='')}:{operation}"


def _gemini_headers(
    definition: ModelDefinition,
    environ: Mapping[str, str],
    *,
    streaming: bool,
) -> dict[str, str]:
    key = _clean(environ.get(definition.api_key_env or ""))
    return {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if streaming else "application/json",
        "x-goog-api-key": key,
    }


def _gemini_chunk(payload: Mapping[str, Any]) -> NormalizedChunk:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise MalformedProviderResponse(
            "Google Gemini returned a response without a valid model candidate.",
            code="malformed_response",
        )
    candidate = candidates[0]
    content_block = candidate.get("content") or {}
    if not isinstance(content_block, dict):
        raise MalformedProviderResponse(
            "Google Gemini returned a malformed model message.",
            code="malformed_response",
        )
    parts = content_block.get("parts") or []
    if not isinstance(parts, list):
        raise MalformedProviderResponse(
            "Google Gemini returned a malformed model message.",
            code="malformed_response",
        )
    content = ""
    thinking = ""
    calls: list[NormalizedToolCall] = []
    for part in parts:
        if not isinstance(part, dict):
            raise MalformedProviderResponse(
                "Google Gemini returned a malformed model message.",
                code="malformed_response",
            )
        if part.get("thought"):
            thought_text = str(part.get("text") or "")
            thinking += thought_text
        elif part.get("plan"):
            thinking += str(part.get("plan") or "")
        elif "text" in part:
            content += normalize_provider_text(part.get("text"))
        elif part.get("functionCall"):
            call = part["functionCall"]
            if not isinstance(call, dict):
                raise MalformedProviderResponse(
                    "Google Gemini returned a malformed function call.",
                    code="malformed_response",
                )
            metadata = {}
            if part.get("thoughtSignature"):
                metadata["thought_signature"] = str(part["thoughtSignature"])
            call_name = str(call.get("name") or "")
            if call_name in ("google_search", "googleSearch"):
                call_name = "web_search"
            calls.append(NormalizedToolCall(
                NormalizedFunction(
                    call_name,
                    _json_arguments(call.get("args")),
                ),
                id=str(call.get("id") or ""),
                provider_metadata=metadata,
            ))
    usage = payload.get("usageMetadata") or {}
    return NormalizedChunk(
        message=NormalizedMessage(
            content=content,
            thinking=thinking,
            tool_calls=calls,
        ),
        prompt_eval_count=int(usage.get("promptTokenCount") or 0),
        eval_count=int(usage.get("candidatesTokenCount") or 0),
        done_reason=str(candidate.get("finishReason") or ""),
    )


def _gemini_stream(response: requests.Response, definition: ModelDefinition) -> Iterator[NormalizedChunk]:
    saw_candidate = False
    for payload in _iter_sse_json(response, definition.provider):
        if not payload.get("candidates"):
            continue
        saw_candidate = True
        yield _gemini_chunk(payload)
    if not saw_candidate:
        raise MalformedProviderResponse(
            "Google Gemini returned an empty response stream.",
            code="malformed_response",
        )


def _remote_chat(
    definition: ModelDefinition,
    *,
    environ: Mapping[str, str],
    operation_timeout: float,
    cancellation_token: CancellationToken | None,
    **kwargs: Any,
) -> NormalizedChunk | Iterator[NormalizedChunk]:
    token = cancellation_token or CancellationToken()
    token.raise_if_cancelled()
    streaming = bool(kwargs.get("stream"))
    kwargs["messages"] = _selene_messages(kwargs.get("messages") or [])
    repair_markdown_fence = _requests_fenced_output(kwargs["messages"])
    # With Show thinking off, inline reasoning tags are still stripped — they
    # must never reach the answer — but the reasoning itself is dropped.
    capture_thinking = bool(kwargs.get("think", True))

    if definition.id.startswith("gemini:"):
        response = _post(
            definition,
            url=_gemini_url(definition, streaming=streaming),
            headers=_gemini_headers(definition, environ, streaming=streaming),
            payload=_gemini_payload(definition, kwargs),
            timeout=operation_timeout,
            stream=streaming,
        )
        release_response = token.add_cancel_callback(response.close)
        try:
            token.raise_if_cancelled()
        except Exception:
            release_response()
            response.close()
            raise
        if not streaming:
            try:
                return _gemini_chunk(_response_json(response, definition.provider))
            finally:
                release_response()
        return _cancellable(
            _gemini_stream(response, definition), token, cleanup=release_response
        )

    response = _post(
        definition,
        url=_openai_url(definition.endpoint),
        headers=_openai_headers(definition, environ),
        payload=_openai_payload(definition, kwargs),
        timeout=operation_timeout,
        stream=streaming,
    )
    release_response = token.add_cancel_callback(response.close)
    try:
        token.raise_if_cancelled()
    except Exception:
        release_response()
        response.close()
        raise
    if not streaming:
        try:
            return _openai_chunk(
                _response_json(response, definition.provider),
                definition,
                repair_markdown_fence=repair_markdown_fence,
                capture_thinking=capture_thinking,
            )
        finally:
            release_response()
    return _cancellable(
        _openai_stream(
            response,
            definition,
            repair_markdown_fence=repair_markdown_fence,
            capture_thinking=capture_thinking,
        ),
        token,
        cleanup=release_response,
    )


def _cancellable(
    chunks: Iterable[NormalizedChunk],
    token: CancellationToken,
    *,
    cleanup: Callable[[], None] | None = None,
) -> Iterator[NormalizedChunk]:
    try:
        for chunk in chunks:
            token.raise_if_cancelled()
            yield chunk
    finally:
        if cleanup is not None:
            cleanup()


def chat_with_model(
    session: Mapping[str, Any] | None,
    runtime: RuntimeConfig,
    *,
    ollama_service_factory: Callable[[RuntimeConfig], Any],
    environ: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> Any:
    """Route a canonical Selene chat call through the selected adapter."""
    environment = os.environ if environ is None else environ
    model = resolve_model((session or {}).get("model_id"), runtime, environment)
    routed_kwargs = dict(kwargs)
    operation_kind = getattr(kwargs.get("kind"), "value", kwargs.get("kind"))
    if operation_kind in (None, "chat"):
        routed_kwargs["messages"] = apply_active_system_prompt(
            kwargs.get("messages") or [],
            active_system_prompt_for_session(session),
        )
    if model.id == LOCAL_MODEL_ID:
        routed_kwargs["model"] = model.model
        return ollama_service_factory(runtime).chat(**routed_kwargs)

    operation_timeout = float(
        routed_kwargs.pop("operation_timeout", runtime.chat_timeout_seconds)
    )
    routed_kwargs.pop("kind", None)
    routed_kwargs.pop("owner", None)
    cancellation_token = routed_kwargs.pop("cancellation_token", None)
    routed_kwargs.pop("model", None)
    return _remote_chat(
        model,
        environ=environment,
        operation_timeout=operation_timeout,
        cancellation_token=cancellation_token,
        **routed_kwargs,
    )


# Load only the repository-local file, and never replace exported environment.
load_dotenv()
