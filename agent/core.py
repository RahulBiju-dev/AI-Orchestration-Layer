"""
agent/core.py — Core chat loop with tool-call interception.

Manages conversation history, sends requests to the custom Ollama model,
intercepts any tool calls, executes them, feeds results back, and
streams the final synthesized answer with visible thinking status.
"""

import glob
import json
import os
import re
import shlex
from agent.terminal import (
    _console,
    _Spinner,
    assistant_stream_panel,
    display_is_tui,
    flush_terminal_input,
    print_assistant_message,
    print_command_help,
    print_content_stream,
    print_error,
    print_generation_stats,
    print_info,
    print_lab_status,
    print_ok,
    print_thinking_delta,
    print_thinking_footer,
    print_thinking_header,
    print_tool_event,
    print_warn,
    print_welcome_header,
    read_user_input,
    thinking_stream_style,
)

import signal
import sys
import threading
import time
import itertools
from datetime import datetime, timezone
try:
    import readline
except ImportError:
    pass

from rich.live import Live

from agent.ollama_runtime import OllamaRuntimeError, OllamaService, OperationKind
from agent.persistence import atomic_write_json, atomic_write_text
from agent.platform_runtime import get_runtime_paths, resource_path
from agent.runtime_config import RuntimeConfigurationError, RuntimeConfig, get_runtime_config
from agent.tool_runner import (
    ToolCallResult,
    ToolCallSpec,
    execute_tool_call,
    execute_tool_calls,
    normalize_tool_arguments,
    normalize_tool_calls,
)
from tools.registry import TOOL_DISPATCH, TOOL_SCHEMAS, get_tool_metadata

# ── Configuration ─────────────────────────────────────────────────────

_RUNTIME_PATHS = get_runtime_paths()
_BASE_RUNTIME_CONFIG = get_runtime_config()
_OLLAMA_SERVICE = OllamaService(_BASE_RUNTIME_CONFIG)
MODEL_NAME = _BASE_RUNTIME_CONFIG.chat_model
_PROJECT_ROOT = str(resource_path("Modelfile").parent)
_DATA_DIR = str(_RUNTIME_PATHS.data_dir)
_SESSIONS_DIR = str(_RUNTIME_PATHS.data_dir / "sessions")
_LEGACY_SESSIONS_DIR = os.path.join(_PROJECT_ROOT, "sessions")
_SYSTEM_PROMPT_CACHE_FILE = os.path.join(_DATA_DIR, "system_prompt_cache.txt")
_DEFAULT_SYSTEM_PROMPT: str | None = None
_DEFAULT_SYSTEM_PROMPT_MTIME: float | None = None
_COMPACT_TOOL_SCHEMAS_CACHE: dict[str, list[dict]] = {}
_SYSTEM_PROMPT_LOCK = threading.RLock()
_COMPACT_TOOL_SCHEMAS_LOCK = threading.Lock()

DEFAULT_NUM_CTX = _BASE_RUNTIME_CONFIG.num_ctx
DEFAULT_NUM_PREDICT = _BASE_RUNTIME_CONFIG.num_predict
MIN_RESPONSE_RESERVE_TOKENS = 256
MIN_EMERGENCY_RESPONSE_TOKENS = 96
CONTEXT_SAFETY_MARGIN_RATIO = 0.08
CONTEXT_TOOL_LOOP_RESERVE = 512
MAX_TOOL_CALL_ROUNDS = 8
MAX_OUTPUT_CONTINUATION_ROUNDS = 8
SYSTEM_PROMPT_ANCHOR_THRESHOLD = 0.75
TOOL_CONTINUATION_PROMPT = (
    "Continue the current user request using the tool result messages already present in this conversation. "
    "Do not repeat an identical tool call to rediscover the same information. "
    "If the available tool results are sufficient, answer the original request directly; "
    "otherwise call only the next distinct tool that is still required.\n\n"
    "Original user request:\n{user_input}"
)
OUTPUT_CONTINUATION_PROMPT = (
    "The previous response reached the model's per-call output limit. Continue exactly where it ended. "
    "Do not restart, repeat earlier text, add a continuation heading, or discuss the limit. "
    "Finish the original request, then stop normally.\n\n"
    "Original user request:\n{user_input}"
)


class ContextWindowError(RuntimeError):
    """Raised when a prompt cannot safely fit inside the configured context."""


def _chunk_done_reason(chunk) -> str:
    """Return Ollama's terminal reason across object and mapping responses."""
    if isinstance(chunk, dict):
        reason = chunk.get("done_reason")
    else:
        reason = getattr(chunk, "done_reason", None)
    return str(reason or "").strip().lower()


def _output_limit_reached(done_reason: str, eval_tokens: int, num_predict: int) -> bool:
    """Detect explicit and legacy Ollama output-budget stops."""
    reason = str(done_reason or "").strip().lower()
    if reason:
        return reason in {"length", "max_tokens", "token_limit"}
    try:
        return int(eval_tokens) >= int(num_predict) > 0
    except (TypeError, ValueError):
        return False

# Parameters that accept float values via /set parameter
_FLOAT_PARAMS = {"temperature", "top_p", "repeat_penalty", "presence_penalty", "frequency_penalty", "min_p", "tfs_z"}
# Parameters that accept integer values via /set parameter
_INT_PARAMS = {"num_ctx", "num_predict", "num_batch", "top_k", "repeat_last_n", "seed", "num_gpu", "num_thread", "num_keep"}
_ALL_PARAMS = _FLOAT_PARAMS | _INT_PARAMS
_CENTRAL_OPTION_NAMES = set(_BASE_RUNTIME_CONFIG.ollama_options())
_EXTRA_FLOAT_RANGES = {
    "presence_penalty": (-2.0, 2.0),
    "frequency_penalty": (-2.0, 2.0),
    "min_p": (0.0, 1.0),
    "tfs_z": (0.0, 2.0),
}
_EXTRA_INT_RANGES = {
    "repeat_last_n": (-1, 131072),
    "seed": (-1, 2_147_483_647),
    "num_gpu": (-1, 1024),
    "num_thread": (1, 512),
    "num_keep": (0, 131072),
}
# terminal helpers (spinner, renderer, ANSI constants) are imported
# from agent.terminal to keep terminal logic modular.

# ── History management ────────────────────────────────────────────────
# Keeps prompt size bounded so tok/s stays consistent across long sessions.


def resolve_session_runtime(session: dict | None = None) -> RuntimeConfig:
    """Resolve centralized settings with this session's explicit overrides."""
    return get_runtime_config(session or {})


def validate_session_options(options: dict | None) -> tuple[dict, tuple[str, ...]]:
    """Validate persisted or interactive Ollama options without silent clamping."""
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise RuntimeConfigurationError("Session options must be a JSON object")

    unknown = set(options) - _ALL_PARAMS
    if unknown:
        raise RuntimeConfigurationError(f"Unknown model option(s): {', '.join(sorted(unknown))}")

    normalized: dict = {}
    for name, value in options.items():
        if name in _FLOAT_PARAMS:
            if isinstance(value, bool):
                raise RuntimeConfigurationError(f"{name} must be a number, not a boolean")
            try:
                normalized[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise RuntimeConfigurationError(f"{name} must be a number") from exc
        else:
            if isinstance(value, bool):
                raise RuntimeConfigurationError(f"{name} must be an integer, not a boolean")
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise RuntimeConfigurationError(f"{name} must be an integer") from exc
            if isinstance(value, float) and not value.is_integer():
                raise RuntimeConfigurationError(f"{name} must be an integer")
            normalized[name] = parsed

    # Central settings are validated together, including num_ctx/num_predict
    # headroom and detected-profile warnings.
    runtime = get_runtime_config(normalized)
    for name, bounds in _EXTRA_FLOAT_RANGES.items():
        if name in normalized and not bounds[0] <= normalized[name] <= bounds[1]:
            raise RuntimeConfigurationError(
                f"{name} must be between {bounds[0]:g} and {bounds[1]:g}"
            )
    for name, bounds in _EXTRA_INT_RANGES.items():
        if name in normalized and not bounds[0] <= normalized[name] <= bounds[1]:
            raise RuntimeConfigurationError(
                f"{name} must be between {bounds[0]} and {bounds[1]}"
            )
    if normalized.get("num_keep", 0) > runtime.num_ctx:
        raise RuntimeConfigurationError("num_keep cannot exceed num_ctx")
    return normalized, runtime.warnings


def effective_model_options(options: dict | None = None) -> tuple[RuntimeConfig, dict]:
    """Merge validated session overrides over the selected runtime profile."""
    normalized, _warnings = validate_session_options(options)
    runtime = get_runtime_config(normalized)
    effective = runtime.ollama_options()
    effective.update({
        name: value for name, value in normalized.items()
        if name not in _CENTRAL_OPTION_NAMES
    })
    return runtime, effective


def effective_session_model_options(session: dict | None = None) -> tuple[RuntimeConfig, dict]:
    """Resolve a complete session mapping, including an optional profile."""
    session_data = session if isinstance(session, dict) else {}
    normalized, _warnings = validate_session_options(session_data.get("options", {}))
    runtime_input = dict(session_data)
    runtime_input["options"] = normalized
    runtime = get_runtime_config(runtime_input)
    effective = runtime.ollama_options()
    effective.update({
        name: value for name, value in normalized.items()
        if name not in _CENTRAL_OPTION_NAMES
    })
    return runtime, effective

def _estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a string using a fast heuristic.
    
    This is used to bound the history length without running a full tokenizer,
    saving compute time on every turn.
    
    Args:
        text (str): The text to estimate tokens for.
        
    Returns:
        int: The estimated token count (~1 token per 4 characters).
    """
    value = str(text or "")
    ascii_chars = sum(1 for character in value if ord(character) < 128)
    non_ascii_chars = len(value) - ascii_chars
    # Roughly four ASCII characters per token is typical for English/code.
    # Non-ASCII scripts can approach one token per code point, so count those
    # separately instead of letting a character-only heuristic under-budget
    # multilingual prompts and Unicode-heavy tool results.
    return ascii_chars // 4 + non_ascii_chars + 1


def _estimate_message_tokens(message: dict) -> int:
    """Estimate serialized chat-message tokens, including role/tool overhead."""
    try:
        serialized = json.dumps(message, ensure_ascii=False, default=str)
    except TypeError:
        serialized = str(message)
    return _estimate_tokens(serialized) + 4


def _estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(_estimate_message_tokens(message) for message in messages)


def _estimate_tool_schema_tokens(tools: list[dict] | None) -> int:
    if not tools:
        return 0
    try:
        return _estimate_tokens(json.dumps(tools, ensure_ascii=False, default=str))
    except TypeError:
        return _estimate_tokens(str(tools))


def _compact_json_schema(value):
    """Remove prose-heavy schema fields while preserving callable structure."""
    if isinstance(value, list):
        return [_compact_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    compact: dict = {}
    for key, child in value.items():
        if key == "description" and isinstance(child, str):
            continue
        compact[key] = _compact_json_schema(child)
    return compact


def compact_tool_schemas(tools: list[dict] | None) -> list[dict] | None:
    """Return lean Ollama tool schemas for lower prompt overhead."""
    if not tools:
        return tools
    try:
        cache_key = json.dumps(tools, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        cache_key = repr(tools)
    with _COMPACT_TOOL_SCHEMAS_LOCK:
        cached = _COMPACT_TOOL_SCHEMAS_CACHE.get(cache_key)
        if cached is not None:
            return cached

    compact_tools: list[dict] = []
    for tool in tools:
        function = dict(tool.get("function") or {})
        parameters = _compact_json_schema(function.get("parameters") or {})
        description = str(function.get("description") or "").strip()
        if len(description) > 100:
            description = description[:97].rstrip() + "..."
        compact_function = {
            "name": function.get("name"),
            "description": description,
            "parameters": parameters,
        }
        compact_tools.append({"type": "function", "function": compact_function})

    with _COMPACT_TOOL_SCHEMAS_LOCK:
        # Bound request-specific schema combinations in a long-running server.
        if len(_COMPACT_TOOL_SCHEMAS_CACHE) >= 16:
            _COMPACT_TOOL_SCHEMAS_CACHE.clear()
        _COMPACT_TOOL_SCHEMAS_CACHE[cache_key] = compact_tools
    return compact_tools


_DEFAULT_TOOL_NAMES = (
    "get_current_datetime",
    "web_search",
    "read_file",
    "vault_search",
    "list_vaults",
)
_TOOL_SELECTION_STOPWORDS = {
    "and", "are", "but", "can", "for", "from", "how", "into", "its",
    "please", "that", "the", "their", "then", "this", "use", "what",
    "when", "where", "which", "with", "would", "you", "your",
}
_TOOL_KEYWORD_HINTS = {
    "get_current_datetime": "date time today tomorrow yesterday timezone current now",
    "spreadsheet": "spreadsheet excel xlsx xls csv worksheet cells table",
    "web_search": "web internet online latest current news search research",
    "web_scrape": "website webpage url link article scrape page",
    "read_document": "pdf docx document pages extract",
    "read_file": "file text lines read inspect path",
    "create_file": "create write save new file",
    "create_pdf": "create write generate export pdf document notes report",
    "export_vault_pdf": "export entire complete vault pdf reference knowledge",
    "build_vault_notes_pdf": "generate refined lecture notes pdf from entire vault recursively exhaustive",
    "spotify_play": "spotify song music album playlist artist play",
    "open_browser": "browser website webapp open url",
    "view_code": "code source implementation function class inspect",
    "describe_image": "image picture screenshot diagram photo vision describe",
    "open_terminal_at_path": "terminal console shell directory folder open",
    "launch_apps": "launch application app desktop open start",
    "google_workspace": "google calendar tasks event birthday schedule",
    "codebase_indexer": "repository repo codebase architecture debug optimization security",
    "index_vault": "index vault document folder embeddings ingest",
    "vault_search": "vault notes knowledge documents semantic search",
    "vault_read": "vault read all exhaustive recursive ordered chunks pages cursor",
    "delete_vault_item": "delete remove vault collection chunks index",
    "list_vaults": "list vault collections indexes",
    "list_vault_aliases": "vault alias aliases list",
    "create_structured_note": "obsidian note markdown wikilink tags create",
    "knowledge_graph_builder": "knowledge graph concepts relationships path",
    "run_simulation": "simulation monte carlo scenario probability model",
    "api_orchestrator": "api http endpoint request integration",
    "context_memory_optimizer": "compact optimize conversation context memory",
    "reasoning_chain_debugger": "audit claim evidence reasoning confidence graph",
    "automated_routine_executor": "routine workflow recurring trigger automation",
}
_TOOL_COMPANIONS = {
    "web_search": ("web_scrape",),
    "index_vault": ("vault_search", "vault_read"),
    "vault_search": ("vault_read",),
}


def _tool_selection_text(messages: list[dict]) -> str:
    """Use the original request, not generic continuation boilerplate."""
    latest_user_text = next(
        (
            str(message.get("content", ""))
            for message in reversed(messages)
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        "",
    )
    marker = "\n\nOriginal user request:\n"
    if latest_user_text.startswith((
        "Continue the current user request using the tool result messages",
        "The previous response reached the model's per-call output limit",
    )) and marker in latest_user_text:
        return latest_user_text.split(marker, 1)[1]
    return latest_user_text


def _recent_called_tool_names(messages: list[dict]) -> list[str]:
    names: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            name = str((function or {}).get("name") or "").strip()
            if name and name not in names:
                names.append(name)
    return names[-4:]


def select_tool_schemas(
    messages: list[dict],
    session: dict | None,
    tools: list[dict] | None,
) -> list[dict] | None:
    """Select a bounded, deterministic schema set for the active request.

    Shipping every schema on every turn consumes most of a 4K context before
    conversation text is considered. Selection affects only model visibility;
    registry dispatch and all confirmations remain unchanged.
    """
    if not tools:
        return tools
    runtime = effective_session_model_options(session)[0]
    maximum = 10 if runtime.num_ctx <= 4096 else 16 if runtime.num_ctx <= 8192 else 24
    if len(tools) <= maximum:
        return list(tools)

    recent_user_text = _tool_selection_text(messages).casefold()
    normalized_text = re.sub(r"[^\w]+", " ", recent_user_text, flags=re.UNICODE)
    request_tokens = {
        token for token in normalized_text.split()
        if len(token) >= 3 and token not in _TOOL_SELECTION_STOPWORDS
    }

    by_name: dict[str, dict] = {}
    scores: dict[str, int] = {}
    for schema in tools:
        function = schema.get("function") if isinstance(schema, dict) else None
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        by_name[name] = schema
        searchable = " ".join((
            name.replace("_", " "),
            str(function.get("description") or ""),
            _TOOL_KEYWORD_HINTS.get(name, ""),
        )).casefold()
        searchable_tokens = {
            token for token in re.sub(r"[^\w]+", " ", searchable).split()
            if len(token) >= 3 and token not in _TOOL_SELECTION_STOPWORDS
        }
        overlap = request_tokens & searchable_tokens
        score = len(overlap)
        if name.casefold() in recent_user_text:
            score += 20
        scores[name] = score

    chosen: list[str] = [
        name for name, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if score > 0
    ]
    if not chosen:
        chosen.extend(_DEFAULT_TOOL_NAMES)
    else:
        for called_name in _recent_called_tool_names(messages):
            if called_name in by_name and called_name not in chosen:
                chosen.append(called_name)
            for companion in _TOOL_COMPANIONS.get(called_name, ()):
                if companion in by_name and companion not in chosen:
                    chosen.append(companion)
        for name in _DEFAULT_TOOL_NAMES:
            if name not in chosen:
                chosen.append(name)

    # A selected date-sensitive tool must retain its mandatory preflight.
    if any(
        (metadata := get_tool_metadata(name)) is not None
        and metadata.requires_temporal_preflight
        for name in chosen[:maximum]
    ) and "get_current_datetime" not in chosen[:maximum]:
        chosen.insert(0, "get_current_datetime")

    selected_names = set(chosen[:maximum])
    # Preserve registry order for stable prompts and model caching.
    return [schema for name, schema in by_name.items() if name in selected_names]


def tool_schemas_for_model(
    messages: list[dict],
    session: dict | None,
    tools: list[dict] | None = None,
) -> list[dict] | None:
    """Return request-selected compact schemas used for budgeting and sending."""
    return compact_tool_schemas(select_tool_schemas(messages, session, tools))


def _context_window_size(options: dict | None = None) -> int:
    try:
        return max(1024, int((options or {}).get("num_ctx") or DEFAULT_NUM_CTX))
    except (TypeError, ValueError):
        return DEFAULT_NUM_CTX


def _requested_response_tokens(options: dict | None = None) -> int:
    try:
        return max(MIN_RESPONSE_RESERVE_TOKENS, int((options or {}).get("num_predict") or DEFAULT_NUM_PREDICT))
    except (TypeError, ValueError):
        return DEFAULT_NUM_PREDICT


def _context_safety_margin(num_ctx: int) -> int:
    return max(256, int(num_ctx * CONTEXT_SAFETY_MARGIN_RATIO))


def _extract_system_prompt_from_modelfile() -> str:
    path = os.path.join(_PROJECT_ROOT, "Modelfile")
    try:
        with open(path, "r", encoding="utf-8") as stream:
            text = stream.read()
    except OSError:
        return ""
    match = re.search(r'SYSTEM\s+"""(.*?)"""', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _modelfile_mtime() -> float | None:
    try:
        return os.path.getmtime(os.path.join(_PROJECT_ROOT, "Modelfile"))
    except OSError:
        return None


def _write_system_prompt_cache(prompt: str) -> None:
    prompt = str(prompt or "").strip()
    if not prompt:
        return
    try:
        atomic_write_text(_SYSTEM_PROMPT_CACHE_FILE, prompt, durable=True)
    except OSError:
        pass


def _read_system_prompt_cache() -> str:
    try:
        with open(_SYSTEM_PROMPT_CACHE_FILE, "r", encoding="utf-8") as stream:
            return stream.read().strip()
    except OSError:
        return ""


def load_default_system_prompt(force_refresh: bool = False) -> str:
    """Load and persist the active model system prompt with durable fallbacks."""
    global _DEFAULT_SYSTEM_PROMPT, _DEFAULT_SYSTEM_PROMPT_MTIME
    with _SYSTEM_PROMPT_LOCK:
        current_mtime = _modelfile_mtime()
        if (
            _DEFAULT_SYSTEM_PROMPT is not None
            and not force_refresh
            and current_mtime == _DEFAULT_SYSTEM_PROMPT_MTIME
        ):
            return _DEFAULT_SYSTEM_PROMPT

        prompt = _extract_system_prompt_from_modelfile()
        if not prompt:
            try:
                result = _OLLAMA_SERVICE.show_model(MODEL_NAME, timeout=10)
                if hasattr(result, "model_dump"):
                    result = result.model_dump()
                prompt = str(
                    result.get("system", "") if isinstance(result, dict)
                    else getattr(result, "system", "")
                ).strip()
            except OllamaRuntimeError:
                pass

        if prompt:
            _write_system_prompt_cache(prompt)
        else:
            prompt = _read_system_prompt_cache()
            if prompt:
                _write_system_prompt_cache(prompt)

        _DEFAULT_SYSTEM_PROMPT = prompt
        _DEFAULT_SYSTEM_PROMPT_MTIME = current_mtime
        return prompt


def _split_system_and_conversation(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    system_messages: list[dict] = []
    conversation_messages: list[dict] = []
    for message in messages:
        if message.get("role") == "system":
            content = str(message.get("content", "")).strip()
            if content:
                system_messages.append({"role": "system", "content": content})
        else:
            conversation_messages.append(message)
    if system_messages:
        # One authoritative system message avoids accumulated stale overrides.
        system_messages = [system_messages[0]]
    return system_messages, conversation_messages


def _is_tool_continuation_prompt(message: dict) -> bool:
    return (
        message.get("role") == "user"
        and str(message.get("content") or "").startswith(
            "Continue the current user request using the tool result messages"
        )
    )


def _is_output_continuation_prompt(message: dict) -> bool:
    return (
        message.get("role") == "user"
        and str(message.get("content") or "").startswith(
            "The previous response reached the model's per-call output limit"
        )
    )


def _output_continuation_tail_start(messages: list[dict]) -> int | None:
    """Find the partial-answer/reminder pair needed for seamless continuation."""
    if len(messages) < 2 or not _is_output_continuation_prompt(messages[-1]):
        return None
    if messages[-2].get("role") != "assistant":
        return None
    return len(messages) - 2


def _fit_output_continuation_tail(messages: list[dict], token_budget: int) -> list[dict]:
    """Keep the latest possible answer suffix beside its continuation prompt."""
    tail = [dict(message) for message in messages]
    if _estimate_messages_tokens(tail) <= token_budget:
        return tail

    content = str(tail[0].get("content") or "")

    def with_suffix(suffix_chars: int) -> list[dict]:
        candidate = [dict(message) for message in tail]
        candidate[0]["content"] = content[-suffix_chars:] if suffix_chars else ""
        return candidate

    low = 0
    high = len(content)
    best = with_suffix(0)
    while low <= high:
        midpoint = (low + high) // 2
        candidate = with_suffix(midpoint)
        if _estimate_messages_tokens(candidate) <= token_budget:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best


def _tool_continuation_tail_start(messages: list[dict]) -> int | None:
    """Find the assistant/tool/user tail that must stay atomic after tools."""
    if not messages or not _is_tool_continuation_prompt(messages[-1]):
        return None
    for index in range(len(messages) - 2, -1, -1):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return index
        if message.get("role") == "user":
            break
    index = len(messages) - 2
    while index >= 0 and messages[index].get("role") == "tool":
        index -= 1
    return index + 1


def _compact_tool_continuation_tail(messages: list[dict]) -> list[dict]:
    """Keep tool-followup essentials while dropping prior thinking text."""
    compacted: list[dict] = []
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            compacted.append({
                "role": "assistant",
                # Thinking/content from the pre-tool response is no longer useful
                # once the structured call and its result are present.
                "content": "",
                "tool_calls": message.get("tool_calls", []),
            })
        else:
            compacted.append(dict(message))
    return compacted


def _truncate_tool_content_for_context(content: str, preview_chars: int) -> str:
    """Return a truthful preview when a tool result must yield response space."""
    value = str(content or "")
    if len(value) <= preview_chars:
        return value
    preview = value[:max(0, int(preview_chars))].rstrip()
    marker = (
        f"...[Tool result truncated from {len(value)} characters to fit the active "
        "context window. Use a narrower follow-up tool call if more detail is required.]"
    )
    return f"{preview}\n{marker}" if preview else marker


def _fit_tool_continuation_tail(messages: list[dict], token_budget: int) -> list[dict]:
    """Bound result previews while preserving the complete tool protocol tail."""
    compacted = _compact_tool_continuation_tail(messages)
    if _estimate_messages_tokens(compacted) <= token_budget:
        return compacted

    tool_indexes = [
        index for index, message in enumerate(compacted)
        if message.get("role") == "tool" and str(message.get("content") or "")
    ]
    if not tool_indexes:
        return compacted

    original_contents = {
        index: str(compacted[index].get("content") or "")
        for index in tool_indexes
    }

    def with_preview_limit(preview_chars: int) -> list[dict]:
        candidate = [dict(message) for message in compacted]
        for index, content in original_contents.items():
            candidate[index]["content"] = _truncate_tool_content_for_context(
                content,
                preview_chars,
            )
        return candidate

    # Use an exact estimate rather than a chars/token conversion: serialized
    # JSON, Unicode results, and escaping can all change the ratio materially.
    low = 0
    high = max(len(content) for content in original_contents.values())
    best = with_preview_limit(0)
    while low <= high:
        midpoint = (low + high) // 2
        candidate = with_preview_limit(midpoint)
        if _estimate_messages_tokens(candidate) <= token_budget:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best


_interrupted = False


def request_generation_interrupt() -> bool:
    """Request cooperative cancellation of the in-flight chat stream.

    Used by the TUI (Ctrl+C) and classic SIGQUIT handler. The streaming loop
    checks the flag between chunks and exits cleanly.

    Returns:
        bool: True if a generation may be in progress (flag was set).
    """
    global _interrupted
    _interrupted = True
    return True


def generation_interrupt_requested() -> bool:
    """Return whether cooperative generation cancel has been requested."""
    return bool(_interrupted)


def _sigquit_handler(signum, frame):
    """Handle SIGQUIT signal to interrupt the current LLM generation stream.
    
    Sets the global _interrupted flag to True, which is checked during
    token streaming to gracefully abort generation without crashing the agent.
    
    Args:
        signum: The signal number.
        frame: The current stack frame.
    """
    request_generation_interrupt()


def _trim_history(
    messages: list[dict],
    num_ctx: int = DEFAULT_NUM_CTX,
    *,
    reserved_tokens: int = DEFAULT_NUM_PREDICT,
    tool_schema_tokens: int = 0,
) -> list[dict]:
    """Trim conversation history to fit within a dynamic token budget.

    Preserves the system prompt (if any) and the most recent messages.
    Tool messages are kept with their associated assistant message.
    
    Args:
        messages (list[dict]): The full conversation history list.
        num_ctx (int, optional): The context window size. Defaults to 8192.
        
    Returns:
        list[dict]: A trimmed list of messages with response/tool headroom.
    """
    if not messages:
        return messages
    margin = _context_safety_margin(num_ctx)
    budget = max(
        512,
        int(num_ctx) - max(0, int(reserved_tokens)) - max(0, int(tool_schema_tokens)) - margin,
    )

    # Separate system prompt from conversation to ensure it is always preserved
    system_msgs, conv_msgs = _split_system_and_conversation(messages)

    # Calculate system prompt cost
    system_cost = _estimate_messages_tokens(system_msgs)
    remaining_budget = budget - system_cost

    if remaining_budget <= 0:
        # System prompt alone exceeds budget; keep it + last user message as a fallback
        return system_msgs + conv_msgs[-1:]

    output_tail_start = _output_continuation_tail_start(conv_msgs)
    tail_start = _tool_continuation_tail_start(conv_msgs)
    if output_tail_start is not None:
        preserved_tail = _fit_output_continuation_tail(
            conv_msgs[output_tail_start:],
            max(0, remaining_budget),
        )
        older_msgs = conv_msgs[:output_tail_start]
    elif tail_start is not None:
        preserved_tail = _fit_tool_continuation_tail(
            conv_msgs[tail_start:],
            max(0, remaining_budget),
        )
        older_msgs = conv_msgs[:tail_start]
    else:
        preserved_tail = []
        older_msgs = conv_msgs

    tail_cost = _estimate_messages_tokens(preserved_tail)
    if tail_cost >= remaining_budget:
        # Keep the complete continuation tail even if older history must go.
        # guarded_options_for_call will refuse cleanly if this still cannot fit.
        return system_msgs + preserved_tail

    # Walk from newest to oldest, accumulating messages until the budget is hit.
    # In a tool-followup turn, never keep the continuation prompt while dropping
    # the assistant tool-call or tool result it references.
    kept: list[dict] = []
    used = tail_cost
    for msg in reversed(older_msgs):
        cost = _estimate_message_tokens(msg)
        if used + cost > remaining_budget and (kept or preserved_tail):
            break
        kept.append(msg)
        used += cost

    kept.reverse()
    return system_msgs + kept + preserved_tail


def _system_prompt_anchor_index(conv_msgs: list[dict]) -> int:
    """Return an insertion point that does not split an atomic continuation tail."""
    output_tail_start = _output_continuation_tail_start(conv_msgs)
    if output_tail_start is not None:
        return output_tail_start

    tool_tail_start = _tool_continuation_tail_start(conv_msgs)
    if tool_tail_start is not None:
        return tool_tail_start

    for index in range(len(conv_msgs) - 1, -1, -1):
        if conv_msgs[index].get("role") == "user":
            return index
    return len(conv_msgs)


def _anchor_system_prompt(messages: list[dict], *, keep_leading_copy: bool) -> list[dict]:
    """Place the exact active system prompt beside the current turn.

    A full-window request can make a head-only system message ineffective even
    when it remains technically present. Repeating the authoritative message is
    preferred; constrained windows relocate that same message so policy is not
    replaced by a lossy summary.
    """
    system_msgs, conv_msgs = _split_system_and_conversation(messages)
    if not system_msgs or not conv_msgs:
        return messages

    anchor = dict(system_msgs[0])
    insert_at = _system_prompt_anchor_index(conv_msgs)
    leading = system_msgs if keep_leading_copy else []
    return [*leading, *conv_msgs[:insert_at], anchor, *conv_msgs[insert_at:]]


def prepare_messages_for_model(
    messages: list[dict],
    session: dict,
    tools: list[dict] | None = None,
    *,
    extra_reserved_tokens: int = 0,
) -> list[dict]:
    """Trim outgoing prompts with system preservation and response headroom."""
    _runtime, options = effective_session_model_options(session)
    num_ctx = _context_window_size(options)
    requested_response = _requested_response_tokens(options)
    runtime_tools = tool_schemas_for_model(messages, session, tools)
    tool_tokens = _estimate_tool_schema_tokens(runtime_tools)
    reserved = requested_response + max(0, int(extra_reserved_tokens))

    margin = _context_safety_margin(num_ctx)
    projected_untrimmed = (
        _estimate_messages_tokens(messages) + tool_tokens + reserved + margin
    )
    system_msgs, _conv_msgs = _split_system_and_conversation(messages)
    should_anchor = (
        bool(system_msgs)
        and projected_untrimmed >= int(num_ctx * SYSTEM_PROMPT_ANCHOR_THRESHOLD)
    )

    if not should_anchor:
        return _trim_history(
            messages,
            num_ctx,
            reserved_tokens=reserved,
            tool_schema_tokens=tool_tokens,
        )

    # Reserve the exact prompt's cost before adding a recent duplicate. This
    # drops older conversation first instead of silently crowding response space.
    anchor_tokens = _estimate_message_tokens(system_msgs[0])
    trimmed = _trim_history(
        messages,
        num_ctx,
        reserved_tokens=reserved + anchor_tokens,
        tool_schema_tokens=tool_tokens,
    )
    prepared = _anchor_system_prompt(trimmed, keep_leading_copy=True)
    projected = _estimate_messages_tokens(prepared) + tool_tokens + reserved + margin
    if projected <= num_ctx:
        return prepared

    # A low-context tool/output continuation may not fit two full copies. Keep
    # one exact prompt immediately before its atomic tail rather than substituting
    # a shortened policy or failing solely because of the duplicate.
    trimmed = _trim_history(
        messages,
        num_ctx,
        reserved_tokens=reserved,
        tool_schema_tokens=tool_tokens,
    )
    return _anchor_system_prompt(trimmed, keep_leading_copy=False)


def guarded_options_for_call(
    messages: list[dict],
    options: dict | None = None,
    tools: list[dict] | None = None,
    *,
    extra_reserved_tokens: int = 0,
) -> dict | None:
    """Return options with a safe per-call num_predict cap."""
    _runtime, base_options = effective_model_options(options)
    num_ctx = _context_window_size(base_options)
    runtime_tools = tool_schemas_for_model(messages, {"options": options or {}}, tools)
    prompt_tokens = _estimate_messages_tokens(messages) + _estimate_tool_schema_tokens(runtime_tools)
    available = (
        num_ctx
        - prompt_tokens
        - _context_safety_margin(num_ctx)
        - max(0, int(extra_reserved_tokens))
    )

    if available < MIN_EMERGENCY_RESPONSE_TOKENS:
        raise ContextWindowError(
            "The outgoing prompt is still too large after trimming. "
            "Increase num_ctx, start a fresh chat, or ask for a narrower answer."
        )

    requested = _requested_response_tokens(base_options)
    safe_predict = max(MIN_EMERGENCY_RESPONSE_TOKENS, min(requested, available))
    if safe_predict < requested:
        base_options["num_predict"] = safe_predict
    return base_options or None


def _compact_history_bg(history: list[dict], session: dict, start_idx: int, end_idx: int) -> None:
    """Compact a stable older-history slice without another model request.

    The former background summarizer raced live session mutation and could load
    a second large model while chat was active. The extractive optimizer is
    bounded, deterministic, and leaves the recent complete turns untouched.
    """
    messages_to_compact = [dict(message) for message in history[start_idx:end_idx]]
    try:
        if not messages_to_compact:
            return
        from tools.context_memory_optimizer import context_memory_optimizer

        optimized = json.loads(context_memory_optimizer(
            messages_to_compact,
            target_tokens=max(
                512,
                int(effective_session_model_options(session)[0].num_ctx * 0.25),
            ),
            preserve_recent=2,
        ))
        replacement = optimized.get("messages")
        if not isinstance(replacement, list) or not replacement:
            return

        # Do not overwrite a slice that changed while compaction was running.
        if history[start_idx:end_idx] == messages_to_compact:
            history[start_idx:end_idx] = replacement
    except Exception:
        # Compaction is an optimization; preserving the original history is the
        # safe failure mode.
        return
    finally:
        session.pop("_is_compacting", None)


def _check_and_compact_history(history: list[dict], session: dict) -> None:
    """Compact complete older turns once history exceeds 75% of context."""
    if session.get("_is_compacting"):
        return
        
    num_ctx = effective_session_model_options(session)[0].num_ctx
    compact_threshold = int(num_ctx * 0.75)
    
    total_tokens = _estimate_messages_tokens(history)
    
    if total_tokens <= compact_threshold:
        return

    start_idx = 1 if history and history[0].get("role") == "system" else 0
    user_indices = [
        index for index in range(start_idx, len(history))
        if history[index].get("role") == "user"
    ]
    # Keep the latest two complete user turns. Ending at a user boundary avoids
    # separating assistant tool calls from their tool results.
    if len(user_indices) < 3:
        return
    end_idx = user_indices[-2]
    if start_idx >= end_idx:
        return

    session["_is_compacting"] = True
    _compact_history_bg(history, session, start_idx, end_idx)

def _stream_thinking_response(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    options: dict | None = None,
    verbose: bool = False,
    think: bool = True,
    fmt: str | None = None,
    extra_reserved_tokens: int = 0,
) -> dict:
    """Stream a response from the Ollama model, displaying thinking progress and final answer.
    
    This function handles the complex logic of parsing an incoming stream of tokens,
    distinguishing between "thinking" tokens and "content" tokens, rendering them in real-time
    with markdown support, and appropriately handling interruptions.
    
    Args:
        model (str): The name of the model to use.
        messages (list[dict]): The chat history to send to the model.
        tools (list[dict] | None, optional): Available tools schema. Defaults to None.
        options (dict | None, optional): Model options (temperature, etc.). Defaults to None.
        verbose (bool, optional): Whether to print token generation stats. Defaults to False.
        think (bool, optional): Whether to enable the model's thinking process. Defaults to True.
        fmt (str | None, optional): Expected output format (e.g. 'json'). Defaults to None.

    Returns:
        dict: The full assistant message containing content, thinking (if any), and tool calls.
    """
    spinner = _Spinner("Thinking").start()
    t_start = time.monotonic()
    runtime_tools = tool_schemas_for_model(messages, {"options": options or {}}, tools)

    thinking_buf = ""
    content_buf = ""
    done_reason = ""
    eval_tokens = 0
    in_thinking = False
    thinking_displayed = False

    try:
        runtime_config, effective_options = effective_model_options(options)
        guarded_options = guarded_options_for_call(
            messages,
            effective_options,
            runtime_tools,
            extra_reserved_tokens=extra_reserved_tokens,
        )
    except (ContextWindowError, RuntimeConfigurationError) as exc:
        spinner.stop()
        message = f"Context window guard stopped this response before generation: {exc}"
        print_warn(message)
        _console.print()
        return {"role": "assistant", "content": message}

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": think,
        "keep_alive": runtime_config.keep_alive,
    }
    if fmt:
        kwargs["format"] = fmt
    if runtime_tools:
        kwargs["tools"] = runtime_tools
    if guarded_options:
        kwargs["options"] = guarded_options

    owner = f"cli:{threading.get_ident()}:{time.monotonic_ns()}"
    try:
        stream = _OLLAMA_SERVICE.chat(
            kind=OperationKind.CHAT,
            owner=owner,
            operation_timeout=runtime_config.chat_timeout_seconds,
            **kwargs,
        )
    except OllamaRuntimeError as exc:
        spinner.stop()
        message = f"Ollama chat failed before streaming: {exc}"
        print_error(message)
        _console.print()
        return {"role": "assistant", "content": message}


    live = None
    _last_render = 0.0  # throttle Live.update() calls
    _RENDER_INTERVAL = 0.08  # seconds between re-renders (~12 FPS)

    global _interrupted
    _interrupted = False
    interrupt_signal = getattr(signal, "SIGQUIT", None) or getattr(signal, "SIGBREAK", None)
    old_handler = None
    if interrupt_signal is not None:
        try:
            old_handler = signal.signal(interrupt_signal, _sigquit_handler)
        except (OSError, ValueError):
            interrupt_signal = None

    try:
        for chunk in stream:
            if _interrupted:
                spinner.stop()
                if in_thinking:
                    in_thinking = False
                    print_thinking_footer("interrupted")
                print_warn("Generation interrupted by user")
                if not display_is_tui():
                    _console.print()
                break

            try:
                done_reason = _chunk_done_reason(chunk) or done_reason
                chunk_eval_count = (
                    chunk.get("eval_count") if isinstance(chunk, dict)
                    else getattr(chunk, "eval_count", None)
                )
                if chunk_eval_count is not None:
                    eval_tokens = int(chunk_eval_count or 0)
                msg = getattr(chunk, "message", None)
                if msg is None and isinstance(chunk, dict):
                    msg = chunk.get("message")
                if msg is None:
                    # Skip keep-alive or heartbeat frames without a message body.
                    continue
                if isinstance(msg, dict):
                    class _MsgView:
                        __slots__ = ("content", "thinking", "tool_calls")

                        def __init__(self, payload: dict) -> None:
                            self.content = payload.get("content") or ""
                            self.thinking = payload.get("thinking") or ""
                            raw_calls = payload.get("tool_calls") or []
                            self.tool_calls = raw_calls

                    msg = _MsgView(msg)

                tool_calls = getattr(msg, "tool_calls", None) or None
                # ── Tool calls come through as non-streamed chunks ────────
                if tool_calls:
                    spinner.stop()
                    assistant_msg = {"role": "assistant", "content": content_buf}
                    if thinking_buf:
                        assistant_msg["thinking"] = thinking_buf
                    normalized_calls = []
                    for tc in tool_calls:
                        try:
                            if isinstance(tc, dict):
                                function = tc.get("function") or {}
                                name = function.get("name") if isinstance(function, dict) else None
                                arguments = function.get("arguments") if isinstance(function, dict) else {}
                            else:
                                function = getattr(tc, "function", None)
                                name = getattr(function, "name", None)
                                arguments = getattr(function, "arguments", {})
                            if not name:
                                continue
                            normalized_calls.append(
                                {"function": {"name": name, "arguments": arguments or {}}}
                            )
                        except Exception:
                            continue
                    if normalized_calls:
                        assistant_msg["tool_calls"] = normalized_calls
                        return assistant_msg
                    continue

                # ── Thinking tokens ───────────────────────────────────────
                thinking_chunk = getattr(msg, "thinking", None) or ""
                if thinking_chunk:
                    if not in_thinking:
                        in_thinking = True
                        spinner.stop()
                        print_thinking_header()
                        thinking_displayed = True

                    thinking_buf += str(thinking_chunk)
                    print_thinking_delta(str(thinking_chunk))
                    continue

                # ── Content tokens ────────────────────────────────────────
                content_chunk = getattr(msg, "content", None) or ""
                if content_chunk:
                    if in_thinking:
                        in_thinking = False
                        print_thinking_footer()
                        spinner.stop()
                    elif spinner._thread and not spinner._stop_event.is_set():
                        spinner.stop()
                        if not thinking_displayed and not display_is_tui():
                            _console.print()

                    content_buf += str(content_chunk)

                    if display_is_tui():
                        now = time.monotonic()
                        if now - _last_render >= _RENDER_INTERVAL:
                            print_content_stream(content_buf)
                            _last_render = now
                    else:
                        if live is None:
                            live = Live(
                                assistant_stream_panel(content_buf),
                                console=_console,
                                auto_refresh=False,
                                screen=False,
                                transient=True,
                                vertical_overflow="visible",
                            )
                            live.start()

                        now = time.monotonic()
                        if now - _last_render >= _RENDER_INTERVAL:
                            live.update(assistant_stream_panel(content_buf), refresh=True)
                            _last_render = now
            except Exception as exc:
                spinner.stop()
                message = (
                    "Ollama returned an unexpected stream chunk shape; "
                    f"generation stopped safely: {exc}"
                )
                print_error(message)
                _console.print()
                if not content_buf:
                    content_buf = message
                break

    except OllamaRuntimeError as exc:
        spinner.stop()
        message = f"Ollama chat failed while streaming: {exc}"
        print_error(message)
        _console.print()
        if not content_buf:
            content_buf = message
    except Exception as exc:
        spinner.stop()
        message = f"Ollama stream ended with an unexpected error: {exc}"
        print_error(message)
        _console.print()
        if not content_buf:
            content_buf = message
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        if interrupt_signal is not None and old_handler is not None:
            try:
                signal.signal(interrupt_signal, old_handler)
            except (OSError, ValueError):
                pass
        if live:
            live.stop()
        flush_terminal_input()

    # End of stream
    spinner.stop()

    if in_thinking:
        # Stream ended while still in thinking (no content followed)
        print_thinking_footer()

    if content_buf:
        # Flush any throttled stream frame, then pin the final markdown card.
        if display_is_tui():
            print_content_stream(content_buf)
        print_assistant_message(content_buf)

    # Verbose stats
    if verbose:
        elapsed = time.monotonic() - t_start
        t_tokens = len(thinking_buf.split()) if thinking_buf else 0
        c_tokens = len(content_buf.split()) if content_buf else 0
        total = t_tokens + c_tokens
        tps = total / elapsed if elapsed > 0 else 0
        print_generation_stats(
            elapsed=elapsed,
            total_tokens=total,
            tokens_per_sec=tps,
        )

    # Build the full message for history
    assistant_msg = {"role": "assistant", "content": content_buf}
    if thinking_buf:
        assistant_msg["thinking"] = thinking_buf
    assistant_msg["_done_reason"] = done_reason
    assistant_msg["_eval_count"] = eval_tokens
    assistant_msg["_num_predict"] = int(guarded_options.get("num_predict", 0))
    return assistant_msg


def _stream_complete_response(
    *,
    model: str,
    messages: list[dict],
    session: dict,
    user_input: str,
    tools: list[dict] | None = None,
    extra_reserved_tokens: int = 0,
) -> dict:
    """Stream one logical answer across bounded Ollama output-limit stops."""
    response = _stream_thinking_response(
        model=model,
        messages=messages,
        tools=tools,
        options=effective_session_model_options(session)[1],
        verbose=session.get("verbose", True),
        think=session.get("think", True),
        fmt=session.get("format") or None,
        extra_reserved_tokens=extra_reserved_tokens,
    )
    combined_content = str(response.get("content") or "")
    combined_thinking = str(response.get("thinking") or "")
    continuation_rounds = 0

    while (
        not response.get("tool_calls")
        and combined_content
        and _output_limit_reached(
            response.get("_done_reason", ""),
            response.get("_eval_count", 0),
            response.get("_num_predict", 0),
        )
        and continuation_rounds < MAX_OUTPUT_CONTINUATION_ROUNDS
    ):
        continuation_rounds += 1
        reminder = {
            "role": "user",
            "content": OUTPUT_CONTINUATION_PROMPT.format(user_input=user_input),
        }
        continuation_messages = prepare_messages_for_model(
            [
                *messages,
                {"role": "assistant", "content": combined_content},
                reminder,
            ],
            session,
            tools=None,
        )
        response = _stream_thinking_response(
            model=model,
            messages=continuation_messages,
            tools=None,
            options=effective_session_model_options(session)[1],
            verbose=session.get("verbose", True),
            think=False,
            fmt=session.get("format") or None,
        )
        next_content = str(response.get("content") or "")
        if not next_content:
            break
        combined_content += next_content
        if response.get("thinking"):
            combined_thinking += str(response["thinking"])

    still_limited = _output_limit_reached(
        response.get("_done_reason", ""),
        response.get("_eval_count", 0),
        response.get("_num_predict", 0),
    )
    if still_limited and continuation_rounds >= MAX_OUTPUT_CONTINUATION_ROUNDS:
        notice = (
            "\n\n[Selene paused after the safe automatic continuation limit. "
            "Ask to continue if you need more.]"
        )
        combined_content += notice
        _console.print(f"[yellow]{notice.strip()}[/]")

    combined = {"role": "assistant", "content": combined_content}
    if combined_thinking:
        combined["thinking"] = combined_thinking
    if response.get("tool_calls"):
        combined["tool_calls"] = response["tool_calls"]
    return combined


def _tool_detail(spec: ToolCallSpec) -> str | None:
    """Compact argument preview for tool status chrome."""
    args = spec.arguments or {}
    for key in ("query", "url", "file_path", "path", "app_name", "collection", "title"):
        value = args.get(key)
        if value:
            text = str(value)
            return text if len(text) <= 72 else text[:69] + "…"
    names = args.get("app_names")
    if isinstance(names, list) and names:
        joined = ", ".join(str(item) for item in names[:4])
        return joined if len(joined) <= 72 else joined[:69] + "…"
    return None


def _tool_start_status(spec: ToolCallSpec) -> None:
    if spec.name not in TOOL_DISPATCH:
        print_tool_event(spec.name or "?", phase="error", message="unknown tool")
        return
    print_tool_event(spec.name, phase="run", detail=_tool_detail(spec))


def _tool_result_has_error(content: str) -> str | None:
    try:
        data = json.loads(content)
    except Exception:
        return None
    if isinstance(data, dict) and data.get("error"):
        return str(data["error"])
    return None


def _tool_end_status(result: ToolCallResult) -> None:
    error = _tool_result_has_error(result.content)
    if error:
        print_tool_event(result.spec.name, phase="error", detail=error[:120])
        return
    print_tool_event(
        result.spec.name,
        phase="ok",
        message="complete · synthesizing",
    )


def _process_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Execute each tool call from the model and format the results.
    
    Iterates over the tool_calls list, dynamically loading the corresponding handler
    from TOOL_DISPATCH, executing it with the provided arguments, and wrapping the
    result in a standard tool-role message to feed back to the model.
    
    Args:
        tool_calls (list[dict]): A list of tool call dictionary objects.
        
    Returns:
        list[dict]: A list of message objects containing the execution results.
    """
    results = execute_tool_calls(
        tool_calls,
        on_start=_tool_start_status,
        on_end=_tool_end_status,
        on_parallel_batch=lambda batch: print_tool_event(
            "batch",
            phase="parallel",
            detail=str(len(batch)),
        ),
    )
    return [result.as_tool_message() for result in results]


def _tool_call_turn_key(call: dict) -> str | None:
    """Return a stable key for suppressing identical calls in one turn."""
    function = call.get("function") or {}
    name = str(function.get("name") or "").strip()
    if not name:
        return None
    arguments, argument_error = normalize_tool_arguments(function.get("arguments"))
    if argument_error:
        arguments = {"_argument_error": argument_error}
    try:
        encoded_args = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        encoded_args = str(arguments)
    return f"{name}:{encoded_args}"


def _process_tool_calls_with_turn_guard(tool_calls: list[dict], executed_tool_calls: dict[str, dict]) -> list[dict]:
    """Execute tool calls while preventing duplicate calls in one turn."""
    pending_calls: list[dict] = []
    pending_positions: list[int] = []
    pending_keys: dict[int, str] = {}
    pending_key_to_index: dict[str, int] = {}
    duplicate_source_by_index: dict[int, int] = {}
    results_by_index: dict[int, dict] = {}

    for index, call in enumerate(tool_calls):
        turn_key = _tool_call_turn_key(call)
        if turn_key and turn_key in executed_tool_calls:
            results_by_index[index] = dict(executed_tool_calls[turn_key])
            continue
        if turn_key and turn_key in pending_key_to_index:
            duplicate_source_by_index[index] = pending_key_to_index[turn_key]
            continue
        if turn_key:
            pending_keys[index] = turn_key
            pending_key_to_index[turn_key] = index
        pending_positions.append(index)
        pending_calls.append(call)

    if pending_calls:
        pending_results = _process_tool_calls(pending_calls)
        for index, result in zip(pending_positions, pending_results):
            results_by_index[index] = result
            if index in pending_keys:
                executed_tool_calls[pending_keys[index]] = dict(result)

    for index, source_index in duplicate_source_by_index.items():
        results_by_index[index] = dict(results_by_index[source_index])

    return [results_by_index[index] for index in sorted(results_by_index)]


# ── Slash commands ────────────────────────────────────────────────────

# Runtime profiles (shared by /set profile and /profile).
_PROFILE_SPECS: tuple[tuple[str, str], ...] = (
    ("manual", "Modelfile defaults (recommended)"),
    ("auto", "Pick low-vram or balanced from VRAM"),
    ("low-vram", "Conservative ~4 GiB settings"),
    ("balanced", "Higher ctx/batch for larger GPUs"),
)
_PROFILE_NAMES = frozenset(name for name, _ in _PROFILE_SPECS)

# Common model knobs surfaced in Tab autocomplete (full list still via /set parameter).
_PARAMETER_SPECS: tuple[tuple[str, str], ...] = (
    ("temperature", "Sampling randomness  ·  e.g. 0.25"),
    ("top_p", "Nucleus sampling  ·  e.g. 0.85"),
    ("top_k", "Top-k sampling  ·  e.g. 40"),
    ("num_ctx", "Context window tokens  ·  e.g. 8192"),
    ("num_predict", "Max output tokens per call  ·  e.g. 2048"),
    ("num_batch", "Prompt batch size  ·  e.g. 128"),
    ("repeat_penalty", "Repetition penalty  ·  e.g. 1.08"),
)


def _build_cli_slash_specs() -> tuple[tuple[str, str], ...]:
    """Catalog for Tab palette + descriptions (scannable, not verbose)."""
    specs: list[tuple[str, str]] = [
        ("/help", "Commands and usage"),
        ("/?", "Commands and usage"),
        ("/clear", "Reset conversation + system override"),
        ("/speech", "Open speech menu  ·  /speech [start|stop]  ·  Ctrl+S in TUI"),
        ("/speech start", "Open speech menu and start listening"),
        ("/speech stop", "Stop listening (menu stays open)"),
        ("/save", "Save session  ·  /save [name]"),
        ("/load", "Load session  ·  /load [name|index]"),
        # Profiles — first-class + /set form for discoverability.
        ("/profile", "Show or set profile  ·  /profile <name>"),
        ("/set profile", "Set profile  ·  manual|auto|low-vram|balanced"),
    ]
    for name, desc in _PROFILE_SPECS:
        specs.append((f"/profile {name}", desc))
        specs.append((f"/set profile {name}", desc))

    # TUI themes (default first via tui_themes.theme_specs_for_slash).
    try:
        from agent.tui_themes import theme_specs_for_slash

        specs.extend(theme_specs_for_slash())
    except Exception:
        specs.append(("/theme", "TUI color theme  ·  /theme <place>"))
        specs.append(("/theme oslo", "Oslo — monochrome grey & white (default)"))

    specs.extend(
        [
            ("/set parameter", "Set model knob  ·  /set parameter <name> <value>"),
        ]
    )
    for name, desc in _PARAMETER_SPECS:
        specs.append((f"/set parameter {name}", desc))

    specs.extend(
        [
            ("/set system", "Session system prompt  ·  /set system \"…\"|default"),
            ("/set history", "Keep multi-turn context (default)"),
            ("/set nohistory", "One-shot turns only"),
            ("/set wordwrap", "Wrap long lines (default)"),
            ("/set nowordwrap", "Disable line wrap"),
            ("/set format", "Force output format  ·  /set format json"),
            ("/set format json", "JSON-only model output"),
            ("/set noformat", "Normal free-form output (default)"),
            ("/set verbose", "Show generation timing/stats (default)"),
            ("/set quiet", "Hide generation stats"),
            ("/set think", "Stream model thinking (default)"),
            ("/set nothink", "Answer without thinking stream"),
            ("/show parameters", "Active profile + model options"),
            ("/show system", "Active system prompt"),
            ("/show model", "Model name / family / size"),
            ("/show profile", "Current runtime profile"),
            ("/vault list", "Indexed vault collections"),
            ("/vault aliases", "Friendly collection aliases"),
            ("/vault alias", "Map name → collection  ·  /vault alias <name> <coll>"),
            ("/vault rename", "Rename collection  ·  /vault rename <old> <new>"),
            ("/vault add", "Index path  ·  /vault add <path> [--collection n]"),
            ("/vault status", "PDF index progress  ·  /vault status <pdf>"),
            ("/vault read", "Read chunks in order  ·  [--cursor n]"),
            ("/vault search", "Semantic search  ·  /vault search <query>"),
            ("/vault delete", "Remove indexed source  ·  /vault delete <path>"),
            ("/quit", "Exit Selene"),
            ("/exit", "Exit Selene"),
            ("/q", "Exit Selene"),
        ]
    )
    return tuple(specs)


CLI_SLASH_SPECS: tuple[tuple[str, str], ...] = _build_cli_slash_specs()
CLI_SLASH_COMPLETIONS = tuple(command for command, _ in CLI_SLASH_SPECS)
CLI_SLASH_DESCRIPTIONS = {command: description for command, description in CLI_SLASH_SPECS}

# /help card — usage forms on the left, short effect on the right.
_COMMAND_HELP_ENTRIES: tuple[tuple[str, str], ...] = (
    ("/help", "Show this help"),
    ("/clear", "Clear history and system override"),
    ("/speech [start|stop]", "Open speech menu (Ctrl+S in TUI)"),
    ("/save [name]", "Save this session"),
    ("/load [name|index]", "Load a session (lists if no arg)"),
    ("/profile [name]", "Show or set profile (manual · auto · low-vram · balanced)"),
    ("/set profile <name>", "Same as /profile <name>"),
    ("/theme [place]", "TUI colors (oslo · tokyo · rome · amazon · …)"),
    ("/set parameter <name> <val>", "Model option (temperature, num_ctx, …)"),
    ("/set system \"…\"|default", "Override or reset system prompt"),
    ("/set history | nohistory", "Multi-turn context on/off"),
    ("/set wordwrap | nowordwrap", "Line wrap on/off"),
    ("/set format json | noformat", "Force JSON or free-form output"),
    ("/set verbose | quiet", "Generation stats on/off"),
    ("/set think | nothink", "Thinking stream on/off"),
    ("/show parameters | system | model | profile", "Inspect session/runtime"),
    ("/vault …", "Vault tools — /vault help for details"),
    ("/quit", "Exit (also /exit, /q)"),
)

_VAULT_HELP_ENTRIES: tuple[tuple[str, str], ...] = (
    ("/vault list", "List indexed collections"),
    ("/vault aliases", "List name → collection maps"),
    ("/vault alias <name> <coll>", "Register an alias"),
    ("/vault rename <old> <new>", "Rename a collection"),
    ("/vault add <path> [--collection n]", "Index a file or folder"),
    ("/vault add <path> [--vision auto|all|off]", "PDF vision capture policy"),
    ("/vault status <path> [--collection n]", "Resumable PDF progress"),
    ("/vault read [--cursor n] [--source p]", "Walk chunks in source order"),
    ("/vault search <query> [--top-k n]", "Semantic search"),
    ("/vault search <query> [--source p]", "Search one source only"),
    ("/vault delete <source> [--collection n]", "Remove indexed chunks"),
    ("/vault delete --all [--collection n]", "Delete a whole collection"),
)

# Back-compat names used by a few older call sites / tests.
_COMMANDS_HELP = _COMMAND_HELP_ENTRIES
_VAULT_HELP = _VAULT_HELP_ENTRIES


def _print_profile_catalog() -> None:
    """List runtime profiles with short descriptions."""
    print_info("Runtime profiles")
    for name, description in _PROFILE_SPECS:
        _console.print(f"    [bold]{name:<10}[/]  [dim]{description}[/]")
    print_info("Usage · /profile <name>  or  /set profile <name>")
    _console.print()


def _handle_speech(args: str, session: dict | None = None) -> None:
    """Toggle or one-shot voice input (Web UI mic parity for CLI/TUI).

    In the full-screen TUI, this opens the centered speech menu (same path as
    Ctrl+S). In classic CLI mode it does a single listen pass and prints text.
    """
    action = str(args or "").strip().lower()
    if action in {"", "toggle", "start", "stop", "on", "off", "status"}:
        pass
    else:
        print_error(f"Unknown /speech argument · {args}")
        print_info("Usage · /speech  |  /speech start  |  /speech stop")
        _console.print()
        return

    # Prefer live TUI controller when available (same menu as Ctrl+S).
    try:
        from agent.terminal import get_display_sink

        sink = get_display_sink()
        if sink is not None and hasattr(sink, "toggle_speech"):
            sink.toggle_speech(action or "toggle")
            return
    except Exception:
        pass

    if action in {"stop", "off"}:
        print_info("Voice input is only active in the TUI (Ctrl+S / /speech)")
        _console.print()
        return

    if action == "status":
        try:
            from agent.speech_input import speech_capability

            cap = speech_capability()
            if cap.available:
                print_ok("Voice input available", detail=cap.detail)
            else:
                print_warn("Voice input unavailable", detail=cap.detail)
        except Exception as exc:
            print_error(f"Voice input check failed · {exc}")
        _console.print()
        return

    # Classic CLI one-shot listen.
    print_lab_status("Listening…", kind="run", detail="speak now")
    try:
        from agent.speech_input import capture_once

        text, error = capture_once()
    except Exception as exc:
        print_error(f"Voice input failed · {exc}")
        _console.print()
        return

    if error:
        print_warn(error)
        _console.print()
        return
    if not text:
        print_warn("I didn't hear anything")
        _console.print()
        return
    print_ok("Heard", detail=text)
    print_info("Tip · in the TUI, Ctrl+S or /speech opens the speech menu")
    _console.print()


def _handle_theme(args: str, session: dict) -> None:
    """Show or apply a TUI place-named color theme (default is Oslo)."""
    try:
        from agent.tui_themes import (
            DEFAULT_THEME,
            is_valid_theme,
            normalize_theme_name,
            theme_catalog,
            theme_label,
        )
    except Exception as exc:
        print_error(f"Themes unavailable · {exc}")
        _console.print()
        return

    raw = str(args or "").strip()
    if not raw:
        current = normalize_theme_name(session.get("tui_theme") or DEFAULT_THEME)
        print_info(f"Current theme · {current}  ({theme_label(current)})")
        print_info("TUI themes (places)")
        for name, label in theme_catalog():
            mark = "  ←" if name == current else ""
            _console.print(f"    [bold]{name:<12}[/]  [dim]{label}{mark}[/]")
        print_info("Usage · /theme <place>")
        _console.print()
        return

    if not is_valid_theme(raw):
        print_error(f"Unknown theme · {raw}")
        print_info("Try · " + " · ".join(name for name, _ in theme_catalog()))
        _console.print()
        return

    key = normalize_theme_name(raw)
    session["tui_theme"] = key
    # Apply live when the full-screen TUI is active.
    try:
        from agent.terminal import get_display_sink

        sink = get_display_sink()
        if sink is not None and hasattr(sink, "apply_theme"):
            sink.apply_theme(key)
            return
    except Exception:
        pass
    print_ok(f"Theme · {key}", detail=theme_label(key))
    print_info("Applies the next time the TUI is opened")
    _console.print()


def _apply_runtime_profile(session: dict, profile: str) -> None:
    """Validate and apply a runtime profile name to the session."""
    normalized = profile.strip().lower().replace("_", "-")
    if not normalized:
        _print_profile_catalog()
        current = str(session.get("runtime_profile") or "manual")
        print_info(f"Current profile · {current}")
        _console.print()
        return
    if normalized not in _PROFILE_NAMES:
        print_error(
            f"Unknown profile · {normalized}",
            detail="manual · auto · low-vram · balanced",
        )
        _print_profile_catalog()
        return

    candidate = dict(session)
    candidate["runtime_profile"] = normalized
    try:
        runtime = get_runtime_config(candidate)
    except RuntimeConfigurationError as exc:
        print_error(f"Invalid runtime profile · {exc}")
        _console.print()
        return

    session["runtime_profile"] = normalized
    print_ok(f"Runtime profile · {runtime.profile.value}")
    if runtime.requested_profile.value != runtime.profile.value:
        print_info(
            f"Resolved · {runtime.requested_profile.value} → {runtime.profile.value}"
        )
    print_info(runtime.selection_reason)
    print_info(
        f"ctx {runtime.num_ctx}  ·  out {runtime.num_predict}  ·  "
        f"batch {runtime.num_batch}  ·  temp {runtime.temperature}"
    )
    for warning in runtime.warnings:
        print_warn(warning)
    _console.print()


def _handle_set(args: str, session: dict, history: list[dict]) -> None:
    """Handle /set sub-commands.
    
    Parses user input to modify session state like toggling flags (verbose, history)
    or updating underlying model options (temperature, top_p).
    
    Args:
        args (str): The raw arguments passed after the '/set ' command.
        session (dict): The current session dictionary containing state.
        history (list[dict]): The conversation history list.
    """
    parts = args.strip().split(None, 1)
    if not parts:
        print_info("Usage · /set <subcommand> [args]")
        print_info(
            "Subcommands · profile · parameter · system · history · nohistory · "
            "wordwrap · nowordwrap · format · noformat · verbose · quiet · think · nothink"
        )
        print_info("Tip · type /help or start with /set and Tab")
        _console.print()
        return

    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    # ── /set verbose / /set quiet ─────────────────────────────────────
    if sub == "verbose":
        session["verbose"] = True
        print_ok("Verbose mode enabled — stats shown after each response")
        _console.print()
        return
    if sub == "quiet":
        session["verbose"] = False
        print_ok("Quiet mode enabled")
        _console.print()
        return

    # ── /set wordwrap / /set nowordwrap ───────────────────────────────
    if sub == "wordwrap":
        session["wordwrap"] = True
        print_ok("Word wrapping enabled")
        _console.print()
        return
    if sub == "nowordwrap":
        session["wordwrap"] = False
        print_ok("Word wrapping disabled")
        _console.print()
        return

    # ── /set history / /set nohistory ─────────────────────────────────
    if sub == "history":
        session["history"] = True
        print_ok("Conversation history enabled")
        _console.print()
        return
    if sub == "nohistory":
        session["history"] = False
        print_ok("History disabled — each turn is now standalone")
        _console.print()
        return

    # ── /set format json / /set noformat ──────────────────────────────
    if sub == "format":
        fmt = rest.strip().lower()
        if fmt == "json":
            session["format"] = "json"
            print_ok("JSON output mode enabled")
            _console.print()
        else:
            _console.print(f"[red]Unsupported format: {fmt}[/]  [dim](supported: json)[/]\n")
        return
    if sub == "noformat":
        session["format"] = ""
        print_ok("Output formatting reset to default")
        _console.print()
        return

    # ── /set think / /set nothink ─────────────────────────────────────
    if sub == "think":
        session["think"] = True
        print_ok("Thinking/reasoning enabled")
        _console.print()
        return
    if sub == "nothink":
        session["think"] = False
        print_ok("Thinking disabled — model will respond directly")
        _console.print()
        return

    # ── /set system "<prompt>" ────────────────────────────────────────
    if sub == "system":
        # Strip surrounding quotes if present
        prompt = rest.strip().strip('"').strip("'")
        
        # Remove any existing system messages from history to avoid duplicates
        history[:] = [m for m in history if m.get("role") != "system"]

        if not prompt or prompt.lower() == "default":
            session["system"] = ""
            print_ok("System prompt reset to default")
            _console.print()
            return

        # Insert new system message at the start
        history.insert(0, {"role": "system", "content": prompt})
        session["system"] = prompt
        
        # Truncate display for confirmation
        display = prompt if len(prompt) <= 80 else prompt[:77] + "…"
        print_ok("System prompt set", detail=display)
        _console.print()
        return

    if sub == "profile":
        _apply_runtime_profile(session, rest)
        return

    # ── /set parameter <name> <value> ─────────────────────────────────
    if sub == "parameter":
        param_parts = rest.strip().split(None, 1)
        if len(param_parts) != 2:
            print_error("Usage · /set parameter <name> <value>")
            print_info("Common · " + " · ".join(name for name, _ in _PARAMETER_SPECS))
            print_info("All · " + ", ".join(sorted(_ALL_PARAMS)))
            _console.print()
            return

        name, raw_val = param_parts[0].lower(), param_parts[1]

        if name not in _ALL_PARAMS:
            print_error(f"Unknown parameter · {name}")
            print_info("Common · " + " · ".join(n for n, _ in _PARAMETER_SPECS))
            print_info("All · " + ", ".join(sorted(_ALL_PARAMS)))
            _console.print()
            return

        try:
            value = float(raw_val) if name in _FLOAT_PARAMS else int(raw_val)
        except ValueError:
            expected = "float" if name in _FLOAT_PARAMS else "integer"
            print_error(f"Invalid value for {name}", detail=f"expected {expected}, got {raw_val!r}")
            _console.print()
            return

        candidate = dict(session.get("options", {}))
        candidate[name] = value
        try:
            normalized, warnings = validate_session_options(candidate)
        except RuntimeConfigurationError as exc:
            _console.print(f"[red]Invalid value for {name}: {exc}[/]\n")
            return
        session["options"] = normalized
        print_ok(f"{name} = {value}")
        for warning in warnings:
            print_warn(warning)
        _console.print()
        return

    print_error(
        f"Unknown /set subcommand · {sub}",
        detail="profile · parameter · system · history · format · verbose · think · …",
    )
    print_info("Tip · /set  or  /help  for the full list")
    _console.print()


def _handle_show(args: str, session: dict, history: list[dict]) -> None:
    """Handle /show sub-commands.
    
    Allows the user to print the current session state, such as active parameters,
    the system prompt, or hardware/model info.
    
    Args:
        args (str): The string following the '/show ' command.
        session (dict): The active session dictionary.
        history (list[dict]): The conversation history list.
    """
    sub = args.strip().lower() or "parameters"

    if sub == "parameters":
        opts = session.get("options", {})
        try:
            runtime, effective = effective_session_model_options(session)
        except RuntimeConfigurationError as exc:
            _console.print(f"[red]Invalid session configuration: {exc}[/]\n")
            return
        _console.print()
        print_info(f"Runtime profile · {runtime.profile.value}")
        print_info(runtime.selection_reason)
        print_info("Effective model parameters")
        for key, value in sorted(effective.items()):
            suffix = "  (session override)" if key in opts else ""
            _console.print(f"    [green]{key}[/] = {value}[dim]{suffix}[/]")
        for warning in runtime.warnings:
            print_warn(warning)
        _console.print()
        # Also show flags
        flags = []
        if session.get("verbose"):
            flags.append("verbose")
        if not session.get("wordwrap", True):
            flags.append("nowordwrap")
        if not session.get("history", True):
            flags.append("nohistory")
        if session.get("format"):
            flags.append(f"format={session['format']}")
        if not session.get("think", True):
            flags.append("nothink")
        if flags:
            print_info(f"Flags: {', '.join(flags)}")
            _console.print()
        return

    if sub == "system":
        prompt = session.get("system", "")
        if not prompt:
            # Check if history has one from the Modelfile
            if history and history[0].get("role") == "system":
                prompt = history[0]["content"]
        if prompt:
            _console.print()
            print_info("System prompt")
            _console.print(f"  [dim]{prompt}[/]\n")
        else:
            print_info("No system prompt set (using Modelfile default)")
            _console.print()
        return

    if sub in ("model", "info"):
        try:
            info = _OLLAMA_SERVICE.show_model(MODEL_NAME, timeout=10)
            if hasattr(info, "model_dump"):
                info = info.model_dump()
            model_info = (
                info.get("modelinfo", {}) if isinstance(info, dict)
                else getattr(info, "modelinfo", None) or {}
            )
            family = model_info.get("general.architecture", "unknown")
            params = model_info.get("general.parameter_count", "unknown")
            _console.print()
            print_info(f"Model · {MODEL_NAME}")
            print_info(f"Family · {family}")
            print_info(f"Params · {params}")
            _console.print()
        except OllamaRuntimeError:
            _console.print()
            print_info(f"Model · {MODEL_NAME}")
            _console.print()
        return

    if sub == "profile":
        try:
            runtime = get_runtime_config(session)
        except RuntimeConfigurationError as exc:
            print_error(f"Invalid session configuration · {exc}")
            _console.print()
            return
        print_info(
            f"Profile · {runtime.profile.value}"
            + (
                f"  (requested {runtime.requested_profile.value})"
                if runtime.requested_profile.value != runtime.profile.value
                else ""
            )
        )
        print_info(runtime.selection_reason)
        print_info(
            f"ctx {runtime.num_ctx}  ·  out {runtime.num_predict}  ·  "
            f"batch {runtime.num_batch}  ·  temp {runtime.temperature}"
        )
        _print_profile_catalog()
        return

    print_error(f"Unknown /show subcommand · {sub}", detail="parameters · system · model · profile")
    _console.print()


def _list_saved_sessions() -> list[str]:
    """Return a sorted list of session file paths (newest first).
    
    Returns:
        list[str]: A list of absolute file paths to saved sessions.
    """
    directories = [_SESSIONS_DIR]
    if os.path.abspath(_LEGACY_SESSIONS_DIR) != os.path.abspath(_SESSIONS_DIR):
        directories.append(_LEGACY_SESSIONS_DIR)
    files: list[str] = []
    seen_names: set[str] = set()
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for filepath in glob.glob(os.path.join(directory, "*.json")):
            name = os.path.basename(filepath).casefold()
            if name in seen_names:
                continue
            seen_names.add(name)
            files.append(filepath)
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def list_session_catalog(*, limit: int = 80) -> list[dict[str, str]]:
    """Return saved conversation rows for menus (newest first).

    Each entry::
        {"path": abs_path, "title": display_title, "detail": "date · N msgs"}
    """
    rows: list[dict[str, str]] = []
    for filepath in _list_saved_sessions():
        if limit > 0 and len(rows) >= limit:
            break
        title = os.path.basename(filepath).replace(".json", "")
        # Drop trailing machine suffixes when present (timestamp / uuid).
        # Keep a readable stem for the menu.
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            when = mtime.strftime("%Y-%m-%d %H:%M")
        except OSError:
            when = "?"
        msg_count = 0
        try:
            with open(filepath, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            history = data.get("history", [])
            if isinstance(history, list):
                msg_count = sum(
                    1 for message in history if isinstance(message, dict) and message.get("role") == "user"
                )
        except Exception:
            msg_count = 0
        label = "msg" if msg_count == 1 else "msgs"
        rows.append(
            {
                "path": filepath,
                "title": title,
                "detail": f"{when} · {msg_count} {label}",
            }
        )
    return rows


def apply_saved_session_file(
    filepath: str,
    session: dict,
    history: list[dict],
) -> tuple[str, int, tuple[str, ...]]:
    """Load a saved session JSON into ``session`` / ``history`` in place.

    Returns:
        (display_name, user_message_count, warnings)

    Raises:
        OSError, ValueError, json.JSONDecodeError, RuntimeConfigurationError
    """
    path = os.path.abspath(str(filepath or "").strip())
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Session file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    saved_history = data.get("history", [])
    saved_session = data.get("session", {})
    if (
        not isinstance(saved_history, list)
        or any(not isinstance(message, dict) for message in saved_history)
        or not isinstance(saved_session, dict)
    ):
        raise ValueError("invalid session structure")

    restored_options, warnings = validate_session_options(saved_session.get("options", {}))
    runtime_profile = saved_session.get("runtime_profile", "manual")
    restored_runtime = get_runtime_config(
        {
            "runtime_profile": runtime_profile,
            "options": restored_options,
        }
    )
    warnings = tuple(dict.fromkeys((*warnings, *restored_runtime.warnings)))

    # Validate before replacing live state so a malformed restore cannot erase
    # the current conversation.
    history.clear()
    history.extend(saved_history)
    session["options"] = restored_options
    session["verbose"] = saved_session.get("verbose", True)
    session["wordwrap"] = saved_session.get("wordwrap", True)
    session["system"] = saved_session.get("system", "")
    session["history"] = saved_session.get("history", True)
    session["format"] = saved_session.get("format", "")
    session["think"] = saved_session.get("think", True)
    session["runtime_profile"] = runtime_profile
    session["tui_theme"] = saved_session.get("tui_theme", "oslo")

    display_name = os.path.basename(path).replace(".json", "")
    msg_count = sum(1 for message in history if message.get("role") == "user")
    return display_name, msg_count, warnings


def start_new_conversation(session: dict, history: list[dict]) -> None:
    """Clear history and system override for a fresh conversation."""
    history.clear()
    session["system"] = ""


def _handle_save(args: str, session: dict, history: list[dict]) -> None:
    """Handle /save [name] — persist current session to a JSON file.
    
    Args:
        args (str): Optional name to use for the saved file.
        session (dict): The active session data to save.
        history (list[dict]): The conversation history to save.
    """
    os.makedirs(_SESSIONS_DIR, exist_ok=True)

    name = args.strip()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if name:
        # Sanitize the name: replace spaces with underscores, strip non-alphanumerics
        safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
        filename = f"{safe_name}_{timestamp}.json"
    else:
        filename = f"session_{timestamp}.json"

    filepath = os.path.join(_SESSIONS_DIR, filename)

    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "session": {
            "options": session.get("options", {}),
            "verbose": session.get("verbose", True),
            "wordwrap": session.get("wordwrap", True),
            "system": session.get("system", ""),
            "history": session.get("history", True),
            "format": session.get("format", ""),
            "think": session.get("think", True),
            "runtime_profile": session.get("runtime_profile", "manual"),
            "tui_theme": session.get("tui_theme", "oslo"),
        },
        "history": history,
    }

    try:
        atomic_write_json(filepath, payload, durable=True)
        display_name = os.path.basename(filepath)
        msg_count = sum(1 for m in history if m.get("role") == "user")
        print_ok(
            f"Session saved · {display_name}",
            detail=f"{msg_count} user message{'s' if msg_count != 1 else ''}",
        )
        _console.print()
    except OSError as exc:
        _console.print(f"[red]Failed to save session: {exc}[/]\n")


def _handle_load(args: str, session: dict, history: list[dict]) -> None:
    """Handle /load [name|index] — load a previously saved session.
    
    Replaces the current session state and history with the loaded data.
    
    Args:
        args (str): The index or partial name of the session to load.
        session (dict): The active session data dict to update.
        history (list[dict]): The conversation history list to update.
    """
    saved = _list_saved_sessions()
    arg = args.strip()

    # No argument: list available sessions
    if not arg:
        catalog = list_session_catalog()
        if not catalog:
            print_info("No saved sessions found")
            _console.print()
            return
        _console.print()
        print_info("Saved sessions")
        for i, row in enumerate(catalog, 1):
            _console.print(
                f"    [green]{i}.[/] {row['title']}  [dim]({row['detail']})[/]"
            )
        print_info("Use /load <number> or /load <name> to restore")
        _console.print()
        return

    # Try as index first
    target_path: str | None = None
    try:
        idx = int(arg)
        if 1 <= idx <= len(saved):
            target_path = saved[idx - 1]
    except ValueError:
        pass

    # Try as name substring match
    if target_path is None:
        matches = [fp for fp in saved if arg.lower() in os.path.basename(fp).lower()]
        if len(matches) == 1:
            target_path = matches[0]
        elif len(matches) > 1:
            _console.print(f"[yellow]Multiple sessions match '{arg}':[/]")
            for fp in matches:
                _console.print(f"  [dim]{os.path.basename(fp)}[/]")
            _console.print(f"[dim]  Be more specific or use /load to list with indices.[/]\n")
            return

    if target_path is None:
        _console.print(f"[red]No session found matching '{arg}'.[/]  [dim](type /load to list available sessions)[/]\n")
        return

    try:
        display_name, msg_count, warnings = apply_saved_session_file(
            target_path, session, history
        )
    except RuntimeConfigurationError as exc:
        _console.print(f"[red]Failed to load session: invalid settings: {exc}[/]\n")
        return
    except (OSError, json.JSONDecodeError, ValueError, FileNotFoundError) as exc:
        _console.print(f"[red]Failed to load session: {exc}[/]\n")
        return

    print_ok(
        f"Session loaded · {display_name}",
        detail=f"{msg_count} user message{'s' if msg_count != 1 else ''}",
    )
    for warning in warnings:
        print_warn(warning)
    _console.print()


def _extract_option(tokens: list[str], names: tuple[str, ...], default: str | None = None) -> str | None:
    """Remove and return a string option from a shlex token list.
    
    Scans for exact matches (e.g. `--collection value`) or inline matches
    (e.g. `--collection=value`). Mutates the tokens list.
    
    Args:
        tokens (list[str]): The list of parsed argument strings.
        names (tuple[str, ...]): The names/aliases of the option to find.
        default (str | None): The value to return if not found.
        
    Returns:
        str | None: The extracted value, or the default.
    """
    index = 0
    while index < len(tokens):
        token = tokens[index]
        for name in names:
            if token == name:
                if index + 1 >= len(tokens):
                    return default
                value = tokens[index + 1]
                del tokens[index:index + 2]
                return value
            if token.startswith(f"{name}="):
                value = token.split("=", 1)[1]
                del tokens[index]
                return value
        index += 1
    return default


def _extract_flag(tokens: list[str], names: tuple[str, ...]) -> bool:
    """Remove and return whether any boolean flag exists in a shlex token list.
    
    Args:
        tokens (list[str]): The list of parsed argument strings.
        names (tuple[str, ...]): Flag names to check for.
        
    Returns:
        bool: True if flag was present and removed, False otherwise.
    """
    for index, token in enumerate(tokens):
        if token in names:
            del tokens[index]
            return True
    return False


def _call_tool_json(tool_name: str, **kwargs) -> dict:
    """Invoke a CLI helper through the shared tool safety contract.
    
    Args:
        tool_name (str): Tool function name to execute.
        **kwargs: Arguments to pass to the tool.
        
    Returns:
        dict: The tool result parsed as JSON, or an error dict.
    """
    spec = normalize_tool_calls([{
        "function": {"name": tool_name, "arguments": kwargs},
    }])[0]
    execution = execute_tool_call(spec)
    try:
        parsed = json.loads(execution.content)
    except json.JSONDecodeError:
        return {"result": execution.content, "ok": execution.ok}
    return parsed if isinstance(parsed, dict) else {"result": parsed, "ok": execution.ok}


def _format_match_snippet(text: str | None, max_chars: int = 260) -> str:
    """Format search match text into a single-line abbreviated snippet.
    
    Args:
        text (str | None): Raw text to format.
        max_chars (int): Max allowed character length.
        
    Returns:
        str: A neat, collapsed snippet.
    """
    snippet = re.sub(r"\s+", " ", (text or "")).strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars - 3].rstrip() + "..."
    return snippet


def _handle_vault(args: str) -> None:
    """Handle /vault sub-commands.
    
    Interacts directly with the vault tools for querying, indexing, and deleting
    local document context.
    
    Args:
        args (str): The raw string arguments passed to the vault command.
    """
    try:
        parts = shlex.split(args)
    except ValueError as exc:
        _console.print(f"[red]Invalid /vault command: {exc}[/]\n")
        return

    if not parts or parts[0].lower() in ("help", "-h", "--help"):
        print_command_help(_VAULT_HELP_ENTRIES, title="vault", subtitle="local knowledge index")
        return

    sub = parts[0].lower()
    tokens = parts[1:]
    collection_option = _extract_option(tokens, ("--collection", "-c"), None)
    collection_raw = collection_option or "vault"
    
    from tools.vault_indexer import resolve_vault_alias
    collection = resolve_vault_alias(collection_raw)

    if sub in ("list", "ls"):
        data = _call_tool_json("list_vaults")
        if "error" in data:
            _console.print(f"[red]Vault list failed: {data['error']}[/]\n")
            return

        vaults = data.get("vaults", [])
        if not vaults:
            _console.print(f"[dim]  No indexed vault collections found.[/]\n")
            return

        _console.print()
        print_info("Indexed vaults")
        for vault in vaults:
            name = vault.get("collection", "unknown")
            chunk_count = vault.get("indexed_chunks")
            if isinstance(chunk_count, int):
                count_text = f"{chunk_count} chunk{'s' if chunk_count != 1 else ''}"
            else:
                count_text = "chunk count unavailable"
            _console.print(f"    [green]{name}[/]  [dim]({count_text})[/]")
        _console.print()
        return

    if sub in ("alias", "register"):
        if len(tokens) < 2:
            _console.print(f"[red]Usage: /vault alias <human-name> <collection-name>[/]\n")
            return
        alias_name = tokens[0]
        coll_name = tokens[1]
        from tools.vault_indexer import register_vault_alias
        register_vault_alias(alias_name, coll_name)
        print_ok(f"Vault alias registered · {alias_name} → {coll_name}")
        _console.print()
        return

    if sub in ("aliases", "list-aliases"):
        from tools.vault_indexer import list_vault_aliases
        try:
            import json as _json
            data = _json.loads(list_vault_aliases())
            aliases = data.get("aliases", [])
            if not aliases:
                _console.print(f"[dim]  No vault aliases registered.[/]\n")
                return
            _console.print()
            print_info("Vault aliases")
            for entry in aliases:
                _console.print(f"    [green]{entry['alias']}[/] → [dim]{entry['collection']}[/]")
            _console.print()
        except Exception as e:
            _console.print(f"[red]Failed to list aliases: {e}[/]\n")
        return

    if sub in ("rename", "mv"):
        if len(tokens) < 2:
            _console.print(f"[red]Usage: /vault rename <old-name> <new-name>[/]\n")
            return
        old_name = tokens[0]
        new_name = tokens[1]
        from tools.vault_indexer import rename_vault
        try:
            import json as _json
            data = _json.loads(rename_vault(old_name, new_name))
            if data.get("error"):
                _console.print(f"[red]✗  {data['error']}[/]\n")
            else:
                print_ok(
                    f"Vault renamed · {data['old_collection']} → {data['new_collection']}",
                    detail=f"{data.get('chunks_moved', 0)} chunks moved",
                )
                if data.get("updated_aliases"):
                    print_info(f"Updated aliases: {', '.join(data['updated_aliases'])}")
                _console.print()
        except Exception as e:
            _console.print(f"[red]Failed to rename vault: {e}[/]\n")
        return

    if sub in ("add", "index"):
        if not tokens:
            _console.print(f"[red]Usage: /vault add <file-or-folder> [--collection name][/]\n")
            return

        vision_mode = _extract_option(tokens, ("--vision",), "auto") or "auto"
        max_pages_raw = _extract_option(tokens, ("--max-pages",), "20") or "20"
        try:
            max_pages = int(max_pages_raw)
        except ValueError:
            _console.print(f"[red]Invalid --max-pages value: {max_pages_raw}[/]\n")
            return
        target = " ".join(tokens)
        if not os.path.exists(target):
            _console.print(f"[red]Vault path not found: {target}[/]\n")
            return

        print_lab_status(f"Indexing vault content", kind="run", detail=target)
        if os.path.isdir(target):
            index_args = {
                "vault_path": target,
                "vision_mode": vision_mode,
                "max_pages": max_pages,
            }
        else:
            index_args = {
                "vault_path": os.path.dirname(target) or ".",
                "file_path": target,
                "vision_mode": vision_mode,
                "max_pages": max_pages,
            }
        if collection_option:
            index_args["collection"] = collection
        data = _call_tool_json("index_vault", **index_args)

        if "error" in data:
            _console.print(f"[red]Vault add failed: {data['error']}[/]\n")
            return

        indexed_files = data.get("indexed_files", 0)
        indexed_chunks = data.get("indexed_chunks", 0)
        skipped_count = data.get("skipped_count", 0)
        incomplete_count = data.get("incomplete_pdf_count", 0)
        status_label = "Index checkpoint saved" if incomplete_count else "Vault indexed"
        print_ok(
            f"{status_label} · {indexed_files} file{'s' if indexed_files != 1 else ''}, "
            f"{indexed_chunks} chunk{'s' if indexed_chunks != 1 else ''}",
            detail=f"collection: {data.get('collection', collection)}",
        )
        if skipped_count:
            print_warn(f"Skipped {skipped_count} file{'s' if skipped_count != 1 else ''}")
        for job in data.get("pdf_jobs", []):
            print_info(
                f"{job.get('source')}: pages {job.get('indexed_pages')}/{job.get('page_count')}"
                f" · vision {job.get('vision_pages')} · next {job.get('next_page')}"
            )
        _console.print()
        return

    if sub == "status":
        target = " ".join(tokens).strip()
        if not target:
            _console.print("[red]Usage: /vault status <pdf-path> [--collection name][/]\n")
            return
        status_args = {
            "vault_path": os.path.dirname(target) or ".",
            "file_path": target,
            "action": "status",
        }
        if collection_option:
            status_args["collection"] = collection
        data = _call_tool_json("index_vault", **status_args)
        if data.get("error"):
            _console.print(f"[red]Vault status failed: {data['error']}[/]\n")
            return
        for job in data.get("jobs", []):
            _console.print(
                f"[cyan]{job.get('source', os.path.basename(target))}[/]  "
                f"pages {job.get('indexed_pages', 0)}/{job.get('page_count', '?')}  "
                f"chunks {job.get('indexed_chunks', 0)}  "
                f"[dim]{'complete' if job.get('complete') else 'next page ' + str(job.get('next_page', 1))}[/]"
            )
        return

    if sub == "read":
        cursor = _extract_option(tokens, ("--cursor",), "0") or "0"
        source = _extract_option(tokens, ("--source", "-s"), None)
        max_chars_raw = _extract_option(tokens, ("--max-chars",), "2800") or "2800"
        try:
            max_chars = int(max_chars_raw)
        except ValueError:
            _console.print(f"[red]Invalid --max-chars value: {max_chars_raw}[/]\n")
            return
        data = _call_tool_json(
            "vault_read", collection=collection, cursor=cursor,
            source=source, max_chars=max_chars,
        )
        if data.get("error"):
            _console.print(f"[red]Vault read failed: {data['error']}[/]\n")
            return
        _console.print(data.get("content", ""), markup=False)
        if data.get("next_cursor") is not None:
            _console.print(f"[dim]Next cursor: {data['next_cursor']}[/]\n")
        return

    if sub in ("search", "find"):
        top_k_raw = _extract_option(tokens, ("--top-k", "-k"), None)
        source = _extract_option(tokens, ("--source", "-s"), None)
        query = " ".join(tokens).strip()
        if not query:
            _console.print(f"[red]Usage: /vault search <query> [--collection name] [--top-k n] [--source path][/]\n")
            return
        try:
            top_k = int(top_k_raw) if top_k_raw is not None else 6
        except ValueError:
            _console.print(f"[red]Invalid --top-k value: {top_k_raw}[/]\n")
            return

        print_lab_status("Searching vault", kind="run", detail=query)
        data = _call_tool_json("vault_search", query=query, collection=collection, top_k=top_k, source=source)

        if "error" in data:
            _console.print(f"[red]Vault search failed: {data['error']}[/]\n")
            return

        matches = data.get("matches", [])
        print_info(
            f"Vault search · {len(matches)} match{'es' if len(matches) != 1 else ''}",
            detail=f"collection: {data.get('collection', collection)}",
        )
        for match in matches:
            source_name = match.get("source") or match.get("filename") or "unknown"
            chunk = match.get("chunk_index", "?")
            distance = match.get("distance")
            distance_text = f" · distance {distance:.4f}" if isinstance(distance, (int, float)) else ""
            _console.print(f"  [green]{match.get('rank', '?')}.[/] {source_name}  [dim](chunk {chunk}{distance_text})[/]")
            snippet = _format_match_snippet(match.get("text"))
            if snippet:
                _console.print(f"     [dim]{snippet}[/]")
        _console.print()
        return

    if sub in ("delete", "remove", "rm"):
        delete_collection = _extract_flag(tokens, ("--all", "--collection-all"))
        source = " ".join(tokens).strip()
        if not delete_collection and not source:
            _console.print(f"[red]Usage: /vault delete <source> [--collection name][/]\n")
            return

        print_lab_status("Deleting vault index entries…", kind="run")
        data = _call_tool_json(
            "delete_vault_item",
            source=source or None,
            collection=collection,
            delete_collection=delete_collection,
        )

        if "error" in data:
            _console.print(f"[red]Vault delete failed: {data['error']}[/]\n")
            return

        if data.get("deleted_collection"):
            print_ok(f"Vault collection deleted · {collection}")
            _console.print()
            return

        deleted_chunks = data.get("deleted_chunks", 0)
        if deleted_chunks:
            print_ok(
                f"Vault entries deleted · {deleted_chunks} chunk{'s' if deleted_chunks != 1 else ''}",
                detail=f"collection: {data.get('collection', collection)}",
            )
            _console.print()
        else:
            print_warn("No indexed chunks matched", detail=str(source))
            _console.print()
        return

    _console.print(f"[red]Unknown /vault subcommand: {sub}[/]  [dim](try: list, aliases, rename, add, search, delete)[/]\n")


def _handle_command(cmd: str, session: dict, history: list[dict]) -> bool | None:
    """Handle a slash command by delegating to specific sub-handlers. 
    
    Args:
        cmd (str): The full command string input by the user.
        session (dict): The current application state and configuration.
        history (list[dict]): The conversation history.
        
    Returns:
        bool | None: True if handled and should continue, None to quit.
    """
    parts = cmd.strip().split(None, 1)
    base = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if base in ("/quit", "/exit", "/q"):
        return None  # Signal to quit

    if base in ("/help", "/?"):
        print_command_help(_COMMAND_HELP_ENTRIES, title="commands")
        return True

    if base == "/clear":
        history.clear()
        # Also clear the custom system prompt override
        session["system"] = ""
        print_ok("Conversation history and system prompt cleared")
        _console.print()
        return True

    if base == "/speech":
        _handle_speech(rest, session)
        return True

    if base == "/theme":
        _handle_theme(rest, session)
        return True

    if base == "/set":
        _handle_set(rest, session, history)
        return True

    if base == "/profile":
        # First-class profile command: bare → show catalog + current; with arg → set.
        if not rest.strip():
            try:
                runtime = get_runtime_config(session)
                print_info(
                    f"Current · {runtime.profile.value}"
                    + (
                        f"  (requested {runtime.requested_profile.value})"
                        if runtime.requested_profile.value != runtime.profile.value
                        else ""
                    )
                )
                print_info(runtime.selection_reason)
            except RuntimeConfigurationError as exc:
                print_error(f"Invalid session configuration · {exc}")
            _print_profile_catalog()
        else:
            _apply_runtime_profile(session, rest)
        return True

    if base == "/show":
        _handle_show(rest, session, history)
        return True

    if base == "/save":
        _handle_save(rest, session, history)
        return True

    if base == "/load":
        _handle_load(rest, session, history)
        return True

    if base == "/vault":
        _handle_vault(rest)
        return True

    # Unknown command
    _console.print(f"[red]Unknown command: {base}[/]  [dim](type /help for available commands)[/]\n")
    return True


# ── Main loop ─────────────────────────────────────────────────────────


def _new_session_state() -> dict:
    """Fresh session options for a CLI/TUI conversation."""
    return {
        "options": {},
        "verbose": True,
        "wordwrap": True,
        "system": "",
        "history": True,
        "format": "",
        "think": True,
        "runtime_profile": "manual",
        "tui_theme": "oslo",
    }


def _boot_status_meta() -> dict[str, str]:
    """Runtime chips for the TUI status bar / classic splash."""
    meta: dict[str, str] = {}
    try:
        from agent.runtime_config import get_runtime_config
        from agent.platform_runtime import platform_family

        runtime = get_runtime_config()
        meta = {
            "profile": runtime.profile.value,
            "model": runtime.chat_model,
            "ctx": str(runtime.num_ctx),
            "out": str(runtime.num_predict),
            "host": platform_family(),
        }
    except Exception:
        pass
    return meta


def process_user_turn(
    user_input: str,
    session: dict,
    history: list[dict],
    default_system_prompt: str | None = None,
) -> None:
    """Run one user→assistant turn (shared by classic CLI and the TUI).

    Handles system-prompt sync, optional vault auto-index, streaming generation,
    tool-call rounds, and background history compaction.
    """
    if default_system_prompt is None:
        default_system_prompt = load_default_system_prompt()

    # ── Sync system prompt ────────────────────────────────────────
    active_system = session.get("system") or default_system_prompt
    if active_system:
        if (
            not history
            or history[0].get("role") != "system"
            or history[0].get("content") != active_system
        ):
            history[:] = [m for m in history if m.get("role") != "system"]
            history.insert(0, {"role": "system", "content": active_system})
    else:
        history[:] = [m for m in history if m.get("role") != "system"]

    # ── Auto-index large or binary files when user inputs a local file path ─
    pre_tool_message = None
    try:
        if os.path.exists(user_input) and os.path.isfile(user_input):
            size = os.path.getsize(user_input)
            ext = os.path.splitext(user_input)[1].lower()
            INDEX_THRESHOLD = 200_000
            if size > INDEX_THRESHOLD or ext in (".pdf", ".docx"):
                print_lab_status(
                    "Large/binary file detected — indexing",
                    kind="run",
                    detail=user_input,
                )
                if "index_vault" in TOOL_DISPATCH:
                    try:
                        execution = execute_tool_calls([{
                            "function": {
                                "name": "index_vault",
                                "arguments": {
                                    "vault_path": os.path.dirname(user_input) or ".",
                                    "file_path": user_input,
                                },
                            }
                        }])[0]
                        tool_content = execution.content
                        tool_msg = {
                            "role": "tool",
                            "tool_name": "index_vault",
                            "name": "index_vault",
                            "content": tool_content,
                        }
                        if session["history"]:
                            history.append(tool_msg)
                        else:
                            pre_tool_message = tool_msg
                        if execution.ok:
                            try:
                                index_payload = json.loads(tool_content)
                            except (TypeError, json.JSONDecodeError):
                                index_payload = {}
                            if index_payload.get("incomplete_pdf_count"):
                                job = (index_payload.get("pdf_jobs") or [{}])[0]
                                print_warn(
                                    f"Index checkpoint saved — pages "
                                    f"{job.get('indexed_pages', '?')}/{job.get('page_count', '?')}; "
                                    "resume required."
                                )
                            else:
                                print_ok("Indexing complete")
                        else:
                            print_warn(
                                "Indexing did not complete; the model will receive the error."
                            )
                    except Exception as e:
                        print_error(f"Indexing failed: {e}")
    except Exception:
        pass

    # ── Build messages to send ────────────────────────────────────
    if session["history"]:
        history.append({"role": "user", "content": user_input})
        messages_to_send = prepare_messages_for_model(history, session, tools=TOOL_SCHEMAS)
    else:
        messages_to_send = []
        if history and history[0].get("role") == "system":
            messages_to_send.append(history[0])
        if pre_tool_message:
            messages_to_send.append(pre_tool_message)
        messages_to_send.append({"role": "user", "content": user_input})
        messages_to_send = prepare_messages_for_model(
            messages_to_send, session, tools=TOOL_SCHEMAS
        )

    # ── LLM call with streaming + thinking ────────────────────────
    assistant_msg = _stream_complete_response(
        model=MODEL_NAME,
        messages=messages_to_send,
        session=session,
        user_input=user_input,
        tools=TOOL_SCHEMAS,
    )

    if session["history"]:
        history.append(assistant_msg)

    # ── Tool-call loop (iterative, in case of chained calls) ──────
    executed_tool_calls: dict[str, dict] = {}
    tool_rounds = 0
    while assistant_msg.get("tool_calls"):
        if tool_rounds >= MAX_TOOL_CALL_ROUNDS:
            message = (
                f"Stopped after {MAX_TOOL_CALL_ROUNDS} tool-call rounds to avoid an unreliable loop. "
                "Please narrow the request or ask me to continue from the latest result."
            )
            print_warn(message)
            if not display_is_tui():
                _console.print()
            if session["history"]:
                history.append({"role": "assistant", "content": message})
            break
        tool_rounds += 1
        tool_results = _process_tool_calls_with_turn_guard(
            assistant_msg["tool_calls"], executed_tool_calls
        )

        if session["history"]:
            history.extend(tool_results)
            reminder = {
                "role": "user",
                "content": TOOL_CONTINUATION_PROMPT.format(user_input=user_input),
            }
            messages_to_send = prepare_messages_for_model(
                [*history, reminder],
                session,
                tools=TOOL_SCHEMAS,
                extra_reserved_tokens=CONTEXT_TOOL_LOOP_RESERVE,
            )
        else:
            messages_to_send.append(assistant_msg)
            messages_to_send.extend(tool_results)
            messages_to_send.append({
                "role": "user",
                "content": TOOL_CONTINUATION_PROMPT.format(user_input=user_input),
            })
            messages_to_send = prepare_messages_for_model(
                messages_to_send,
                session,
                tools=TOOL_SCHEMAS,
                extra_reserved_tokens=CONTEXT_TOOL_LOOP_RESERVE,
            )

        assistant_msg = _stream_complete_response(
            model=MODEL_NAME,
            messages=messages_to_send,
            session=session,
            user_input=user_input,
            tools=TOOL_SCHEMAS,
            extra_reserved_tokens=CONTEXT_TOOL_LOOP_RESERVE,
        )
        if session["history"]:
            history.append(assistant_msg)

    if session["history"]:
        _check_and_compact_history(history, session)


def _should_use_tui() -> bool:
    """Prefer the full-screen TUI on interactive TTYs when Textual is available."""
    if "--classic" in sys.argv or "--classic-cli" in sys.argv:
        return False
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True


def run_classic() -> None:
    """Scrollback-friendly classic CLI loop (no alternate screen)."""
    default_system_prompt = load_default_system_prompt()
    history: list[dict] = []
    session = _new_session_state()

    meta = _boot_status_meta()
    try:
        print_welcome_header(
            {
                "profile": meta.get("profile", ""),
                "model": meta.get("model", ""),
                "num_ctx": meta.get("ctx", ""),
                "num_predict": meta.get("out", ""),
                "platform": meta.get("host", ""),
            }
            if meta
            else None
        )
    except Exception:
        print_welcome_header()

    while True:
        try:
            user_input = read_user_input(
                completions=CLI_SLASH_COMPLETIONS,
                descriptions=CLI_SLASH_DESCRIPTIONS,
            )
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            result = _handle_command(user_input, session, history)
            if result is None:
                break
            continue

        process_user_turn(user_input, session, history, default_system_prompt)


def run() -> None:
    """Run the interactive agent — full-screen TUI by default, classic fallback.

    Use ``--classic`` / ``--classic-cli`` to force the legacy scrollback prompt.
    """
    default_system_prompt = load_default_system_prompt()
    history: list[dict] = []
    session = _new_session_state()
    meta = _boot_status_meta()

    if _should_use_tui():
        from agent.tui import run_tui

        run_tui(
            session=session,
            history=history,
            default_system_prompt=default_system_prompt,
            process_turn=process_user_turn,
            handle_command=_handle_command,
            slash_completions=CLI_SLASH_COMPLETIONS,
            slash_descriptions=CLI_SLASH_DESCRIPTIONS,
            status_meta=meta,
        )
        return

    # Classic path (or TUI unavailable).
    if not any(flag in sys.argv for flag in ("--classic", "--classic-cli")):
        # Non-TTY or missing Textual — still try to be helpful.
        try:
            import textual  # noqa: F401
        except ImportError:
            print_warn(
                "Full TUI unavailable (install textual: pip install 'textual>=1.0.0'). "
                "Using classic CLI."
            )

    # Re-enter classic with the same session objects for a clean start.
    run_classic()
