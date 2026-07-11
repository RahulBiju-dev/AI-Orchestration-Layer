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
    _print_status,
    _Spinner,
    assistant_stream_panel,
    flush_terminal_input,
    print_assistant_message,
    print_lab_status,
    print_thinking_footer,
    print_thinking_header,
    print_tool_event,
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
SYSTEM_PERSISTENCE_REMINDER = (
    "Runtime system reminder: Follow the active system prompt exactly. "
    "Use tools over memory for verifiable state, preserve the vault > web > internal knowledge hierarchy, "
    "do not invent facts or tool results, and answer the latest user request directly."
)
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

    recent_user_text = next(
        (
            str(message.get("content", ""))
            for message in reversed(messages)
            if isinstance(message, dict) and message.get("role") == "user"
        ),
        "",
    ).casefold()
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

def _sigquit_handler(signum, frame):
    """Handle SIGQUIT signal to interrupt the current LLM generation stream.
    
    Sets the global _interrupted flag to True, which is checked during
    token streaming to gracefully abort generation without crashing the agent.
    
    Args:
        signum: The signal number.
        frame: The current stack frame.
    """
    global _interrupted
    # Mark as interrupted so the streaming loop can break
    _interrupted = True


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


def _insert_system_persistence_reminder(messages: list[dict]) -> list[dict]:
    """Keep a compact system-policy anchor near the active turn in long prompts."""
    system_msgs, conv_msgs = _split_system_and_conversation(messages)
    if not system_msgs or not conv_msgs:
        return messages
    if _is_tool_continuation_prompt(conv_msgs[-1]) or _is_output_continuation_prompt(conv_msgs[-1]):
        return [*system_msgs, *conv_msgs]

    reminder = {"role": "system", "content": SYSTEM_PERSISTENCE_REMINDER}
    insert_at = len(conv_msgs)
    for index in range(len(conv_msgs) - 1, -1, -1):
        if conv_msgs[index].get("role") == "user":
            insert_at = index
            break
    return [*system_msgs, *conv_msgs[:insert_at], reminder, *conv_msgs[insert_at:]]


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

    trimmed = _trim_history(
        messages,
        num_ctx,
        reserved_tokens=reserved,
        tool_schema_tokens=tool_tokens,
    )
    prepared = _insert_system_persistence_reminder(trimmed)

    # If the tail reminder pushes the prompt too close to the edge, trim once
    # more with the reminder overhead included, then reinsert it.
    projected = _estimate_messages_tokens(prepared) + tool_tokens + reserved + _context_safety_margin(num_ctx)
    if projected > num_ctx:
        reminder_tokens = _estimate_message_tokens({"role": "system", "content": SYSTEM_PERSISTENCE_REMINDER})
        trimmed = _trim_history(
            messages,
            num_ctx,
            reserved_tokens=reserved + reminder_tokens,
            tool_schema_tokens=tool_tokens,
        )
        prepared = _insert_system_persistence_reminder(trimmed)
    return prepared


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
        _console.print(f"\n[yellow]⚠ {message}[/]\n")
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
        _console.print(f"\n[red]⚠ {message}[/]\n")
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
                shortcut = "Ctrl+\\" if hasattr(signal, "SIGQUIT") else "Ctrl+Break"
                _console.print(f"\n[yellow]⚠ Generation interrupted by user ({shortcut}).[/]\n")
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
                    from rich.markup import escape
                    _console.print(escape(str(thinking_chunk)), style=thinking_stream_style(), end="")
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
                        if not thinking_displayed:
                            _console.print()

                    content_buf += str(content_chunk)

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
                _console.print(f"\n[red]⚠ {message}[/]\n")
                if not content_buf:
                    content_buf = message
                break

    except OllamaRuntimeError as exc:
        spinner.stop()
        message = f"Ollama chat failed while streaming: {exc}"
        _console.print(f"\n[red]⚠ {message}[/]\n")
        if not content_buf:
            content_buf = message
    except Exception as exc:
        spinner.stop()
        message = f"Ollama stream ended with an unexpected error: {exc}"
        _console.print(f"\n[red]⚠ {message}[/]\n")
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
        # Print the final complete markdown to persistent scrollback after the
        # transient live panel is removed.
        print_assistant_message(content_buf)

    # Verbose stats
    if verbose:
        elapsed = time.monotonic() - t_start
        t_tokens = len(thinking_buf.split()) if thinking_buf else 0
        c_tokens = len(content_buf.split()) if content_buf else 0
        total = t_tokens + c_tokens
        tps = total / elapsed if elapsed > 0 else 0
        _console.print(
                f"[dim]  ⏱  {elapsed:.1f}s  ·  ~{total} tokens  ·  ~{tps:.1f} tok/s[/]\n",
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
        verbose=session.get("verbose", False),
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
            verbose=session.get("verbose", False),
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

CLI_SLASH_COMPLETIONS = (
    "/help",
    "/?",
    "/clear",
    "/save",
    "/load",
    "/set parameter",
    "/set profile",
    "/set system",
    "/set history",
    "/set nohistory",
    "/set wordwrap",
    "/set nowordwrap",
    "/set format",
    "/set noformat",
    "/set verbose",
    "/set quiet",
    "/set think",
    "/set nothink",
    "/show parameters",
    "/show system",
    "/show model",
    "/vault alias",
    "/vault aliases",
    "/vault rename",
    "/vault add",
    "/vault list",
    "/vault search",
    "/vault delete",
    "/quit",
    "/exit",
    "/q",
)

_COMMANDS_HELP = f"""
[cyan][bold]Available commands:[/]
  [green]/help[/]                          — Show this help message
  [green]/clear[/]                         — Clear conversation history
  [green]/save [name][/]                   — Save current session  [dim](optional name)[/]
  [green]/load [name|index][/]             — Load a saved session  [dim](lists sessions if no arg)[/]
  [green]/set parameter <name> <val>[/]    — Set a model parameter  [dim](e.g. temperature 0.7)[/]
  [green]/set profile <name>[/]             — Select auto, low-vram, balanced, or manual
  [green]/set system "<prompt>"[/]         — Set the system prompt for this session
  [green]/set history[/]                   — Enable conversation history  [dim](default)[/]
  [green]/set nohistory[/]                 — Disable history  [dim](each turn is standalone)[/]
  [green]/set wordwrap[/]                  — Enable word wrapping  [dim](default)[/]
  [green]/set nowordwrap[/]                — Disable word wrapping
  [green]/set format json[/]               — Force JSON output from the model
  [green]/set noformat[/]                  — Disable forced output format  [dim](default)[/]
  [green]/set verbose[/]                   — Show generation stats after each response
  [green]/set quiet[/]                     — Hide generation stats  [dim](default)[/]
  [green]/set think[/]                     — Enable model thinking/reasoning  [dim](default)[/]
  [green]/set nothink[/]                   — Disable model thinking
  [green]/show parameters[/]               — Show current session parameters
  [green]/show system[/]                   — Show the active system prompt
  [green]/show model[/]                    — Show model info
  [green]/vault alias <name> <coll>[/]     — Register a friendly alias for a collection
  [green]/vault aliases[/]                  — List registered vault aliases
  [green]/vault rename <old> <new>[/]       — Rename a vault collection
  [green]/vault add <path>[/]               — Add a file or folder to the searchable vault
  [green]/vault list[/]                     — List indexed vault collections
  [green]/vault search <query>[/]           — Search the indexed vault
  [green]/vault delete <source>[/]          — Delete indexed vault chunks by source/path
  [green]/quit[/]                          — Exit the agent  [dim](also /exit, /q)[/]
"""

_VAULT_HELP = f"""
[cyan][bold]Vault commands:[/]
  [green]/vault list[/]                                  — List indexed vault collections
  [green]/vault aliases[/]                               — List registered vault aliases
  [green]/vault alias <name> <coll>[/]                  — Register a friendly alias for a collection
  [green]/vault rename <old> <new>[/]                   — Rename a vault collection
  [green]/vault add <path> [--collection name][/]        — Index a file or folder
  [green]/vault search <query> [--top-k n][/]            — Search indexed content
  [green]/vault search <query> [--source path][/]        — Restrict search to a source
  [green]/vault delete <source> [--collection name][/]   — Remove indexed chunks
  [green]/vault delete --all [--collection name][/]      — Delete a collection
"""


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
        _console.print(f"[red]Usage: /set <subcommand> [args][/]  [dim](type /help for details)[/]\n")
        return

    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    # ── /set verbose / /set quiet ─────────────────────────────────────
    if sub == "verbose":
        session["verbose"] = True
        _console.print(f"[cyan][bold]✓  Verbose mode enabled — stats shown after each response.[/]\n")
        return
    if sub == "quiet":
        session["verbose"] = False
        _console.print(f"[cyan][bold]✓  Quiet mode enabled.[/]\n")
        return

    # ── /set wordwrap / /set nowordwrap ───────────────────────────────
    if sub == "wordwrap":
        session["wordwrap"] = True
        _console.print(f"[cyan][bold]✓  Word wrapping enabled.[/]\n")
        return
    if sub == "nowordwrap":
        session["wordwrap"] = False
        _console.print(f"[cyan][bold]✓  Word wrapping disabled.[/]\n")
        return

    # ── /set history / /set nohistory ─────────────────────────────────
    if sub == "history":
        session["history"] = True
        _console.print(f"[cyan][bold]✓  Conversation history enabled.[/]\n")
        return
    if sub == "nohistory":
        session["history"] = False
        _console.print(f"[cyan][bold]✓  History disabled — each turn is now standalone.[/]\n")
        return

    # ── /set format json / /set noformat ──────────────────────────────
    if sub == "format":
        fmt = rest.strip().lower()
        if fmt == "json":
            session["format"] = "json"
            _console.print(f"[cyan][bold]✓  JSON output mode enabled.[/]\n")
        else:
            _console.print(f"[red]Unsupported format: {fmt}[/]  [dim](supported: json)[/]\n")
        return
    if sub == "noformat":
        session["format"] = ""
        _console.print(f"[cyan][bold]✓  Output formatting reset to default.[/]\n")
        return

    # ── /set think / /set nothink ─────────────────────────────────────
    if sub == "think":
        session["think"] = True
        _console.print(f"[cyan][bold]✓  Thinking/reasoning enabled.[/]\n")
        return
    if sub == "nothink":
        session["think"] = False
        _console.print(f"[cyan][bold]✓  Thinking disabled — model will respond directly.[/]\n")
        return

    # ── /set system "<prompt>" ────────────────────────────────────────
    if sub == "system":
        # Strip surrounding quotes if present
        prompt = rest.strip().strip('"').strip("'")
        
        # Remove any existing system messages from history to avoid duplicates
        history[:] = [m for m in history if m.get("role") != "system"]

        if not prompt or prompt.lower() == "default":
            session["system"] = ""
            _console.print(f"[cyan][bold]✓  System prompt reset to default.[/]\n")
            return

        # Insert new system message at the start
        history.insert(0, {"role": "system", "content": prompt})
        session["system"] = prompt
        
        # Truncate display for confirmation
        display = prompt if len(prompt) <= 80 else prompt[:77] + "…"
        _console.print(f"[cyan][bold]✓  System prompt set:[/] [dim]{display}[/]\n")
        return

    if sub == "profile":
        profile = rest.strip().lower().replace("_", "-")
        candidate = dict(session)
        candidate["runtime_profile"] = profile
        try:
            runtime = get_runtime_config(candidate)
        except RuntimeConfigurationError as exc:
            _console.print(f"[red]Invalid runtime profile: {exc}[/]\n")
            return
        session["runtime_profile"] = profile
        _console.print(f"[cyan][bold]✓  Runtime profile = {runtime.profile.value}[/]")
        _console.print(f"[dim]{runtime.selection_reason}[/]")
        for warning in runtime.warnings:
            _console.print(f"[yellow]⚠ {warning}[/]")
        _console.print()
        return

    # ── /set parameter <name> <value> ─────────────────────────────────
    if sub == "parameter":
        param_parts = rest.strip().split(None, 1)
        if len(param_parts) != 2:
            _console.print(f"[red]Usage: /set parameter <name> <value>[/]")
            _console.print(f"[dim]  Available: {', '.join(sorted(_ALL_PARAMS))}[/]\n")
            return

        name, raw_val = param_parts[0].lower(), param_parts[1]

        if name not in _ALL_PARAMS:
            _console.print(f"[red]Unknown parameter: {name}[/]")
            _console.print(f"[dim]  Available: {', '.join(sorted(_ALL_PARAMS))}[/]\n")
            return

        try:
            value = float(raw_val) if name in _FLOAT_PARAMS else int(raw_val)
        except ValueError:
            expected = "float" if name in _FLOAT_PARAMS else "integer"
            _console.print(f"[red]Invalid value for {name}: expected {expected}, got '{raw_val}'[/]\n")
            return

        candidate = dict(session.get("options", {}))
        candidate[name] = value
        try:
            normalized, warnings = validate_session_options(candidate)
        except RuntimeConfigurationError as exc:
            _console.print(f"[red]Invalid value for {name}: {exc}[/]\n")
            return
        session["options"] = normalized
        _console.print(f"[cyan][bold]✓  {name} = {value}[/]\n")
        for warning in warnings:
            _console.print(f"[yellow]⚠ {warning}[/]")
        if warnings:
            _console.print()
        return

    _console.print(f"[red]Unknown /set subcommand: {sub}[/]  [dim](try: parameter, profile, system, verbose, quiet, wordwrap, nowordwrap, history, nohistory, format, noformat, think, nothink)[/]\n")


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
        _console.print(f"\n[cyan][bold]Runtime profile:[/] {runtime.profile.value}")
        _console.print(f"[dim]{runtime.selection_reason}[/]")
        _console.print(f"[cyan][bold]Effective model parameters:[/]")
        for key, value in sorted(effective.items()):
            suffix = " [dim](session override)[/]" if key in opts else ""
            _console.print(f"  [green]{key}[/] = {value}{suffix}")
        for warning in runtime.warnings:
            _console.print(f"[yellow]⚠ {warning}[/]")
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
            _console.print(f"[dim]  Flags: {', '.join(flags)}[/]\n")
        return

    if sub == "system":
        prompt = session.get("system", "")
        if not prompt:
            # Check if history has one from the Modelfile
            if history and history[0].get("role") == "system":
                prompt = history[0]["content"]
        if prompt:
            _console.print(f"\n[cyan][bold]System prompt:[/]\n[dim]{prompt}[/]\n")
        else:
            _console.print(f"[dim]  No system prompt set (using Modelfile default).[/]\n")
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
            _console.print(f"\n[cyan][bold]Model:[/]  {MODEL_NAME}")
            _console.print(f"[cyan][bold]Family:[/] {family}")
            _console.print(f"[cyan][bold]Params:[/] {params}\n")
        except OllamaRuntimeError:
            _console.print(f"\n[cyan][bold]Model:[/]  {MODEL_NAME}\n")
        return

    _console.print(f"[red]Unknown /show subcommand: {sub}[/]  [dim](try: parameters, system, model)[/]\n")


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
            "verbose": session.get("verbose", False),
            "wordwrap": session.get("wordwrap", True),
            "system": session.get("system", ""),
            "history": session.get("history", True),
            "format": session.get("format", ""),
            "think": session.get("think", True),
            "runtime_profile": session.get("runtime_profile", "auto"),
        },
        "history": history,
    }

    try:
        atomic_write_json(filepath, payload, durable=True)
        display_name = os.path.basename(filepath)
        msg_count = sum(1 for m in history if m.get("role") == "user")
        _console.print(f"[cyan][bold]✓  Session saved:[/] [dim]{display_name}[/]  ({msg_count} user message{'s' if msg_count != 1 else ''})\n")
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
        if not saved:
            _console.print(f"[dim]  No saved sessions found.[/]\n")
            return
        _console.print(f"\n[cyan][bold]Saved sessions:[/]")
        for i, fp in enumerate(saved, 1):
            name = os.path.basename(fp).replace(".json", "")
            mtime = datetime.fromtimestamp(os.path.getmtime(fp)).strftime("%Y-%m-%d %H:%M")
            # Peek at message count
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                msg_count = sum(1 for m in data.get("history", []) if m.get("role") == "user")
                info = f"{msg_count} msg{'s' if msg_count != 1 else ''}"
            except Exception:
                info = "?"
            _console.print(f"  [green]{i}.[/] {name}  [dim]({mtime} · {info})[/]")
        _console.print(f"\n[dim]  Use /load <number> or /load <name> to restore.[/]\n")
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

    # Load the session
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _console.print(f"[red]Failed to load session: {exc}[/]\n")
        return

    saved_history = data.get("history", [])
    saved_session = data.get("session", {})
    if (
        not isinstance(saved_history, list)
        or any(not isinstance(message, dict) for message in saved_history)
        or not isinstance(saved_session, dict)
    ):
        _console.print("[red]Failed to load session: invalid session structure.[/]\n")
        return
    try:
        restored_options, warnings = validate_session_options(saved_session.get("options", {}))
    except RuntimeConfigurationError as exc:
        _console.print(f"[red]Failed to load session: invalid model settings: {exc}[/]\n")
        return
    runtime_profile = saved_session.get("runtime_profile", "auto")
    try:
        restored_runtime = get_runtime_config({
            "runtime_profile": runtime_profile,
            "options": restored_options,
        })
    except RuntimeConfigurationError as exc:
        _console.print(f"[red]Failed to load session: invalid runtime profile: {exc}[/]\n")
        return
    warnings = tuple(dict.fromkeys((*warnings, *restored_runtime.warnings)))

    # Validate before replacing live state so a malformed restore cannot erase
    # the current conversation.
    history.clear()
    history.extend(saved_history)
    session["options"] = restored_options
    session["verbose"] = saved_session.get("verbose", False)
    session["wordwrap"] = saved_session.get("wordwrap", True)
    session["system"] = saved_session.get("system", "")
    session["history"] = saved_session.get("history", True)
    session["format"] = saved_session.get("format", "")
    session["think"] = saved_session.get("think", True)
    session["runtime_profile"] = runtime_profile

    display_name = os.path.basename(target_path).replace(".json", "")
    msg_count = sum(1 for m in history if m.get("role") == "user")
    _console.print(f"[cyan][bold]✓  Session loaded:[/] [dim]{display_name}[/]  ({msg_count} user message{'s' if msg_count != 1 else ''})\n")
    for warning in warnings:
        _console.print(f"[yellow]⚠ {warning}[/]")
    if warnings:
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
        print(_VAULT_HELP)
        return

    sub = parts[0].lower()
    tokens = parts[1:]
    collection_raw = _extract_option(tokens, ("--collection", "-c"), "vault") or "vault"
    
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

        _console.print(f"\n[cyan][bold]Indexed vaults:[/]")
        for vault in vaults:
            name = vault.get("collection", "unknown")
            chunk_count = vault.get("indexed_chunks")
            if isinstance(chunk_count, int):
                count_text = f"{chunk_count} chunk{'s' if chunk_count != 1 else ''}"
            else:
                count_text = "chunk count unavailable"
            _console.print(f"  [green]{name}[/]  [dim]({count_text})[/]")
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
        _console.print(f"[cyan][bold]✓  Vault alias registered:[/] [green]{alias_name}[/] -> [dim]{coll_name}[/]\n")
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
            _console.print(f"\n[cyan][bold]Vault aliases:[/]")
            for entry in aliases:
                _console.print(f"  [green]{entry['alias']}[/] -> [dim]{entry['collection']}[/]")
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
                _console.print(f"[cyan][bold]✓  Vault renamed:[/] [dim]{data['old_collection']}[/] -> [green]{data['new_collection']}[/]  ({data.get('chunks_moved', 0)} chunks moved)")
                if data.get("updated_aliases"):
                    _console.print(f"  [dim]Updated aliases: {', '.join(data['updated_aliases'])}[/]")
                _console.print()
        except Exception as e:
            _console.print(f"[red]Failed to rename vault: {e}[/]\n")
        return

    if sub in ("add", "index"):
        if not tokens:
            _console.print(f"[red]Usage: /vault add <file-or-folder> [--collection name][/]\n")
            return

        target = " ".join(tokens)
        if not os.path.exists(target):
            _console.print(f"[red]Vault path not found: {target}[/]\n")
            return

        _print_status("🔧", f"Indexing vault content: [dim]{target}[/]", "yellow")
        if os.path.isdir(target):
            data = _call_tool_json("index_vault", vault_path=target, collection=collection)
        else:
            data = _call_tool_json(
                "index_vault",
                vault_path=os.path.dirname(target) or ".",
                file_path=target,
                collection=collection,
            )

        if "error" in data:
            _console.print(f"[red]Vault add failed: {data['error']}[/]\n")
            return

        indexed_files = data.get("indexed_files", 0)
        indexed_chunks = data.get("indexed_chunks", 0)
        skipped_count = data.get("skipped_count", 0)
        _console.print(
                f"[cyan][bold]✓  Vault indexed:[/] "
            f"{indexed_files} file{'s' if indexed_files != 1 else ''}, "
            f"{indexed_chunks} chunk{'s' if indexed_chunks != 1 else ''} "
            f"[dim](collection: {data.get('collection', collection)})[/]"
        )
        if skipped_count:
            _console.print(f"[yellow]  Skipped {skipped_count} file{'s' if skipped_count != 1 else ''}.[/]")
        _console.print()
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

        _print_status("🔍", f"Searching vault: [dim]{query}[/]", "yellow")
        data = _call_tool_json("vault_search", query=query, collection=collection, top_k=top_k, source=source)

        if "error" in data:
            _console.print(f"[red]Vault search failed: {data['error']}[/]\n")
            return

        matches = data.get("matches", [])
        _console.print(
                f"\n[cyan][bold]Vault search:[/] "
            f"{len(matches)} match{'es' if len(matches) != 1 else ''} "
            f"[dim](collection: {data.get('collection', collection)})[/]"
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

        _print_status("🗑", "Deleting vault index entries…", "yellow")
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
            _console.print(f"[cyan][bold]✓  Vault collection deleted:[/] [dim]{collection}[/]\n")
            return

        deleted_chunks = data.get("deleted_chunks", 0)
        if deleted_chunks:
            _console.print(
                f"[cyan][bold]✓  Vault entries deleted:[/] "
                f"{deleted_chunks} chunk{'s' if deleted_chunks != 1 else ''} "
                f"[dim](collection: {data.get('collection', collection)})[/]\n"
            )
        else:
            _console.print(f"[yellow]No indexed chunks matched:[/] [dim]{source}[/]\n")
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
        print(_COMMANDS_HELP)
        return True

    if base == "/clear":
        history.clear()
        # Also clear the custom system prompt override
        session["system"] = ""
        _console.print(f"[cyan][bold]✓  Conversation history and system prompt cleared.[/]\n")
        return True

    if base == "/set":
        _handle_set(rest, session, history)
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

def run() -> None:
    """Run the interactive agent loop.
    
    This acts as the primary entry point for terminal interaction.
    Initializes state, parses user input, handles commands, constructs history,
    and runs the streaming generator loop.
    """
    default_system_prompt = load_default_system_prompt()

    history: list[dict] = []
    session: dict = {
        "options": {},       # Runtime model parameters (temperature, etc.)
        "verbose": False,    # Show generation stats
        "wordwrap": True,    # Word wrapping (reserved for future use)
        "system": "",        # Custom system prompt override
        "history": True,     # Whether to keep conversation history across turns
        "format": "",        # Output format ("" = default, "json" = JSON mode)
        "think": True,       # Whether to enable model thinking/reasoning
        "runtime_profile": "auto",  # Hardware-aware profile; explicit options still win
    }

    try:
        from agent.runtime_config import get_runtime_config

        _boot_runtime = get_runtime_config()
        print_welcome_header(
            {
                "profile": _boot_runtime.profile.value,
                "model": _boot_runtime.chat_model,
                "num_ctx": str(_boot_runtime.num_ctx),
                "num_predict": str(_boot_runtime.num_predict),
            }
        )
    except Exception:
        print_welcome_header()

    while True:
        # ── User input ────────────────────────────────────────────────
        try:
            user_input = read_user_input(completions=CLI_SLASH_COMPLETIONS)
        except EOFError:
            # Exit loop if EOF (Ctrl+D) is encountered
            break

        if not user_input:
            continue

        # ── Handle slash commands ─────────────────────────────────────
        if user_input.startswith("/"):
            result = _handle_command(user_input, session, history)
            if result is None:
                break  # /quit
            continue  # Command was handled, skip LLM call

        # ── Sync system prompt ────────────────────────────────────────
        # Ensure the custom or default system prompt is consistently present in history
        active_system = session.get("system") or default_system_prompt
        if active_system:
            if not history or history[0].get("role") != "system" or history[0].get("content") != active_system:
                # Remove any stray system messages elsewhere and insert at front
                history[:] = [m for m in history if m.get("role") != "system"]
                history.insert(0, {"role": "system", "content": active_system})
        else:
            # If no system prompt is available at all, strip any injected ones
            history[:] = [m for m in history if m.get("role") != "system"]

        # ── Auto-index large or binary files when user inputs a local file path ─
        pre_tool_message = None
        try:
            if os.path.exists(user_input) and os.path.isfile(user_input):
                size = os.path.getsize(user_input)
                ext = os.path.splitext(user_input)[1].lower()
                # Threshold in bytes for auto-indexing (tunable)
                INDEX_THRESHOLD = 200_000
                if size > INDEX_THRESHOLD or ext in (".pdf", ".docx"):
                    _print_status("🔧", f"Large/binary file detected — indexing: [dim]{user_input}[/]", "yellow")
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
                                _print_status("✓", "Indexing complete.", "green")
                            else:
                                _print_status("⚠", "Indexing did not complete; the model will receive the error.", "yellow")
                        except Exception as e:
                            _print_status("⚠", f"Indexing failed: {e}", "red")
        except Exception:
            # Best-effort; don't let indexing errors stop the agent
            pass

        # ── Build messages to send ────────────────────────────────────
        # When history is disabled, send only the current message (+ system if set)
        if session["history"]:
            history.append({"role": "user", "content": user_input})
            # Trim history to keep prompt size bounded for consistent tok/s
            messages_to_send = prepare_messages_for_model(history, session, tools=TOOL_SCHEMAS)
        else:
            messages_to_send = []
            if history and history[0].get("role") == "system":
                messages_to_send.append(history[0])
            # If we have a pre-tool message (index result) and history is disabled,
            # insert it before the user message so the model sees it in the same turn.
            if pre_tool_message:
                messages_to_send.append(pre_tool_message)
            messages_to_send.append({"role": "user", "content": user_input})
            messages_to_send = prepare_messages_for_model(messages_to_send, session, tools=TOOL_SCHEMAS)

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
                _console.print(f"\n[yellow]⚠ {message}[/]\n")
                if session["history"]:
                    history.append({"role": "assistant", "content": message})
                break
            tool_rounds += 1
            tool_results = _process_tool_calls_with_turn_guard(assistant_msg["tool_calls"], executed_tool_calls)

            if session["history"]:
                history.extend(tool_results)
                # Trim history to keep follow-up requests within token budget
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

            # Follow-up call after tool results — also streamed
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
                
        # ── Trigger automatic history compaction in background ──────
        if session["history"]:
            _check_and_compact_history(history, session)
