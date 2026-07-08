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
    print_thinking_footer,
    print_thinking_header,
    print_welcome_header,
    read_user_input,
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

import ollama
from rich.live import Live

from tools.registry import TOOL_DISPATCH, TOOL_SCHEMAS

# ── Configuration ─────────────────────────────────────────────────────

MODEL_NAME = "selene"
_SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sessions")

# Parameters that accept float values via /set parameter
_FLOAT_PARAMS = {"temperature", "top_p", "top_k", "repeat_penalty", "presence_penalty", "frequency_penalty", "min_p", "tfs_z"}
# Parameters that accept integer values via /set parameter
_INT_PARAMS = {"num_ctx", "num_predict", "repeat_last_n", "seed", "num_gpu", "num_thread", "num_keep"}
_ALL_PARAMS = _FLOAT_PARAMS | _INT_PARAMS
# terminal helpers (spinner, renderer, ANSI constants) are imported
# from agent.terminal to keep terminal logic modular.

# ── History management ────────────────────────────────────────────────
# Keeps prompt size bounded so tok/s stays consistent across long sessions.

def _estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a string using a fast heuristic.
    
    This is used to bound the history length without running a full tokenizer,
    saving compute time on every turn.
    
    Args:
        text (str): The text to estimate tokens for.
        
    Returns:
        int: The estimated token count (~1 token per 4 characters).
    """
    return len(text) // 4 + 1


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


def _trim_history(messages: list[dict], num_ctx: int = 8192) -> list[dict]:
    """Trim conversation history to fit within a dynamic token budget.

    Preserves the system prompt (if any) and the most recent messages.
    Tool messages are kept with their associated assistant message.
    
    Args:
        messages (list[dict]): The full conversation history list.
        num_ctx (int, optional): The context window size. Defaults to 8192.
        
    Returns:
        list[dict]: A trimmed list of messages fitting the 95% budget.
    """
    budget = int(num_ctx * 0.95)
    if not messages:
        return messages

    # Separate system prompt from conversation to ensure it is always preserved
    system_msgs = []
    conv_msgs = []
    for msg in messages:
        if msg.get("role") == "system":
            system_msgs.append(msg)
        else:
            conv_msgs.append(msg)

    # Calculate system prompt cost
    system_cost = sum(_estimate_tokens(m.get("content", "")) for m in system_msgs)
    remaining_budget = budget - system_cost

    if remaining_budget <= 0:
        # System prompt alone exceeds budget; keep it + last user message as a fallback
        return system_msgs + conv_msgs[-1:]

    # Walk from newest to oldest, accumulating messages until the budget is hit
    kept: list[dict] = []
    used = 0
    for msg in reversed(conv_msgs):
        content = msg.get("content", "")
        thinking = msg.get("thinking", "")
        cost = _estimate_tokens(content) + _estimate_tokens(thinking)
        if used + cost > remaining_budget and kept:
            break
        kept.append(msg)
        used += cost

    kept.reverse()
    return system_msgs + kept


def _compact_history_bg(history: list[dict], session: dict, start_idx: int, end_idx: int) -> None:
    """Run background summarization and replace older messages."""
    import ollama
    messages_to_compact = history[start_idx:end_idx]
    
    # We want to format the previous conversation for the LLM to summarize
    text_to_summarize = ""
    for msg in messages_to_compact:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        text_to_summarize += f"[{role.upper()}]: {content}\n\n"
        
    system_prompt = session.get("system", "")
    if not system_prompt and history and history[0].get("role") == "system":
        system_prompt = history[0].get("content", "")
        
    summarize_prompt = (
        "Please provide a concise, factual summary of the following conversation history. "
        "Retain all key context, decisions, and factual information so it can be used "
        "in place of the full history without losing context.\n\n"
        f"Conversation:\n{text_to_summarize}"
    )
    
    compact_messages = []
    if system_prompt:
        compact_messages.append({"role": "system", "content": system_prompt})
    compact_messages.append({"role": "user", "content": summarize_prompt})
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=compact_messages,
            options=session.get("options", {})
        )
        summary = response.message.content or ""
        if summary:
            from tools.context_memory_optimizer import context_memory_optimizer
            optimized = json.loads(context_memory_optimizer(
                [{"role": "assistant", "content": summary}],
                target_tokens=max(512, int(session.get("options", {}).get("num_ctx", 8192) * 0.25)),
                preserve_recent=1,
            ))
            history[start_idx:end_idx] = optimized.get("messages") or [
                {"role": "assistant", "content": f"[System Note: Older conversation compacted]\n{summary}"}
            ]
    except Exception as e:
        # Ignore errors in background compaction to prevent terminal spam
        pass
    finally:
        session["_is_compacting"] = False


def _check_and_compact_history(history: list[dict], session: dict) -> None:
    """Check token usage and trigger background compaction if > 90%."""
    if session.get("_is_compacting"):
        return
        
    num_ctx = session.get("options", {}).get("num_ctx", 8192)
    compact_threshold = int(num_ctx * 0.90)
    
    total_tokens = sum(_estimate_tokens(m.get("content", "") + m.get("thinking", "")) for m in history)
    
    if total_tokens > compact_threshold:
        # Determine the slice to compact. Keep system prompt (idx 0) and at least the 4 most recent messages (e.g. 2 turns).
        if len(history) >= 6:
            start_idx = 1 if history[0].get("role") == "system" else 0
            end_idx = len(history) - 4
            if start_idx < end_idx:
                session["_is_compacting"] = True
                import threading
                t = threading.Thread(target=_compact_history_bg, args=(history, session, start_idx, end_idx))
                t.daemon = True
                t.start()

def _stream_thinking_response(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    options: dict | None = None,
    verbose: bool = False,
    think: bool = True,
    fmt: str | None = None,
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

    thinking_buf = ""
    content_buf = ""
    in_thinking = False
    thinking_displayed = False

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": think,
        "keep_alive": "30m",
    }
    if fmt:
        kwargs["format"] = fmt
    if tools:
        kwargs["tools"] = tools
    if options:
        kwargs["options"] = options

    stream = ollama.chat(**kwargs)

    live = None
    _last_render = 0.0  # throttle Live.update() calls
    _RENDER_INTERVAL = 0.08  # seconds between re-renders (~12 FPS)

    global _interrupted
    _interrupted = False
    old_handler = signal.signal(signal.SIGQUIT, _sigquit_handler)

    try:
        for chunk in stream:
            if _interrupted:
                spinner.stop()
                if in_thinking:
                    in_thinking = False
                    print_thinking_footer("interrupted")
                _console.print(f"\n[yellow]⚠ Generation interrupted by user (Ctrl+\\).[/]\n")
                break
            
            msg = chunk.message

            # ── Tool calls come through as non-streamed chunks ────────
            if msg.tool_calls:
                spinner.stop()
                # Build the assistant message with any accumulated content
                assistant_msg = {"role": "assistant", "content": content_buf}
                if thinking_buf:
                    assistant_msg["thinking"] = thinking_buf
                assistant_msg["tool_calls"] = [
                    {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
                return assistant_msg

            # ── Thinking tokens ───────────────────────────────────────
            thinking_chunk = getattr(msg, "thinking", None) or ""
            if thinking_chunk:
                if not in_thinking:
                    in_thinking = True
                    spinner.stop()
                    print_thinking_header()
                    thinking_displayed = True

                thinking_buf += thinking_chunk
                # Print thinking content in dim magenta
                from rich.markup import escape
                _console.print(escape(thinking_chunk), style="dim magenta", end="")
                continue

            # ── Content tokens ────────────────────────────────────────
            content_chunk = msg.content or ""
            if content_chunk:
                if in_thinking:
                    # Transition from thinking to answering
                    in_thinking = False
                    print_thinking_footer()
                    spinner.stop()
                elif spinner._thread and not spinner._stop_event.is_set():
                    spinner.stop()
                    if not thinking_displayed:
                        _console.print()  # newline before answer

                content_buf += content_chunk

                # Initialize Live display on the first content chunk
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

                # Throttle Markdown re-renders to reduce CPU overhead
                now = time.monotonic()
                if now - _last_render >= _RENDER_INTERVAL:
                    # Update Markdown rendering in real-time
                    live.update(assistant_stream_panel(content_buf), refresh=True)
                    _last_render = now

    finally:
        signal.signal(signal.SIGQUIT, old_handler)
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
    return assistant_msg


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
    tool_messages: list[dict] = []

    for call in tool_calls:
        fn_name = call["function"]["name"]
        fn_args = call["function"]["arguments"]

        handler = TOOL_DISPATCH.get(fn_name)
        if handler is None:
            _print_status("⚠", f"Unknown tool: {fn_name}", "red")
            result = json.dumps({"error": f"Unknown tool '{fn_name}'"})
        else:
            try:
                if fn_name == "web_search":
                    _print_status("🔍", f"Searching the web: [dim]{fn_args.get('query', '')}[/]", "yellow")
                    result = handler(**fn_args)
                    _print_status("✓", "Search complete — synthesizing answer…", "green")
                elif fn_name == "web_scrape":
                    _print_status("🌐", f"Reading web page: [dim]{fn_args.get('url', '')}[/]", "yellow")
                    result = handler(**fn_args)
                    _print_status("✓", "Page read — synthesizing answer…", "green")
                elif fn_name == "read_document":
                    _print_status("📄", f"Reading document: [dim]{fn_args.get('file_path', '')}[/]", "yellow")
                    result = handler(**fn_args)
                    _print_status("✓", "Document read — synthesizing answer…", "green")
                elif fn_name == "read_file":
                    _print_status("📂", f"Reading file: [dim]{fn_args.get('file_path', '')}[/]", "yellow")
                    result = handler(**fn_args)
                    _print_status("✓", "File read — synthesizing answer…", "green")
                elif fn_name == "spotify_play":
                    _print_status("🎵", f"Opening Spotify: [dim]{fn_args.get('query', '')}[/]", "yellow")
                    result = handler(**fn_args)
                    _print_status("✓", "Spotify action complete — synthesizing answer…", "green")
                elif fn_name == "open_app":
                    _print_status("🚀", f"Opening app: [dim]{fn_args.get('app_name', '')}[/]", "yellow")
                    result = handler(**fn_args)
                    _print_status("✓", "App launch request sent — synthesizing answer…", "green")
                elif fn_name == "launch_apps":
                    names = ", ".join(str(value) for value in fn_args.get("app_names", []))
                    _print_status("🚀", f"Opening apps: [dim]{names}[/]", "yellow")
                    result = handler(**fn_args)
                    _print_status("✓", "App launch requests processed — synthesizing answer…", "green")
                elif fn_name == "open_terminal_at_path":
                    _print_status("💻", f"Opening terminal at: [dim]{fn_args.get('path', '')}[/]", "yellow")
                    result = handler(**fn_args)
                    _print_status("✓", "Terminal launch request sent — synthesizing answer…", "green")
                else:
                    _print_status("⚙️", f"Executing {fn_name}…", "yellow")
                    result = handler(**fn_args)
                    _print_status("✓", "Tool execution complete — synthesizing answer…", "green")
            except Exception as e:
                _print_status("❌", f"Error executing {fn_name}: {e}", "red")
                result = json.dumps({"error": f"Tool execution failed: {str(e)}"})

        tool_messages.append({"role": "tool", "tool_name": fn_name, "content": result})

    return tool_messages


# ── Slash commands ────────────────────────────────────────────────────

_COMMANDS_HELP = f"""
[cyan][bold]Available commands:[/]
  [green]/help[/]                          — Show this help message
  [green]/clear[/]                         — Clear conversation history
  [green]/save [name][/]                   — Save current session  [dim](optional name)[/]
  [green]/load [name|index][/]             — Load a saved session  [dim](lists sessions if no arg)[/]
  [green]/set parameter <name> <val>[/]    — Set a model parameter  [dim](e.g. temperature 0.7)[/]
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

        session["options"][name] = value
        _console.print(f"[cyan][bold]✓  {name} = {value}[/]\n")
        return

    _console.print(f"[red]Unknown /set subcommand: {sub}[/]  [dim](try: parameter, system, verbose, quiet, wordwrap, nowordwrap, history, nohistory, format, noformat, think, nothink)[/]\n")


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
        if not opts:
            _console.print(f"[dim]  No custom parameters set (using model defaults).[/]\n")
        else:
            _console.print(f"\n[cyan][bold]Session parameters:[/]")
            for k, v in sorted(opts.items()):
                _console.print(f"  [green]{k}[/] = {v}")
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
            info = ollama.show(MODEL_NAME)
            model_info = getattr(info, "modelinfo", None) or {}
            family = model_info.get("general.architecture", "unknown")
            params = model_info.get("general.parameter_count", "unknown")
            _console.print(f"\n[cyan][bold]Model:[/]  {MODEL_NAME}")
            _console.print(f"[cyan][bold]Family:[/] {family}")
            _console.print(f"[cyan][bold]Params:[/] {params}\n")
        except Exception:
            _console.print(f"\n[cyan][bold]Model:[/]  {MODEL_NAME}\n")
        return

    _console.print(f"[red]Unknown /show subcommand: {sub}[/]  [dim](try: parameters, system, model)[/]\n")


def _list_saved_sessions() -> list[str]:
    """Return a sorted list of session file paths (newest first).
    
    Returns:
        list[str]: A list of absolute file paths to saved sessions.
    """
    if not os.path.isdir(_SESSIONS_DIR):
        return []
    files = glob.glob(os.path.join(_SESSIONS_DIR, "*.json"))
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
        },
        "history": history,
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
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

    # Restore history
    history.clear()
    history.extend(data.get("history", []))

    # Restore session parameters
    saved_session = data.get("session", {})
    session["options"] = saved_session.get("options", {})
    session["verbose"] = saved_session.get("verbose", False)
    session["wordwrap"] = saved_session.get("wordwrap", True)
    session["system"] = saved_session.get("system", "")
    session["history"] = saved_session.get("history", True)
    session["format"] = saved_session.get("format", "")
    session["think"] = saved_session.get("think", True)

    display_name = os.path.basename(target_path).replace(".json", "")
    msg_count = sum(1 for m in history if m.get("role") == "user")
    _console.print(f"[cyan][bold]✓  Session loaded:[/] [dim]{display_name}[/]  ({msg_count} user message{'s' if msg_count != 1 else ''})\n")


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
    """Helper method to invoke a tool locally and ensure the output is a dict.
    
    Args:
        tool_name (str): Tool function name to execute.
        **kwargs: Arguments to pass to the tool.
        
    Returns:
        dict: The tool result parsed as JSON, or an error dict.
    """
    handler = TOOL_DISPATCH.get(tool_name)
    if handler is None:
        return {"error": f"Tool not found: {tool_name}"}
    try:
        result = handler(**kwargs)
    except Exception as exc:
        return {"error": str(exc)}
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"result": result}
    if isinstance(result, dict):
        return result
    return {"result": result}


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
    import subprocess
    default_system_prompt = ""
    try:
        res = subprocess.run(["ollama", "show", MODEL_NAME, "--system"], capture_output=True, text=True)
        if res.returncode == 0:
            default_system_prompt = res.stdout.strip()
    except Exception:
        pass

    history: list[dict] = []
    session: dict = {
        "options": {},       # Runtime model parameters (temperature, etc.)
        "verbose": False,    # Show generation stats
        "wordwrap": True,    # Word wrapping (reserved for future use)
        "system": "",        # Custom system prompt override
        "history": True,     # Whether to keep conversation history across turns
        "format": "",        # Output format ("" = default, "json" = JSON mode)
        "think": True,       # Whether to enable model thinking/reasoning
    }

    print_welcome_header()

    while True:
        # ── User input ────────────────────────────────────────────────
        try:
            user_input = read_user_input()
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
                    handler = TOOL_DISPATCH.get("index_vault")
                    if handler:
                        try:
                            res = handler(vault_path=os.path.dirname(user_input) or ".", file_path=user_input)
                            # Ensure we push a tool-style message into history so the model knows indexing occurred
                            if isinstance(res, str):
                                tool_content = res
                            else:
                                import json as _json
                                tool_content = _json.dumps(res)
                            tool_msg = {"role": "tool", "content": tool_content}
                            if session["history"]:
                                history.append(tool_msg)
                            else:
                                pre_tool_message = tool_msg
                            _print_status("✓", "Indexing complete.", "green")
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
            num_ctx = session["options"].get("num_ctx", 8192)
            messages_to_send = _trim_history(history, num_ctx)
        else:
            messages_to_send = []
            if history and history[0].get("role") == "system":
                messages_to_send.append(history[0])
            # If we have a pre-tool message (index result) and history is disabled,
            # insert it before the user message so the model sees it in the same turn.
            if pre_tool_message:
                messages_to_send.append(pre_tool_message)
            messages_to_send.append({"role": "user", "content": user_input})

        # ── LLM call with streaming + thinking ────────────────────────
        assistant_msg = _stream_thinking_response(
            model=MODEL_NAME,
            messages=messages_to_send,
            tools=TOOL_SCHEMAS,
            options=session["options"] or None,
            verbose=session["verbose"],
            think=session["think"],
            fmt=session["format"] or None,
        )

        if session["history"]:
            history.append(assistant_msg)

        # ── Tool-call loop (iterative, in case of chained calls) ──────
        while assistant_msg.get("tool_calls"):
            tool_results = _process_tool_calls(assistant_msg["tool_calls"])
            if session["history"]:
                history.extend(tool_results)
                # Trim history to keep follow-up requests within token budget
                num_ctx = session["options"].get("num_ctx", 8192)
                reminder = {
                    "role": "user",
                    "content": (
                        "Continue the current turn using the tool result above. Answer this "
                        "original request directly; do not ask the user to repeat it:\n"
                        f"{user_input}"
                    ),
                }
                messages_to_send = _trim_history([*history, reminder], num_ctx)
            else:
                messages_to_send.append(assistant_msg)
                messages_to_send.extend(tool_results)
                messages_to_send.append({
                    "role": "user",
                    "content": (
                        "Continue the current turn using the tool result above. Answer this "
                        "original request directly; do not ask the user to repeat it:\n"
                        f"{user_input}"
                    ),
                })

            # Follow-up call after tool results — also streamed
            assistant_msg = _stream_thinking_response(
                model=MODEL_NAME,
                messages=messages_to_send,
                tools=TOOL_SCHEMAS,
                options=session["options"] or None,
                verbose=session["verbose"],
                think=session["think"],
                fmt=session["format"] or None,
            )
            if session["history"]:
                history.append(assistant_msg)
                
        # ── Trigger automatic history compaction in background ──────
        if session["history"]:
            _check_and_compact_history(history, session)
