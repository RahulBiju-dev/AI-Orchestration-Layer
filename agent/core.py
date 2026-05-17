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
    _render_terminal_markdown,
    _BOLD,
    _DIM,
    _CYAN,
    _YELLOW,
    _GREEN,
    _RED,
    _MAGENTA,
    _RESET,
    _CLEAR_LINE,
)

import signal
import sys
import threading
import time
import itertools
from datetime import datetime, timezone

import ollama
from rich.markdown import Markdown
from rich.live import Live

from tools.registry import TOOL_DISPATCH, TOOL_SCHEMAS

# ── Configuration ─────────────────────────────────────────────────────

MODEL_NAME = "gemma-agent"
_SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sessions")

# Parameters that accept float values via /set parameter
_FLOAT_PARAMS = {"temperature", "top_p", "top_k", "repeat_penalty", "presence_penalty", "frequency_penalty", "min_p", "tfs_z"}
# Parameters that accept integer values via /set parameter
_INT_PARAMS = {"num_ctx", "num_predict", "repeat_last_n", "seed", "num_gpu", "num_thread", "num_keep"}
_ALL_PARAMS = _FLOAT_PARAMS | _INT_PARAMS
# terminal helpers (spinner, renderer, ANSI constants) are imported
# from agent.terminal to keep terminal logic modular.

# ── History management ────────────────────────────────────────────────
# Rough token budget for conversation history.
# Keeps prompt size bounded so tok/s stays consistent across long sessions.
_HISTORY_TOKEN_BUDGET = 6000  # ~6k tokens ≈ leaves room for response in 8192 ctx


def _estimate_tokens(text: str) -> int:
    """Fast heuristic: ~1 token per 4 characters (close enough for trimming)."""
    return len(text) // 4 + 1


_interrupted = False

def _sigquit_handler(signum, frame):
    global _interrupted
    _interrupted = True


def _trim_history(messages: list[dict], budget: int = _HISTORY_TOKEN_BUDGET) -> list[dict]:
    """Trim conversation history to fit within a token budget.

    Preserves the system prompt (if any) and the most recent messages.
    Tool messages are kept with their associated assistant message.
    """
    if not messages:
        return messages

    # Separate system prompt from conversation
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
        # System prompt alone exceeds budget; keep it + last user message
        return system_msgs + conv_msgs[-1:]

    # Walk from newest to oldest, accumulating messages
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

def _stream_thinking_response(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    options: dict | None = None,
    verbose: bool = False,
    think: bool = True,
    fmt: str | None = None,
) -> dict:
    """Stream a response, showing thinking progress and the final answer.

    Returns the full assistant message dict (with thinking + content)
    for appending to history.
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
                    print(
                        f"\n{_MAGENTA}{_DIM}└─ [Interrupted] ────────────────────────{_RESET}\n",
                        file=sys.stderr,
                    )
                print(f"\n{_YELLOW}⚠ Generation interrupted by user (Ctrl+\\).{_RESET}\n", file=sys.stderr)
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
                    # Print thinking header
                    print(
                        f"\n{_MAGENTA}{_DIM}┌─ thinking ─────────────────────────────{_RESET}",
                        file=sys.stderr,
                    )
                    thinking_displayed = True

                thinking_buf += thinking_chunk
                # Print thinking content in dim magenta
                print(
                    f"{_MAGENTA}{_DIM}{thinking_chunk}{_RESET}",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            # ── Content tokens ────────────────────────────────────────
            content_chunk = msg.content or ""
            if content_chunk:
                if in_thinking:
                    # Transition from thinking to answering
                    in_thinking = False
                    print(
                        f"\n{_MAGENTA}{_DIM}└────────────────────────────────────────{_RESET}\n",
                        file=sys.stderr,
                    )
                    spinner.stop()
                elif spinner._thread and not spinner._stop_event.is_set():
                    spinner.stop()
                    if not thinking_displayed:
                        print()  # newline before answer

                content_buf += content_chunk

                # Initialize Live display on the first content chunk
                if live is None:
                    live = Live(
                        Markdown(_render_terminal_markdown(content_buf)),
                        console=_console,
                        auto_refresh=False,
                        screen=True,
                    )
                    live.start()

                # Throttle Markdown re-renders to reduce CPU overhead
                now = time.monotonic()
                if now - _last_render >= _RENDER_INTERVAL:
                    # Auto-scroll logic: keep the output size within the terminal bounds
                    max_lines = max(5, _console.height - 6)
                    lines = content_buf.rsplit("\n", max_lines)
                    if len(lines) > max_lines:
                        display_buf = "...\n" + "\n".join(lines[-max_lines:])
                    else:
                        display_buf = content_buf

                    # Update Markdown rendering in real-time
                    live.update(Markdown(_render_terminal_markdown(display_buf)), refresh=True)
                    _last_render = now

    finally:
        signal.signal(signal.SIGQUIT, old_handler)
        if live:
            live.stop()
            # Print the final complete markdown to the terminal so it remains in the scrollback buffer.
            # Using screen=True during streaming prevents the scrolling terminal duplication bug entirely.
            if content_buf:
                _console.print(Markdown(_render_terminal_markdown(content_buf)))

    # End of stream
    spinner.stop()

    if in_thinking:
        # Stream ended while still in thinking (no content followed)
        print(
            f"\n{_MAGENTA}{_DIM}└────────────────────────────────────────{_RESET}\n",
            file=sys.stderr,
        )

    if content_buf:
        print()  # final newline after streamed answer

    # Verbose stats
    if verbose:
        elapsed = time.monotonic() - t_start
        t_tokens = len(thinking_buf.split()) if thinking_buf else 0
        c_tokens = len(content_buf.split()) if content_buf else 0
        total = t_tokens + c_tokens
        tps = total / elapsed if elapsed > 0 else 0
        print(
            f"{_DIM}  ⏱  {elapsed:.1f}s  ·  ~{total} tokens  ·  ~{tps:.1f} tok/s{_RESET}\n",
            file=sys.stderr,
        )

    # Build the full message for history
    assistant_msg = {"role": "assistant", "content": content_buf}
    if thinking_buf:
        assistant_msg["thinking"] = thinking_buf
    return assistant_msg


def _process_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Execute each tool call and return the corresponding tool-role messages."""
    tool_messages: list[dict] = []

    for call in tool_calls:
        fn_name = call["function"]["name"]
        fn_args = call["function"]["arguments"]

        handler = TOOL_DISPATCH.get(fn_name)
        if handler is None:
            _print_status("⚠", f"Unknown tool: {fn_name}", _RED)
            result = json.dumps({"error": f"Unknown tool '{fn_name}'"})
        else:
            if fn_name == "web_search":
                _print_status("🔍", f"Searching the web: {_DIM}{fn_args.get('query', '')}{_RESET}", _YELLOW)
                result = handler(**fn_args)
                _print_status("✓", "Search complete — synthesizing answer…", _GREEN)
            elif fn_name == "read_document":
                _print_status("📄", f"Reading document: {_DIM}{fn_args.get('file_path', '')}{_RESET}", _YELLOW)
                result = handler(**fn_args)
                _print_status("✓", "Document read — synthesizing answer…", _GREEN)
            elif fn_name == "read_file":
                _print_status("📂", f"Reading file: {_DIM}{fn_args.get('file_path', '')}{_RESET}", _YELLOW)
                result = handler(**fn_args)
                _print_status("✓", "File read — synthesizing answer…", _GREEN)
            elif fn_name == "spotify_play":
                _print_status("🎵", f"Opening Spotify: {_DIM}{fn_args.get('query', '')}{_RESET}", _YELLOW)
                result = handler(**fn_args)
                _print_status("✓", "Spotify action complete — synthesizing answer…", _GREEN)
            else:
                _print_status("⚙️", f"Executing {fn_name}…", _YELLOW)
                result = handler(**fn_args)
                _print_status("✓", "Tool execution complete — synthesizing answer…", _GREEN)

        tool_messages.append({"role": "tool", "content": result})

    return tool_messages


# ── Slash commands ────────────────────────────────────────────────────

_COMMANDS_HELP = f"""
{_CYAN}{_BOLD}Available commands:{_RESET}
  {_GREEN}/help{_RESET}                          — Show this help message
  {_GREEN}/clear{_RESET}                         — Clear conversation history
  {_GREEN}/save [name]{_RESET}                   — Save current session  {_DIM}(optional name){_RESET}
  {_GREEN}/load [name|index]{_RESET}             — Load a saved session  {_DIM}(lists sessions if no arg){_RESET}
  {_GREEN}/set parameter <name> <val>{_RESET}    — Set a model parameter  {_DIM}(e.g. temperature 0.7){_RESET}
  {_GREEN}/set system "<prompt>"{_RESET}         — Set the system prompt for this session
  {_GREEN}/set history{_RESET}                   — Enable conversation history  {_DIM}(default){_RESET}
  {_GREEN}/set nohistory{_RESET}                 — Disable history  {_DIM}(each turn is standalone){_RESET}
  {_GREEN}/set wordwrap{_RESET}                  — Enable word wrapping  {_DIM}(default){_RESET}
  {_GREEN}/set nowordwrap{_RESET}                — Disable word wrapping
  {_GREEN}/set format json{_RESET}               — Force JSON output from the model
  {_GREEN}/set noformat{_RESET}                  — Disable forced output format  {_DIM}(default){_RESET}
  {_GREEN}/set verbose{_RESET}                   — Show generation stats after each response
  {_GREEN}/set quiet{_RESET}                     — Hide generation stats  {_DIM}(default){_RESET}
  {_GREEN}/set think{_RESET}                     — Enable model thinking/reasoning  {_DIM}(default){_RESET}
  {_GREEN}/set nothink{_RESET}                   — Disable model thinking
  {_GREEN}/show parameters{_RESET}               — Show current session parameters
  {_GREEN}/show system{_RESET}                   — Show the active system prompt
  {_GREEN}/show model{_RESET}                    — Show model info
  {_GREEN}/vault alias <name> <coll>{_RESET}     — Register a friendly alias for a collection
  {_GREEN}/vault aliases{_RESET}                  — List registered vault aliases
  {_GREEN}/vault rename <old> <new>{_RESET}       — Rename a vault collection
  {_GREEN}/vault add <path>{_RESET}               — Add a file or folder to the searchable vault
  {_GREEN}/vault list{_RESET}                     — List indexed vault collections
  {_GREEN}/vault search <query>{_RESET}           — Search the indexed vault
  {_GREEN}/vault delete <source>{_RESET}          — Delete indexed vault chunks by source/path
  {_GREEN}/quit{_RESET}                          — Exit the agent  {_DIM}(also /exit, /q){_RESET}
"""

_VAULT_HELP = f"""
{_CYAN}{_BOLD}Vault commands:{_RESET}
  {_GREEN}/vault list{_RESET}                                  — List indexed vault collections
  {_GREEN}/vault aliases{_RESET}                               — List registered vault aliases
  {_GREEN}/vault alias <name> <coll>{_RESET}                  — Register a friendly alias for a collection
  {_GREEN}/vault rename <old> <new>{_RESET}                   — Rename a vault collection
  {_GREEN}/vault add <path> [--collection name]{_RESET}        — Index a file or folder
  {_GREEN}/vault search <query> [--top-k n]{_RESET}            — Search indexed content
  {_GREEN}/vault search <query> [--source path]{_RESET}        — Restrict search to a source
  {_GREEN}/vault delete <source> [--collection name]{_RESET}   — Remove indexed chunks
  {_GREEN}/vault delete --all [--collection name]{_RESET}      — Delete a collection
"""


def _handle_set(args: str, session: dict, history: list[dict]) -> None:
    """Handle /set sub-commands."""
    parts = args.strip().split(None, 1)
    if not parts:
        print(f"{_RED}Usage: /set <subcommand> [args]{_RESET}  {_DIM}(type /help for details){_RESET}\n")
        return

    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    # ── /set verbose / /set quiet ─────────────────────────────────────
    if sub == "verbose":
        session["verbose"] = True
        print(f"{_CYAN}{_BOLD}✓  Verbose mode enabled — stats shown after each response.{_RESET}\n")
        return
    if sub == "quiet":
        session["verbose"] = False
        print(f"{_CYAN}{_BOLD}✓  Quiet mode enabled.{_RESET}\n")
        return

    # ── /set wordwrap / /set nowordwrap ───────────────────────────────
    if sub == "wordwrap":
        session["wordwrap"] = True
        print(f"{_CYAN}{_BOLD}✓  Word wrapping enabled.{_RESET}\n")
        return
    if sub == "nowordwrap":
        session["wordwrap"] = False
        print(f"{_CYAN}{_BOLD}✓  Word wrapping disabled.{_RESET}\n")
        return

    # ── /set history / /set nohistory ─────────────────────────────────
    if sub == "history":
        session["history"] = True
        print(f"{_CYAN}{_BOLD}✓  Conversation history enabled.{_RESET}\n")
        return
    if sub == "nohistory":
        session["history"] = False
        print(f"{_CYAN}{_BOLD}✓  History disabled — each turn is now standalone.{_RESET}\n")
        return

    # ── /set format json / /set noformat ──────────────────────────────
    if sub == "format":
        fmt = rest.strip().lower()
        if fmt == "json":
            session["format"] = "json"
            print(f"{_CYAN}{_BOLD}✓  JSON output mode enabled.{_RESET}\n")
        else:
            print(f"{_RED}Unsupported format: {fmt}{_RESET}  {_DIM}(supported: json){_RESET}\n")
        return
    if sub == "noformat":
        session["format"] = ""
        print(f"{_CYAN}{_BOLD}✓  Output formatting reset to default.{_RESET}\n")
        return

    # ── /set think / /set nothink ─────────────────────────────────────
    if sub == "think":
        session["think"] = True
        print(f"{_CYAN}{_BOLD}✓  Thinking/reasoning enabled.{_RESET}\n")
        return
    if sub == "nothink":
        session["think"] = False
        print(f"{_CYAN}{_BOLD}✓  Thinking disabled — model will respond directly.{_RESET}\n")
        return

    # ── /set system "<prompt>" ────────────────────────────────────────
    if sub == "system":
        # Strip surrounding quotes if present
        prompt = rest.strip().strip('"').strip("'")
        
        # Remove any existing system messages from history to avoid duplicates
        history[:] = [m for m in history if m.get("role") != "system"]

        if not prompt or prompt.lower() == "default":
            session["system"] = ""
            print(f"{_CYAN}{_BOLD}✓  System prompt reset to default.{_RESET}\n")
            return

        # Insert new system message at the start
        history.insert(0, {"role": "system", "content": prompt})
        session["system"] = prompt
        
        # Truncate display for confirmation
        display = prompt if len(prompt) <= 80 else prompt[:77] + "…"
        print(f"{_CYAN}{_BOLD}✓  System prompt set:{_RESET} {_DIM}{display}{_RESET}\n")
        return

    # ── /set parameter <name> <value> ─────────────────────────────────
    if sub == "parameter":
        param_parts = rest.strip().split(None, 1)
        if len(param_parts) != 2:
            print(f"{_RED}Usage: /set parameter <name> <value>{_RESET}")
            print(f"{_DIM}  Available: {', '.join(sorted(_ALL_PARAMS))}{_RESET}\n")
            return

        name, raw_val = param_parts[0].lower(), param_parts[1]

        if name not in _ALL_PARAMS:
            print(f"{_RED}Unknown parameter: {name}{_RESET}")
            print(f"{_DIM}  Available: {', '.join(sorted(_ALL_PARAMS))}{_RESET}\n")
            return

        try:
            value = float(raw_val) if name in _FLOAT_PARAMS else int(raw_val)
        except ValueError:
            expected = "float" if name in _FLOAT_PARAMS else "integer"
            print(f"{_RED}Invalid value for {name}: expected {expected}, got '{raw_val}'{_RESET}\n")
            return

        session["options"][name] = value
        print(f"{_CYAN}{_BOLD}✓  {name} = {value}{_RESET}\n")
        return

    print(f"{_RED}Unknown /set subcommand: {sub}{_RESET}  {_DIM}(try: parameter, system, verbose, quiet, wordwrap, nowordwrap, history, nohistory, format, noformat, think, nothink){_RESET}\n")


def _handle_show(args: str, session: dict, history: list[dict]) -> None:
    """Handle /show sub-commands."""
    sub = args.strip().lower() or "parameters"

    if sub == "parameters":
        opts = session.get("options", {})
        if not opts:
            print(f"{_DIM}  No custom parameters set (using model defaults).{_RESET}\n")
        else:
            print(f"\n{_CYAN}{_BOLD}Session parameters:{_RESET}")
            for k, v in sorted(opts.items()):
                print(f"  {_GREEN}{k}{_RESET} = {v}")
            print()
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
            print(f"{_DIM}  Flags: {', '.join(flags)}{_RESET}\n")
        return

    if sub == "system":
        prompt = session.get("system", "")
        if not prompt:
            # Check if history has one from the Modelfile
            if history and history[0].get("role") == "system":
                prompt = history[0]["content"]
        if prompt:
            print(f"\n{_CYAN}{_BOLD}System prompt:{_RESET}\n{_DIM}{prompt}{_RESET}\n")
        else:
            print(f"{_DIM}  No system prompt set (using Modelfile default).{_RESET}\n")
        return

    if sub in ("model", "info"):
        try:
            info = ollama.show(MODEL_NAME)
            model_info = getattr(info, "modelinfo", None) or {}
            family = model_info.get("general.architecture", "unknown")
            params = model_info.get("general.parameter_count", "unknown")
            print(f"\n{_CYAN}{_BOLD}Model:{_RESET}  {MODEL_NAME}")
            print(f"{_CYAN}{_BOLD}Family:{_RESET} {family}")
            print(f"{_CYAN}{_BOLD}Params:{_RESET} {params}\n")
        except Exception:
            print(f"\n{_CYAN}{_BOLD}Model:{_RESET}  {MODEL_NAME}\n")
        return

    print(f"{_RED}Unknown /show subcommand: {sub}{_RESET}  {_DIM}(try: parameters, system, model){_RESET}\n")


def _list_saved_sessions() -> list[str]:
    """Return a sorted list of session file paths (newest first)."""
    if not os.path.isdir(_SESSIONS_DIR):
        return []
    files = glob.glob(os.path.join(_SESSIONS_DIR, "*.json"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def _handle_save(args: str, session: dict, history: list[dict]) -> None:
    """Handle /save [name] — persist current session to a JSON file."""
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
        print(f"{_CYAN}{_BOLD}✓  Session saved:{_RESET} {_DIM}{display_name}{_RESET}  ({msg_count} user message{'s' if msg_count != 1 else ''})\n")
    except OSError as exc:
        print(f"{_RED}Failed to save session: {exc}{_RESET}\n")


def _handle_load(args: str, session: dict, history: list[dict]) -> None:
    """Handle /load [name|index] — load a previously saved session."""
    saved = _list_saved_sessions()
    arg = args.strip()

    # No argument: list available sessions
    if not arg:
        if not saved:
            print(f"{_DIM}  No saved sessions found.{_RESET}\n")
            return
        print(f"\n{_CYAN}{_BOLD}Saved sessions:{_RESET}")
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
            print(f"  {_GREEN}{i}.{_RESET} {name}  {_DIM}({mtime} · {info}){_RESET}")
        print(f"\n{_DIM}  Use /load <number> or /load <name> to restore.{_RESET}\n")
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
            print(f"{_YELLOW}Multiple sessions match '{arg}':{_RESET}")
            for fp in matches:
                print(f"  {_DIM}{os.path.basename(fp)}{_RESET}")
            print(f"{_DIM}  Be more specific or use /load to list with indices.{_RESET}\n")
            return

    if target_path is None:
        print(f"{_RED}No session found matching '{arg}'.{_RESET}  {_DIM}(type /load to list available sessions){_RESET}\n")
        return

    # Load the session
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{_RED}Failed to load session: {exc}{_RESET}\n")
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
    print(f"{_CYAN}{_BOLD}✓  Session loaded:{_RESET} {_DIM}{display_name}{_RESET}  ({msg_count} user message{'s' if msg_count != 1 else ''})\n")


def _extract_option(tokens: list[str], names: tuple[str, ...], default: str | None = None) -> str | None:
    """Remove and return a string option from a shlex token list."""
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
    """Remove and return whether any boolean flag exists in a shlex token list."""
    for index, token in enumerate(tokens):
        if token in names:
            del tokens[index]
            return True
    return False


def _call_tool_json(tool_name: str, **kwargs) -> dict:
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
    snippet = re.sub(r"\s+", " ", (text or "")).strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars - 3].rstrip() + "..."
    return snippet


def _handle_vault(args: str) -> None:
    """Handle /vault sub-commands."""
    try:
        parts = shlex.split(args)
    except ValueError as exc:
        print(f"{_RED}Invalid /vault command: {exc}{_RESET}\n")
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
            print(f"{_RED}Vault list failed: {data['error']}{_RESET}\n")
            return

        vaults = data.get("vaults", [])
        if not vaults:
            print(f"{_DIM}  No indexed vault collections found.{_RESET}\n")
            return

        print(f"\n{_CYAN}{_BOLD}Indexed vaults:{_RESET}")
        for vault in vaults:
            name = vault.get("collection", "unknown")
            chunk_count = vault.get("indexed_chunks")
            if isinstance(chunk_count, int):
                count_text = f"{chunk_count} chunk{'s' if chunk_count != 1 else ''}"
            else:
                count_text = "chunk count unavailable"
            print(f"  {_GREEN}{name}{_RESET}  {_DIM}({count_text}){_RESET}")
        print()
        return

    if sub in ("alias", "register"):
        if len(tokens) < 2:
            print(f"{_RED}Usage: /vault alias <human-name> <collection-name>{_RESET}\n")
            return
        alias_name = tokens[0]
        coll_name = tokens[1]
        from tools.vault_indexer import register_vault_alias
        register_vault_alias(alias_name, coll_name)
        print(f"{_CYAN}{_BOLD}✓  Vault alias registered:{_RESET} {_GREEN}{alias_name}{_RESET} -> {_DIM}{coll_name}{_RESET}\n")
        return

    if sub in ("aliases", "list-aliases"):
        from tools.vault_indexer import list_vault_aliases
        try:
            import json as _json
            data = _json.loads(list_vault_aliases())
            aliases = data.get("aliases", [])
            if not aliases:
                print(f"{_DIM}  No vault aliases registered.{_RESET}\n")
                return
            print(f"\n{_CYAN}{_BOLD}Vault aliases:{_RESET}")
            for entry in aliases:
                print(f"  {_GREEN}{entry['alias']}{_RESET} -> {_DIM}{entry['collection']}{_RESET}")
            print()
        except Exception as e:
            print(f"{_RED}Failed to list aliases: {e}{_RESET}\n")
        return

    if sub in ("rename", "mv"):
        if len(tokens) < 2:
            print(f"{_RED}Usage: /vault rename <old-name> <new-name>{_RESET}\n")
            return
        old_name = tokens[0]
        new_name = tokens[1]
        from tools.vault_indexer import rename_vault
        try:
            import json as _json
            data = _json.loads(rename_vault(old_name, new_name))
            if data.get("error"):
                print(f"{_RED}✗  {data['error']}{_RESET}\n")
            else:
                print(f"{_CYAN}{_BOLD}✓  Vault renamed:{_RESET} {_DIM}{data['old_collection']}{_RESET} -> {_GREEN}{data['new_collection']}{_RESET}  ({data.get('chunks_moved', 0)} chunks moved)")
                if data.get("updated_aliases"):
                    print(f"  {_DIM}Updated aliases: {', '.join(data['updated_aliases'])}{_RESET}")
                print()
        except Exception as e:
            print(f"{_RED}Failed to rename vault: {e}{_RESET}\n")
        return

    if sub in ("add", "index"):
        if not tokens:
            print(f"{_RED}Usage: /vault add <file-or-folder> [--collection name]{_RESET}\n")
            return

        target = " ".join(tokens)
        if not os.path.exists(target):
            print(f"{_RED}Vault path not found: {target}{_RESET}\n")
            return

        _print_status("🔧", f"Indexing vault content: {_DIM}{target}{_RESET}", _YELLOW)
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
            print(f"{_RED}Vault add failed: {data['error']}{_RESET}\n")
            return

        indexed_files = data.get("indexed_files", 0)
        indexed_chunks = data.get("indexed_chunks", 0)
        skipped_count = data.get("skipped_count", 0)
        print(
            f"{_CYAN}{_BOLD}✓  Vault indexed:{_RESET} "
            f"{indexed_files} file{'s' if indexed_files != 1 else ''}, "
            f"{indexed_chunks} chunk{'s' if indexed_chunks != 1 else ''} "
            f"{_DIM}(collection: {data.get('collection', collection)}){_RESET}"
        )
        if skipped_count:
            print(f"{_YELLOW}  Skipped {skipped_count} file{'s' if skipped_count != 1 else ''}.{_RESET}")
        print()
        return

    if sub in ("search", "find"):
        top_k_raw = _extract_option(tokens, ("--top-k", "-k"), None)
        source = _extract_option(tokens, ("--source", "-s"), None)
        query = " ".join(tokens).strip()
        if not query:
            print(f"{_RED}Usage: /vault search <query> [--collection name] [--top-k n] [--source path]{_RESET}\n")
            return
        try:
            top_k = int(top_k_raw) if top_k_raw is not None else 6
        except ValueError:
            print(f"{_RED}Invalid --top-k value: {top_k_raw}{_RESET}\n")
            return

        _print_status("🔍", f"Searching vault: {_DIM}{query}{_RESET}", _YELLOW)
        data = _call_tool_json("vault_search", query=query, collection=collection, top_k=top_k, source=source)

        if "error" in data:
            print(f"{_RED}Vault search failed: {data['error']}{_RESET}\n")
            return

        matches = data.get("matches", [])
        print(
            f"\n{_CYAN}{_BOLD}Vault search:{_RESET} "
            f"{len(matches)} match{'es' if len(matches) != 1 else ''} "
            f"{_DIM}(collection: {data.get('collection', collection)}){_RESET}"
        )
        for match in matches:
            source_name = match.get("source") or match.get("filename") or "unknown"
            chunk = match.get("chunk_index", "?")
            distance = match.get("distance")
            distance_text = f" · distance {distance:.4f}" if isinstance(distance, (int, float)) else ""
            print(f"  {_GREEN}{match.get('rank', '?')}.{_RESET} {source_name}  {_DIM}(chunk {chunk}{distance_text}){_RESET}")
            snippet = _format_match_snippet(match.get("text"))
            if snippet:
                print(f"     {_DIM}{snippet}{_RESET}")
        print()
        return

    if sub in ("delete", "remove", "rm"):
        delete_collection = _extract_flag(tokens, ("--all", "--collection-all"))
        source = " ".join(tokens).strip()
        if not delete_collection and not source:
            print(f"{_RED}Usage: /vault delete <source> [--collection name]{_RESET}\n")
            return

        _print_status("🗑", "Deleting vault index entries…", _YELLOW)
        data = _call_tool_json(
            "delete_vault_item",
            source=source or None,
            collection=collection,
            delete_collection=delete_collection,
        )

        if "error" in data:
            print(f"{_RED}Vault delete failed: {data['error']}{_RESET}\n")
            return

        if data.get("deleted_collection"):
            print(f"{_CYAN}{_BOLD}✓  Vault collection deleted:{_RESET} {_DIM}{collection}{_RESET}\n")
            return

        deleted_chunks = data.get("deleted_chunks", 0)
        if deleted_chunks:
            print(
                f"{_CYAN}{_BOLD}✓  Vault entries deleted:{_RESET} "
                f"{deleted_chunks} chunk{'s' if deleted_chunks != 1 else ''} "
                f"{_DIM}(collection: {data.get('collection', collection)}){_RESET}\n"
            )
        else:
            print(f"{_YELLOW}No indexed chunks matched:{_RESET} {_DIM}{source}{_RESET}\n")
        return

    print(f"{_RED}Unknown /vault subcommand: {sub}{_RESET}  {_DIM}(try: list, aliases, rename, add, search, delete){_RESET}\n")


def _handle_command(cmd: str, session: dict, history: list[dict]) -> bool | None:
    """Handle a slash command. Returns True if handled, None to quit."""
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
        print(f"{_CYAN}{_BOLD}✓  Conversation history and system prompt cleared.{_RESET}\n")
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
    print(f"{_RED}Unknown command: {base}{_RESET}  {_DIM}(type /help for available commands){_RESET}\n")
    return True


# ── Main loop ─────────────────────────────────────────────────────────

def run() -> None:
    """Run the interactive agent loop."""
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

    print(f"\n{_CYAN}{_BOLD}╭───────────────────────────────────────╮{_RESET}")
    print(f"{_CYAN}{_BOLD}│   Gemma CLI Agent  ·  type /help      │{_RESET}")
    print(f"{_CYAN}{_BOLD}╰───────────────────────────────────────╯{_RESET}\n")

    while True:
        # ── User input ────────────────────────────────────────────────
        try:
            user_input = input(f"{_GREEN}{_BOLD}>>> {_RESET}").strip()
        except EOFError:
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
        # Ensure the custom system prompt is present in history if set
        if session.get("system"):
            if not history or history[0].get("role") != "system":
                # Remove any stray system messages elsewhere and insert at front
                history[:] = [m for m in history if m.get("role") != "system"]
                history.insert(0, {"role": "system", "content": session["system"]})
        else:
            # If using Modelfile default, strip any injected system prompts from history
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
                    _print_status("🔧", f"Large/binary file detected — indexing: {_DIM}{user_input}{_RESET}", _YELLOW)
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
                            _print_status("✓", "Indexing complete.", _GREEN)
                        except Exception as e:
                            _print_status("⚠", f"Indexing failed: {e}", _RED)
        except Exception:
            # Best-effort; don't let indexing errors stop the agent
            pass

        # ── Build messages to send ────────────────────────────────────
        # When history is disabled, send only the current message (+ system if set)
        if session["history"]:
            history.append({"role": "user", "content": user_input})
            # Trim history to keep prompt size bounded for consistent tok/s
            messages_to_send = _trim_history(history)
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
                messages_to_send = _trim_history(history)
            else:
                messages_to_send.append(assistant_msg)
                messages_to_send.extend(tool_results)

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
