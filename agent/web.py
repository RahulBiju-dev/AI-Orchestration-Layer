#!/usr/bin/env python3
"""
agent/web.py — Web Server for the Selene AI Agent UI.

Serves static frontend files and exposes HTTP endpoints / SSE streaming for interaction.
"""

import http.server
import socketserver
import json
import os
import glob
import sys
import time
import socket
import re
import webbrowser
import threading
from datetime import datetime, timezone
import ollama

# Import agent configurations and helpers from core
from agent.core import MODEL_NAME, _trim_history, _check_and_compact_history
from tools.registry import TOOL_DISPATCH, TOOL_SCHEMAS

# Setup directories
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_SESSIONS_DIR = os.path.expanduser("~/.selene-agent/sessions")

# Global Application State
GLOBAL_STATE = {
    "history": [],
    "session": {
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40
        },
        "verbose": False,
        "wordwrap": True,
        "system": "",
        "history": True,
        "format": "",
        "think": True,
    },
    "active_session_name": "Active Session"
}

# ── Session Management Functions ──────────────────────────────────────

def list_saved_sessions() -> list[str]:
    """Return a sorted list of session filenames (newest first).
    
    Returns:
        list[str]: A list of saved session filenames.
    """
    if not os.path.isdir(_SESSIONS_DIR):
        return []
    files = glob.glob(os.path.join(_SESSIONS_DIR, "*.json"))
    files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.basename(f) for f in files]


def save_session(name: str) -> str:
    """Persist current state to a JSON file.
    
    Args:
        name (str): An optional name for the session.
        
    Returns:
        str: The filename of the saved session.
    """
    os.makedirs(_SESSIONS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if name:
        # Sanitize name
        safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
        filename = f"{safe_name}_{timestamp}.json"
    else:
        filename = f"session_{timestamp}.json"
        
    filepath = os.path.join(_SESSIONS_DIR, filename)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "session": GLOBAL_STATE["session"],
        "history": GLOBAL_STATE["history"],
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        
    GLOBAL_STATE["active_session_name"] = filename
    return filename


def autosave_session() -> str | None:
    """Create or update the active chat without requiring a manual save."""
    if not GLOBAL_STATE["history"]:
        return None

    filename = GLOBAL_STATE.get("active_session_name", "")
    filepath = os.path.join(_SESSIONS_DIR, os.path.basename(filename))
    if filename in ("", "Active Session", "New conversation") or not os.path.isfile(filepath):
        # Use a temporary name until the first response is complete and Selene
        # can name the conversation from its actual subject.
        return save_session("")

    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "session": GLOBAL_STATE["session"],
        "history": GLOBAL_STATE["history"],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return filename


def save_session_snapshot(
    filename: str,
    session_data: dict,
    history_data: list[dict],
) -> str:
    """Persist one conversation without depending on the globally selected chat."""
    os.makedirs(_SESSIONS_DIR, exist_ok=True)
    safe_filename = os.path.basename(filename or "")
    if safe_filename in ("", "Active Session", "New conversation") or not safe_filename.endswith(".json"):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        safe_filename = f"session_{timestamp}.json"

    filepath = os.path.join(_SESSIONS_DIR, safe_filename)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "session": session_data,
        "history": history_data,
    }
    temporary = f"{filepath}.tmp-{threading.get_ident()}"
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        os.replace(temporary, filepath)
    finally:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass
    return safe_filename


def _normalize_agent_title(value: str) -> str:
    """Constrain model output to a safe, human-readable 2-3 word title."""
    first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    first_line = re.sub(r"^(?:conversation\s+)?title\s*:\s*", "", first_line, flags=re.I)
    words = re.findall(r"[^\W_]+(?:['’][^\W_]+)?", first_line, flags=re.UNICODE)[:3]
    if len(words) == 1:
        words.append("Discussion" if words[0].casefold() == "chat" else "Chat")
    if len(words) < 2:
        return "New Conversation"
    return " ".join(words)


def generate_conversation_title(history: list[dict]) -> str:
    """Ask the local agent for a short semantic title, with a stable fallback."""
    first_user = next(
        (str(message.get("content", "")) for message in history if message.get("role") == "user"),
        "",
    )
    last_assistant = next(
        (str(message.get("content", "")) for message in reversed(history) if message.get("role") == "assistant"),
        "",
    )
    if not first_user and not last_assistant:
        return "New Conversation"

    prompt = (
        "Create a descriptive title for this conversation. Output only the title, "
        "using exactly 2 or 3 words. Do not output punctuation, quotes, or a label.\n\n"
        f"User topic:\n{first_user[:1200]}\n\nAssistant response:\n{last_assistant[:1200]}"
    )
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            think=False,
            keep_alive="30m",
            options={"temperature": 0.2, "num_predict": 12},
        )
        message = getattr(response, "message", None)
        content = getattr(message, "content", "")
        if isinstance(response, dict):
            content = response.get("message", {}).get("content", content)
        return _normalize_agent_title(str(content or ""))
    except Exception:
        return "New Conversation"


def title_temporary_session(history: list[dict], filename: str | None = None) -> str | None:
    """Replace a first-turn temporary filename with an agent-generated title."""
    filename = os.path.basename(filename or GLOBAL_STATE.get("active_session_name", ""))
    match = re.fullmatch(r"session_(\d{8}_\d{6}(?:_\d{6})?)\.json", filename)
    if not match:
        return None

    old_path = os.path.join(_SESSIONS_DIR, filename)
    if not os.path.isfile(old_path):
        return None

    title = generate_conversation_title(history)
    safe_title = "_".join(_normalize_agent_title(title).split())
    target = f"{safe_title}_{match.group(1)}.json"
    target_path = os.path.join(_SESSIONS_DIR, target)
    suffix = 2
    while os.path.exists(target_path) and target_path != old_path:
        target = f"{safe_title}_{match.group(1)}_{suffix}.json"
        target_path = os.path.join(_SESSIONS_DIR, target)
        suffix += 1

    os.replace(old_path, target_path)
    if GLOBAL_STATE["active_session_name"] == filename:
        GLOBAL_STATE["active_session_name"] = target
    return target


def load_session(filename: str) -> None:
    """Load session from a JSON file.
    
    Args:
        filename (str): The filename of the session to load.
        
    Raises:
        FileNotFoundError: If the specified session file does not exist.
    """
    filepath = os.path.join(_SESSIONS_DIR, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Session file not found: {filename}")
        
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    GLOBAL_STATE["history"] = data.get("history", [])
    GLOBAL_STATE["session"] = {
        **GLOBAL_STATE["session"],
        **data.get("session", {}),
        "options": {
            **GLOBAL_STATE["session"].get("options", {}),
            **data.get("session", {}).get("options", {}),
        },
    }
    GLOBAL_STATE["active_session_name"] = filename


# ── Slash Command Handler ─────────────────────────────────────────────

_COMMANDS_HELP_MD = """
### Available Commands
* `/help` or `/?` — Show this help message
* `/clear` — Clear conversation history and system prompt override
* `/save [name]` — Save current session (optional name)
* `/load [name|index]` — Load a saved session (lists sessions if no arg)
* `/set parameter <name> <val>` — Set a model parameter (e.g., `temperature 0.7`)
* `/set system "<prompt>"` — Set custom system prompt for this session
* `/set history` / `/set nohistory` — Enable/disable conversation history
* `/set wordwrap` / `/set nowordwrap` — Enable/disable word wrapping
* `/set format json` / `/set noformat` — Enable/disable JSON output format
* `/set verbose` / `/set quiet` — Enable/disable generation stats
* `/set think` / `/set nothink` — Enable/disable model thinking/reasoning
* `/show parameters` — Show current session parameters
* `/show system` — Show the active system prompt
* `/show model` — Show model info
* `/vault list` — List indexed vault collections
* `/vault aliases` — List registered vault aliases
* `/vault alias <name> <coll>` — Register a friendly alias for a collection
* `/vault rename <old> <new>` — Rename a vault collection
* `/vault add <path>` — Add a file or folder to the searchable vault
* `/vault search <query>` — Search the indexed vault
* `/vault delete <source>` — Delete indexed chunks
* `/quit` or `/exit` or `/q` — Quit/exit the session
"""

def execute_command_web(cmd: str, session: dict, history: list[dict]) -> str:
    """Execute a slash command from the web interface.
    
    Similar to the terminal's `_handle_command`, but formats output as markdown
    strings suitable for rendering in the web UI.
    
    Args:
        cmd (str): The slash command string.
        session (dict): The active session state dictionary.
        history (list[dict]): The conversation history list.
        
    Returns:
        str: The markdown-formatted response of the command execution.
    """
    import os
    import json
    import glob
    import shlex
    from datetime import datetime
    from tools.registry import TOOL_DISPATCH
    
    parts = cmd.strip().split(None, 1)
    base = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    
    if base in ("/quit", "/exit", "/q"):
        return "Session quit signal received."
        
    elif base in ("/help", "/?"):
        return _COMMANDS_HELP_MD
        
    elif base == "/clear":
        history.clear()
        session["system"] = ""
        return "✓ Conversation history and system prompt cleared."
        
    elif base == "/save":
        filename = save_session(rest)
        display_name = filename.replace(".json", "")
        return f"✓ Session saved as `{display_name}`"
        
    elif base == "/load":
        saved = list_saved_sessions()
        if not rest:
            if not saved:
                return "No saved sessions found."
            out = ["### Saved Sessions"]
            for i, fp in enumerate(saved, 1):
                name = os.path.basename(fp).replace(".json", "")
                out.append(f"{i}. `{name}`")
            out.append("\nUse `/load <number>` or `/load <name>` to restore.")
            return "\n".join(out)
            
        target_filename = None
        # Try as index first
        try:
            idx = int(rest)
            if 1 <= idx <= len(saved):
                target_filename = os.path.basename(saved[idx - 1])
        except ValueError:
            pass
            
        # Try substring match
        if not target_filename:
            matches = [fp for fp in saved if rest.lower() in os.path.basename(fp).lower()]
            if len(matches) == 1:
                target_filename = os.path.basename(matches[0])
            elif len(matches) > 1:
                out = [f"Multiple sessions match '{rest}':"]
                for fp in matches:
                    out.append(f"- `{os.path.basename(fp).replace('.json', '')}`")
                return "\n".join(out)
                
        if not target_filename:
            return f"No session found matching '{rest}'."
            
        load_session(target_filename)
        return f"✓ Session loaded: `{target_filename.replace('.json', '')}`"
        
    elif base == "/set":
        if not rest:
            return "Usage: `/set <subcommand> [args]` (type `/help` for details)"
        subparts = rest.split(None, 1)
        sub = subparts[0].lower()
        args = subparts[1].strip() if len(subparts) > 1 else ""
        
        if sub == "verbose":
            session["verbose"] = True
            return "✓ Verbose mode enabled — stats shown after each response."
        elif sub == "quiet":
            session["verbose"] = False
            return "✓ Quiet mode enabled."
        elif sub == "wordwrap":
            session["wordwrap"] = True
            return "✓ Word wrapping enabled."
        elif sub == "nowordwrap":
            session["wordwrap"] = False
            return "✓ Word wrapping disabled."
        elif sub == "history":
            session["history"] = True
            return "✓ Conversation history enabled."
        elif sub == "nohistory":
            session["history"] = False
            return "✓ History disabled — each turn is standalone."
        elif sub == "think":
            session["think"] = True
            return "✓ Thinking/reasoning enabled."
        elif sub == "nothink":
            session["think"] = False
            return "✓ Thinking disabled — model will respond directly."
        elif sub == "format":
            if args.lower() == "json":
                session["format"] = "json"
                return "✓ JSON output mode enabled."
            else:
                return f"Unsupported format: `{args}` (supported: json)"
        elif sub == "noformat":
            session["format"] = ""
            return "✓ Output formatting reset to default."
        elif sub == "system":
            if args.startswith('"') and args.endswith('"'):
                args = args[1:-1]
            elif args.startswith("'") and args.endswith("'"):
                args = args[1:-1]
            session["system"] = args
            return f"✓ System prompt set to: {args}" if args else "✓ System prompt reset to default."
        elif sub == "parameter":
            _ALL_PARAMS = {
                "temperature": float,
                "top_p": float,
                "top_k": int,
                "num_predict": int,
                "num_ctx": int,
                "repeat_penalty": float,
                "seed": int,
            }
            subparts = args.split(None, 1)
            if len(subparts) != 2:
                return f"Usage: `/set parameter <name> <value>`\nAvailable: {', '.join(sorted(_ALL_PARAMS.keys()))}"
            name, raw_val = subparts[0].lower(), subparts[1]
            if name not in _ALL_PARAMS:
                return f"Unknown parameter: `{name}`\nAvailable: {', '.join(sorted(_ALL_PARAMS.keys()))}"
            try:
                val = _ALL_PARAMS[name](raw_val)
                if "options" not in session:
                    session["options"] = {}
                session["options"][name] = val
                return f"✓ `{name}` = `{val}`"
            except ValueError:
                expected = _ALL_PARAMS[name].__name__
                return f"Invalid value for {name}: expected {expected}, got '{raw_val}'"
        else:
            return f"Unknown /set subcommand: `{sub}`"
            
    elif base == "/show":
        if not rest:
            return "Usage: `/show <subcommand>` (parameters, system, model)"
        sub = rest.lower()
        if sub == "parameters":
            if "options" not in session or not session["options"]:
                return "No custom parameters set (using model defaults)."
            out = ["**Session parameters:**"]
            for k, v in session["options"].items():
                out.append(f"- `{k}` = `{v}`")
            return "\n".join(out)
        elif sub == "system":
            prompt = session.get("system", "")
            if prompt:
                return f"**System prompt:**\n{prompt}"
            else:
                return "No system prompt set (using Modelfile default)."
        elif sub == "model":
            return f"**Model:** `{MODEL_NAME}`"
        else:
            return f"Unknown /show subcommand: `{sub}` (try: parameters, system, model)"
            
    elif base == "/vault":
        try:
            parts = shlex.split(rest)
        except ValueError as exc:
            return f"Invalid /vault command: {exc}"
            
        if not parts or parts[0].lower() in ("help", "-h", "--help"):
            return """
### Vault Commands
* `/vault list` — List indexed vault collections
* `/vault aliases` — List registered vault aliases
* `/vault alias <name> <coll>` — Register a friendly alias for a collection
* `/vault rename <old> <new>` — Rename a vault collection
* `/vault add <path>` — Add a file or folder to the searchable vault
* `/vault search <query>` — Search the indexed vault
* `/vault delete <source>` — Delete indexed chunks
"""
        sub = parts[0].lower()
        tokens = parts[1:]
        
        def extract_option(tkns, names, default=None):
            for name in names:
                if name in tkns:
                    idx = tkns.index(name)
                    if idx + 1 < len(tkns):
                        val = tkns[idx + 1]
                        tkns.pop(idx + 1)
                        tkns.pop(idx)
                        return val
            return default
            
        collection_raw = extract_option(tokens, ("--collection", "-c"), "vault") or "vault"
        
        try:
            from tools.vault_indexer import resolve_vault_alias
            collection = resolve_vault_alias(collection_raw)
        except ImportError:
            collection = collection_raw
            
        def call_tool(tool_name, **kwargs):
            handler = TOOL_DISPATCH.get(tool_name)
            if not handler:
                return {"error": f"Tool not found: {tool_name}"}
            try:
                result = handler(**kwargs)
                if isinstance(result, str):
                    return json.loads(result)
                return result
            except Exception as exc:
                return {"error": str(exc)}
                
        if sub in ("list", "ls"):
            data = call_tool("list_vaults")
            if "error" in data:
                return f"Vault list failed: {data['error']}"
            vaults = data.get("vaults", [])
            if not vaults:
                return "No indexed vault collections found."
            out = ["### Indexed Vaults"]
            for vault in vaults:
                name = vault.get("collection", "unknown")
                chunk_count = vault.get("indexed_chunks")
                count_text = f"{chunk_count} chunk{'s' if chunk_count != 1 else ''}" if isinstance(chunk_count, int) else "unknown chunks"
                out.append(f"- `{name}` ({count_text})")
            return "\n".join(out)
            
        elif sub == "aliases":
            data = call_tool("list_vault_aliases")
            if "error" in data:
                return f"Failed to list aliases: {data['error']}"
            aliases = data.get("aliases", {})
            if not aliases:
                return "No registered vault aliases found."
            out = ["### Vault Aliases"]
            for name, coll in aliases.items():
                out.append(f"- `{name}` → `{coll}`")
            return "\n".join(out)
            
        elif sub == "alias":
            if len(tokens) < 2:
                return "Usage: `/vault alias <name> <collection>`"
            alias_name, target_coll = tokens[0], tokens[1]
            data = call_tool("register_vault_alias", alias=alias_name, collection=target_coll)
            if "error" in data:
                return f"Failed to register alias: {data['error']}"
            return f"✓ Alias `{alias_name}` registered to collection `{target_coll}`"
            
        elif sub == "rename":
            if len(tokens) < 2:
                return "Usage: `/vault rename <old> <new>`"
            old_name, new_name = tokens[0], tokens[1]
            data = call_tool("rename_vault_collection", old_collection=old_name, new_collection=new_name)
            if "error" in data:
                return f"Failed to rename vault: {data['error']}"
            return f"✓ Vault collection `{old_name}` renamed to `{new_name}`"
            
        elif sub == "add":
            if not tokens:
                return "Usage: `/vault add <path>`"
            path = tokens[0]
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                return f"Path does not exist: `{path}`"
                
            if os.path.isfile(abs_path):
                vault_path = os.path.dirname(abs_path) or "."
                file_path = abs_path
            else:
                vault_path = abs_path
                file_path = None
                
            data = call_tool("index_vault", vault_path=vault_path, file_path=file_path, collection=collection)
            if "error" in data:
                return f"Vault indexing failed: {data['error']}"
                
            indexed = data.get("indexed_files", [])
            out = [f"✓ Vault indexing complete (collection: `{collection}`):"]
            for f in indexed:
                out.append(f"- `{f}`")
            return "\n".join(out)
            
        elif sub == "search":
            if not tokens:
                return "Usage: `/vault search <query> [--top-k n] [--source path]`"
                
            top_k_str = extract_option(tokens, ("--top-k", "-k"), "4")
            source_filter = extract_option(tokens, ("--source", "-s"), None)
            
            try:
                top_k = int(top_k_str)
            except ValueError:
                top_k = 4
                
            query = " ".join(tokens)
            data = call_tool("search_vault", query=query, collection=collection, top_k=top_k, source=source_filter)
            if "error" in data:
                return f"Vault search failed: {data['error']}"
                
            results = data.get("results", [])
            if not results:
                return f"No results found for query `{query}` in vault `{collection}`."
                
            out = [f"### Vault Search Results for '{query}'"]
            for idx, res in enumerate(results, 1):
                src = res.get("source", "unknown")
                score = res.get("score", 0.0)
                text = res.get("text", "")
                snippet = text[:260] + "..." if len(text) > 260 else text
                out.append(f"{idx}. **{src}** (score: {score:.3f})\n>{snippet}\n")
            return "\n".join(out)
            
        elif sub == "delete":
            delete_all = False
            if "--all" in tokens:
                delete_all = True
                tokens.remove("--all")
                
            if not delete_all and not tokens:
                return "Usage: `/vault delete <source> [--collection name]` or `/vault delete --all [--collection name]`"
                
            source = tokens[0] if tokens else None
            data = call_tool("delete_vault", source=source, collection=collection, delete_collection=delete_all)
            if "error" in data:
                return f"Vault delete failed: {data['error']}"
                
            if data.get("deleted_collection"):
                return f"✓ Vault collection deleted: `{collection}`"
            deleted_chunks = data.get("deleted_chunks", 0)
            return f"✓ Vault chunks deleted: {deleted_chunks} chunk(s) (collection: `{collection}`)"
            
        else:
            return f"Unknown /vault subcommand: `{sub}` (try: list, aliases, rename, add, search, delete)"
            
    else:
        return f"Unknown command: `{base}` (type `/help` for available commands)"


# ── Chat Stream Generator ─────────────────────────────────────────────

def generate_chat_events(
    user_input: str,
    session_data: dict,
    history_data: list[dict],
    session_name: str | None = None,
):
    """Generator yielding dictionary objects representing the progress of agent generation.
    
    Supports tool execution and chained follow-up model runs. Yields Server-Sent Events (SSE)
    compatible dictionary objects for streaming status, thinking chunks, content chunks,
    and tool execution updates to the frontend.
    
    Args:
        user_input (str): The raw text submitted by the user.
        session_data (dict): The session configuration and state.
        history_data (list[dict]): The conversation history.
        session_name (str | None): Conversation file that owns this generation.
        
    Yields:
        dict: A dictionary representing an event in the generation process.
    """
    origin_name = session_name or GLOBAL_STATE.get("active_session_name", "Active Session")

    if user_input.startswith('/'):
        output = execute_command_web(user_input, session_data, history_data)
        yield {"type": "content_chunk", "content": output}
        yield {"type": "done"}
        return

    previous_name = origin_name
    origin_name = save_session_snapshot(origin_name, session_data, history_data)
    if GLOBAL_STATE.get("active_session_name") == previous_name:
        GLOBAL_STATE["active_session_name"] = origin_name
    yield {"type": "conversation_started", "session_name": origin_name}

    # Exact, persistently approved routine triggers are deterministic. Do this
    # before asking the model so a saved phrase cannot be overlooked by tool
    # selection, while the routine executor still enforces its safety policy.
    routine_handler = TOOL_DISPATCH.get("automated_routine_executor")
    if routine_handler:
        try:
            preview = json.loads(routine_handler(action="show", trigger=user_input))
        except (TypeError, ValueError, json.JSONDecodeError):
            preview = {}
        if preview.get("automatic_trigger") is True:
            routine_name = str(preview.get("name", "routine"))
            yield {"type": "tool_start", "name": "automated_routine_executor"}
            try:
                result = json.loads(routine_handler(
                    action="run",
                    trigger=user_input,
                    dry_run=False,
                ))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                result = {"ok": False, "error": str(exc)}
            yield {
                "type": "tool_end",
                "name": "automated_routine_executor",
                "result": json.dumps(result, ensure_ascii=False),
            }
            if result.get("ok") is True:
                output = f"✓ Routine **{routine_name}** completed."
            else:
                detail = result.get("error") or "One or more routine actions failed."
                output = f"Routine **{routine_name}** failed: {detail}"

            if session_data.get("history", True):
                history_data.append({"role": "user", "content": user_input})
                history_data.append({"role": "assistant", "content": output})
                origin_name = save_session_snapshot(origin_name, session_data, history_data)
                viewing_origin = GLOBAL_STATE.get("active_session_name") == origin_name
                titled_name = title_temporary_session(history_data, origin_name)
                if titled_name:
                    origin_name = titled_name
                save_session_snapshot(origin_name, session_data, history_data)
                if viewing_origin:
                    GLOBAL_STATE["active_session_name"] = origin_name
                    GLOBAL_STATE["session"] = session_data
                    GLOBAL_STATE["history"] = history_data
            yield {"type": "content_chunk", "content": output}
            yield {
                "type": "done",
                "history": history_data,
                "active_session_name": origin_name,
                "saved_sessions": list_saved_sessions(),
            }
            return
    # 1. Sync system prompt override
    default_system_prompt = ""
    try:
        import subprocess
        res = subprocess.run(["ollama", "show", MODEL_NAME, "--system"], capture_output=True, text=True)
        if res.returncode == 0:
            default_system_prompt = res.stdout.strip()
    except Exception:
        pass
        
    active_system = session_data.get("system") or default_system_prompt
    if active_system:
        if not history_data or history_data[0].get("role") != "system" or history_data[0].get("content") != active_system:
            history_data[:] = [m for m in history_data if m.get("role") != "system"]
            history_data.insert(0, {"role": "system", "content": active_system})
    else:
        history_data[:] = [m for m in history_data if m.get("role") != "system"]
        
    # 2. Check for local file auto-indexing
    pre_tool_message = None
    try:
        if os.path.exists(user_input) and os.path.isfile(user_input):
            size = os.path.getsize(user_input)
            ext = os.path.splitext(user_input)[1].lower()
            INDEX_THRESHOLD = 200_000
            if size > INDEX_THRESHOLD or ext in (".pdf", ".docx"):
                yield {"type": "status", "message": f"Large/binary file detected — indexing {user_input}...", "color": "yellow"}
                handler = TOOL_DISPATCH.get("index_vault")
                if handler:
                    res = handler(vault_path=os.path.dirname(user_input) or ".", file_path=user_input)
                    if isinstance(res, str):
                        tool_content = res
                    else:
                        tool_content = json.dumps(res)
                    tool_msg = {"role": "tool", "content": tool_content}
                    if session_data.get("history", True):
                        history_data.append(tool_msg)
                    else:
                        pre_tool_message = tool_msg
                    yield {"type": "status", "message": "Indexing complete.", "color": "green"}
    except Exception as e:
        yield {"type": "status", "message": f"Indexing failed: {e}", "color": "red"}
        
    # 3. Build messages to send
    if session_data.get("history", True):
        history_data.append({"role": "user", "content": user_input})
        # Persist the prompt immediately so stopping a generation cannot lose it.
        save_session_snapshot(origin_name, session_data, history_data)
        messages_to_send = _trim_history(history_data)
    else:
        messages_to_send = []
        if history_data and history_data[0].get("role") == "system":
            messages_to_send.append(history_data[0])
        if pre_tool_message:
            messages_to_send.append(pre_tool_message)
        messages_to_send.append({"role": "user", "content": user_input})
        
    # 4. Stream response loop (supports tool execution and model chain-calling)
    while True:
        kwargs = {
            "model": MODEL_NAME,
            "messages": messages_to_send,
            "stream": True,
            "think": session_data.get("think", True),
            "keep_alive": "30m",
        }
        if session_data.get("format"):
            kwargs["format"] = session_data["format"]
        if TOOL_SCHEMAS:
            kwargs["tools"] = TOOL_SCHEMAS
        if session_data.get("options"):
            kwargs["options"] = session_data["options"]
            
        try:
            stream = ollama.chat(**kwargs)
        except Exception as e:
            yield {"type": "status", "message": f"Ollama Chat error: {e}", "color": "red"}
            break
            
        thinking_buf = ""
        content_buf = ""
        tool_calls = []
        in_thinking = False
        thinking_started = False
        prompt_tokens = 0
        eval_tokens = 0
        
        for chunk in stream:
            msg = chunk.message
            
            if getattr(chunk, "prompt_eval_count", None):
                prompt_tokens = chunk.prompt_eval_count
            if getattr(chunk, "eval_count", None):
                eval_tokens = chunk.eval_count
            
            # Intercept tool calls
            if getattr(msg, "tool_calls", None):
                tool_calls = [
                    {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
                break
                
            # Stream thinking text
            thinking_chunk = getattr(msg, "thinking", None) or ""
            if thinking_chunk:
                if not thinking_started:
                    thinking_started = True
                    in_thinking = True
                    yield {"type": "thinking_start"}
                thinking_buf += thinking_chunk
                yield {"type": "thinking_chunk", "text": thinking_chunk}
                continue
                
            # Stream final content text
            content_chunk = getattr(msg, "content", None) or ""
            if content_chunk:
                if in_thinking:
                    in_thinking = False
                    yield {"type": "thinking_end"}
                content_buf += content_chunk
                yield {"type": "content_chunk", "text": content_chunk}
                
        if in_thinking:
            yield {"type": "thinking_end"}
            
        # Send token usage if available
        if prompt_tokens or eval_tokens:
            yield {
                "type": "token_usage", 
                "total": prompt_tokens + eval_tokens,
                "budget": int(session_data.get("options", {}).get("num_ctx", 8192))
            }
            
        # Compile assistant message
        assistant_msg = {"role": "assistant", "content": content_buf}
        if thinking_buf:
            assistant_msg["thinking"] = thinking_buf
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
            
        if session_data.get("history", True):
            history_data.append(assistant_msg)
            
        # If there are no tool calls, this turn is completed
        if not tool_calls:
            viewing_origin = GLOBAL_STATE.get("active_session_name") == origin_name
            titled_name = title_temporary_session(history_data, origin_name)
            if titled_name:
                origin_name = titled_name
            save_session_snapshot(origin_name, session_data, history_data)
            if viewing_origin:
                GLOBAL_STATE["active_session_name"] = origin_name
                GLOBAL_STATE["session"] = session_data
                GLOBAL_STATE["history"] = history_data
            yield {
                "type": "done",
                "history": history_data,
                "active_session_name": origin_name,
                "saved_sessions": list_saved_sessions(),
            }
            break
            
        # Execute tool calls
        yield {"type": "tool_calls_start", "calls": tool_calls}
        
        tool_results = []
        for tc in tool_calls:
            fn = tc["function"]
            fn_name = fn["name"]
            fn_args = fn["arguments"]
            
            yield {"type": "tool_start", "name": fn_name, "arguments": fn_args}
            
            handler = TOOL_DISPATCH.get(fn_name)
            if not handler:
                result_str = f"Error: Tool {fn_name} not found in registry."
            else:
                try:
                    res = handler(**fn_args)
                    if isinstance(res, str):
                        result_str = res
                    else:
                        result_str = json.dumps(res)
                except Exception as e:
                    result_str = f"Error: {e}"
                    
            yield {"type": "tool_end", "name": fn_name, "result": result_str}
            
            tool_results.append({
                "role": "tool",
                "tool_name": fn_name,
                "content": result_str
            })
            
        if session_data.get("history", True):
            history_data.extend(tool_results)
            num_ctx = session_data.get("options", {}).get("num_ctx", 8192)
            reminder = {
                "role": "user",
                "content": (
                    "Continue the current turn using the tool result above. Answer this "
                    "original request directly; do not ask the user to repeat it:\n"
                    f"{user_input}"
                ),
            }
            messages_to_send = _trim_history([*history_data, reminder], num_ctx)
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


# ── HTTP Handler ──────────────────────────────────────────────────────

class AgentHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """Custom HTTP request handler for the agent's web interface.
    
    Handles serving static assets, managing session state via REST API,
    and providing an SSE endpoint for streaming chat generation.
    """
    
    def log_message(self, format, *args):
        """Mute standard output logs to keep the server output clean."""
        pass

    def send_json_response(self, status_code: int, data: dict):
        """Helper to send a JSON-encoded HTTP response.
        
        Args:
            status_code (int): The HTTP status code to return.
            data (dict): The dictionary to serialize to JSON.
        """
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def read_json_body(self) -> dict:
        """Helper to read and parse the JSON body of an HTTP POST request.
        
        Returns:
            dict: The parsed JSON body.
        """
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        return json.loads(body.decode('utf-8'))

    def serve_static_file(self, filename: str, content_type: str):
        """Serve a static file from the STATIC_DIR.
        
        Args:
            filename (str): The name of the file to serve.
            content_type (str): The MIME type of the file.
        """
        filepath = os.path.join(STATIC_DIR, filename)
        if not os.path.isfile(filepath):
            self.send_error(404, "File Not Found")
            return
            
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)
        except OSError:
            self.send_error(500, "Internal Server Error")

    def do_GET(self):
        """Handle incoming HTTP GET requests for static files and settings."""
        # 1. Routing for Home and Assets
        if self.path == '/' or self.path == '/index.html':
            self.serve_static_file('index.html', 'text/html')
            return
        elif self.path == '/style.css':
            self.serve_static_file('style.css', 'text/css')
            return
        elif self.path == '/app.js':
            self.serve_static_file('app.js', 'application/javascript')
            return
            
        # 2. Routing for Settings/State load
        elif self.path == '/api/settings':
            saved = list_saved_sessions()
            
            ollama_status = "Online"
            try:
                ollama.list()
            except Exception:
                ollama_status = "Offline"
                
            response_data = {
                "settings": GLOBAL_STATE["session"],
                "history": GLOBAL_STATE["history"],
                "saved_sessions": saved,
                "active_session_name": GLOBAL_STATE["active_session_name"],
                "model_name": MODEL_NAME,
                "ollama_status": ollama_status
            }
            self.send_json_response(200, response_data)
            return
            
        elif self.path == '/favicon.ico' or self.path == '/favicon.png':
            self.serve_static_file('favicon.png', 'image/png')
            return
            
        elif self.path == '/avatar.png':
            self.serve_static_file('avatar.png', 'image/jpeg')
            return
            
        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """Handle incoming HTTP POST requests for API endpoints (chat, save/load/clear session)."""
        # 1. Save Settings
        if self.path == '/api/settings':
            try:
                body = self.read_json_body()
                GLOBAL_STATE["session"] = body
                autosave_session()
                self.send_json_response(200, {"status": "success", "settings": GLOBAL_STATE["session"]})
            except Exception as e:
                self.send_json_response(400, {"status": "error", "error": str(e)})
            return
            
        # 2. Chat SSE Stream
        elif self.path == '/api/chat':
            try:
                body = self.read_json_body()
                user_input = body.get("message", "").strip()
                # Pin this request to the conversation that initiated it. The
                # user may select another chat while this thread is streaming.
                requested_name = os.path.basename(str(body.get("session_name", "")))
                current_name = GLOBAL_STATE["active_session_name"]
                aliases = {"", "Active Session", "New conversation"}
                if requested_name == current_name or ({requested_name, current_name} <= aliases):
                    generation_session_name = current_name
                    generation_session = GLOBAL_STATE["session"]
                    generation_history = GLOBAL_STATE["history"]
                else:
                    requested_path = os.path.join(_SESSIONS_DIR, requested_name)
                    if not requested_name.endswith(".json") or not os.path.isfile(requested_path):
                        raise FileNotFoundError("The originating conversation no longer exists")
                    with open(requested_path, "r", encoding="utf-8") as stream:
                        requested_session = json.load(stream)
                    generation_session_name = requested_name
                    generation_session = requested_session.get("session", {})
                    generation_history = requested_session.get("history", [])
            except Exception as e:
                self.send_json_response(400, {"error": "Invalid JSON body"})
                return
                
            if not user_input:
                self.send_json_response(400, {"error": "Message cannot be empty"})
                return
                
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            try:
                for event in generate_chat_events(
                    user_input,
                    generation_session,
                    generation_history,
                    generation_session_name,
                ):
                    data_line = f"data: {json.dumps(event)}\n\n"
                    self.wfile.write(data_line.encode('utf-8'))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
            
        # 3. Save Session
        elif self.path == '/api/save-session':
            try:
                body = self.read_json_body()
                name = body.get("name", "").strip()
                filename = save_session(name)
                self.send_json_response(200, {"status": "success", "filename": filename})
            except Exception as e:
                self.send_json_response(500, {"status": "error", "error": str(e)})
            return
            
        # 4. Load Session
        elif self.path == '/api/load-session':
            try:
                body = self.read_json_body()
                name = body.get("name", "").strip()
                load_session(name)
                self.send_json_response(200, {"status": "success"})
            except Exception as e:
                self.send_json_response(500, {"status": "error", "error": str(e)})
            return

        # 5. Start a fresh session after safely persisting the current one
        elif self.path == '/api/new-session':
            autosave_session()
            GLOBAL_STATE["history"] = []
            GLOBAL_STATE["active_session_name"] = "Active Session"
            self.send_json_response(200, {
                "status": "success",
                "saved_sessions": list_saved_sessions(),
            })
            return

        # 6. Permanently delete a saved conversation
        elif self.path == '/api/delete-session':
            try:
                body = self.read_json_body()
                name = os.path.basename(body.get("name", ""))
                if not name or name not in list_saved_sessions():
                    raise FileNotFoundError("Session not found")
                os.remove(os.path.join(_SESSIONS_DIR, name))
                if GLOBAL_STATE["active_session_name"] == name:
                    GLOBAL_STATE["history"] = []
                    GLOBAL_STATE["active_session_name"] = "Active Session"
                self.send_json_response(200, {
                    "status": "success",
                    "saved_sessions": list_saved_sessions(),
                    "active_session_name": GLOBAL_STATE["active_session_name"],
                })
            except Exception as e:
                self.send_json_response(404, {"status": "error", "error": str(e)})
            return
            
        # 7. Clear Session history
        elif self.path == '/api/clear-session':
            GLOBAL_STATE["history"].clear()
            GLOBAL_STATE["session"]["system"] = ""
            GLOBAL_STATE["active_session_name"] = "Active Session"
            self.send_json_response(200, {"status": "success"})
            return
            
        else:
            self.send_error(404, "Not Found")


# ── Threaded HTTP Server ──────────────────────────────────────────────

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """A threading version of the standard HTTPServer to handle concurrent requests."""
    daemon_threads = True


def find_free_port() -> int:
    """Find and return an available port on the local system."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_web_server():
    """Starts the multi-threaded web server and opens the browser.
    
    Attempts to bind to port 5005. If unavailable, falls back to a random free port.
    It will also launch the default web browser automatically.
    """
    # Keep the user's latest UI/model preferences without reopening its chat.
    # Every application launch starts with a blank conversation; previous
    # conversations remain autosaved and can still be opened from the sidebar.
    saved_sessions = list_saved_sessions()
    if saved_sessions:
        try:
            filepath = os.path.join(_SESSIONS_DIR, saved_sessions[0])
            with open(filepath, "r", encoding="utf-8") as stream:
                saved_session = json.load(stream).get("session", {})
            GLOBAL_STATE["session"] = {
                **GLOBAL_STATE["session"],
                **saved_session,
                "options": {
                    **GLOBAL_STATE["session"].get("options", {}),
                    **saved_session.get("options", {}),
                },
            }
        except (OSError, AttributeError, TypeError, json.JSONDecodeError):
            pass
    GLOBAL_STATE["history"] = []
    GLOBAL_STATE["active_session_name"] = "Active Session"

    # Attempt to bind to default port 5005 first, then fall back to random port
    host = '127.0.0.1'
    if '--public' in sys.argv:
        host = '0.0.0.0'
        
    try:
        server = ThreadingHTTPServer((host, 5005), AgentHTTPRequestHandler)
        port = 5005
    except OSError:
        port = find_free_port()
        server = ThreadingHTTPServer((host, port), AgentHTTPRequestHandler)
        
    url = f"http://127.0.0.1:{port}"
    print(f"\n🚀 Starting Selene Web Interface at {url}")
    print(f"ELECTRON_PORT:{port}", flush=True)
    
    if host == '0.0.0.0':
        local_ip = "127.0.0.1"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass
        print(f"📡 Accessible across your local network at: http://{local_ip}:{port}")
        
    if "--no-browser" not in sys.argv:
        print(f"Opening default web browser...\n")
        
        def open_browser():
            time.sleep(0.5)
            webbrowser.open(url)
            
        threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server...")
        server.server_close()
