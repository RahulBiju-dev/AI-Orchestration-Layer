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
import webbrowser
import threading
from datetime import datetime, timezone
import ollama

# Import agent configurations and helpers from core
from agent.core import MODEL_NAME, _trim_history
from tools.registry import TOOL_DISPATCH, TOOL_SCHEMAS

# Setup directories
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sessions")

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
    """Return a sorted list of session filenames (newest first)."""
    if not os.path.isdir(_SESSIONS_DIR):
        return []
    files = glob.glob(os.path.join(_SESSIONS_DIR, "*.json"))
    files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.basename(f) for f in files]


def save_session(name: str) -> str:
    """Persist current state to a JSON file."""
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


def load_session(filename: str) -> None:
    """Load session from a JSON file."""
    filepath = os.path.join(_SESSIONS_DIR, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Session file not found: {filename}")
        
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    GLOBAL_STATE["session"] = data.get("session", {})
    GLOBAL_STATE["history"] = data.get("history", [])
    GLOBAL_STATE["active_session_name"] = filename


# ── Chat Stream Generator ─────────────────────────────────────────────

def generate_chat_events(user_input: str, session_data: dict, history_data: list[dict]):
    """
    Generator yielding dictionary objects representing the progress of agent generation.
    Supports tool execution and chained follow-up model runs.
    """
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
        
        for chunk in stream:
            msg = chunk.message
            
            # Intercept tool calls
            if msg.tool_calls:
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
            content_chunk = msg.content or ""
            if content_chunk:
                if in_thinking:
                    in_thinking = False
                    yield {"type": "thinking_end"}
                content_buf += content_chunk
                yield {"type": "content_chunk", "text": content_chunk}
                
        if in_thinking:
            yield {"type": "thinking_end"}
            
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
            yield {"type": "done", "history": history_data}
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
                "content": result_str
            })
            
        if session_data.get("history", True):
            history_data.extend(tool_results)
            messages_to_send = _trim_history(history_data)
        else:
            messages_to_send.append(assistant_msg)
            messages_to_send.extend(tool_results)


# ── HTTP Handler ──────────────────────────────────────────────────────

class AgentHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        # Mute standard output logs to keep the server output clean
        pass

    def send_json_response(self, status_code: int, data: dict):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def read_json_body(self) -> dict:
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        return json.loads(body.decode('utf-8'))

    def serve_static_file(self, filename: str, content_type: str):
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
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)
        except OSError:
            self.send_error(500, "Internal Server Error")

    def do_GET(self):
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
            
        elif self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
            return
            
        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        # 1. Save Settings
        if self.path == '/api/settings':
            try:
                body = self.read_json_body()
                GLOBAL_STATE["session"] = body
                self.send_json_response(200, {"status": "success", "settings": GLOBAL_STATE["session"]})
            except Exception as e:
                self.send_json_response(400, {"status": "error", "error": str(e)})
            return
            
        # 2. Chat SSE Stream
        elif self.path == '/api/chat':
            try:
                body = self.read_json_body()
                user_input = body.get("message", "").strip()
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
                for event in generate_chat_events(user_input, GLOBAL_STATE["session"], GLOBAL_STATE["history"]):
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
            
        # 5. Clear Session history
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
    daemon_threads = True


def find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_web_server():
    """Starts the multi-threaded web server and opens the browser."""
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
