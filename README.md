# AI CLI Agent

A modular, tool-augmented local AI agent built with Python and [Ollama](https://ollama.com/). Runs entirely on your machine — no cloud APIs, no telemetry, no subscriptions. The agent wraps a customised [Gemma 4](https://ai.google.dev/gemma) model with an autonomous tool-calling loop, real-time streaming output, and a persistent RAG (Retrieval-Augmented Generation) vault for long-term document memory.

By default the agent launches a **modern browser-based Web UI** with live streaming, collapsible thinking panels, and an interactive sidebar. The classic terminal interface is still available via `--cli`.

---

## Table of Contents

- [How It Works — Theory](#how-it-works--theory)
  - [The Agentic Loop](#the-agentic-loop)
  - [Tool Calling](#tool-calling)
  - [Streaming Inference](#streaming-inference)
  - [RAG Vault](#rag-vault--retrieval-augmented-generation)
  - [Context Window Management](#context-window-management)
- [Features](#features)
  - [Web UI](#web-ui)
  - [Terminal Interface](#terminal-interface)
  - [Tool Suite](#tool-suite)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Web UI (Default)](#web-ui-default)
  - [Terminal CLI](#terminal-cli)
  - [Slash Commands](#slash-commands)
  - [Vault Commands](#vault-commands)
  - [Runtime Configuration](#runtime-configuration)
- [Performance Tuning](#performance-tuning)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## How It Works — Theory

### The Agentic Loop

Traditional chatbots are stateless request-response pipes: you send a prompt, the model returns text, done. An **agent** adds a decision loop on top. After generating a response, the agent inspects it for **tool-call signals** — structured instructions the model emits when it determines it needs external data or side-effects to answer properly. If tool calls are detected, the agent:

1. Executes each tool (web search, file read, Spotify playback, etc.)
2. Injects the tool results back into the conversation as `tool` role messages
3. Calls the model **again** with the augmented context
4. Repeats until the model produces a final text-only response

This creates an **iterative refinement loop** where the model can chain multiple tools before composing its answer. For example, if asked *"What's the latest Python release and how does its new feature compare to Rust's approach?"*, the agent might:
- Call `web_search` for the Python release
- Call `web_search` again for the Rust comparison
- Synthesise both results into a single coherent answer

```
User Prompt ──→ LLM ──→ Tool Calls? ──Yes──→ Execute Tools ──→ Inject Results ──→ LLM ──┐
                              │                                                         │
                              No                                                        │
                              │                                                         │
                              ▼                                                         │
                        Stream Answer ◀────────────────────────────────────────────────┘
```

### Tool Calling

The agent uses the **OpenAI-compatible function calling** format supported by Ollama. Each tool is defined as a JSON schema that describes its name, purpose, and parameters. These schemas are sent to the model alongside every prompt, enabling the model to "know" what tools are available and how to invoke them.

When the model decides a tool is needed, it emits a structured JSON object instead of text:

```json
{
  "function": {
    "name": "web_search",
    "arguments": {"query": "Python 3.14 release date"}
  }
}
```

The agent intercepts this, dispatches to the corresponding Python function via a **dispatch table** (`TOOL_DISPATCH`), and feeds the return value back to the model as a `tool` role message. The model never executes code directly — it only emits structured requests that the agent mediates.

**Why this matters:** The model's training data has a knowledge cutoff. Tool calling allows it to bridge that gap with real-time data, local filesystem access, and system integration — all while keeping execution sandboxed in Python handlers.

### Streaming Inference

LLM inference happens in two phases:

1. **Prefill (prompt evaluation):** The model processes all input tokens in parallel. Cost is proportional to `num_ctx` (context window size) × number of input tokens. This is the "thinking" phase.

2. **Decode (token generation):** The model generates output tokens one at a time, each conditioned on all previous tokens via the **KV cache** (a matrix of key-value attention states). This is the bottleneck for tokens-per-second (tok/s).

The agent streams both phases to the terminal in real-time:

- **Thinking tokens** (from Gemma's `thinking` field) are displayed in dim magenta as the model reasons through the problem
- **Content tokens** are accumulated into a buffer and rendered as rich Markdown using `rich.Live`

To avoid CPU overhead from re-rendering the entire Markdown buffer on every token, the renderer is **throttled** to approximately 12 FPS (~80ms intervals). Tokens still accumulate in the buffer at full speed — only the visual update is debounced.

### RAG Vault — Retrieval-Augmented Generation

The vault system implements a local RAG pipeline for long-term document memory:

```
Documents ──→ Chunk ──→ Embed ──→ ChromaDB
                                      │
                          Query ──→ Embed ──→ Vector Search ──→ Top-K Chunks ──→ LLM Context
```

**How RAG works:**

1. **Chunking:** Documents (PDF, DOCX, Markdown, reStructuredText, plain text) are split into overlapping segments of ~1800 characters. The splitter prefers natural boundaries (paragraphs, sentences) over hard character cuts, producing more coherent retrieval snippets.

2. **Embedding:** Each chunk is converted into a dense vector (a list of floating-point numbers) using an embedding model (`embeddinggemma` by default, running locally via Ollama). This vector captures the *semantic meaning* of the text — chunks about similar topics will have vectors that are close together in the embedding space, regardless of exact wording.

3. **Storage:** Vectors and their source metadata (file path, chunk index, character offsets) are stored in [ChromaDB](https://www.trychroma.com/), a persistent vector database that lives in the project's `.chroma/` directory.

4. **Retrieval:** When the agent (or user via `/vault search`) queries the vault, the query text is embedded with the same model, and ChromaDB performs an **approximate nearest neighbour (ANN) search** to find the top-K most semantically similar chunks.

5. **Augmentation:** Retrieved chunks are injected into the model's context window alongside the user's question, giving the model grounded, relevant information to draw from.

**Why local RAG?** Unlike cloud RAG services, everything runs on your machine. Your documents never leave your filesystem. The embedding model and vector database are both local — no API keys, no data exfiltration risks.

### Context Window Management

The **context window** (`num_ctx`) is the maximum number of tokens the model can "see" at once — both input and output combined. It's the model's working memory. Larger windows let the model consider more conversation history but come with costs:

- **KV cache memory** scales linearly with context length. For an 8B parameter model at Q4 quantisation, each additional 1K of context costs roughly 32-64MB of memory.
- **Prefill latency** increases with more input tokens.
- If the KV cache exceeds GPU VRAM, it spills to system RAM, causing a dramatic throughput drop.

The agent manages this automatically:

- **Default `num_ctx` is 8192** — sized to fit the model + KV cache entirely in GPU VRAM on consumer GPUs (4-8GB).
- **History trimming and compaction** keep the prompt within the active token budget. Near 90% usage, older turns are summarized and passed through the context optimizer while system instructions and recent exchanges remain intact; hard trimming remains the final bound.
- Both values can be overridden at runtime via `/set parameter num_ctx <value>`.

---

## Features

### Core
- **Fully local:** All inference runs through Ollama on your machine. No cloud dependencies for core functionality.
- **Custom model:** Uses a `Modelfile` to wrap Gemma 4 (8B, Q4_K_M quantisation) with a tailored system prompt and optimised sampling parameters.
- **Thinking visibility:** Streams the model's internal chain-of-thought reasoning before the final answer.
- **Dual interface:** Launches the Web UI by default; pass `--cli` for the classic terminal experience.

### Web UI

The default interface — launch with `python main.py` and the agent opens in your browser automatically.

- **Cyberpunk-Obsidian aesthetic** — deep charcoal backgrounds, glassmorphism cards, and neon glowing accents in Cyan, Magenta, Teal, and Amber.
- **Enhanced 3D Elements** — realistic layered shadows (`--shadow-subtle`, `--shadow-heavy`) and tactile hover/active states that simulate physical lift for cards and message bubbles.
- **Context window usage indicators** — visual tracking of the model's context capacity in real time.
- **Live SSE streaming** — tokens and thinking blocks are pushed to the browser in real-time via Server-Sent Events; no polling, no page reloads.
- **Smart generation states** — dynamic site behaviour that intelligently adapts while a response is actively generating.
- **Collapsible thinking panel** — while the model reasons, a dedicated magenta panel shows the chain-of-thought with animated dots. After completion it collapses into a togglable bar (DeepSeek/Gemini style).
- **Interactive tool cards** — each tool invocation renders a visual card: `⟳ Running [tool]` → `✓ Executed [tool]`. Click the header to expand raw JSON parameters and output.
- **Sidebar control panel:**
  - Real-time sliders for `Temperature`, `Top-P`, and `Top-K`
  - System-prompt override with a one-click reset to default
  - Toggles for conversation history and model thinking
  - Save / restore named sessions without leaving the browser
- **Markdown & code highlighting** — responses are rendered with `marked.js` and syntax-highlighted with `highlight.js`.
- **Responsive layout** — sidebar collapses on narrow viewports; works on desktop and tablet.

### Tool Suite
The agent autonomously decides when to call tools based on the user's query:

| Tool | Description |
|------|-------------|
| 🔍 **Web Search** | Real-time DuckDuckGo search with adaptive depth (easy/medium/hard) for current events, docs, and post-cutoff information |
| 🌐 **Browser** | Open URLs or search queries in the system's default browser |
| 💻 **Code Viewer** | Read source files with line numbers; scan directories by extension |
| 📄 **Document Reader** | Extract text from PDFs (`pypdf`) and Word docs (`python-docx`) with page/chunk/query navigation |
| 📂 **File Manager** | Read text files with line range and search controls; create new files (auto-vaulted to `vaults/`) |
| 🎵 **Spotify** | Search and play songs natively on Windows, macOS, and Linux |
| 👁️ **Vision Describer** | Describes images, diagrams, and slides using the local `moondream` vision model |
| 🗄️ **Vault Index** | Chunk and embed local files into ChromaDB for semantic search; auto-registers aliases |
| 🔎 **Vault Search** | Query the vault using vector similarity; resolves friendly aliases automatically |
| 🗑️ **Vault Delete** | Remove indexed entries by source path or delete entire collections |
| 🏷️ **Vault Aliases** | List registered human-friendly names that map to vault collections |
| 📓 **Obsidian Notes** | Create structured Obsidian-optimised notes with YAML frontmatter, WikiLinks, and version control |
| 🕸️ **Knowledge Graph Builder** | Map typed relationships and discover evidence-traceable causal paths, conflicts, central concepts, and feedback cycles |
| 📈 **Simulation Runner** | Execute recurrence, Euler, scenario, and Monte Carlo models with deterministic seeds and distribution summaries |
| 🔌 **API Orchestrator** | Manage API auth refresh, bounded retries, deprecation signals, response limits, and endpoint failover |
| 🧠 **Context Memory Optimizer** | Compact conversations while preserving instructions, recent turns, decisions, constraints, facts, and links |
| 🧭 **Reasoning Chain Debugger** | Audit explicit claim/evidence graphs for unsupported leaps, missing references, cycles, and confidence problems |
| ⚙️ **Automated Routine Executor** | Define natural-language workflow macros, preview their actions, and execute approved local commands/apps/URLs |

### Advanced Tool Safety Model

The advanced tools are deliberately bounded:

- Graph inferences include the exact supporting edge path; the builder does not invent edges from labels.
- Simulation equations use a restricted arithmetic parser—never Python `eval`—and workloads are capped. Forecasts remain conditional on the supplied assumptions.
- API credentials are referenced by environment-variable name rather than passed as literal secrets. Retries, timeouts, response sizes, and failover endpoints are capped.
- Memory optimisation is extractive and reports before/after token estimates. Automatic background compaction uses the same optimizer after generating its factual summary.
- The reasoning debugger audits supplied claims, dependencies, assumptions, and evidence IDs. It does not expose private model chain-of-thought; it produces an accountable evidence graph and Mermaid diagram.
- Routines live in gitignored `.selene/routines.json`. Runs default to a dry-run preview and require `dry_run=false` plus `confirmed=true` after user approval. Commands use argument arrays with `shell=False` and remain in the project workspace.

Example simulation model:

```json
{
  "variables": {"inventory": 100, "demand": 12},
  "equations": {
    "inventory": "max(0, inventory - demand)",
    "demand": "max(0, demand + normal(0, 1.5))"
  },
  "steps": 30,
  "trials": 200,
  "seed": 42
}
```

Example routine definition:

```json
{
  "action": "define",
  "name": "morning workspace",
  "routine": {
    "triggers": ["start my morning"],
    "actions": [
      {"type": "open_app", "argv": ["code", "."]},
      {"type": "open_url", "url": "https://example.com/dashboard"},
      {"type": "command", "argv": ["git", "status", "--short"]}
    ]
  }
}
```

### Terminal Interface
- **Rich Markdown streaming** via `rich.Live` with automatic scroll management
- **LaTeX math rendering** — Greek letters, fractions (`\frac`), roots (`\sqrt`), super/subscripts, arrays, and 220+ symbol mappings converted to Unicode for terminal display
- **Animated spinner** during model loading and thinking phases
- **Session persistence** — save and restore full conversation state including history, parameters, and system prompts
- **Graceful Interrupts** — use `Ctrl+\` to safely stop the model's generation midway while preserving the partial response in your conversation context (leaving `Ctrl+C` free to exit the application).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.py                                │
│          (entry point — model init, CLI/Web routing)            │
└────────────────────┬──────────────────────┬─────────────────────┘
                     │  --cli flag          │  default
                     ▼                      ▼
        ┌────────────────────┐   ┌─────────────────────────────┐
        │   agent/core.py    │   │       agent/web.py           │
        │  (Terminal CLI)    │   │  (Threaded HTTP + SSE)       │
        │  Chat Loop         │   │  /api/chat   → SSE stream    │
        │  /commands         │   │  /api/session/*  → JSON      │
        │  Streaming         │   │  /static/*   → HTML/CSS/JS   │
        │  Session Mgmt      │   └──────────────┬──────────────┘
        └────────┬───────────┘                  │
                 │                              │
                 └──────────────┬───────────────┘
                                ▼
               ┌────────────────────────────────┐
               │         tools/registry          │
               │  TOOL_SCHEMAS   TOOL_DISPATCH   │
               └────────────────┬───────────────┘
                                │
     ┌──────┬──────┬────────┬───┴──┬─────────┬───────────┐
     ▼      ▼      ▼        ▼      ▼         ▼           ▼
  search  browser  file  document spotify   vision     vault
  .py     .py      .py   .py      .py    describer  (index/search/
                                          .py        embeddings)
                                    obsi_vault          │
                                    _writer.py           ▼
                                                  ┌──────────┐
                                                  │ ChromaDB │
                                                  │ (.chroma)│
                                                  └──────────┘

  advanced tools: knowledge graph · simulation · API orchestration
                  context memory · reasoning audit · routine macros

 Browser (Web UI)
 ┌──────────────────────────────────────────────┐
 │  index.html  +  style.css  +  app.js          │
 │  ┌──────────┐  ┌─────────────────────────┐   │
 │  │ Sidebar  │  │  Chat Panel              │   │
 │  │ Sliders  │  │  SSE token stream        │   │
 │  │ Toggles  │  │  Thinking block          │   │
 │  │ Sessions │  │  Tool cards              │   │
 │  └──────────┘  └─────────────────────────┘   │
 └──────────────────────────────────────────────┘
```

### Data Flow — Web UI

1. Browser opens automatically at `http://localhost:5005` (or next free port)
2. User message is `POST`ed to `/api/chat`
3. `web.py` builds the Ollama request with the current session's parameters, history, and tool schemas
4. If the model returns tool calls, `web.py` executes them via `TOOL_DISPATCH` and yields SSE `tool_*` events to the browser
5. Thinking tokens arrive as `thinking` SSE events; content tokens as `token` events
6. `app.js` renders tokens in real time, assembles the thinking panel, and builds tool cards
7. On completion a `done` SSE event is fired and the response is added to history

### Data Flow — Terminal CLI

1. **User input** enters the chat loop in `agent/core.py`
2. Slash commands (`/help`, `/save`, `/vault`, etc.) are intercepted and handled locally — they never touch the LLM
3. Natural language input is sent to Ollama with the full tool schema list and (trimmed) conversation history
4. If the model returns tool calls, the dispatch table routes them to the appropriate Python handler
5. Tool results are appended to the conversation and the model is called again
6. Final text output is streamed through the terminal renderer with Markdown and LaTeX processing

---

## Prerequisites

- **[Ollama](https://ollama.com/)** installed and running (`ollama serve`)
- **Python 3.10+**
- **Gemma 4 E4B model:** `ollama pull gemma4:e4b`
- **Embedding model (for vault):** `ollama pull embeddinggemma`
- **For Spotify:** Spotify desktop app. On Linux, `dbus-python` is also required (pre-installed on most GNOME/Fedora systems).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/RahulBiju-dev/AI-CLI-Agent.git
cd AI-CLI-Agent

# Install system dependencies (for PDF vision support)
sudo dnf install poppler-utils -y

# Install Python dependencies
pip install -r requirements.txt

# Ensure Ollama has the required models
ollama pull gemma4:e4b
ollama pull embeddinggemma
ollama pull moondream

# Start the agent (auto-builds the custom model on first run)
python main.py
```

### Multimodal Vision Capabilities
The agent supports memory-safe multimodal vision, allowing it to read slides, diagrams, and architectures from large PDFs without RAM exhaustion.
> **Note:** You MUST run `ollama pull moondream` in your terminal before using the agent with PDFs or images to enable this feature!

### What happens on first run

The agent calls `ollama create selene -f Modelfile` to build a custom model variant that bundles:
- The Gemma 4 base weights
- A system prompt with personality, knowledge cutoff rules, and tool-use instructions
- Tuned sampling parameters (`temperature`, `num_ctx`, `num_batch`, `num_predict`)

This custom model is cached by Ollama and reused on subsequent runs.

---

## Building the Desktop App (Electron)

Selene can be built into a standalone desktop application using Electron and PyInstaller. This bundles the Python backend and web UI into a single executable that you can distribute.

> **Note**: The Ollama engine and models (like Gemma 4) are **not** bundled in the app to keep the file size reasonable. Users must have Ollama installed and running on their system.

### Step-by-Step Build Instructions

1. **Install Node.js & Python dependencies**:
   Ensure you have Node.js installed. Then, install the required packages:
   ```bash
   bun install
   pip install pyinstaller
   ```

2. **Build the Python Backend**:
   First, compile the Python code into a standalone executable using the provided PyInstaller spec:
   ```bash
   bun run build:backend
   ```
   This creates the backend executable inside the `dist/` folder.

3. **Test in Development Mode (Optional)**:
   You can run the Electron app locally to ensure the backend spawns correctly:
   ```bash
   bun start
   ```

4. **Build the Electron App**:
   Package the application for your operating system:
   ```bash
   bun run build
   ```
   The final binaries (AppImage, deb, nsis, etc.) will be output to the `dist-electron/` folder.

## Usage

### Web UI (Default)

```bash
python main.py
```

The server binds to port `5005` (or the next free port) and automatically opens your default browser. You'll land on the chat interface immediately — no configuration required.

> **Port:** If `5005` is occupied the agent finds a free port and launches the browser pointing to that port.

**Sidebar controls** (click `⚙` to open):

| Control | Description |
|---------|-------------|
| Temperature slider | Adjust response creativity (0.0 – 1.0) |
| Top-P / Top-K sliders | Fine-tune nucleus and top-k sampling |
| System prompt | Override or reset the model's instructions |
| History toggle | Enable / disable conversation memory |
| Thinking toggle | Show / hide the model's reasoning panel |
| Save session | Persist the current conversation with a custom name |
| Load session | Restore any previously saved session |

### Terminal CLI

Pass `--cli` to skip the Web UI and use the classic terminal interface:

```bash
python main.py --cli
```

You'll see the chat prompt:

```
╭───────────────────────────────────────╮
│   Gemma CLI Agent  ·  type /help      │
╰───────────────────────────────────────╯

>>>
```

Type naturally. The agent will decide whether to answer directly or use tools. When tools are called, you'll see status indicators:

```
🔍  Searching the web: Python 3.14 new features
✓  Search complete — synthesizing answer…
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands (also `/?`) |
| `/clear` | Clear conversation history and reset system prompt |
| `/save [name]` | Save session to a JSON file |
| `/load [name\|index]` | Load a saved session (lists available if no arg) |
| `/set parameter <name> <val>` | Set a model parameter (e.g., `temperature 0.7`) |
| `/set system "<prompt>"` | Override the system prompt (use `default` to reset) |
| `/set history` / `/set nohistory` | Toggle conversation memory |
| `/set think` / `/set nothink` | Toggle thinking/reasoning visibility |
| `/set verbose` / `/set quiet` | Toggle generation stats (tok/s, elapsed time) |
| `/set format json` / `/set noformat` | Force JSON output mode |
| `/set wordwrap` / `/set nowordwrap` | Toggle word wrapping |
| `/show parameters` | View active session parameters and flags |
| `/show system` | Display the current system prompt |
| `/show model` | Show model architecture and parameter count (also `/show info`) |
| `/quit` | Exit the agent (also `/exit`, `/q`) |

### Vault Commands

The vault provides persistent semantic search over your local documents:

| Command | Aliases | Description |
|---------|---------|-------------|
| `/vault list` | `ls` | List indexed vault collections |
| `/vault aliases` | `list-aliases` | List registered vault aliases |
| `/vault alias <name> <coll>` | `register` | Register a friendly alias for a collection |
| `/vault rename <old> <new>` | `mv` | Rename a vault collection |
| `/vault add <path>` | `index` | Index a file or folder into the vault |
| `/vault search <query>` | `find` | Search indexed content for relevant chunks |
| `/vault delete <source>` | `remove`, `rm` | Remove indexed entries by source path |
| `/vault help` | `-h`, `--help` | Show vault command help |
| `/vault add <path> --collection notes` | | Index into a named collection |
| `/vault search <query> --top-k 10` | | Return more results |
| `/vault search <query> --source file.md` | | Restrict search to a specific source |
| `/vault delete --all` | | Delete an entire collection |

**Auto-indexing:** When you paste a file path as input and the file is large (>200KB) or binary (PDF/DOCX), the agent automatically indexes it into its own vault collection before processing. The collection name is derived from the filename (e.g., `DAA_Notes.pdf` → collection `DAA_Notes`).

**Auto-naming:** When no collection name is specified, the vault automatically derives one from the filename or folder name instead of dumping everything into a generic bucket. This means each document gets its own isolated, searchable collection:

| Input | Auto-derived collection |
|-------|------------------------|
| `DAA_Notes.pdf` | `DAA_Notes` |
| `Compression Notes.pdf` | `Compression_Notes` |
| `physics_notes.md` | `physics_notes` |
| Folder `/docs/` | `docs` |

**Auto-vaulting on file creation:** Every file created with the `create_file` tool is automatically saved into the `vaults/` directory (using only the file's basename, regardless of the path specified), indexed into its own ChromaDB collection, and registered with a human-friendly alias. This means you can immediately search any file the agent creates for you without manually indexing it.

**Vault Aliases:** Vaults are automatically given friendly aliases derived from the filename. When searching, you can use the original name (e.g., `"physics_notes"`) instead of remembering the sanitized ChromaDB collection name. Aliases are stored in `vaults/.vault_aliases.json` and support exact and substring matching.

**Multimodal Support:** For PDFs, the agent uses `moondream` via Ollama to generate visual descriptions of diagrams and slides. This is integrated directly into the vault indexing pipeline. Ensure you have run `ollama pull moondream` and installed `poppler-utils`.

### Runtime Configuration

All model parameters can be adjusted without restarting:

```bash
>>> /set parameter temperature 0.8
✓  temperature = 0.8

>>> /set parameter num_ctx 16384
✓  num_ctx = 16384

>>> /set verbose
✓  Verbose mode enabled — stats shown after each response.
```

Available parameters: `temperature`, `top_p`, `top_k`, `num_ctx`, `num_predict`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `min_p`, `tfs_z`, `repeat_last_n`, `seed`, `num_gpu`, `num_thread`, `num_keep`.

---

## Performance Tuning

The default configuration is optimised for consumer hardware (4-8GB VRAM GPUs). Key parameters in the `Modelfile`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `num_ctx` | 8192 | Context window size. Increase for more history, decrease for faster tok/s |
| `num_predict` | 2048 | Max tokens per response. Prevents runaway generation |
| `num_batch` | 512 | Prompt evaluation batch size. Higher = faster prefill on capable GPUs |
| `temperature` | 0.4 | Sampling temperature. Lower = more deterministic |

### Ollama Environment Variables

For additional throughput gains, set these before running `ollama serve`:

```bash
# Enable flash attention (major speedup if supported)
export OLLAMA_FLASH_ATTENTION=1

# Single user mode (max throughput)
export OLLAMA_NUM_PARALLEL=1

# Keep model loaded between requests (note: the agent also sets keep_alive=30m per-request)
export OLLAMA_KEEP_ALIVE=30m
```

### Memory Budgets

| `num_ctx` | Approx. KV Cache | Recommended VRAM |
|-----------|-------------------|------------------|
| 2048 | ~0.5 GB | 4 GB+ |
| 4096 | ~1.0 GB | 4 GB+ |
| 8192 | ~2.0 GB | 6 GB+ |
| 16384 | ~4.0 GB | 8 GB+ |
| 32768 | ~8.0 GB | 12 GB+ |
| 65536 | ~16 GB | 24 GB+ |

> **Rule of thumb:** If your tok/s drops below ~10 for simple queries, your KV cache is probably spilling to system RAM. Lower `num_ctx` until it fits in VRAM.

---

## Project Structure

```
AI-CLI-Agent/
├── main.py                    # Entry point — model init, --cli flag, web launch
├── Modelfile                  # Ollama model definition (system prompt, parameters)
├── requirements.txt           # Python dependencies
│
├── agent/
│   ├── __init__.py
│   ├── core.py                # Terminal chat loop, tool dispatch, streaming, session mgmt
│   ├── terminal.py            # ANSI helpers, spinner, LaTeX renderer, Markdown
│   ├── web.py                 # Threaded HTTP server, SSE generator, session/API routes
│   └── static/
│       ├── index.html         # Web UI layout — sidebar, chat panel, modals
│       ├── style.css          # Design system — dark mode, glassmorphism, animations
│       └── app.js             # Browser controller — SSE stream, tool cards, Markdown render
│
├── tools/
│   ├── __init__.py
│   ├── registry.py            # Tool JSON schemas + dispatch table
│   ├── search.py              # DuckDuckGo web search
│   ├── browser.py             # System browser control
│   ├── code.py                # Source code viewer with line numbers
│   ├── document.py            # PDF/DOCX extraction with chunking
│   ├── file.py                # Text file read/write with search; auto-vaults created files
│   ├── spotify.py             # Spotify cross-platform desktop control
│   ├── vision_describer.py    # Multimodal image description via moondream
│   ├── obsi_vault_writer.py   # Obsidian-optimised structured note creation
│   ├── vault_indexer.py       # Document chunking, ChromaDB indexing, alias registry
│   ├── vault_search.py        # Vector similarity search with alias resolution
│   ├── vault_embeddings.py    # Ollama embedding API helpers
│   ├── knowledge_graph_builder.py # Typed semantic graph inference
│   ├── run_simulation.py      # Safe dynamic and Monte Carlo models
│   ├── api_orchestrator.py    # Resilient authenticated HTTP lifecycle
│   ├── context_memory_optimizer.py # Long-context compaction
│   ├── reasoning_chain_debugger.py # Explicit evidence-graph audit
│   └── automated_routine_executor.py # Persistent preview-first macros
│
├── .agents/                   # Agent configuration
├── sessions/                  # Saved session JSON files
├── vaults/                    # Default vault document storage (also Obsidian vault)
├── .chroma/                   # ChromaDB persistent vector database
└── .gitignore
```

### Key Design Decisions

- **Web UI is the default** — `python main.py` starts the browser interface; the terminal CLI is opt-in via `--cli`. This keeps the richer interface front-and-centre without breaking existing workflows.
- **SSE over WebSockets** — Server-Sent Events are used for token streaming because they need only a standard HTTP connection, require no upgrade handshake, and reconnect automatically on drop.
- **`ThreadingMixIn` HTTP server** — `web.py` uses Python's `socketserver.ThreadingMixIn` so each request (including long-lived SSE connections) runs in its own daemon thread, preventing a slow generation from blocking the session API.
- **Shared `GLOBAL_STATE`** — a single in-process dict holds history and session parameters, making state trivially accessible across request handlers without an external store.
- **Tool schemas are compressed** to minimise prompt token overhead — they're sent with every LLM call, so shorter descriptions directly improve prefill speed.
- **Streaming is throttled** at ~12 FPS in the terminal to avoid CPU-bound Markdown re-rendering from bottlenecking the token pipeline. The Web UI receives every token immediately via SSE.
- **All regex patterns are pre-compiled** at module load time in `terminal.py`, not on each rendering pass.
- **History is trimmed** using a token budget heuristic (1 token ≈ 4 characters) to keep prompt size bounded without requiring a tokeniser dependency.
- **Vault embeddings** use the local `embeddinggemma` model via Ollama's HTTP API, falling back to the Python client if the HTTP endpoint changes.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-tool`)
3. Add your tool in `tools/` following the existing pattern:
   - Implement the tool function
   - Add a JSON schema to `TOOL_SCHEMAS` in `tools/registry.py`
   - Add the function to `TOOL_DISPATCH`
4. Test with `python main.py`
5. Submit a pull request

---

## License

Distributed under the Apache License 2.0. See [LICENSE](LICENSE) for full terms.
