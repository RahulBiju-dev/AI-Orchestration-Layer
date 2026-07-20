# Selene (AI Orchestration Layer)

A modular, tool-augmented **local AI orchestration layer** built with Python and [Ollama](https://ollama.com/). Selene sits between you and a local model: it owns the agentic loop, tool registry, context management, RAG vault, and session lifecycle, then surfaces the same capabilities through multiple interfaces.

Inference, conversation storage, and document memory stay local; optional integrations such as Google Calendar and Google Tasks contact their provider only after user-authorized OAuth. The orchestration core wraps a customised [Gemma 4](https://ai.google.dev/gemma) model with autonomous tool-calling, real-time streaming, and a persistent RAG (Retrieval-Augmented Generation) vault for long-term document memory.

**Fedora Linux is the primary development and reference platform.** Windows 10/11 are supported natively (no WSL or Unix compatibility layer). See [docs/platform-support.md](docs/platform-support.md) for the full tool and backend matrix.

### Interfaces (same orchestration core)

| Interface | How to launch | Role |
|-----------|---------------|------|
| **Web UI** (default) | `python main.py` | Browser chat with SSE streaming, thinking panels, tool cards, and sidebar controls |
| **Terminal CLI** | `python main.py --cli` | Rich terminal loop with slash commands and Markdown streaming |
| **Desktop app** | Electron build (`package.json`) | Packaged Web UI + PyInstaller backend for installable distribution |

All interfaces share `tool_runner`, the Ollama coordinator, managed model lifecycle, vault/RAG, and generation ownership rules — the product is the orchestration layer, not a single UI.

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
  - [Codebase Indexer](#codebase-indexer)
  - [Google Calendar and Tasks](#google-calendar-and-tasks)
- [Architecture](#architecture)
- [Platform support](#platform-support)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Diagnostics](#diagnostics)
- [Building the Desktop App (Electron)](#building-the-desktop-app-electron)
- [Usage](#usage)
  - [Web UI (Default)](#web-ui-default)
  - [Terminal CLI](#terminal-cli)
  - [Slash Commands](#slash-commands)
  - [Vault Commands](#vault-commands)
  - [Runtime Configuration](#runtime-configuration)
- [Performance Tuning](#performance-tuning)
- [Project Structure](#project-structure)
- [Testing](#testing)
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
    "arguments": {
      "query": "Python 3.14 release date",
      "include_content": true
    }
  }
}
```

The agent intercepts this and routes execution through **`agent/tool_runner.py`**, using schemas, dispatch, and `ToolMetadata` from `tools/registry.py`. Metadata covers side effects, parallel safety, resource weight, cancellation, timeouts, output bounds, platform support, and optional dependencies. The model never executes code directly — it only emits structured requests that the agent mediates.

When the model emits multiple independent read-only tool calls in the same response, Selene runs the safe calls concurrently and feeds the results back in the original order. Side-effecting tools and dependency-sensitive chains, such as current-date preflights before web search or scraping, remain ordered. A non-idempotent side-effect failure or timeout blocks later side effects in the same batch.

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
Documents ──→ Page/Text/Vision ──→ Chunk ──→ Embed ──→ ChromaDB
                                                     │
                    Focused query ──→ Vector Search ──┤
                    Complete read  ──→ Ordered Cursor ┴──→ Notes / PDF
```

**How RAG works:**

1. **Extraction and chunking:** Documents (PDF, DOCX, Markdown, reStructuredText, plain text) are split into overlapping segments of ~1800 characters. PDFs are processed page by page. Extracted text and optional Moondream visual analysis retain page, content-kind, chunk, and character-offset metadata.

2. **Embedding:** Each chunk is converted into a dense vector (a list of floating-point numbers) using an embedding model (`embeddinggemma` by default, running locally via Ollama). This vector captures the *semantic meaning* of the text — chunks about similar topics will have vectors that are close together in the embedding space, regardless of exact wording.

3. **Storage and checkpoints:** Vectors and source metadata are stored in [ChromaDB](https://www.trychroma.com/). Large-PDF progress is atomically checkpointed after every committed page under `vaults/.index_jobs/`. Pass each returned `next_page` as `resume_page`; changed files get a new fingerprint, and stale chunks are removed only after the replacement generation completes.

4. **Retrieval:** `vault_search` performs approximate nearest-neighbour retrieval for focused questions. `vault_read` instead walks every matching source/page/chunk through a stable `next_cursor`; oversized chunks use a character sub-cursor so exhaustive reads do not skip text.

5. **Long-form output:** `export_vault_pdf` creates a complete source-preserving reference PDF. `build_vault_notes_pdf` maps bounded ordered excerpts into grounded note sections, saves each section durably under `vaults/.pdf_jobs/`, and automatically assembles the final PDF after the last cursor. `create_pdf` handles ordinary Markdown-like content or a UTF-8 content file.

**Why local RAG?** Unlike cloud RAG services, everything runs on your machine. Your documents never leave your filesystem. The embedding model and vector database are both local — no API keys, no data exfiltration risks.

### Context Window Management

The **context window** (`num_ctx`) is the maximum number of tokens the model can "see" at once — both input and output combined. It's the model's working memory. Larger windows let the model consider more conversation history but come with costs:

- **KV cache memory** scales linearly with context length. For an 8B parameter model at Q4 quantisation, each additional 1K of context costs roughly 32-64MB of memory.
- **Prefill latency** increases with more input tokens.
- If the KV cache exceeds GPU VRAM, it spills to system RAM, causing a dramatic throughput drop.

The agent manages this automatically:

- **Conservative default `num_ctx` is 4096** under the low-VRAM profile (the safe default when VRAM cannot be measured or is ~4 GiB). The balanced profile raises this to **8192** on larger GPUs. Override with `SELENE_NUM_CTX` or `/set parameter num_ctx`.
- **System prompt persistence** keeps the active model system prompt in every model request. Selene reads the local `Modelfile` system prompt first, then falls back to Ollama's built-model prompt, then to `~/.selene-agent/system_prompt_cache.txt` (or `$SELENE_DATA_DIR/system_prompt_cache.txt`). This makes prompt edits effective at runtime even before the model is rebuilt, while still keeping a durable fallback.
- **System reminder anchoring** adds a compact runtime system reminder near the active user turn while preserving the full system prompt at the front. This helps long conversations retain tool/evidence rules even when the beginning of the context is far away.
- **History trimming and compaction** keep the prompt within the active token budget. Near 75% usage, older turns are summarized and passed through the context optimizer while system instructions and recent exchanges remain intact; hard trimming remains the final bound.
- **Context preflight guards** reserve output space before every Ollama call, including follow-up calls after tools. The guard counts serialized chat messages, the runtime tool schema list, a safety margin, and the requested `num_predict` output budget; if necessary it lowers `num_predict` for that call. When Ollama reaches the per-call output limit, Selene automatically continues from the latest answer suffix and streams one combined response instead of silently ending mid-output.
- **Compact runtime tool schemas** are sent to Ollama instead of the verbose documentation schemas. Function names, descriptions, parameters, required fields, and enums are preserved, but prose-heavy parameter descriptions are stripped because the detailed tool explanations already live in the system prompt and README.
- **Graceful overflow handling** stops before generation if the prompt still cannot safely fit after trimming. In that case Selene returns a controlled warning asking for a narrower request, a fresh chat, or a larger `num_ctx`, rather than producing unstable output.
- Both values can be overridden at runtime via `/set parameter num_ctx <value>`.

---

## Features

### Core
- **Fully local:** All inference runs through Ollama on your machine. No cloud dependencies for core functionality.
- **Custom model:** Uses a `Modelfile` to wrap Gemma 4 (8B, Q4_K_M quantisation) with a tailored system prompt and optimised sampling parameters.
- **Thinking visibility:** Streams the model's internal chain-of-thought reasoning before the final answer.
- **Multi-interface orchestration:** One agent core powering Web UI (default), Terminal CLI (`--cli`), and optional Electron desktop packaging.

### Web UI

The default interface — launch with `python main.py` and the agent opens in your browser automatically.

- **Cyberpunk-Obsidian aesthetic** — deep charcoal backgrounds, glassmorphism cards, and neon glowing accents in Cyan, Magenta, Teal, and Amber.
- **Enhanced 3D Elements** — realistic layered shadows (`--shadow-subtle`, `--shadow-heavy`) and tactile hover/active states that simulate physical lift for cards and message bubbles.
- **Context window usage indicators** — visual tracking of the model's context capacity in real time.
- **Live SSE streaming** — tokens and thinking blocks are pushed to the browser in real-time via Server-Sent Events; no polling, no page reloads.
- **Generation ownership** — each browser tab sends an `X-Selene-Client-ID`. Unsaved chats are isolated per tab; a **saved** session allows only one active generation across tabs. Each run has a generation ID and ends in exactly one terminal SSE state: `completed`, `cancelled`, or `failed`.
- **Safe cancellation** — stop an in-flight reply without tearing down the server; cancellation tokens cover model work and tool execution.
- **Smart generation states** — dynamic site behaviour that intelligently adapts while a response is actively generating.
- **Composer mode picker** — switch between Fast (the default), Ultra Thinking, and Deep Research beside the context meter; enhanced modes use a restrained live text shine while running, and the compact `×` returns the conversation to Fast.
- **Ultra Thinking** — forces difficulty-aware tools to their highest level, suspends the ordinary eight-round tool cap, and runs an independent second reasoning/review pass before exposing the final answer. Cancellation, context guards, confirmations, timeouts, and repeated-no-progress protection remain active.
- **Deep Research** — plans three to eight initial hard-difficulty searches according to the active context window, gathers the web evidence in parallel, and continues with as many distinct follow-up search rounds as the evidence requires before synthesizing a source-linked response. Raw research transcripts are compacted silently after every three searches or two page scrapes, while the exact original request is re-anchored. The ordinary eight-round cap is suspended while cancellation, context, timeout, and repeated-no-progress protections remain active.
- **Collapsible thinking panel** — while the model reasons, a dedicated magenta panel shows the chain-of-thought with animated dots. After completion it collapses into a togglable bar (DeepSeek/Gemini style).
- **Interactive tool cards** — each tool invocation renders a visual card: `⟳ Running [tool]` → `✓ Executed [tool]`. Click the header to expand raw JSON parameters and output.
- **Sidebar control panel:**
  - Real-time sliders for `Temperature`, `Top-P`, and `Top-K`
  - System-prompt override with a one-click reset to default
  - Toggles for conversation history and model thinking
  - Automatically saved conversations with agent-generated 2–3-word sidebar titles
  - Save / restore named sessions without leaving the browser
- **Markdown rendering** — responses support headings, emphasis, links, blockquotes, lists, task lists, fenced code with copy buttons, and responsive GFM-style tables.
- **LaTeX symbol rendering** — common commands such as `\oplus`, `\alpha`, `\subseteq`, and `\Rightarrow` render as Unicode outside code spans and fenced code.
- **Responsive layout** — sidebar collapses on narrow viewports; works on desktop and tablet.

### Tool Suite
The agent autonomously decides when to call tools based on the user's query:

| Tool | Description |
|------|-------------|
| 🔍 **Web Search** | Real-time DuckDuckGo search with adaptive depth (easy/medium/hard), plus optional top-result content extraction for current events, docs, and post-cutoff information |
| 🕸️ **Web Scraper** | Fetch and extract readable text, headings, metadata, and optional links from public HTTP(S) pages with byte/character limits and local-network safeguards |
| 🌐 **Browser** | Open URLs or search queries in the system's default browser |
| 💻 **Code Viewer** | Read source files with line numbers; scan directories by extension |
| 🧬 **Codebase Indexer** | Persistently index an entire repository, auto-refresh it after 24 hours, and retrieve grounded code context for architecture questions, fault finding, and optimisation |
| 📄 **Document Reader** | Extract text from PDFs (`pypdf`) and Word docs (`python-docx`) with page/chunk/query navigation |
| 📊 **Spreadsheet Tool** | View, read, search, and create bounded `.csv`, `.xls`, and `.xlsx` files with sheet and A1-range controls |
| 📂 **File Manager** | Stream line ranges, navigate/search bounded text files, and create non-overwriting files auto-vaulted under `~/.selene-agent/vaults/` |
| 🎵 **Spotify** | Search and play songs natively on Windows, macOS, and Linux |
| 👁️ **Vision Describer** | Describes images, diagrams, and slides using the local `moondream` vision model |
| 🗄️ **Vault Index** | Chunk and embed local files into ChromaDB for semantic search; auto-registers aliases |
| 🔎 **Vault Search** | Query the vault using vector similarity; resolves friendly aliases automatically |
| 📚 **Vault Read** | Traverse every indexed chunk in deterministic source/page order with a lossless resume cursor |
| 🧾 **PDF Creator** | Atomically render styled, paginated PDFs from Markdown-like content without silent overwrite |
| 📑 **Vault PDF Export** | Export an entire vault as a source-preserving reference PDF, or build resumable model-refined lecture notes |
| 🗑️ **Vault Delete** | Remove indexed entries by source path or delete entire collections |
| 🏷️ **Vault Aliases** | List registered human-friendly names that map to vault collections |
| 📓 **Obsidian Notes** | Create structured Obsidian-optimised notes with YAML frontmatter, WikiLinks, and version control |
| 🕸️ **Knowledge Graph Builder** | Map typed relationships and discover evidence-traceable causal paths, conflicts, central concepts, and feedback cycles |
| 📈 **Simulation Runner** | Execute recurrence, Euler, scenario, and Monte Carlo models with deterministic seeds and distribution summaries |
| 🔌 **API Orchestrator** | Manage API auth refresh, bounded retries, deprecation signals, response limits, and endpoint failover |
| 🧠 **Context Memory Optimizer** | Compact conversations while preserving instructions, recent turns, decisions, constraints, facts, and links |
| 🧭 **Reasoning Chain Debugger** | Audit explicit claim/evidence graphs for unsupported leaps, missing references, cycles, and confidence problems |
| ⚙️ **Automated Routine Executor** | Define natural-language workflow macros, preview their actions, and execute approved local commands/apps/URLs |
| 🚀 **App Launcher** | Launch up to ten installed desktop apps by display name (Linux desktop entries; Windows Start Menu shortcuts with target validation), with confirmation and command-injection safeguards |
| 🕒 **Current Date & Time** | Return the current local date/time or convert it to a requested IANA timezone |
| 💻 **Terminal Launcher** | Open a supported terminal at an existing directory only (no command execution): GNOME/KDE/Xfce on Linux; Windows Terminal / PowerShell / cmd on Windows |
| 📅 **Google Calendar** | List calendars and upcoming events, search a time range, and create or edit events; deletion requires explicit confirmation |
| ✅ **Google Tasks** | List task lists and tasks, create tasks with notes or due dates, and update status or details; deletion requires explicit confirmation |

Large PDF indexing is deliberately resumable rather than one enormous tool call. The conservative default processes up to 20 pages per call so full Moondream runs stay within the bounded tool timeout on slower local GPUs. `vision_mode=auto` uses Moondream on image-bearing or low-text pages, `vision_mode=all` analyzes every page and is required for handwritten-document requests, and `vision_mode=off` is text-only. Call the tool again with the same file/collection and `resume_page=next_page` until `complete=true`, or use `action=status` to inspect the compact durable checkpoint. Progressing `index_vault` continuations are not stopped by the ordinary eight-round tool cap; malformed, mixed, failed, or repeated no-progress checkpoints remain bounded. This is suitable for 900–1,200-page collections without retaining rendered page batches or the whole extracted document in RAM.

`file_path` is always the source file; omit `vault_path` for single-file indexing. Selene creates a managed collection folder under `SELENE_DATA_DIR/vaults/` automatically and leaves external source documents in place. `vault_path` is reserved for a source folder that should be indexed recursively. New installations keep Chroma data under `vaults/.chroma`; an existing legacy `.chroma` store continues in place without migration or copying.

### Spreadsheet Tool

Use `spreadsheet` with `action=view` for metadata and bounded previews, `action=read` for a worksheet, A1 range, or value query, and `action=create` to write a new `.csv`, `.xls`, or `.xlsx` file from JSON rows. CSV is treated as one worksheet, supports delimiter detection/selection, and can use the top-level `rows` convenience argument. Creation requires `confirmed=true`, is non-overwriting by default, and treats formula-looking strings as text unless `allow_formulas=true` is explicitly requested.

```json
{
  "action": "create",
  "file_path": "reports/scores.xlsx",
  "sheets": [
    {"name": "Scores", "rows": [["Name", "Score"], ["Ada", 10], ["Lin", 9]]}
  ],
  "confirmed": true
}
```

For CSV, pass rows directly and optionally choose a delimiter:

```json
{"action": "create", "file_path": "reports/scores.csv", "rows": [["Name", "Score"], ["Ada", 10]], "delimiter": ",", "confirmed": true}
```

### Codebase Indexer

`codebase_indexer` gives the agent persistent, repository-wide context for architecture questions, implementation tracing, fault finding, security review, and optimisation. Point the agent at a local repository and ask naturally, for example: *“In `/projects/shop`, trace checkout from the HTTP endpoint to the database and identify likely failure points.”*

The tool has three actions:

| Action | Behaviour |
|--------|-----------|
| `query` | Refreshes when necessary, then retrieves the most relevant code and repository-map chunks for the model to analyse. This is the default. |
| `index` | Explicitly indexes or refreshes a repository. Set `force_reindex=true` to bypass the cooldown when querying. |
| `status` | Reports the collection name, last successful index time, age, and next refresh time without indexing. |

Each absolute repository path receives its own stable ChromaDB collection. On the first reference, the tool recursively indexes supported source, configuration, and documentation files, records symbols and line ranges, and builds a chunked repository map. Later references reuse that collection. The first reference after the index becomes 24 hours old automatically refreshes it; this is a rolling 24-hour cooldown rather than a calendar-day reset.

Refreshes update changed files and remove chunks for deleted files. If embedding a particular file fails, its previous valid chunks are retained. Simultaneous first-use queries share a refresh lock so they do not duplicate the full embedding job.

Common dependency, cache, VCS, and build directories—such as `.git`, `node_modules`, `.venv`, `dist`, `build`, and `target`—are excluded. A single file is capped at 2 MiB, with repository caps of 5,000 files and 50 MiB of source text. These bounds keep accidental generated trees from overwhelming local Ollama and ChromaDB.

Indexes use the same local embedding model and Chroma storage as the document vault. Refresh metadata is stored in `~/.selene-agent/codebase_indexes.json`; vectors remain under `~/.selene-agent/.chroma/`. `SELENE_DATA_DIR` relocates both.

### Google Calendar and Tasks

The `google_workspace` tool exposes Google Calendar and Google Tasks as two user-facing capabilities through one encrypted OAuth connection.

| Capability | Supported operations |
|------------|----------------------|
| **Google Calendar** | Check connection status, list calendars, list/search events by time range, list upcoming birthdays with annual dates normalized into the requested window, create events, edit events, and delete confirmed events |
| **Google Tasks** | List task lists, list tasks with optional completed items, create tasks, edit titles/notes/due dates/status, and delete confirmed tasks |

Event times accept RFC 3339 date-times or `YYYY-MM-DD` for all-day events. Task due dates accept either form; Google Tasks retains the date portion. Calendar IDs default to `primary`, while task-list IDs default to `@default`. Selene sends attendee updates when an event with guests is created, changed, or deleted.

On first use, Selene opens Google's Desktop OAuth flow in the browser. The OAuth client configuration, access token, and refresh token are stored as AES-GCM ciphertext in `~/.selene-agent/google_oauth.enc` (or `$SELENE_DATA_DIR/google_oauth.enc`). Its encryption key has an owner-only mode-`0600` recovery copy beside the ciphertext and is mirrored to the OS keyring when that backend is available, so a locked or unavailable Linux keyring cannot strand future tokens. Refreshed tokens are immediately re-encrypted; credential values are redacted from tool errors.

The `status` action checks local decryptability without contacting Google or refreshing a token. If an older keyring-only credential cannot be unlocked, Selene preserves the encrypted file and asks for one explicit `authorize` flow before replacing it.

Setup:

1. Enable the **Google Calendar API** and **Google Tasks API** in a Google Cloud project.
2. Configure the OAuth consent screen, create a **Desktop app** OAuth client, and download its JSON outside this repository.
3. Install dependencies with `pip install -r requirements.txt`.
4. Tell Selene: `Connect my Google account using /absolute/path/to/client_secret.json`.
5. After Selene confirms the encrypted credential was saved, delete the downloaded source JSON.

Example requests:

- `What is on my primary calendar tomorrow?`
- `Create a project review on Monday from 2 PM to 3 PM in Asia/Kolkata.`
- `Show my incomplete Google Tasks.`
- `Add “submit expense report” to my default task list, due Friday.`

### Advanced Tool Safety Model

The advanced tools are deliberately bounded:

- Graph inferences include the exact supporting edge path; the builder does not invent edges from labels.
- Simulation equations use a restricted arithmetic parser—never Python `eval`—and workloads are capped. Forecasts remain conditional on the supplied assumptions.
- API credentials are referenced by environment-variable name rather than passed as literal secrets. Retries, timeouts, response sizes, and failover endpoints are capped.
- Memory optimisation is extractive and reports before/after token estimates. Automatic background compaction uses the same optimizer after generating its factual summary.
- The reasoning debugger audits supplied claims, dependencies, assumptions, and evidence IDs. It does not expose private model chain-of-thought; it produces an accountable evidence graph and Mermaid diagram.
- Routines live in `~/.selene-agent/routines.json` (or `$SELENE_DATA_DIR/routines.json`) so they persist across conversations, application restarts, and upgrades. Existing routines from `.selene/routines.json` are imported automatically. Routine actions can invoke registered agent tools; app actions are dispatched through `app_launcher.py`, including batched `launch_apps` calls. Use `action=show` (or `dry_run=true`) for the required preview of command, URL, and general tool runs, then use `action=run` with `confirmed=true` after user approval. App/delay-only routines can receive persistent approval when defined, allowing an exact saved trigger to run them later without another prompt. Commands use argument arrays with `shell=False` and remain in the project workspace.
- Google Calendar and Tasks use a bounded Desktop OAuth loopback flow. The downloaded client configuration and refresh token are stored together as AES-GCM ciphertext at `~/.selene-agent/google_oauth.enc` (or under `$SELENE_DATA_DIR`). A mode-`0600` local recovery key is mirrored to the OS keyring when available; unreadable legacy ciphertext is preserved until a successful explicit reauthorization. The downloaded source JSON is never copied into the repository and can be deleted after authorization.
- App actions accept only installed application display names. Shells, terminals, paths, URLs, command flags, uninstallers, and arbitrary PATH binaries are rejected; all launches are detached and shell-free. On Windows, discovery uses bounded Start Menu `.lnk` resolution with target validation.

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
    "description": "Open Antigravity and VS Code for the morning workspace.",
    "allow_automatic": true,
    "triggers": ["start my morning"],
    "actions": [
      {"type": "open_app", "app_name": "Antigravity"},
      {"type": "open_app", "app_name": "VS Code"}
    ]
  },
  "confirmed": true
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

Selene is an **orchestration layer**: thin UI shells (terminal, web, desktop) sit on a shared agent runtime that owns tools, model lifecycle, context, and memory.

```
┌──────────────────────────────────────────────────────────────────┐
│  Interfaces                                                      │
│  Web UI (default)  ·  Terminal CLI (--cli)  ·  Electron desktop  │
└───────────────┬──────────────────────────────┬───────────────────┘
                │                              │
     agent/core.py                    agent/web.py + web_runtime
     Terminal loop · /commands        Threaded HTTP + SSE
     session save                     generation ownership · APIs
                │                              │
                └────────────────┬─────────────┘
                                 ▼
              ┌────────────────────────────────────┐
              │  Orchestration core (main.py entry)│
              │  doctor · profiles · managed model │
              │  agent/ollama_runtime.py           │
              │  shared coordinator (chat/title/   │
              │  summary/embed/vision) + keep-alive│
              └────────────────┬───────────────────┘
                               │
              ┌────────────────┼────────────────────┐
              ▼                ▼                    ▼
   agent/tool_runner.py   tools/registry.py   agent/platform_runtime.py
   ordered/parallel       schemas·dispatch    paths·process·desktop
   cancel + timeouts      ·ToolMetadata       terminals·browser
              │
              ▼
         tools/* handlers  ──► optional Chroma / Ollama / OS APIs

 Supporting modules: runtime_config · model_lifecycle · diagnostics ·
 persistence · cancellation

 Electron desktop (optional): electron/main.js spawns selene-backend,
 waits for stdout port announce, single-instance lock, owned shutdown
```

Authoritative module map for contributors: [AGENTS.md](AGENTS.md).

### Data Flow — Web UI

1. Browser opens automatically at `http://localhost:5005` (or next free port)
2. User message is `POST`ed to `/api/chat` with client and generation identity headers
3. `web_runtime` leases generation ownership (one active generation per saved session; unsaved chats isolated per tab)
4. `web.py` builds the Ollama request via the shared coordinator with session parameters, trimmed history, and compact tool schemas
5. Tool calls run through `tool_runner` (metadata-aware parallelism, cancellation, bounds); SSE `tool_*` events stream to the browser
6. Thinking tokens arrive as `thinking` SSE events; content tokens as `token` events
7. `app.js` renders tokens in real time, assembles the thinking panel, and builds tool cards
8. The stream ends in exactly one terminal state (`completed`, `cancelled`, or `failed`) and history is updated when appropriate

### Data Flow — Terminal CLI

1. **User input** enters the chat loop in `agent/core.py`
2. Slash commands (`/help`, `/save`, `/vault`, etc.) are intercepted and handled locally — they never touch the LLM
3. Natural language input is sent through the shared Ollama coordinator with compact tool schemas and (trimmed) conversation history
4. Tool calls go through `tool_runner` → registry dispatch → Python handlers
5. Tool results are appended to the conversation and the model is called again
6. Final text output is streamed through the terminal renderer with Markdown and LaTeX processing
7. `Ctrl+\` cancels the active generation while keeping partial context; `Ctrl+C` exits the process

---

## Platform support

| Platform | Role | Notes |
|----------|------|-------|
| **Fedora Linux** | Primary / reference | DBus Spotify, desktop entries, XDG paths, GNOME terminals, AppImage |
| **Windows 10/11** | Native secondary | No WSL required; Start Menu apps, Windows Terminal/PowerShell/cmd, LocalAppData runtime |

Authoritative matrix (every registered tool): [docs/platform-support.md](docs/platform-support.md). Contributor architecture rules: [AGENTS.md](AGENTS.md).

### Runtime data paths

Selection order (no silent migration or copy):

1. `SELENE_DATA_DIR` (if set)
2. Existing legacy store `~/.selene-agent` (kept in place when present)
3. Platform default:
   - Linux: XDG data/state/config/cache under the Selene app name
   - Windows: `%LOCALAPPDATA%\Selene`

Critical JSON (sessions, aliases, routines metadata, and similar) is written with **atomic temp + fsync + replace**. Malformed files are preserved rather than silently overwritten. Selene never starts a second Ollama server and never stops an external Ollama process it did not own.

### Spotify / PDF notes

- **Fedora:** Spotify discovers active and instance-qualified MPRIS players over DBus (`dbus-python` is Linux-only in `requirements.txt`) and briefly waits for startup registration. If MPRIS remains unavailable, Selene uses the native `gio` Spotify URI handler and reports playback as unverified.
- **Windows:** Spotify uses a URI launch backend and never claims confirmed playback.
- **PDF text** works with `pypdf` alone. **PDF-to-image** needs Poppler (`poppler-utils` on Fedora; set `POPPLER_PATH` / `SELENE_POPPLER_PATH` on Windows if needed). **PDF creation** uses `reportlab` from `requirements.txt`.
- Optional packages (Google APIs, Chroma, vision/PDF image tooling, PDF writing, `dbus-python`) fail with capability errors for those features only — they must not block core import or startup.

---

## Prerequisites

- **[Ollama](https://ollama.com/)** installed and running (`ollama serve`)
- **Python 3.10+** (CI validates 3.11/3.12; local Fedora hosts may be newer)
- **Gemma 4 E4B model:** `ollama pull gemma4:e4b`
- **Embedding model (for vault):** `ollama pull embeddinggemma`
- **Vision model (optional):** `ollama pull moondream`
- **For Spotify:** Spotify desktop app. On Linux, `dbus-python` is also required (pre-installed on most GNOME/Fedora systems).

---

## Installation

### Fedora

```bash
# Clone the repository
git clone https://github.com/RahulBiju-dev/AI-Orchestration-Layer.git
cd AI-Orchestration-Layer

# Optional: PDF page images (text extraction works without this)
sudo dnf install poppler-utils -y

# Install Python dependencies
pip install -r requirements.txt

# Ensure Ollama has the required models
ollama pull gemma4:e4b
ollama pull embeddinggemma
ollama pull moondream   # optional vision

# Non-destructive environment check
python main.py --doctor

# Start the agent (auto-builds the custom model on first run)
python main.py
```

### Windows (native)

```powershell
git clone https://github.com/RahulBiju-dev/AI-Orchestration-Layer.git
cd AI-Orchestration-Layer
python -m pip install -r requirements.txt
# dbus-python is skipped automatically via environment markers
python main.py --doctor
python main.py
```

Optional PDF images on Windows: install Poppler and set `SELENE_POPPLER_PATH` to its `bin` directory.

### Multimodal Vision Capabilities
The agent supports memory-safe multimodal vision, allowing it to read slides, diagrams, and architectures from large PDFs without RAM exhaustion.
> **Note:** You MUST run `ollama pull moondream` in your terminal before using the agent with PDFs or images to enable this feature!

### What happens on first run

The agent uses a **staged managed-model lifecycle**: build under a temporary alias, inspect, publish to the live `selene` alias, then record Modelfile hash metadata. It never pre-deletes the live alias. The Modelfile bundles:
- The Gemma 4 base weights
- A system prompt with personality, knowledge cutoff rules, and tool-use instructions
- Conservative sampling parameters aligned with the selected hardware profile

This custom model is cached by Ollama and reused on subsequent runs when the Modelfile hash matches.

---

## Diagnostics

```bash
python main.py --doctor
python main.py --doctor --json
```

Reports Python/OS, runtime paths and writability, hardware profile, Ollama availability, model presence, GPU probe (when safe), tool-registry consistency, optional dependencies, terminal/Spotify capabilities, Poppler, port availability, and packaged resources. Secrets and personal document contents are never printed. Individual check failures do not abort the rest of the report.

---

## Building the Desktop App (Electron)

Selene can be built into a standalone desktop application using Electron and PyInstaller. This bundles the Python backend and web UI into a single executable that you can distribute.

> **Note**: The Ollama engine and models (like Gemma 4) are **not** bundled in the app to keep the file size reasonable. Users must have Ollama installed and running on their system.

### Scripts (`package.json`)

| Script | Purpose |
|--------|---------|
| `bun run build:backend` | PyInstaller via `selene-backend.spec` → `dist/selene-backend` (or `.exe`) |
| `bun run build:desktop` | `electron/build_desktop.py` helper (Electron's bundled Node + electron-builder) |
| `bun run build:appimage` | Linux AppImage only |
| `bun run build` | Backend + Linux AppImage |
| `bun run build:windows` | Backend + Windows NSIS (run on a Windows host) |
| `bun start` | Electron development shell against a local/backend setup |

### Step-by-Step Build Instructions

1. **Install Node.js & Python dependencies**:
   Ensure you have Bun (or Node) installed. Then:
   ```bash
   bun install
   pip install -r requirements-dev.txt   # includes PyInstaller
   ```

2. **Build the Python Backend**:
   ```bash
   bun run build:backend
   ```
   This creates the backend executable inside the `dist/` folder (`selene-backend` on Linux, `selene-backend.exe` on Windows). The spec keeps `console=True` so Electron can read the **stdout port announcement**; Windows Electron hides that console via spawn flags.

3. **Test in Development Mode (Optional)**:
   ```bash
   bun start
   ```
   Electron takes a single-instance lock, spawns only the Selene-owned backend, waits for the port line on stdout, and shuts that process tree down on quit (`/api/shutdown` with owner token, then SIGTERM/`taskkill /T` if needed). External Ollama is never stopped.

4. **Build the Electron App**:
   ```bash
   bun run build          # backend + Linux AppImage
   bun run build:windows  # backend + Windows NSIS (on Windows hosts)
   ```
   Artifact names follow `package.json` `version` (currently **1.0.0**):
   - Linux: `dist-electron/Selene-1.0.0.AppImage`
   - Windows: `dist-electron/Selene-1.0.0-Setup.exe`

   `electron/build_desktop.py` always forces `--publish never`. NSIS uninstall leaves user runtime data intact (`deleteAppDataOnUninstall: false`).

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
│   Selene  ·  type /help               │
╰───────────────────────────────────────╯

>>>
```

Type naturally. The agent will decide whether to answer directly or use tools. When tools are called, you'll see status indicators:

```
🔍  Searching the web: Python 3.14 new features
✓  Search complete — synthesizing answer…
```

### Slash Commands

Type `/` in the terminal CLI to open the filtered command menu. Use **Up/Down** to select a command and **Tab** to autofill it; continuing to type narrows the suggestions.

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
| `/vault status <path>` | | Show the durable large-PDF page checkpoint |
| `/vault read --cursor <n>` | | Read the vault exhaustively; repeat with the returned cursor |
| `/vault search <query>` | `find` | Search indexed content for relevant chunks |
| `/vault delete <source>` | `remove`, `rm` | Remove indexed entries by source path |
| `/vault help` | `-h`, `--help` | Show vault command help |
| `/vault add <path> --collection notes` | | Index into a named collection |
| `/vault add <path> --vision all --max-pages 25` | | Analyze every PDF page with Moondream in 25-page resumable runs |
| `/vault search <query> --top-k 10` | | Return more results |
| `/vault search <query> --source file.md` | | Restrict search to a specific source |
| `/vault delete --all` | | Delete an entire collection |

**Auto-indexing:** When you paste a file path as input and the file is large (>200KB) or binary (PDF/DOCX), the agent indexes it into its own vault collection before processing. The collection name is derived from the filename (e.g., `DAA_Notes.pdf` → collection `DAA_Notes`). Large PDFs return a truthful page checkpoint rather than claiming the entire document finished in one call.

**Auto-naming:** When no collection name is specified, the vault automatically derives one from the filename or folder name instead of dumping everything into a generic bucket. This means each document gets its own isolated, searchable collection:

| Input | Auto-derived collection |
|-------|------------------------|
| `DAA_Notes.pdf` | `DAA_Notes` |
| `Compression Notes.pdf` | `Compression_Notes` |
| `physics_notes.md` | `physics_notes` |
| Folder `/docs/` | `docs` |

**Auto-vaulting on file creation:** Every file created with `create_file` is saved into `~/.selene-agent/vaults/` by default (using only the basename), indexed into its own ChromaDB collection, and registered with a friendly alias. Existing files are never overwritten. If indexing is temporarily unavailable, file creation still succeeds and reports `indexed: false`.

**Vault Aliases:** Vaults are automatically given friendly aliases derived from the filename. When searching, you can use the original name (e.g., `"physics_notes"`) instead of remembering the sanitized ChromaDB collection name. Aliases are atomically stored in `~/.selene-agent/vaults/.vault_aliases.json`; substring resolution is used only when it identifies one unique collection.

**Multimodal Support:** For PDFs, the agent combines `pypdf` text with local Moondream descriptions of visible text, equations, labels, tables, chart values, diagrams, and relationships. `auto` is the efficient default; choose `all` when every page must be sent through Moondream. Ensure you have run `ollama pull moondream` and installed Poppler.

**Complete notes workflow:** Finish the resumable index first. Use `vault_search` for focused questions, `vault_read` with `next_cursor` for exhaustive logs, `export_vault_pdf` for a lossless reference export, or `build_vault_notes_pdf` for refined notes. The refined workflow returns a new cursor until every ordered excerpt has a durable note section; it never treats one top-K search as the whole lecture deck.

### Runtime Configuration

Settings resolve in this order (highest wins):

1. Session / slash-command override (`/set …`)
2. Environment variables (`SELENE_*`)
3. Selected hardware profile (`auto`, `low-vram`, `balanced`, `manual`)
4. Bundled Modelfile defaults (`manual`)
4. Bundled Modelfile defaults (`manual`)

The desktop app and browser UI ask which runtime profile to use when they open. **Manual** is preselected and keeps the bundled Modelfile values; choose **Auto** to inspect the device, or explicitly select **Low VRAM** or **Balanced**. The same choice remains available under Settings → Model.

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

Common environment overrides (see `agent/runtime_config.py` for the full list):

| Variable | Effect |
|----------|--------|
| `SELENE_RUNTIME_PROFILE` | `auto` · `low-vram` · `balanced` · `manual` |
| `SELENE_NUM_CTX` / `SELENE_NUM_PREDICT` / `SELENE_NUM_BATCH` | Context, output ceiling, prefill batch |
| `SELENE_TEMPERATURE` / `SELENE_TOP_P` / `SELENE_TOP_K` | Sampling |
| `SELENE_KEEP_ALIVE` | Ollama keep-alive (`10m`, `30m`, `0`, `-1`, …) |
| `SELENE_MODEL_CONCURRENCY` / `SELENE_TOOL_WORKERS` | Coordinator and tool-pool limits |
| `SELENE_CHAT_MODEL` / `SELENE_EMBEDDING_MODEL` / `SELENE_VISION_MODEL` | Model names |
| `SELENE_DATA_DIR` | Runtime data root |

---

## Performance Tuning

Selene defaults to the **manual** profile and bundled Modelfile parameters. Hardware inspection only selects a profile when **auto** is explicitly chosen; in Auto mode, unmeasurable VRAM or a ~4 GiB class GPU selects the conservative **low-vram** profile. All chat, title, summary, embedding, and vision work shares one Ollama coordinator; low-VRAM mode serializes model-heavy work without serializing ordinary tools.
Selene defaults to the **manual** profile and the bundled Modelfile parameters. Hardware inspection only selects a profile when you choose **auto**; in Auto mode, unmeasurable VRAM or a ~4 GiB class GPU selects the conservative **low-vram** profile. All chat, title, summary, embedding, and vision work shares one Ollama coordinator; low-VRAM mode serializes model-heavy work without serializing ordinary tools.

| Setting | low-vram (Auto safeguard) | balanced (larger GPU) | Purpose |
| Setting | low-vram (Auto safeguard) | balanced (larger GPU) | Purpose |
|---------|------------------------------|------------------------|---------|
| `num_ctx` | 4096 | 8192 | Context window |
| `num_predict` | 768 | 2048 | Output ceiling (per call; manual/Modelfile default is 2048) |
| `num_batch` | 128 | 512 | Prefill batch size |
| model slots | 1 | 2 | Concurrent model-heavy ops under the coordinator |
| tool workers | 2 | 4 | Bounded parallel tool execution |
| keep-alive | 10m | 30m | How long Ollama retains weights in memory |

These are **safeguards**, not a claim of measured optimality for every host. Override via environment (`SELENE_RUNTIME_PROFILE`, `SELENE_NUM_CTX`, …) or session `/set` commands after reading `python main.py --doctor`.

### Ollama Environment Variables

For additional throughput gains, set these before running `ollama serve`:

```bash
# Enable flash attention (major speedup if supported)
export OLLAMA_FLASH_ATTENTION=1

# Single user mode (max throughput)
export OLLAMA_NUM_PARALLEL=1

# Keep model loaded between requests (note: the agent also sets keep_alive per-request)
export OLLAMA_KEEP_ALIVE=30m
```

### Memory Budgets (approximate)

| `num_ctx` | Approx. KV Cache | Recommended VRAM |
|-----------|-------------------|------------------|
| 2048 | ~0.5 GB | 4 GB+ |
| 4096 | ~1.0 GB | 4 GB+ (Selene low-vram default) |
| 8192 | ~2.0 GB | 6 GB+ |
| 16384 | ~4.0 GB | 8 GB+ |

Exact VRAM use depends on model weights, quantization, and concurrent workloads.

> **Rule of thumb:** If your tok/s drops below ~10 for simple queries, your KV cache is probably spilling to system RAM. Lower `num_ctx` until it fits in VRAM.

---

## Project Structure

```
AI-Orchestration-Layer/
├── main.py                    # Entry — doctor, profile, managed model, multi-interface routing
├── Modelfile                  # Ollama model definition (system prompt, parameters)
├── package.json               # Electron packaging; version drives artifact names
├── selene-backend.spec        # PyInstaller backend bundle
├── requirements.txt           # Runtime Python deps (dbus-python Linux-only marker)
├── requirements-dev.txt       # Dev extras (PyInstaller, …)
├── AGENTS.md                  # Contributor / coding-agent architecture rules
├── docs/
│   └── platform-support.md    # Tool + platform capability matrix
│
├── agent/
│   ├── core.py                # Terminal chat loop, slash commands, sessions
│   ├── web.py                 # Threaded HTTP server, SSE generator, API routes
│   ├── web_runtime.py         # Client/session generation ownership leases
│   ├── tool_runner.py         # Metadata-aware ordered/parallel tool execution
│   ├── ollama_runtime.py      # Shared Ollama coordinator + API client
│   ├── model_lifecycle.py     # Staged alias build → inspect → publish
│   ├── runtime_config.py      # Hardware profiles and SELENE_* resolution
│   ├── platform_runtime.py    # Paths, process trees, desktop open helpers
│   ├── persistence.py         # Atomic JSON write helpers
│   ├── cancellation.py        # Cooperative cancellation tokens
│   ├── diagnostics.py         # python main.py --doctor
│   ├── terminal.py            # ANSI helpers, spinner, LaTeX, Markdown
│   └── static/                # Web UI (index.html, style.css, app.js)
│
├── tools/
│   ├── registry.py            # TOOL_SCHEMAS, TOOL_DISPATCH, TOOL_METADATA
│   ├── search.py · web_scraper.py · browser.py · code.py · codebase_indexer.py
│   ├── document.py · file.py · spreadsheet.py · spotify.py · vision_describer.py
│   ├── vault_*.py · obsi_vault_writer.py · app_launcher.py · terminal_launcher.py
│   ├── current_datetime.py · knowledge_graph_builder.py · run_simulation.py
│   ├── api_orchestrator.py · context_memory_optimizer.py · reasoning_chain_debugger.py
│   ├── automated_routine_executor.py · google_workspace.py
│   └── …
│
├── electron/
│   ├── main.js                # Spawn backend, port wait, single-instance, shutdown
│   ├── preload.js
│   └── build_desktop.py       # electron-builder via Electron's Node; --publish never
│
├── tests/                     # unittest suite (no live GPU/OAuth/Spotify)
├── .github/workflows/ci.yml   # Linux + Windows CI matrices
└── .gitignore
```

Runtime data is kept outside the checkout (see [Runtime data paths](#runtime-data-paths)). Default contents include conversations, `routines.json`, `system_prompt_cache.txt`, `google_oauth.enc`, `vaults/`, `.chroma/`, and `codebase_indexes.json`.

### Key Design Decisions

- **Multi-interface by design** — Web UI is the default (`python main.py`); Terminal CLI is opt-in via `--cli`; Electron packages the same orchestration backend for desktop distribution.
- **Shared Ollama coordinator** — chat, titles, summaries, embeddings, and vision share one queue/slot policy so low-VRAM hosts stay stable.
- **Managed model lifecycle** — rebuilds stage under a temporary alias, inspect, then publish to `selene`; the live alias is never pre-deleted.
- **SSE over WebSockets** — token streaming uses standard HTTP with automatic reconnect behaviour and no upgrade handshake.
- **`ThreadingMixIn` HTTP server** — long-lived SSE connections do not block session APIs.
- **Generation ownership** — `web_runtime` isolates unsaved tabs and serializes generations on a saved session name; terminal SSE state is exclusive.
- **`GLOBAL_STATE` + client stores** — in-process session/history defaults remain for the primary view, while per-client unsaved state and generation leases prevent cross-tab races.
- **Tool schemas are compacted at runtime** — detailed prose stays in the system prompt/docs; model-facing schemas keep callable structure only.
- **All normal tool execution goes through `tool_runner`** — CLI, web, slash commands, and routines share timeouts, cancellation, and side-effect batching rules.
- **Platform adapters own OS specifics** — paths, process groups/`taskkill /T`, terminals, browsers, and app launch live in `platform_runtime` rather than ad-hoc `os.system` calls.
- **Streaming is throttled** at ~12 FPS in the terminal; the Web UI receives every token immediately via SSE.
- **History is trimmed** with a conservative serialized-message heuristic and preflight output reservation before every Ollama request.
- **Vault embeddings** use local `embeddinggemma` via Ollama's HTTP API, with a Python-client fallback if the endpoint changes.

---

## Testing

```bash
python -m compileall . -q
python -m unittest discover -s tests -v
python main.py --doctor
```

CI (`.github/workflows/ci.yml`) runs Linux and Windows Python matrices, registry validation, frontend bundle checks, and packaging configuration smoke tests. Tests never download Ollama models or require a GPU, OAuth credentials, Spotify, or GUI interaction.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-tool`)
3. Read [AGENTS.md](AGENTS.md) for architecture freeze rules and platform policy
4. Add your tool in `tools/` following the existing pattern:
   - Implement the tool function
   - Add a JSON schema to `TOOL_SCHEMAS` in `tools/registry.py` (if model-exposed)
   - Add the function to `TOOL_DISPATCH` **and** `TOOL_METADATA`
   - Route execution only through `agent/tool_runner.py`
5. Test with `python -m unittest discover -s tests -v` and `python main.py --doctor`
6. Submit a pull request

Also see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

Distributed under the Apache License 2.0. See [LICENSE](LICENSE) for full terms.
