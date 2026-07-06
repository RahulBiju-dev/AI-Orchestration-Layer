"""
tools/registry.py — Tool JSON schemas for the Ollama API.

This module acts as the central registry for all available tools in the agent.
Each entry follows the Ollama tool-calling format (OpenAI-compatible
function schema). Add new tools here as the agent grows. These schemas are
passed directly to the LLM so it knows what tools are available and how to call them.
"""

from tools.search import web_search
from tools.document import read_document
from tools.file import read_file, create_file
from tools.code import view_code
from tools.spotify import spotify_play
from tools.browser import open_browser
from tools.app_launcher import launch_apps, open_app
from tools.terminal_launcher import open_terminal_at_path
from tools.current_datetime import get_current_datetime
from tools.vault_indexer import delete_vault_item, index_vault, list_vault_aliases, list_vaults
from tools.vault_search import search_vault
from tools.obsi_vault_writer import create_structured_note
from tools.vision_describer import describe_image
from tools.knowledge_graph_builder import knowledge_graph_builder
from tools.run_simulation import run_simulation
from tools.api_orchestrator import api_orchestrator
from tools.context_memory_optimizer import context_memory_optimizer
from tools.reasoning_chain_debugger import reasoning_chain_debugger
from tools.automated_routine_executor import automated_routine_executor
from tools.codebase_indexer import codebase_indexer
from tools.google_workspace import google_workspace

# ── Schema definitions ────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date, time, weekday, UTC offset, and Unix timestamp in the computer's local timezone or a requested IANA timezone. Use this for questions involving now, today, or the current time/date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Optional IANA timezone such as Asia/Kolkata, Europe/London, or America/New_York. Omit for the computer's local timezone."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web via DuckDuckGo for current information, "
                "events, docs, or anything beyond training data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard"],
                        "description": (
                            "Search depth. 'easy' (5 results) for quick facts; "
                            "'medium' (8 results, default) for general questions; "
                            "'hard' (15 results) for deep "
                            "research or niche/complex queries."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Read text from a PDF or .docx file with page, chunk, and query controls. "
                "For large docs, call with file_path only first for a preview."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the PDF or Word document.",
                    },
                    "pages": {
                        "type": "string",
                        "description": "PDF page selection, e.g. '1-3,8'.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query for relevant snippets.",
                    },
                    "chunk": {
                        "type": "integer",
                        "description": "0-based chunk number for large text.",
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Chars per chunk (default 12000).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max chars returned (default 14000).",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file with line range, chunk, and query controls. "
                "For large files, use query to find relevant lines first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                    "lines": {
                        "type": "string",
                        "description": "Line range, e.g. '20-80' or '42'.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query for matching snippets.",
                    },
                    "chunk": {
                        "type": "integer",
                        "description": "0-based chunk number.",
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Chars per chunk (default 12000).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max chars returned (default 14000).",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new, non-overwriting file in Selene's vault, then index it for search. Use only when explicitly asked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Filename for the new vault file; directory components are ignored.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write.",
                    }
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_play",
            "description": "Play a song on Spotify. Accepts a URI, URL, or search query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Spotify URI, URL, or search query (e.g. 'Bohemian Rhapsody Queen').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_browser",
            "description": "Open the default browser to a URL, a named web app (e.g. Gmail or Google Docs), or a search query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "URL, common web-app name, or search term to open.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_code",
            "description": (
                "View source code files with line numbers. Supports all common languages. "
                "Can scan folders for files by extension."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to a source file or folder.",
                    },
                    "lines": {
                        "type": "string",
                        "description": "Line range, e.g. '1-50'.",
                    },
                    "extension": {
                        "type": "string",
                        "description": "File extension to scan for in a folder (e.g. '.py').",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_image",
            "description": "Describe a local PNG, JPEG, WebP, GIF, or BMP image with the local moondream vision model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to a local image file (maximum 25 MiB)."},
                    "prompt": {"type": "string", "description": "Optional focused question or description instruction."}
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_terminal_at_path",
            "description": "Open a new terminal window with its working directory set to an existing local directory. Requires explicit user approval and never accepts a command to execute.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Existing local directory to open in the terminal."
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Must be true only when the user explicitly requested this terminal launch."
                    }
                },
                "required": ["path", "confirmed"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "launch_apps",
            "description": (
                "Launch one or more installed desktop applications by display name (for example, "
                "['Antigravity', 'VS Code']). Use only when the user explicitly asks to open them, "
                "or while executing a previously approved automatic app routine. This tool cannot "
                "launch terminals, shells, paths, URLs, or arbitrary command arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "app_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 10,
                        "description": "Installed application display names. Do not include paths or command arguments."
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Must be true only when the user explicitly requested this launch."
                    },
                },
                "required": ["app_names", "confirmed"],
            },
        },
    }
]

# Add RAG tooling: index the vault and search it
TOOL_SCHEMAS.extend([
    {
        "type": "function",
        "function": {
            "name": "google_workspace",
            "description": "Connect Google once, then view/create/edit Calendar events and Google Tasks. OAuth tokens and the client configuration are AES-encrypted in Selene's data directory. Use status before authorize; deletions and disconnect require explicit confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["status", "authorize", "disconnect", "list_calendars", "list_events", "list_birthdays", "create_event", "update_event", "delete_event", "list_task_lists", "list_tasks", "create_task", "update_task", "delete_task"], "description": "Use list_birthdays for upcoming contact birthdays; it normalizes annual recurrences into the current requested window."},
                    "client_secrets_file": {"type": "string", "description": "Path to a downloaded Google Desktop OAuth client JSON; only needed for authorize."},
                    "calendar_id": {"type": "string", "description": "Calendar ID; defaults to primary."},
                    "event_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "description": {"type": "string"},
                    "location": {"type": "string"},
                    "start": {"type": "string", "description": "RFC3339 date-time, or YYYY-MM-DD for an all-day event."},
                    "end": {"type": "string", "description": "RFC3339 date-time, or exclusive YYYY-MM-DD end date for an all-day event."},
                    "timezone": {"type": "string", "description": "IANA timezone such as Asia/Kolkata."},
                    "days_ahead": {"type": "integer", "description": "Upcoming birthday window in days for list_birthdays; defaults to 90."},
                    "attendees": {"type": "array", "items": {"type": "string"}},
                    "time_min": {"type": "string", "description": "RFC3339 lower bound for event listing."},
                    "time_max": {"type": "string", "description": "RFC3339 upper bound for event listing."},
                    "query": {"type": "string"},
                    "tasklist_id": {"type": "string", "description": "Task-list ID; defaults to @default."},
                    "task_id": {"type": "string"},
                    "title": {"type": "string"},
                    "notes": {"type": "string"},
                    "due": {"type": "string", "description": "YYYY-MM-DD or RFC3339 task due time (Google stores the date portion)."},
                    "status": {"type": "string", "enum": ["needsAction", "completed"]},
                    "show_completed": {"type": "boolean"},
                    "max_results": {"type": "integer", "description": "1-100, default 25."},
                    "confirmed": {"type": "boolean", "description": "Must be true for delete or disconnect after explicit user approval."}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "codebase_indexer",
            "description": (
                "Index and deeply inspect a local codebase using a persistent semantic code vault. "
                "Use for repository architecture, implementation questions, fault finding, security review, or optimisation. "
                "Query calls automatically reindex on first use after a 24-hour cooldown. Answer from returned context and cite files/lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "codebase_path": {"type": "string", "description": "Absolute or relative path to the repository root."},
                    "query": {"type": "string", "description": "Focused question about the codebase. Required for query action."},
                    "action": {"type": "string", "enum": ["query", "index", "status"], "description": "Defaults to query."},
                    "force_reindex": {"type": "boolean", "description": "Ignore the 24-hour cooldown and refresh now."},
                    "top_k": {"type": "integer", "description": "Relevant code chunks to retrieve (1-20, default 10)."},
                    "max_chars": {"type": "integer", "description": "Maximum retrieved context size (1000-30000, default 14000)."}
                },
                "required": ["codebase_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "index_vault",
            "description": "Index a folder or file into ChromaDB vault. Use before vault_search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "description": "ChromaDB collection name."},
                    "vault_path": {"type": "string", "description": "Folder to index recursively."},
                    "file_path": {"type": "string", "description": "File to index."},
                    "chunk_size": {"type": "integer", "description": "Chunk size (default 1800)."},
                    "chunk_overlap": {"type": "integer", "description": "Overlap between chunks (default 250)."},
                    "include_vision": {"type": "boolean", "description": "Include moondream page descriptions for PDFs (default true; false is faster for text-only PDFs)."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vault_search",
            "description": "Search indexed vault for relevant chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "collection": {"type": "string"},
                    "top_k": {"type": "integer", "description": "Chunks to return (default 6)."},
                    "max_chars": {"type": "integer", "description": "Max chars in context (default 7000)."},
                    "source": {"type": "string", "description": "Source path to restrict search."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_vault_item",
            "description": (
                "Delete indexed chunks from a vault collection by source/source_path, or delete a collection. "
                "This does not delete files from disk. Use only when explicitly asked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Indexed source or file path to delete."},
                    "collection": {"type": "string", "description": "ChromaDB collection name."},
                    "delete_collection": {"type": "boolean", "description": "Delete the entire collection."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_vaults",
            "description": "List existing indexed vault collections and their chunk counts.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_vault_aliases",
            "description": "List all registered vault aliases — friendly names that map to indexed collections. Use this when the user wants to know what vaults are available or what name to use for vault_search.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_structured_note",
            "description": "Create an autonomous Obsidian note structured for graph view, including YAML tags, internal links, and versioned filenames.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The title of the note."},
                    "content": {"type": "string", "description": "The main markdown content of the note."},
                    "incoming_links": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of incoming WikiLink note titles."
                    },
                    "outgoing_links": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of outgoing WikiLink note titles to include in a Related Concepts section."
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of tags for the YAML frontmatter."
                    }
                },
                "required": ["title", "content"]
            }
        }
    },
])

# Reasoning, prediction, integration, memory, and workflow tools.
TOOL_SCHEMAS.extend([
    {
        "type": "function",
        "function": {
            "name": "knowledge_graph_builder",
            "description": "Build a concept/relationship graph and discover explainable multi-hop causal paths, conflicts, feedback cycles, and central concepts. Inferences are grounded only in supplied edges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concepts": {"type": "array", "items": {"type": "object"}, "description": "Concept objects with id, optional label, and optional attributes."},
                    "relationships": {"type": "array", "items": {"type": "object"}, "description": "Edges with source, target, type, optional weight (0-1), and evidence."},
                    "query": {"type": "object", "description": "Optional source, target, and relation_types filter."},
                    "max_depth": {"type": "integer", "description": "Maximum inference path length (1-8, default 4)."}
                },
                "required": ["concepts", "relationships"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_simulation",
            "description": "Run bounded discrete-time what-if or Monte Carlo simulations from explicit variables and equations. Use recurrence for next-state equations or euler for rates of change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "variables": {"type": "object", "description": "Initial numeric state keyed by variable name."},
                    "equations": {"type": "object", "description": "Safe arithmetic expression for each updated variable; may use step, time, dt, normal(), and uniform()."},
                    "steps": {"type": "integer"}, "dt": {"type": "number"},
                    "mode": {"type": "string", "enum": ["recurrence", "euler"]},
                    "scenarios": {"type": "array", "items": {"type": "object"}, "description": "Named scenarios containing variable overrides."},
                    "trials": {"type": "integer", "description": "Monte Carlo trials (1-1000)."},
                    "seed": {"type": "integer", "description": "Optional reproducibility seed."}
                },
                "required": ["variables", "equations"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "api_orchestrator",
            "description": "Manage a resilient HTTP API call with environment-based authentication, OAuth client-credential refresh, exponential backoff, deprecation detection, and alternative endpoint failover.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {"type": "object", "description": "HTTP method, url, headers, params, json/data body, timeout, and response limit."},
                    "auth": {"type": "object", "description": "Auth config. Use environment-variable names, never literal secrets. Types: none, bearer, api_key, basic, oauth2_client_credentials."},
                    "retry": {"type": "object", "description": "max_attempts (up to 6) and base_delay."},
                    "alternative_endpoints": {"type": "array", "items": {"type": "string"}},
                    "documentation": {"type": "object", "description": "Optional OpenAPI-like base_url and paths map used to find non-deprecated alternatives."}
                },
                "required": ["request"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "context_memory_optimizer",
            "description": "Compress explicit conversation messages into compact memory while retaining system instructions, recent turns, decisions, constraints, facts, tool results, and semantic links.",
            "parameters": {
                "type": "object",
                "properties": {
                    "messages": {"type": "array", "items": {"type": "object"}},
                    "target_tokens": {"type": "integer"},
                    "preserve_recent": {"type": "integer"},
                    "critical_terms": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["messages"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reasoning_chain_debugger",
            "description": "Audit an explicit claim/evidence rationale for missing support, bad references, circular dependencies, ambiguity, and unjustified confidence. It does not expose hidden model chain-of-thought.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conclusion": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "object"}, "description": "Steps with id, claim, depends_on, evidence_ids, assumption, and confidence."},
                    "evidence": {"type": "array", "items": {"type": "object"}, "description": "Evidence records with stable id and source."}
                },
                "required": ["conclusion", "steps"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "automated_routine_executor",
            "description": "Define, store, list, preview, run, or delete reusable local workflow macros. App-only routines may be persistently approved at definition time; command and URL routines always require confirmation for each run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "define", "show", "run", "delete"]},
                    "name": {"type": "string"},
                    "trigger": {"type": "string", "description": "Legacy single trigger accepted for compatibility. For new definitions, put every phrase in routine.triggers."},
                    "routine": {
                        "type": "object",
                        "description": "Complete routine definition. Infer a useful description from the user's request and preserve their trigger wording and example usages. Never leave description or triggers empty.",
                        "properties": {
                            "description": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Concise explanation of the routine's purpose and actions, derived from the user's request."
                            },
                            "triggers": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 25,
                                "items": {"type": "string", "minLength": 1},
                                "description": "Natural-language phrases that identify this routine. Include the canonical phrase and each example usage supplied by the user."
                            },
                            "actions": {
                                "type": "array",
                                "minItems": 1,
                                "description": "Ordered actions. Use {type: open_app, app_name: <display name>} for one app, or {type: tool, tool_name: launch_apps, arguments: {app_names: [...]}} for several. Other registered tools may be called with type: tool."
                            },
                            "allow_automatic": {
                                "type": "boolean",
                                "description": "Set true only when the user explicitly requests persistent automatic execution; only app-launch and delay actions qualify."
                            }
                        },
                        "required": ["description", "triggers", "actions"]
                    },
                    "dry_run": {"type": "boolean", "description": "Optional. Set true to preview a run; action=show always previews and action=run executes by default."},
                    "confirmed": {"type": "boolean", "description": "Must be true for execution/deletion after explicit user approval, and when granting persistent approval to an automatic app-only routine."}
                },
                "required": ["action"]
            }
        }
    }
])

# ── Dispatch map ──────────────────────────────────────────────────────
# Maps function names to their Python callables.

TOOL_DISPATCH: dict[str, callable] = {
    "get_current_datetime": get_current_datetime,
    "web_search": web_search,
    "read_document": read_document,
    "read_file": read_file,
    "create_file": create_file,
    "view_code": view_code,
    "spotify_play": spotify_play,
    "open_browser": open_browser,
    "open_app": open_app,
    "launch_apps": launch_apps,
    "open_terminal_at_path": open_terminal_at_path,
    "describe_image": describe_image,
}

# Dispatch RAG tools
TOOL_DISPATCH.update({
    "index_vault": index_vault,
    "vault_search": search_vault,
    "delete_vault_item": delete_vault_item,
    "list_vaults": list_vaults,
    "list_vault_aliases": list_vault_aliases,
    "create_structured_note": create_structured_note,
    "knowledge_graph_builder": knowledge_graph_builder,
    "run_simulation": run_simulation,
    "api_orchestrator": api_orchestrator,
    "context_memory_optimizer": context_memory_optimizer,
    "reasoning_chain_debugger": reasoning_chain_debugger,
    "automated_routine_executor": automated_routine_executor,
    "codebase_indexer": codebase_indexer,
    "google_workspace": google_workspace,
})
