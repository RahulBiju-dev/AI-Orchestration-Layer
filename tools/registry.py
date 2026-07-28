"""
tools/registry.py — Tool JSON schemas for the Ollama API.

This module acts as the central registry for all available tools in the agent.
Each entry follows the Ollama tool-calling format (OpenAI-compatible
function schema). Add new tools here as the agent grows. These schemas are
passed directly to the LLM so it knows what tools are available and how to call them.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass, replace
from typing import Callable

from tools.search import web_search
from tools.web_scraper import web_scrape
from tools.document import read_document
from tools.file import read_file, create_file
from tools.code import view_code
from tools.spotify import spotify_play
from tools.browser import open_browser
from tools.app_launcher import launch_apps, open_app
from tools.terminal_launcher import open_terminal_at_path
from tools.current_datetime import get_current_datetime
from tools.spreadsheet import spreadsheet
from tools.vault_indexer import (
    delete_vault_item,
    index_vault,
    list_vault_aliases,
    list_vaults,
    register_vault_alias_tool,
    rename_vault,
)
from tools.vault_search import read_vault, search_vault
from tools.pdf_writer import build_vault_notes_pdf, create_pdf, export_vault_pdf
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
                        "maxLength": 255,
                        "description": "Optional IANA timezone such as Asia/Kolkata, Europe/London, or America/New_York. Omit for the computer's local timezone."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "spreadsheet",
            "description": "Reliably view, read, search, or write .csv, legacy .xls, and .xlsx spreadsheets. Use action=create with rows/sheets and confirmed=true when the user asks to create, save, write, or export spreadsheet data. New output is reopened and verified before atomic installation; existing files are preserved unless overwrite is explicit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["view", "read", "create"]},
                    "file_path": {"type": "string", "description": "Path to a .csv, .xls, or .xlsx file."},
                    "sheet": {"type": "string", "description": "Optional worksheet name. Read defaults to the first sheet; view defaults to all sheets."},
                    "cell_range": {"type": "string", "description": "Optional A1 range such as A1:D20 for read/query."},
                    "query": {"type": "string", "description": "Optional case-insensitive value search for read."},
                    "sheets": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "description": "For create: worksheets with name and rows. Cell values must be strings, numbers, booleans, or null.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "minLength": 1, "maxLength": 31},
                                "rows": {
                                    "type": "array",
                                    "maxItems": 10000,
                                    "items": {
                                        "type": "array",
                                        "maxItems": 256,
                                        "items": {
                                            "type": ["string", "number", "boolean", "null"],
                                            "maxLength": 1000000
                                        }
                                    }
                                }
                            },
                            "required": ["name", "rows"],
                            "additionalProperties": False
                        }
                    },
                    "rows": {
                        "type": "array",
                        "maxItems": 10000,
                        "items": {
                            "type": "array",
                            "maxItems": 256,
                            "items": {
                                "type": ["string", "number", "boolean", "null"],
                                "maxLength": 1000000
                            }
                        },
                        "description": "Convenience rows for creating a single-sheet workbook, especially CSV. Use sheets for multiple Excel worksheets."
                    },
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 200, "description": "Rows returned per preview/read (1-200, default 50)."},
                    "max_columns": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Columns returned per preview/read (1-100, default 30)."},
                    "data_only": {"type": "boolean", "description": "For .xlsx reads, return cached formula results instead of formulas when available."},
                    "overwrite": {"type": "boolean", "description": "For create, replace an existing file only when explicitly requested."},
                    "allow_formulas": {"type": "boolean", "description": "For Excel create, interpret strings beginning with = as formulas. For CSV, leave formula-looking fields unescaped. Defaults false to prevent spreadsheet formula injection."},
                    "delimiter": {"type": "string", "description": "For CSV, optional delimiter: comma, semicolon, tab (or \\t), or pipe. Reading auto-detects when omitted; writing defaults to comma."},
                    "confirmed": {"type": "boolean", "description": "Must be true for create after explicit user approval."}
                },
                "required": ["action", "file_path"]
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
                        "minLength": 1,
                        "maxLength": 1000,
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
                    "include_content": {
                        "type": "boolean",
                        "description": (
                            "When true, fetch readable text from the top search results. "
                            "Use this when snippets are insufficient and detailed source content is needed."
                        ),
                    },
                    "max_pages": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Optional number of top results to scrape when include_content=true (1-5). Defaults by difficulty.",
                    },
                    "max_chars_per_page": {
                        "type": "integer",
                        "minimum": 1000,
                        "maximum": 20000,
                        "description": "Maximum extracted text characters per scraped page when include_content=true (1000-20000, default 6000).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_scrape",
            "description": (
                "Read a specific public webpage or article URL; return grounded text, headings, metadata, and links."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 4096,
                        "description": "Public http/https URL to read. Bare domains are treated as https:// domains.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "minimum": 1000,
                        "maximum": 50000,
                        "description": "Maximum extracted text characters to return (1000-50000, default 20000).",
                    },
                    "include_links": {
                        "type": "boolean",
                        "description": "Include up to 40 extracted page links with anchor text.",
                    },
                },
                "required": ["url"],
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
            "description": (
                "Index/resume vaults. Handwriting: vision_mode=all. "
                "Pass next_page as resume_page until complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "description": "ChromaDB collection name."},
                    "vault_path": {"type": "string", "description": "Source folder to index recursively. Omit when file_path is supplied."},
                    "file_path": {"type": "string", "description": "Source file to index; managed vault storage is created automatically."},
                    "chunk_size": {"type": "integer", "description": "Chunk size (default 1800)."},
                    "chunk_overlap": {"type": "integer", "description": "Overlap between chunks (default 250)."},
                    "include_vision": {"type": "boolean", "description": "Enable automatic Moondream page selection (default true; false is text-only)."},
                    "vision_mode": {"type": "string", "enum": ["off", "auto", "all"], "description": "PDF vision policy. auto analyzes image-bearing/low-text pages; all runs Moondream on every page."},
                    "max_pages": {"type": "integer", "minimum": 1, "maximum": 500, "description": "PDF pages processed per resumable call (default 20)."},
                    "resume_page": {"type": "integer", "minimum": 1, "description": "For recursive resume calls, pass the previous PDF job's next_page exactly."},
                    "action": {"type": "string", "enum": ["index", "status"], "description": "Index/resume content or inspect durable PDF checkpoints."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vault_read",
            "description": (
                "Read a vault exhaustively in deterministic source/page/chunk order. "
                "Use next_cursor repeatedly to traverse every character without semantic-search gaps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "cursor": {"type": ["string", "integer"], "description": "Start at 0, then pass the returned next_cursor exactly."},
                    "source": {"type": "string", "description": "Optional source filename/path filter."},
                    "start_page": {"type": "integer", "minimum": 1},
                    "end_page": {"type": "integer", "minimum": 1},
                    "max_chunks": {"type": "integer", "minimum": 1, "maximum": 50},
                    "max_chars": {"type": "integer", "minimum": 500, "maximum": 3200}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_pdf",
            "description": "Create a styled, paginated PDF atomically from Markdown-like content or a UTF-8 text file. Existing PDFs are never overwritten without explicit confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative paths go under Selene vault exports; absolute paths must stay in the workspace or Selene data directory."},
                    "title": {"type": "string"},
                    "content": {"type": "string", "description": "Markdown-like PDF body."},
                    "content_file": {"type": "string", "description": "Optional UTF-8 source file for a long PDF body."},
                    "overwrite": {"type": "boolean"},
                    "confirmed": {"type": "boolean", "description": "Required when overwriting an existing PDF after explicit user approval."}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "export_vault_pdf",
            "description": (
                "Export every matching vault chunk in source/page order to a complete reference PDF without model summarization. "
                "For slide PDFs, set require_vision=true after indexing with vision_mode=all."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "file_path": {"type": "string"},
                    "title": {"type": "string"},
                    "source": {"type": "string"},
                    "start_page": {"type": "integer", "minimum": 1},
                    "end_page": {"type": "integer", "minimum": 1},
                    "require_vision": {
                        "type": "boolean",
                        "description": "Require verified complete vision_mode=all coverage for selected PDF pages."
                    },
                    "overwrite": {"type": "boolean"},
                    "confirmed": {"type": "boolean"}
                },
                "required": ["collection", "file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "build_vault_notes_pdf",
            "description": (
                "Recursively turn an entire vault into refined grounded notes and a final PDF. "
                "Processes bounded ordered windows, saves every note section durably, and returns next_cursor until complete. "
                "For slides, first index_vault with vision_mode=all until complete=true, then call this with require_vision=true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "file_path": {"type": "string"},
                    "title": {"type": "string"},
                    "source": {"type": "string"},
                    "cursor": {"type": ["string", "integer"], "description": "Omit initially; on resume pass returned next_cursor exactly."},
                    "sections_per_run": {"type": "integer", "minimum": 1, "maximum": 12},
                    "require_vision": {
                        "type": "boolean",
                        "description": "For slides/diagram-heavy PDFs, refuse notes unless every page completed vision_mode=all."
                    },
                    "action": {"type": "string", "enum": ["build", "status"]},
                    "overwrite": {"type": "boolean"},
                    "confirmed": {"type": "boolean"}
                },
                "required": ["collection", "file_path"]
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
            "description": "Call to map supplied concepts/entities and typed relationships, trace dependency or causal paths, find direct links, contradictory effects, feedback regions, and central nodes. Construct edges only from relationships stated or evidenced in the request/tool results; this tool analyzes them but does not invent facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concepts": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 500,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "minLength": 1, "maxLength": 200},
                                "label": {"type": "string", "minLength": 1, "maxLength": 2000},
                                "attributes": {
                                    "type": "object",
                                    "additionalProperties": True
                                }
                            },
                            "required": ["id"],
                            "additionalProperties": False
                        },
                        "description": "Concept/entity records. IDs must be stable exact identifiers used by relationship endpoints."
                    },
                    "relationships": {
                        "type": "array",
                        "maxItems": 3000,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "minLength": 1, "maxLength": 200},
                                "source": {"type": "string", "minLength": 1, "maxLength": 200},
                                "target": {"type": "string", "minLength": 1, "maxLength": 200},
                                "type": {"type": "string", "minLength": 1, "maxLength": 200},
                                "weight": {"type": "number", "minimum": 0, "maximum": 1},
                                "evidence": {}
                            },
                            "required": ["source", "target", "type"],
                            "additionalProperties": False
                        },
                        "description": "Directed typed edges. Use domain-specific types such as depends_on, causes, part_of, enables, prevents, or contradicts; weight is optional confidence from 0 to 1."
                    },
                    "query": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "minLength": 1, "maxLength": 200},
                            "target": {"type": "string", "minLength": 1, "maxLength": 200},
                            "relation_types": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 100,
                                "items": {"type": "string", "minLength": 1, "maxLength": 200}
                            }
                        },
                        "additionalProperties": False,
                        "description": "Optional endpoint/type filter. Omit relation_types to traverse every supplied edge type."
                    },
                    "max_depth": {"type": "integer", "minimum": 1, "maximum": 8, "description": "Maximum inference path length (1-8, default 4)."}
                },
                "required": ["concepts", "relationships"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_simulation",
            "description": "Call for multi-step numerical what-if questions, scenario comparisons, sensitivity analysis, projections, or Monte Carlo uncertainty ranges. Translate user-supplied values and stated assumptions into transparent variables/equations; use recurrence for next-state equations or euler for rates. Do not present results as independently validated forecasts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "object",
                        "minProperties": 1,
                        "maxProperties": 100,
                        "additionalProperties": {"type": "number"},
                        "description": "Initial numeric state keyed by identifier, for example {inventory: 100, demand: 12}."
                    },
                    "equations": {
                        "type": "object",
                        "minProperties": 1,
                        "maxProperties": 100,
                        "additionalProperties": {"type": "string", "minLength": 1, "maxLength": 2000},
                        "description": "Updated-variable expressions using state names, step, time, dt, pi, e, min, max, abs, sqrt, log, exp, sin, cos, floor, ceil, normal, or uniform. Every target must exist in variables."
                    },
                    "steps": {"type": "integer", "minimum": 1, "maximum": 10000},
                    "dt": {"type": "number", "minimum": 0.000000001, "maximum": 1000000},
                    "mode": {"type": "string", "enum": ["recurrence", "euler"]},
                    "scenarios": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "minLength": 1, "maxLength": 200},
                                "overrides": {
                                    "type": "object",
                                    "additionalProperties": {"type": "number"}
                                }
                            },
                            "required": ["name", "overrides"],
                            "additionalProperties": False
                        },
                        "description": "Optional named cases that override initial variables, such as baseline, optimistic, and pessimistic."
                    },
                    "trials": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Monte Carlo trials (1-1000)."},
                    "seed": {
                        "type": "integer",
                        "minimum": -9223372036854775807,
                        "maximum": 9223372036854775807,
                        "description": "Optional reproducibility seed. Supply one for repeatable stochastic results."
                    }
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
                    "request": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]},
                            "url": {"type": "string", "minLength": 1, "maxLength": 4096},
                            "headers": {"type": "object"},
                            "params": {}, "json": {}, "data": {},
                            "timeout": {"type": "number", "minimum": 0.5, "maximum": 120},
                            "max_response_chars": {"type": "integer", "minimum": 1000, "maximum": 100000},
                            "allow_redirects": {"type": "boolean"}
                        },
                        "required": ["url"],
                        "additionalProperties": False,
                        "description": "HTTP method, url, headers, params, json/data body, timeout, and response limit."
                    },
                    "auth": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["none", "bearer", "api_key", "basic", "oauth2_client_credentials"]},
                            "token_env": {"type": "string"}, "value_env": {"type": "string"},
                            "header": {"type": "string"}, "username_env": {"type": "string"},
                            "password_env": {"type": "string"}, "token_url": {"type": "string", "maxLength": 4096},
                            "client_id_env": {"type": "string"}, "client_secret_env": {"type": "string"},
                            "scope": {"type": "string"}
                        },
                        "additionalProperties": False,
                        "description": "Auth config. Use environment-variable names, never literal secrets. Types: none, bearer, api_key, basic, oauth2_client_credentials."
                    },
                    "retry": {
                        "type": "object",
                        "properties": {
                            "max_attempts": {"type": "integer", "minimum": 1, "maximum": 6},
                            "base_delay": {"type": "number", "minimum": 0, "maximum": 10}
                        },
                        "additionalProperties": False,
                        "description": "max_attempts (up to 6) and base_delay."
                    },
                    "alternative_endpoints": {"type": "array", "maxItems": 9, "items": {"type": "string", "maxLength": 4096}},
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
            "description": "Compact supplied chat messages to a token budget while preserving policy, recent tool turns, and key facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "maxItems": 10000,
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "minLength": 1, "maxLength": 50},
                                "content": {},
                                "tool_calls": {"type": "array"}
                            },
                            "required": ["role"]
                        }
                    },
                    "target_tokens": {"type": "integer", "minimum": 256, "maximum": 8000},
                    "preserve_recent": {"type": "integer", "minimum": 0, "maximum": 50},
                    "critical_terms": {"type": "array", "maxItems": 100, "items": {"type": "string", "maxLength": 200}}
                },
                "required": ["messages"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reasoning_chain_debugger",
            "description": "Audit an explicit claim/evidence graph for unsupported conclusions, bad links, cycles, and confidence gaps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conclusion": {"type": "string", "minLength": 1, "maxLength": 10000},
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 500,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "maxLength": 200},
                                "claim": {"type": "string", "maxLength": 10000},
                                "depends_on": {"type": "array", "maxItems": 200, "items": {"type": "string", "maxLength": 200}},
                                "evidence_ids": {"type": "array", "maxItems": 200, "items": {"type": "string", "maxLength": 200}},
                                "assumption": {"type": "boolean"},
                                "confidence": {"type": "number"}
                            },
                            "required": ["claim"],
                            "additionalProperties": False
                        }
                    },
                    "evidence": {
                        "type": "array",
                        "maxItems": 1000,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "minLength": 1, "maxLength": 200},
                                "source": {"type": "string"},
                                "quality": {"type": "number", "minimum": 0, "maximum": 1}
                            },
                            "required": ["id"],
                            "additionalProperties": True
                        }
                    }
                },
                "required": ["conclusion", "steps"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "automated_routine_executor",
            "description": "Reliably define, store, list, preview, run, replace, or delete reusable local workflow macros. Use this when the user asks to create or run a routine/workflow. Registered tool actions are schema-validated before saving and revalidated before execution. App-only routines may be persistently approved; commands, URLs, and other tools require per-run confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "define", "show", "run", "delete"]},
                    "name": {"type": "string", "minLength": 1, "maxLength": 200},
                    "trigger": {"type": "string", "minLength": 1, "maxLength": 200, "description": "Legacy single trigger accepted for compatibility. For new definitions, put every phrase in routine.triggers."},
                    "routine": {
                        "type": "object",
                        "description": "Complete routine definition. Infer a useful description from the user's request and preserve their trigger wording and example usages. Never leave description or triggers empty.",
                        "properties": {
                            "description": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 500,
                                "description": "Concise explanation of the routine's purpose and actions, derived from the user's request."
                            },
                            "triggers": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 25,
                                "items": {"type": "string", "minLength": 1, "maxLength": 200},
                                "description": "Natural-language phrases that identify this routine. Include the canonical phrase and each example usage supplied by the user."
                            },
                            "actions": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 50,
                                "description": "Ordered actions. Use open_app for one app, tool/launch_apps for several, or tool with another registered tool name and its normal arguments.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string", "enum": ["command", "open_app", "open_url", "delay", "tool"]},
                                        "argv": {
                                            "type": "array",
                                            "minItems": 1,
                                            "maxItems": 100,
                                            "items": {"type": "string", "minLength": 1, "maxLength": 4096}
                                        },
                                        "cwd": {"type": "string", "minLength": 1, "maxLength": 4096},
                                        "timeout": {"type": "number", "minimum": 1, "maximum": 600},
                                        "app_name": {"type": "string", "minLength": 1, "maxLength": 128},
                                        "url": {"type": "string", "minLength": 1, "maxLength": 4096},
                                        "seconds": {"type": "number", "minimum": 0, "maximum": 30},
                                        "tool_name": {"type": "string", "minLength": 1, "maxLength": 200},
                                        "arguments": {
                                            "type": "object",
                                            "additionalProperties": True
                                        },
                                        "continue_on_error": {"type": "boolean"}
                                    },
                                    "required": ["type"],
                                    "additionalProperties": False
                                }
                            },
                            "allow_automatic": {
                                "type": "boolean",
                                "description": "Set true only when the user explicitly requests persistent automatic execution; only app-launch and delay actions qualify."
                            }
                        },
                        "required": ["description", "triggers", "actions"]
                    },
                    "dry_run": {"type": "boolean", "description": "Optional. Set true to preview a run; action=show always previews and action=run executes by default."},
                    "overwrite": {"type": "boolean", "description": "For define, replace an existing routine only when explicitly requested together with confirmed=true."},
                    "confirmed": {"type": "boolean", "description": "Must be true for execution/deletion after explicit user approval, and when granting persistent approval to an automatic app-only routine."}
                },
                "required": ["action"]
            }
        }
    }
])

# ── Dispatch map ──────────────────────────────────────────────────────
# Maps function names to their Python callables.

TOOL_DISPATCH: dict[str, Callable] = {
    "get_current_datetime": get_current_datetime,
    "spreadsheet": spreadsheet,
    "web_search": web_search,
    "web_scrape": web_scrape,
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
    "create_pdf": create_pdf,
    "export_vault_pdf": export_vault_pdf,
    "build_vault_notes_pdf": build_vault_notes_pdf,
}

# Dispatch RAG tools
TOOL_DISPATCH.update({
    "index_vault": index_vault,
    "vault_search": search_vault,
    "vault_read": read_vault,
    "delete_vault_item": delete_vault_item,
    "list_vaults": list_vaults,
    "list_vault_aliases": list_vault_aliases,
    # Slash-command administrative helpers (not model-exposed).
    "register_vault_alias": register_vault_alias_tool,
    "rename_vault": rename_vault,
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


# ── Execution metadata ────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolMetadata:
    """Resource and safety contract used by every shared tool executor."""

    name: str
    read_only: bool
    side_effecting: bool
    parallel_safe: bool
    idempotent: bool
    network_bound: bool = False
    cpu_heavy: bool = False
    gpu_heavy: bool = False
    requires_temporal_preflight: bool = False
    supports_cancellation: bool = False
    default_timeout_seconds: float = 60.0
    max_output_chars: int = 20_000
    fedora_support: str = "supported"
    windows_support: str = "supported"
    optional_dependencies: tuple[str, ...] = ()
    model_exposed: bool = True

    def __post_init__(self) -> None:
        if self.read_only == self.side_effecting:
            raise ValueError(f"{self.name}: exactly one of read_only/side_effecting must be true")
        if self.parallel_safe and (self.side_effecting or self.cpu_heavy or self.gpu_heavy):
            raise ValueError(f"{self.name}: heavy or side-effecting tools cannot be parallel-safe")
        if self.default_timeout_seconds <= 0:
            raise ValueError(f"{self.name}: default timeout must be positive")
        if self.max_output_chars < 256:
            raise ValueError(f"{self.name}: output bound is too small")


def _metadata(
    name: str,
    *,
    read_only: bool = True,
    parallel_safe: bool = False,
    idempotent: bool | None = None,
    **values,
) -> ToolMetadata:
    side_effecting = not read_only
    if idempotent is None:
        idempotent = read_only
    return ToolMetadata(
        name=name,
        read_only=read_only,
        side_effecting=side_effecting,
        parallel_safe=parallel_safe,
        idempotent=idempotent,
        **values,
    )


# This map is deliberately explicit. A new dispatch entry should not silently
# inherit optimistic parallel/resource behavior.
TOOL_METADATA: dict[str, ToolMetadata] = {
    "get_current_datetime": _metadata(
        "get_current_datetime", parallel_safe=True, default_timeout_seconds=5, max_output_chars=4_000
    ),
    "spreadsheet": _metadata(
        "spreadsheet", read_only=False, default_timeout_seconds=120, max_output_chars=30_000,
        optional_dependencies=("openpyxl", "xlrd", "xlwt"),
    ),
    "web_search": _metadata(
        "web_search", parallel_safe=True, network_bound=True,
        requires_temporal_preflight=True, default_timeout_seconds=90, max_output_chars=35_000,
        optional_dependencies=("ddgs",),
    ),
    "web_scrape": _metadata(
        "web_scrape", parallel_safe=True, network_bound=True,
        requires_temporal_preflight=True, default_timeout_seconds=90, max_output_chars=50_000,
        optional_dependencies=("requests",),
    ),
    "read_document": _metadata(
        "read_document", cpu_heavy=True, default_timeout_seconds=120, max_output_chars=20_000,
        optional_dependencies=("pypdf", "python-docx"),
    ),
    "read_file": _metadata(
        "read_file", parallel_safe=True, default_timeout_seconds=20, max_output_chars=20_000
    ),
    "create_file": _metadata(
        "create_file", read_only=False, default_timeout_seconds=120, max_output_chars=12_000
    ),
    "spotify_play": _metadata(
        "spotify_play", read_only=False, network_bound=True, default_timeout_seconds=30,
        max_output_chars=8_000, windows_support="partial", optional_dependencies=("dbus-python",),
    ),
    "open_browser": _metadata(
        "open_browser", read_only=False, default_timeout_seconds=15, max_output_chars=8_000
    ),
    "view_code": _metadata(
        "view_code", parallel_safe=True, default_timeout_seconds=20, max_output_chars=20_000
    ),
    "describe_image": _metadata(
        "describe_image", gpu_heavy=True, supports_cancellation=True,
        default_timeout_seconds=300, max_output_chars=24_000,
        optional_dependencies=("ollama",),
    ),
    "create_pdf": _metadata(
        "create_pdf", read_only=False, cpu_heavy=True, default_timeout_seconds=180,
        max_output_chars=8_000, optional_dependencies=("reportlab",),
    ),
    "export_vault_pdf": _metadata(
        "export_vault_pdf", read_only=False, cpu_heavy=True, supports_cancellation=True,
        default_timeout_seconds=900, max_output_chars=10_000,
        optional_dependencies=("chromadb", "reportlab"),
    ),
    "build_vault_notes_pdf": _metadata(
        "build_vault_notes_pdf", read_only=False, cpu_heavy=True, supports_cancellation=True,
        default_timeout_seconds=900, max_output_chars=12_000,
        optional_dependencies=("chromadb", "ollama", "reportlab"),
    ),
    "open_terminal_at_path": _metadata(
        "open_terminal_at_path", read_only=False, default_timeout_seconds=15, max_output_chars=8_000
    ),
    "launch_apps": _metadata(
        "launch_apps", read_only=False, default_timeout_seconds=45, max_output_chars=12_000
    ),
    "open_app": _metadata(
        "open_app", read_only=False, default_timeout_seconds=20, max_output_chars=8_000,
        model_exposed=False,
    ),
    "google_workspace": _metadata(
        "google_workspace", read_only=False, network_bound=True,
        requires_temporal_preflight=True, default_timeout_seconds=120, max_output_chars=25_000,
        optional_dependencies=(
            "google-api-python-client", "google-auth-oauthlib", "cryptography", "keyring",
        ),
    ),
    "codebase_indexer": _metadata(
        "codebase_indexer", read_only=False, cpu_heavy=True, gpu_heavy=True,
        supports_cancellation=True,
        default_timeout_seconds=900, max_output_chars=25_000,
        optional_dependencies=("chromadb", "ollama"),
    ),
    "index_vault": _metadata(
        "index_vault", read_only=False, cpu_heavy=True, gpu_heavy=True,
        supports_cancellation=True,
        default_timeout_seconds=900, max_output_chars=20_000,
        optional_dependencies=("chromadb", "ollama"),
    ),
    "vault_search": _metadata(
        "vault_search", gpu_heavy=True, supports_cancellation=True,
        default_timeout_seconds=180, max_output_chars=20_000,
        optional_dependencies=("chromadb", "ollama"),
    ),
    "vault_read": _metadata(
        "vault_read", cpu_heavy=True, supports_cancellation=True,
        default_timeout_seconds=120, max_output_chars=4_000,
        optional_dependencies=("chromadb",),
    ),
    "delete_vault_item": _metadata(
        "delete_vault_item", read_only=False, cpu_heavy=True,
        default_timeout_seconds=120, max_output_chars=12_000, optional_dependencies=("chromadb",),
    ),
    "list_vaults": _metadata(
        "list_vaults", default_timeout_seconds=60, max_output_chars=12_000,
        optional_dependencies=("chromadb",),
    ),
    "list_vault_aliases": _metadata(
        "list_vault_aliases", parallel_safe=True, default_timeout_seconds=10, max_output_chars=12_000
    ),
    "register_vault_alias": _metadata(
        "register_vault_alias", read_only=False, default_timeout_seconds=15, max_output_chars=8_000,
        model_exposed=False,
    ),
    "rename_vault": _metadata(
        "rename_vault", read_only=False, cpu_heavy=True, default_timeout_seconds=180,
        max_output_chars=12_000, model_exposed=False, optional_dependencies=("chromadb",),
    ),
    "create_structured_note": _metadata(
        "create_structured_note", read_only=False, default_timeout_seconds=60, max_output_chars=12_000
    ),
    "knowledge_graph_builder": _metadata(
        "knowledge_graph_builder", cpu_heavy=True, supports_cancellation=True,
        default_timeout_seconds=60, max_output_chars=100_000
    ),
    "run_simulation": _metadata(
        "run_simulation", cpu_heavy=True, supports_cancellation=True,
        default_timeout_seconds=120, max_output_chars=100_000
    ),
    "api_orchestrator": _metadata(
        "api_orchestrator", read_only=False, network_bound=True,
        requires_temporal_preflight=True, default_timeout_seconds=180, max_output_chars=30_000
    ),
    "context_memory_optimizer": _metadata(
        "context_memory_optimizer", cpu_heavy=True, default_timeout_seconds=30, max_output_chars=25_000
    ),
    "reasoning_chain_debugger": _metadata(
        "reasoning_chain_debugger", cpu_heavy=True, default_timeout_seconds=30, max_output_chars=25_000
    ),
    "automated_routine_executor": _metadata(
        "automated_routine_executor", read_only=False, supports_cancellation=True,
        default_timeout_seconds=300, max_output_chars=20_000
    ),
}


# Tool handlers are keyword-bound callables. Closing each model-visible root
# schema ensures invented parameter names are rejected before they reach a
# handler as an opaque Python ``TypeError``. Nested free-form objects retain
# their explicitly declared JSON-schema behavior.
for _entry in TOOL_SCHEMAS:
    _function = _entry.get("function") if isinstance(_entry, dict) else None
    _parameters = _function.get("parameters") if isinstance(_function, dict) else None
    if isinstance(_parameters, dict) and _parameters.get("type") == "object":
        _parameters.setdefault("additionalProperties", False)


TOOL_SCHEMA_BY_NAME: dict[str, dict] = {
    str(entry.get("function", {}).get("name")): entry.get("function", {}).get("parameters", {})
    for entry in TOOL_SCHEMAS
    if entry.get("type") == "function" and entry.get("function", {}).get("name")
}


def get_tool_metadata(name: str, arguments: dict | None = None) -> ToolMetadata | None:
    """Resolve metadata, including the few operations whose safety varies by action."""
    metadata = TOOL_METADATA.get(name)
    if metadata is None:
        return None
    action = str((arguments or {}).get("action") or "").casefold()
    if name == "spreadsheet" and action in {"view", "read"}:
        return replace(
            metadata,
            read_only=True,
            side_effecting=False,
            parallel_safe=True,
            idempotent=True,
        )
    if name == "google_workspace" and action == "authorize":
        # The local OAuth listener stops itself after 240 seconds. Leave time
        # for the token exchange and atomic encrypted save before the runner's
        # non-cooperative timeout is reached.
        return replace(metadata, default_timeout_seconds=300)
    if name == "google_workspace" and action in {
        "status", "list_calendars", "list_events", "list_birthdays",
        "list_tasks", "list_task_lists",
    }:
        return replace(
            metadata,
            read_only=True,
            side_effecting=False,
            parallel_safe=False,
            idempotent=True,
        )
    if name == "automated_routine_executor" and action in {"list", "show"}:
        return replace(
            metadata,
            read_only=True,
            side_effecting=False,
            parallel_safe=False,
            idempotent=True,
        )
    if action == "status" and name in {
        "build_vault_notes_pdf", "codebase_indexer", "index_vault"
    }:
        return replace(
            metadata,
            read_only=True,
            side_effecting=False,
            parallel_safe=True,
            idempotent=True,
            cpu_heavy=False,
            gpu_heavy=False,
        )
    return metadata


_VALID_PLATFORM_SUPPORT = frozenset({"supported", "partial", "unsupported", "limited"})


def validate_tool_registry() -> list[str]:
    """Return contract violations without making optional tools fail at import time."""
    errors: list[str] = []
    dispatch_names = set(TOOL_DISPATCH)
    metadata_names = set(TOOL_METADATA)
    schema_names = set(TOOL_SCHEMA_BY_NAME)
    if dispatch_names - metadata_names:
        errors.append(f"Dispatch tools missing metadata: {sorted(dispatch_names - metadata_names)}")
    if metadata_names - dispatch_names:
        errors.append(f"Metadata without dispatch handlers: {sorted(metadata_names - dispatch_names)}")
    exposed_metadata = {name for name, value in TOOL_METADATA.items() if value.model_exposed}
    if schema_names != exposed_metadata:
        errors.append(
            f"Model schema/metadata mismatch: schemas_only={sorted(schema_names - exposed_metadata)}, "
            f"metadata_only={sorted(exposed_metadata - schema_names)}"
        )
    schema_name_list = [
        str(entry.get("function", {}).get("name"))
        for entry in TOOL_SCHEMAS
        if entry.get("type") == "function" and entry.get("function", {}).get("name")
    ]
    duplicate_schema_names = sorted({
        name for name in schema_name_list if schema_name_list.count(name) > 1
    })
    if duplicate_schema_names:
        errors.append(f"Duplicate model tool schemas: {duplicate_schema_names}")
    for name, schema in TOOL_SCHEMA_BY_NAME.items():
        properties = schema.get("properties")
        if schema.get("type") != "object" or not isinstance(properties, dict):
            errors.append(f"{name}: parameters must be an object schema with properties")
            continue
        if schema.get("additionalProperties") is not False:
            errors.append(f"{name}: root parameters must reject additional properties")
        required = schema.get("required", [])
        if not isinstance(required, list) or any(item not in properties for item in required):
            errors.append(f"{name}: required parameters must be a subset of properties")
        handler = TOOL_DISPATCH.get(name)
        if callable(handler):
            try:
                signature = inspect.signature(handler)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                keyword_parameters = {
                    parameter_name: parameter
                    for parameter_name, parameter in signature.parameters.items()
                    if parameter_name != "cancellation_token"
                    and parameter.kind in {
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    }
                }
                accepts_extras = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
                unknown_schema_parameters = sorted(set(properties) - set(keyword_parameters))
                if unknown_schema_parameters and not accepts_extras:
                    errors.append(
                        f"{name}: schema parameters are not accepted by the handler: {unknown_schema_parameters}"
                    )
                required_handler_parameters = {
                    parameter_name
                    for parameter_name, parameter in keyword_parameters.items()
                    if parameter.default is inspect.Parameter.empty
                }
                missing_required_schema = sorted(required_handler_parameters - set(required))
                if missing_required_schema:
                    errors.append(
                        f"{name}: required handler parameters are not required by the schema: {missing_required_schema}"
                    )
    for name, handler in TOOL_DISPATCH.items():
        if not callable(handler):
            errors.append(f"Dispatch handler for '{name}' is not callable")
        module = getattr(handler, "__module__", None)
        if module and module not in sys.modules:
            # Imported callables always have a module; require it still exists.
            try:
                importlib.import_module(module)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"Dispatch module for '{name}' ({module}) is not importable: {exc}")
    for name, metadata in TOOL_METADATA.items():
        if metadata.fedora_support not in _VALID_PLATFORM_SUPPORT:
            errors.append(f"{name}: invalid fedora_support={metadata.fedora_support!r}")
        if metadata.windows_support not in _VALID_PLATFORM_SUPPORT:
            errors.append(f"{name}: invalid windows_support={metadata.windows_support!r}")
        if not metadata.name or metadata.name != name:
            errors.append(f"Metadata key '{name}' does not match metadata.name={metadata.name!r}")
    return errors
