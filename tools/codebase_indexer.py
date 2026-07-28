"""Code-aware repository indexing and retrieval backed by the local Chroma vault."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.cancellation import CancellationToken, OperationCancelled
from agent.persistence import PersistenceError, atomic_write_json, read_json_preserved

from tools.vault_embeddings import DEFAULT_EMBED_MODEL, embed_query, embed_texts
from tools.vault_indexer import CHROMA_DIR, DATA_DIR, chunk_text_with_offsets, get_chroma_client

REFRESH_SECONDS = 24 * 60 * 60
DEFAULT_TOP_K = 10
DEFAULT_MAX_CHARS = 14000
DEFAULT_BATCH_SIZE = 16
STATE_FILE = os.path.join(DATA_DIR, "codebase_indexes.json")
_STATE_LOCK = threading.RLock()
_INDEX_LOCK = threading.RLock()

CODE_EXTENSIONS = {
    ".asm", ".bash", ".bat", ".c", ".cc", ".cfg", ".cjs", ".clj", ".cljc",
    ".cljs", ".cmake", ".cmd", ".coffee", ".conf", ".config", ".cpp", ".cs",
    ".css", ".csv", ".cxx", ".dart", ".edn", ".erl", ".ex", ".exs", ".f",
    ".f03", ".f08", ".f90", ".f95", ".fish", ".fs", ".fsx", ".gql", ".go",
    ".gradle", ".graphql", ".graphqls", ".groovy", ".h", ".hh", ".hpp", ".hs",
    ".html", ".htm", ".hxx", ".ini", ".ipynb", ".java", ".jl", ".js", ".jsx",
    ".json", ".ksh", ".kt", ".kts", ".less", ".lhs", ".lisp", ".lock", ".lua",
    ".m", ".markdown", ".md", ".mdown", ".mjs", ".mm", ".pas", ".php", ".pl",
    ".pm", ".pp", ".properties", ".proto", ".ps1", ".psd1", ".psm1", ".py",
    ".pyi", ".pyw", ".r", ".rake", ".rb", ".rst", ".rs", ".s", ".sass",
    ".scala", ".scss", ".sh", ".sql", ".svelte", ".swift", ".t", ".tex",
    ".tf", ".tfvars", ".thrift", ".toml", ".ts", ".tsv", ".tsx", ".vue",
    ".xml", ".yaml", ".yml", ".zig", ".zsh",
}
SPECIAL_FILES = {
    "dockerfile", "makefile", "procfile", "rakefile", "gemfile", "cmakelists.txt",
    "package.json", "pyproject.toml", "requirements.txt", "cargo.toml", "go.mod",
    "composer.json", "build.gradle", "settings.gradle", ".gitignore", ".dockerignore",
    ".editorconfig", ".env", ".gitattributes", ".npmrc", ".nvmrc", "authors",
    "copying", "justfile", "license", "notice", "vagrantfile",
}
IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".chroma", ".idea", ".vscode", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".nox", ".venv", ".gradle",
    ".parcel-cache", ".svelte-kit", ".turbo", "venv", "env",
    "__pycache__", "node_modules", "bower_components", "vendor", "dist", "build",
    "dist-electron", "coverage", "htmlcov", "target", "out", ".next", ".nuxt",
    ".cache", "deriveddata", "pods",
}
LANGUAGES = {
    ".py": "Python", ".pyi": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".mjs": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript", ".rs": "Rust",
    ".go": "Go", ".java": "Java", ".kt": "Kotlin", ".cpp": "C++", ".cc": "C++",
    ".c": "C", ".cs": "C#", ".rb": "Ruby", ".php": "PHP", ".swift": "Swift",
    ".scala": "Scala", ".sh": "Shell", ".sql": "SQL", ".html": "HTML",
    ".css": "CSS", ".vue": "Vue", ".svelte": "Svelte",
}


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False)


def _utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _collection_name(root: str) -> str:
    basename = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.basename(root)).strip("_") or "repo"
    digest = hashlib.sha256(root.encode("utf-8")).hexdigest()[:12]
    return f"codebase_{basename[:36]}_{digest}"[:63]


def _load_state() -> dict:
    with _STATE_LOCK:
        try:
            return read_json_preserved(STATE_FILE, expected_type=dict)
        except FileNotFoundError:
            return {}


def _save_state(state: dict) -> None:
    with _STATE_LOCK:
        atomic_write_json(STATE_FILE, state, private=True)


def _is_indexable(path: Path) -> bool:
    name = path.name.casefold()
    return (
        path.suffix.lower() in CODE_EXTENSIONS
        or name in SPECIAL_FILES
        or name.startswith(".env.")
    )


def _discover_files(
    root: str,
    cancellation_token: CancellationToken | None = None,
) -> tuple[list[Path], list[dict]]:
    files: list[Path] = []
    skipped: list[dict] = []
    for current, dirs, names in os.walk(root, followlinks=False):
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        dirs[:] = sorted(
            name
            for name in dirs
            if name.casefold() not in IGNORED_DIRS
            and not os.path.islink(os.path.join(current, name))
        )
        for name in sorted(names):
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            path = Path(current, name)
            if not _is_indexable(path) or path.is_symlink():
                continue
            try:
                path.stat()
            except OSError as exc:
                skipped.append({"file": str(path), "reason": str(exc)})
                continue
            files.append(path)
    return files, skipped


def _read_source(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if b"\x00" in raw:
        raise ValueError("binary file")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return raw.decode("cp1252")
        except UnicodeDecodeError:
            return raw.decode("latin-1")


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _symbol_hints(text: str) -> list[str]:
    patterns = (
        r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)",
        r"^\s*class\s+([A-Za-z_]\w*)",
        r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)",
        r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)",
        r"^\s*(?:public\s+|private\s+|protected\s+)?(?:class|interface|enum)\s+([A-Za-z_]\w*)",
    )
    found: list[str] = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text, flags=re.MULTILINE))
    return list(dict.fromkeys(found))[:30]


def _overview_document(root: str, records: list[dict]) -> str:
    extensions: dict[str, int] = {}
    lines = [f"Repository: {os.path.basename(root)}", f"Root: {root}", "", "Files:"]
    for record in records:
        extension = record["extension"] or "[no extension]"
        extensions[extension] = extensions.get(extension, 0) + 1
        symbols = ", ".join(record["symbols"][:12])
        suffix = f" | symbols: {symbols}" if symbols else ""
        lines.append(f"- {record['source']} | {record['language']} | {record['line_count']} lines{suffix}")
    summary = ", ".join(f"{key}: {value}" for key, value in sorted(extensions.items()))
    lines[2:2] = [f"Indexed files: {len(records)}", f"File types: {summary}"]
    return "\n".join(lines)


def _existing_ids(collection: Any) -> tuple[set[str], dict[str, set[str]]]:
    existing = collection.get(include=["metadatas"])
    if not isinstance(existing, dict):
        raise RuntimeError("ChromaDB returned malformed existing-index metadata")
    ids = existing.get("ids", []) or []
    metas = existing.get("metadatas", []) or []
    if not isinstance(ids, list) or not isinstance(metas, list):
        raise RuntimeError("ChromaDB returned malformed existing-index metadata")
    by_source: dict[str, set[str]] = {}
    normalized_ids: set[str] = set()
    for index, item_id in enumerate(ids):
        if not isinstance(item_id, str) or not item_id:
            raise RuntimeError("ChromaDB returned an invalid existing chunk ID")
        metadata = metas[index] if index < len(metas) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        source = str((metadata or {}).get("source", ""))
        by_source.setdefault(source, set()).add(item_id)
        normalized_ids.add(item_id)
    return normalized_ids, by_source


def _index_repository(
    root: str,
    collection_name: str,
    model: str,
    now: float,
    cancellation_token: CancellationToken | None = None,
    *,
    refresh_reason: str = "forced or index older than 24 hours",
) -> dict:
    files, discovery_skips = _discover_files(root, cancellation_token)
    if not files:
        return {"error": "No supported UTF-8 source or project files were found", "codebase_path": root}
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=collection_name)
    except Exception as exc:
        return {"error": f"Could not open ChromaDB: {exc}", "persist_directory": CHROMA_DIR}

    try:
        previous_ids, previous_by_source = _existing_ids(collection)
    except Exception as exc:
        return {
            "error": f"Could not inspect the existing codebase index: {exc}",
            "codebase_path": root,
            "collection": collection_name,
            "preserved": True,
        }
    active_ids: set[str] = set()
    records: list[dict] = []
    skipped = list(discovery_skips)
    for item in discovery_skips:
        source_path = str(item.get("file") or "")
        try:
            source = Path(source_path).relative_to(root).as_posix()
        except (TypeError, ValueError):
            source = os.path.relpath(source_path, root).replace(os.sep, "/")
        active_ids.update(previous_by_source.get(source, set()))
    indexed_chunks = 0

    for path in files:
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        relative = path.relative_to(root).as_posix()
        try:
            text = _read_source(path)
            extension = path.suffix.lower()
            language = LANGUAGES.get(extension, extension.lstrip(".").upper() or "Text")
            symbols = _symbol_hints(text)
            record = {
                "source": relative,
                "extension": extension,
                "language": language,
                "line_count": text.count("\n") + (1 if text else 0),
                "symbols": symbols,
            }
            if not text.strip():
                # Emptying a tracked file is a valid repository change. Do not
                # preserve its old chunks as though the read or embedding failed.
                records.append(record)
                continue
            chunks = chunk_text_with_offsets(text, chunk_size=2200, chunk_overlap=300)
            if not chunks:
                raise ValueError("source produced no indexable text")
            ids = [f"{hashlib.sha256(relative.encode()).hexdigest()[:20]}::{item['index']}" for item in chunks]
            documents = [
                f"File: {relative}\nLanguage: {language}\n\n{item['text']}"
                for item in chunks
            ]
            metadatas = [{
                "source": relative,
                "source_path": str(path),
                "filename": path.name,
                "extension": extension,
                "language": language,
                "chunk_index": item["index"],
                "char_start": item["char_start"],
                "char_end": item["char_end"],
                "line_start": _line_number(text, item["char_start"]),
                "line_end": _line_number(text, item["char_end"]),
                "kind": "code",
            } for item in chunks]
            embeddings: list[list[float]] = []
            for start in range(0, len(documents), DEFAULT_BATCH_SIZE):
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                embeddings.extend(embed_texts(
                    documents[start:start + DEFAULT_BATCH_SIZE],
                    model=model,
                    cancellation_token=cancellation_token,
                ))
            collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            active_ids.update(ids)
            indexed_chunks += len(ids)
            records.append(record)
        except OperationCancelled:
            raise
        except Exception as exc:
            active_ids.update(previous_by_source.get(relative, set()))
            skipped.append({"file": relative, "reason": f"previous index preserved: {exc}"})

    if not records:
        return {
            "error": "Every source file failed to index; any previous index was preserved",
            "codebase_path": root,
            "skipped_files": skipped[:30],
        }

    if skipped:
        # A newly generated overview would falsely omit files whose last good
        # chunks were deliberately preserved. Keep the previous overview until
        # every supported source has been read and embedded successfully.
        active_ids.update(previous_by_source.get("[repository overview]", set()))
    else:
        overview = _overview_document(root, records)
        overview_chunks = chunk_text_with_offsets(overview, chunk_size=4000, chunk_overlap=100)
        overview_ids = [f"__repository_overview__::{item['index']}" for item in overview_chunks]
        overview_docs = [item["text"] for item in overview_chunks]
        try:
            collection.upsert(
                ids=overview_ids,
                documents=overview_docs,
                embeddings=embed_texts(
                    overview_docs,
                    model=model,
                    cancellation_token=cancellation_token,
                ),
                metadatas=[{
                    "source": "[repository overview]", "source_path": root, "filename": "[repository overview]",
                    "extension": "", "language": "Repository map", "chunk_index": item["index"],
                    "char_start": item["char_start"], "char_end": item["char_end"],
                    "line_start": _line_number(overview, item["char_start"]),
                    "line_end": _line_number(overview, item["char_end"]),
                    "kind": "overview",
                } for item in overview_chunks],
            )
            active_ids.update(overview_ids)
            indexed_chunks += len(overview_ids)
        except OperationCancelled:
            raise
        except Exception as exc:
            skipped.append({"file": "[repository overview]", "reason": str(exc)})
            active_ids.update(previous_by_source.get("[repository overview]", set()))

    stale_ids = previous_ids - active_ids
    if stale_ids:
        try:
            collection.delete(ids=sorted(stale_ids))
        except Exception as exc:
            return {
                "error": f"The code index was updated, but stale chunks could not be removed: {exc}",
                "codebase_path": root,
                "collection": collection_name,
                "preserved": True,
                "stale_chunks": len(stale_ids),
            }

    try:
        state = _load_state()
    except PersistenceError as exc:
        return {
            "error": str(exc),
            "state_file": STATE_FILE,
            "preserved": True,
        }
    complete = not skipped
    previous_entry = state.get(root) if isinstance(state.get(root), dict) else {}
    state[root] = {
        **previous_entry,
        "collection": collection_name,
        "indexed_files": len(records),
        "indexed_chunks": indexed_chunks,
        "complete": complete,
        "last_attempted_at": now,
    }
    if complete:
        state[root]["last_indexed_at"] = now
    try:
        _save_state(state)
    except (OSError, PersistenceError) as exc:
        return {
            "error": f"The code index was updated, but refresh state could not be saved: {exc}",
            "state_file": STATE_FILE,
            "preserved": True,
            "collection": collection_name,
        }
    return {
        "refreshed": complete,
        "complete": complete,
        "needs_retry": not complete,
        "refresh_reason": refresh_reason,
        "codebase_path": root,
        "collection": collection_name,
        "indexed_at": _utc_iso(now),
        "indexed_files": len(records),
        "indexed_chunks": indexed_chunks,
        "removed_chunks": len(stale_ids),
        "skipped_count": len(skipped),
        "skipped_files": skipped[:30],
        "persist_directory": CHROMA_DIR,
    }


def _refresh_status(
    root: str,
    now: float,
    *,
    index_available: bool | None = None,
) -> dict:
    entry = _load_state().get(root)
    if not isinstance(entry, dict):
        return {
            "needs_refresh": True,
            "reason": "not indexed",
            **({"index_available": index_available} if index_available is not None else {}),
        }
    if entry.get("complete") is False:
        return {
            **entry,
            "needs_refresh": True,
            "reason": "previous indexing attempt was incomplete",
            **({"index_available": index_available} if index_available is not None else {}),
        }
    try:
        indexed_at = float(entry["last_indexed_at"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return {
            **entry,
            "needs_refresh": True,
            "reason": "index timestamp missing",
            **({"index_available": index_available} if index_available is not None else {}),
        }
    if not math.isfinite(indexed_at):
        return {
            **entry,
            "needs_refresh": True,
            "reason": "index timestamp is invalid",
            **({"index_available": index_available} if index_available is not None else {}),
        }
    try:
        indexed_at_iso = _utc_iso(indexed_at)
        refresh_after_iso = _utc_iso(indexed_at + REFRESH_SECONDS)
    except (OSError, OverflowError, ValueError):
        return {
            **entry,
            "needs_refresh": True,
            "reason": "index timestamp is invalid",
            **({"index_available": index_available} if index_available is not None else {}),
        }
    age = max(0.0, now - indexed_at)
    needs_refresh = age >= REFRESH_SECONDS
    reason = "index is at least 24 hours old" if needs_refresh else "inside 24-hour cooldown"
    if index_available is False:
        needs_refresh = True
        reason = "index collection is missing or unavailable"
    return {
        **entry,
        "needs_refresh": needs_refresh,
        "reason": reason,
        "last_indexed_at": indexed_at_iso,
        "age_seconds": round(age),
        "refresh_after": refresh_after_iso,
        **({"index_available": index_available} if index_available is not None else {}),
    }


def _index_available(collection_name: str) -> bool:
    try:
        return get_chroma_client().get_collection(name=collection_name).count() > 0
    except Exception:
        return False


def _search(
    root: str,
    collection_name: str,
    query: str,
    model: str,
    top_k: int,
    max_chars: int,
    cancellation_token: CancellationToken | None = None,
) -> dict:
    try:
        collection = get_chroma_client().get_collection(name=collection_name)
        count = collection.count()
        if not count:
            return {"error": "The codebase index is empty", "collection": collection_name}
        results = collection.query(
            query_embeddings=[embed_query(
                query,
                model=model,
                cancellation_token=cancellation_token,
            )],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
    except OperationCancelled:
        raise
    except Exception as exc:
        return {"error": f"Could not search codebase index: {exc}", "collection": collection_name}

    if not isinstance(results, dict):
        return {"error": "Codebase search returned malformed results", "collection": collection_name}

    def first_result_list(name: str) -> list:
        values = results.get(name)
        if not isinstance(values, list) or not values:
            return []
        return values[0] if isinstance(values[0], list) else []

    documents = first_result_list("documents")
    metadatas = first_result_list("metadatas")
    distances = first_result_list("distances")
    matches: list[dict] = []
    context_parts: list[str] = []
    used = 0
    for rank, document in enumerate(documents, start=1):
        metadata = metadatas[rank - 1] if rank <= len(metadatas) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        distance = distances[rank - 1] if rank <= len(distances) else None
        try:
            distance = float(distance) if distance is not None else None
        except (TypeError, ValueError, OverflowError):
            distance = None
        if isinstance(distance, float) and not math.isfinite(distance):
            distance = None
        document = str(document or "")
        header = f"Source: {metadata.get('source', 'unknown')} | lines {metadata.get('line_start', '?')}-{metadata.get('line_end', '?')}"
        entry = f"{header}\n{document}\n---"
        separator_size = 2 if context_parts else 0
        remaining = max_chars - used - separator_size
        if remaining <= 0:
            break
        if len(entry) > remaining:
            if remaining <= 3:
                break
            entry = entry[:remaining - 3].rstrip()
            if not entry:
                break
            entry += "..."
        context_parts.append(entry)
        used += separator_size + len(entry)
        matches.append({
            "rank": rank,
            "source": metadata.get("source"),
            "source_path": metadata.get("source_path"),
            "language": metadata.get("language"),
            "line_start": metadata.get("line_start"),
            "line_end": metadata.get("line_end"),
            "distance": distance,
        })
    return {
        "codebase_path": root,
        "collection": collection_name,
        "query": query,
        "match_count": len(matches),
        "matches": matches,
        "context": "\n\n".join(context_parts),
        "guidance": (
            "Answer from the returned code context, cite source paths and line ranges, and distinguish observed code from inference. "
            "For fault or optimisation reviews, explain impact and propose a concrete fix; retrieve another focused query if evidence is incomplete."
        ),
    }


def codebase_indexer(
    codebase_path: str,
    query: str | None = None,
    action: str = "query",
    force_reindex: bool = False,
    model: str = DEFAULT_EMBED_MODEL,
    top_k: int = DEFAULT_TOP_K,
    max_chars: int = DEFAULT_MAX_CHARS,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """Index, inspect, or semantically query a local source repository.

    Query calls automatically refresh the repository when it has never been indexed
    or its successful index is at least 24 hours old.
    """
    if not codebase_path or not str(codebase_path).strip():
        return _json({"error": "codebase_path is required"})
    root = os.path.realpath(os.path.abspath(os.path.expanduser(str(codebase_path).strip())))
    if not os.path.isdir(root):
        return _json({"error": f"codebase path is not a directory: {root}"})
    action = str(action or "query").strip().lower()
    if action not in {"query", "index", "status"}:
        return _json({"error": "action must be query, index, or status"})
    if query is not None and len(str(query)) > 4000:
        return _json({"error": "query exceeds the 4000-character limit"})
    if (
        not isinstance(model, str)
        or not model.strip()
        or len(model) > 200
        or any(character.isspace() or ord(character) < 32 for character in model)
    ):
        return _json({
            "error": "model must be a non-empty name of at most 200 characters without whitespace or control characters"
        })
    model = model.strip()
    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    collection_name = _collection_name(root)
    now = time.time()
    index_available = _index_available(collection_name)
    try:
        status = _refresh_status(root, now, index_available=index_available)
    except PersistenceError as exc:
        return _json({"error": str(exc), "state_file": STATE_FILE, "preserved": True})
    if action == "status":
        return _json({**status, "codebase_path": root, "collection": collection_name})

    refresh = bool(force_reindex) or action == "index" or status["needs_refresh"]
    index_result: dict | None = None
    if refresh:
        while not _INDEX_LOCK.acquire(timeout=0.1):
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
        try:
            # A second query may have waited while the first refreshed this repo.
            # Recheck inside the lock so simultaneous first-use requests do not
            # duplicate all embedding work.
            locked_index_available = _index_available(collection_name)
            try:
                locked_status = _refresh_status(
                    root,
                    time.time(),
                    index_available=locked_index_available,
                )
            except PersistenceError as exc:
                return _json({"error": str(exc), "state_file": STATE_FILE, "preserved": True})
            should_refresh = (
                bool(force_reindex)
                or action == "index"
                or locked_status["needs_refresh"]
            )
            if should_refresh:
                if force_reindex:
                    refresh_reason = "forced by caller"
                elif action == "index":
                    refresh_reason = "explicit index action"
                else:
                    refresh_reason = str(locked_status.get("reason") or "refresh required")
                index_result = _index_repository(
                    root,
                    collection_name,
                    model,
                    time.time(),
                    cancellation_token,
                    refresh_reason=refresh_reason,
                )
                if "error" in index_result:
                    return _json(index_result)
            else:
                status = locked_status
        finally:
            _INDEX_LOCK.release()
    if action == "index" and not query:
        return _json(index_result or {"refreshed": False, "codebase_path": root, "collection": collection_name})
    if not query or not str(query).strip():
        return _json({"error": "query is required when action is query"})

    try:
        top_k = int(top_k or DEFAULT_TOP_K)
    except (TypeError, ValueError, OverflowError):
        top_k = DEFAULT_TOP_K
    try:
        max_chars = int(max_chars or DEFAULT_MAX_CHARS)
    except (TypeError, ValueError, OverflowError):
        max_chars = DEFAULT_MAX_CHARS
    top_k = max(1, min(20, top_k))
    max_chars = max(1000, min(30000, max_chars))
    result = _search(
        root,
        collection_name,
        str(query).strip(),
        model,
        top_k,
        max_chars,
        cancellation_token,
    )
    result["refresh"] = index_result or {**status, "refreshed": False}
    return _json(result)
