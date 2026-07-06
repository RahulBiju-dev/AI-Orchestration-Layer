"""Code-aware repository indexing and retrieval backed by the local Chroma vault."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.vault_embeddings import DEFAULT_EMBED_MODEL, embed_query, embed_texts
from tools.vault_indexer import CHROMA_DIR, DATA_DIR, chunk_text_with_offsets, get_chroma_client

REFRESH_SECONDS = 24 * 60 * 60
DEFAULT_TOP_K = 10
DEFAULT_MAX_CHARS = 14000
DEFAULT_BATCH_SIZE = 16
MAX_FILES = 5000
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 50 * 1024 * 1024
STATE_FILE = os.path.join(DATA_DIR, "codebase_indexes.json")
_STATE_LOCK = threading.RLock()
_INDEX_LOCK = threading.RLock()

CODE_EXTENSIONS = {
    ".c", ".cc", ".clj", ".cljs", ".cmake", ".coffee", ".cpp", ".cs", ".css",
    ".dart", ".ex", ".exs", ".fs", ".fsx", ".go", ".graphql", ".h", ".hpp",
    ".html", ".htm", ".java", ".jl", ".js", ".jsx", ".json", ".kt", ".kts",
    ".less", ".lua", ".md", ".markdown", ".mjs", ".mm", ".php", ".pl", ".pm",
    ".proto", ".py", ".pyi", ".r", ".rb", ".rs", ".rst", ".sass", ".scala",
    ".scss", ".sh", ".sql", ".svelte", ".swift", ".toml", ".ts", ".tsx", ".vue",
    ".xml", ".yaml", ".yml", ".zig",
}
SPECIAL_FILES = {
    "dockerfile", "makefile", "procfile", "rakefile", "gemfile", "cmakelists.txt",
    "package.json", "pyproject.toml", "requirements.txt", "cargo.toml", "go.mod",
    "composer.json", "build.gradle", "settings.gradle", ".gitignore", ".dockerignore",
}
IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".chroma", ".idea", ".vscode", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".nox", ".venv", "venv", "env",
    "__pycache__", "node_modules", "bower_components", "vendor", "dist", "build",
    "coverage", "htmlcov", "target", "out", ".next", ".nuxt", ".cache",
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
            with open(STATE_FILE, "r", encoding="utf-8") as stream:
                value = json.load(stream)
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}


def _save_state(state: dict) -> None:
    with _STATE_LOCK:
        directory = os.path.dirname(STATE_FILE)
        os.makedirs(directory, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix="codebase-index-", suffix=".json", dir=directory)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(state, stream, indent=2, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, STATE_FILE)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _is_indexable(path: Path) -> bool:
    return path.suffix.lower() in CODE_EXTENSIONS or path.name.casefold() in SPECIAL_FILES


def _discover_files(root: str) -> tuple[list[Path], list[dict]]:
    files: list[Path] = []
    skipped: list[dict] = []
    total_bytes = 0
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in IGNORED_DIRS and not os.path.islink(os.path.join(current, name)))
        for name in sorted(names):
            path = Path(current, name)
            if not _is_indexable(path) or path.is_symlink():
                continue
            try:
                size = path.stat().st_size
            except OSError as exc:
                skipped.append({"file": str(path), "reason": str(exc)})
                continue
            if size > MAX_FILE_BYTES:
                skipped.append({"file": str(path), "reason": "file exceeds 2 MiB"})
                continue
            if len(files) >= MAX_FILES or total_bytes + size > MAX_TOTAL_BYTES:
                skipped.append({"file": str(path), "reason": "repository indexing limit reached"})
                continue
            files.append(path)
            total_bytes += size
    return files, skipped


def _read_source(path: Path) -> str:
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        raise ValueError("binary file")
    return raw.decode("utf-8")


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
    try:
        existing = collection.get(include=["metadatas"])
    except Exception:
        return set(), {}
    ids = existing.get("ids", []) or []
    metas = existing.get("metadatas", []) or []
    by_source: dict[str, set[str]] = {}
    for item_id, metadata in zip(ids, metas):
        source = str((metadata or {}).get("source", ""))
        by_source.setdefault(source, set()).add(item_id)
    return set(ids), by_source


def _index_repository(root: str, collection_name: str, model: str, now: float) -> dict:
    files, discovery_skips = _discover_files(root)
    if not files:
        return {"error": "No supported UTF-8 source or project files were found", "codebase_path": root}
    try:
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=collection_name)
    except Exception as exc:
        return {"error": f"Could not open ChromaDB: {exc}", "persist_directory": CHROMA_DIR}

    previous_ids, previous_by_source = _existing_ids(collection)
    active_ids: set[str] = set()
    records: list[dict] = []
    skipped = list(discovery_skips)
    indexed_chunks = 0

    for path in files:
        relative = path.relative_to(root).as_posix()
        try:
            text = _read_source(path)
            chunks = chunk_text_with_offsets(text, chunk_size=2200, chunk_overlap=300)
            if not chunks:
                raise ValueError("empty file")
            extension = path.suffix.lower()
            symbols = _symbol_hints(text)
            ids = [f"{hashlib.sha256(relative.encode()).hexdigest()[:20]}::{item['index']}" for item in chunks]
            documents = [
                f"File: {relative}\nLanguage: {LANGUAGES.get(extension, extension.lstrip('.').upper() or 'Text')}\n\n{item['text']}"
                for item in chunks
            ]
            metadatas = [{
                "source": relative,
                "source_path": str(path),
                "filename": path.name,
                "extension": extension,
                "language": LANGUAGES.get(extension, extension.lstrip(".").upper() or "Text"),
                "chunk_index": item["index"],
                "char_start": item["char_start"],
                "char_end": item["char_end"],
                "line_start": _line_number(text, item["char_start"]),
                "line_end": _line_number(text, item["char_end"]),
                "kind": "code",
            } for item in chunks]
            embeddings: list[list[float]] = []
            for start in range(0, len(documents), DEFAULT_BATCH_SIZE):
                embeddings.extend(embed_texts(documents[start:start + DEFAULT_BATCH_SIZE], model=model))
            collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            active_ids.update(ids)
            indexed_chunks += len(ids)
            records.append({
                "source": relative,
                "extension": extension,
                "language": metadatas[0]["language"],
                "line_count": text.count("\n") + (1 if text else 0),
                "symbols": symbols,
            })
        except Exception as exc:
            active_ids.update(previous_by_source.get(relative, set()))
            skipped.append({"file": relative, "reason": f"previous index preserved: {exc}"})

    if not records:
        return {
            "error": "Every source file failed to index; any previous index was preserved",
            "codebase_path": root,
            "skipped_files": skipped[:30],
        }

    overview = _overview_document(root, records)
    overview_chunks = chunk_text_with_offsets(overview, chunk_size=4000, chunk_overlap=100)
    overview_ids = [f"__repository_overview__::{item['index']}" for item in overview_chunks]
    overview_docs = [item["text"] for item in overview_chunks]
    try:
        collection.upsert(
            ids=overview_ids, documents=overview_docs, embeddings=embed_texts(overview_docs, model=model),
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
    except Exception as exc:
        skipped.append({"file": "[repository overview]", "reason": str(exc)})
        active_ids.update(previous_by_source.get("[repository overview]", set()))

    stale_ids = previous_ids - active_ids
    if stale_ids:
        try:
            collection.delete(ids=sorted(stale_ids))
        except Exception as exc:
            skipped.append({"file": "[stale index entries]", "reason": str(exc)})

    state = _load_state()
    state[root] = {
        "collection": collection_name,
        "last_indexed_at": now,
        "indexed_files": len(records),
        "indexed_chunks": indexed_chunks,
    }
    _save_state(state)
    return {
        "refreshed": True,
        "refresh_reason": "forced or index older than 24 hours",
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


def _refresh_status(root: str, now: float) -> dict:
    entry = _load_state().get(root)
    if not isinstance(entry, dict):
        return {"needs_refresh": True, "reason": "not indexed"}
    try:
        indexed_at = float(entry["last_indexed_at"])
    except (KeyError, TypeError, ValueError):
        return {"needs_refresh": True, "reason": "index timestamp missing"}
    age = max(0.0, now - indexed_at)
    return {
        "needs_refresh": age >= REFRESH_SECONDS,
        "reason": "index is at least 24 hours old" if age >= REFRESH_SECONDS else "inside 24-hour cooldown",
        "last_indexed_at": _utc_iso(indexed_at),
        "age_seconds": round(age),
        "refresh_after": _utc_iso(indexed_at + REFRESH_SECONDS),
        **entry,
    }


def _index_available(collection_name: str) -> bool:
    try:
        return get_chroma_client().get_collection(name=collection_name).count() > 0
    except Exception:
        return False


def _search(root: str, collection_name: str, query: str, model: str, top_k: int, max_chars: int) -> dict:
    try:
        collection = get_chroma_client().get_collection(name=collection_name)
        count = collection.count()
        if not count:
            return {"error": "The codebase index is empty", "collection": collection_name}
        results = collection.query(
            query_embeddings=[embed_query(query, model=model)],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        return {"error": f"Could not search codebase index: {exc}", "collection": collection_name}

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    matches: list[dict] = []
    context_parts: list[str] = []
    used = 0
    for rank, document in enumerate(documents, start=1):
        metadata = metadatas[rank - 1] if rank <= len(metadatas) else {}
        distance = distances[rank - 1] if rank <= len(distances) else None
        header = f"Source: {metadata.get('source', 'unknown')} | lines {metadata.get('line_start', '?')}-{metadata.get('line_end', '?')}"
        entry = f"{header}\n{document}\n---"
        remaining = max_chars - used
        if remaining <= 0:
            break
        entry = entry if len(entry) <= remaining else entry[:max(0, remaining - 3)].rstrip() + "..."
        context_parts.append(entry)
        used += len(entry)
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

    now = time.time()
    collection_name = _collection_name(root)
    status = _refresh_status(root, now)
    if action == "status":
        return _json({"codebase_path": root, "collection": collection_name, **status})

    refresh = bool(force_reindex) or action == "index" or status["needs_refresh"] or not _index_available(collection_name)
    index_result: dict | None = None
    if refresh:
        with _INDEX_LOCK:
            # A second query may have waited while the first refreshed this repo.
            # Recheck inside the lock so simultaneous first-use requests do not
            # duplicate all embedding work.
            locked_status = _refresh_status(root, time.time())
            should_refresh = (
                bool(force_reindex)
                or action == "index"
                or locked_status["needs_refresh"]
                or not _index_available(collection_name)
            )
            if should_refresh:
                index_result = _index_repository(root, collection_name, model, now)
                if "error" in index_result:
                    return _json(index_result)
            else:
                status = locked_status
    if action == "index" and not query:
        return _json(index_result or {"refreshed": False, "codebase_path": root, "collection": collection_name})
    if not query or not str(query).strip():
        return _json({"error": "query is required when action is query"})

    try:
        top_k = int(top_k or DEFAULT_TOP_K)
    except (TypeError, ValueError):
        top_k = DEFAULT_TOP_K
    try:
        max_chars = int(max_chars or DEFAULT_MAX_CHARS)
    except (TypeError, ValueError):
        max_chars = DEFAULT_MAX_CHARS
    top_k = max(1, min(20, top_k))
    max_chars = max(1000, min(30000, max_chars))
    result = _search(root, collection_name, str(query).strip(), model, top_k, max_chars)
    result["refresh"] = index_result or {"refreshed": False, **status}
    return _json(result)
