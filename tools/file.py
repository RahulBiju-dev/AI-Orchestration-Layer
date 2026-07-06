from __future__ import annotations

import json
import math
import os
import re

DEFAULT_MAX_CHARS = 14000
DEFAULT_CHUNK_SIZE = 12000
MAX_CHARS_CAP = 50000
MAX_QUERY_MATCHES = 12
MAX_DIRECT_FILE_BYTES = 50 * 1024 * 1024
BINARY_DOCUMENT_TYPES = {
    ".pdf": "pdf",
    ".docx": "docx",
}


def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _positive_int(value: int | str | None, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def _parse_line_range(lines: str | None, total_lines: int) -> tuple[int, int] | None:
    if not lines:
        return None
    raw = lines.strip()
    try:
        if "-" in raw:
            start_raw, end_raw = raw.split("-", 1)
            start_line = int(start_raw)
            end_line = int(end_raw)
        else:
            start_line = end_line = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid line range format: {lines}. Use '10-20' or '10'.") from exc

    if start_line > end_line:
        start_line, end_line = end_line, start_line
    start_line = max(1, start_line)
    end_line = min(total_lines, end_line)
    if start_line > total_lines:
        raise ValueError(f"Line range starts after end of file. This file has {total_lines} line(s).")
    return start_line, end_line


def _line_numbered(lines: list[str], start_line: int) -> str:
    return "\n".join(f"{start_line + index:4d} | {line.rstrip()}" for index, line in enumerate(lines))


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    if not text:
        return [""]
    return [text[start:start + chunk_size] for start in range(0, len(text), chunk_size)]


def _snippet(text: str, query: str, max_chars: int = 800) -> str:
    lower = text.lower()
    query_lower = query.lower().strip()
    pos = lower.find(query_lower) if query_lower else -1
    if pos < 0:
        terms = [term for term in re.findall(r"\w+", query_lower) if len(term) > 2]
        positions = [lower.find(term) for term in terms if lower.find(term) >= 0]
        pos = min(positions) if positions else 0

    start = max(0, pos - max_chars // 3)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    result = text[start:end].strip()
    if start:
        result = "..." + result
    if end < len(text):
        result += "..."
    return result


def _search_lines(file_lines: list[str], query: str) -> list[dict]:
    query_lower = query.lower().strip()
    terms = [term for term in re.findall(r"\w+", query_lower) if len(term) > 2]
    matches = []
    for index, line in enumerate(file_lines, start=1):
        lower = line.lower()
        score = lower.count(query_lower) * 8 if query_lower else 0
        score += sum(lower.count(term) for term in terms)
        if score <= 0:
            continue

        context_start = max(1, index - 2)
        context_end = min(len(file_lines), index + 2)
        context = "".join(file_lines[context_start - 1:context_end])
        matches.append({
            "line": index,
            "score": score,
            "snippet": _snippet(context, query),
            "context_lines": f"{context_start}-{context_end}",
        })

    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:MAX_QUERY_MATCHES]


def read_file(
    file_path: str,
    lines: str | None = None,
    query: str | None = None,
    chunk: int | str | None = None,
    chunk_size: int | str = DEFAULT_CHUNK_SIZE,
    max_chars: int | str = DEFAULT_MAX_CHARS,
) -> str:
    """
    Read a text file with line, chunk, and query controls for large files.
    
    This function reads standard text files, allowing the caller to limit the
    returned output by specifying line ranges, searching for queries to find
    relevant snippets, or paging through the file in chunks.
    
    Args:
        file_path (str): The absolute or relative path to the file.
        lines (str | None): Optional line range, such as "20-80" or "42".
        query (str | None): Optional text search query; if provided, returns
            matching text snippets rather than contiguous blocks.
        chunk (int | str | None): Optional 0-based chunk number for iterating
            through large files.
        chunk_size (int | str): Approximate characters per chunk (default 12000).
        max_chars (int | str): Maximum characters returned in text fields,
            used to protect the LLM context from overflowing.
            
    Returns:
        str: A JSON-encoded string with file contents or search matches, or an error.
    """
    if not os.path.exists(file_path):
        return _json({"error": f"File not found: {file_path}"})
    if not os.path.isfile(file_path):
        return _json({"error": f"Not a file: {file_path}"})

    ext = os.path.splitext(file_path)[1].lower()
    file_size_bytes = os.path.getsize(file_path)
    max_chars_int = _positive_int(max_chars, DEFAULT_MAX_CHARS, minimum=1000, maximum=MAX_CHARS_CAP)
    chunk_size_int = _positive_int(chunk_size, DEFAULT_CHUNK_SIZE, minimum=1000, maximum=MAX_CHARS_CAP)
    if query is not None:
        query = str(query).strip()
        if len(query) > 4000:
            return _json({"error": "query exceeds the 4000-character limit"})

    if ext in BINARY_DOCUMENT_TYPES:
        from tools.document import read_document
        if lines:
            return _json({"error": "The lines parameter is for text files. Use read_document pages for PDF files."})
        return read_document(file_path, query=query, chunk=chunk, chunk_size=chunk_size, max_chars=max_chars)

    if lines:
        try:
            with open(file_path, "r", encoding="utf-8") as stream:
                total_lines = sum(1 for _ in stream)
            line_range = _parse_line_range(lines, total_lines)
            start_line, end_line = line_range
            selected = []
            with open(file_path, "r", encoding="utf-8") as stream:
                for line_number, value in enumerate(stream, start=1):
                    if line_number > end_line:
                        break
                    if line_number >= start_line:
                        selected.append(value)
            display_text = _line_numbered(selected, start_line)
            truncated = len(display_text) > max_chars_int
            if truncated:
                display_text = display_text[:max_chars_int].rstrip() + "\n\n...[Line range truncated. Request fewer lines.]"
            return _json({
                "file": file_path, "size_bytes": file_size_bytes, "line_count": total_lines,
                "mode": "lines", "lines": f"{start_line}-{end_line}", "text": display_text,
                "returned_chars": len(display_text), "truncated": truncated,
            })
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            return _json({"error": str(exc), "file": file_path, "size_bytes": file_size_bytes})

    if file_size_bytes > MAX_DIRECT_FILE_BYTES:
        return _json({
            "error": f"File is too large for direct full/query reading ({file_size_bytes} bytes)",
            "max_direct_bytes": MAX_DIRECT_FILE_BYTES,
            "guidance": "Use index_vault and vault_search, or request a specific line range.",
        })
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        return _json({"error": f"Cannot read file as UTF-8 text: {file_path}", "binary": True, "hint": "This appears to be a binary file."})
    except OSError as exc:
        return _json({"error": f"Error reading file: {exc}"})

    file_lines = text.splitlines(keepends=True)

    base = {
        "file": file_path,
        "size_bytes": file_size_bytes,
        "char_count": len(text),
        "line_count": len(file_lines),
    }

    try:
        if query:
            matches = _search_lines(file_lines, query)
            base.update({
                "mode": "query",
                "query": query,
                "matches": matches,
                "match_count": len(matches),
                "guidance": "Use the lines parameter with a returned context_lines range for fuller surrounding text.",
            })
            return _json(base)

    except ValueError as exc:
        return _json({"error": str(exc), **base})

    chunks = _chunk_text(text, chunk_size_int)
    total_chunks = len(chunks)
    if chunk is None and len(text) <= max_chars_int:
        base.update({
            "mode": "full",
            "text": text,
            "returned_chars": len(text),
            "truncated": False,
        })
        return _json(base)

    requested_chunk = _positive_int(chunk, 0, minimum=0)
    selected_chunk = min(requested_chunk, total_chunks - 1)
    selected_text = chunks[selected_chunk]
    truncated = len(selected_text) > max_chars_int
    if truncated:
        selected_text = selected_text[:max_chars_int].rstrip() + "\n\n...[Chunk truncated. Request a smaller chunk_size or use lines/query.]"

    base.update({
        "mode": "chunk",
        "text": selected_text,
        "returned_chars": len(selected_text),
        "truncated": True,
        "navigation": {
            "chunk": selected_chunk,
            "chunk_size": chunk_size_int,
            "total_chunks": total_chunks,
            "estimated_total_chunks": math.ceil(len(text) / chunk_size_int) if chunk_size_int else total_chunks,
            "next_chunk": selected_chunk + 1 if selected_chunk + 1 < total_chunks else None,
            "previous_chunk": selected_chunk - 1 if selected_chunk > 0 else None,
            "lines_parameter": "Use lines like '120-180' when the relevant location is known.",
            "query_parameter": "Use query to locate relevant lines before reading a large file.",
        },
    })
    if requested_chunk >= total_chunks:
        base["warning"] = f"Requested chunk {requested_chunk}, but only {total_chunks} chunk(s) exist; returned the last chunk."
    return _json(base)


def create_file(file_path: str, content: str) -> str:
    """
    Create a new file with the given content.

    The file is written into the project's ``vaults/`` directory so that it
    is automatically available for semantic search. A dedicated ChromaDB
    collection is created for the file and a human-friendly alias is
    registered so the user can reference the vault by name later.

    Args:
        file_path (str): The absolute or relative path where the file should be created.
            The basename is used to place the file inside the ``vaults/`` directory.
        content (str): The text content to write to the file.

    Returns:
        str: A JSON-encoded string indicating success or failure, including
            vault connection parameters like 'collection' and 'alias'.
    """
    from tools.vault_indexer import (
        VAULTS_DIR,
        index_vault,
        register_vault_alias,
        sanitize_collection_name,
    )

    try:
        basename = os.path.basename(str(file_path).strip())
        if basename in {"", ".", "..", ".vault_aliases.json"}:
            return _json({"error": "file_path must contain a valid filename"})
        if not isinstance(content, str):
            content = str(content)
        if len(content.encode("utf-8")) > 10 * 1024 * 1024:
            return _json({"error": "File content exceeds the 10 MiB creation limit"})
        vault_file_path = os.path.join(VAULTS_DIR, basename)
        os.makedirs(VAULTS_DIR, exist_ok=True)

        with open(vault_file_path, "x", encoding="utf-8") as f:
            f.write(content)

        # Derive a collection name from the filename (without extension)
        stem = os.path.splitext(basename)[0]
        collection_name = sanitize_collection_name(stem)

        # Index the file into its own vault collection
        try:
            index_result = index_vault(
                vault_path=VAULTS_DIR,
                file_path=vault_file_path,
                collection=collection_name,
            )
            parsed_index = json.loads(index_result) if isinstance(index_result, str) else index_result
        except Exception as exc:
            parsed_index = {"error": str(exc)}
        index_error = parsed_index.get("error") if isinstance(parsed_index, dict) else None
        if not index_error:
            try:
                register_vault_alias(alias=stem, collection_name=collection_name, file_path=vault_file_path)
            except Exception as exc:
                parsed_index = {**parsed_index, "alias_warning": str(exc)}

        return _json({
            "success": True,
            "indexed": not bool(index_error),
            "message": f"Created file at {vault_file_path}" + (" but indexing failed" if index_error else " and indexed it"),
            "vault_path": vault_file_path,
            "collection": collection_name,
            "alias": stem,
            "index_result": parsed_index,
            "hint": f"Use vault_search with collection='{collection_name}' or reference the alias '{stem}' to search this file.",
        })
    except FileExistsError:
        return _json({"error": f"File already exists and was not overwritten: {vault_file_path}"})
    except Exception as exc:
        return _json({"error": f"Error creating file: {exc}"})
