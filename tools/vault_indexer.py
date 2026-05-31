"""Vault indexer: chunk local files and index embeddings into ChromaDB using Ollama."""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import chromadb

from tools.document import extract_document_text
from tools.vault_embeddings import DEFAULT_EMBED_MODEL, embed_texts

SUPPORTED_INDEX_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".pdf",
    ".docx",
}
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 250
DEFAULT_BATCH_SIZE = 16
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".chroma")
VAULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vaults")


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


def sanitize_collection_name(name: str) -> str:
    """Sanitize the collection name to meet ChromaDB requirements.
    Expected a name containing 3-63 characters from [a-zA-Z0-9._-],
    starting and ending with an alphanumeric character.
    """
    if not name:
        return "vault"
    # Replace invalid chars with underscores
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    # Strip leading/trailing non-alphanumeric chars
    name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
    name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
    
    if not name:
        return "vault"
    if len(name) < 3:
        name = name.ljust(3, '0')
        
    return name[:63]


def get_chroma_client(path: str | None = None):
    """Return a persistent Chroma client shared by index and search tools."""
    persist_directory = path or CHROMA_DIR
    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=persist_directory)

    from chromadb.config import Settings

    return chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory,
    ))


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    return [chunk["text"] for chunk in chunk_text_with_offsets(text, chunk_size, chunk_overlap)]


def chunk_text_with_offsets(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Split text into readable chunks with character offsets.

    The splitter prefers paragraph/newline/sentence boundaries near the end of
    each window, which gives retrieval snippets more coherent context than hard
    character slicing.
    """
    chunk_size = _positive_int(chunk_size, DEFAULT_CHUNK_SIZE, minimum=500, maximum=20000)
    chunk_overlap = _positive_int(chunk_overlap, DEFAULT_CHUNK_OVERLAP, minimum=0, maximum=max(0, chunk_size // 2))
    text = text.replace("\r\n", "\n")
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)
    while start < length:
        hard_end = min(length, start + chunk_size)
        end = hard_end
        if hard_end < length:
            window = text[start:hard_end]
            boundary_candidates = [
                window.rfind("\n\n"),
                window.rfind("\n"),
                window.rfind(". "),
                window.rfind("? "),
                window.rfind("! "),
            ]
            boundary = max(boundary_candidates)
            if boundary >= chunk_size // 2:
                end = start + boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "index": len(chunks),
                "text": chunk,
                "char_start": start,
                "char_end": end,
            })

        if end >= length:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    match = re.match(r"^---\s*\n.*?\n---\s*\n", text, flags=re.DOTALL)
    return text[match.end():].lstrip() if match else text


def extract_pdf_with_vision(path: str) -> tuple[str, dict]:
    """Extract regular text and visual descriptions page-by-page from a PDF in a memory-safe batching loop."""
    try:
        import pypdf
        from pdf2image import convert_from_path
        import tempfile
        from tools.vision_describer import describe_image
        
        text_stream = []
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            total_pages = len(reader.pages)
            
        chunk_size = 10
        for i in range(0, total_pages, chunk_size):
            first_page = i + 1
            last_page = min(i + chunk_size, total_pages)
            
            # Extract regular text using pypdf
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page_num in range(first_page - 1, last_page):
                    try:
                        page_text = reader.pages[page_num].extract_text()
                        if page_text:
                            text_stream.append(f"--- Page {page_num + 1} Text ---\n{page_text.strip()}\n")
                    except Exception:
                        pass
                        
            # Extract multimodal vision text
            try:
                images = convert_from_path(path, first_page=first_page, last_page=last_page)
                for img_idx, img in enumerate(images):
                    page_num = first_page + img_idx
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
                        temp_img_path = temp_img.name
                        img.save(temp_img_path, "PNG")
                    try:
                        desc = describe_image(temp_img_path)
                        text_stream.append(f"--- Page {page_num} Visual Description ---\n{desc}\n")
                    finally:
                        os.remove(temp_img_path)
            except Exception as e:
                text_stream.append(f"Error processing visual descriptions for pages {first_page}-{last_page}: {e}")
                
        text = "\n".join(text_stream)
        return text, {"document_type": "pdf", "char_count": len(text)}
    except Exception as e:
        return f"Error reading PDF: {e}", {"document_type": "pdf", "char_count": 0}


def _read_text_for_index(path: str) -> tuple[str, dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf_with_vision(path)
            
    if ext in {".pdf", ".docx"}:
        return extract_document_text(path)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = _strip_frontmatter(text)
    return text, {"document_type": ext.lstrip(".") or "text", "char_count": len(text)}


def _iter_indexable_files(vault_path: str):
    for root, dirs, files in os.walk(vault_path):
        dirs[:] = [name for name in dirs if name not in {".git", ".chroma", "__pycache__", "node_modules"}]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_INDEX_EXTENSIONS:
                yield os.path.join(root, fname)


def _flush_batch(collection, ids: list[str], docs: list[str], metadatas: list[dict], model: str) -> int:
    if not docs:
        return 0
    embeddings = embed_texts(docs, model=model)
    collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
    return len(docs)


def _delete_existing_source(collection, source: str) -> None:
    try:
        collection.delete(where={"source": source})
    except Exception:
        # Older Chroma versions may raise when no rows match. Stale chunks are
        # less harmful than failing the entire indexing operation.
        pass


def index_vault(
    vault_path: Optional[str] = None,
    collection_name: str = "vault",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    file_path: Optional[str] = None,
    collection: Optional[str] = None,
):
    """Index either a vault folder or a single file.

    Returns a JSON string with status and next-step guidance for tool consumers.
    """
    if collection:
        collection_name = collection
    elif collection_name == "vault":
        # Auto-derive a meaningful name instead of the generic "vault"
        if file_path:
            collection_name = os.path.splitext(os.path.basename(file_path))[0]
        elif vault_path:
            collection_name = os.path.basename(os.path.abspath(vault_path))

    collection_name = sanitize_collection_name(collection_name)

    if not vault_path:
        vault_path = os.path.dirname(file_path) if file_path else VAULTS_DIR
    vault_path = os.path.abspath(vault_path)
    if not os.path.exists(vault_path):
        return _json({"error": f"vault path not found: {vault_path}"})

    if file_path:
        candidates = [os.path.abspath(file_path)]
    else:
        if not os.path.isdir(vault_path):
            return _json({"error": f"vault_path must be a folder when file_path is not provided: {vault_path}"})
        candidates = list(_iter_indexable_files(vault_path))

    batch_size = _positive_int(batch_size, DEFAULT_BATCH_SIZE, minimum=1, maximum=128)
    chunk_size = _positive_int(chunk_size, DEFAULT_CHUNK_SIZE, minimum=500, maximum=20000)
    chunk_overlap = _positive_int(chunk_overlap, DEFAULT_CHUNK_OVERLAP, minimum=0, maximum=max(0, chunk_size // 2))

    client = get_chroma_client()
    collection_obj = client.get_or_create_collection(name=collection_name)

    ids: list[str] = []
    docs: list[str] = []
    metadatas: list[dict] = []
    indexed_chunks = 0
    indexed_files = 0
    skipped_files: list[dict] = []

    for path in candidates:
        if not os.path.exists(path):
            skipped_files.append({"file": path, "error": "file not found"})
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_INDEX_EXTENSIONS:
            skipped_files.append({"file": path, "error": f"unsupported extension: {ext}"})
            continue

        try:
            text, info = _read_text_for_index(path)
        except UnicodeDecodeError:
            skipped_files.append({"file": path, "error": "not UTF-8 text; use PDF/DOCX or plain text"})
            continue
        except Exception as exc:
            skipped_files.append({"file": path, "error": str(exc)})
            continue

        chunks = chunk_text_with_offsets(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            skipped_files.append({"file": path, "error": "no extractable text"})
            continue

        rel = os.path.relpath(path, vault_path) if os.path.isdir(vault_path) else os.path.basename(path)
        _delete_existing_source(collection_obj, rel)
        indexed_files += 1

        for chunk in chunks:
            chunk_index = chunk["index"]
            ids.append(f"{rel}::chunk::{chunk_index}")
            docs.append(chunk["text"])
            metadatas.append({
                "source": rel,
                "source_path": path,
                "filename": os.path.basename(path),
                "extension": ext,
                "chunk_index": chunk_index,
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "document_type": info.get("document_type", ext.lstrip(".")),
            })

            if len(docs) >= batch_size:
                indexed_chunks += _flush_batch(collection_obj, ids, docs, metadatas, model=model)
                ids, docs, metadatas = [], [], []

    if docs:
        indexed_chunks += _flush_batch(collection_obj, ids, docs, metadatas, model=model)

    # Auto-register an alias when a single file was indexed
    if file_path and indexed_files == 1:
        stem = os.path.splitext(os.path.basename(file_path))[0]
        register_vault_alias(
            alias=stem,
            collection_name=collection_name,
            file_path=os.path.abspath(file_path),
        )

    return _json({
        "collection": collection_name,
        "persist_directory": CHROMA_DIR,
        "indexed_files": indexed_files,
        "indexed_chunks": indexed_chunks,
        "skipped_files": skipped_files[:20],
        "skipped_count": len(skipped_files),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "alias": os.path.splitext(os.path.basename(file_path))[0] if file_path and indexed_files == 1 else None,
        "guidance": "Use vault_search with a focused query to retrieve relevant chunks from large indexed files.",
    })


def delete_vault_item(
    source: Optional[str] = None,
    collection_name: str = "vault",
    collection: Optional[str] = None,
    delete_collection: bool = False,
) -> str:
    """Delete indexed vault chunks by source path, or delete an entire collection."""
    if collection:
        collection_name = collection

    collection_name = sanitize_collection_name(collection_name)
    client = get_chroma_client()

    if delete_collection:
        try:
            client.delete_collection(name=collection_name)
            return _json({
                "collection": collection_name,
                "deleted_collection": True,
                "persist_directory": CHROMA_DIR,
            })
        except Exception as exc:
            return _json({"error": str(exc), "collection": collection_name, "persist_directory": CHROMA_DIR})

    if not source or not source.strip():
        return _json({"error": "source is required unless delete_collection is true", "collection": collection_name})

    raw_source = source.strip()
    possible_sources = [raw_source]
    if os.path.exists(raw_source):
        possible_sources.insert(0, os.path.abspath(raw_source))
    elif not os.path.isabs(raw_source):
        possible_sources.append(os.path.abspath(raw_source))

    try:
        collection_obj = client.get_collection(name=collection_name)
    except Exception as exc:
        return _json({"error": str(exc), "collection": collection_name, "persist_directory": CHROMA_DIR})

    deleted_ids: set[str] = set()
    attempted_filters: list[dict] = []

    for candidate in dict.fromkeys(possible_sources):
        filters = [{"source": candidate}, {"source_path": candidate}]
        for where in filters:
            attempted_filters.append(where)
            try:
                existing = collection_obj.get(where=where, include=["metadatas"])
                ids = existing.get("ids", [])
                if not ids:
                    continue
                collection_obj.delete(ids=ids)
                deleted_ids.update(ids)
            except Exception:
                continue

    return _json({
        "collection": collection_name,
        "source": raw_source,
        "deleted_chunks": len(deleted_ids),
        "deleted": len(deleted_ids) > 0,
        "attempted_filters": attempted_filters,
        "guidance": "Use /vault search to confirm the source no longer appears in results.",
    })


def list_vaults() -> str:
    """List existing ChromaDB vault collections with basic index counts."""
    client = get_chroma_client()
    try:
        collections = client.list_collections()
    except Exception as exc:
        return _json({"error": str(exc), "persist_directory": CHROMA_DIR})

    vaults: list[dict] = []
    for item in collections:
        name = getattr(item, "name", item)
        if not isinstance(name, str):
            continue

        chunk_count = None
        try:
            collection_obj = client.get_collection(name=name)
            chunk_count = collection_obj.count()
        except Exception:
            pass

        vaults.append({
            "collection": name,
            "indexed_chunks": chunk_count,
        })

    vaults.sort(key=lambda item: item["collection"].lower())
    return _json({
        "persist_directory": CHROMA_DIR,
        "vault_count": len(vaults),
        "vaults": vaults,
    })


# ── Vault alias registry ──────────────────────────────────────────────
# Maps human-friendly names (e.g. "DAA Notes") to collection names and
# file paths so that users can reference vaults without remembering the
# sanitized ChromaDB collection name.

_ALIAS_FILE = os.path.join(VAULTS_DIR, ".vault_aliases.json")


def _load_aliases() -> dict:
    """Load the alias registry from disk."""
    if os.path.isfile(_ALIAS_FILE):
        try:
            with open(_ALIAS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_aliases(aliases: dict) -> None:
    """Persist the alias registry to disk."""
    os.makedirs(VAULTS_DIR, exist_ok=True)
    with open(_ALIAS_FILE, "w", encoding="utf-8") as f:
        json.dump(aliases, f, indent=2, ensure_ascii=False)


def register_vault_alias(alias: str, collection_name: str, file_path: str | None = None) -> None:
    """Register a human-friendly alias for a vault collection."""
    aliases = _load_aliases()
    key = alias.strip().lower()
    aliases[key] = {
        "alias": alias.strip(),
        "collection": collection_name,
        "file_path": file_path,
    }
    _save_aliases(aliases)


def resolve_vault_alias(name: str) -> str:
    """Resolve a name to a collection name.

    Tries, in order:
      1. Exact alias match (case-insensitive)
      2. Substring alias match
      3. Return the name itself (assumed to already be a collection name)
    """
    if not name:
        return "vault"
    aliases = _load_aliases()
    key = name.strip().lower()

    # Exact match
    if key in aliases:
        return aliases[key]["collection"]

    # Substring match
    for alias_key, entry in aliases.items():
        if key in alias_key or alias_key in key:
            return entry["collection"]

    # Fall through — treat as a raw collection name
    return sanitize_collection_name(name)


def list_vault_aliases() -> str:
    """Return a JSON listing of all registered vault aliases."""
    aliases = _load_aliases()
    entries = []
    for _key, entry in sorted(aliases.items()):
        entries.append({
            "alias": entry.get("alias", _key),
            "collection": entry.get("collection"),
            "file_path": entry.get("file_path"),
        })
    return _json({
        "alias_count": len(entries),
        "aliases": entries,
    })


def rename_vault(old_name: str, new_name: str) -> str:
    """Rename a vault collection and update any aliases that reference it.

    Copies all documents, embeddings, and metadata from the old collection
    into a new one, deletes the old collection, and updates the alias
    registry so existing aliases point to the new name.

    Returns a JSON string with the result.
    """
    old_collection = sanitize_collection_name(old_name)
    new_collection = sanitize_collection_name(new_name)

    if old_collection == new_collection:
        return _json({"error": "Old and new names resolve to the same collection name.",
                       "old": old_collection, "new": new_collection})

    client = get_chroma_client()

    # Verify old collection exists
    try:
        old_coll = client.get_collection(name=old_collection)
    except Exception:
        return _json({"error": f"Collection '{old_collection}' not found.",
                       "persist_directory": CHROMA_DIR})

    count = old_coll.count()
    if count == 0:
        # Empty collection — just delete and create the new one
        client.delete_collection(name=old_collection)
        client.get_or_create_collection(name=new_collection)
    else:
        # Fetch all data from the old collection
        data = old_coll.get(include=["documents", "metadatas", "embeddings"])
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        embeddings = data.get("embeddings", [])

        # Create the new collection and upsert everything
        new_coll = client.get_or_create_collection(name=new_collection)
        new_coll.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

        # Delete the old collection
        client.delete_collection(name=old_collection)

    # Update aliases that referenced the old collection
    aliases = _load_aliases()
    updated_aliases = []
    for key, entry in aliases.items():
        if entry.get("collection") == old_collection:
            entry["collection"] = new_collection
            updated_aliases.append(entry.get("alias", key))
    if updated_aliases:
        _save_aliases(aliases)

    # Register the new name as an alias too
    register_vault_alias(new_name, new_collection)

    return _json({
        "renamed": True,
        "old_collection": old_collection,
        "new_collection": new_collection,
        "chunks_moved": count,
        "updated_aliases": updated_aliases,
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index a local vault into ChromaDB using Ollama embeddings.")
    parser.add_argument("--vault-path", default=None)
    parser.add_argument("--collection", default="vault")
    parser.add_argument("--file-path")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    print(index_vault(
        vault_path=args.vault_path,
        collection_name=args.collection,
        file_path=args.file_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model=args.model,
        batch_size=args.batch_size,
    ))
