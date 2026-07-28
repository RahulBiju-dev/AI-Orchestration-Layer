"""Vault indexer: chunk local files and index embeddings into ChromaDB using Ollama."""

from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Optional

from agent.cancellation import CancellationToken, OperationCancelled
from agent.persistence import PersistenceError, atomic_write_json, read_json_preserved
from agent.platform_runtime import get_runtime_paths

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
DEFAULT_PDF_PAGES_PER_RUN = 20
PDF_VISION_TEXT_THRESHOLD = 500
MAX_RETURNED_WARNING_SAMPLES = 3
MAX_RETURNED_FAILED_PAGES = 25
DATA_DIR = str(get_runtime_paths().data_dir)
VAULTS_DIR = os.path.join(DATA_DIR, "vaults")


def _select_chroma_directory(data_dir: str, vaults_dir: str) -> str:
    """Keep an existing legacy store; use the managed layout only when fresh."""
    legacy_directory = os.path.join(data_dir, ".chroma")
    managed_directory = os.path.join(vaults_dir, ".chroma")
    return legacy_directory if os.path.exists(legacy_directory) else managed_directory


# Preserve an existing pre-vault-folder store in place. New installations keep
# the Chroma database with the rest of the managed vault state; no user data is
# silently migrated, copied, merged, or deleted.
CHROMA_DIR = _select_chroma_directory(DATA_DIR, VAULTS_DIR)
_ALIAS_LOCK = threading.RLock()
_INDEX_JOB_LOCK = threading.RLock()


def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _positive_int(value: int | str | None, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError, OverflowError):
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
    name = str(name or "")
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


def _path_is_within(path: str, root: str) -> bool:
    """Return whether path is contained by root on the active platform."""
    try:
        normalized_path = os.path.normcase(os.path.abspath(path))
        normalized_root = os.path.normcase(os.path.abspath(root))
        return os.path.commonpath((normalized_root, normalized_path)) == normalized_root
    except (OSError, ValueError):
        return False


def _managed_collection_directory(collection_name: str) -> str:
    """Return the durable user-visible folder for a vault collection."""
    return os.path.join(VAULTS_DIR, sanitize_collection_name(collection_name))


def get_chroma_client(path: str | None = None):
    """Return a persistent Chroma client shared by index and search tools."""
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "ChromaDB is unavailable. Install the optional 'chromadb' package "
            "to use vault and codebase indexing tools."
        ) from exc

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

        chunk = text[start:end]
        if chunk.strip():
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


def extract_pdf_with_vision(
    path: str,
    cancellation_token: CancellationToken | None = None,
) -> tuple[str, dict]:
    """Extract PDF text and page images without retaining image batches in RAM."""
    try:
        import pypdf
        from pdf2image import convert_from_path
        from tools.vision_describer import describe_image

        text_stream = []
        warnings = []
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            total_pages = len(reader.pages)
            for page_num, page in enumerate(reader.pages, start=1):
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_stream.append(f"--- Page {page_num} Text ---\n{page_text.strip()}\n")
                except Exception as exc:
                    warnings.append(f"Page {page_num} text extraction failed: {exc}")

        # Render one page at a time to cap peak memory. paths_only lets Poppler
        # write directly to a temporary directory instead of creating PIL images.
        with tempfile.TemporaryDirectory(prefix="selene-pdf-") as image_dir:
            for page_num in range(1, total_pages + 1):
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                try:
                    image_paths = convert_from_path(
                        path, first_page=page_num, last_page=page_num, dpi=140,
                        fmt="png", output_folder=image_dir, paths_only=True,
                        thread_count=1,
                    )
                    if not image_paths:
                        continue
                    description = describe_image(
                        str(image_paths[0]),
                        cancellation_token=cancellation_token,
                    )
                    if description and not description.startswith("Error"):
                        text_stream.append(f"--- Page {page_num} Visual Description ---\n{description}\n")
                    elif description:
                        warnings.append(f"Page {page_num} vision skipped: {description}")
                except OperationCancelled:
                    raise
                except Exception as exc:
                    warnings.append(f"Page {page_num} visual extraction failed: {exc}")

        text = "\n".join(text_stream)
        return text, {"document_type": "pdf", "page_count": total_pages, "char_count": len(text), "warnings": warnings[:20], "warning_count": len(warnings)}
    except OperationCancelled:
        raise
    except Exception as exc:
        raise RuntimeError(f"Error reading PDF: {exc}") from exc


def _read_text_for_index(
    path: str,
    include_vision: bool = True,
    cancellation_token: CancellationToken | None = None,
) -> tuple[str, dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return (
            extract_pdf_with_vision(path, cancellation_token)
            if include_vision
            else extract_document_text(path)
        )
            
    if ext in {".pdf", ".docx"}:
        return extract_document_text(path)

    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
        encoding = "utf-16"
    else:
        if b"\x00" in raw:
            raise ValueError("binary content cannot be indexed as text")
        try:
            text = raw.decode("utf-8-sig")
            encoding = "utf-8"
        except UnicodeDecodeError:
            try:
                text = raw.decode("cp1252")
                encoding = "cp1252"
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
                encoding = "latin-1"
    return text, {
        "document_type": ext.lstrip(".") or "text",
        "char_count": len(text),
        "encoding": encoding,
    }


def _iter_indexable_files(vault_path: str):
    for root, dirs, files in os.walk(vault_path):
        dirs[:] = sorted(
            name for name in dirs
            if name.casefold() not in {".git", ".chroma", "__pycache__", "node_modules"}
            and not os.path.islink(os.path.join(root, name))
        )
        for fname in sorted(files):
            candidate = os.path.join(root, fname)
            if os.path.islink(candidate):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_INDEX_EXTENSIONS:
                yield candidate


def _pdf_fingerprint(path: str) -> str:
    stat = os.stat(path)
    payload = f"{os.path.abspath(path)}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _text_generation(
    path: str,
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    model: str,
) -> str:
    """Identify one complete text/config generation without relying on mtimes."""
    digest = hashlib.sha256()
    digest.update(os.path.abspath(path).encode("utf-8", errors="surrogatepass"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8", errors="surrogatepass"))
    digest.update(f"\0{chunk_size}:{chunk_overlap}:{model}".encode("utf-8"))
    return digest.hexdigest()[:20]


def _existing_source_ids(
    collection_obj,
    source: str,
    source_path: str,
) -> set[str]:
    """Read and validate existing IDs for one physical source file."""
    raw = collection_obj.get(include=["metadatas"])
    if not isinstance(raw, dict):
        raise RuntimeError("ChromaDB returned malformed source metadata")
    ids = raw.get("ids")
    metadatas = raw.get("metadatas")
    if not isinstance(ids, list) or not isinstance(metadatas, list) or len(ids) != len(metadatas):
        raise RuntimeError("ChromaDB returned misaligned source metadata")
    matching_ids: set[str] = set()
    normalized_path = os.path.normcase(os.path.abspath(source_path))
    for item_id, metadata in zip(ids, metadatas):
        if not isinstance(item_id, str) or not item_id:
            raise RuntimeError("ChromaDB returned an invalid source chunk ID")
        meta = metadata if isinstance(metadata, dict) else {}
        stored_path = str(meta.get("source_path") or "").strip()
        same_source = (
            os.path.normcase(os.path.abspath(stored_path)) == normalized_path
            if stored_path
            else str(meta.get("source") or "") == source
        )
        if same_source:
            matching_ids.add(item_id)
    return matching_ids


def _index_job_path(collection_name: str, path: str) -> Path:
    key = hashlib.sha256(
        f"{collection_name}:{os.path.abspath(path)}".encode("utf-8")
    ).hexdigest()[:24]
    return Path(VAULTS_DIR) / ".index_jobs" / f"{key}.json"


def _load_index_job(collection_name: str, path: str) -> dict:
    job_path = _index_job_path(collection_name, path)
    try:
        return read_json_preserved(job_path, expected_type=dict)
    except FileNotFoundError:
        return {}


def _collection_index_job_summaries(collection_name: str) -> list[dict]:
    """List durable PDF jobs for a collection without requiring source mounts."""
    checkpoint_directory = Path(VAULTS_DIR) / ".index_jobs"
    if not checkpoint_directory.is_dir():
        return []
    jobs: list[dict] = []
    for checkpoint_path in sorted(checkpoint_directory.glob("*.json")):
        try:
            state = read_json_preserved(checkpoint_path, expected_type=dict)
        except (FileNotFoundError, PersistenceError):
            continue
        stored_collection = str(state.get("collection") or "").strip()
        if not stored_collection or sanitize_collection_name(stored_collection) != collection_name:
            continue
        source_path = str(state.get("source_path") or "").strip()
        if not source_path:
            continue
        job = _index_job_summary(collection_name, source_path, state)
        job["checkpoint_exists"] = True
        jobs.append(job)
    return jobs


def _save_index_job(collection_name: str, path: str, state: dict) -> None:
    state["revision"] = max(0, int(state.get("revision", 0))) + 1
    job_path = _index_job_path(collection_name, path)
    job_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(job_path, state)


def _index_job_summary(
    collection_name: str,
    path: str,
    state: dict,
    *,
    source: str | None = None,
    error: str | None = None,
) -> dict:
    """Return the compact, model-facing control state for one PDF job."""
    failed_pages = sorted({
        int(value) for value in state.get("vision_failed_pages", [])
        if str(value).lstrip("-").isdigit()
    })
    warnings = [str(value) for value in state.get("warnings", []) if str(value)]
    config = state.get("config") if isinstance(state.get("config"), dict) else {}
    complete = bool(state.get("complete"))
    next_page = None if complete else state.get("next_page")
    try:
        next_page = None if next_page is None else int(next_page)
    except (TypeError, ValueError):
        next_page = None
    vision_mode = str(config.get("vision_mode") or state.get("vision_mode") or "auto")

    summary = {
        # Put control fields first so even legacy prefix previews remain useful.
        "complete": complete,
        "next_page": next_page,
        "revision": _positive_int(state.get("revision"), 0),
        "source": source or state.get("source") or os.path.basename(path),
        "source_path": os.path.abspath(path),
        "page_count": _positive_int(state.get("page_count"), 0),
        "indexed_pages": _positive_int(state.get("indexed_pages"), 0),
        "indexed_chunks": _positive_int(state.get("indexed_chunks"), 0),
        "chunk_size": _positive_int(
            config.get("chunk_size"), DEFAULT_CHUNK_SIZE, minimum=500, maximum=20000
        ),
        "chunk_overlap": _positive_int(
            config.get("chunk_overlap"),
            DEFAULT_CHUNK_OVERLAP,
            minimum=0,
            maximum=max(
                0,
                _positive_int(
                    config.get("chunk_size"),
                    DEFAULT_CHUNK_SIZE,
                    minimum=500,
                    maximum=20000,
                ) // 2,
            ),
        ),
        "vision_mode": vision_mode,
        "vision_pages": _positive_int(state.get("vision_pages"), 0),
        "vision_failed_count": len(failed_pages),
        "vision_failed_pages": failed_pages[:MAX_RETURNED_FAILED_PAGES],
        "vision_failed_pages_truncated": len(failed_pages) > MAX_RETURNED_FAILED_PAGES,
        "vision_complete": bool(complete and not failed_pages),
        "cleanup_pending": bool(state.get("cleanup_pending")),
        "warning_count": max(_positive_int(state.get("warning_count"), 0), len(warnings)),
        "warnings": warnings[-MAX_RETURNED_WARNING_SAMPLES:],
        "checkpoint": str(_index_job_path(collection_name, path)),
        "fingerprint": state.get("fingerprint"),
    }
    if error:
        summary["error"] = str(error)
        summary["retryable"] = next_page is not None
    return summary


def _page_has_images(page) -> bool:
    """Inspect PDF resources without decoding embedded image streams."""
    try:
        resources = page.get("/Resources") or {}
        resources = resources.get_object() if hasattr(resources, "get_object") else resources
        xobjects = resources.get("/XObject") or {}
        xobjects = xobjects.get_object() if hasattr(xobjects, "get_object") else xobjects
        for value in xobjects.values():
            item = value.get_object() if hasattr(value, "get_object") else value
            if item.get("/Subtype") == "/Image":
                return True
    except Exception:
        return False
    return False


def _describe_pdf_page(
    path: str,
    page_number: int,
    image_dir: str,
    cancellation_token: CancellationToken | None,
) -> str:
    try:
        from pdf2image import convert_from_path
        from tools.vision_describer import describe_image
    except ImportError as exc:
        raise RuntimeError(
            "PDF visual indexing requires pdf2image and Poppler; text indexing remains available"
        ) from exc

    conversion_options = {
        "first_page": page_number,
        "last_page": page_number,
        "dpi": 120,
        "fmt": "jpeg",
        "output_folder": image_dir,
        "paths_only": True,
        "thread_count": 1,
        "jpegopt": {"quality": 82, "optimize": True},
    }
    poppler_path = os.environ.get("SELENE_POPPLER_PATH") or os.environ.get("POPPLER_PATH")
    if poppler_path:
        conversion_options["poppler_path"] = poppler_path
    image_paths = convert_from_path(path, **conversion_options)
    if not image_paths:
        raise RuntimeError("PDF page rendering returned no image")
    prompts = (
        (
            "Capture this document page completely for a searchable knowledge vault. "
            "Transcribe visible text, equations, labels, tables, chart values, diagram relationships, "
            "and any information conveyed visually. Be factual and dense; do not add outside knowledge."
        ),
        # Moondream can return a clean stop with empty content for long OCR-style
        # prompts. Its short native description prompt recovers those pages.
        "Describe this page in detail.",
    )
    last_error = "Vision model returned no page description"
    for prompt in prompts:
        description = describe_image(
            str(image_paths[0]),
            prompt=prompt,
            cancellation_token=cancellation_token,
        )
        value = str(description or "").strip()
        if value and not value.startswith("Error"):
            return value
        last_error = value or last_error
        if value and "empty response" not in value.casefold():
            break
    raise RuntimeError(last_error)


def _index_pdf_incremental(
    *,
    path: str,
    rel: str,
    collection_name: str,
    collection_obj,
    chunk_size: int,
    chunk_overlap: int,
    model: str,
    vision_mode: str,
    max_pages: int,
    resume_page: int | None,
    cancellation_token: CancellationToken | None,
    vision_mode_explicit: bool = True,
    chunk_size_explicit: bool = True,
    chunk_overlap_explicit: bool = True,
    model_explicit: bool = True,
) -> dict:
    """Index a bounded PDF page range and checkpoint after every committed page."""
    try:
        import pypdf
    except ImportError as exc:
        raise RuntimeError("Missing required dependency: pypdf") from exc

    fingerprint = _pdf_fingerprint(path)
    with _INDEX_JOB_LOCK:
        state = _load_index_job(collection_name, path)
    if state.get("fingerprint") != fingerprint:
        state = {
            "version": 2,
            "collection": collection_name,
            "source": rel,
            "source_path": os.path.abspath(path),
            "fingerprint": fingerprint,
            "revision": 0,
            "next_page": 1,
            "indexed_pages": 0,
            "indexed_chunks": 0,
            "vision_pages": 0,
            "page_chunk_counts": {},
            "vision_completed_pages": [],
            "vision_failed_pages": [],
            "warning_count": 0,
            "warnings": [],
            "complete": False,
            "config": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": model,
                "vision_mode": vision_mode,
            },
        }
    else:
        # The checkpoint owns the source label for this generation. Callers in
        # older releases sometimes supplied the Chroma directory as vault_path,
        # which changed the relative label between resume calls. Retaining the
        # durable label prevents duplicate IDs and split source names.
        stored_source = str(state.get("source") or "").strip()
        if stored_source:
            rel = stored_source
        stored_config = state.get("config") if isinstance(state.get("config"), dict) else {}
        if stored_config:
            stored_mode = str(stored_config.get("vision_mode") or vision_mode)
            if vision_mode_explicit and stored_mode != vision_mode:
                raise ValueError(
                    f"vision_mode {vision_mode!r} does not match the durable job setting {stored_mode!r}"
                )
            stored_chunk_size = int(stored_config.get("chunk_size", chunk_size))
            stored_chunk_overlap = int(stored_config.get("chunk_overlap", chunk_overlap))
            stored_model = str(stored_config.get("embedding_model") or model)
            if (
                (chunk_size_explicit and stored_chunk_size != chunk_size)
                or (chunk_overlap_explicit and stored_chunk_overlap != chunk_overlap)
                or (model_explicit and stored_model != model)
            ):
                raise ValueError(
                    "chunk or embedding settings do not match the durable PDF job; "
                    "resume with the original settings or use a new collection"
                )
            chunk_size = stored_chunk_size
            chunk_overlap = stored_chunk_overlap
            model = stored_model
            vision_mode = stored_mode
        else:
            # Adopt settings for checkpoints created before configuration became
            # part of the durable job contract.
            state["version"] = 2
            state["config"] = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": model,
                "vision_mode": vision_mode,
            }

    with open(path, "rb") as stream:
        reader = pypdf.PdfReader(stream)
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise ValueError("password required")
            except Exception as exc:
                raise ValueError("PDF is encrypted and cannot be indexed without a password") from exc
        total_pages = len(reader.pages)
        next_page = max(1, min(int(state.get("next_page", 1)), total_pages + 1))
        state["page_count"] = total_pages
        state["config"] = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": model,
            "vision_mode": vision_mode,
        }
        if total_pages == 0:
            state["complete"] = True
            state["next_page"] = 1
        # Persist the initialized job before page work so a failure on page one
        # still has a truthful, resumable checkpoint and durable configuration.
        with _INDEX_JOB_LOCK:
            _save_index_job(collection_name, path, state)
        if resume_page is not None and int(resume_page) != next_page:
            raise ValueError(
                f"resume_page {resume_page} does not match durable checkpoint next_page {next_page}"
            )
        generation_prefix = f"{rel}::pdf::{fingerprint}::"
        failed_at_start = {
            int(value) for value in state.get("vision_failed_pages", [])
        }
        retrying_vision = bool(failed_at_start) and int(state.get("indexed_pages", 0)) >= total_pages
        retry_order: list[int] = []
        if retrying_vision and vision_mode == "off":
            state["vision_failed_pages"] = []
            state["complete"] = False
            state["cleanup_pending"] = True
            state["next_page"] = total_pages + 1
            with _INDEX_JOB_LOCK:
                _save_index_job(collection_name, path, state)
            next_page = total_pages + 1

        if retrying_vision and vision_mode != "off":
            ordered_failed = sorted(failed_at_start)
            split_index = next(
                (index for index, page in enumerate(ordered_failed) if page >= next_page),
                0,
            )
            retry_order = ordered_failed[split_index:] + ordered_failed[:split_index]
            page_numbers = retry_order[:max_pages]
        else:
            stop_page = min(total_pages, next_page + max_pages - 1)
            page_numbers = range(next_page, stop_page + 1)

        with tempfile.TemporaryDirectory(prefix="selene-pdf-page-") as image_dir:
            for page_number in page_numbers:
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                page = reader.pages[page_number - 1]
                warnings = state.setdefault("warnings", [])
                warning_count = max(int(state.get("warning_count", 0)), len(warnings))
                try:
                    page_text = (page.extract_text() or "").strip()
                except Exception as exc:
                    page_text = ""
                    warnings.append(f"Page {page_number} text extraction failed: {exc}")
                    warning_count += 1

                should_describe = retrying_vision or vision_mode == "all" or (
                    vision_mode == "auto"
                    and (len(page_text) < PDF_VISION_TEXT_THRESHOLD or _page_has_images(page))
                )
                visual_text = ""
                if should_describe:
                    try:
                        visual_text = _describe_pdf_page(
                            path, page_number, image_dir, cancellation_token
                        )
                        if not str(visual_text or "").strip():
                            raise RuntimeError("Vision model returned no page description")
                        completed_vision = {
                            int(value) for value in state.get("vision_completed_pages", [])
                        }
                        completed_vision.add(page_number)
                        state["vision_completed_pages"] = sorted(completed_vision)
                        state["vision_pages"] = len(completed_vision)
                        state["vision_failed_pages"] = [
                            value for value in state.get("vision_failed_pages", [])
                            if int(value) != page_number
                        ]
                    except OperationCancelled:
                        raise
                    except Exception as exc:
                        warnings.append(f"Page {page_number} visual extraction failed: {exc}")
                        warning_count += 1
                        failed_pages = {
                            int(value) for value in state.get("vision_failed_pages", [])
                        }
                        failed_pages.add(page_number)
                        state["vision_failed_pages"] = sorted(failed_pages)

                parts = []
                if page_text:
                    parts.append(f"[Page {page_number} extracted text]\n{page_text}")
                if visual_text:
                    parts.append(f"[Page {page_number} visual analysis]\n{visual_text}")
                page_content = "\n\n".join(parts)
                chunks = chunk_text_with_offsets(
                    page_content,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ) if page_content else []

                if chunks:
                    documents = [chunk["text"] for chunk in chunks]
                    embeddings = embed_texts(
                        documents,
                        model=model,
                        cancellation_token=cancellation_token,
                    )
                    ids = [
                        f"{generation_prefix}page::{page_number}::chunk::{chunk['index']}"
                        for chunk in chunks
                    ]
                    source_ids = _existing_source_ids(collection_obj, rel, path)
                    page_prefix = f"{generation_prefix}page::{page_number}::chunk::"
                    previous_page_ids = {
                        item_id for item_id in source_ids if item_id.startswith(page_prefix)
                    }
                    metadatas = [
                        {
                            "source": rel,
                            "source_path": os.path.abspath(path),
                            "filename": os.path.basename(path),
                            "extension": ".pdf",
                            "document_type": "pdf",
                            "page": page_number,
                            "page_count": total_pages,
                            "chunk_index": chunk["index"],
                            "char_start": chunk["char_start"],
                            "char_end": chunk["char_end"],
                            "content_kind": (
                                "text+vision" if page_text and visual_text
                                else "vision" if visual_text else "text"
                            ),
                            "index_generation": fingerprint,
                        }
                        for chunk in chunks
                    ]
                    try:
                        collection_obj.upsert(
                            ids=ids,
                            documents=documents,
                            embeddings=embeddings,
                            metadatas=metadatas,
                        )
                        stale_page_ids = previous_page_ids - set(ids)
                        if stale_page_ids:
                            collection_obj.delete(ids=sorted(stale_page_ids))
                    except Exception:
                        # Generation IDs keep the previous complete PDF intact.
                        # Remove only records that did not exist before this page
                        # attempt; a subsequent resume can safely retry the page.
                        new_ids = set(ids) - previous_page_ids
                        if new_ids:
                            try:
                                collection_obj.delete(ids=sorted(new_ids))
                            except Exception:
                                pass
                        raise
                page_chunk_counts = dict(state.get("page_chunk_counts", {}))
                page_chunk_counts[str(page_number)] = len(chunks)
                state["page_chunk_counts"] = page_chunk_counts
                state["indexed_chunks"] = sum(int(value) for value in page_chunk_counts.values())

                state["indexed_pages"] = max(int(state.get("indexed_pages", 0)), page_number)
                state["page_count"] = total_pages
                state["warning_count"] = warning_count
                state["warnings"] = warnings[-50:]
                failed_vision_pages = state.get("vision_failed_pages", [])
                if retrying_vision:
                    retry_position = retry_order.index(page_number)
                    remaining_in_pass = retry_order[retry_position + 1:]
                    if not failed_vision_pages:
                        state["complete"] = False
                        state["cleanup_pending"] = True
                        state["next_page"] = total_pages + 1
                    elif remaining_in_pass:
                        state["complete"] = False
                        state["next_page"] = remaining_in_pass[0]
                    else:
                        # A full failed-page pass made no guarantee that transient
                        # vision failures recovered. A progress-aware runtime may
                        # retry this next cycle, but will stop if the signature repeats.
                        state["complete"] = False
                        state["next_page"] = min(int(value) for value in failed_vision_pages)
                else:
                    reached_end = page_number >= total_pages
                    if reached_end and vision_mode != "off" and failed_vision_pages:
                        state["complete"] = False
                        state["cleanup_pending"] = False
                        state["next_page"] = min(int(value) for value in failed_vision_pages)
                    else:
                        state["complete"] = False
                        state["cleanup_pending"] = reached_end
                        state["next_page"] = total_pages + 1 if reached_end else page_number + 1
                with _INDEX_JOB_LOCK:
                    _save_index_job(collection_name, path, state)

    ready_for_cleanup = (
        total_pages == 0
        or (
            int(state.get("indexed_pages", 0)) >= total_pages
            and not state.get("vision_failed_pages", [])
        )
    )
    if ready_for_cleanup:
        try:
            existing_ids = _existing_source_ids(collection_obj, rel, path)
            stale_ids = [
                item_id for item_id in existing_ids
                if not str(item_id).startswith(generation_prefix)
            ]
            if stale_ids:
                collection_obj.delete(ids=stale_ids)
            state["complete"] = True
            state["cleanup_pending"] = False
            state["next_page"] = total_pages + 1
        except Exception as exc:
            state.setdefault("warnings", []).append(
                f"New index completed but stale-generation cleanup failed: {exc}"
            )
            state["warning_count"] = max(
                int(state.get("warning_count", 0)) + 1,
                len(state["warnings"]),
            )
            state["warnings"] = state["warnings"][-50:]
            state["complete"] = False
            state["cleanup_pending"] = True
            state["next_page"] = total_pages + 1
        with _INDEX_JOB_LOCK:
            _save_index_job(collection_name, path, state)

    return _index_job_summary(collection_name, path, state, source=rel)


def index_vault(
    vault_path: Optional[str] = None,
    collection_name: str = "vault",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    file_path: Optional[str] = None,
    collection: Optional[str] = None,
    include_vision: bool = True,
    vision_mode: str | None = None,
    max_pages: int = DEFAULT_PDF_PAGES_PER_RUN,
    resume_page: int | None = None,
    action: str = "index",
    cancellation_token: CancellationToken | None = None,
):
    """
    Index either a vault folder or a single file into a ChromaDB collection.

    This function reads text from the target documents, splits it into overlapping
    semantic chunks, generates vector embeddings using Ollama, and stores them in
    ChromaDB for later similarity search retrieval.

    Args:
        vault_path (str | None): Source directory to index recursively. Omit it
            when file_path is supplied; Selene creates managed vault storage
            separately under VAULTS_DIR.
        collection_name (str): The name of the ChromaDB collection to use.
        chunk_size (int | None): The approximate character limit for each text
            chunk. Omitted resumes inherit the durable job setting.
        chunk_overlap (int | None): The number of characters to overlap between
            chunks. Omitted resumes inherit the durable job setting.
        model (str | None): The embedding model name. Omitted resumes inherit
            the durable job setting.
        batch_size (int): Number of chunks to process in a single batch.
        file_path (str | None): A single explicit file path to index.
        collection (str | None): An alias for collection_name.
        include_vision (bool): Backward-compatible visual indexing toggle.
        vision_mode (str | None): ``off``, ``auto``, or ``all``. Auto uses
            Moondream for low-text or image-bearing pages; all analyzes every page.
        max_pages (int): PDF pages processed per invocation (default 20). Progress
            is checkpointed after every page, so repeated calls resume safely.
        resume_page (int | None): Optional optimistic checkpoint token. Pass the
            prior result's next_page so recursive calls remain distinct and safe.
        action (str): ``index`` (default) or ``status`` for checkpoint inspection.

    Returns:
        str: A JSON-encoded string containing status metrics (e.g., number of indexed
             files and chunks) and guidance for using the index.
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
    action = str(action or "index").strip().lower()
    if action not in {"index", "status"}:
        return _json({"error": "action must be 'index' or 'status'"})
    chunk_size_explicit = chunk_size is not None
    chunk_overlap_explicit = chunk_overlap is not None
    model_explicit = model is not None
    vision_mode_explicit = vision_mode is not None
    if vision_mode is None:
        vision_mode = "auto" if include_vision else "off"
    vision_mode = str(vision_mode).strip().lower()
    if vision_mode not in {"off", "auto", "all"}:
        return _json({"error": "vision_mode must be off, auto, or all"})
    max_pages = _positive_int(
        max_pages,
        DEFAULT_PDF_PAGES_PER_RUN,
        minimum=1,
        maximum=500,
    )
    if resume_page is not None:
        resume_page = _positive_int(resume_page, 1, minimum=1, maximum=1_000_000)

    requested_vault_path = os.path.abspath(vault_path) if vault_path else None
    managed_vault_directory = _managed_collection_directory(collection_name)

    if action == "status" and not file_path and requested_vault_path is None:
        jobs = _collection_index_job_summaries(collection_name)
        return _json({
            "complete": all(job.get("complete", False) for job in jobs) if jobs else False,
            "collection": collection_name,
            "action": "status",
            "jobs": jobs,
            "job_count": len(jobs),
            "source_root": None,
        })

    if file_path:
        file_path = os.path.abspath(file_path)
        if action == "index" and not os.path.exists(file_path):
            return _json({
                "error": f"source file not found: {file_path}",
                "file_path": file_path,
            })
        if action == "index" and not os.path.isfile(file_path):
            return _json({
                "error": f"source file is not a regular file: {file_path}",
                "file_path": file_path,
            })
        # A single explicit file is the complete source scope. vault_path is not
        # a destination and must never prevent a valid file from being indexed.
        source_root = os.path.dirname(file_path) or os.curdir
        candidates = [file_path]
    else:
        source_root = requested_vault_path or os.path.abspath(VAULTS_DIR)
        if not os.path.exists(source_root):
            can_create_managed_source = (
                action == "index" and _path_is_within(source_root, VAULTS_DIR)
            )
            if can_create_managed_source:
                try:
                    os.makedirs(source_root, exist_ok=True)
                except OSError as exc:
                    return _json({
                        "error": f"Could not create managed vault folder: {exc}",
                        "vault_directory": source_root,
                    })
            else:
                return _json({"error": f"source folder not found: {source_root}"})
        if not os.path.isdir(source_root):
            return _json({
                "error": (
                    "vault_path must be a source folder when file_path is omitted: "
                    f"{source_root}"
                )
            })
        candidates = list(_iter_indexable_files(source_root))

    if action == "index":
        try:
            os.makedirs(managed_vault_directory, exist_ok=True)
        except OSError as exc:
            return _json({
                "error": f"Could not create managed vault folder: {exc}",
                "vault_directory": managed_vault_directory,
            })

    if action == "index" and not candidates:
        return _json({
            "complete": False,
            "continuation_required": False,
            "next_page": None,
            "continuation": None,
            "collection": collection_name,
            "source_root": source_root,
            "indexed_files": 0,
            "indexed_chunks": 0,
            "skipped_files": [],
            "skipped_count": 0,
            "pdf_jobs": [],
            "incomplete_pdf_count": 0,
            "failed_pdf_count": 0,
            "error": f"no supported files found in source folder: {source_root}",
        })

    if action == "status":
        jobs = []
        for path in candidates:
            if os.path.splitext(path)[1].lower() != ".pdf":
                continue
            try:
                state = _load_index_job(collection_name, path)
            except PersistenceError as exc:
                jobs.append({"source_path": path, "error": str(exc), "preserved": True})
                continue
            job = _index_job_summary(collection_name, path, state)
            job["checkpoint_exists"] = bool(state)
            jobs.append(job)
        return _json({
            "complete": all(job.get("complete", False) for job in jobs) if jobs else False,
            "collection": collection_name,
            "action": "status",
            "jobs": jobs,
            "job_count": len(jobs),
            "source_root": source_root,
        })

    batch_size = _positive_int(batch_size, DEFAULT_BATCH_SIZE, minimum=1, maximum=128)
    chunk_size = _positive_int(chunk_size, DEFAULT_CHUNK_SIZE, minimum=500, maximum=20000)
    chunk_overlap = _positive_int(chunk_overlap, DEFAULT_CHUNK_OVERLAP, minimum=0, maximum=max(0, chunk_size // 2))
    model = str(model or DEFAULT_EMBED_MODEL).strip()
    if (
        not model
        or len(model) > 200
        or any(character.isspace() or ord(character) < 32 for character in model)
    ):
        return _json({
            "error": "model must be a non-empty name of at most 200 characters without whitespace or control characters"
        })

    try:
        client = get_chroma_client()
        collection_obj = client.get_or_create_collection(name=collection_name)
    except Exception as exc:
        return _json({"error": f"Could not open ChromaDB: {exc}", "persist_directory": CHROMA_DIR})

    indexed_chunks = 0
    indexed_files = 0
    skipped_files: list[dict] = []
    pdf_jobs: list[dict] = []

    for path in candidates:
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        if not os.path.exists(path):
            skipped_files.append({"file": path, "error": "file not found"})
            continue
        if not os.path.isfile(path):
            skipped_files.append({"file": path, "error": "not a regular file"})
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_INDEX_EXTENSIONS:
            skipped_files.append({"file": path, "error": f"unsupported extension: {ext}"})
            continue

        rel = (
            os.path.relpath(path, source_root)
            if os.path.isdir(source_root)
            else os.path.basename(path)
        )
        if ext == ".pdf":
            try:
                job = _index_pdf_incremental(
                    path=path,
                    rel=rel,
                    collection_name=collection_name,
                    collection_obj=collection_obj,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    model=model,
                    vision_mode=vision_mode,
                    max_pages=max_pages,
                    resume_page=resume_page,
                    cancellation_token=cancellation_token,
                    vision_mode_explicit=vision_mode_explicit,
                    chunk_size_explicit=chunk_size_explicit,
                    chunk_overlap_explicit=chunk_overlap_explicit,
                    model_explicit=model_explicit,
                )
                pdf_jobs.append(job)
                indexed_chunks += int(job.get("indexed_chunks", 0))
                if job.get("complete"):
                    indexed_files += 1
            except OperationCancelled:
                raise
            except Exception as exc:
                try:
                    failed_state = _load_index_job(collection_name, path)
                except PersistenceError:
                    failed_state = {}
                preserved = bool(failed_state)
                error_text = (
                    "incremental PDF indexing failed; durable checkpoint and previous "
                    f"index preserved: {exc}"
                    if preserved
                    else f"incremental PDF indexing failed before a checkpoint could be saved: {exc}"
                )
                skipped_files.append({
                    "file": path,
                    "error": error_text,
                })
                failed_job = _index_job_summary(
                    collection_name,
                    path,
                    failed_state,
                    source=rel,
                    error=error_text,
                )
                pdf_jobs.append(failed_job)
                indexed_chunks += int(failed_job.get("indexed_chunks", 0))
            continue

        try:
            text, info = _read_text_for_index(
                path,
                include_vision=bool(include_vision),
                cancellation_token=cancellation_token,
            )
        except UnicodeDecodeError:
            skipped_files.append({"file": path, "error": "not UTF-8 text; use PDF/DOCX or plain text"})
            continue
        except OperationCancelled:
            raise
        except Exception as exc:
            skipped_files.append({"file": path, "error": str(exc)})
            continue

        try:
            previous_ids = _existing_source_ids(collection_obj, rel, path)
        except Exception as exc:
            skipped_files.append({
                "file": path,
                "error": f"could not inspect previous index; existing chunks preserved: {exc}",
            })
            continue

        chunks = chunk_text_with_offsets(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            # Emptying a tracked document is a valid source change. Only mark it
            # indexed after all previous chunks have been removed successfully.
            try:
                if previous_ids:
                    collection_obj.delete(ids=sorted(previous_ids))
            except Exception as exc:
                skipped_files.append({
                    "file": path,
                    "error": f"empty source could not remove previous chunks: {exc}",
                })
                continue
            indexed_files += 1
            continue

        generation = _text_generation(
            path,
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model=model,
        )
        generation_prefix = f"{rel}::text::{generation}::chunk::"
        file_ids: list[str] = []
        file_docs: list[str] = []
        file_metadatas: list[dict] = []
        for chunk in chunks:
            chunk_index = chunk["index"]
            file_ids.append(f"{generation_prefix}{chunk_index}")
            file_docs.append(chunk["text"])
            file_metadatas.append({
                "source": rel,
                "source_path": os.path.abspath(path),
                "filename": os.path.basename(path),
                "extension": ext,
                "chunk_index": chunk_index,
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "document_type": info.get("document_type", ext.lstrip(".")),
                "index_generation": generation,
                **({"encoding": info["encoding"]} if info.get("encoding") else {}),
            })

        # Complete all potentially failing embedding work before modifying the
        # existing source, so a model outage cannot erase a valid old index.
        prepared_batches = []
        try:
            for start in range(0, len(file_docs), batch_size):
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                batch_docs = file_docs[start:start + batch_size]
                prepared_batches.append((
                    file_ids[start:start + batch_size],
                    batch_docs,
                    file_metadatas[start:start + batch_size],
                    embed_texts(
                        batch_docs,
                        model=model,
                        cancellation_token=cancellation_token,
                    ),
                ))
        except OperationCancelled:
            raise
        except Exception as exc:
            skipped_files.append({"file": path, "error": f"embedding failed; previous index preserved: {exc}"})
            continue

        written_ids: set[str] = set()
        try:
            for batch_ids, batch_docs, batch_metadata, batch_embeddings in prepared_batches:
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                collection_obj.upsert(ids=batch_ids, documents=batch_docs, embeddings=batch_embeddings, metadatas=batch_metadata)
                written_ids.update(batch_ids)
            stale_ids = previous_ids - set(file_ids)
            if stale_ids:
                collection_obj.delete(ids=sorted(stale_ids))
        except OperationCancelled:
            rollback_ids = written_ids - previous_ids
            if rollback_ids:
                try:
                    collection_obj.delete(ids=sorted(rollback_ids))
                except Exception:
                    pass
            raise
        except Exception as exc:
            rollback_ids = written_ids - previous_ids
            rollback_error = None
            if rollback_ids:
                try:
                    collection_obj.delete(ids=sorted(rollback_ids))
                except Exception as rollback_exc:
                    rollback_error = str(rollback_exc)
            error = f"index write failed; previous index preserved: {exc}"
            if rollback_error:
                error += f"; partial-generation cleanup also failed: {rollback_error}"
            skipped_files.append({"file": path, "error": error})
            continue

        indexed_files += 1
        indexed_chunks += len(file_docs)

    removed_chunks = 0
    pdf_work_complete = all(job.get("complete", False) for job in pdf_jobs)
    if not file_path and not skipped_files and pdf_work_complete:
        # Once the whole folder has indexed successfully, reconcile documents
        # removed since the prior run. Restrict cleanup to absolute source paths
        # inside this folder so unrelated documents in a shared collection stay.
        try:
            raw = collection_obj.get(include=["metadatas"])
            if not isinstance(raw, dict):
                raise RuntimeError("ChromaDB returned malformed collection metadata")
            existing_ids = raw.get("ids")
            existing_metadatas = raw.get("metadatas")
            if (
                not isinstance(existing_ids, list)
                or not isinstance(existing_metadatas, list)
                or len(existing_ids) != len(existing_metadatas)
            ):
                raise RuntimeError("ChromaDB returned misaligned collection metadata")
            candidate_paths = {
                os.path.normcase(os.path.abspath(candidate)) for candidate in candidates
            }
            removed_source_ids = []
            for item_id, metadata in zip(existing_ids, existing_metadatas):
                if not isinstance(item_id, str) or not item_id:
                    raise RuntimeError("ChromaDB returned an invalid chunk ID")
                meta = metadata if isinstance(metadata, dict) else {}
                source_path = str(meta.get("source_path") or "").strip()
                if not source_path or not _path_is_within(source_path, source_root):
                    continue
                if os.path.normcase(os.path.abspath(source_path)) not in candidate_paths:
                    removed_source_ids.append(item_id)
            if removed_source_ids:
                collection_obj.delete(ids=sorted(removed_source_ids))
                removed_chunks = len(removed_source_ids)
        except Exception as exc:
            skipped_files.append({
                "file": source_root,
                "error": f"folder cleanup failed; removed-source chunks may remain: {exc}",
            })

    # Auto-register an alias when a single file was indexed
    has_pdf_chunks = any(int(job.get("indexed_chunks", 0)) > 0 for job in pdf_jobs)
    alias_warning = None
    if file_path and (indexed_files == 1 or has_pdf_chunks):
        stem = os.path.splitext(os.path.basename(file_path))[0]
        try:
            register_vault_alias(
                alias=stem,
                collection_name=collection_name,
                file_path=os.path.abspath(file_path),
            )
        except (OSError, PersistenceError, ValueError) as exc:
            alias_warning = f"Content was indexed, but its friendly alias could not be saved: {exc}"

    incomplete_pdf_count = sum(not job.get("complete", False) for job in pdf_jobs)
    failed_pdf_count = sum(bool(job.get("error")) for job in pdf_jobs)
    complete = bool(candidates) and not skipped_files and incomplete_pdf_count == 0
    resumable_jobs = [
        job for job in pdf_jobs
        if not job.get("complete") and not job.get("error") and job.get("next_page") is not None
    ]
    continuation = None
    if file_path and len(resumable_jobs) == 1:
        job = resumable_jobs[0]
        continuation = {
            "tool": "index_vault",
            "arguments": {
                "action": "index",
                "collection": collection_name,
                "file_path": os.path.abspath(file_path),
                "vision_mode": job.get("vision_mode", vision_mode),
                "max_pages": max_pages,
                "chunk_size": int(job.get("chunk_size", chunk_size)),
                "chunk_overlap": int(job.get("chunk_overlap", chunk_overlap)),
                "resume_page": int(job["next_page"]),
            },
        }
    effective_vision_mode = (
        str(pdf_jobs[0].get("vision_mode") or vision_mode)
        if len(pdf_jobs) == 1 else vision_mode
    )
    effective_chunk_size = (
        int(pdf_jobs[0].get("chunk_size", chunk_size))
        if len(pdf_jobs) == 1 else chunk_size
    )
    effective_chunk_overlap = (
        int(pdf_jobs[0].get("chunk_overlap", chunk_overlap))
        if len(pdf_jobs) == 1 else chunk_overlap
    )

    result = {
        "complete": complete,
        "continuation_required": continuation is not None,
        "next_page": continuation["arguments"]["resume_page"] if continuation else None,
        "continuation": continuation,
        "collection": collection_name,
        "source_root": source_root,
        "indexed_files": indexed_files,
        "indexed_chunks": indexed_chunks,
        "removed_chunks": removed_chunks,
        "skipped_files": skipped_files[:20],
        "skipped_count": len(skipped_files),
        "chunk_size": effective_chunk_size,
        "chunk_overlap": effective_chunk_overlap,
        "include_vision": bool(include_vision),
        "vision_mode": effective_vision_mode,
        "max_pages_per_run": max_pages,
        "pdf_jobs": pdf_jobs,
        "incomplete_pdf_count": incomplete_pdf_count,
        "failed_pdf_count": failed_pdf_count,
        "alias": os.path.splitext(os.path.basename(file_path))[0] if file_path and (indexed_files == 1 or has_pdf_chunks) else None,
        "guidance": (
            "Call index_vault again with the same file and collection plus resume_page=next_page to resume incomplete PDF jobs. "
            "Use action=status to inspect progress, vault_search for semantic lookup, and vault_read for exhaustive ordered retrieval."
        ),
    }
    if alias_warning:
        result["warning"] = alias_warning
    if skipped_files and len(skipped_files) == len(candidates):
        result["error"] = skipped_files[0]["error"]
    return _json(result)


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
    try:
        client = get_chroma_client()
    except Exception as exc:
        return _json({"error": f"Could not open ChromaDB: {exc}", "persist_directory": CHROMA_DIR})

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
        attempted_filters.append({"source": candidate})
        attempted_filters.append({"source_path": candidate})

    if attempted_filters:
        where_clause = {"$or": attempted_filters} if len(attempted_filters) > 1 else attempted_filters[0]
        try:
            existing = collection_obj.get(where=where_clause, include=["metadatas"])
            ids = existing.get("ids", [])
            if ids:
                collection_obj.delete(ids=ids)
                deleted_ids.update(ids)
        except Exception:
            pass

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
    try:
        client = get_chroma_client()
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
    with _ALIAS_LOCK:
        try:
            return read_json_preserved(_ALIAS_FILE, expected_type=dict)
        except FileNotFoundError:
            return {}


def _save_aliases(aliases: dict) -> None:
    """Persist the alias registry to disk."""
    with _ALIAS_LOCK:
        atomic_write_json(_ALIAS_FILE, aliases)


def register_vault_alias(alias: str, collection_name: str, file_path: str | None = None) -> None:
    """Register a human-friendly alias for a vault collection."""
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        raise ValueError("alias is required")
    with _ALIAS_LOCK:
        aliases = _load_aliases()
        aliases[clean_alias.casefold()] = {
            "alias": clean_alias,
            "collection": sanitize_collection_name(collection_name),
            "file_path": os.path.abspath(file_path) if file_path else None,
        }
        _save_aliases(aliases)


def register_vault_alias_tool(
    alias: str,
    collection: str | None = None,
    collection_name: str | None = None,
    file_path: str | None = None,
) -> str:
    """JSON tool entry-point for slash-command and runner registration of aliases."""
    target = collection if collection is not None else collection_name
    if not target:
        return _json({"error": "collection is required"})
    try:
        register_vault_alias(alias=alias, collection_name=str(target), file_path=file_path)
    except (OSError, PersistenceError, ValueError) as exc:
        return _json({"error": str(exc), "alias_file": _ALIAS_FILE})
    resolved = sanitize_collection_name(str(target))
    return _json({
        "success": True,
        "alias": str(alias or "").strip(),
        "collection": resolved,
        "file_path": os.path.abspath(file_path) if file_path else None,
    })


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
    key = name.strip().casefold()

    # Exact match
    if key in aliases:
        return aliases[key]["collection"]

    # Substring match
    matches = {entry.get("collection") for alias_key, entry in aliases.items() if key in alias_key or alias_key in key}
    matches.discard(None)
    if len(matches) == 1:
        return next(iter(matches))

    # Fall through — treat as a raw collection name
    return sanitize_collection_name(name)


def list_vault_aliases() -> str:
    """Return a JSON listing of all registered vault aliases."""
    try:
        aliases = _load_aliases()
    except PersistenceError as exc:
        return _json({"error": str(exc), "alias_file": _ALIAS_FILE, "preserved": True})
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
    clean_new_name = str(new_name or "").strip()
    if not clean_new_name:
        return _json({"error": "new_name is required."})
    old_collection = sanitize_collection_name(old_name)
    new_collection = sanitize_collection_name(clean_new_name)

    if old_collection == new_collection:
        return _json({"error": "Old and new names resolve to the same collection name.",
                       "old": old_collection, "new": new_collection})

    # Read critical metadata before any collection mutation. If the file is
    # malformed, preserve both it and the original Chroma collection.
    try:
        aliases = _load_aliases()
    except PersistenceError as exc:
        return _json({"error": str(exc), "alias_file": _ALIAS_FILE, "preserved": True})

    try:
        client = get_chroma_client()
    except Exception as exc:
        return _json({"error": f"Could not open ChromaDB: {exc}", "persist_directory": CHROMA_DIR})

    # Verify old collection exists
    try:
        old_coll = client.get_collection(name=old_collection)
    except Exception:
        return _json({"error": f"Collection '{old_collection}' not found.",
                       "persist_directory": CHROMA_DIR})

    try:
        client.get_collection(name=new_collection)
        return _json({"error": f"Destination collection '{new_collection}' already exists."})
    except Exception:
        pass

    count = old_coll.count()
    new_coll = client.create_collection(name=new_collection)
    try:
        batch_size = 500
        for offset in range(0, count, batch_size):
            data = old_coll.get(limit=batch_size, offset=offset, include=["documents", "metadatas", "embeddings"])
            ids = data.get("ids", [])
            if ids:
                new_coll.upsert(
                    ids=ids,
                    documents=data.get("documents", []),
                    metadatas=data.get("metadatas", []),
                    embeddings=data.get("embeddings", []),
                )
        if new_coll.count() != count:
            raise RuntimeError(f"Copied {new_coll.count()} of {count} chunks")
    except Exception as exc:
        try:
            client.delete_collection(name=new_collection)
        except Exception:
            pass
        return _json({"error": f"Rename copy failed; original collection was preserved: {exc}"})

    # Commit alias metadata before deleting the original collection. A full or
    # locked disk must never leave aliases pointing at a collection we already
    # destroyed. Reload under the alias lock so unrelated concurrent aliases are
    # merged instead of overwritten by the earlier preflight snapshot.
    try:
        with _ALIAS_LOCK:
            aliases = _load_aliases()
            updated_aliases = []
            for key, entry in aliases.items():
                if entry.get("collection") == old_collection:
                    entry["collection"] = new_collection
                    updated_aliases.append(entry.get("alias", key))
            aliases[clean_new_name.casefold()] = {
                "alias": clean_new_name,
                "collection": new_collection,
                "file_path": None,
            }
            _save_aliases(aliases)
    except (OSError, PersistenceError, ValueError) as exc:
        try:
            client.delete_collection(name=new_collection)
        except Exception:
            pass
        return _json({
            "error": f"Rename metadata update failed; original collection was preserved: {exc}",
            "alias_file": _ALIAS_FILE,
            "preserved": True,
        })

    try:
        client.delete_collection(name=old_collection)
        original_retained = False
    except Exception as exc:
        # Both complete collections are safer than deleting either after the
        # copy and metadata commit succeeded.
        original_retained = True
        deletion_warning = str(exc)

    result = {
        "renamed": True,
        "old_collection": old_collection,
        "new_collection": new_collection,
        "chunks_moved": count,
        "updated_aliases": updated_aliases,
        "original_retained": original_retained,
    }
    if original_retained:
        result["warning"] = f"The new collection is active, but the original could not be removed: {deletion_warning}"
    return _json(result)


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
    parser.add_argument("--no-vision", action="store_true")
    args = parser.parse_args()

    print(index_vault(
        vault_path=args.vault_path,
        collection_name=args.collection,
        file_path=args.file_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model=args.model,
        batch_size=args.batch_size,
        include_vision=not args.no_vision,
    ))
