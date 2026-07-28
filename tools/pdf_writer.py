"""Safe PDF creation and exhaustive vault export tools."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

from agent.cancellation import CancellationToken, OperationCancelled
from agent.platform_runtime import get_runtime_paths
from agent.persistence import atomic_write_json, atomic_write_text, read_json_preserved

MAX_PDF_CONTENT_CHARS = 10_000_000
MAX_PDF_TITLE_CHARS = 1_000
EXPORTS_DIR = get_runtime_paths().data_dir / "vaults" / "exports"
PDF_JOBS_DIR = get_runtime_paths().data_dir / "vaults" / ".pdf_jobs"
NOTES_SOURCE_CHARS = 6000
MAX_NOTES_CURSOR_VALUE = 10_000_000
_PDF_OUTPUT_LOCK = threading.RLock()
_NOTES_LOCKS_GUARD = threading.Lock()
_NOTES_JOB_LOCKS: dict[str, threading.Lock] = {}


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_output_path(file_path: str) -> Path:
    raw = str(file_path or "").strip()
    if not raw:
        raise ValueError("file_path is required")
    if len(raw) > 4096 or "\0" in raw:
        raise ValueError("file_path is invalid")
    requested = Path(raw).expanduser()
    if requested.suffix.lower() != ".pdf":
        requested = requested.with_suffix(".pdf")
    if requested.is_absolute():
        resolved = requested.resolve()
        allowed_roots = (
            Path.cwd().resolve(),
            get_runtime_paths().data_dir.resolve(),
            EXPORTS_DIR.resolve(),
        )
        if not any(_is_within(resolved, root) for root in allowed_roots):
            raise ValueError(
                "Absolute PDF output must stay inside the current workspace or Selene data directory"
            )
        return resolved
    resolved = (EXPORTS_DIR / requested).resolve()
    if not _is_within(resolved, EXPORTS_DIR.resolve()):
        raise ValueError("Relative PDF output cannot escape Selene's vault exports directory")
    return resolved


def _register_unicode_font(pdfmetrics, TTFont) -> str:
    candidates = (
        Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            pdfmetrics.registerFont(TTFont("SeleneUnicode", str(candidate)))
            return "SeleneUnicode"
        except Exception:
            continue
    return "Helvetica"


def _markdown_story(content: str, title: str, styles, font_name: str):
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Preformatted, Spacer

    body = ParagraphStyle(
        "SeleneBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9.5,
        leading=13,
        spaceAfter=5,
        wordWrap="CJK",
    )
    heading1 = ParagraphStyle(
        "SeleneHeading1", parent=body, fontSize=17, leading=21, spaceBefore=10,
        spaceAfter=7, textColor="#17365D",
    )
    heading2 = ParagraphStyle(
        "SeleneHeading2", parent=body, fontSize=13, leading=17, spaceBefore=8,
        spaceAfter=5, textColor="#24527A",
    )
    heading3 = ParagraphStyle(
        "SeleneHeading3", parent=body, fontSize=11, leading=14, spaceBefore=6,
        spaceAfter=4, textColor="#2F5D73",
    )
    title_style = ParagraphStyle(
        "SeleneTitle", parent=heading1, fontSize=22, leading=27,
        alignment=TA_CENTER, spaceAfter=12,
    )
    code_style = ParagraphStyle(
        "SeleneCode", parent=body, fontName="Courier", fontSize=8, leading=10,
        leftIndent=4 * mm, backColor="#F3F5F7", borderPadding=5,
    )
    table_style = ParagraphStyle(
        "SeleneTable", parent=code_style, fontSize=7.5, leading=9,
        leftIndent=0, backColor="#F8FAFC",
    )

    story = [Paragraph(html.escape(title), title_style)] if title else []
    paragraph_lines: list[str] = []
    code_lines: list[str] = []
    table_lines: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = " ".join(line.strip() for line in paragraph_lines).strip()
        paragraph_lines.clear()
        if text:
            story.append(Paragraph(html.escape(text), body))

    def flush_code() -> None:
        if code_lines:
            story.append(Preformatted("\n".join(code_lines), code_style, maxLineLength=120))
            code_lines.clear()

    def flush_table() -> None:
        if table_lines:
            story.append(Preformatted("\n".join(table_lines), table_style, maxLineLength=160))
            table_lines.clear()

    for raw_line in content.replace("\r\n", "\n").split("\n"):
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraph()
                flush_table()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            table_lines.append(stripped)
            continue
        flush_table()
        if not line.strip():
            flush_paragraph()
            story.append(Spacer(1, 2 * mm))
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            style = {1: heading1, 2: heading2, 3: heading3}[len(heading.group(1))]
            story.append(Paragraph(html.escape(heading.group(2).strip()), style))
            continue
        bullet = re.match(r"^\s*[-*+]\s+(.+)$", line)
        if bullet:
            flush_paragraph()
            story.append(Paragraph(
                f"•&nbsp;&nbsp;{html.escape(bullet.group(1).strip())}",
                ParagraphStyle("SeleneBullet", parent=body, leftIndent=5 * mm, firstLineIndent=-3 * mm),
            ))
            continue
        numbered = re.match(r"^\s*(\d+[.)])\s+(.+)$", line)
        if numbered:
            flush_paragraph()
            story.append(Paragraph(
                f"{html.escape(numbered.group(1))}&nbsp;&nbsp;{html.escape(numbered.group(2).strip())}",
                ParagraphStyle("SeleneNumbered", parent=body, leftIndent=7 * mm, firstLineIndent=-5 * mm),
            ))
            continue
        paragraph_lines.append(line)

    flush_paragraph()
    flush_code()
    flush_table()
    if len(story) == (1 if title else 0):
        story.append(Paragraph("No content was provided.", body))
    return story


def _render_pdf(path: Path, title: str, content: str) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import SimpleDocTemplate
    except ImportError as exc:
        raise RuntimeError("PDF creation requires reportlab. Install project requirements first.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    font_name = _register_unicode_font(pdfmetrics, TTFont)
    story = _markdown_story(content, title, getSampleStyleSheet(), font_name)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)

    def add_page_number(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.setFillColor("#667085")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    try:
        document = SimpleDocTemplate(
            str(temporary), pagesize=A4,
            rightMargin=18 * mm, leftMargin=18 * mm,
            topMargin=18 * mm, bottomMargin=17 * mm,
            title=title,
            author="Selene",
        )
        document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _create_pdf_locked(
    file_path: str,
    content: str | None = None,
    title: str = "",
    content_file: str | None = None,
    overwrite: bool = False,
    confirmed: bool = False,
) -> str:
    """Create a styled PDF atomically from Markdown-like text or a text file."""
    try:
        output = _resolve_output_path(file_path)
        resolved_title = str(title or output.stem).strip()
        if len(resolved_title) > MAX_PDF_TITLE_CHARS:
            return _json({"error": f"PDF title exceeds the {MAX_PDF_TITLE_CHARS}-character limit"})
        if output.exists() and not overwrite:
            return _json({"error": f"PDF already exists: {output}", "overwrite_required": True})
        if output.exists() and overwrite and not confirmed:
            return _json({"error": "confirmed=true is required to overwrite an existing PDF"})

        if content_file:
            source = Path(content_file).expanduser().resolve()
            if not source.is_file():
                return _json({"error": f"content_file was not found: {source}"})
            if source.stat().st_size > MAX_PDF_CONTENT_CHARS:
                return _json({"error": "content_file exceeds the 10 MB PDF input limit"})
            value = source.read_text(encoding="utf-8")
        else:
            value = str(content or "")
        if not value.strip():
            return _json({"error": "content or content_file is required"})
        if len(value) > MAX_PDF_CONTENT_CHARS:
            return _json({"error": "PDF content exceeds the 10,000,000-character limit"})

        _render_pdf(output, resolved_title, value)
        return _json({
            "created": True,
            "file_path": str(output),
            "title": resolved_title,
            "input_characters": len(value),
            "bytes": output.stat().st_size,
        })
    except Exception as exc:
        return _json({"error": str(exc)})


def create_pdf(
    file_path: str,
    content: str | None = None,
    title: str = "",
    content_file: str | None = None,
    overwrite: bool = False,
    confirmed: bool = False,
) -> str:
    """Create one PDF at a time so local existence checks remain atomic."""
    with _PDF_OUTPUT_LOCK:
        return _create_pdf_locked(
            file_path=file_path,
            content=content,
            title=title,
            content_file=content_file,
            overwrite=overwrite,
            confirmed=confirmed,
        )


def _validated_batch_by_id(collection_obj, records: list[dict]) -> dict[str, tuple[str, dict]]:
    """Fetch an exact record batch without allowing zip truncation or omissions."""
    requested_ids = [item.get("id") for item in records]
    if any(not isinstance(item_id, str) or not item_id for item_id in requested_ids):
        raise RuntimeError("Vault records contain an invalid chunk ID")
    raw = collection_obj.get(
        ids=requested_ids,
        include=["documents", "metadatas"],
    )
    if not isinstance(raw, dict):
        raise RuntimeError("Vault collection returned malformed records")
    ids = raw.get("ids")
    documents = raw.get("documents")
    metadatas = raw.get("metadatas")
    if not isinstance(ids, list) or not isinstance(documents, list) or not isinstance(metadatas, list):
        raise RuntimeError("Vault collection returned malformed records")
    if not (len(ids) == len(documents) == len(metadatas)):
        raise RuntimeError("Vault collection returned misaligned records")
    by_id: dict[str, tuple[str, dict]] = {}
    for item_id, document, metadata in zip(ids, documents, metadatas):
        if not isinstance(item_id, str) or not item_id:
            raise RuntimeError("Vault collection returned an invalid chunk ID")
        if item_id in by_id:
            raise RuntimeError(f"Vault collection returned duplicate chunk ID {item_id!r}")
        by_id[item_id] = (
            str(document or ""),
            metadata if isinstance(metadata, dict) else {},
        )
    missing = [item_id for item_id in requested_ids if item_id not in by_id]
    if missing:
        raise RuntimeError(f"Vault collection omitted {len(missing)} requested chunk(s)")
    return by_id


def _source_page_key(metadata: dict) -> tuple[str, str, object]:
    return (
        str(metadata.get("source_path") or ""),
        str(metadata.get("source") or ""),
        metadata.get("page"),
    )


def _metadata_offset(metadata: dict, name: str, default: int) -> int:
    value = metadata.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"Vault chunk has invalid {name}: {value!r}") from exc
    if parsed < 0:
        raise RuntimeError(f"Vault chunk has negative {name}: {parsed}")
    return parsed


def export_vault_pdf(
    collection: str,
    file_path: str,
    title: str = "",
    source: str | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
    overwrite: bool = False,
    confirmed: bool = False,
    require_vision: bool = False,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """Export every ordered vault chunk to a source-preserving reference PDF."""
    try:
        from tools.vault_indexer import get_chroma_client, resolve_vault_alias
        from tools.vault_search import ordered_vault_records

        collection_name = resolve_vault_alias(collection)
        records = ordered_vault_records(
            collection_name,
            source=source,
            start_page=start_page,
            end_page=end_page,
            cancellation_token=cancellation_token,
        )
        if not records:
            return _json({"error": "No vault chunks matched the requested collection/source/pages"})
        preflight = _vault_notes_preflight(
            collection_name,
            records,
            require_vision=bool(require_vision),
        )
        if preflight.get("error"):
            return _json(preflight)

        collection_obj = get_chroma_client().get_collection(name=collection_name)
        sections: list[str] = []
        previous_key = None
        previous_end = 0
        for start in range(0, len(records), 100):
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            batch = records[start:start + 100]
            by_id = _validated_batch_by_id(collection_obj, batch)
            for item in batch:
                document, metadata = by_id[item["id"]]
                key = _source_page_key(metadata)
                char_start = _metadata_offset(metadata, "char_start", 0)
                char_end = _metadata_offset(
                    metadata,
                    "char_end",
                    char_start + len(document),
                )
                if char_end < char_start:
                    raise RuntimeError("Vault chunk char_end precedes char_start")
                if key != previous_key:
                    page_label = f" — Page {metadata.get('page')}" if metadata.get("page") else ""
                    sections.append(f"\n## {metadata.get('source', 'Unknown source')}{page_label}\n")
                    previous_end = 0
                    previous_key = key
                overlap = max(0, previous_end - char_start)
                text = str(document)[overlap:]
                if text.strip():
                    sections.append(text)
                previous_end = max(previous_end, char_end)

        content = "\n\n".join(sections)
        if len(content) > MAX_PDF_CONTENT_CHARS:
            return _json({
                "error": "Vault export exceeds the 10,000,000-character PDF input limit",
                "chunks": len(records),
            })
        result = json.loads(create_pdf(
            file_path=file_path,
            content=content,
            title=title or f"{collection_name} Knowledge Export",
            overwrite=overwrite,
            confirmed=confirmed,
        ))
        result.update({
            "collection": collection_name,
            "source": source,
            "exported_chunks": len(records),
            "export_kind": "ordered_source_preserving",
            "vision_verified": preflight.get("vision_verified", False),
        })
        return _json(result)
    except OperationCancelled:
        raise
    except Exception as exc:
        return _json({"error": str(exc)})


def _notes_job_paths(collection: str, source: str | None, output: Path) -> tuple[Path, Path]:
    key = hashlib.sha256(
        f"{collection}:{source or ''}:{output}".encode("utf-8")
    ).hexdigest()[:24]
    directory = PDF_JOBS_DIR / key
    return directory, directory / "state.json"


def _record_is_pdf(metadata: dict) -> bool:
    return (
        str(metadata.get("extension") or "").casefold() == ".pdf"
        or str(metadata.get("source") or "").casefold().endswith(".pdf")
        or str(metadata.get("source_path") or "").casefold().endswith(".pdf")
    )


def _job_matches_records(job: dict, records: list[dict]) -> bool:
    job_path = str(job.get("source_path") or "").strip()
    job_source = str(job.get("source") or "").strip().casefold()
    normalized_job_path = os.path.normcase(os.path.abspath(job_path)) if job_path else ""
    for item in records:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        record_path = str(metadata.get("source_path") or "").strip()
        if (
            normalized_job_path
            and record_path
            and os.path.normcase(os.path.abspath(record_path)) == normalized_job_path
        ):
            return True
        if (
            (not normalized_job_path or not record_path)
            and job_source
            and str(metadata.get("source") or "").strip().casefold() == job_source
        ):
            return True
    return False


def _pdf_source_identities(records: list[dict]) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    for item in records:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if not _record_is_pdf(metadata):
            continue
        source_path = str(metadata.get("source_path") or "").strip()
        if source_path:
            identities.add(("path", os.path.normcase(os.path.abspath(source_path))))
            continue
        source = str(metadata.get("source") or "").strip().casefold()
        if source:
            identities.add(("source", source))
    return identities


def _job_source_identities(job: dict) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    source_path = str(job.get("source_path") or "").strip()
    if source_path:
        identities.add(("path", os.path.normcase(os.path.abspath(source_path))))
    source = str(job.get("source") or "").strip().casefold()
    if source:
        identities.add(("source", source))
    return identities


def _index_continuation(job: dict, collection_name: str) -> dict | None:
    next_page = job.get("next_page")
    source_path = str(job.get("source_path") or "").strip()
    if next_page is None or not source_path:
        return None
    return {
        "tool": "index_vault",
        "arguments": {
            "action": "index",
            "collection": collection_name,
            "file_path": source_path,
            "vision_mode": str(job.get("vision_mode") or "auto"),
            "chunk_size": int(job.get("chunk_size") or 1800),
            "chunk_overlap": int(job.get("chunk_overlap") or 250),
            "resume_page": int(next_page),
        },
    }


def _vault_notes_preflight(
    collection_name: str,
    records: list[dict],
    *,
    require_vision: bool,
) -> dict:
    """Verify that selected PDF chunks come from complete durable index jobs."""
    from tools.vault_indexer import _collection_index_job_summaries

    pdf_records = [
        item for item in records
        if _record_is_pdf(item.get("metadata") if isinstance(item.get("metadata"), dict) else {})
    ]
    jobs = [
        job for job in _collection_index_job_summaries(collection_name)
        if _job_matches_records(job, records)
    ]
    incomplete = [job for job in jobs if not job.get("complete")]
    if incomplete:
        continuations = [
            continuation
            for continuation in (
                _index_continuation(job, collection_name) for job in incomplete
            )
            if continuation is not None
        ]
        return {
            "error": "Selected PDF indexing is incomplete; refusing to create notes with missing pages or vision data",
            "error_code": "vault_index_incomplete",
            "collection": collection_name,
            "complete": False,
            "require_vision": require_vision,
            "pdf_jobs": incomplete,
            "continuation_required": bool(continuations),
            "continuation": continuations[0] if len(continuations) == 1 else None,
            "continuations": continuations,
            "guidance": (
                "Resume every returned index_vault continuation until each job reports complete=true, "
                "then retry the PDF operation."
            ),
        }

    if not require_vision:
        return {
            "vision_verified": False,
            "pdf_job_count": len(jobs),
        }
    if not pdf_records:
        return {
            "error": "require_vision=true was requested, but the selected vault records are not identifiable as PDF pages",
            "error_code": "vision_source_unverified",
            "collection": collection_name,
            "require_vision": True,
        }
    if not jobs:
        return {
            "error": "All-page vision coverage cannot be verified because the selected PDF has no durable index checkpoint",
            "error_code": "vision_source_unverified",
            "collection": collection_name,
            "require_vision": True,
            "guidance": (
                "Index the slide PDF with index_vault using vision_mode=all and wait for complete=true, "
                "then retry this operation."
            ),
        }

    expected_sources = _pdf_source_identities(pdf_records)
    verified_sources = set().union(*(_job_source_identities(job) for job in jobs))
    missing_sources = sorted(expected_sources - verified_sources)
    if missing_sources:
        return {
            "error": "All-page vision coverage cannot be verified for every selected PDF",
            "error_code": "vision_source_unverified",
            "collection": collection_name,
            "require_vision": True,
            "unverified_sources": [
                {"identity_kind": kind, "identity": identity}
                for kind, identity in missing_sources
            ],
            "guidance": (
                "Index every selected slide PDF with vision_mode=all and wait for complete=true, "
                "or narrow the source selection, then retry this operation."
            ),
        }

    unverified = [
        job for job in jobs
        if (
            str(job.get("vision_mode") or "") != "all"
            or not job.get("vision_complete")
            or int(job.get("vision_pages") or 0) < int(job.get("page_count") or 0)
        )
    ]
    if unverified:
        return {
            "error": "The selected PDF was not indexed with successful all-page vision coverage",
            "error_code": "vision_coverage_incomplete",
            "collection": collection_name,
            "require_vision": True,
            "pdf_jobs": unverified,
            "guidance": (
                "For slide notes, index the source into a fresh collection with vision_mode=all, "
                "resume until complete=true, then build the notes from that collection."
            ),
        }
    return {
        "vision_verified": True,
        "pdf_job_count": len(jobs),
        "vision_pages": sum(int(job.get("vision_pages") or 0) for job in jobs),
    }


def _parse_notes_cursor(value: str | int | None) -> tuple[int, int]:
    raw = str(value if value is not None else "0:0").strip()
    if len(raw) > 100:
        raise ValueError("cursor exceeds the 100-character limit")
    try:
        if ":" in raw:
            record, char = raw.split(":", 1)
            parsed = int(record), int(char)
        else:
            parsed = int(raw), 0
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cursor must be an integer or '<chunk>:<character>'") from exc
    if any(value < 0 or value > MAX_NOTES_CURSOR_VALUE for value in parsed):
        raise ValueError(
            f"cursor values must be between 0 and {MAX_NOTES_CURSOR_VALUE}"
        )
    return parsed


def _cursor_text(cursor: tuple[int, int]) -> str:
    return f"{cursor[0]}:{cursor[1]}"


def _vault_source_window(
    collection_obj,
    records: list[dict],
    cursor: tuple[int, int],
    max_chars: int = NOTES_SOURCE_CHARS,
) -> tuple[str, tuple[int, int]]:
    record_index, char_offset = cursor
    candidates = records[record_index:record_index + 20]
    if not candidates:
        return "", cursor
    by_id = _validated_batch_by_id(collection_obj, candidates)
    parts: list[str] = []
    used = 0
    next_cursor = cursor
    previous_key = None
    previous_end = 0
    if record_index > 0 and char_offset == 0:
        previous_meta = records[record_index - 1]["metadata"]
        previous_key = _source_page_key(previous_meta)
        previous_end = _metadata_offset(previous_meta, "char_end", 0)
    for relative_index, item in enumerate(candidates):
        document, metadata = by_id[item["id"]]
        offset = char_offset if relative_index == 0 else 0
        if offset > len(document):
            raise RuntimeError(
                f"Notes cursor offset {offset} exceeds chunk length {len(document)}"
            )
        key = _source_page_key(metadata)
        char_start = _metadata_offset(metadata, "char_start", 0)
        char_end = _metadata_offset(
            metadata,
            "char_end",
            char_start + len(document),
        )
        if char_end < char_start:
            raise RuntimeError("Vault chunk char_end precedes char_start")
        if offset == 0 and key == previous_key:
            offset = min(len(document), max(0, previous_end - char_start))
        if offset >= len(document):
            next_cursor = (record_index + relative_index + 1, 0)
            previous_end = max(previous_end, char_end) if key == previous_key else char_end
            previous_key = key
            continue
        header = (
            f"[Source: {metadata.get('source', 'unknown')} | "
            f"Page: {metadata.get('page', '?')} | "
            f"Chunk: {metadata.get('chunk_index', '?')} | "
            f"Kind: {metadata.get('content_kind', 'text')}]"
        )
        available = max_chars - used - len(header) - 2
        if available <= 0:
            break
        text_slice = document[offset:offset + available]
        parts.append(f"{header}\n{text_slice}")
        used += len(header) + len(text_slice) + 2
        if offset + len(text_slice) < len(document):
            next_cursor = (record_index + relative_index, offset + len(text_slice))
            break
        next_cursor = (record_index + relative_index + 1, 0)
        previous_end = max(previous_end, char_end) if key == previous_key else char_end
        previous_key = key
        if used >= max_chars:
            break
    return "\n\n".join(parts), next_cursor


def _generate_note_section(
    source_text: str,
    title: str,
    cancellation_token: CancellationToken | None,
) -> str:
    from agent.ollama_runtime import OllamaService, OperationKind
    from agent.runtime_config import get_runtime_config

    runtime = get_runtime_config()
    service = OllamaService(runtime)
    owner = f"vault-notes:{threading.get_ident()}:{time.monotonic_ns()}"
    response = service.chat(
        kind=OperationKind.CHAT,
        owner=owner,
        cancellation_token=cancellation_token,
        operation_timeout=runtime.summary_timeout_seconds,
        model=runtime.chat_model,
        stream=False,
        think=False,
        messages=[
            {
                "role": "system",
                "content": (
                    "Convert the supplied slide or document excerpt into complete, dense lecture notes. "
                    "The excerpt may contain vision-model transcriptions of diagrams, charts, tables, equations, "
                    "layouts, and other visual relationships; treat those as source evidence and preserve them. "
                    "Retain every definition, equation, step, label, value, example, caveat, and source/page label. "
                    "Remove only exact repetition and presentation filler. Do not add outside facts or silently "
                    "drop uncertain material; mark uncertainty explicitly. Use clear Markdown headings and bullets."
                ),
            },
            {
                "role": "user",
                "content": f"Notes document: {title}\n\nVault excerpt:\n{source_text}",
            },
        ],
    )
    message = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if not content:
        raise RuntimeError("The local chat model returned an empty notes section")
    return str(content).strip()


def _section_integrity(text: str) -> dict:
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
    }


def _pdf_output_is_valid(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < 8:
            return False
        with path.open("rb") as stream:
            return stream.read(5) == b"%PDF-"
    except OSError:
        return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _notes_job_lock(key: str) -> threading.Lock:
    with _NOTES_LOCKS_GUARD:
        return _NOTES_JOB_LOCKS.setdefault(key, threading.Lock())


def _build_vault_notes_pdf_locked(
    collection: str,
    file_path: str,
    title: str = "",
    source: str | None = None,
    cursor: str | int | None = None,
    sections_per_run: int = 4,
    action: str = "build",
    overwrite: bool = False,
    confirmed: bool = False,
    require_vision: bool = False,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """Build grounded notes over an entire vault through resumable model sections."""
    try:
        from tools.vault_indexer import get_chroma_client, resolve_vault_alias
        from tools.vault_search import ordered_vault_records

        action = str(action or "build").strip().lower()
        if action not in {"build", "status"}:
            return _json({"error": "action must be build or status"})
        if len(str(title or "")) > MAX_PDF_TITLE_CHARS:
            return _json({"error": f"PDF title exceeds the {MAX_PDF_TITLE_CHARS}-character limit"})
        collection_name = resolve_vault_alias(collection)
        output = _resolve_output_path(file_path)
        job_dir, state_path = _notes_job_paths(collection_name, source, output)
        try:
            state = read_json_preserved(state_path, expected_type=dict)
        except FileNotFoundError:
            state = {}
        if action == "status":
            from tools.vault_indexer import _collection_index_job_summaries

            return _json({
                **state,
                "collection": collection_name,
                "file_path": str(output),
                "job_directory": str(job_dir),
                "exists": bool(state),
                "require_vision": bool(state.get("require_vision", False)),
                "pdf_index_jobs": _collection_index_job_summaries(collection_name),
            })

        if state and bool(state.get("require_vision", False)) != bool(require_vision):
            return _json({
                "error": (
                    "require_vision does not match this durable notes job; "
                    "use the original setting or a different file_path"
                ),
                "require_vision": bool(state.get("require_vision", False)),
                "job_directory": str(job_dir),
                "preserved": True,
            })
        if state.get("finalizing") and output.exists():
            if not _pdf_output_is_valid(output):
                return _json({
                    "error": "The notes job reached finalization, but the output is not a valid PDF",
                    "file_path": str(output),
                    "job_directory": str(job_dir),
                    "sections_preserved": len(state.get("section_files", [])),
                })
            output_unchanged = (
                state.get("output_existed_before_finalizing") is True
                and isinstance(state.get("previous_output_sha256"), str)
                and _file_sha256(output) == state["previous_output_sha256"]
            )
            if output_unchanged:
                # The process may have stopped after recording intent but before
                # atomically replacing the old output. Resume finalization below.
                state["finalizing"] = False
                atomic_write_json(state_path, state)
            else:
                state["complete"] = True
                state["finalizing"] = False
                state["next_cursor"] = None
                state["pdf_bytes"] = output.stat().st_size
                atomic_write_json(state_path, state)

        if state.get("complete") and _pdf_output_is_valid(output):
            return _json({
                "created": True,
                "complete": True,
                "already_complete": True,
                "collection": collection_name,
                "source": source,
                "file_path": str(output),
                "bytes": output.stat().st_size,
                "completed_sections": state.get("completed_sections", 0),
                "job_directory": str(job_dir),
                "vision_verified": bool(state.get("vision_verified", False)),
            })
        if state.get("complete") and not output.exists():
            state["complete"] = False
            state["next_cursor"] = f"{state.get('total_chunks', 0)}:0"
            atomic_write_json(state_path, state)
        elif state.get("complete"):
            return _json({
                "error": "The durable notes job is complete, but its output PDF is invalid",
                "file_path": str(output),
                "job_directory": str(job_dir),
                "sections_preserved": len(state.get("section_files", [])),
            })

        if output.exists() and not overwrite and not state.get("complete"):
            return _json({"error": f"PDF already exists: {output}", "overwrite_required": True})
        if output.exists() and overwrite and not confirmed:
            return _json({"error": "confirmed=true is required to overwrite an existing PDF"})

        records = ordered_vault_records(
            collection_name,
            source=source,
            cancellation_token=cancellation_token,
        )
        if not records:
            return _json({"error": "No vault chunks matched the requested collection/source"})
        preflight = _vault_notes_preflight(
            collection_name,
            records,
            require_vision=bool(require_vision),
        )
        if preflight.get("error"):
            return _json(preflight)
        selection_digest = hashlib.sha256(
            json.dumps(
                [
                    {
                        "id": item.get("id"),
                        "metadata": item.get("metadata"),
                    }
                    for item in records
                ],
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        legacy_selection_digest = hashlib.sha256(
            "\n".join(str(item["id"]) for item in records).encode("utf-8")
        ).hexdigest()
        if (
            state
            and int(state.get("version") or 1) < 2
            and state.get("selection_digest") == legacy_selection_digest
        ):
            state["version"] = 2
            state["selection_digest"] = selection_digest
            state["require_vision"] = False
            state["vision_verified"] = False
            state.setdefault("section_manifest", {})
            state.setdefault("finalizing", False)
            atomic_write_json(state_path, state)
        if state and state.get("selection_digest") != selection_digest:
            return _json({
                "error": "The vault changed after this notes job started; use a different file_path to start a fresh grounded export",
                "current_cursor": state.get("next_cursor"),
            })

        if not state:
            job_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "version": 2,
                "collection": collection_name,
                "source": source,
                "file_path": str(output),
                "title": str(title or f"{collection_name} Notes"),
                "require_vision": bool(require_vision),
                "vision_verified": bool(preflight.get("vision_verified", False)),
                "vision_pages": int(preflight.get("vision_pages") or 0),
                "selection_digest": selection_digest,
                "total_chunks": len(records),
                "next_cursor": "0:0",
                "completed_sections": 0,
                "section_files": [],
                "section_manifest": {},
                "finalizing": False,
                "complete": False,
            }
            atomic_write_json(state_path, state)

        section_names = state.get("section_files", [])
        if not isinstance(section_names, list) or any(
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".md")
            for name in section_names
        ) or len(section_names) != len(set(section_names)):
            return _json({
                "error": "The durable notes checkpoint contains invalid section records and was preserved",
                "job_directory": str(job_dir),
                "preserved": True,
            })
        missing_sections = [name for name in section_names if not (job_dir / name).is_file()]
        if missing_sections:
            return _json({
                "error": "A committed notes section is missing; refusing to create an incomplete PDF",
                "missing_sections": missing_sections[:20],
                "job_directory": str(job_dir),
            })
        section_manifest = state.get("section_manifest", {})
        if not isinstance(section_manifest, dict):
            return _json({
                "error": "The durable notes checkpoint contains an invalid section manifest and was preserved",
                "job_directory": str(job_dir),
                "preserved": True,
            })
        manifest_changed = False
        for name in section_names:
            section_text = (job_dir / name).read_text(encoding="utf-8")
            if not section_text.strip():
                return _json({
                    "error": "A committed notes section is empty; refusing to create an incomplete PDF",
                    "section": name,
                    "job_directory": str(job_dir),
                })
            actual = _section_integrity(section_text)
            expected = section_manifest.get(name)
            if expected is None:
                # Upgrade version-1 checkpoints after validating their section
                # paths and non-empty contents.
                section_manifest[name] = actual
                manifest_changed = True
            elif not isinstance(expected, dict) or expected != actual:
                return _json({
                    "error": "A committed notes section failed its integrity check",
                    "section": name,
                    "job_directory": str(job_dir),
                    "preserved": True,
                })
        if manifest_changed:
            state["version"] = 2
            state["section_manifest"] = section_manifest
            atomic_write_json(state_path, state)

        current_cursor = _parse_notes_cursor(state.get("next_cursor"))
        if (
            current_cursor[0] > len(records)
            or (current_cursor[0] == len(records) and current_cursor[1] != 0)
        ):
            return _json({
                "error": "The durable notes cursor is beyond the selected vault content",
                "next_cursor": _cursor_text(current_cursor),
                "total_chunks": len(records),
                "job_directory": str(job_dir),
                "preserved": True,
            })
        if cursor is not None and _parse_notes_cursor(cursor) != current_cursor:
            return _json({
                "error": "cursor does not match the durable job checkpoint",
                "next_cursor": _cursor_text(current_cursor),
            })
        try:
            sections_per_run = max(1, min(int(sections_per_run), 12))
        except (TypeError, ValueError, OverflowError):
            return _json({"error": "sections_per_run must be an integer between 1 and 12"})
        collection_obj = get_chroma_client().get_collection(name=collection_name)

        created_this_run = 0
        while current_cursor[0] < len(records) and created_this_run < sections_per_run:
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            source_text, next_cursor = _vault_source_window(
                collection_obj, records, current_cursor
            )
            if not source_text or next_cursor == current_cursor:
                raise RuntimeError(f"Could not advance vault notes cursor {_cursor_text(current_cursor)}")
            section_name = (
                f"section-{current_cursor[0]:08d}-{current_cursor[1]:08d}--"
                f"{next_cursor[0]:08d}-{next_cursor[1]:08d}.md"
            )
            section_path = job_dir / section_name
            if not section_path.exists():
                notes = _generate_note_section(
                    source_text,
                    state["title"],
                    cancellation_token,
                )
                atomic_write_text(section_path, notes, durable=True)
            section_text = section_path.read_text(encoding="utf-8")
            if not section_text.strip():
                raise RuntimeError(f"Generated notes section is empty: {section_name}")
            section_files = list(state.get("section_files", []))
            if section_name not in section_files:
                section_files.append(section_name)
            state["section_files"] = sorted(section_files)
            section_manifest = dict(state.get("section_manifest", {}))
            section_manifest[section_name] = _section_integrity(section_text)
            state["section_manifest"] = section_manifest
            current_cursor = next_cursor
            created_this_run += 1
            state["next_cursor"] = _cursor_text(current_cursor)
            state["completed_sections"] = len(state["section_files"])
            atomic_write_json(state_path, state)

        if current_cursor[0] >= len(records):
            section_paths = [job_dir / name for name in state.get("section_files", [])]
            notes_content = "\n\n".join(
                path.read_text(encoding="utf-8") for path in section_paths
            )
            state["finalizing"] = True
            state["final_content_sha256"] = hashlib.sha256(
                notes_content.encode("utf-8")
            ).hexdigest()
            state["output_existed_before_finalizing"] = output.is_file()
            state["previous_output_sha256"] = (
                _file_sha256(output) if output.is_file() else None
            )
            state["next_cursor"] = _cursor_text(current_cursor)
            atomic_write_json(state_path, state)
            result = json.loads(create_pdf(
                file_path=str(output),
                content=notes_content,
                title=state["title"],
                overwrite=overwrite,
                confirmed=confirmed,
            ))
            if result.get("error"):
                state["finalizing"] = False
                state["last_finalize_error"] = str(result.get("error"))
                atomic_write_json(state_path, state)
                result.update({
                    "job_directory": str(job_dir),
                    "sections_preserved": len(section_paths),
                    "next_cursor": _cursor_text(current_cursor),
                })
                return _json(result)
            state["complete"] = True
            state["finalizing"] = False
            state["next_cursor"] = None
            state["pdf_bytes"] = result.get("bytes")
            atomic_write_json(state_path, state)
            result.update({
                "collection": collection_name,
                "source": source,
                "refined": True,
                "completed_sections": len(section_paths),
                "job_directory": str(job_dir),
                "vision_verified": bool(state.get("vision_verified", False)),
            })
            return _json(result)

        return _json({
            "collection": collection_name,
            "source": source,
            "file_path": str(output),
            "complete": False,
            "next_cursor": _cursor_text(current_cursor),
            "total_chunks": len(records),
            "completed_sections": state["completed_sections"],
            "sections_created_this_run": created_this_run,
            "job_directory": str(job_dir),
            "vision_verified": bool(state.get("vision_verified", False)),
            "guidance": (
                "Call build_vault_notes_pdf again with cursor=next_cursor and the same arguments. "
                "Each call resumes from durable section checkpoints; finalize occurs automatically at the end."
            ),
        })
    except OperationCancelled:
        raise
    except Exception as exc:
        return _json({"error": str(exc)})


def build_vault_notes_pdf(
    collection: str,
    file_path: str,
    title: str = "",
    source: str | None = None,
    cursor: str | int | None = None,
    sections_per_run: int = 4,
    action: str = "build",
    overwrite: bool = False,
    confirmed: bool = False,
    require_vision: bool = False,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """Serialize concurrent mutations of the same durable notes job."""
    arguments = {
        "collection": collection,
        "file_path": file_path,
        "title": title,
        "source": source,
        "cursor": cursor,
        "sections_per_run": sections_per_run,
        "action": action,
        "overwrite": overwrite,
        "confirmed": confirmed,
        "require_vision": require_vision,
        "cancellation_token": cancellation_token,
    }
    if str(action or "build").strip().lower() == "status":
        return _build_vault_notes_pdf_locked(**arguments)
    try:
        from tools.vault_indexer import resolve_vault_alias

        collection_name = resolve_vault_alias(collection)
        output = _resolve_output_path(file_path)
        _, state_path = _notes_job_paths(collection_name, source, output)
        lock = _notes_job_lock(str(state_path))
        while not lock.acquire(timeout=0.05):
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
        try:
            return _build_vault_notes_pdf_locked(**arguments)
        finally:
            lock.release()
    except OperationCancelled:
        raise
    except Exception as exc:
        return _json({"error": str(exc)})
