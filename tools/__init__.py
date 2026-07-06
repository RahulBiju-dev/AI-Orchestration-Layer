"""Tool package with lazy exports to keep lightweight imports inexpensive."""

from typing import Any

__all__ = [
    "index_vault",
    "chunk_text",
    "search_vault",
    "format_for_gemma",
]


def __getattr__(name: str) -> Any:
    if name in {"index_vault", "chunk_text"}:
        from . import vault_indexer
        return getattr(vault_indexer, name)
    if name in {"search_vault", "format_for_gemma"}:
        from . import vault_search
        return getattr(vault_search, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
