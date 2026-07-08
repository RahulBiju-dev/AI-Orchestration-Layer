"""Shared Ollama embedding helpers for vault indexing/search."""

from __future__ import annotations

import math
from typing import Any, List, Sequence

import requests

DEFAULT_EMBED_MODEL = "embeddinggemma"
OLLAMA_EMBED_URLS = (
    "http://127.0.0.1:11434/api/embed",
)
OLLAMA_LEGACY_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"

# Persistent session for connection pooling
_SESSION = requests.Session()


def _as_plain_data(response: Any) -> Any:
    """Convert Ollama client response objects into plain Python data."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return response


def normalize_embeddings(response: Any) -> List[List[float]]:
    """Extract a list of embedding vectors from common Ollama response shapes."""
    data = _as_plain_data(response)

    if isinstance(data, dict):
        if "embeddings" in data:
            return [list(embedding) for embedding in data["embeddings"]]
        if "embedding" in data:
            return [list(data["embedding"])]

    if isinstance(data, list):
        if all(isinstance(item, (list, tuple)) for item in data):
            return [list(item) for item in data]
        if all(isinstance(item, dict) and "embedding" in item for item in data):
            return [list(item["embedding"]) for item in data]

    raise RuntimeError("Unexpected embedding response shape: %s" % repr(data)[:500])


def _clean_inputs(texts: Sequence[str]) -> list[str]:
    cleaned = []
    for text in texts:
        value = str(text or "").strip()
        cleaned.append(value if value else " ")
    return cleaned


def _validate_embedding_count(embeddings: list[list[float]], expected: int) -> list[list[float]]:
    if len(embeddings) != expected:
        raise RuntimeError(f"Ollama returned {len(embeddings)} embedding(s) for {expected} input(s)")
    dimensions = set()
    normalized = []
    for index, embedding in enumerate(embeddings):
        if not embedding:
            raise RuntimeError(f"Ollama returned an empty embedding at index {index}")
        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Ollama returned a non-numeric embedding at index {index}") from exc
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError(f"Ollama returned a non-finite embedding at index {index}")
        dimensions.add(len(vector))
        normalized.append(vector)
    if len(dimensions) > 1:
        raise RuntimeError(f"Ollama returned inconsistent embedding dimensions: {sorted(dimensions)}")
    return normalized


def embed_texts(texts: Sequence[str], model: str = DEFAULT_EMBED_MODEL, timeout: int = 60) -> List[List[float]]:
    """
    Embed one or more texts using Ollama, returning Chroma-compatible vectors.
    
    This function communicates with a local Ollama instance to generate vector
    embeddings for the provided strings. It supports batching and handles fallback
    to the Ollama Python client if direct HTTP requests fail.
    
    Args:
        texts (Sequence[str]): A list or tuple of string documents to embed.
        model (str): The Ollama model name to use for embeddings (e.g., 'embeddinggemma').
        timeout (int): The maximum number of seconds to wait for a network response.
        
    Returns:
        List[List[float]]: A list of floating-point vectors corresponding to the inputs.
        
    Raises:
        RuntimeError: If all connection methods to Ollama fail or if the shape of the
                      returned embeddings does not match the inputs.
    """
    inputs = _clean_inputs(texts)
    if not inputs:
        return []

    timeout = max(1, min(int(timeout), 300))
    last_error: Exception | None = None
    payload = {"model": model, "input": inputs}

    for url in OLLAMA_EMBED_URLS:
        try:
            resp = _SESSION.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return _validate_embedding_count(normalize_embeddings(resp.json()), len(inputs))
        except Exception as exc:
            last_error = exc

    # Ollama versions before /api/embed accepted one prompt per request at
    # /api/embeddings. Keep this compatibility path explicit, but concurrent
    # to significantly improve latency on bulk embedding tasks.
    try:
        from concurrent.futures import ThreadPoolExecutor

        legacy_embeddings = []

        def _fetch_legacy_embedding(text: str) -> list[list[float]]:
            resp = _SESSION.post(OLLAMA_LEGACY_EMBED_URL, json={"model": model, "prompt": text}, timeout=timeout)
            resp.raise_for_status()
            return normalize_embeddings(resp.json())

        # Use up to 10 threads to avoid overwhelming a local LLM server
        with ThreadPoolExecutor(max_workers=10) as executor:
            # map ensures results are returned in the exact order of `inputs`
            results = executor.map(_fetch_legacy_embedding, inputs)

            for emb_list in results:
                legacy_embeddings.extend(emb_list)

        return _validate_embedding_count(legacy_embeddings, len(inputs))
    except Exception as exc:
        last_error = exc

    try:
        import ollama

        if hasattr(ollama, "embed"):
            return _validate_embedding_count(normalize_embeddings(ollama.embed(model=model, input=inputs)), len(inputs))
        if hasattr(ollama, "Embeddings"):
            client = ollama.Embeddings()
            return _validate_embedding_count(normalize_embeddings(client.create(model=model, input=inputs)), len(inputs))
    except Exception as exc:
        last_error = exc

    raise RuntimeError("Failed to obtain embeddings via HTTP or ollama client: %s" % last_error)


def embed_query(text: str, model: str = DEFAULT_EMBED_MODEL, timeout: int = 30) -> List[float]:
    embeddings = embed_texts([text], model=model, timeout=timeout)
    if not embeddings:
        raise RuntimeError("Ollama returned no embedding for query")
    return embeddings[0]
