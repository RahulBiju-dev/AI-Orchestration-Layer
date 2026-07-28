"""Shared Ollama embedding helpers for vault indexing/search."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Sequence

from agent.cancellation import CancellationToken, OperationCancelled
from agent.ollama_runtime import (
    CoordinatorError,
    OllamaRequestTimeout,
    OllamaRuntimeError,
    OllamaService,
    OllamaUnavailableError,
)
from agent.runtime_config import get_runtime_config

_RUNTIME_CONFIG = get_runtime_config()
DEFAULT_EMBED_MODEL = _RUNTIME_CONFIG.embedding_model
_EMBED_SERVICE = OllamaService(_RUNTIME_CONFIG)
MAX_INPUTS = 1_000
MAX_INPUT_CHARS = 1_000_000
MAX_TOTAL_INPUT_CHARS = 20_000_000
MAX_EMBED_BATCH_SIZE = 128
MAX_EMBEDDING_DIMENSIONS = 65_536
MAX_TOTAL_EMBEDDING_VALUES = 5_000_000
MAX_RETRIES = 2
MAX_TRACKED_MODELS = 256
_MODEL_DIMENSIONS: dict[str, int] = {}
_MODEL_DIMENSIONS_LOCK = threading.Lock()


def _as_plain_data(response: Any) -> Any:
    """Convert Ollama client response objects into plain Python data."""
    try:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "dict"):
            return response.dict()
    except Exception as exc:
        raise RuntimeError(f"Could not serialize the embedding response: {exc}") from exc
    return response


def _copy_embedding(value: Any, index: int) -> list[Any]:
    if (
        isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sequence)
    ):
        raise RuntimeError(f"Embedding at index {index} must be an array")
    if not value:
        raise RuntimeError(f"Ollama returned an empty embedding at index {index}")
    if len(value) > MAX_EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Embedding at index {index} exceeds the "
            f"{MAX_EMBEDDING_DIMENSIONS}-dimension limit"
        )
    return list(value)


def normalize_embeddings(response: Any) -> list[list[Any]]:
    """Extract a list of embedding vectors from common Ollama response shapes."""
    data = _as_plain_data(response)
    raw_embeddings: Any = None

    if isinstance(data, dict):
        if "embeddings" in data:
            raw_embeddings = data["embeddings"]
        elif "embedding" in data:
            raw_embeddings = [data["embedding"]]
        elif "data" in data:
            rows = data["data"]
            if isinstance(rows, list) and all(
                isinstance(item, dict) and "embedding" in item for item in rows
            ):
                raw_embeddings = [item["embedding"] for item in rows]
    elif isinstance(data, list):
        if all(isinstance(item, dict) and "embedding" in item for item in data):
            raw_embeddings = [item["embedding"] for item in data]
        else:
            raw_embeddings = data

    if not isinstance(raw_embeddings, list):
        raise RuntimeError("Unexpected embedding response shape: %s" % repr(data)[:500])
    if not raw_embeddings:
        return []
    if not all(
        isinstance(item, Sequence)
        and not isinstance(item, (str, bytes, bytearray))
        for item in raw_embeddings
    ):
        raise RuntimeError("Embedding response must contain an array of vectors")
    normalized: list[list[Any]] = []
    value_count = 0
    for index, embedding in enumerate(raw_embeddings):
        vector = _copy_embedding(embedding, index)
        value_count += len(vector)
        if value_count > MAX_TOTAL_EMBEDDING_VALUES:
            raise RuntimeError(
                "Embedding response exceeds the "
                f"{MAX_TOTAL_EMBEDDING_VALUES}-value limit"
            )
        normalized.append(vector)
    return normalized


def _clean_inputs(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes, bytearray)):
        raise TypeError("texts must be a sequence of strings, not a single string")
    if not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence")
    if len(texts) > MAX_INPUTS:
        raise ValueError(f"At most {MAX_INPUTS} texts may be embedded in one call")
    cleaned: list[str] = []
    total_chars = 0
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"texts[{index}] must be a string")
        if not text.strip():
            raise ValueError(f"texts[{index}] must not be empty or whitespace-only")
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(
                f"texts[{index}] exceeds the {MAX_INPUT_CHARS}-character limit"
            )
        total_chars += len(text)
        if total_chars > MAX_TOTAL_INPUT_CHARS:
            raise ValueError(
                f"combined embedding input exceeds the "
                f"{MAX_TOTAL_INPUT_CHARS}-character limit"
            )
        # Preserve the exact caller-provided text; stripping can change source
        # offsets or code/document meaning.
        cleaned.append(text)
    return cleaned


def _validate_embedding_count(embeddings: list[list[float]], expected: int) -> list[list[float]]:
    if len(embeddings) != expected:
        raise RuntimeError(f"Ollama returned {len(embeddings)} embedding(s) for {expected} input(s)")
    dimensions = set()
    normalized = []
    for index, embedding in enumerate(embeddings):
        if not embedding:
            raise RuntimeError(f"Ollama returned an empty embedding at index {index}")
        if len(embedding) > MAX_EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                f"Ollama returned an embedding above the "
                f"{MAX_EMBEDDING_DIMENSIONS}-dimension limit at index {index}"
            )
        try:
            vector = [
                float(value)
                for value in embedding
                if not isinstance(value, bool)
            ]
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"Ollama returned a non-numeric embedding at index {index}") from exc
        if len(vector) != len(embedding):
            raise RuntimeError(f"Ollama returned a boolean embedding value at index {index}")
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError(f"Ollama returned a non-finite embedding at index {index}")
        squared_norm = math.fsum(value * value for value in vector)
        if not math.isfinite(squared_norm) or squared_norm <= 0:
            raise RuntimeError(
                f"Ollama returned a zero-length or numerically unstable embedding at index {index}"
            )
        dimensions.add(len(vector))
        normalized.append(vector)
    if len(dimensions) > 1:
        raise RuntimeError(f"Ollama returned inconsistent embedding dimensions: {sorted(dimensions)}")
    return normalized


def _validate_model(model: Any) -> str:
    if (
        not isinstance(model, str)
        or not model.strip()
        or len(model) > 200
        or any(character.isspace() or ord(character) < 32 for character in model)
    ):
        raise ValueError(
            "model must be a non-empty name of at most 200 characters "
            "without whitespace or control characters"
        )
    return model.strip()


def _record_model_dimension(model: str, embeddings: list[list[float]]) -> None:
    if not embeddings:
        return
    dimension = len(embeddings[0])
    with _MODEL_DIMENSIONS_LOCK:
        previous = _MODEL_DIMENSIONS.get(model)
        if previous is not None and previous != dimension:
            raise RuntimeError(
                f"Embedding dimension for model {model!r} changed from "
                f"{previous} to {dimension}; refusing to mix incompatible vectors"
            )
        if previous is None and len(_MODEL_DIMENSIONS) >= MAX_TRACKED_MODELS:
            _MODEL_DIMENSIONS.pop(next(iter(_MODEL_DIMENSIONS)))
        _MODEL_DIMENSIONS[model] = dimension


def _retry_delay(
    attempt: int,
    deadline: float,
    cancellation_token: CancellationToken | None,
) -> bool:
    delay = 0.2 * attempt
    if time.monotonic() + delay >= deadline:
        return False
    if cancellation_token:
        if cancellation_token.wait(delay):
            cancellation_token.raise_if_cancelled()
    else:
        time.sleep(delay)
    return True


def embed_texts(
    texts: Sequence[str],
    model: str = DEFAULT_EMBED_MODEL,
    timeout: int = 60,
    cancellation_token: CancellationToken | None = None,
) -> list[list[float]]:
    """
    Embed one or more texts using Ollama, returning Chroma-compatible vectors.
    
    This function communicates with a local Ollama instance to generate vector
    embeddings for the provided strings through the shared resource coordinator.
    
    Args:
        texts (Sequence[str]): A list or tuple of string documents to embed.
        model (str): The Ollama model name to use for embeddings (e.g., 'embeddinggemma').
        timeout (int): The maximum number of seconds for all batches and retries.
        
    Returns:
        list[list[float]]: Floating-point vectors corresponding exactly to the inputs.
        
    Raises:
        RuntimeError: If Ollama fails or the returned vectors do not match the inputs.
    """
    inputs = _clean_inputs(texts)
    if not inputs:
        return []
    model_name = _validate_model(model)
    if isinstance(timeout, bool) or not isinstance(timeout, int):
        raise ValueError("timeout must be an integer between 1 and 300 seconds")
    timeout_seconds = timeout
    if not 1 <= timeout_seconds <= 300:
        raise ValueError("timeout must be between 1 and 300 seconds")
    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    thread_id = threading.get_ident()
    coordinator = _EMBED_SERVICE.coordinator
    owner = (
        coordinator.current_context_owner()
        or f"embedding:{thread_id}:{time.monotonic_ns()}"
    )
    deadline = time.monotonic() + timeout_seconds
    embeddings: list[list[float]] = []
    embedding_value_count = 0
    batch_count = math.ceil(len(inputs) / MAX_EMBED_BATCH_SIZE)

    for batch_index, start in enumerate(
        range(0, len(inputs), MAX_EMBED_BATCH_SIZE),
        start=1,
    ):
        batch = inputs[start:start + MAX_EMBED_BATCH_SIZE]
        attempt = 0
        while True:
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"Embedding deadline expired before batch "
                    f"{batch_index}/{batch_count} completed"
                )
            attempt += 1
            try:
                response = _EMBED_SERVICE.embed(
                    batch,
                    owner=owner,
                    model=model_name,
                    cancellation_token=cancellation_token,
                    wait_timeout=remaining,
                    operation_timeout=remaining,
                )
                batch_embeddings = _validate_embedding_count(
                    normalize_embeddings(response),
                    len(batch),
                )
                embedding_value_count += sum(len(vector) for vector in batch_embeddings)
                if embedding_value_count > MAX_TOTAL_EMBEDDING_VALUES:
                    raise RuntimeError(
                        "Combined embedding response exceeds the "
                        f"{MAX_TOTAL_EMBEDDING_VALUES}-value limit"
                    )
                embeddings.extend(batch_embeddings)
                break
            except OperationCancelled:
                raise
            except (OllamaRequestTimeout, OllamaUnavailableError) as exc:
                if attempt > MAX_RETRIES or not _retry_delay(
                    attempt,
                    deadline,
                    cancellation_token,
                ):
                    raise RuntimeError(
                        f"Embedding batch {batch_index}/{batch_count} failed "
                        f"after {attempt} attempt(s): {exc}"
                    ) from exc
            except CoordinatorError as exc:
                raise RuntimeError(
                    f"Embedding batch {batch_index}/{batch_count} could not acquire "
                    f"the coordinated model runtime: {exc}"
                ) from exc
            except OllamaRuntimeError as exc:
                raise RuntimeError(
                    f"Embedding batch {batch_index}/{batch_count} failed: {exc}"
                ) from exc
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Embedding batch {batch_index}/{batch_count} returned invalid data: {exc}"
                ) from exc

    normalized = _validate_embedding_count(embeddings, len(inputs))
    _record_model_dimension(model_name, normalized)
    return normalized


def embed_query(
    text: str,
    model: str = DEFAULT_EMBED_MODEL,
    timeout: int = 30,
    cancellation_token: CancellationToken | None = None,
) -> list[float]:
    embeddings = embed_texts(
        [text],
        model=model,
        timeout=timeout,
        cancellation_token=cancellation_token,
    )
    if not embeddings:
        raise RuntimeError("Ollama returned no embedding for query")
    return embeddings[0]
