"""Provider-aware default system prompts shared by every Selene interface."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Mapping, Sequence

from agent.platform_runtime import resource_path


LOCAL_MODEL_ID = "local:default"
EXTERNAL_SYSTEM_PROMPT_PATH = "agent/prompts/external_models.md"


def extract_local_system_prompt() -> str:
    """Read the local model's authoritative prompt from the bundled Modelfile."""
    try:
        text = resource_path("Modelfile").read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(r'SYSTEM\s+"""(.*?)"""', text, re.DOTALL)
    return match.group(1).strip() if match else ""


@lru_cache(maxsize=1)
def load_external_system_prompt() -> str:
    """Load the provider-neutral prompt used by configured API models."""
    try:
        return resource_path(EXTERNAL_SYSTEM_PROMPT_PATH).read_text(
            encoding="utf-8"
        ).strip()
    except OSError as exc:
        raise RuntimeError(
            "The bundled external-model system prompt is unavailable."
        ) from exc


def default_system_prompt_for_model(
    model_id: object,
    *,
    local_prompt: str = "",
) -> str:
    """Return the default prompt belonging to the selected inference model."""
    selected = str(model_id or LOCAL_MODEL_ID).strip() or LOCAL_MODEL_ID
    if selected == LOCAL_MODEL_ID or selected.startswith("local:"):
        return str(local_prompt or extract_local_system_prompt()).strip()
    return load_external_system_prompt()


def active_system_prompt_for_session(
    session: Mapping[str, object] | None,
    *,
    local_prompt: str = "",
) -> str:
    """Resolve an explicit conversation override or the model-owned default."""
    source = session or {}
    override = str(source.get("system") or "").strip()
    if override:
        return override
    return default_system_prompt_for_model(
        source.get("model_id"),
        local_prompt=local_prompt,
    )


def apply_active_system_prompt(
    messages: Sequence[Mapping[str, object]],
    prompt: str,
) -> list[dict]:
    """Replace every anchored system copy without disturbing message order."""
    active = str(prompt or "").strip()
    normalized = [dict(message) for message in messages]
    system_indexes = [
        index
        for index, message in enumerate(normalized)
        if message.get("role") == "system"
    ]
    if not active:
        return [
            message for message in normalized if message.get("role") != "system"
        ]
    if not system_indexes:
        return [{"role": "system", "content": active}, *normalized]
    for index in system_indexes:
        normalized[index]["content"] = active
    return normalized
