"""Central runtime configuration and optional hardware profile selection.

The resolver in this module is intentionally independent from the CLI and web
frontends.  Both frontends can therefore resolve the same settings while still
supplying session-scoped overrides.

Default behaviour follows the bundled ``Modelfile`` parameters. Hardware
auto-selection only runs when the user (or a session) explicitly requests the
``auto`` profile. Explicit settings are never silently reduced.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, MutableMapping


class RuntimeConfigurationError(ValueError):
    """Raised when a known runtime setting has an unsafe or invalid value."""


class RuntimeProfile(str, Enum):
    AUTO = "auto"
    LOW_VRAM = "low-vram"
    BALANCED = "balanced"
    MANUAL = "manual"


@dataclass(frozen=True)
class HardwareInfo:
    """Small, local-only hardware snapshot used for profile selection."""

    gpu_name: str | None = None
    gpu_vram_mb: int | None = None
    system_memory_mb: int | None = None
    detector: str = "none"
    reason: str = "Hardware detection was not run."


@dataclass(frozen=True)
class UserConfigLoadResult:
    """Result of reading (never rewriting) a user runtime configuration file."""

    values: Mapping[str, Any]
    path: Path
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved model/resource settings plus their selection evidence."""

    requested_profile: RuntimeProfile
    profile: RuntimeProfile
    selection_reason: str
    hardware: HardwareInfo

    chat_model: str
    base_model: str
    embedding_model: str
    vision_model: str

    num_ctx: int
    num_predict: int
    num_batch: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    keep_alive: str

    model_concurrency: int
    heavy_tool_concurrency: int
    tool_workers: int
    serialize_embeddings: bool
    serialize_vision: bool

    chat_timeout_seconds: float
    title_timeout_seconds: float
    summary_timeout_seconds: float
    embedding_timeout_seconds: float
    vision_timeout_seconds: float
    build_timeout_seconds: float

    warnings: tuple[str, ...] = ()
    sources: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    def ollama_options(self) -> dict[str, int | float]:
        """Return the centrally managed Ollama option subset."""
        return {
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "num_batch": self.num_batch,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
        }

    def timeout_for(self, operation: str | Enum) -> float:
        """Return the configured timeout for a model operation name."""
        name = str(getattr(operation, "value", operation)).strip().lower()
        mapping = {
            "chat": self.chat_timeout_seconds,
            "title": self.title_timeout_seconds,
            "summary": self.summary_timeout_seconds,
            "embedding": self.embedding_timeout_seconds,
            "vision": self.vision_timeout_seconds,
            "build": self.build_timeout_seconds,
        }
        try:
            return mapping[name]
        except KeyError as exc:
            raise RuntimeConfigurationError(f"Unknown model operation: {name!r}") from exc

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["requested_profile"] = self.requested_profile.value
        data["profile"] = self.profile.value
        return data


# Defaults mirror the bundled Modelfile PARAMETER lines (plus non-Modelfile
# runtime knobs that Ollama/Selene still need centrally). Hardware profiles may
# diverge only when the user explicitly selects low-vram / balanced / auto.
_MODELFILE_DEFAULTS: dict[str, Any] = {
    "chat_model": "selene",
    "base_model": "gemma4:e4b",
    "embedding_model": "embeddinggemma",
    "vision_model": "moondream",
    "num_ctx": 8192,
    "num_predict": 768,
    "num_batch": 128,
    "temperature": 0.25,
    "top_p": 0.85,
    "top_k": 40,
    "repeat_penalty": 1.08,
    "keep_alive": "10m",
    "model_concurrency": 1,
    "heavy_tool_concurrency": 1,
    "tool_workers": 2,
    "serialize_embeddings": True,
    "serialize_vision": True,
    "chat_timeout_seconds": 180.0,
    "title_timeout_seconds": 30.0,
    "summary_timeout_seconds": 90.0,
    "embedding_timeout_seconds": 180.0,
    "vision_timeout_seconds": 180.0,
    "build_timeout_seconds": 900.0,
}

# Backward-compatible alias used by older call sites / docs.
_CONSERVATIVE_DEFAULTS: dict[str, Any] = dict(_MODELFILE_DEFAULTS)

_PROFILE_DEFAULTS: dict[RuntimeProfile, dict[str, Any]] = {
    # Explicit opt-in for constrained GPUs (~4 GiB class).
    RuntimeProfile.LOW_VRAM: {
        **_MODELFILE_DEFAULTS,
        "num_ctx": 4096,
        "model_concurrency": 1,
        "heavy_tool_concurrency": 1,
        "tool_workers": 2,
        "serialize_embeddings": True,
        "serialize_vision": True,
    },
    RuntimeProfile.BALANCED: {
        **_MODELFILE_DEFAULTS,
        "num_ctx": 8192,
        "num_predict": 1536,
        "num_batch": 512,
        "keep_alive": "30m",
        "model_concurrency": 2,
        "tool_workers": 4,
    },
    # Manual / default: stay on Modelfile parameters unless the user overrides.
    RuntimeProfile.MANUAL: dict(_MODELFILE_DEFAULTS),
}

_ENV_SETTINGS = {
    "SELENE_CHAT_MODEL": "chat_model",
    "SELENE_BASE_MODEL": "base_model",
    "SELENE_EMBEDDING_MODEL": "embedding_model",
    "SELENE_VISION_MODEL": "vision_model",
    "SELENE_NUM_CTX": "num_ctx",
    "SELENE_NUM_PREDICT": "num_predict",
    "SELENE_NUM_BATCH": "num_batch",
    "SELENE_TEMPERATURE": "temperature",
    "SELENE_TOP_P": "top_p",
    "SELENE_TOP_K": "top_k",
    "SELENE_REPEAT_PENALTY": "repeat_penalty",
    "SELENE_KEEP_ALIVE": "keep_alive",
    "SELENE_MODEL_CONCURRENCY": "model_concurrency",
    "SELENE_HEAVY_TOOL_CONCURRENCY": "heavy_tool_concurrency",
    "SELENE_TOOL_WORKERS": "tool_workers",
    "SELENE_SERIALIZE_EMBEDDINGS": "serialize_embeddings",
    "SELENE_SERIALIZE_VISION": "serialize_vision",
    "SELENE_CHAT_TIMEOUT": "chat_timeout_seconds",
    "SELENE_TITLE_TIMEOUT": "title_timeout_seconds",
    "SELENE_SUMMARY_TIMEOUT": "summary_timeout_seconds",
    "SELENE_EMBEDDING_TIMEOUT": "embedding_timeout_seconds",
    "SELENE_VISION_TIMEOUT": "vision_timeout_seconds",
    "SELENE_BUILD_TIMEOUT": "build_timeout_seconds",
}

_INT_RANGES = {
    "num_ctx": (1024, 131072),
    "num_predict": (64, 32768),
    "num_batch": (1, 4096),
    "top_k": (0, 200),
    "model_concurrency": (1, 8),
    "heavy_tool_concurrency": (1, 8),
    "tool_workers": (1, 32),
}
_FLOAT_RANGES = {
    "chat_timeout_seconds": (1.0, 3600.0),
    "title_timeout_seconds": (1.0, 600.0),
    "summary_timeout_seconds": (1.0, 1800.0),
    "embedding_timeout_seconds": (1.0, 3600.0),
    "vision_timeout_seconds": (1.0, 3600.0),
    "build_timeout_seconds": (30.0, 7200.0),
}
_NUMBER_RANGES = {
    "temperature": (0.0, 2.0),
    "top_p": (0.01, 1.0),
    "repeat_penalty": (0.1, 2.0),
}
_BOOL_FIELDS = {"serialize_embeddings", "serialize_vision"}
_MODEL_FIELDS = {"chat_model", "base_model", "embedding_model", "vision_model"}
_SETTING_FIELDS = set(_CONSERVATIVE_DEFAULTS)
_KEEP_ALIVE_RE = re.compile(r"^(?:-1|0|[0-9]+(?:\.[0-9]+)?(?:ms|s|m|h))$")


def _system_memory_mb() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        return max(1, (page_size * pages) // (1024 * 1024))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def detect_hardware(timeout_seconds: float = 2.0) -> HardwareInfo:
    """Inspect NVIDIA VRAM with one bounded command, falling back safely.

    Absence or failure of ``nvidia-smi`` is not an error and never triggers a
    benchmark or network request.  Unknown/non-NVIDIA systems select the
    conservative profile unless the user chooses otherwise.
    """
    try:
        timeout = max(0.2, min(float(timeout_seconds), 5.0))
    except (TypeError, ValueError):
        timeout = 2.0
    memory_mb = _system_memory_mb()
    executable = shutil.which("nvidia-smi")
    if not executable:
        return HardwareInfo(
            system_memory_mb=memory_mb,
            reason="nvidia-smi is unavailable; using the conservative low-VRAM profile.",
        )

    command = [
        executable,
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "timeout": timeout,
        "check": False,
    }
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        result = subprocess.run(command, **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        return HardwareInfo(
            system_memory_mb=memory_mb,
            detector="nvidia-smi",
            reason=f"nvidia-smi could not be queried ({type(exc).__name__}); using the conservative low-VRAM profile.",
        )

    candidates: list[tuple[str, int]] = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            try:
                name, raw_mb = line.rsplit(",", 1)
                amount = int(float(raw_mb.strip()))
            except (TypeError, ValueError):
                continue
            if amount > 0:
                candidates.append((name.strip() or "NVIDIA GPU", amount))
    if not candidates:
        return HardwareInfo(
            system_memory_mb=memory_mb,
            detector="nvidia-smi",
            reason="nvidia-smi returned no usable VRAM value; using the conservative low-VRAM profile.",
        )

    # Choose the smallest visible GPU so automatic settings remain safe if the
    # runtime's eventual GPU choice differs from our assumption.
    gpu_name, gpu_vram_mb = min(candidates, key=lambda item: item[1])
    return HardwareInfo(
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        system_memory_mb=memory_mb,
        detector="nvidia-smi",
        reason=f"Detected {gpu_name} with {gpu_vram_mb} MiB VRAM.",
    )


def _hardware_profile(hardware: HardwareInfo) -> tuple[RuntimeProfile, str]:
    if hardware.gpu_vram_mb is None:
        return RuntimeProfile.LOW_VRAM, hardware.reason
    if hardware.gpu_vram_mb < 7168:
        return (
            RuntimeProfile.LOW_VRAM,
            f"Detected {hardware.gpu_vram_mb} MiB VRAM; selected the conservative low-VRAM profile.",
        )
    return (
        RuntimeProfile.BALANCED,
        f"Detected {hardware.gpu_vram_mb} MiB VRAM; selected the balanced profile.",
    )


def _parse_profile(value: Any) -> RuntimeProfile:
    if isinstance(value, RuntimeProfile):
        return value
    # Empty / missing values default to manual (Modelfile parameters), not auto.
    normalized = str(value if value is not None else RuntimeProfile.MANUAL.value).strip().lower().replace("_", "-")
    if not normalized:
        return RuntimeProfile.MANUAL
    try:
        return RuntimeProfile(normalized)
    except ValueError as exc:
        choices = ", ".join(profile.value for profile in RuntimeProfile)
        raise RuntimeConfigurationError(f"Invalid runtime profile {value!r}; expected one of: {choices}") from exc


def _flatten_settings(values: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        return {}
    flattened: dict[str, Any] = {}
    runtime = values.get("runtime")
    if isinstance(runtime, Mapping):
        flattened.update(runtime)
    options = values.get("options")
    if isinstance(options, Mapping):
        flattened.update(options)
    flattened.update({key: value for key, value in values.items() if key not in {"runtime", "options"}})
    if "runtime_profile" in flattened and "profile" not in flattened:
        flattened["profile"] = flattened["runtime_profile"]
    return flattened


def _environment_settings(environ: Mapping[str, str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if "SELENE_RUNTIME_PROFILE" in environ:
        values["profile"] = environ["SELENE_RUNTIME_PROFILE"]
    for environment_name, setting_name in _ENV_SETTINGS.items():
        if environment_name in environ:
            values[setting_name] = environ[environment_name]
    return values


def _coerce_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeConfigurationError(f"{name} must be true or false, not {value!r}")


def _coerce_setting(name: str, value: Any) -> Any:
    if name in _INT_RANGES:
        if isinstance(value, bool):
            raise RuntimeConfigurationError(f"{name} must be an integer, not a boolean")
        if isinstance(value, float) and not value.is_integer():
            raise RuntimeConfigurationError(f"{name} must be an integer, not {value!r}")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeConfigurationError(f"{name} must be an integer, not {value!r}") from exc
        minimum, maximum = _INT_RANGES[name]
        if not minimum <= parsed <= maximum:
            raise RuntimeConfigurationError(f"{name} must be between {minimum} and {maximum}, got {parsed}")
        return parsed

    if name in _FLOAT_RANGES or name in _NUMBER_RANGES:
        if isinstance(value, bool):
            raise RuntimeConfigurationError(f"{name} must be a number, not a boolean")
        try:
            parsed_float = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeConfigurationError(f"{name} must be a number, not {value!r}") from exc
        minimum, maximum = (_FLOAT_RANGES | _NUMBER_RANGES)[name]
        if not minimum <= parsed_float <= maximum:
            unit = " seconds" if name in _FLOAT_RANGES else ""
            raise RuntimeConfigurationError(f"{name} must be between {minimum:g} and {maximum:g}{unit}")
        return parsed_float

    if name in _BOOL_FIELDS:
        return _coerce_bool(name, value)

    if name in _MODEL_FIELDS:
        model = str(value or "").strip()
        if not model or len(model) > 255 or any(character.isspace() or ord(character) < 32 for character in model):
            raise RuntimeConfigurationError(f"{name} is not a valid Ollama model name")
        return model

    if name == "keep_alive":
        keep_alive = str(value).strip().lower()
        if not _KEEP_ALIVE_RE.fullmatch(keep_alive):
            raise RuntimeConfigurationError(
                "keep_alive must be 0, -1, or a duration such as 30s, 10m, or 1h"
            )
        return keep_alive

    raise RuntimeConfigurationError(f"Unknown centralized runtime setting: {name}")


def _apply_layer(
    resolved: MutableMapping[str, Any],
    sources: MutableMapping[str, str],
    raw_values: Mapping[str, Any],
    source: str,
) -> None:
    for name, value in raw_values.items():
        if name in _SETTING_FIELDS:
            resolved[name] = _coerce_setting(name, value)
            sources[name] = source


def _configuration_warnings(
    resolved: Mapping[str, Any],
    hardware_profile: RuntimeProfile,
    hardware: HardwareInfo,
) -> list[str]:
    reference = _PROFILE_DEFAULTS[hardware_profile]
    warnings: list[str] = []
    for name, label in (
        ("num_ctx", "context size"),
        ("num_batch", "prompt batch size"),
        ("model_concurrency", "model concurrency"),
        ("heavy_tool_concurrency", "heavy-tool concurrency"),
    ):
        if resolved[name] > reference[name]:
            warnings.append(
                f"Configured {label} ({resolved[name]}) exceeds the conservative "
                f"{hardware_profile.value} reference ({reference[name]}) for the detected hardware."
            )
    if hardware.gpu_vram_mb is not None and hardware.gpu_vram_mb <= 4096:
        if not resolved["serialize_embeddings"] or not resolved["serialize_vision"]:
            warnings.append(
                "Embedding and vision serialization is recommended for GPUs with 4 GiB VRAM or less."
            )
    return warnings


def resolve_runtime_config(
    session_overrides: Mapping[str, Any] | None = None,
    *,
    user_config: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
    hardware: HardwareInfo | None = None,
    inherited_warnings: tuple[str, ...] | list[str] = (),
) -> RuntimeConfig:
    """Resolve config with session > environment > user > profile precedence.

    Unconfigured systems default to the bundled Modelfile parameter set and the
    ``manual`` profile. Hardware-based selection only runs when ``auto`` is
    requested explicitly.
    """
    environment = os.environ if environ is None else environ
    user_values = _flatten_settings(user_config)
    environment_values = _environment_settings(environment)
    session_values = _flatten_settings(session_overrides)

    profile_value: Any = RuntimeProfile.MANUAL.value
    profile_source = "Modelfile defaults"
    if "profile" in user_values:
        profile_value, profile_source = user_values["profile"], "user config"
    if "profile" in environment_values:
        profile_value, profile_source = environment_values["profile"], "environment"
    if "profile" in session_values:
        profile_value, profile_source = session_values["profile"], "session override"
    requested_profile = _parse_profile(profile_value)

    # Hardware inspection is advisory for warnings. It only *selects* a profile
    # when the user explicitly requests ``auto``.
    detected = hardware or detect_hardware()
    detected_profile, detected_reason = _hardware_profile(detected)
    if requested_profile is RuntimeProfile.AUTO:
        selected_profile = detected_profile
        selection_reason = detected_reason
    elif requested_profile is RuntimeProfile.MANUAL:
        selected_profile = RuntimeProfile.MANUAL
        if profile_source == "Modelfile defaults":
            selection_reason = (
                "Using bundled Modelfile parameter defaults "
                "(hardware profile auto-selection is off unless profile=auto)."
            )
        else:
            selection_reason = (
                f"Manual profile selected by {profile_source}; "
                "explicit values are preserved over hardware defaults."
            )
    else:
        selected_profile = requested_profile
        selection_reason = f"{selected_profile.value} profile selected by {profile_source}."

    values = dict(_MODELFILE_DEFAULTS)
    sources: dict[str, str] = {name: "Modelfile default" for name in values}
    if selected_profile in (
        RuntimeProfile.LOW_VRAM,
        RuntimeProfile.BALANCED,
        RuntimeProfile.MANUAL,
    ):
        for name, value in _PROFILE_DEFAULTS[selected_profile].items():
            values[name] = value
            sources[name] = (
                "Modelfile default"
                if selected_profile is RuntimeProfile.MANUAL
                else f"{selected_profile.value} profile"
            )

    _apply_layer(values, sources, user_values, "user config")
    _apply_layer(values, sources, environment_values, "environment")
    _apply_layer(values, sources, session_values, "session override")

    if values["num_predict"] > values["num_ctx"] - 512:
        raise RuntimeConfigurationError(
            "num_predict must leave at least 512 tokens in num_ctx for prompt and safety overhead"
        )
    if values["heavy_tool_concurrency"] > values["tool_workers"]:
        raise RuntimeConfigurationError("heavy_tool_concurrency cannot exceed tool_workers")

    warnings = list(inherited_warnings)
    warnings.extend(_configuration_warnings(values, detected_profile, detected))
    return RuntimeConfig(
        requested_profile=requested_profile,
        profile=selected_profile,
        selection_reason=selection_reason,
        hardware=detected,
        warnings=tuple(dict.fromkeys(str(warning) for warning in warnings if warning)),
        sources=sources,
        **values,
    )


def default_user_config_path(environ: Mapping[str, str] | None = None) -> Path:
    """Return the config path without creating directories or files."""
    environment = os.environ if environ is None else environ
    explicit = environment.get("SELENE_CONFIG_FILE")
    if explicit:
        return Path(explicit).expanduser()

    # platform_runtime is optional during early startup and unit tests.  Its
    # contract is consulted lazily so this module does not create an import
    # cycle with packaged-path handling.
    try:
        from agent.platform_runtime import get_runtime_paths, resolve_runtime_paths  # type: ignore

        paths = get_runtime_paths() if environment is os.environ else resolve_runtime_paths(environ=environment)
        config_dir = getattr(paths, "config_dir", None)
        if config_dir:
            return Path(config_dir) / "runtime_config.json"
    except (ImportError, AttributeError, OSError, TypeError):
        pass

    data_dir = environment.get("SELENE_DATA_DIR")
    if data_dir:
        return Path(data_dir).expanduser() / "runtime_config.json"
    return Path.home() / ".selene-agent" / "runtime_config.json"


def load_user_config(path: str | os.PathLike[str]) -> UserConfigLoadResult:
    """Read a JSON config without modifying malformed or unreadable input."""
    config_path = Path(path).expanduser()
    if not config_path.exists():
        return UserConfigLoadResult(values={}, path=config_path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return UserConfigLoadResult(
            values={},
            path=config_path,
            warnings=(
                f"Runtime config {config_path} was preserved but not loaded: {type(exc).__name__}.",
            ),
        )
    if not isinstance(payload, Mapping):
        return UserConfigLoadResult(
            values={},
            path=config_path,
            warnings=(f"Runtime config {config_path} was preserved but not loaded: root must be a JSON object.",),
        )
    return UserConfigLoadResult(values=dict(payload), path=config_path)


class RuntimeConfigManager:
    """Thread-safe, lazy owner of hardware/user configuration snapshots."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._hardware: HardwareInfo | None = None
        self._user_result: UserConfigLoadResult | None = None
        self._config_path: Path | None = None

    def refresh(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        config_path: str | os.PathLike[str] | None = None,
    ) -> RuntimeConfig:
        environment = os.environ if environ is None else environ
        path = Path(config_path).expanduser() if config_path is not None else default_user_config_path(environment)
        hardware = detect_hardware()
        loaded = load_user_config(path)
        with self._lock:
            self._hardware = hardware
            self._user_result = loaded
            self._config_path = path
        return resolve_runtime_config(
            user_config=loaded.values,
            environ=environment,
            hardware=hardware,
            inherited_warnings=loaded.warnings,
        )

    def resolve(
        self,
        session_overrides: Mapping[str, Any] | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimeConfig:
        environment = os.environ if environ is None else environ
        expected_path = default_user_config_path(environment)
        with self._lock:
            needs_refresh = (
                self._hardware is None
                or self._user_result is None
                or self._config_path != expected_path
            )
        if needs_refresh:
            self.refresh(environ=environment, config_path=expected_path)
        with self._lock:
            assert self._hardware is not None and self._user_result is not None
            hardware = self._hardware
            user_result = self._user_result
        return resolve_runtime_config(
            session_overrides,
            user_config=user_result.values,
            environ=environment,
            hardware=hardware,
            inherited_warnings=user_result.warnings,
        )


_RUNTIME_CONFIG_MANAGER = RuntimeConfigManager()


def get_runtime_config(
    session_overrides: Mapping[str, Any] | None = None,
    *,
    refresh: bool = False,
    environ: Mapping[str, str] | None = None,
) -> RuntimeConfig:
    """Resolve the process/user configuration, optionally with session values."""
    if refresh:
        _RUNTIME_CONFIG_MANAGER.refresh(environ=environ)
    return _RUNTIME_CONFIG_MANAGER.resolve(session_overrides, environ=environ)
