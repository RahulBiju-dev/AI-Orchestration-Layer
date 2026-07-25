"""Bounded, non-destructive Selene environment diagnostics (``--doctor``)."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable

from agent.platform_runtime import (
    capability_report,
    get_runtime_paths,
    platform_family,
    resource_path,
)
from agent.runtime_config import RuntimeConfigurationError, get_runtime_config
from tools.registry import TOOL_DISPATCH, TOOL_METADATA, validate_tool_registry


CheckFn = Callable[[], dict[str, Any]]


def _status(ok: bool, **fields: Any) -> dict[str, Any]:
    payload = {"ok": bool(ok), **fields}
    return payload


def _check_python() -> dict[str, Any]:
    return _status(
        True,
        python_version=platform.python_version(),
        implementation=platform.python_implementation(),
        executable=sys.executable,
    )


def _check_operating_system() -> dict[str, Any]:
    family = platform_family()
    detail: dict[str, Any] = {
        "family": family,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    if family == "linux":
        release_file = Path("/etc/os-release")
        if release_file.is_file():
            try:
                values: dict[str, str] = {}
                for line in release_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    if "=" not in line or line.startswith("#"):
                        continue
                    key, raw = line.split("=", 1)
                    values[key] = raw.strip().strip('"')
                detail["distro"] = values.get("PRETTY_NAME") or values.get("NAME")
                detail["id"] = values.get("ID")
                detail["version_id"] = values.get("VERSION_ID")
            except OSError as exc:
                detail["distro_error"] = str(exc)
    elif family == "windows":
        detail["windows_version"] = platform.version()
    return _status(True, **detail)


def _check_runtime_paths() -> dict[str, Any]:
    paths = get_runtime_paths()
    writable: dict[str, bool] = {}
    for label, directory in (
        ("data_dir", paths.data_dir),
        ("state_dir", paths.state_dir),
        ("config_dir", paths.config_dir),
        ("cache_dir", paths.cache_dir),
    ):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / f".selene-doctor-{os.getpid()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            writable[label] = True
        except OSError:
            writable[label] = False
    ok = all(writable.values())
    return _status(
        ok,
        source=paths.source,
        paths=paths.report(),
        writable=writable,
        remedy=None if ok else "Ensure the runtime data directories are writable, or set SELENE_DATA_DIR.",
    )


def _check_runtime_config() -> dict[str, Any]:
    try:
        config = get_runtime_config(refresh=True)
    except RuntimeConfigurationError as exc:
        return _status(False, error=str(exc), remedy="Check SELENE_* profile and option overrides.")
    return _status(
        True,
        profile=config.profile.value,
        selection_reason=config.selection_reason,
        chat_model=config.chat_model,
        embedding_model=config.embedding_model,
        vision_model=getattr(config, "vision_model", None),
        num_ctx=config.num_ctx,
        num_predict=config.num_predict,
        num_batch=config.num_batch,
        warnings=list(config.warnings),
    )


def _check_ollama() -> dict[str, Any]:
    cli = shutil.which("ollama")
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    api_ok = False
    version = None
    models: list[str] = []
    error = None
    try:
        import urllib.request

        request = urllib.request.Request(f"{host}/api/version", method="GET")
        with urllib.request.urlopen(request, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
            version = payload.get("version")
            api_ok = True
        tags_request = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(tags_request, timeout=2.0) as response:
            tags = json.loads(response.read().decode("utf-8", errors="replace"))
            models = sorted(
                str(item.get("name"))
                for item in tags.get("models", [])
                if isinstance(item, dict) and item.get("name")
            )[:40]
    except Exception as exc:
        error = str(exc)
    return _status(
        api_ok,
        cli_available=bool(cli),
        cli_path=cli,
        host=host,
        version=version,
        model_count=len(models),
        models=models,
        error=error,
        remedy=None if api_ok else "Start Ollama (`ollama serve`) and ensure OLLAMA_HOST is reachable.",
    )


def _check_models(config_payload: dict[str, Any], ollama_payload: dict[str, Any]) -> dict[str, Any]:
    models = {name.casefold() for name in ollama_payload.get("models") or []}
    chat = str(config_payload.get("chat_model") or "")
    embedding = str(config_payload.get("embedding_model") or "")
    vision = str(config_payload.get("vision_model") or "")

    def present(name: str) -> bool | None:
        if not name:
            return None
        if not ollama_payload.get("ok"):
            return None
        base = name.casefold()
        return any(base == item or item.startswith(base + ":") or base.startswith(item) for item in models)

    chat_ok = present(chat)
    embed_ok = present(embedding)
    vision_ok = present(vision)
    ok = all(value is not False for value in (chat_ok, embed_ok, vision_ok))
    return _status(
        ok if ollama_payload.get("ok") else False,
        chat_model={"name": chat, "available": chat_ok},
        embedding_model={"name": embedding, "available": embed_ok},
        vision_model={"name": vision, "available": vision_ok},
        remedy=None if ok else "Pull or rebuild the missing models with Ollama before running chat/RAG/vision.",
    )


def _check_gpu() -> dict[str, Any]:
    nvidia = shutil.which("nvidia-smi")
    if not nvidia:
        return _status(True, available=False, detail="nvidia-smi not found; CPU or non-NVIDIA GPU assumed.")
    try:
        completed = __import__("subprocess").run(
            [nvidia, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
            shell=False,
        )
    except Exception as exc:
        return _status(True, available=False, error=str(exc), detail="GPU probe failed safely.")
    if completed.returncode != 0:
        return _status(
            True,
            available=False,
            detail="nvidia-smi could not report usable GPU data; low-vram profile may be selected.",
            stderr=(completed.stderr or "")[:400],
        )
    gpus = []
    for line in (completed.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "memory_total_mib": parts[1], "memory_free_mib": parts[2]})
    return _status(True, available=bool(gpus), gpus=gpus)


def _check_tool_registry() -> dict[str, Any]:
    errors = validate_tool_registry()
    support = {
        name: {
            "fedora": meta.fedora_support,
            "windows": meta.windows_support,
            "model_exposed": meta.model_exposed,
            "optional_dependencies": list(meta.optional_dependencies),
        }
        for name, meta in sorted(TOOL_METADATA.items())
    }
    return _status(
        not errors,
        tool_count=len(TOOL_DISPATCH),
        errors=errors,
        support=support,
        remedy=None if not errors else "Fix tools/registry.py metadata/dispatch mismatches.",
    )


def _check_optional_dependencies() -> dict[str, Any]:
    modules = {
        "chromadb": "chromadb",
        "ddgs": "ddgs",
        "pypdf": "pypdf",
        "python-docx": "docx",
        "openpyxl": "openpyxl",
        "xlrd": "xlrd",
        "xlwt": "xlwt",
        "pdf2image": "pdf2image",
        "reportlab": "reportlab",
        "cryptography": "cryptography",
        "google-api-python-client": "googleapiclient",
        "google-auth-oauthlib": "google_auth_oauthlib",
        "keyring": "keyring",
        "dbus-python": "dbus",
        "ollama": "ollama",
        "requests": "requests",
        "rich": "rich",
    }
    present: dict[str, bool] = {}
    for label, module_name in modules.items():
        if label == "dbus-python" and platform_family() != "linux":
            present[label] = False
            continue
        present[label] = importlib.util.find_spec(module_name) is not None
    return _status(True, dependencies=present)


def _check_poppler() -> dict[str, Any]:
    pdftoppm = shutil.which("pdftoppm")
    poppler_path = os.environ.get("POPPLER_PATH") or os.environ.get("SELENE_POPPLER_PATH")
    available = bool(pdftoppm or (poppler_path and Path(poppler_path).exists()))
    return _status(
        True,
        available=available,
        pdftoppm=pdftoppm,
        poppler_path=poppler_path,
        detail="Text PDF extraction works without Poppler; PDF-to-image needs Poppler.",
        remedy=None if available else "Install Poppler (Fedora: poppler-utils) for PDF page images.",
    )


def _check_capabilities() -> dict[str, Any]:
    report = capability_report()
    return _status(True, capabilities=report)


def _check_port_availability() -> dict[str, Any]:
    preferred = int(os.environ.get("SELENE_PORT", "8765") or "8765")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            sock.bind(("127.0.0.1", preferred))
        return _status(True, preferred_port=preferred, available=True)
    except OSError as exc:
        return _status(
            False,
            preferred_port=preferred,
            available=False,
            error=str(exc),
            remedy=f"Free port {preferred} or set SELENE_PORT to an open local port.",
        )


def _check_packaged_resources() -> dict[str, Any]:
    try:
        modelfile = resource_path("Modelfile")
        external_prompt = resource_path("agent/prompts/external_models.md")
        static_dir = resource_path("agent/static")
    except Exception as exc:
        return _status(
            False,
            error=str(exc),
            remedy=(
                "Repackage with Modelfile, agent/prompts, and agent/static included."
            ),
        )
    ok = (
        modelfile.is_file()
        and external_prompt.is_file()
        and static_dir.is_dir()
    )
    return _status(
        ok,
        modelfile=str(modelfile),
        external_prompt=str(external_prompt),
        static_dir=str(static_dir),
        remedy=(
            None
            if ok
            else "Ensure Modelfile, agent/prompts, and agent/static are present in the install tree."
        ),
    )


def run_diagnostics(*, include_network: bool = True) -> dict[str, Any]:
    """Run all diagnostic checks and return a redacted structured report."""
    started = time.monotonic()
    checks: dict[str, dict[str, Any]] = {}
    checks["python"] = _check_python()
    checks["operating_system"] = _check_operating_system()
    checks["runtime_paths"] = _check_runtime_paths()
    checks["runtime_config"] = _check_runtime_config()
    checks["ollama"] = _check_ollama() if include_network else _status(True, skipped=True)
    checks["models"] = _check_models(checks["runtime_config"], checks["ollama"])
    checks["gpu"] = _check_gpu()
    checks["tool_registry"] = _check_tool_registry()
    checks["optional_dependencies"] = _check_optional_dependencies()
    checks["poppler"] = _check_poppler()
    checks["capabilities"] = _check_capabilities()
    checks["port"] = _check_port_availability()
    checks["packaged_resources"] = _check_packaged_resources()

    failed = [name for name, payload in checks.items() if not payload.get("ok", False)]
    return {
        "ok": not failed,
        "failed_checks": failed,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "checks": checks,
    }


def format_diagnostics_report(report: dict[str, Any]) -> str:
    """Render a concise human-readable diagnostics summary."""
    lines = ["Selene diagnostics", "=================="]
    overall = "PASS" if report.get("ok") else "ISSUES FOUND"
    lines.append(f"Overall: {overall}")
    if report.get("failed_checks"):
        lines.append("Failed: " + ", ".join(report["failed_checks"]))
    lines.append("")
    for name, payload in (report.get("checks") or {}).items():
        marker = "✓" if payload.get("ok") else "✗"
        summary_bits = []
        for key in ("profile", "family", "source", "version", "detail", "error", "remedy"):
            if payload.get(key):
                summary_bits.append(f"{key}={payload[key]}")
        if name == "tool_registry":
            summary_bits.append(f"tools={payload.get('tool_count')}")
        if name == "runtime_config":
            summary_bits.append(f"num_ctx={payload.get('num_ctx')}")
            summary_bits.append(f"chat_model={payload.get('chat_model')}")
        suffix = f" ({'; '.join(summary_bits)})" if summary_bits else ""
        lines.append(f"{marker} {name}{suffix}")
    lines.append("")
    lines.append(f"Elapsed: {report.get('elapsed_seconds')}s")
    return "\n".join(lines)


def main_doctor(*, as_json: bool = False) -> int:
    """CLI entry used by ``python main.py --doctor``."""
    report = run_diagnostics()
    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        print(format_diagnostics_report(report))
    return 0 if report.get("ok") else 1
