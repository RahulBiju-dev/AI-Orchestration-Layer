#!/usr/bin/env python3
"""Selene entry point: validate local runtime, then launch CLI or web UI."""

import sys

from agent.model_lifecycle import ModelStartupResult, ensure_managed_model
from agent.ollama_runtime import InvalidModelfileError, OllamaRuntimeError, OllamaService
from agent.platform_runtime import get_runtime_paths, resource_path
from agent.runtime_config import RuntimeConfig, RuntimeConfigurationError, get_runtime_config

from agent.terminal import _console, print_error, print_info, print_lab_status, print_ok, print_warn


def _get_modelfile_path() -> str:
    """Return the path to the Modelfile, handling PyInstaller's temp directory.
    
    When running as a packaged PyInstaller executable, files are extracted to a 
    temporary _MEIPASS directory. Otherwise, they are relative to this script.
    
    Returns:
        str: Absolute path to the Modelfile.
    """
    return str(resource_path("Modelfile"))

def _ensure_model(
    config: RuntimeConfig | None = None,
    service: OllamaService | None = None,
) -> ModelStartupResult:
    """Verify or safely stage-build Selene's managed model alias."""
    runtime = config or get_runtime_config()
    return ensure_managed_model(
        config=runtime,
        service=service,
        modelfile_path=_get_modelfile_path(),
    )


def main() -> None:
    """Main execution point for the application."""
    if "--doctor" in sys.argv:
        from agent.diagnostics import main_doctor

        raise SystemExit(main_doctor(as_json="--json" in sys.argv))

    service: OllamaService | None = None
    try:
        runtime = get_runtime_config(refresh=True)
        runtime_paths = get_runtime_paths()
        print_info(f"Runtime profile · {runtime.profile.value}")
        print_info(runtime.selection_reason)
        print_info(f"Runtime data · {runtime_paths.data_dir}", detail=runtime_paths.source)
        for warning in runtime.warnings:
            print_warn(warning)

        service = OllamaService(runtime)
        print_lab_status(
            f"Verifying local model '{runtime.chat_model}'…",
            kind="run",
        )
        model_result = _ensure_model(runtime, service)
        if model_result.action in {"built", "rebuilt"}:
            print_ok(
                f"Model {model_result.action} through a verified staging alias"
            )
        else:
            print_ok(f"Model ready ({model_result.action})")
        _console.print()

        if "--cli" in sys.argv:
            from agent.core import run

            run()
        else:
            from agent.web import start_web_server

            start_web_server() 
    except KeyboardInterrupt:
        _console.print()
        print_info("Interrupted — goodbye")
    except (RuntimeConfigurationError, InvalidModelfileError, OllamaRuntimeError) as exc:
        _console.print()
        print_error(f"Selene startup failed · {exc}")
        raise SystemExit(1) from None
    finally:
        if service is not None:
            service.coordinator.shutdown(cancel_active=True, wait=False)
        tool_runner = sys.modules.get("agent.tool_runner")
        shutdown = getattr(tool_runner, "shutdown_tool_runner", None)
        if callable(shutdown):
            shutdown(wait=False)


if __name__ == "__main__":
    main()
