#!/usr/bin/env python3
"""
main.py — Entry point for the Gemma CLI Agent.

Creates the custom Ollama model from the Modelfile (if needed)
and launches the interactive chat loop.
"""

import sys

import ollama

from agent.core import MODEL_NAME, run

from rich.console import Console

_console = Console()


def _ensure_model() -> None:
    """Create the custom model from the Modelfile if it doesn't already exist."""
    try:
        ollama.show(MODEL_NAME)
    except ollama.ResponseError:
        _console.print(f"[cyan bold]⟳  Building model '{MODEL_NAME}' from Modelfile…[/]")
        ollama.create(model=MODEL_NAME, from_="./Modelfile")
        _console.print("[cyan bold]✓  Model ready.[/]\n")


def main() -> None:
    try:
        _ensure_model()
        run()
    except KeyboardInterrupt:
        _console.print("\n[dim]Interrupted — goodbye.[/]")
        sys.exit(0)
    except ollama.ResponseError as exc:
        _console.print(f"\n[red bold]Ollama error:[/red bold] {exc}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
