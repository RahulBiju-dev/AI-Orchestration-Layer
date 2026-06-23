#!/usr/bin/env python3
"""
main.py — Entry point for the Selene AI Agent.

Creates the custom Ollama model from the Modelfile (if needed)
and launches the interactive chat loop.
"""

import sys
import os

import ollama

from agent.core import MODEL_NAME, run

from rich.console import Console

_console = Console()


def _get_modelfile_path() -> str:
    """Return the path to the Modelfile, handling PyInstaller's temp directory."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, 'Modelfile')
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Modelfile')

def _ensure_model() -> None:
    """Create the custom model from the Modelfile if it doesn't already exist."""
    try:
        # Check if the required custom model is already present in Ollama
        ollama.show(MODEL_NAME)
    except ollama.ResponseError:
        # If the model is not found, build it from the local Modelfile
        _console.print(f"[cyan bold]⟳  Building model '{MODEL_NAME}' from Modelfile…[/]")
        ollama.create(model=MODEL_NAME, from_=_get_modelfile_path())
        _console.print("[cyan bold]✓  Model ready.[/]\n")


def main() -> None:
    """Main execution point for the application."""
    try:
        # Ensure the model is available before launching any interfaces
        _ensure_model()
        
        # Check arguments to determine which interface to launch
        if "--cli" in sys.argv:
            # Launch the terminal-based interactive CLI
            run()
        else:
            # Launch the threaded web server for the browser interface
            from agent.web import start_web_server
            start_web_server() 
    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C to exit without stack traces
        _console.print("\n[dim]Interrupted — goodbye.[/]")
        sys.exit(0)
    except ollama.ResponseError as exc:
        # Catch and display Ollama-specific connection or model errors
        _console.print(f"\n[red bold]Ollama error:[/red bold] {exc}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
