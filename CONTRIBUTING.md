# Contributing to Selene (AI Orchestration Layer)

Thank you for your interest in contributing to Selene! We welcome contributions of all forms, including bug fixes, feature requests, documentation improvements, new tool integrations, and interface work across Web UI, Terminal CLI, and desktop packaging.

The following guide outlines the process for proposing changes to this codebase.

## Code of Conduct

All contributors must adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it to understand the community standards we expect.

## How Can I Contribute?

### Reporting Bugs or Feature Requests

If you find a bug or have a feature idea, please search the existing issues before creating a new one. If no existing issue covers the topic, open a new one using our issue templates.

### Contributing Code

1. **Fork the Repository**: Create your own copy of the repository on GitHub.
2. **Create a Feature Branch**: Build your changes on a dedicated branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Set Up the Environment**:
   - Ensure you have Python 3.10+ installed.
   - Install the project dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Ensure Ollama is running and has the models pulled:
     ```bash
     ollama pull gemma4:e4b
     ollama pull embeddinggemma
     ollama pull moondream
     ```
4. **Implement Your Changes**:
   - Follow standard Python naming and design conventions (snake_case, clean docstrings, logical layouts).
   - Keep tool schemas and implementations modular.
5. **Test Your Changes**:
   - Run unit tests (no pytest required):
     ```bash
     python -m unittest discover -s tests -v
     python main.py --doctor
     ```
   - Optionally run the agent locally with Ollama available:
     ```bash
     python main.py
     ```
   - Verify tool calls work as expected. Prefer Fedora-first native behavior and native Windows parity without WSL.
6. **Submit a Pull Request**: Submit your pull request to the `main` branch. Provide a clear description of the changes and references to any related issues.

See [AGENTS.md](AGENTS.md) for architecture ownership rules and [docs/platform-support.md](docs/platform-support.md) for the tool support matrix.

---

## Adding New Tools

Selene's orchestration core is designed to be highly modular. Tools registered once are available to every interface (Web, CLI, and desktop). You can add a new tool by following these steps:

### 1. Implement the Tool Function
Create or edit a python file under the `tools/` directory (e.g., `tools/my_new_tool.py`).
Implement your function. It should accept arguments and return a string (or JSON string/dict representation) representing the tool's result.

```python
# tools/my_new_tool.py

def fetch_weather(location: str) -> str:
    """
    Fetches the current weather for a specific location.
    """
    # ... call weather API or run local command ...
    return f"The weather in {location} is sunny, 22°C."
```

### 2. Register the Schema
Open [tools/registry.py](tools/registry.py) and add your tool's JSON schema to `TOOL_SCHEMAS`. Ensure the parameter descriptions are concise to minimize context token overhead.

```python
# tools/registry.py

TOOL_SCHEMAS = [
    # ... other schemas ...
    {
        "name": "fetch_weather",
        "description": "Fetch the current weather conditions for a given location",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "location": {
                    "type": "STRING",
                    "description": "The city and country/state, e.g., 'London, UK'"
                }
            },
            "required": ["location"]
        }
    }
]
```

### 3. Register the Dispatch Route
In the same file ([tools/registry.py](tools/registry.py)), import your function and map it in the `TOOL_DISPATCH` dictionary.

```python
from tools.my_new_tool import fetch_weather

TOOL_DISPATCH = {
    # ... other routes ...
    "fetch_weather": fetch_weather,
}
```

### 4. Test Tool Integration
Start the agent and ask a question that triggers the tool:
```
>>> What's the weather like in Seattle right now?
```
The agent should automatically recognize the prompt, call `fetch_weather`, display status updates, and synthesize a final response using the tool output.
