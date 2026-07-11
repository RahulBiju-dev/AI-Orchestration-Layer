# Selene Agent Guide

Concise guidance for contributors and coding agents working in this repository.

## Platform policy

- **Fedora Linux is the primary reference platform.** Preserve native DBus, desktop-entry discovery, XDG paths, POSIX process groups, GNOME terminals, and AppImage packaging.
- **Windows 10/11 are first-class secondary platforms** with native parity (no WSL, Git Bash, Cygwin, MSYS2, or containers required).
- Prefer platform-specific native backends over weakest-common-denominator generics.
- Do not replace Fedora-native behavior to simplify Windows support.

## Reliability priorities

1. Correctness and data integrity
2. Safe cancellation / shutdown
3. Structured, truthful error reporting
4. Optional-dependency isolation (missing packages must not block core startup)
5. Fedora regression prevention
6. Windows parity or explicit capability limits

## Architecture map

| Area | Authoritative module |
|------|----------------------|
| Entry / doctor | `main.py`, `agent/diagnostics.py` |
| Runtime profiles | `agent/runtime_config.py` |
| Ollama API + coordinator | `agent/ollama_runtime.py` |
| Managed model lifecycle | `agent/model_lifecycle.py` |
| CLI agent loop | `agent/core.py` |
| Web/SSE runtime | `agent/web.py`, `agent/web_runtime.py` |
| Tool execution | `agent/tool_runner.py` |
| Tool registry | `tools/registry.py` |
| Platform contracts | `agent/platform_runtime.py` |
| Atomic persistence | `agent/persistence.py` |
| Desktop packaging | `electron/`, `selene-backend.spec` |

## Configuration hierarchy

Resolve settings in this order:

1. Session / command override
2. Environment or user config (`SELENE_*`)
3. Selected profile (`manual`, `low-vram`, `balanced`, or explicit `auto`)
4. Bundled Modelfile parameter defaults (no hardware auto-pick unless `profile=auto`)

Key knobs: `num_ctx`, `num_predict`, `num_batch`, model concurrency, tool workers, keep-alive.

## Ollama coordinator rules

- All chat, title, summary, embedding, and vision work goes through the shared coordinator.
- Never start a second Ollama server or stop an external Ollama process.
- Managed model rebuilds use staged alias → inspect → publish → metadata.
- Low-VRAM mode serializes model-heavy operations without serializing ordinary tools.

## Session ownership

- Saved sessions: one active generation across tabs.
- Unsaved sessions: isolated per browser client/tab.
- Generation IDs own cancellation and have one terminal SSE state: `completed`, `cancelled`, or `failed`.

## Tool metadata

- `tools/registry.py` is authoritative for schemas, dispatch, and `ToolMetadata`.
- All normal execution (CLI, web, slash commands, routines) must use `agent/tool_runner.py`.
- Metadata covers side effects, parallel safety, resource weight, cancellation, timeout, output bounds, platform support, and optional dependencies.
- Non-idempotent side-effect failures and timeouts block later side effects in the same batch.

## Platform adapter usage

- Paths, process spawn/kill, terminals, browsers, and native open use `agent/platform_runtime.py`.
- Linux: desktop entries, DBus Spotify (lazy), XDG roots, process groups.
- Windows: `%LOCALAPPDATA%\Selene`, Start Menu discovery with shortcut-target validation, Terminal/PowerShell/cmd, ShellExecute, `taskkill /T` only for owned trees.

## Persistence rules

- Runtime root: `SELENE_DATA_DIR` > existing `~/.selene-agent` > platform default (XDG or LocalAppData).
- Never silently migrate, copy, merge, or delete user data stores.
- Critical JSON uses atomic temp + fsync + replace; malformed files are preserved.

## Optional dependency rules

- `dbus-python` is Linux-only (`sys_platform == "linux"` marker).
- Google, Poppler, pdf2image, Chroma, and vision tooling fail with capability errors — they must not prevent import or core startup.
- Electron deps are separate (`package.json` / Bun).

## Testing requirements

- Prefer `unittest` (no pytest required for collection).
- Do not download models, use GPUs, real OAuth, live Spotify, or personal data in tests.
- Mock platforms; use temp directories.
- Full suite: `python -m unittest discover -s tests -v`
- Diagnostics: `python main.py --doctor`

## Packaging expectations

- PyInstaller backend keeps `console=True` for Electron port announcement; Windows hides the console via Electron spawn flags.
- Artifact names follow `package.json` version (`Selene-${version}...`).
- Desktop helper always forces `--publish never`.
- Uninstall must not delete user runtime data (`deleteAppDataOnUninstall: false`).

## Do not edit manually

- Personal runtime data under `~/.selene-agent`, XDG Selene dirs, or `%LOCALAPPDATA%\Selene`
- Generated `dist/`, `dist-electron/`, `node_modules/`, `__pycache__/`
- Live Chroma indexes or encrypted Google credential stores as part of ordinary development