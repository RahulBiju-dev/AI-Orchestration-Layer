"""
tools/code.py - Source code reading and viewing tool.

This module provides functionality to read and display source code files
with line numbers, handling truncation for very large files. It also allows
scanning directories for files matching specific extensions.
"""

import json
import os

# Comprehensive supported source code file extensions covering all common languages
SUPPORTED_EXTENSIONS = {
    # Web
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".html", ".htm", ".css", ".scss", ".sass", ".less",
    # Python
    ".py", ".pyw", ".pyi",
    # C/C++/Objective-C
    ".c", ".cpp", ".cc", ".cxx", ".c++", ".h", ".hpp", ".hh", ".hxx", ".m", ".mm",
    # Rust
    ".rs",
    # Go
    ".go",
    # Java
    ".java",
    # C#
    ".cs",
    # PHP
    ".php", ".phtml", ".php3", ".php4", ".php5", ".php7", ".phps",
    # Ruby
    ".rb", ".rbw", ".rake",
    # Swift
    ".swift",
    # Kotlin
    ".kt", ".kts",
    # R/Statistics
    ".r", ".R",
    # Scala
    ".scala",
    # Clojure
    ".clj", ".cljs", ".cljc", ".edn",
    # Lisp
    ".lisp", ".lsp", ".cl", ".fasl",
    # Haskell
    ".hs", ".lhs",
    # Erlang/Elixir
    ".erl", ".ex", ".exs",
    # Julia
    ".jl",
    # Perl
    ".pl", ".pm", ".t",
    # Lua
    ".lua",
    # Shell/Bash
    ".sh", ".bash", ".zsh", ".fish", ".ksh",
    # PowerShell
    ".ps1", ".psd1", ".psm1",
    # Groovy/Gradle
    ".groovy", ".gradle",
    # Dart
    ".dart",
    # Fortran
    ".f", ".f90", ".f95", ".f03", ".f08", ".f15", ".f18",
    # Pascal
    ".pas", ".pp", ".inc",
    # Assembly
    ".s", ".asm", ".S", ".ASM",
    # YAML
    ".yaml", ".yml",
    # JSON
    ".json",
    # XML
    ".xml",
    # Configuration
    ".conf", ".config", ".cfg", ".ini", ".toml",
    # Markdown
    ".md", ".markdown", ".mdown", ".mkdn", ".mkd",
    # SQL
    ".sql",
    # GraphQL
    ".graphql", ".gql",
    # Vue
    ".vue",
    # Dockerfile
    "Dockerfile",
    # Makefile
    "Makefile",
    # Properties/Env
    ".properties", ".env",
    # Protocol Buffers
    ".proto",
    # Thrift
    ".thrift",
}


def _read_single_file(file_path: str, lines: str | None = None) -> dict:
    """
    Read a single source code file and return structured data including its contents.
    
    This helper function reads the file line by line, applies any requested line
    ranges, and prepends line numbers to the output. It also enforces a size limit
    on the returned output string to avoid overwhelming output.
    
    Args:
        file_path (str): The path to the file to read.
        lines (str | None): An optional line range string (e.g., "10-20" or "42").
            If None, the entire file is processed.
            
    Returns:
        dict: A dictionary containing the file metadata and its contents (or error).
            Keys on success: "file", "extension", "total_lines", "displayed_lines", "code".
            Keys on failure: "error".
    """
    try:
        # Read all lines from the file
        with open(file_path, "r", encoding="utf-8") as f:
            file_lines = f.readlines()

        # Initialize line range to encompass the entire file
        start_line = 1
        end_line = len(file_lines)
        
        # Parse line range if provided
        if lines:
            try:
                if "-" in lines:
                    start_line, end_line = map(int, lines.split("-"))
                    # Bound start and end lines to valid ranges
                    start_line = max(1, start_line)
                    end_line = min(len(file_lines), end_line)
                else:
                    start_line = int(lines)
                    end_line = start_line
            except ValueError:
                return {"error": f"Invalid line range format: {lines}. Use '10-20' or '10'."}

        # Build output string with prepended line numbers
        output_lines = []
        for i in range(start_line - 1, end_line):
            if i < len(file_lines):
                line_num = i + 1
                line_content = file_lines[i].rstrip("\n")
                output_lines.append(f"{line_num:4d} | {line_content}")

        code_output = "\n".join(output_lines)

        # Add truncation notice if output text is too large
        if len(code_output) > 25000:
            code_output = code_output[:25000] + f"\n\n...[File truncated due to size limit. Showing first 25000 characters.]"

        file_ext = os.path.splitext(file_path)[1].lower()
        return {
            "file": file_path,
            "extension": file_ext if file_ext else os.path.basename(file_path),
            "total_lines": len(file_lines),
            "displayed_lines": f"{start_line}-{end_line}",
            "code": code_output
        }

    except UnicodeDecodeError:
        return {"error": f"Cannot read file: {file_path} (appears to be binary)"}
    except Exception as e:
        return {"error": f"Error reading file {file_path}: {str(e)}"}


def view_code(file_path: str, lines: str | None = None, extension: str | None = None) -> str:
    """
    View and read source code files with line numbers, or scan folders for files.

    Args:
        file_path: Path to a source code file or folder.
        lines: Optional line range for single files (e.g., "1-50" or "10-20").
        extension: When file_path is a folder, scan for files with this extension (e.g., ".py", ".js").

    Returns:
        A JSON string containing the code with line numbers, file list, or an error message.
    """
    if not os.path.exists(file_path):
        return json.dumps({"error": f"Path not found: {file_path}"})

    # Handle folder scanning
    if os.path.isdir(file_path):
        if not extension:
            return json.dumps({
                "error": f"{file_path} is a directory. Please provide an extension parameter (e.g., '.py', '.js') to scan for files."
            })

        # Normalize extension
        if not extension.startswith("."):
            extension = f".{extension}"

        # Scan folder recursively for files with given extension
        matching_files = []
        for root, dirs, files in os.walk(file_path):
            for file in files:
                if file.endswith(extension) or (extension == "Dockerfile" and file == "Dockerfile") or (extension == "Makefile" and file == "Makefile"):
                    matching_files.append(os.path.join(root, file))

        matching_files.sort()

        if not matching_files:
            return json.dumps({
                "folder": file_path,
                "extension": extension,
                "files_found": 0,
                "message": f"No files with extension '{extension}' found in {file_path}"
            })

        # Prepare file list with sizes
        file_info = []
        total_size = 0
        for file in matching_files:
            try:
                size = os.path.getsize(file)
                total_size += size
                file_info.append({
                    "path": file,
                    "size_bytes": size,
                    "size_kb": round(size / 1024, 2)
                })
            except Exception as e:
                file_info.append({
                    "path": file,
                    "error": str(e)
                })

        # If not too many files, read them all; otherwise just list them
        if len(matching_files) <= 10 and total_size <= 100000:
            # Read all files
            files_data = []
            for file_path_item in matching_files:
                file_data = _read_single_file(file_path_item, None)
                files_data.append(file_data)

            return json.dumps({
                "folder": file_path,
                "extension": extension,
                "files_found": len(matching_files),
                "files": files_data
            })
        else:
            # Just list the files
            return json.dumps({
                "folder": file_path,
                "extension": extension,
                "files_found": len(matching_files),
                "total_size_kb": round(total_size / 1024, 2),
                "file_list": file_info,
                "message": "Too many or too large files to read all at once. To view a specific file, use view_code with the full file path."
            })

    # Handle single file
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Check if it's a special file (Dockerfile, Makefile, etc.)
    basename = os.path.basename(file_path)
    if basename not in SUPPORTED_EXTENSIONS and file_ext not in SUPPORTED_EXTENSIONS:
        return json.dumps({
            "error": f"Unsupported file type: {file_ext if file_ext else basename}. Supported types: {', '.join(sorted(list(SUPPORTED_EXTENSIONS)[:20]))}... (and {len(SUPPORTED_EXTENSIONS) - 20} more)"
        })

    result = _read_single_file(file_path, lines)
    return json.dumps(result)
