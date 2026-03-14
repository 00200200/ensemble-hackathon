from __future__ import annotations

from typing import Dict


def generate_abstract(chunk: Dict) -> str:
    """
    Lightweight, deterministic "abstract" generator for a code chunk.

    This is a placeholder for a true LLM-driven abstraction step. It compresses
    the chunk into a short, human-readable description based on its type and
    name, without relying on external models or network calls.
    """
    if not isinstance(chunk, dict):
        return ""

    chunk_type = chunk.get("type") or "unknown"
    name = (chunk.get("name") or "").strip()
    rel_path = (
        chunk.get("rel_filepath") or chunk.get("filepath") or "unknown"
    ).strip()
    if not rel_path:
        rel_path = "unknown"

    # Grab just the first non-empty line of the chunk text as a hint.
    text = chunk.get("text") or ""
    if not isinstance(text, str):
        text = str(text)
    first_line = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break

    if chunk_type == "function":
        sig = first_line if (first_line and first_line.startswith("def ")) else f"def {name}(...)"
        return f"Function `{name}` in `{rel_path}`: {sig}"

    if chunk_type == "class":
        sig = first_line if (first_line and first_line.startswith("class ")) else f"class {name}: ..."
        return f"Class `{name}` in `{rel_path}`: {sig}"

    if chunk_type == "import":
        hint = first_line or name or "import"
        return f"Import in `{rel_path}`: {hint}"

    if chunk_type == "file_content":
        return f"File-level code in `{rel_path}`: {first_line or '(content)'}"

    return f"Code chunk `{name}` in `{rel_path}`: {first_line or '(code)'}"

