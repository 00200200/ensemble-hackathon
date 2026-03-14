import os
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm
import tree_sitter_python
from tree_sitter import Language, Node, Parser


# Initialize tree-sitter parser for Python
PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PY_LANGUAGE)

# Core Tree-sitter queries for Python code structure
CORE_QUERY = """
(function_definition name: (identifier) @func.name) @function.def
(class_definition name: (identifier) @class.name) @class.def
"""

BASE_QUERY = """
(class_definition superclasses: (argument_list (identifier) @class.base)) @class.with_base
"""

CALL_QUERY = """
(call function: (identifier) @called_func)
(call function: (attribute attribute: (identifier) @called_func))
"""

IMPORT_QUERY = """
(import_statement) @import.stmt
(import_from_statement) @importfrom.stmt
"""

def extract_repo(zip_path: str, extract_to: str):
    """Extracts a repository zip file to a specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def chunk_python_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Parses a Python file using tree-sitter and extracts classes and functions as individual chunks.
    Returns a list of dictionaries containing chunk text and metadata.
    """
    chunks = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source_bytes = f.read().encode('utf-8')
    except Exception:
        return chunks

    if not source_bytes.strip():
        return chunks

    tree = parser.parse(source_bytes)
    source_text = source_bytes.decode("utf-8", errors="ignore")

    # Core definitions: functions and classes (independent of inheritance)
    core_query = PY_LANGUAGE.query(CORE_QUERY)
    core_captures = core_query.captures(tree.root_node)

    # Reconstruct blocks based on captures. `captures` is a list of tuples:
    # (Node, capture_name). The list is ordered by occurrence. A `function.def`
    # or `class.def` is typically matched along with its name (`func.name` or
    # `class.name`).

    def_spans: List[Tuple[Node, Dict[str, Any]]] = []

    for node, capture_name in core_captures:
        if capture_name in ("function.def", "class.def"):
            chunk_type = "function" if capture_name == "function.def" else "class"

            # Find the identifier child node which holds the name.
            name_node: Optional[Node] = None
            for child in node.children:
                if child.type == "identifier":
                    name_node = child
                    break

            block_name = ""
            if name_node:
                block_name = source_bytes[
                    name_node.start_byte:name_node.end_byte
                ].decode("utf-8", errors="ignore")

            chunk_text = source_bytes[
                node.start_byte:node.end_byte
            ].decode("utf-8", errors="ignore")

            chunk: Dict[str, Any] = {
                "filepath": filepath,
                "name": block_name,
                "text": chunk_text,
                "type": chunk_type,
            }
            chunks.append(chunk)
            def_spans.append((node, chunk))

    # If the file has no functions/classes, include the whole file
    if not chunks:
        chunks.append({
            "filepath": filepath,
            "text": source_text,
            "type": "file_content",
        })
        return chunks

    # ------------------------------------------------------------------
    # 1. AST-based call graph: identify real call sites inside each chunk
    # ------------------------------------------------------------------
    #
    # We collect identifiers that appear in call position, either as:
    #   func(...)  -> identifier
    #   obj.method(...) -> attribute with an identifier `method`
    #
    # and then match them against locally defined names.
    calls_per_chunk: Dict[int, Set[str]] = {id(c): set() for c in chunks}
    defined_names = {c.get("name") for c in chunks if c.get("name")}

    call_query = PY_LANGUAGE.query(CALL_QUERY)
    call_captures = call_query.captures(tree.root_node)

    def find_chunk_for_node(node: Node) -> Optional[Dict[str, Any]]:
        start = node.start_byte
        end = node.end_byte
        for def_node, chunk in def_spans:
            if def_node.start_byte <= start and end <= def_node.end_byte:
                return chunk
        return None

    for node, cap_name in call_captures:
        if cap_name != "called_func":
            continue
        name = source_bytes[node.start_byte:node.end_byte].decode(
            "utf-8",
            errors="ignore",
        )
        if name not in defined_names:
            continue
        owner_chunk = find_chunk_for_node(node)
        if owner_chunk is None:
            continue
        calls_per_chunk[id(owner_chunk)].add(name)

    for chunk in chunks:
        chunk["calls"] = sorted(calls_per_chunk.get(id(chunk), set()))

    # ------------------------------------------------------------------
    # 2. Base class extraction: record simple inheritance dependencies
    # ------------------------------------------------------------------
    base_query = PY_LANGUAGE.query(BASE_QUERY)
    base_captures = base_query.captures(tree.root_node)

    bases_per_chunk: Dict[int, Set[str]] = {}

    def find_class_chunk(node: Node) -> Optional[Dict[str, Any]]:
        # Walk up to the enclosing class_definition, then map via def_spans.
        current: Optional[Node] = node
        while current is not None and current.type != "class_definition":
            current = current.parent
        if current is None:
            return None
        for def_node, chunk in def_spans:
            if def_node is current:
                return chunk
        return None

    for node, cap_name in base_captures:
        if cap_name != "class.base":
            continue
        base_name = source_bytes[node.start_byte:node.end_byte].decode(
            "utf-8",
            errors="ignore",
        )
        owner_chunk = find_class_chunk(node)
        if owner_chunk is None:
            continue
        bases_per_chunk.setdefault(id(owner_chunk), set()).add(base_name)

    for chunk in chunks:
        chunk["bases"] = sorted(bases_per_chunk.get(id(chunk), set()))

    # ------------------------------------------------------------------
    # 3. Import chunks: treat import statements as independent retrievable
    #    units, since they often carry high-signal library information.
    # ------------------------------------------------------------------
    import_chunks: List[Dict[str, Any]] = []
    import_query = PY_LANGUAGE.query(IMPORT_QUERY)
    import_captures = import_query.captures(tree.root_node)

    def node_text(n: Node) -> str:
        return source_bytes[n.start_byte:n.end_byte].decode(
            "utf-8",
            errors="ignore",
        )

    for node, cap_name in import_captures:
        stmt_text = node_text(node)
        module_name = ""

        # Try to extract a dotted module name if present.
        for child in node.children:
            if child.type == "dotted_name":
                module_name = node_text(child)
                break

        if not module_name:
            module_name = stmt_text.strip()

        import_chunks.append({
            "filepath": filepath,
            "name": module_name,
            "text": stmt_text,
            "type": "import",
        })

    chunks.extend(import_chunks)
    return chunks


def process_repository(repo_dir: str) -> List[Dict[str, Any]]:
    """
    Walks through the extracted repository, chunking all Python files.
    """
    all_chunks = []
    repo_path = Path(repo_dir)
    
    for file_path in repo_path.rglob("*.py"):
        if file_path.is_file():
            rel_path = str(file_path.relative_to(repo_path))
            chunks = chunk_python_file(str(file_path))
            
            for chunk in chunks:
                chunk["rel_filepath"] = rel_path
                all_chunks.append(chunk)
                
    return all_chunks


if __name__ == "__main__":
    data_dir = Path("data/python-practice")
    sample_zip = next(data_dir.glob("*.zip"), None)
    
    if sample_zip:
        print(f"Testing tree-sitter chunking on {sample_zip.name}")
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_repo(str(sample_zip), temp_dir)
            chunks = process_repository(temp_dir)
            print(f"Extracted {len(chunks)} chunks from the repository.")
            if chunks:
                print("Sample chunk:")
                print(json.dumps(chunks[0], indent=2))
    else:
        print("No zip files found in data/python-practice/")
