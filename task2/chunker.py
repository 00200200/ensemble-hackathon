import os
import json
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import tree_sitter_python
from tree_sitter import Language, Parser


# Initialize tree-sitter parser for Python
PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PY_LANGUAGE)

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
    
    # We query for function definitions and class definitions
    query_str = """
    (function_definition
      name: (identifier) @func.name
    ) @function.def
    
    (class_definition
      name: (identifier) @class.name
    ) @class.def
    """
    
    try:
        query = PY_LANGUAGE.query(query_str)
        captures = query.captures(tree.root_node)
    except Exception:
        # Fallback to whole file if querying fails
        chunks.append({
            "filepath": filepath,
            "text": source_bytes.decode('utf-8', errors='ignore'),
            "type": "file_content"
        })
        return chunks

    # Reconstruct blocks based on captures. captures is a list of tuples: (Node, string_tag)
    # The list is ordered by occurrence. A function.def or class.def is typically 
    # matched along with its name (func.name or class.name).
    
    # Simple strategy: iterate captures and find full block definitions
    for node, name in captures.items():
        if name in ["function.def", "class.def"]:
            chunk_type = "function" if name == "function.def" else "class"
            
            # Find the name node which is a child
            name_node = None
            for child in node.children:
                if child.type == "identifier":
                    name_node = child
                    break
            
            block_name = ""
            if name_node:
                block_name = source_bytes[name_node.start_byte:name_node.end_byte].decode('utf-8')

            chunk_text = source_bytes[node.start_byte:node.end_byte].decode('utf-8')
            chunks.append({
                "filepath": filepath,
                "name": block_name,
                "text": chunk_text,
                "type": chunk_type
            })

    # If the file has no functions/classes, include the whole file
    if not chunks:
        chunks.append({
            "filepath": filepath,
            "text": source_bytes.decode('utf-8', errors='ignore'),
            "type": "file_content"
        })

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
