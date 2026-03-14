#!/usr/bin/env python3
"""
verify_dag_architecture.py

A deterministic diagnostic tool for the refactored RAG pipeline.
Creates a temporary mock repository, runs the AST chunker,
verifies that dependency edges (calls, bases) are correctly extracted,
and then tests that the topological sort places definitions before usages.
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Ensure we can import the local modules (chunker, main)
sys.path.insert(0, str(Path(__file__).parent))

from chunker import process_repository
from main import build_dependency_order, format_context, FILE_SEP_TOKEN


def create_mock_repo(temp_dir: Path) -> None:
    """Create the three Python files inside temp_dir."""
    files = {
        "utils.py": """
def matrix_multiply(a, b):
    return a @ b
""",
        "base_models.py": """
class GraphNode:
    def __init__(self, val):
        self.val = val
""",
        "core.py": """
from utils import matrix_multiply
from base_models import GraphNode

class NeuralGraph(GraphNode):
    def forward(self, x, weights):
        # <gap>
        return matrix_multiply(x, weights)
"""
    }
    for fname, content in files.items():
        (temp_dir / fname).write_text(content, encoding="utf-8")


def find_chunk_by_name(chunks, name: str, expected_type: str = None):
    """Return the first chunk with matching name and optional type."""
    for c in chunks:
        if c.get("name") == name:
            if expected_type is None or c.get("type") == expected_type:
                return c
    return None


def main():
    print("=== DAG Architecture Verification ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)
        print(f"Created temporary repository at: {repo_path}")

        # 1. Create mock files
        create_mock_repo(repo_path)
        print("Mock repository files created:\n  utils.py, base_models.py, core.py\n")

        # 2. Run chunker
        print("Running process_repository...")
        chunks = process_repository(str(repo_path))
        print(f"Extracted {len(chunks)} chunks total.\n")

        # 3. Locate the three key chunks
        neural = find_chunk_by_name(chunks, "NeuralGraph", "class")
        graph = find_chunk_by_name(chunks, "GraphNode", "class")
        matrix = find_chunk_by_name(chunks, "matrix_multiply", "function")

        if not all([neural, graph, matrix]):
            missing = []
            if not neural: missing.append("NeuralGraph")
            if not graph: missing.append("GraphNode")
            if not matrix: missing.append("matrix_multiply")
            print(f"ERROR: Could not find chunks for: {', '.join(missing)}")
            sys.exit(1)

        print("--- PHASE 1: AST EXTRACTION (NODES & EDGES) ---")
        print(f"NeuralGraph chunk   : {neural.get('filepath')} (type={neural['type']})")
        print(f"GraphNode chunk     : {graph.get('filepath')} (type={graph['type']})")
        print(f"matrix_multiply chunk: {matrix.get('filepath')} (type={matrix['type']})")
        print()

        # Verify dependencies
        print("Checking dependency edges:")
        bases = neural.get("bases", [])
        calls = neural.get("calls", [])
        print(f"  NeuralGraph.bases   = {bases}")
        print(f"  NeuralGraph.calls   = {calls}")

        assert "GraphNode" in bases, "NeuralGraph should inherit from GraphNode"
        assert "matrix_multiply" in calls, "NeuralGraph.forward should call matrix_multiply"
        print("✅ AST edges correct.\n")

        # 4. Topological sort test
        print("--- PHASE 2: TOPOLOGICAL SORT (DAG) ---")
        # Create an unsorted list (simulating retrieval order)
        unsorted = [neural, graph, matrix]   # NeuralGraph first, then GraphNode, then matrix_multiply
        print("Unsorted list (simulated retrieval):")
        for i, c in enumerate(unsorted):
            print(f"  {i}: {c['name']} ({c['type']})")

        sorted_chunks = build_dependency_order(unsorted)
        print("\nTopologically sorted list:")
        for i, c in enumerate(sorted_chunks):
            print(f"  {i}: {c['name']} ({c['type']})")

        # Verify order: matrix and GraphNode must come before NeuralGraph
        names_sorted = [c['name'] for c in sorted_chunks]
        try:
            pos_matrix = names_sorted.index("matrix_multiply")
            pos_graph = names_sorted.index("GraphNode")
            pos_neural = names_sorted.index("NeuralGraph")
            assert pos_matrix < pos_neural, "matrix_multiply must appear before NeuralGraph"
            assert pos_graph < pos_neural, "GraphNode must appear before NeuralGraph"
            print("✅ Topological order respects dependencies (definitions before usage).")
        except ValueError as e:
            print(f"ERROR: Missing expected chunk in sorted list: {e}")
            sys.exit(1)

        # 5. Final context assembly
        print("\n--- PHASE 3: FINAL FIM CONTEXT ---")
        context_str = format_context(sorted_chunks, FILE_SEP_TOKEN)
        print(context_str)

        # Optional: verify file separator tokens appear
        if FILE_SEP_TOKEN in context_str:
            print(f"\n✅ File separator '{FILE_SEP_TOKEN}' present.")
        else:
            print(f"\n⚠️  File separator '{FILE_SEP_TOKEN}' missing!")

    print("\n=== Verification complete. All assertions passed. ===")


if __name__ == "__main__":
    main()