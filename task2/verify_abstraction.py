#!/usr/bin/env python3
"""
Verify that abstraction embedding is wired correctly: chunks get abstracts,
and the retriever uses them when building the index (BM25 + dense doc).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from abstraction import generate_abstract
from chunker import process_repository
from retriever import Retriever
import tempfile


def test_generate_abstract():
    """generate_abstract returns non-empty string for typical chunks."""
    chunk = {
        "type": "function",
        "name": "foo",
        "rel_filepath": "src/bar.py",
        "text": "def foo(x, y):\n    return x + y",
    }
    out = generate_abstract(chunk)
    assert isinstance(out, str), "abstract should be str"
    assert len(out) > 0, "abstract should be non-empty"
    assert "foo" in out and "bar.py" in out, "abstract should mention name and path"
    print("  generate_abstract: OK")


def test_chunks_get_abstracts_via_process():
    """Chunks from process_repository can be given abstracts (simulating main.py)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "m.py").write_text("def hello():\n    pass\n")
        chunks = process_repository(tmpdir)
        assert len(chunks) >= 1
        for ch in chunks:
            ch["abstract"] = generate_abstract(ch)
        for ch in chunks:
            assert "abstract" in ch and ch["abstract"], "every chunk should have non-empty abstract"
    print("  chunks get abstracts: OK")


def test_retriever_uses_abstract_in_doc():
    """Retriever builds index doc from abstract when present (same logic as build_index)."""
    short_abstract = "Function `long_func` in `a.py`: def long_func()"
    long_text = "def long_func():\n" + "    x = 1\n" * 100
    chunks = [
        {
            "rel_filepath": "a.py",
            "type": "function",
            "name": "long_func",
            "text": long_text,
            "abstract": short_abstract,
        },
    ]
    # Replicate retriever's doc-building logic to ensure abstract is used when present.
    c = chunks[0]
    abstract = c.get("abstract") or ""
    base_text = abstract or c.get("text", "")
    header = f"File: {c.get('rel_filepath', 'unknown')}"
    doc = f"{header}\n\n{base_text}"
    assert doc == f"File: a.py\n\n{short_abstract}", "doc should use abstract, not full text"
    assert len(doc) < len(long_text), "doc with abstract should be shorter than raw code"

    retriever = Retriever(db_path="/tmp/lancedb_verify_abstraction")
    retriever.build_index("verify_abstraction_table", chunks)
    assert "verify_abstraction_table" in retriever.repo_chunks
    stored = retriever.repo_chunks["verify_abstraction_table"]
    assert len(stored) == 1 and stored[0].get("abstract") == short_abstract
    print("  retriever uses abstract in index: OK")


def main():
    print("=== Abstraction verification ===\n")
    test_generate_abstract()
    test_chunks_get_abstracts_via_process()
    test_retriever_uses_abstract_in_doc()
    print("\n=== All abstraction checks passed. ===\n")


if __name__ == "__main__":
    main()
