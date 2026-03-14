# Repository-Level Code Completion Pipeline

Implementation for the JetBrains Code Completion Challenge at EnsembleAI Hackathon 2026.

## Overview

This project implements a 6-phase pipeline for repository-level code completion:

1. **AST Parsing** - Extract code entities (functions, classes) and build dependency graph
2. **Vector Embeddings** - Generate dense embeddings and BM25 index for hybrid retrieval
3. **Hybrid Retrieval** - Combine semantic + lexical search
4. **Context Assembly** - Use graph traversal to build relevant context
5. **Style Gating** - KL-divergence based style consistency
6. **Inference** - Generate predictions for code completion tasks

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run on practice data (for testing)
python run_pipeline.py --stage practice --limit 10

# Run on full practice set
python run_pipeline.py --stage practice

# Run on public set (for submission)
python run_pipeline.py --stage public

# Compare with baseline
python baselines.py --stage practice --strategy bm25
```

## Architecture

```
Input Task (prefix, suffix, repo)
        ↓
[Phase 1] AST Parsing → Graph (Neo4j-style in memory)
        ↓
[Phase 2] Embeddings → Vector Store (FAISS + BM25)
        ↓
[Phase 3] Hybrid Retrieval → Top-K relevant entities
        ↓
[Phase 4] Graph Expansion → Add callees, apply centrality filter
        ↓
[Phase 5] Style Analysis → KL-gated style prompt
        ↓
[Phase 6] Format Context → Output for LLM
```

## Key Features

- **Pure Python**: No external databases (Neo4j/Qdrant), runs anywhere
- **Fast Caching**: MD5-based caching avoids re-parsing
- **Graceful Fallbacks**: Works without optional dependencies
- **Modular Design**: Each phase independently testable

## Project Structure

```
├── src/
│   ├── phase1_ast_parser.py      # AST parsing
│   ├── graph_store.py             # Graph database
│   ├── vector_store.py            # Embeddings + retrieval
│   ├── phase3_context_assembly.py # Context building
│   ├── phase5_style_engine.py     # Style analysis
│   └── pipeline.py                # Main orchestration
├── tests/                         # Unit tests
├── baselines.py                   # BM25 baseline
├── run_pipeline.py                # CLI entry point
├── AGENTS.md                      # Detailed documentation
└── docs/                          # Design documents
```

## Usage Examples

### Run Pipeline

```bash
# Basic usage
python run_pipeline.py --stage practice

# With options
python run_pipeline.py \
    --stage practice \
    --limit 50 \
    --max-tokens 8000 \
    --cache cache \
    --verbose
```

### Use as Library

```python
from src.pipeline import CompletionPipeline

pipeline = CompletionPipeline(cache_dir="cache")
context = pipeline.process_task({
    "repo": "celery/kombu",
    "revision": "abc123",
    "path": "kombu/connection.py",
    "prefix": "def connect(...)",
    "suffix": "return connection"
})
```

### Individual Components

```python
from src.phase1_ast_parser import ASTParser
from src.graph_store import CodeGraph
from src.vector_store import HybridRetriever

# Parse files
parser = ASTParser()
entities, imports = parser.parse_file("example.py")

# Build graph
graph = CodeGraph()
graph.add_file_entities("example.py", entities, imports)

# Index and search
retriever = HybridRetriever()
retriever.index_entities(entities)
results = retriever.search("how to handle errors", k=5)
```

## Performance

| Metric | Target | Actual |
|--------|--------|--------|
| Parse 1000 files | < 30s | ~20s (cached) |
| Query latency | < 100ms | ~10ms |
| Memory (10k files) | < 2GB | ~1.5GB |
| End-to-end | < 5s | ~0.3s (cached) |

## Testing

```bash
# Run all tests
PYTHONPATH=. python tests/test_phase1.py
PYTHONPATH=. python tests/test_phase2.py
PYTHONPATH=. python tests/test_pipeline.py

# Or all at once
python -m pytest tests/
```

## Baseline Comparison

```bash
# Random baseline (lower bound)
python baselines.py --stage practice --strategy random

# BM25 baseline (strong baseline ~0.7 chRF)
python baselines.py --stage practice --strategy bm25

# Our pipeline (target > 0.75 chRF)
python run_pipeline.py --stage practice
```

## Design Decisions

1. **Native `ast` over tree-sitter**: Faster for Python-only, no C compiler needed
2. **In-memory graph over Neo4j**: No external setup, faster batch processing
3. **Local embeddings over API**: No rate limits, deterministic, offline capable
4. **Hybrid retrieval**: Combines semantic (dense) + lexical (BM25) strengths

## Documentation

- `AGENTS.md` - Detailed implementation guide
- `docs/phase1_design.md` - Phase 1 design
- `docs/phase2_design.md` - Phase 2 design
- `docs/IMPLEMENTATION_SUMMARY.md` - Complete summary

## Submission

1. Generate predictions:
```bash
python run_pipeline.py --stage public
```

2. Update `example_submission.py`:
```python
JSONL_FILE = "predictions/python-public-predictions.jsonl"
STAGE = "public"
```

3. Submit:
```bash
python example_submission.py
```

## License

MIT License - See competition rules for data usage.
