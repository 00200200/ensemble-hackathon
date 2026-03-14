# AGENTS.md - Project Documentation and Guidelines

## Project Overview

**Task**: JetBrains Code Completion Challenge at EnsembleAI Hackathon 2026
**Goal**: Implement a context collection strategy that yields the most accurate code completions when provided to LLMs
**Evaluation**: ChrF score metric, averaged across three different models
**Target**: Beat SOTA baseline of ~0.7 chRF

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run baseline BM25
python baselines.py --stage practice --strategy bm25

# Run our pipeline
python run_pipeline.py --stage practice --limit 10

# Submit predictions
# (Update JSONL_FILE in example_submission.py, then run it)
```

## Architecture: The 6-Phase Pipeline

### Phase 1: Structural Ingestion & AST Partitioning ✅

**Files:**
- `src/phase1_ast_parser.py` - AST parsing and entity extraction
- `src/graph_store.py` - Graph database for relationships

**Key Features:**
- Native Python `ast` module for parsing (fast, no deps)
- Extracts: functions, classes, methods, constants
- Graph edges: DEFINES, CONTAINS, CALLS, IMPORTS
- Caching with MD5 hash invalidation

**Usage:**
```python
from src.phase1_ast_parser import ASTParser, ParseCache
from src.graph_store import CodeGraph

parser = ASTParser()
entities, imports = parser.parse_file("example.py")

graph = CodeGraph()
graph.add_file_entities("example.py", entities, imports)
```

### Phase 2: LLM-Driven Functional Abstraction ✅

**Files:**
- `src/vector_store.py` - Embeddings and hybrid retrieval
- `src/llm_abstraction.py` - LLM-generated function summaries

**Key Features:**
- Dense embeddings: sentence-transformers/all-MiniLM-L6-v2 (fallback to simple embedder)
- Sparse retrieval: BM25Okapi (fallback to simple BM25)
- Hybrid scoring: S(q,d) = α·Dense + (1-α)·BM25, α=0.5
- FAISS for fast nearest neighbor search
- **LLM Abstraction**: GPT-4o-mini, DeepSeek, or any OpenAI-compatible API

**LLM Abstraction Usage:**

OpenAI (default):
```bash
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"  # optional
python run_pipeline.py --stage practice --limit 10
```

DeepSeek (cheaper):
```bash
export DEEPSEEK_API_KEY="sk-..."
export LLM_MODEL="deepseek-chat"  # or deepseek-coder
python run_pipeline.py --stage practice --limit 10
```

Or any OpenAI-compatible API:
```bash
export LLM_API_KEY="sk-..."
export LLM_MODEL="your-model"
export LLM_BASE_URL="https://api.your-provider.com/v1"
python run_pipeline.py --stage practice --limit 10
```

**Test your setup:**
```bash
python test_llm.py
```

**LLM Provider Pricing (approximate):**
| Provider | Model | Input | Output | Cost per 1K functions* |
|----------|-------|-------|--------|------------------------|
| OpenAI | gpt-4o-mini | $0.15/M | $0.60/M | ~$0.05-0.10 |
| OpenAI | gpt-4o | $2.50/M | $10.00/M | ~$0.80-1.50 |
| DeepSeek | deepseek-chat | $0.14/M | $0.28/M | ~$0.04-0.08 |
| DeepSeek | deepseek-coder | $0.14/M | $0.28/M | ~$0.04-0.08 |

*Assuming ~500 tokens per function on average. DeepSeek is ~5-10x cheaper!

Without API key, falls back to signature + docstring + dependencies.

**Usage:**
```python
from src.vector_store import HybridRetriever

retriever = HybridRetriever(alpha=0.5)
retriever.index_entities(entities)
results = retriever.search("handle errors", k=10)
```

### Phase 3 & 4: Topological Context Assembly ✅

**Files:**
- `src/phase3_context_assembly.py` - Context assembly

**Key Features:**
- Seed selection: Top-K from hybrid retrieval
- Neighborhood expansion: 1-hop graph traversal
- Full code for seeds, abstracts for neighbors
- Centrality filtering: Penalty for high-degree nodes (utility functions)
- Token budget management

**Usage:**
```python
from src.phase3_context_assembly import ContextAssembler

assembler = ContextAssembler(graph, retriever)
context_items = assembler.assemble(
    query="how to handle errors",
    current_file="main.py",
    max_tokens=8000
)
context = assembler.format_context(context_items)
```

### Phase 5: Dynamic Style Engine (KL-Gated) ✅

**Files:**
- `src/phase5_style_engine.py` - Style analysis

**Key Features:**
- Style profile: naming conventions, type hints, docstring style
- KL divergence calculation with smoothing (ε=1e-10)
- Threshold gating: ε=0.15, only update if D_KL > ε

**Usage:**
```python
from src.phase5_style_engine import StyleGatedPrompt

style_gated = StyleGatedPrompt(divergence_threshold=0.15)
should_update, divergence = style_gated.should_update_style(
    local_entities, global_entities
)
```

### Phase 6: Inference Pipeline ✅

**Files:**
- `src/pipeline.py` - Main pipeline orchestration
- `run_pipeline.py` - CLI entry point

**Key Features:**
- Repository processing with caching
- Task processing from JSONL
- Automatic zip extraction
- Fallback strategies

**Usage:**
```python
from src.pipeline import CompletionPipeline

pipeline = CompletionPipeline(cache_dir="cache")
pipeline.process_jsonl(
    input_path="input.jsonl",
    output_path="predictions.jsonl"
)
```

## Project Structure

```
project/
├── src/
│   ├── __init__.py
│   ├── phase1_ast_parser.py      # AST parsing
│   ├── graph_store.py             # Graph database
│   ├── vector_store.py            # Embeddings + retrieval
│   ├── phase3_context_assembly.py # Context building
│   ├── phase5_style_engine.py     # Style analysis
│   └── pipeline.py                # Main pipeline
├── tests/
│   ├── test_phase1.py
│   ├── test_phase2.py
│   └── test_pipeline.py
├── baselines.py                   # BM25 baseline
├── run_pipeline.py                # CLI for our pipeline
├── example_submission.py          # Submission script
├── cache/                         # Parsed ASTs, embeddings
├── predictions/                   # Output files
├── docs/                          # Design docs
│   ├── phase1_design.md
│   ├── phase2_design.md
│   └── progress.md
├── AGENTS.md                      # This file
├── pyproject.toml                 # Dependencies
└── data/                          # Competition data
```

## Key Design Decisions

### 1. Why native `ast` instead of tree-sitter?
- Python-only for this competition
- Faster parsing, no C compiler needed
- Part of standard library
- Can extend to tree-sitter later if needed

### 2. Why in-memory graph instead of Neo4j?
- No external database setup
- Faster for batch processing
- Easy serialization with pickle
- NetworkX provides all needed graph algorithms

### 3. Why local embeddings instead of API calls?
- No API rate limits or costs
- sentence-transformers is fast on CPU
- Offline capability
- Deterministic results

### 4. Why hybrid retrieval?
- Dense: captures semantic similarity
- Sparse (BM25): captures exact identifier matches
- Combines strengths of both
- α parameter allows tuning

## Implementation Checklist

### Completed ✅
- [x] Phase 1: AST parsing with caching
- [x] Phase 2: Dense + sparse embeddings
- [x] Phase 3: Hybrid retrieval
- [x] Phase 4: Topological context assembly
- [x] Phase 5: KL-gated style engine
- [x] Phase 6: End-to-end pipeline
- [x] Baseline implementations (random, BM25)
- [x] Comprehensive tests

### Performance Targets
- Parse 1000 files: ~20 seconds (with cache)
- Single query: < 2 seconds
- Memory: < 2GB for 10k files

## Testing

```bash
# Run all tests
PYTHONPATH=/home/bn/Documents/Hackatony/Ensemble2026/task2 python tests/test_phase1.py
PYTHONPATH=/home/bn/Documents/Hackatony/Ensemble2026/task2 python tests/test_phase2.py
PYTHONPATH=/home/bn/Documents/Hackatony/Ensemble2026/task2 python tests/test_pipeline.py

# Run on small subset
python run_pipeline.py --stage practice --limit 5 --verbose
```

## Submission

1. Generate predictions:
```bash
python run_pipeline.py --stage practice
```

2. Update `example_submission.py`:
```python
JSONL_FILE = "predictions/python-practice-predictions.jsonl"
STAGE = "practice"  # or "public"
```

3. Submit:
```bash
python example_submission.py
```

## References

- Competition: JetBrains Challenge at EnsembleAI 2026
- Data: `data/drive-download-20260314T192239Z-1-001/`
- Baseline: BM25 ~0.7 chRF
- Target: > 0.75 chRF

## Notes for Future Improvements

1. **Better embeddings**: Fine-tune on code-specific corpus
2. **Cross-file dependencies**: Use import analysis for better retrieval
3. **Test file handling**: Special handling for test files (different style)
4. **Incremental updates**: Only re-parse changed files in large repos
5. **Parallel processing**: Multi-threading for repository processing
