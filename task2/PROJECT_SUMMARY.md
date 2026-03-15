# Project Summary: JetBrains Code Completion Pipeline

## Original Design Goals

The core idea was to build a context collection system that:
1. **Parses repository structure** using AST to extract entities (functions, classes)
2. **Builds a code graph** with relationships (DEFINES, CONTAINS, CALLS, IMPORTS)
3. **Retrieves relevant context** using hybrid BM25 + embedding search
4. **Assembles context** prioritizing the most relevant code for LLM completion

## What Was Actually Implemented

### Phase 1: AST Parsing (`src/phase1_ast_parser.py`)
- ✅ Native Python AST parser with tree-sitter fallback
- ✅ Entity extraction (functions, classes, methods)
- ✅ Relative paths for cache portability
- ❌ **ISSUE**: Python 2 syntax files fail to parse (empty entities)

### Phase 2: Graph Store (`src/graph_store.py`)
- ✅ NetworkX-based graph with entity relationships
- ✅ File-to-entity and name-to-entity indexing
- ✅ Import resolution for cross-file dependencies

### Phase 3: Vector Store (`src/vector_store.py`)
- ✅ BM25Okapi for sparse retrieval
- ✅ FAISS for dense embeddings
- ✅ Reciprocal Rank Fusion for hybrid scoring
- ❌ **ISSUE**: Simple embedder fallback lacks code understanding

### Phase 4: Context Assembly (`src/phase3_context_assembly.py`)
- ✅ Multi-source context assembly
- ✅ Token budget management
- ❌ **MAJOR ISSUE**: Initial implementation used USELESS ABSTRACTS instead of full code

## Critical Issues Found

### Issue 1: Path Mismatch Bug (FIXED)
**Problem**: Graph stored absolute paths with temp directories, but task lookup used relative paths.
```
Stored:   /tmp/tmpXXX/celery__kombu-.../t/file.py
Lookup:   t/integration/test_redis.py
Result:   0 entities found for current file
```
**Fix**: Store relative paths in entities.

### Issue 2: Missing Current File Context (FIXED)
**Problem**: Current file entities were not being included in context.
```
Task: Editing _pbs.py
Context: _relaxator.py, _input.py, _population.py (WRONG!)
```
**Fix**: Always include current file entities with highest priority.

### Issue 3: Useless Abstracts (FIXED)
**Problem**: Context showed function signatures + LLM-generated summaries instead of actual code.
```python
# BEFORE (Useless):
# method: __init__
def __init__(self, target_forces)
Summary: Initializes an instance by storing the provided target_forces...

# AFTER (Useful):
def __init__(self, target_forces):
    self.target_forces = target_forces
    self.forces = None
    ...
```
**Fix**: Use FULL CODE instead of abstracts.

### Issue 4: Missing Modified Files Context (FIXED)
**Problem**: The `modified` field from task data (files changed in same commit) was ignored.
**Impact**: LLM couldn't see related code changes.
**Fix**: Added modified files as high-priority context source.

### Issue 5: Token Budget Imbalance (FIXED)
**Problem**: Full code for large files consumed entire budget, crowding out other files.
```
optimize.py: 7,380 tokens
Modified file budget: 2,000 tokens
Result: Only optimize.py in context, data/__init__.py excluded
```
**Fix**: Use ABSTRACTS for modified files to fit ALL of them.

## Current Architecture

### Token Budget Allocation
- **35%** - Current file entities (FULL CODE)
- **40%** - Modified files from same commit (ABSTRACTS - to fit all)
- **15%** - Import definitions (ABSTRACTS)
- **8%** - Retrieved code from hybrid search (ABSTRACTS)
- **2%** - Buffer

### Context Priority Order
1. Current file (highest priority)
2. Modified files (co-changed in commit)
3. Import definitions
4. Retrieved seeds

## The "Slop" Problem

The bad context examples (empty test functions, random utilities) came from:
1. **Retrieved seeds** using BM25 without filtering
2. **No relevance filtering** - any file could be included
3. **Centrality penalty** wasn't working effectively

Example bad context:
```python
<|file_sep|>
# File: test_voxel.py
# function: empty_remove_invalid
def empty_remove_invalid()  # USELESS!

# function: empty_shift_change
def empty_shift_change()  # USELESS!
```

## Current Status

### Working Features
- ✅ Repository extraction and caching
- ✅ AST parsing (for Python 3 files)
- ✅ Graph construction with relationships
- ✅ Hybrid retrieval (BM25 + FAISS)
- ✅ Modified files inclusion
- ✅ Current file priority
- ✅ Full code for current file

### Remaining Issues
- ⚠️ Python 2 files parse with 0 entities (legacy repos)
- ⚠️ Simple embedder lacks code semantic understanding
- ⚠️ Retrieved seeds can still include low-relevance files
- ⚠️ No special handling for test files vs source files

### Recommended Improvements
1. **Better retrieval filtering**: Exclude test utilities, empty functions
2. **Code-aware embeddings**: Use code-specific embedder (codebert, etc.)
3. **Python 2 support**: Use tree-sitter with Python 2 grammar
4. **File type weighting**: Prioritize source files over test files
5. **Call graph traversal**: Follow actual function call chains

## Usage

```bash
# Run on dataset with specific task IDs
python run_pipeline.py --stage dataset --filter-ids "id1,id2,id3"

# Resume interrupted run
python run_pipeline.py --stage dataset --resume

# Run with custom input
python run_pipeline.py --input custom.jsonl --output output.jsonl
```

## Key Files
- `src/pipeline.py` - Main pipeline orchestration
- `src/phase3_context_assembly.py` - Context assembly logic
- `src/graph_store.py` - Code graph storage
- `src/vector_store.py` - Hybrid retrieval
- `run_pipeline.py` - CLI entry point
