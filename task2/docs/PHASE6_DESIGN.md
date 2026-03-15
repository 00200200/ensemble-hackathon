# Phase 6 Design: Inference Pipeline

## Goal
Execute the edit and refine the system based on performance.

## Design Decisions

### Pipeline Flow

1. **Repository Processing**
   - Extract zip file
   - Parse all Python files (with caching)
   - Build graph and index

2. **Task Processing**
   - For each task in JSONL:
     - Build query from prefix/suffix
     - Retrieve relevant context
     - Assemble formatted context
     - Write to output

3. **Caching Strategy**
   - Cache parsed ASTs by file hash
   - Cache graph by repository revision
   - Cache embeddings

### Output Format

```json
{"context": "<|file_sep|>\n# File: utils.py\ndef helper():\n    pass"}
```

## Implementation Notes

### Performance Targets

- Parse 1000 files: ~20 seconds (with cache)
- Single query: < 2 seconds
- Memory: < 2GB for 10k files

### Error Handling

- Skip files with syntax errors
- Continue on individual task failures
- Log warnings for debugging

### Fallback Strategies

1. If graph not available: Use BM25 only
2. If embeddings fail: Use simple embedder
3. If cache corrupted: Re-parse
