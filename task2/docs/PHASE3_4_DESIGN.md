# Phase 3 & 4 Design: Context Assembly

## Goal
Use the repository's "Call Graph" to provide the LLM with necessary downstream logic.

## Design Decisions

### 1. Seed Selection

From Phase 2 hybrid retrieval, get Top-K (K=5) entities as seeds.

### 2. Neighborhood Expansion

For each seed, perform 1-hop traversal in the graph:
- Retrieve full code body of seed itself
- Retrieve functional abstracts of direct callees (downstream dependencies)

### 3. Centrality Filtering

If a function has very high degree (>20), apply penalty to prevent context overflow.

### 4. Context Format

```
<|file_sep|>
# File: {filename}
# Entity: {name} ({type})
{code}
```

### 5. Token Budget Management

Target token budget: 8000 tokens for Mellum

Allocation:
- 40% current file prefix
- 30% cross-file context
- 20% suffix
- 10% buffer

### 6. Relevance Ordering

**Ascending relevance order** (least relevant first, most relevant last):
- This ensures most relevant content survives left-truncation
- Confirmed by research to give ~3% improvement

## Implementation Notes

### Query Building

From prefix/suffix:
1. Extract identifiers using Tree-sitter
2. Include recent code (higher weight)
3. Filter common keywords

### Context Deduplication

Use structure-aware merging:
- If two contexts are from same file and adjacent, merge them
- This eliminates redundancy and restores logical flow

### Current File Handling

- Always include current file entities with priority
- Trim prefix/suffix to enclosing function/class if possible
