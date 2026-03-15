# Phase 2 Design: Vector Store & Hybrid Retrieval

## Goal
Implement embeddings and hybrid retrieval (BM25 + Dense) for code entities.

## Design Decisions

### 1. Dense Embeddings

**Primary**: sentence-transformers/all-MiniLM-L6-v2
- 384 dimensions
- Fast on CPU
- Good for code snippets

**Fallback**: Simple keyword-based embedder if sentence-transformers unavailable

### 2. Sparse Retrieval (BM25)

**Primary**: rank_bm25.BM25Okapi
- Industry standard
- Good for exact identifier matching

**Fallback**: Simple BM25 implementation

### 3. Vector Store

**Primary**: FAISS (Facebook AI Similarity Search)
- Fast nearest neighbor search
- Efficient for large datasets

**Fallback**: numpy + cosine similarity

### 4. Hybrid Scoring

Formula: `S(q,d) = α·Dense(q,d) + (1-α)·BM25(q,d)`

Initial α = 0.5 (balanced)

For identifier-heavy queries: α < 0.3
For conceptual queries: α > 0.7

## Implementation Notes

### Text Preparation for Embedding

For each entity, create embedding text:
```
Name: {name}
Type: {type}
Signature: {signature}
Docstring: {docstring}
Code Preview: {first 10 lines of raw_code}
```

### BM25 Tokenization

- Lowercase
- Split on non-alphanumeric
- Include identifier names (camelCase/Pascal_case splits)

### Query Construction

From prefix/suffix:
1. Extract identifiers using Tree-sitter
2. Combine with raw text
3. Weight recent code higher

## Testing Strategy

Unit tests for:
1. Embedding generation
2. BM25 scoring
3. Hybrid retrieval ranking
4. FAISS index operations
5. Query building from prefix/suffix
