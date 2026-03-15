# Research Notes: Winning Strategies for Code Completion

## Date: 2026-03-14
## Competition: JetBrains Context Collection Challenge at EnsembleAI 2026

---

## Key Insight

**Symbol-aware retrieval at chunk granularity, ordered by ascending relevance to survive left-truncation, outperforms sophisticated embedding-based approaches.**

The ASE 2025 competition proved that GPU-free keyword search with good query construction beats dense retrieval.

---

## Competition Winners Analysis

### SpareCodeComplete (1st Kotlin, 2nd Python, 0.725 ChrF)
- **Strategy**: Zoekt keyword search only - NO embeddings
- **Pipeline**:
  1. Build Zoekt index from repository
  2. Extract identifiers via Tree-sitter AST parsing
  3. Construct Zoekt queries from extracted symbols
  4. Apply gradual query relaxation (exact → regex → OR)
  5. Concatenate top results
- **Key Insight**: GPU-free, keyword-only approach can achieve top results

### NoMoreActimel (1st Python)
- **Strategy**: Query reformulation with Qwen2.5-Coder-1.5B
- Reformulate queries from prefix/suffix
- Embedding-based retrieval with heuristic boosting
- Count tokens using actual Mellum tokenizer
- Index cross-repository/cross-revision data

### WSPR_NCSU (3rd)
- **Strategy**: Hybrid BM25 + FAISS with Reciprocal Rank Fusion
- BM25 weight: 0.2, FAISS weight: 0.8
- **Key Innovation**: Relative positioning - retrieve adjacent chunks to captured results

### REALISE Lab Findings
- **Chunk-level retrieval beats file-level by 6%**
- **Reversing BM25-ranked file order** (ascending relevance) yields ~3% improvement
- Left-truncation preserves most relevant content placed last
- Local-scope trimming to nearest enclosing block helps

---

## Critical Implementation Insights

### 1. Left-Truncation Strategy
- Context is trimmed from LEFT when exceeding window
- **Place most relevant context CLOSEST to cursor** (at the end)
- Ascending relevance order: least relevant first, most relevant last
- This alone yields ~3% improvement

### 2. Chunk Granularity
- Function-level or logical-block-level chunks outperform file-level by 6%+
- Use AST-aware chunking (cAST method)
- Sweet spot: 10-20 lines or one function/class method

### 3. Static Analysis > Similarity
- Import resolution and definition retrieval is highest-value context
- STALL+ showed file-level dependency extraction roughly doubles EM
- Definitions of symbols in target completion = highest leverage

### 4. FIM Format Differences
- **Mellum**: SPM (Suffix-Prefix-Middle)
- **Codestral & Qwen2.5**: PSM (Prefix-Suffix-Middle)
- Must handle both formats for competition

### 5. Context Budget Allocation (Mellum 8K)
- 40% current-file prefix
- 30% cross-file context
- 20% suffix
- 10% buffer for special tokens

### 6. ChrF Metric Insights
- β=2 (recall weighted twice as heavily as precision)
- Missing characters hurts more than extra characters
- Sensitive to exact variable names and function names
- Character n-grams of order 1-6

---

## Architecture Decisions

### Why NOT Neo4j/Qdrant?
- External databases add complexity and latency
- NetworkX + in-memory FAISS/numpy is faster for batch processing
- No setup required - works offline
- Can persist with pickle

### Why Tree-sitter over Python ast?
- Handles incomplete/broken code (critical at cursor positions)
- Supports 100+ languages (future-proof)
- Incremental parsing support
- S-expression queries for efficient extraction

### Why Hybrid Retrieval (BM25 + Dense)?
- BM25: exact identifier matching
- Dense: semantic similarity
- α=0.5 initial balance
- Reciprocal Rank Fusion for combining

---

## Implementation Checklist from Research

- [x] Use Tree-sitter for AST parsing (handles broken code)
- [x] Chunk-level retrieval, not file-level
- [x] Import resolution and definition retrieval (static analysis)
- [x] Hybrid BM25 + embeddings with RRF
- [x] Gradual query relaxation
- [x] Bidirectional retrieval (prefix + suffix)
- [x] Ascending relevance ordering for left-truncation
- [x] Token counting with actual model tokenizers
- [ ] Top-5 chunks is optimal (more introduces noise)
- [ ] Deduplication with structure-aware merging

---

## References

1. STALL+ - Static analysis + retrieval combination
2. RepoCoder - Iterative retrieval-generation
3. AlignCoder - Query enhancement with candidates
4. RepoHyper - Graph-based retrieval
5. CoCoMIC - CCFinder static analysis tool
6. Repoformer - Only ~20% of retrievals help
