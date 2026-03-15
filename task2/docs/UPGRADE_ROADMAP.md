# Upgrade Roadmap for Higher ChrF

## Priority 1: HIGH IMPACT (Do These First)

### 1.1 Import Resolution & Definition Lookup ⭐⭐⭐
**Current**: We extract imports but don't resolve them  
**Problem**: Missing definitions of symbols used in completion  
**Research**: STALL+ showed this **doubles EM** (exact match)  
**Implementation**:
```python
# When we see: from kombu import Connection
# Retrieve: def Connection(hostname, port, ...): ...
```
**Expected Gain**: +3-5% ChrF  
**Effort**: Medium

### 1.2 Query Reformulation with LLM ⭐⭐⭐
**Current**: Simple concatenation of identifiers from prefix/suffix  
**Winner**: NoMoreActimel (1st place) used Qwen2.5-Coder-1.5B for this  
**Implementation**:
```python
prompt = f"""
Given this code context, what is the developer trying to do?
Generate a search query to find relevant code.

Prefix: {prefix[-500:]}
Suffix: {suffix[:200]}

Query: """
```
**Expected Gain**: +2-4% ChrF  
**Effort**: Low (if already using LLM)

### 1.3 Adjacent Chunk Retrieval ⭐⭐
**Current**: Only retrieve matched entities  
**Winner**: WSPR_NCSU (3rd) retrieved adjacent chunks  
**Logic**: If we match a function, also get next/prev functions from same file  
**Implementation**: For each match, also retrieve entities at line_start±10  
**Expected Gain**: +2-3% ChrF  
**Effort**: Low

---

## Priority 2: MEDIUM IMPACT

### 2.1 Model-Specific Tokenizers ⭐⭐
**Current**: Rough estimate (4 chars/token)  
**Problem**: Mellum (8K), Codestral (16K+), Qwen (varies) have different tokenizers  
**Research**: NoMoreActimel counted tokens with actual Mellum tokenizer  
**Implementation**:
```python
# Load actual tokenizers for each model
mellum_tokenizer = AutoTokenizer.from_pretrained("jetbrains/mellum-4b")
codestral_tokenizer = AutoTokenizer.from_pretrained("mistralai/codestral")
```
**Expected Gain**: +1-2% ChrF (better budget utilization)  
**Effort**: Medium

### 2.2 Cross-Encoder Reranking ⭐⭐
**Current**: BM25 + Dense embeddings (bi-encoder)  
**Research**: Cross-encoders give better relevance scoring  
**Implementation**:
```python
# Initial retrieval: Top-50 with bi-encoder
# Rerank: Cross-encoder scores top-50 → select top-10
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```
**Expected Gain**: +1-3% ChrF  
**Effort**: Medium

### 2.3 Better Identifier Extraction ⭐
**Current**: Simple regex + Tree-sitter  
**Problem**: Missing context from type annotations, decorators  
**Upgrade**: Extract more signals:
- Type hints in signatures
- Exception types in except clauses
- Decorator names
- Class inheritance (bases)
**Expected Gain**: +1-2% ChrF  
**Effort**: Low

---

## Priority 3: NICE TO HAVE

### 3.1 Local Scope Trimming ⭐
**Current**: Use full prefix/suffix from input  
**Research**: REALISE Lab trimmed to enclosing function/block  
**Implementation**: Parse AST to find nearest function/class boundaries  
**Expected Gain**: +0.5-1% ChrF  
**Effort**: Medium

### 3.2 Cross-Repository Indexing ⭐
**Current**: Index per repository  
**Winner**: NoMoreActimel indexed cross-repository data  
**Logic**: Similar patterns across repos (e.g., all Django projects)  
**Expected Gain**: +0.5-1% ChrF  
**Effort**: High

### 3.3 Iterative Retrieval (RepoCoder style) ⭐
**Research**: RepoCoder showed +10% improvement  
**Logic**: 
1. Generate initial completion
2. Use completion as query for better retrieval
3. Regenerate with better context
**Problem**: Requires model access, not just context  
**Expected Gain**: +5-10% ChrF  
**Effort**: Very High (requires inference API)

---

## Quick Wins (Implement Today)

### 1. Import Resolution
```python
# In phase3_context_assembly.py

for import_stmt in current_file_imports:
    if import_stmt['type'] == 'from':
        module = import_stmt['module']
        for name in import_stmt['names']:
            # Look up definition in graph
            definition = graph.get_definition(module, name['name'])
            if definition:
                context_items.append(definition)
```

### 2. Adjacent Chunks
```python
# After retrieving seed_entity, also get:
adjacent_entities = graph.get_entities_in_range(
    file_path=seed_entity.file_path,
    line_range=(seed_entity.line_start - 20, seed_entity.line_end + 20)
)
```

### 3. Better Query
```python
# Use LLM to reformulate query if available
if self.llm_abstractor:
    reformulated = self.llm_abstractor.reformulate_query(prefix, suffix)
    if reformulated:
        query = reformulated
```

---

## Estimated Total Impact

| Upgrade | ChrF Gain | Effort | Priority |
|---------|-----------|--------|----------|
| Import Resolution | +3-5% | Medium | ⭐⭐⭐ |
| Query Reformulation | +2-4% | Low | ⭐⭐⭐ |
| Adjacent Chunks | +2-3% | Low | ⭐⭐ |
| Model Tokenizers | +1-2% | Medium | ⭐⭐ |
| Cross-Encoder | +1-3% | Medium | ⭐⭐ |
| Better Identifiers | +1-2% | Low | ⭐ |
| **TOTAL** | **+10-19%** | | |

**Conservative estimate**: Current ~0.75 → **0.80-0.85** with top 3 upgrades

---

## Implementation Order

1. **Today**: Import resolution (biggest bang for buck)
2. **Tomorrow**: Query reformulation (if using LLM)
3. **This week**: Adjacent chunks + better identifiers
4. **Next week**: Cross-encoder + model tokenizers

---

## Cost-Benefit Analysis

### Free Upgrades (No API cost)
- Import resolution: ⭐⭐⭐ HIGH impact
- Adjacent chunks: ⭐⭐ MEDIUM impact  
- Better identifiers: ⭐ LOW impact

### With LLM API
- Query reformulation: ⭐⭐⭐ HIGH impact
- Better abstracts (already done): ⭐⭐ MEDIUM impact

### Hardware/Time Cost
- Cross-encoder: ⭐⭐ MEDIUM impact (slower)
- Model tokenizers: ⭐⭐ MEDIUM impact (download models)

**Recommendation**: Start with free upgrades, then add LLM features.
