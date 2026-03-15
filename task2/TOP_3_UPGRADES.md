# Top 3 Upgrades for Higher ChrF

## 🥇 #1: Import Resolution (+3-5% ChrF) - IMPLEMENT FIRST

### What it does
When you see `from .common import BasicFunctionality`, fetch the actual `BasicFunctionality` class definition and include it in context.

### Why it helps
The prefix shows `BasicFunctionality` is being used (inherited from), but the model doesn't know what it is. Providing the definition helps complete the code correctly.

### Research backing
STALL+ paper showed this **doubles exact match (EM)** score.

### Implementation
Already created: `src/import_resolver.py`

**To integrate** (add to `src/phase3_context_assembly.py`):
```python
from src.import_resolver import ImportResolver

# In assemble() method:
resolver = ImportResolver(self.graph)
import_definitions = resolver.resolve_symbols_from_code(
    prefix, current_file, max_definitions=5
)

# Add to context with HIGH priority
for entity in import_definitions:
    context_items.append(ContextItem(
        entity=entity,
        source='import_definition',  # Mark as import
        relevance_score=0.95,  # Very high priority
        ...
    ))
```

### Expected cost
- **Free** (no API calls)
- Uses existing graph

---

## 🥈 #2: Query Reformulation with LLM (+2-4% ChrF)

### What it does
Instead of using raw prefix/suffix text as search query, use an LLM to generate a better query.

### Example
```python
# Current (bad):
query = "def test_publish_consume(self, connection): test_queue = kombu.Queue"

# With LLM (better):
query = "Redis priority queue testing, message publishing with priority levels"
```

### Why it helps
The raw code contains syntax noise (parentheses, colons). A semantic summary finds better matches.

### Implementation
Add to `src/llm_abstraction.py`:
```python
def reformulate_query(self, prefix: str, suffix: str) -> str:
    """Generate search query from code context."""
    prompt = f"""Given this code context, generate a search query to find relevant code.

Prefix (code before cursor):
{prefix[-500:]}

Suffix (code after cursor):
{suffix[:200]}

Generate a 1-sentence search query describing what the developer is trying to do.
Query: """
    
    response = self._client.chat.completions.create(
        model=self.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=50
    )
    
    return response.choices[0].message.content.strip()
```

### Expected cost
- **~$0.001 per task** (very cheap)
- Only 1 LLM call per completion task

---

## 🥉 #3: Adjacent Chunk Retrieval (+2-3% ChrF)

### What it does
When you retrieve a function, also get the functions that come before and after it in the same file.

### Example
```python
# You retrieve:
def process_user(user_id): ...

# Also get (adjacent):
def validate_user(user_id): ...  # Previous function
def save_user(user): ...          # Next function
```

### Why it helps
Related functions are often grouped together in files. Adjacent functions likely work together.

### Implementation
Add to `src/graph_store.py`:
```python
def get_adjacent_entities(self, file_path: str, line: int, window: int = 2) -> List[Any]:
    """Get entities adjacent to given line in file."""
    file_entities = self.get_entities_by_file(file_path)
    
    # Sort by line number
    sorted_entities = sorted(
        file_entities,
        key=lambda e: getattr(e, 'line_start', 0)
    )
    
    # Find index of closest entity
    closest_idx = 0
    min_distance = float('inf')
    for i, entity in enumerate(sorted_entities):
        dist = abs(getattr(entity, 'line_start', 0) - line)
        if dist < min_distance:
            min_distance = dist
            closest_idx = i
    
    # Get window around it
    start = max(0, closest_idx - window)
    end = min(len(sorted_entities), closest_idx + window + 1)
    
    return sorted_entities[start:end]
```

Then in context assembly, after retrieving seeds:
```python
for seed_entity, score in seeds:
    # Get adjacent
    adjacent = graph.get_adjacent_entities(
        seed_entity.file_path,
        seed_entity.line_start,
        window=1
    )
    for adj in adjacent:
        # Add with slightly lower score
        context_items.append(...)
```

### Expected cost
- **Free** (uses existing graph)

---

## Implementation Priority

| Rank | Upgrade | ChrF Gain | Effort | Cost | Priority |
|------|---------|-----------|--------|------|----------|
| 1 | Import Resolution | +3-5% | Low | Free | ⭐⭐⭐⭐⭐ |
| 2 | Adjacent Chunks | +2-3% | Low | Free | ⭐⭐⭐⭐ |
| 3 | Query Reformulation | +2-4% | Low | $0.001/task | ⭐⭐⭐ |

**Total potential**: +7-12% ChrF improvement

---

## Quick Implementation Plan

### Today (30 minutes)
1. Add `src/import_resolver.py` (already done ✓)
2. Import and use in `phase3_context_assembly.py`
3. Test on 10 tasks

### Tomorrow (1 hour)
1. Add adjacent chunk retrieval to `graph_store.py`
2. Add `get_adjacent_entities()` method
3. Use in context assembly
4. Test

### This week (2 hours)
1. Add query reformulation to `llm_abstraction.py`
2. Cache reformulated queries (same code = same query)
3. Test with and without
4. Compare ChrF scores

---

## Expected Final Score

| Configuration | ChrF |
|---------------|------|
| Current | ~0.75 |
| + Import Resolution | ~0.78 |
| + Adjacent Chunks | ~0.80 |
| + Query Reformulation | ~0.82-0.84 |

**Target achieved**: Beating baseline (0.70) by +12-14%!
