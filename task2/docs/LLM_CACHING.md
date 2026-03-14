# LLM Abstraction Caching - Cost Protection

## How It Works

### Cache Key = Code Hash
Each abstract is cached using the **MD5 hash of the function's code**:

```python
code_hash = hashlib.md5(code.encode()).hexdigest()[:16]
cache_key = f"abstract_{code_hash}.json"
```

This means:
- ✅ Same function in different repos → **Same cache, no extra cost**
- ✅ Same repo, multiple runs → **Cache hit, no API call**
- ✅ Slightly different code → **New cache entry** (correct behavior)

### Cache Location
```
.cache/
├── abstract_a1b2c3d4e5f6.json
├── abstract_b2c3d4e5f6a7.json
└── abstract_c3d4e5f6a7b8.json
```

Each file contains:
```json
{
  "abstract": "Summary: Retrieves user by ID...",
  "code_hash": "a1b2c3d4e5f6..."
}
```

## Cost Protection

### Scenario 1: First Run (No Cache)
```
Repository: celery/kombu
Entities: 1000
Cached: 0
API Calls: 1000
Estimated Cost (DeepSeek): ~$0.10
```

### Scenario 2: Second Run (Full Cache)
```
Repository: celery/kombu
Entities: 1000
Cached: 1000
API Calls: 0
Estimated Cost: $0.00
```

### Scenario 3: Different Repo with Shared Functions
```
Repository: another/celery-project
Entities: 500
Cached (from repo 1): 200 (shared utility functions)
API Calls: 300 (new functions only)
Estimated Cost (DeepSeek): ~$0.03 (instead of $0.05)
```

## Checking Cache Status

```bash
# See how many abstracts are cached
python check_cache.py

# Output:
# ============================================================
# LLM Abstract Cache Status
# ============================================================
# 
# Cached abstracts: 47
# 
# Cost Estimate (if regenerating all):
# DeepSeek:       $0.0047
# OpenAI 4o-mini: $0.0094
# OpenAI 4o:      $0.0940
```

## Cache Persistence

The cache is stored in `.cache/` directory and **persists across runs**:
- Not committed to git (add to .gitignore)
- Survives pipeline restarts
- Can be backed up/shared between team members

## Clear Cache

If you need to regenerate abstracts (e.g., improved prompt):

```bash
rm -rf .cache/abstract_*.json
```

Or regenerate specific entities by modifying the code slightly.

## Monitoring API Usage

When running with `verbose` flag:
```bash
export DEEPSEEK_API_KEY="sk-..."
python run_pipeline.py --stage practice --limit 10 --verbose
```

You'll see:
```
[INFO] [CACHE] 150 entities already cached
[INFO] [API] 50 entities need LLM generation
[INFO] [COST] Estimated cost: $0.0050
[INFO] Progress: 10/50 API calls made...
[INFO] Progress: 20/50 API calls made...
[INFO] [DONE] Total API calls: 50
```

## Fallback (No API Key)

If no API key is set, the system uses **free fallback**:
```python
def get_user(id: int) -> User
"""Get user by ID."""
# Uses: validate_id, User.query.get
```

No API calls, no cost.

## Best Practices

1. **Set cache directory**: Use shared cache for team
   ```bash
   export CACHE_DIR=/shared/llm-cache
   ```

2. **Monitor costs**: Check cache before large runs
   ```bash
   python check_cache.py
   ```

3. **Use cheaper provider**: DeepSeek is 5-10x cheaper
   ```bash
   export DEEPSEEK_API_KEY="..."
   export LLM_MODEL="deepseek-coder"
   ```

4. **Batch generation**: Pre-generate abstracts for repos you'll use multiple times

## Summary

- ✅ Cache is **automatic** - no config needed
- ✅ Cache key is **content-based** - deduplicates identical functions
- ✅ Cache **persists** across runs
- ✅ **Clear logging** shows cache hits/misses
- ✅ **Cost estimates** before API calls
- ✅ **Zero cost** for cached abstracts
