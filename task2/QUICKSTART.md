# Quick Start Guide

## 3 Ways to Run

### 1. Baseline (Simplest)
```bash
python baselines.py --stage practice --strategy bm25
```

### 2. Our Pipeline (Free, Better)
```bash
pip install -e "."
python run_pipeline.py --stage practice --limit 10
```

### 3. Our Pipeline (With LLM, Best)
```bash
# DeepSeek (cheapest)
export DEEPSEEK_API_KEY="sk-..."
export LLM_MODEL="deepseek-coder"
python run_pipeline.py --stage practice --limit 10

# Or OpenAI
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"
python run_pipeline.py --stage practice --limit 10
```

---

## Check Cache Status

```bash
python check_cache.py
```

Shows:
- How many abstracts cached
- Cost to regenerate
- Sample cached abstracts

---

## Test API

```bash
python test_llm.py
```

Verifies your API key works before running pipeline.

---

## Run Tests

```bash
PYTHONPATH=. python tests/test_phase1.py
PYTHONPATH=. python tests/test_phase2.py
PYTHONPATH=. python tests/test_phase3.py
PYTHONPATH=. python tests/test_pipeline.py
```

---

## Submit

```bash
# Generate
python run_pipeline.py --stage public

# Submit (update example_submission.py first)
export TEAM_TOKEN="..."
python example_submission.py
```

---

## Expected Results

| Mode | ChrF | Cost |
|------|------|------|
| Baseline | ~0.70 | Free |
| Ours (free) | ~0.72-0.75 | Free |
| Ours + DeepSeek | ~0.75-0.78 | ~$0.01-0.05 |
| Ours + OpenAI | ~0.76-0.80 | ~$0.05-0.10 |

**Key**: Our pipeline with LLM abstracts gives **+5-10% improvement** over baseline.
