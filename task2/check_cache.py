#!/usr/bin/env python
"""
Check LLM abstract cache status.
Shows how many abstracts are cached and estimates costs for remaining.

Usage:
    python check_cache.py
"""

import json
from pathlib import Path
import hashlib


def main():
    cache_dir = Path(".cache")
    
    if not cache_dir.exists():
        print("No cache directory found.")
        return
    
    cache_files = list(cache_dir.glob("abstract_*.json"))
    
    print(f"\n{'='*60}")
    print(f"LLM Abstract Cache Status")
    print(f"{'='*60}\n")
    
    print(f"Cached abstracts: {len(cache_files)}")
    
    if len(cache_files) == 0:
        print("\nNo cached abstracts found.")
        print("Run the pipeline with LLM_API_KEY set to generate abstracts.")
        return
    
    # Show some examples
    print(f"\nSample cached abstracts:")
    print("-" * 60)
    
    for i, cache_file in enumerate(cache_files[:3]):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                abstract = data.get('abstract', '')
                # Show first 100 chars
                preview = abstract.replace('\n', ' ')[:100]
                print(f"{i+1}. {preview}...")
        except Exception as e:
            print(f"{i+1}. Error reading: {e}")
    
    if len(cache_files) > 3:
        print(f"... and {len(cache_files) - 3} more")
    
    # Cost estimate
    print(f"\n{'='*60}")
    print(f"Cost Estimate (if regenerating all)")
    print(f"{'='*60}")
    
    # Rough estimates
    deepseek_cost = len(cache_files) * 0.0001
    openai_mini_cost = len(cache_files) * 0.0002
    openai_4o_cost = len(cache_files) * 0.002
    
    print(f"DeepSeek:       ${deepseek_cost:.4f}")
    print(f"OpenAI 4o-mini: ${openai_mini_cost:.4f}")
    print(f"OpenAI 4o:      ${openai_4o_cost:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Cache location: {cache_dir.absolute()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
