#!/usr/bin/env python
"""
Test LLM abstraction setup.
Usage:
    # Test OpenAI
    export OPENAI_API_KEY="sk-..."
    python test_llm.py

    # Test DeepSeek
    export DEEPSEEK_API_KEY="sk-..."
    export LLM_MODEL="deepseek-chat"
    python test_llm.py

    # Or explicitly
    python test_llm.py --model deepseek-coder --api-key "sk-..." --base-url https://api.deepseek.com
"""

import argparse
import os
from src.llm_abstraction import CachedLLMAbstractor
from src.phase1_ast_parser import Entity


def main():
    parser = argparse.ArgumentParser(description="Test LLM abstraction")
    parser.add_argument("--model", default=None, help="Model name (e.g., gpt-4o-mini, deepseek-chat)")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--base-url", default=None, help="Base URL (e.g., https://api.deepseek.com)")
    args = parser.parse_args()
    
    # Get from env vars or args
    model = args.model or os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL", "gpt-4o-mini")
    api_key = args.api_key or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
    
    # Auto-detect DeepSeek
    if not base_url and "deepseek" in model.lower():
        base_url = "https://api.deepseek.com"
    
    print(f"Testing LLM abstraction with:")
    print(f"  Model: {model}")
    print(f"  Base URL: {base_url or 'default (OpenAI)'}")
    print(f"  API Key: {'*' * 10}..." if api_key else "  API Key: NOT SET")
    print()
    
    if not api_key:
        print("ERROR: No API key found. Set OPENAI_API_KEY or DEEPSEEK_API_KEY env var.")
        return 1
    
    # Create abstractor
    abstractor = CachedLLMAbstractor(
        model=model,
        api_key=api_key,
        base_url=base_url
    )
    
    # Test entity
    test_entity = Entity(
        name='calculate_total',
        type='function',
        file_path='/test.py',
        line_start=1,
        line_end=10,
        signature='def calculate_total(items: List[Item]) -> Decimal',
        raw_code='''def calculate_total(items: List[Item]) -> Decimal:
    """Calculate total price with tax."""
    subtotal = sum(item.price for item in items)
    tax = subtotal * 0.08
    return subtotal + tax''',
        docstring='Calculate total price with tax.',
        dependencies=['sum', 'Item']
    )
    
    print("Generating abstract for test function...")
    print("-" * 60)
    
    try:
        abstract = abstractor.get_abstract(test_entity)
        print(f"\nGenerated abstract:\n{abstract}\n")
        print("-" * 60)
        print("SUCCESS! LLM abstraction is working.")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
