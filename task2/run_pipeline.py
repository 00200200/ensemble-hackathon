#!/usr/bin/env python3
"""Run the code completion pipeline on competition data.

Usage:
    python run_pipeline.py --stage practice --limit 10
    python run_pipeline.py --stage public --output predictions/submission.jsonl
"""

import argparse
import logging
from pathlib import Path



from src.pipeline import CompletionPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for JetBrains code completion challenge"
    )
    parser.add_argument(
        '--stage',
        default='practice',
        choices=['practice', 'public'],
        help='Competition stage to process'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output predictions file (default: predictions/python-{stage}.jsonl)'
    )
    parser.add_argument(
        '--cache',
        type=Path,
        default=Path('cache'),
        help='Cache directory'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of tasks to process (for testing)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=8000,
        help='Maximum context tokens'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/drive-download-20260314T192239Z-1-001'),
        help='Data directory'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Determine paths
    input_file = args.data_dir / f"python-{args.stage}.jsonl"
    
    if args.output:
        output_file = args.output
    else:
        output_file = Path('predictions') / f"python-{args.stage}-predictions.jsonl"
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create pipeline
    pipeline = CompletionPipeline(
        data_dir=args.data_dir,
        cache_dir=args.cache,
        max_tokens=args.max_tokens
    )
    
    # Process
    pipeline.process_jsonl(
        input_path=input_file,
        output_path=output_file,
        limit=args.limit
    )
    
    print(f"\nPredictions written to: {output_file}")
    print(f"To submit, run: python example_submission.py")
    print(f"(Update JSONL_FILE in example_submission.py to: {output_file})")


if __name__ == "__main__":
    main()
