#!/usr/bin/env python
"""
Validate that pipeline processes correct repositories.
Checks:
1. Repo zips exist
2. Target files exist in extracted repos
3. Context is generated for correct files

Usage:
    python validate_repos.py --stage practice --limit 10
"""

import argparse
import json
from pathlib import Path
from src.pipeline import CompletionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='practice')
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--data-dir', default='data/drive-download-20260314T192239Z-1-001')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    input_file = data_dir / f"python-{args.stage}.jsonl"
    
    print(f"\n{'='*70}")
    print(f"Repository Validation")
    print(f"{'='*70}\n")
    
    # Read tasks
    tasks = []
    with open(input_file) as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            tasks.append(json.loads(line))
    
    print(f"Validating {len(tasks)} tasks from {input_file}\n")
    
    # Create pipeline
    pipeline = CompletionPipeline(data_dir=data_dir)
    
    errors = []
    success = 0
    
    for i, task in enumerate(tasks):
        task_id = task['id']
        repo = task['repo']
        revision = task['revision']
        target_path = task['path']
        
        print(f"Task {i+1}/{len(tasks)}: {task_id}")
        print(f"  Repo: {repo}@{revision[:8]}")
        print(f"  Target: {target_path}")
        
        # Check repo exists
        repo_name = repo.replace('/', '__')
        zip_name = f"{repo_name}-{revision}.zip"
        zip_path = data_dir / f"python-{args.stage}" / zip_name
        
        if not zip_path.exists():
            print(f"  ❌ ERROR: Zip not found: {zip_path}")
            errors.append(f"Task {task_id}: Zip not found - {zip_name}")
            continue
        
        # Get repo path (extracts if needed)
        repo_path = pipeline.get_repo_path(task)
        
        if not repo_path:
            print(f"  ❌ ERROR: Failed to get repo path")
            errors.append(f"Task {task_id}: Failed to get repo path")
            continue
        
        # Check target file exists
        target_file = repo_path / target_path
        if not target_file.exists():
            print(f"  ❌ ERROR: Target file not found: {target_file}")
            # List available files
            py_files = list(repo_path.rglob('*.py'))
            print(f"     Available .py files: {len(py_files)}")
            print(f"     Sample: {[str(f.relative_to(repo_path)) for f in py_files[:3]]}")
            errors.append(f"Task {task_id}: Target file not found - {target_path}")
            continue
        
        print(f"  ✓ Repo extracted: {repo_path}")
        print(f"  ✓ Target exists: {target_file}")
        
        # Try to generate context
        try:
            context = pipeline.process_task(task)
            if context:
                print(f"  ✓ Context generated: {len(context)} chars")
                success += 1
            else:
                print(f"  ⚠ Warning: Empty context")
                errors.append(f"Task {task_id}: Empty context")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            errors.append(f"Task {task_id}: Exception - {e}")
        
        print()
    
    # Summary
    print(f"{'='*70}")
    print(f"Validation Summary")
    print(f"{'='*70}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successful:  {success}")
    print(f"Errors:      {len(errors)}")
    
    if errors:
        print(f"\nErrors:")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print(f"\n✓ All validations passed!")
        return 0


if __name__ == '__main__':
    exit(main())
