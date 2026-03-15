"""Baseline strategies for code completion context retrieval.

This module provides baseline implementations:
1. Random file selection
2. BM25-based file selection
"""

import argparse
import json
import logging
import random
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple
import tempfile

logger = logging.getLogger(__name__)


def extract_repo(zip_path: Path, extract_to: Path) -> Path:
    """Extract repository zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    
    # Find repo root
    subdirs = [d for d in extract_to.iterdir() if d.is_dir()]
    return subdirs[0] if subdirs else extract_to


def get_python_files(repo_path: Path) -> List[Path]:
    """Get all Python files in repository."""
    skip_patterns = {'__pycache__', '.git', 'venv', '.venv', 'node_modules'}
    files = []
    for f in repo_path.rglob('*.py'):
        if not any(p in skip_patterns for p in f.parts):
            files.append(f)
    return files


def random_strategy(
    task: Dict,
    data_dir: Path,
    max_chars: int = 8000 * 4
) -> str:
    """Random file selection baseline.
    
    Selects a random Python file from the repository as context.
    """
    repo_name = task['repo'].replace('/', '__')
    revision = task['revision']
    stage = task.get('stage', 'practice')
    
    zip_path = data_dir / f"python-{stage}" / f"{repo_name}-{revision}.zip"
    
    if not zip_path.exists():
        logger.warning(f"Zip not found: {zip_path}")
        return ""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = extract_repo(zip_path, Path(tmpdir))
        py_files = get_python_files(repo_path)
        
        if not py_files:
            return ""
        
        # Select random file
        selected = random.choice(py_files)
        content = selected.read_text(errors='replace')
        
        return content[:max_chars]


def bm25_strategy(
    task: Dict,
    data_dir: Path,
    max_chars: int = 8000 * 4
) -> str:
    """BM25-based file selection baseline.
    
    Selects files based on BM25 relevance to the query.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed, using simple fallback")
        return random_strategy(task, data_dir, max_chars)
    
    repo_name = task['repo'].replace('/', '__')
    revision = task['revision']
    stage = task.get('stage', 'practice')
    
    zip_path = data_dir / f"python-{stage}" / f"{repo_name}-{revision}.zip"
    
    if not zip_path.exists():
        logger.warning(f"Zip not found: {zip_path}")
        return ""
    
    # Build query from prefix/suffix
    query = task.get('prefix', '')[-1000:] + ' ' + task.get('suffix', '')[:500]
    query_tokens = query.lower().split()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = extract_repo(zip_path, Path(tmpdir))
        py_files = get_python_files(repo_path)
        
        if not py_files:
            return ""
        
        # Read all files
        file_contents = []
        file_tokens = []
        
        for f in py_files:
            content = f.read_text(errors='replace')
            tokens = content.lower().split()
            file_contents.append((f, content))
            file_tokens.append(tokens)
        
        # Score with BM25
        bm25 = BM25Okapi(file_tokens)
        scores = bm25.get_scores(query_tokens)
        
        # Sort by score
        scored_files = list(zip(py_files, file_contents, scores))
        scored_files.sort(key=lambda x: x[2], reverse=True)
        
        # Select top files until max_chars
        context_parts = []
        total_chars = 0
        
        for _, (f, content), score in scored_files[:3]:  # Top 3 files
            if total_chars + len(content) > max_chars:
                remaining = max_chars - total_chars
                content = content[:remaining]
            
            context_parts.append(f"<|file_sep|>\n# File: {f.name}\n{content}")
            total_chars += len(content)
            
            if total_chars >= max_chars:
                break
        
        return '\n\n'.join(context_parts)


def process_jsonl(
    input_path: Path,
    output_path: Path,
    data_dir: Path,
    strategy: str,
    limit: int = None
):
    """Process JSONL file with given strategy."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {input_path} with {strategy} strategy...")
    
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        count = 0
        
        for line in f_in:
            if limit and count >= limit:
                break
            
            task = json.loads(line.strip())
            logger.info(f"Task {count + 1}: {task.get('id', 'unknown')}")
            
            try:
                if strategy == 'random':
                    context = random_strategy(task, data_dir)
                elif strategy == 'bm25':
                    context = bm25_strategy(task, data_dir)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                prediction = {"context": context}
                f_out.write(json.dumps(prediction) + '\n')
                
            except Exception as e:
                logger.error(f"Error processing task: {e}")
                f_out.write(json.dumps({"context": ""}) + '\n')
            
            count += 1
            
            if count % 10 == 0:
                logger.info(f"Processed {count} tasks...")
    
    logger.info(f"Complete! Wrote {count} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Baseline strategies")
    parser.add_argument('--stage', default='practice', 
                       choices=['practice', 'public'])
    parser.add_argument('--strategy', default='bm25',
                       choices=['random', 'bm25'])
    parser.add_argument('--lang', default='python',
                       choices=['python', 'kotlin'])
    parser.add_argument('--data-dir', 
                       default='data/drive-download-20260314T192239Z-1-001',
                       type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--trim-prefix', action='store_true',
                       help='Trim prefix to 10 lines')
    parser.add_argument('--trim-suffix', action='store_true',
                       help='Trim suffix to 10 lines')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    input_file = args.data_dir / f"{args.lang}-{args.stage}.jsonl"
    output_file = Path('predictions') / f"{args.lang}-{args.stage}-{args.strategy}.jsonl"
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    process_jsonl(input_file, output_file, args.data_dir, args.strategy, args.limit)


if __name__ == "__main__":
    main()
