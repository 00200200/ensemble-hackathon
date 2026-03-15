"""
Phase 6: Inference Pipeline

End-to-end pipeline for processing code completion tasks.
Integrates all phases: parsing, graph building, retrieval, and context assembly.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.phase1_ast_parser import ASTParser, ParseCache
from src.graph_store import CodeGraph
from src.vector_store import HybridRetriever
from src.phase3_context_assembly import ContextAssembler

logger = logging.getLogger(__name__)


class RepositoryProcessor:
    """Process a repository for code completion."""
    
    def __init__(
        self,
        cache_dir: str = ".cache",
        use_tree_sitter: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.parser = ASTParser(use_tree_sitter=use_tree_sitter)
        self.parse_cache = ParseCache(cache_dir)
    
    def process_repository(
        self,
        repo_path: Path,
        repo_name: str,
        revision: str
    ) -> tuple[CodeGraph, HybridRetriever]:
        """
        Process a repository and build graph + index.
        
        Args:
            repo_path: Path to repository directory
            repo_name: Repository name
            revision: Git revision
        
        Returns:
            (graph, retriever)
        """
        cache_key = f"{repo_name.replace('/', '__')}_{revision}"
        graph_cache = self.cache_dir / f"{cache_key}_graph.pkl"
        index_cache = self.cache_dir / f"{cache_key}_index.pkl"
        
        # Try to load from cache
        logger.info(f"DEBUG: Looking for cache at {graph_cache}, exists={graph_cache.exists()}")
        if graph_cache.exists():
            try:
                logger.info(f"Loading cached graph from {graph_cache}")
                graph = CodeGraph()
                graph.load(graph_cache)
                
                # Also load retriever if available
                retriever = HybridRetriever()
                if index_cache.exists():
                    # TODO: Implement retriever save/load
                    logger.info("Cached index found but reloading")
                
                logger.info(f"Loaded {len(graph)} entities from cache")
                logger.info(f"DEBUG: Loaded graph has {len(graph._file_to_entities)} files")
                
                # Still need to index entities
                all_entities = [e for _, e in graph.get_all_entities()]
                retriever.index_entities(all_entities)
                
                return graph, retriever
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, reprocessing")
        
        # Process repository
        logger.info(f"Processing repository: {repo_path}")
        
        # Parse all files
        parse_results = self.parser.parse_directory(repo_path)
        
        # Build graph
        graph = CodeGraph()
        all_entities = []
        
        for result in parse_results:
            graph.add_file_entities(
                result.file_path,
                result.entities,
                result.imports
            )
            all_entities.extend(result.entities)
        
        logger.info(f"Built graph with {len(graph)} entities")
        
        # Build index
        retriever = HybridRetriever()
        if all_entities:
            logger.info(f"Indexing {len(all_entities)} entities...")
            retriever.index_entities(all_entities)
        
        # Save to cache
        try:
            graph.save(graph_cache)
            logger.info(f"Saved graph to {graph_cache}")
        except Exception as e:
            logger.warning(f"Failed to save graph cache: {e}")
        
        return graph, retriever


class CompletionPipeline:
    """End-to-end pipeline for code completion."""
    
    def __init__(
        self,
        data_dir: Path,
        cache_dir: str = ".cache",
        max_tokens: int = 8000,
        stage: str = "practice"
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.max_tokens = max_tokens
        self.stage = stage
        self.repo_processor = RepositoryProcessor(cache_dir)
        
        # Cache for processed repositories
        self._repo_cache: Dict[str, tuple[CodeGraph, HybridRetriever]] = {}
    
    def get_repo_path(self, task: Dict[str, Any]) -> Optional[Path]:
        """Get repository path for a task."""
        repo = task.get('repo', '')
        revision = task.get('revision', '')
        stage = task.get('stage', self.stage)  # Use pipeline's stage as default
        archive = task.get('archive')  # For dataset stage
        
        repo_name = repo.replace('/', '__')
        
        # For dataset stage, use archive field if available
        if archive and stage == 'dataset':
            archive_name = archive.replace('.zip', '')
            
            # Check multiple possible locations for extracted directory
            for dataset_dir in ["python-dataset", "python-dataset1", "dataset"]:
                extracted_dir = self.data_dir / dataset_dir / archive_name
                if extracted_dir.exists():
                    return extracted_dir
            
            # Check for zip file in multiple locations
            for dataset_dir in ["python-dataset", "python-dataset1", "dataset"]:
                zip_path = self.data_dir / dataset_dir / archive
                if zip_path.exists():
                    # Extract to temp location
                    import tempfile
                    extract_dir = Path(tempfile.mkdtemp()) / archive_name
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(extract_dir)
                    
                    logger.info(f"Extracted repository to: {extract_dir}")
                    return extract_dir
        
        # Standard stage handling (practice, public)
        # Check for extracted directory first
        extracted_dir = self.data_dir / f"python-{stage}" / f"{repo_name}-{revision}"
        if extracted_dir.exists():
            return extracted_dir
        
        # Check for zip file
        zip_path = self.data_dir / f"python-{stage}" / f"{repo_name}-{revision}.zip"
        if zip_path.exists():
            # Extract to temp location
            import tempfile
            extract_dir = Path(tempfile.mkdtemp()) / f"{repo_name}-{revision}"
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            # The zip extracts directly to extract_dir (no nested root folder)
            # So we return extract_dir itself
            logger.info(f"Extracted repository to: {extract_dir}")
            return extract_dir
        
        logger.warning(f"Repository not found: {repo} @ {revision}")
        return None
    
    def process_task(
        self,
        task: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Process a single completion task.
        
        Args:
            task: Task dictionary with prefix, suffix, path, etc.
        
        Returns:
            Dictionary with context, prefix, and suffix
        """
        repo = task.get('repo', '')
        revision = task.get('revision', '')
        cache_key = f"{repo}@{revision}"
        
        # Get prefix and suffix from task
        prefix = task.get('prefix', '')
        suffix = task.get('suffix', '')
        
        # Get or build graph and retriever
        if cache_key not in self._repo_cache:
            repo_path = self.get_repo_path(task)
            if not repo_path:
                # Fallback: return empty context
                return {"context": ""}
            
            graph, retriever = self.repo_processor.process_repository(
                repo_path,
                repo,
                revision
            )
            self._repo_cache[cache_key] = (graph, retriever)
        else:
            graph, retriever = self._repo_cache[cache_key]
        
        # Build context
        assembler = ContextAssembler(
            graph=graph,
            retriever=retriever,
            max_tokens=self.max_tokens
        )
        
        context = assembler.assemble_for_task(task, graph, retriever)
        
        # Return dict with context only
        return {
            "context": context
        }
    
    def process_jsonl(
        self,
        input_path: Path,
        output_path: Path,
        limit: Optional[int] = None,
        skip: int = 0,
        append: bool = False,
        filter_ids: Optional[set] = None
    ) -> None:
        """
        Process a JSONL file of tasks.
        
        Args:
            input_path: Path to input JSONL
            output_path: Path to output JSONL
            limit: Optional limit on number of tasks
            skip: Number of tasks to skip from beginning (for resuming)
            append: If True, append to output file instead of overwriting
            filter_ids: Optional set of task IDs to process (skip others)
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing {input_path} -> {output_path}")
        if skip > 0:
            logger.info(f"Skipping first {skip} tasks")
        if filter_ids:
            logger.info(f"Filtering to {len(filter_ids)} specific task IDs")
        
        mode = 'a' if append else 'w'
        
        with open(input_path, 'r') as f_in, open(output_path, mode) as f_out:
            count = 0
            skipped = 0
            line_num = 0
            
            for line in f_in:
                line_num += 1
                
                task = json.loads(line.strip())
                task_id = task.get('id', f'task_{line_num}')
                
                # Handle filter logic
                if filter_ids and task_id not in filter_ids:
                    continue
                
                # Handle skip logic
                if skipped < skip:
                    skipped += 1
                    continue
                
                if limit and count >= limit:
                    break
                
                task_num = skip + count + 1
                
                logger.info(f"Processing task {task_num} (line {line_num}): {task_id}")
                
                try:
                    prediction = self.process_task(task)
                    f_out.write(json.dumps(prediction) + '\n')
                    
                except Exception as e:
                    logger.error(f"Error processing task {task_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Write empty context on error
                    prediction = {
                        "context": ""
                    }
                    f_out.write(json.dumps(prediction) + '\n')
                
                count += 1
                
                if count % 10 == 0:
                    logger.info(f"Processed {count} tasks...")
        
        logger.info(f"Complete! Wrote {count} predictions to {output_path}")


def run_pipeline(
    data_dir: str,
    stage: str = "practice",
    output_dir: str = "predictions",
    limit: Optional[int] = None,
    verbose: bool = False
) -> Path:
    """
    Run the full pipeline.
    
    Args:
        data_dir: Data directory
        stage: Competition stage (practice, public, etc.)
        output_dir: Output directory
        limit: Optional limit on number of tasks
        verbose: Enable verbose logging
    
    Returns:
        Path to output file
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = data_dir / f"python-{stage}.jsonl"
    output_file = output_dir / f"python-{stage}-predictions.jsonl"
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    pipeline = CompletionPipeline(data_dir)
    pipeline.process_jsonl(input_file, output_file, limit)
    
    return output_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run completion pipeline")
    parser.add_argument('--data-dir', required=True, help='Data directory')
    parser.add_argument('--stage', default='practice', help='Stage (practice, public)')
    parser.add_argument('--output-dir', default='predictions', help='Output directory')
    parser.add_argument('--limit', type=int, help='Limit number of tasks')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    output = run_pipeline(
        data_dir=args.data_dir,
        stage=args.stage,
        output_dir=args.output_dir,
        limit=args.limit,
        verbose=args.verbose
    )
    
    print(f"Output written to: {output}")
