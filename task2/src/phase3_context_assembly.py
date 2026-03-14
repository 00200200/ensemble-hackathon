"""
Phase 3 & 4: Context Assembly

Uses the repository's Call Graph to provide the LLM with necessary context.
Implements:
- Seed selection from hybrid retrieval
- Neighborhood expansion via graph traversal
- Centrality filtering
- Token budget management
- Ascending relevance ordering (for left-truncation)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import tree_sitter
    from tree_sitter_language_pack import get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

from src.llm_abstraction import CachedLLMAbstractor

logger = logging.getLogger(__name__)


@dataclass
class ContextItem:
    """A single context item."""
    entity: Any
    entity_id: str
    source: str  # 'current_file', 'retrieval_seed', 'neighbor', 'import'
    relevance_score: float
    code_to_show: str  # Full code or abstract
    token_estimate: int


class QueryBuilder:
    """Build search queries from prefix/suffix."""
    
    def __init__(self):
        self.parser = None
        if TREE_SITTER_AVAILABLE:
            try:
                self.parser = get_parser('python')
            except Exception as e:
                logger.warning(f"Failed to load tree-sitter parser: {e}")
    
    def extract_identifiers(self, code: str) -> List[str]:
        """Extract identifiers from code using Tree-sitter."""
        if not self.parser or not code:
            # Fallback: simple regex extraction
            return self._extract_identifiers_regex(code)
        
        try:
            tree = self.parser.parse(code.encode('utf-8'))
            root = tree.root_node
            
            identifiers = []
            self._extract_identifiers_recursive(root, code, identifiers)
            
            # Filter duplicates and common keywords
            seen = set()
            filtered = []
            keywords = {
                'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'return',
                'import', 'from', 'as', 'try', 'except', 'finally', 'with',
                'pass', 'break', 'continue', 'lambda', 'yield', 'raise',
                'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is',
                'self', 'cls', 'print', 'len', 'range', 'enumerate', 'zip',
                'self', 'super', 'object', 'int', 'str', 'list', 'dict', 'set'
            }
            
            for ident in identifiers:
                ident_lower = ident.lower()
                if ident_lower not in keywords and ident_lower not in seen and len(ident) > 1:
                    seen.add(ident_lower)
                    filtered.append(ident)
            
            return filtered
        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed: {e}")
            return self._extract_identifiers_regex(code)
    
    def _extract_identifiers_recursive(self, node, code: str, identifiers: List[str]):
        """Recursively extract identifiers from AST."""
        if node.type == 'identifier':
            text = code[node.start_byte:node.end_byte]
            identifiers.append(text)
        
        for child in node.children:
            self._extract_identifiers_recursive(child, code, identifiers)
    
    def _extract_identifiers_regex(self, code: str) -> List[str]:
        """Fallback regex-based identifier extraction."""
        if not code:
            return []
        
        # Match identifiers
        pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        matches = re.findall(pattern, code)
        
        # Filter keywords
        keywords = {
            'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'return',
            'import', 'from', 'as', 'try', 'except', 'finally', 'with',
            'pass', 'break', 'continue', 'lambda', 'yield', 'raise',
            'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is',
            'self', 'cls', 'print', 'len', 'range', 'enumerate', 'zip'
        }
        
        seen = set()
        filtered = []
        for m in matches:
            if m not in keywords and m not in seen and len(m) > 1:
                seen.add(m)
                filtered.append(m)
        
        return filtered
    
    def build_query(
        self,
        prefix: str,
        suffix: str,
        prefix_weight: float = 0.6
    ) -> str:
        """
        Build search query from prefix and suffix.
        
        Args:
            prefix: Code before cursor
            suffix: Code after cursor
            prefix_weight: Weight for prefix vs suffix
        
        Returns:
            Query string
        """
        # Extract identifiers from both
        prefix_ids = self.extract_identifiers(prefix[-1000:] if len(prefix) > 1000 else prefix)
        suffix_ids = self.extract_identifiers(suffix[:500] if len(suffix) > 500 else suffix)
        
        # Combine with weights
        query_parts = []
        
        # Add prefix identifiers (multiple times for weight)
        for _ in range(int(prefix_weight * 3)):
            query_parts.extend(prefix_ids)
        
        # Add suffix identifiers
        for _ in range(int((1 - prefix_weight) * 3)):
            query_parts.extend(suffix_ids)
        
        # Add raw code context (truncated)
        prefix_lines = prefix.split('\n')[-10:]  # Last 10 lines
        suffix_lines = suffix.split('\n')[:5]    # First 5 lines
        
        query_parts.extend(prefix_lines)
        query_parts.extend(suffix_lines)
        
        return ' '.join(query_parts)


class ContextAssembler:
    """Assemble context from graph and retrieval results."""
    
    def __init__(
        self,
        graph: Any,
        retriever: Any,
        max_tokens: int = 8000,
        seed_count: int = 5,
        expansion_depth: int = 1,
        centrality_threshold: int = 20,
        centrality_penalty: float = 0.5,
        use_llm_abstracts: bool = True,
        cache_dir: str = ".cache",
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None
    ):
        self.graph = graph
        self.retriever = retriever
        self.max_tokens = max_tokens
        self.seed_count = seed_count
        self.expansion_depth = expansion_depth
        self.centrality_threshold = centrality_threshold
        self.centrality_penalty = centrality_penalty
        self.query_builder = QueryBuilder()
        self.use_llm_abstracts = use_llm_abstracts
        
        # Initialize LLM abstractor if enabled
        self.llm_abstractor = None
        if use_llm_abstracts:
            try:
                import os
                # Auto-detect DeepSeek from env vars
                model = llm_model or os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL", "gpt-4o-mini")
                api_key = llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
                base_url = llm_base_url or os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
                
                # Auto-set DeepSeek base URL if not provided
                if not base_url and "deepseek" in model.lower():
                    base_url = "https://api.deepseek.com"
                
                if not api_key:
                    logger.info("No LLM API key found. Using fallback abstracts (signature + docstring + deps).")
                    logger.info("To enable LLM abstracts, set OPENAI_API_KEY or DEEPSEEK_API_KEY")
                    self.use_llm_abstracts = False
                else:
                    self.llm_abstractor = CachedLLMAbstractor(
                        cache_dir=cache_dir,
                        model=model,
                        api_key=api_key,
                        base_url=base_url
                    )
                    logger.info(f"LLM abstractor initialized: {model}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM abstractor: {e}")
                self.use_llm_abstracts = False
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimate: 1 token ≈ 4 characters for code
        return len(text) // 4
    
    def assemble(
        self,
        query: str,
        current_file: str,
        current_file_entities: Optional[List[Any]] = None,
        prefix: str = "",
        suffix: str = ""
    ) -> List[ContextItem]:
        """
        Assemble context for a completion task.
        
        Args:
            query: Search query
            current_file: Path of file being edited
            current_file_entities: Entities from current file
            prefix: Code before cursor
            suffix: Code after cursor
        
        Returns:
            List of context items sorted by ascending relevance
        """
        context_items: List[ContextItem] = []
        seen_entities: Set[str] = set()
        
        # 1. Add current file entities (highest priority)
        if current_file_entities:
            for entity in current_file_entities:
                entity_id = entity.id if hasattr(entity, 'id') else str(entity)
                if entity_id in seen_entities:
                    continue
                seen_entities.add(entity_id)
                
                code = entity.raw_code if hasattr(entity, 'raw_code') else str(entity)
                tokens = self.estimate_tokens(code)
                
                item = ContextItem(
                    entity=entity,
                    entity_id=entity_id,
                    source='current_file',
                    relevance_score=1.0,  # Highest relevance
                    code_to_show=code,
                    token_estimate=tokens
                )
                context_items.append(item)
        
        # 2. Retrieve seeds from hybrid retriever
        seeds = self.retriever.search(query, k=self.seed_count)
        
        for seed_entity, seed_score in seeds:
            entity_id = seed_entity.id if hasattr(seed_entity, 'id') else str(seed_entity)
            if entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)
            
            # Apply centrality penalty if high-degree
            degree = self.graph.get_degree(entity_id) if hasattr(self.graph, 'get_degree') else 0
            if degree > self.centrality_threshold:
                seed_score *= self.centrality_penalty
            
            # Use abstract for retrieved seeds (not full code) to save tokens
            code = self._create_abstract(seed_entity)
            tokens = self.estimate_tokens(code)
            
            item = ContextItem(
                entity=seed_entity,
                entity_id=entity_id,
                source='retrieval_seed',
                relevance_score=seed_score,
                code_to_show=code,
                token_estimate=tokens
            )
            context_items.append(item)
            
            # 3. Expand neighbors
            if hasattr(self.graph, 'get_neighbors'):
                neighbors = self.graph.get_neighbors(
                    entity_id,
                    relation_types=['CALLS', 'CONTAINS'],
                    depth=self.expansion_depth
                )
                
                for neighbor_id, neighbor_entity, depth in neighbors:
                    if neighbor_id in seen_entities:
                        continue
                    seen_entities.add(neighbor_id)
                    
                    # For neighbors, show abstract instead of full code
                    code = self._create_abstract(neighbor_entity)
                    tokens = self.estimate_tokens(code)
                    
                    # Score decays with depth
                    neighbor_score = seed_score * (0.7 ** depth)
                    
                    item = ContextItem(
                        entity=neighbor_entity,
                        entity_id=neighbor_id,
                        source='neighbor',
                        relevance_score=neighbor_score,
                        code_to_show=code,
                        token_estimate=tokens
                    )
                    context_items.append(item)
        
        # 4. Sort by ASCENDING relevance (least relevant first)
        # This ensures most relevant content survives left-truncation
        context_items.sort(key=lambda x: x.relevance_score)
        
        # 5. Apply token budget
        return self._apply_token_budget(context_items)
    
    def _create_abstract(self, entity: Any) -> str:
        """Create a functional abstract for an entity."""
        # Try LLM abstractor first if enabled
        if self.use_llm_abstracts and self.llm_abstractor:
            try:
                llm_abstract = self.llm_abstractor.get_abstract(entity)
                if llm_abstract and len(llm_abstract) > 10:
                    # Add signature as header if not in abstract
                    signature = getattr(entity, 'signature', '')
                    if signature and signature not in llm_abstract:
                        return f"{signature}\n{llm_abstract}"
                    return llm_abstract
            except Exception as e:
                logger.debug(f"LLM abstract failed: {e}")
        
        # Fallback: manual abstract
        parts = []
        
        if hasattr(entity, 'signature') and entity.signature:
            parts.append(entity.signature)
        
        if hasattr(entity, 'docstring') and entity.docstring:
            parts.append(f'"""{entity.docstring}"""')
        
        # If no signature/docstring, show first few lines
        if not parts and hasattr(entity, 'raw_code') and entity.raw_code:
            lines = entity.raw_code.split('\n')[:5]
            parts.append('\n'.join(lines))
            parts.append("# ... (truncated)")
        
        return '\n'.join(parts)
    
    def _apply_token_budget(
        self,
        items: List[ContextItem],
        current_file_budget: float = 0.4,
        cross_file_budget: float = 0.3
    ) -> List[ContextItem]:
        """
        Apply token budget constraints.
        
        Budget allocation:
        - 40% current file
        - 30% cross-file context
        - 20% suffix (handled separately)
        - 10% buffer
        """
        current_file_tokens = int(self.max_tokens * current_file_budget)
        cross_file_tokens = int(self.max_tokens * cross_file_budget)
        
        result = []
        current_file_used = 0
        cross_file_used = 0
        
        for item in items:
            if item.source == 'current_file':
                if current_file_used + item.token_estimate <= current_file_tokens:
                    result.append(item)
                    current_file_used += item.token_estimate
                else:
                    # Try to truncate
                    remaining = current_file_tokens - current_file_used
                    if remaining > 100:  # Only if meaningful amount left
                        truncated_code = self._truncate_code(item.code_to_show, remaining)
                        if truncated_code:
                            item.code_to_show = truncated_code
                            item.token_estimate = remaining
                            result.append(item)
                            current_file_used += remaining
            else:
                if cross_file_used + item.token_estimate <= cross_file_tokens:
                    result.append(item)
                    cross_file_used += item.token_estimate
                else:
                    # Skip if over budget
                    pass
        
        return result
    
    def _truncate_code(self, code: str, max_tokens: int) -> str:
        """Truncate code to fit within token budget."""
        # Rough estimate: 4 chars per token
        max_chars = max_tokens * 4
        
        if len(code) <= max_chars:
            return code
        
        # Truncate and add indicator
        truncated = code[:max_chars]
        # Try to end at a newline
        last_newline = truncated.rfind('\n')
        if last_newline > max_chars * 0.8:
            truncated = truncated[:last_newline]
        
        return truncated + "\n# ... (truncated)"
    
    def format_context(self, items: List[ContextItem]) -> str:
        """Format context items into a string."""
        if not items:
            return ""
        
        parts = []
        
        # Group by file for better organization
        file_groups: Dict[str, List[ContextItem]] = {}
        for item in items:
            file_path = getattr(item.entity, 'file_path', 'unknown')
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(item)
        
        for file_path, file_items in file_groups.items():
            file_name = file_path.split('/')[-1] if '/' in file_path else file_path
            parts.append(f"<|file_sep|>\n# File: {file_name}")
            
            for item in file_items:
                entity_name = getattr(item.entity, 'name', 'unknown')
                entity_type = getattr(item.entity, 'type', 'unknown')
                
                parts.append(f"# {entity_type}: {entity_name}")
                parts.append(item.code_to_show)
                parts.append("")  # Empty line between entities
        
        return '\n'.join(parts)
    
    def assemble_for_task(
        self,
        task: Dict[str, Any],
        graph: Any,
        retriever: Any
    ) -> str:
        """
        Convenience method to assemble context for a task.
        
        Args:
            task: Task dictionary with prefix, suffix, path, etc.
            graph: Code graph
            retriever: Hybrid retriever
        
        Returns:
            Formatted context string
        """
        prefix = task.get('prefix', '')
        suffix = task.get('suffix', '')
        current_file = task.get('path', '')
        
        # Build query
        query = self.query_builder.build_query(prefix, suffix)
        
        # Get current file entities from graph
        current_file_entities = None
        if hasattr(graph, 'get_entities_by_file'):
            current_file_entities = graph.get_entities_by_file(current_file)
        
        # Assemble context
        context_items = self.assemble(
            query=query,
            current_file=current_file,
            current_file_entities=current_file_entities,
            prefix=prefix,
            suffix=suffix
        )
        
        return self.format_context(context_items)
