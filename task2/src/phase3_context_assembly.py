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
import os
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
from src.import_resolver import ImportResolver

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
        # More accurate estimate: 1 token ≈ 3.5 characters for code
        # Code has many special chars that are separate tokens
        return int(len(text) / 3.5)
    
    def assemble(
        self,
        query: str,
        current_file: str,
        current_file_entities: Optional[List[Any]] = None,
        prefix: str = "",
        suffix: str = "",
        modified_files: Optional[List[str]] = None
    ) -> List[ContextItem]:
        """
        Assemble context for a completion task.
        
        Args:
            query: Search query
            current_file: Path of file being edited
            current_file_entities: Entities from current file
            prefix: Code before cursor
            suffix: Code after cursor
            modified_files: List of files modified in the same commit (high relevance)
        
        Returns:
            List of context items sorted by ascending relevance
        """
        context_items: List[ContextItem] = []
        seen_entities: Set[str] = set()
        modified_files = modified_files or []
        
        # 1. Add current file entities - TRIMMED to relevant scope
        # Only include entity containing the cursor + parent class
        # This saves tokens for cross-file context (imports, related files)
        logger.info(f"DEBUG: current_file_entities = {len(current_file_entities) if current_file_entities else None}")
        if current_file_entities:
            # Find entity containing the cursor based on prefix line count
            cursor_line = len(prefix.split('\n')) if prefix else 1
            relevant_entities = self._find_relevant_entities_at_cursor(
                current_file_entities, cursor_line
            )
            
            logger.info(f"Adding {len(relevant_entities)} relevant entities from current file (cursor at line {cursor_line})")
            
            for entity in relevant_entities:
                entity_id = entity.id if hasattr(entity, 'id') else str(entity)
                if entity_id in seen_entities:
                    continue
                seen_entities.add(entity_id)
                
                # Use FULL CODE for current file entity
                code = entity.raw_code if hasattr(entity, 'raw_code') else str(entity)
                tokens = self.estimate_tokens(code)
                
                item = ContextItem(
                    entity=entity,
                    entity_id=entity_id,
                    source='current_file',
                    relevance_score=2.0,  # HIGHEST relevance
                    code_to_show=code,  # FULL CODE
                    token_estimate=tokens
                )
                context_items.append(item)
        else:
            logger.warning(f"No entities found for current file: {current_file}")
        
        # 2. Add entities from MODIFIED FILES - files changed in same commit
        # These are HIGHLY RELEVANT because they were modified together!
        # BUT limit to top N files to save tokens for imports
        MAX_MODIFIED_FILES = 10  # Limit to prevent token overflow
        
        if modified_files and hasattr(self.graph, 'get_entities_by_file'):
            mod_files_only = [f for f in modified_files if f != current_file]
            # Limit number of modified files
            mod_files_limited = mod_files_only[:MAX_MODIFIED_FILES]
            
            logger.info(f"Adding entities from {len(mod_files_limited)}/{len(mod_files_only)} modified files")
            
            if mod_files_limited:
                # Only take FIRST entity from each modified file (most important)
                for mod_file in mod_files_limited:
                    try:
                        mod_entities = self.graph.get_entities_by_file(mod_file)
                        if mod_entities:
                            # Only take first entity (likely class or main function)
                            entity = mod_entities[0]
                            entity_id = entity.id if hasattr(entity, 'id') else str(entity)
                            if entity_id in seen_entities:
                                continue
                            seen_entities.add(entity_id)
                            
                            # Use ABSTRACT for modified files to save tokens
                            code = self._create_abstract(entity)
                            tokens = self.estimate_tokens(code)
                            
                            item = ContextItem(
                                entity=entity,
                                entity_id=entity_id,
                                source='modified_file',
                                relevance_score=1.7,  # HIGH but below imports
                                code_to_show=code,
                                token_estimate=tokens
                            )
                            context_items.append(item)
                        else:
                            logger.warning(f"  - {mod_file}: NO entities found!")
                    except Exception as e:
                        logger.debug(f"Could not get entities for modified file {mod_file}: {e}")
        
        # 2. Resolve imports/dependencies - CRITICAL for base classes!
        # Parse imports from prefix and resolve them
        if prefix and hasattr(self.graph, 'get_entities_by_name'):
            try:
                # Parse imports from prefix
                from src.import_resolver import ImportParser
                imports = ImportParser.extract_imports(prefix)
                
                logger.info(f"Found {len(imports)} imports in prefix")
                
                # Resolve each import
                import_definitions = []
                for imp in imports[:10]:  # Limit to top 10
                    name = imp.get('name', '')
                    if name and hasattr(self.graph, 'get_entities_by_name'):
                        entities = self.graph.get_entities_by_name(name)
                        if entities:
                            # Prefer entities not in current file (cross-file)
                            for e in entities:
                                e_file = getattr(e, 'file_path', '')
                                if e_file != current_file:
                                    import_definitions.append(e)
                                    break
                            else:
                                # If all in current file, take first
                                import_definitions.append(entities[0])
                
                # Also look at class inheritance from raw_code
                for entity in current_file_entities:
                    code = getattr(entity, 'raw_code', '')
                    if code:
                        # Look for class inheritance patterns: class X(Y):
                        import re
                        match = re.search(r'class\s+\w+\s*\(([^)]+)\)', code)
                        if match:
                            parent_class = match.group(1).strip()
                            if parent_class and hasattr(self.graph, 'get_entities_by_name'):
                                entities = self.graph.get_entities_by_name(parent_class)
                                for e in entities:
                                    e_file = getattr(e, 'file_path', '')
                                    if e_file != current_file:
                                        import_definitions.append(e)
                                        logger.info(f"Found parent class: {parent_class}")
                                        break
                
                if import_definitions:
                    logger.info(f"Resolved {len(import_definitions)} imports/dependencies")
                    
                    for entity in import_definitions:
                        entity_id = entity.id if hasattr(entity, 'id') else str(entity)
                        if entity_id in seen_entities:
                            continue
                        seen_entities.add(entity_id)
                        
                        # Use FULL CODE for import definitions
                        code = entity.raw_code if hasattr(entity, 'raw_code') else str(entity)
                        tokens = self.estimate_tokens(code)
                        
                        item = ContextItem(
                            entity=entity,
                            entity_id=entity_id,
                            source='import_definition',
                            relevance_score=1.85,  # VERY HIGH
                            code_to_show=code,
                            token_estimate=tokens
                        )
                        context_items.append(item)
                        logger.info(f"Added import: {getattr(entity, 'name', 'unknown')} ({getattr(entity, 'file_path', 'unknown')})")
            except Exception as e:
                logger.warning(f"Import resolution failed: {e}")
        
        # 3. Retrieve seeds from hybrid retriever
        # Increase seed count to get more candidates
        seeds = self.retriever.search(query, k=self.seed_count * 5)  # Get more candidates for filtering
        
        # Filter and score seeds
        filtered_seeds = []
        for seed_entity, seed_score in seeds:
            entity_id = seed_entity.id if hasattr(seed_entity, 'id') else str(seed_entity)
            if entity_id in seen_entities:
                continue
            
            file_path = getattr(seed_entity, 'file_path', '')
            entity_name = getattr(seed_entity, 'name', '')
            
            # Skip low-value entities
            # Skip empty test functions
            if 'test' in file_path.lower() and entity_name.startswith(('test_', 'empty_')):
                raw_code = getattr(seed_entity, 'raw_code', '')
                if len(raw_code) < 100:  # Skip tiny test functions
                    continue
            
            # Skip utility functions with generic names
            if entity_name in ('setup', 'teardown', '__init__', 'main') and seed_score < 0.5:
                continue
            
            # Boost source files over test files
            if 'test' not in file_path.lower() and '/tests/' not in file_path:
                seed_score *= 1.2
            
            # Apply centrality penalty if high-degree
            degree = self.graph.get_degree(entity_id) if hasattr(self.graph, 'get_degree') else 0
            if degree > self.centrality_threshold:
                seed_score *= self.centrality_penalty
            
            filtered_seeds.append((seed_entity, seed_score))
        
        # Sort by score and take top k
        filtered_seeds.sort(key=lambda x: x[1], reverse=True)
        
        for seed_entity, seed_score in filtered_seeds[:self.seed_count]:
            entity_id = seed_entity.id if hasattr(seed_entity, 'id') else str(seed_entity)
            seen_entities.add(entity_id)
            
            # Use ABSTRACT for retrieved seeds (full code would be too long)
            code = self._create_abstract(seed_entity)
            tokens = self.estimate_tokens(code)
            
            item = ContextItem(
                entity=seed_entity,
                entity_id=entity_id,
                source='retrieval_seed',
                relevance_score=seed_score,
                code_to_show=code,  # Abstract
                token_estimate=tokens
            )
            context_items.append(item)
            
            # 4. Add adjacent chunks - functions before/after in same file
            # This provides local context around the retrieved entity
            try:
                seed_file = getattr(seed_entity, 'file_path', '')
                seed_line = getattr(seed_entity, 'line_start', 0)
                if seed_file and seed_line > 0:
                    # Get all entities from the same file
                    file_entities = self.graph.get_entities_by_file(seed_file)
                    for adj_entity in file_entities:
                        adj_id = adj_entity.id if hasattr(adj_entity, 'id') else str(adj_entity)
                        if adj_id in seen_entities:
                            continue
                        adj_line = getattr(adj_entity, 'line_start', 0)
                        # Check if within 50 lines (adjacent)
                        if abs(adj_line - seed_line) <= 50:
                            seen_entities.add(adj_id)
                            adj_code = self._create_abstract(adj_entity)
                            adj_tokens = self.estimate_tokens(adj_code)
                            adj_item = ContextItem(
                                entity=adj_entity,
                                entity_id=adj_id,
                                source='adjacent_chunk',
                                relevance_score=seed_score * 0.8,  # Slightly lower than seed
                                code_to_show=adj_code,
                                token_estimate=adj_tokens
                            )
                            context_items.append(adj_item)
            except Exception as e:
                logger.debug(f"Failed to add adjacent chunks: {e}")
            
            # 5. Expand neighbors (graph traversal)
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
        
        # 5. Sort by ASCENDING relevance (least relevant first)
        # This ensures most relevant content survives left-truncation
        context_items.sort(key=lambda x: x.relevance_score)
        
        # DEBUG: Check what we have before budget
        logger.info(f"DEBUG before budget: {len(context_items)} items")
        sources = {}
        for item in context_items:
            sources[item.source] = sources.get(item.source, 0) + 1
        logger.info(f"DEBUG sources: {sources}")
        
        # 6. Apply token budget
        return self._apply_token_budget(context_items)
    
    def _find_relevant_entities_at_cursor(
        self, 
        entities: List[Any], 
        cursor_line: int
    ) -> List[Any]:
        """
        Find entities relevant at cursor position.
        
        Returns the entity containing the cursor + parent class if applicable.
        This avoids including the entire file which wastes tokens.
        """
        if not entities:
            return []
        
        # Find the deepest entity containing the cursor
        containing = []
        for entity in entities:
            line_start = getattr(entity, 'line_start', 0)
            line_end = getattr(entity, 'line_end', 0)
            
            if line_start <= cursor_line <= line_end:
                containing.append(entity)
        
        if not containing:
            # Cursor not in any entity, return class/module level if exists
            classes = [e for e in entities if getattr(e, 'type', '') == 'class']
            if classes:
                # Return the class with most methods (likely the main one)
                return [max(classes, key=lambda e: len(getattr(e, 'dependencies', [])))]
            return entities[:1]  # Return first entity as fallback
        
        # Find the deepest (smallest range) containing entity
        containing.sort(key=lambda e: 
            getattr(e, 'line_end', 0) - getattr(e, 'line_start', 0)
        )
        deepest = containing[0]
        result = [deepest]
        
        # If deepest is a method, also include parent class
        if getattr(deepest, 'type', '') == 'method':
            parent_name = getattr(deepest, 'parent', None)
            if parent_name:
                for entity in entities:
                    if (getattr(entity, 'type', '') == 'class' and 
                        getattr(entity, 'name', '') == parent_name):
                        result.insert(0, entity)  # Parent first
                        break
        
        return result
    
    def _create_abstract(self, entity: Any) -> str:
        """Create context for an entity - ALWAYS use full code for important entities."""
        # For retrieved/secondary entities, use full code (not useless signatures!)
        code = getattr(entity, 'raw_code', '')
        if code:
            # Return full code but truncated reasonably
            return code
        
        # Ultimate fallback
        signature = getattr(entity, 'signature', '')
        docstring = getattr(entity, 'docstring', '')
        
        parts = []
        if signature:
            parts.append(signature)
        if docstring:
            parts.append(f'"""{docstring}"""')
        
        return '\n'.join(parts) if parts else str(entity)
    
    def _apply_token_budget(
        self,
        items: List[ContextItem],
        current_file_budget: float = 0.30,  # 30% for current file (trimmed)
        import_budget: float = 0.30,  # 30% for imports - CRITICAL for base classes!
        modified_file_budget: float = 0.25,  # 25% for modified files
        cross_file_budget: float = 0.12  # 12% for other retrieved code
    ) -> List[ContextItem]:
        """
        Apply token budget constraints.
        
        Budget allocation:
        - 30% current file (trimmed to relevant scope)
        - 30% import definitions - CRITICAL for base classes like AttributeGetter!
        - 25% modified files (co-changed)
        - 12% other cross-file context
        - 3% buffer
        """
        current_file_tokens = int(self.max_tokens * current_file_budget)
        modified_file_tokens = int(self.max_tokens * modified_file_budget)
        import_tokens = int(self.max_tokens * import_budget)
        cross_file_tokens = int(self.max_tokens * cross_file_budget)
        
        result = []
        current_file_used = 0
        modified_file_used = 0
        import_used = 0
        cross_file_used = 0
        
        for item in items:
            if item.source == 'current_file':
                # Highest priority - current file entities
                logger.debug(f"Current file entity: {item.entity_id}, tokens: {item.token_estimate}")
                if current_file_used + item.token_estimate <= current_file_tokens:
                    result.append(item)
                    current_file_used += item.token_estimate
                    logger.debug(f"  -> Added, used: {current_file_used}/{current_file_tokens}")
                else:
                    # Try to truncate
                    remaining = current_file_tokens - current_file_used
                    logger.debug(f"  -> Too big, remaining: {remaining}")
                    if remaining > 100:
                        truncated_code = self._truncate_code(item.code_to_show, remaining)
                        if truncated_code:
                            item.code_to_show = truncated_code
                            item.token_estimate = remaining
                            result.append(item)
                            current_file_used += remaining
                            logger.debug(f"  -> Truncated and added")
                            
            elif item.source == 'modified_file':
                # Second priority - modified files (co-changed)
                if modified_file_used + item.token_estimate <= modified_file_tokens:
                    result.append(item)
                    modified_file_used += item.token_estimate
                else:
                    remaining = modified_file_tokens - modified_file_used
                    if remaining > 100:
                        truncated_code = self._truncate_code(item.code_to_show, remaining)
                        if truncated_code:
                            item.code_to_show = truncated_code
                            item.token_estimate = remaining
                            result.append(item)
                            modified_file_used += remaining
                            
            elif item.source == 'import_definition':
                # Third priority - imported symbols
                if import_used + item.token_estimate <= import_tokens:
                    result.append(item)
                    import_used += item.token_estimate
                else:
                    remaining = import_tokens - import_used
                    if remaining > 100:
                        truncated_code = self._truncate_code(item.code_to_show, remaining)
                        if truncated_code:
                            item.code_to_show = truncated_code
                            item.token_estimate = remaining
                            result.append(item)
                            import_used += remaining
                            
            else:
                # Lowest priority - other retrieved code
                if cross_file_used + item.token_estimate <= cross_file_tokens:
                    result.append(item)
                    cross_file_used += item.token_estimate
                else:
                    # Skip if over budget
                    pass
        
        logger.info(f"Token budget: current_file={current_file_used}/{current_file_tokens}, "
                   f"modified={modified_file_used}/{modified_file_tokens}, "
                   f"imports={import_used}/{import_tokens}, cross_file={cross_file_used}/{cross_file_tokens}")
        
        # DEBUG: Log relevance ordering
        if result:
            scores = [item.relevance_score for item in result]
            sources = [item.source for item in result]
            logger.info(f"DEBUG: {len(result)} items, scores min={min(scores):.3f}, max={max(scores):.3f}")
            logger.info(f"DEBUG: First 3 sources: {sources[:3]}, Last 3 sources: {sources[-3:]}")
            
            # Check if current file is at the end (highest relevance)
            current_file_positions = [i for i, item in enumerate(result) if item.source == 'current_file']
            if current_file_positions:
                logger.info(f"DEBUG: Current file entities at positions: {current_file_positions} (should be at end)")
        
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
    
    def _is_duplicate(self, item: ContextItem, seen_ranges: Dict[str, List[Tuple[int, int]]]) -> bool:
        """Check if entity is already covered by a parent class."""
        file_path = getattr(item.entity, 'file_path', '')
        line_start = getattr(item.entity, 'line_start', 0)
        line_end = getattr(item.entity, 'line_end', 0)
        
        if file_path in seen_ranges:
            for (start, end) in seen_ranges[file_path]:
                # If this entity is fully contained within a seen range, it's a duplicate
                if start <= line_start and line_end <= end:
                    return True
        return False
    
    def format_context(self, items: List[ContextItem]) -> str:
        """Format context items into a string."""
        if not items:
            return ""
        
        parts = []
        seen_ranges: Dict[str, List[Tuple[int, int]]] = {}  # Track covered line ranges
        
        # Sort items for ASCENDING relevance order (least relevant first)
        # This ensures most relevant survive left-truncation
        def get_sort_key(item):
            # Primary: source priority (HIGHER = more important = appears LATER in list)
            # We want: retrieval_seed (lowest) < import < modified_file < current_file (highest)
            if item.source == 'current_file':
                priority = 3  # HIGHEST - appears LAST
            elif item.source == 'modified_file':
                priority = 2
            elif item.source == 'import_definition':
                priority = 1
            else:
                priority = 0  # retrieval_seed - appears FIRST
            
            # Secondary: relevance score (lower score = less relevant = earlier in list)
            return (priority, item.relevance_score)
        
        sorted_items = sorted(items, key=get_sort_key)
        
        # Log ordering for debugging
        if sorted_items:
            scores = [item.relevance_score for item in sorted_items]
            sources = [item.source for item in sorted_items]
            logger.debug(f"Ordering: scores min={min(scores):.2f}, max={max(scores):.2f}")
            logger.debug(f"Sources order: {sources[:5]}...{sources[-5:]}")
        
        # Group by file for better organization (but maintain priority order)
        file_groups: Dict[str, List[ContextItem]] = {}
        file_order = []
        for item in sorted_items:
            # Skip duplicates (methods already in class)
            if self._is_duplicate(item, seen_ranges):
                logger.debug(f"Skipping duplicate: {item.entity_id}")
                continue
            
            # Track this entity's range
            file_path = getattr(item.entity, 'file_path', 'unknown')
            line_start = getattr(item.entity, 'line_start', 0)
            line_end = getattr(item.entity, 'line_end', 0)
            
            if file_path not in seen_ranges:
                seen_ranges[file_path] = []
            seen_ranges[file_path].append((line_start, line_end))
            
            if file_path not in file_groups:
                file_groups[file_path] = []
                file_order.append(file_path)
            file_groups[file_path].append(item)
        
        for file_path in file_order:
            file_items = file_groups[file_path]
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
            task: Task dictionary with prefix, suffix, path, modified, etc.
            graph: Code graph
            retriever: Hybrid retriever
        
        Returns:
            Formatted context string
        """
        prefix = task.get('prefix', '')
        suffix = task.get('suffix', '')
        current_file = task.get('path', '')
        modified_files = task.get('modified', [])  # Files changed in same commit
        
        # Build query
        query = self.query_builder.build_query(prefix, suffix)
        
        # Get current file entities from graph
        current_file_entities = None
        logger.info(f"DEBUG assemble_for_task: graph type = {type(graph)}, has get_entities_by_file = {hasattr(graph, 'get_entities_by_file')}")
        if hasattr(graph, 'get_entities_by_file'):
            current_file_entities = graph.get_entities_by_file(current_file)
            logger.info(f"DEBUG assemble_for_task: got {len(current_file_entities) if current_file_entities else 0} entities for {current_file}")
        
        # Assemble context with modified files
        context_items = self.assemble(
            query=query,
            current_file=current_file,
            current_file_entities=current_file_entities,
            prefix=prefix,
            suffix=suffix,
            modified_files=modified_files  # NEW: Pass modified files
        )
        
        return self.format_context(context_items)
