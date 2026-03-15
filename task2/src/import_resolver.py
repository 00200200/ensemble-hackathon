"""
Import Resolution for Context Enhancement

Resolves imports in the current file to fetch definitions of used symbols.
This is the highest-impact upgrade according to research (STALL+ paper).
"""

from __future__ import annotations

import ast
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ImportParser:
    """Parse imports from code."""
    
    @staticmethod
    def extract_imports(code: str) -> List[Dict[str, Any]]:
        """Extract all imports from code."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'name': alias.name,
                        'alias': alias.asname,
                        'full_name': alias.asname or alias.name
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'type': 'from',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'full_name': alias.asname or alias.name,
                        'level': node.level  # Relative import level (0=absolute)
                    })
        
        return imports


class ImportResolver:
    """Resolve imports to their definitions."""
    
    def __init__(self, graph: Any):
        self.graph = graph
    
    def resolve_imports(
        self,
        imports: List[Dict[str, Any]],
        current_file: str,
        max_definitions: int = 10
    ) -> List[Any]:
        """
        Resolve imports to their definitions in the graph.
        
        Args:
            imports: List of import statements from ImportParser
            current_file: Path to current file (for relative imports)
            max_definitions: Maximum definitions to return
        
        Returns:
            List of entity definitions
        """
        definitions = []
        seen = set()
        
        for imp in imports:
            # Try to resolve this import
            entity = self._resolve_single_import(imp, current_file)
            
            if entity and entity not in seen:
                definitions.append(entity)
                seen.add(entity)
                
                if len(definitions) >= max_definitions:
                    break
        
        return definitions
    
    def _resolve_single_import(
        self,
        imp: Dict[str, Any],
        current_file: str
    ) -> Optional[Any]:
        """Resolve a single import to its definition."""
        name = imp.get('name', '')
        module = imp.get('module', '')
        imp_type = imp.get('type', '')
        
        if imp_type == 'from':
            # from module import name
            # Look for: module.name or just name
            candidates = []
            
            # Try fully qualified name
            if module:
                full_name = f"{module}.{name}"
                candidates.extend(self._find_entities_by_name(full_name))
            
            # Try just the name
            candidates.extend(self._find_entities_by_name(name))
            
            # Prefer entities from imported modules
            for entity in candidates:
                entity_module = getattr(entity, 'file_path', '')
                if module.replace('.', '/') in entity_module:
                    return entity
            
            # Return first match
            if candidates:
                return candidates[0]
        
        else:
            # import module
            # For now, skip whole module imports (too broad)
            # Could return module-level docstring or __init__.py
            pass
        
        return None
    
    def _find_entities_by_name(self, name: str) -> List[Any]:
        """Find entities by name in the graph."""
        if not hasattr(self.graph, 'get_entities_by_name'):
            return []
        
        entities = self.graph.get_entities_by_name(name)
        return entities if entities else []
    
    def resolve_symbols_from_code(
        self,
        code: str,
        current_file: str,
        max_definitions: int = 10
    ) -> List[Any]:
        """
        Convenience method: extract imports from code and resolve them.
        
        Args:
            code: Source code (prefix of current file)
            current_file: Path to current file
            max_definitions: Maximum definitions to return
        
        Returns:
            List of resolved entity definitions
        """
        imports = ImportParser.extract_imports(code)
        
        if not imports:
            return []
        
        logger.info(f"Resolving {len(imports)} imports from {current_file}")
        
        definitions = self.resolve_imports(
            imports,
            current_file,
            max_definitions
        )
        
        logger.info(f"Resolved {len(definitions)} import definitions")
        
        return definitions


def enhance_context_with_imports(
    context_items: List[Any],
    prefix: str,
    current_file: str,
    graph: Any,
    max_import_definitions: int = 5
) -> List[Any]:
    """
    Enhance context with resolved import definitions.
    
    This is the main entry point - add import definitions to context.
    
    Args:
        context_items: Existing context items
        prefix: Code prefix (contains imports)
        current_file: Current file path
        graph: Code graph
        max_import_definitions: Max definitions to add
    
    Returns:
        Enhanced context items with import definitions
    """
    resolver = ImportResolver(graph)
    
    # Resolve imports from prefix
    import_definitions = resolver.resolve_symbols_from_code(
        prefix,
        current_file,
        max_import_definitions
    )
    
    # Add to context (avoid duplicates)
    seen_ids = {getattr(item, 'id', str(item)) for item in context_items}
    
    for entity in import_definitions:
        entity_id = getattr(entity, 'id', str(entity))
        if entity_id not in seen_ids:
            # Wrap in ContextItem-like object
            context_items.append(entity)
            seen_ids.add(entity_id)
    
    return context_items
