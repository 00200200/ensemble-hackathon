"""
Phase 1: Structural Ingestion & AST Partitioning

Transforms raw repository into structured entities using AST parsing.
Supports both Tree-sitter (preferred) and Python native ast (fallback).
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from tree_sitter import Language, Parser, Tree
    from tree_sitter_language_pack import get_language, get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents a code entity (function, class, etc.)."""
    
    name: str
    type: str  # 'function', 'class', 'method', 'constant'
    file_path: str
    line_start: int
    line_end: int
    signature: str
    raw_code: str
    docstring: Optional[str] = None
    parent: Optional[str] = None  # For methods (class name)
    dependencies: List[str] = field(default_factory=list)  # Called functions/classes
    imports: List[str] = field(default_factory=list)  # Imported modules/names
    
    def __hash__(self) -> int:
        return hash((self.file_path, self.name, self.line_start))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return (self.file_path == other.file_path and 
                self.name == other.name and 
                self.line_start == other.line_start)
    
    @property
    def id(self) -> str:
        """Unique identifier for this entity."""
        return f"{self.file_path}:{self.name}:{self.line_start}"


@dataclass
class FileParseResult:
    """Result of parsing a single file."""
    
    file_path: str
    entities: List[Entity]
    imports: List[Dict[str, Any]]
    module_docstring: Optional[str] = None


class TreeSitterExtractor:
    """Extract entities using Tree-sitter (handles incomplete code)."""
    
    def __init__(self):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("Tree-sitter not available")
        try:
            self.language = get_language('python')
            self.parser = get_parser('python')
        except Exception as e:
            logger.warning(f"Failed to load tree-sitter language: {e}")
            raise
    
    def parse_file(self, file_path: Path, content: str) -> FileParseResult:
        """Parse a file and extract entities."""
        tree = self.parser.parse(content.encode('utf-8'))
        root_node = tree.root_node
        
        entities = []
        imports = []
        module_docstring = None
        
        # Extract module-level docstring
        module_docstring = self._extract_module_docstring(root_node, content)
        
        # Extract imports
        imports = self._extract_imports(root_node)
        
        # Extract functions and classes
        for node in root_node.children:
            if node.type == 'function_definition':
                entity = self._extract_function(node, content, str(file_path))
                if entity:
                    entities.append(entity)
            elif node.type == 'class_definition':
                class_entity, method_entities = self._extract_class(
                    node, content, str(file_path)
                )
                if class_entity:
                    entities.append(class_entity)
                    entities.extend(method_entities)
        
        return FileParseResult(
            file_path=str(file_path),
            entities=entities,
            imports=imports,
            module_docstring=module_docstring
        )
    
    def _extract_module_docstring(self, root_node, content: str) -> Optional[str]:
        """Extract module-level docstring."""
        for child in root_node.children:
            if child.type == 'expression_statement':
                expr = child.child_by_field_name('expression')
                if expr and expr.type == 'string':
                    return self._get_node_text(expr, content)
        return None
    
    def _extract_imports(self, root_node) -> List[Dict[str, Any]]:
        """Extract import statements."""
        imports = []
        
        def traverse(node):
            if node.type == 'import_statement':
                # import x, y, z
                names = []
                for child in node.children:
                    if child.type == 'dotted_name':
                        name = self._get_dotted_name(child)
                        names.append({'name': name, 'alias': None})
                    elif child.type == 'aliased_import':
                        name = self._get_dotted_name(child.child_by_field_name('name'))
                        alias_node = child.child_by_field_name('alias')
                        alias = self._get_node_text(alias_node, '') if alias_node else None
                        names.append({'name': name, 'alias': alias})
                imports.append({'type': 'import', 'names': names})
            
            elif node.type == 'import_from_statement':
                # from x import y, z
                module_node = node.child_by_field_name('module')
                module = self._get_dotted_name(module_node) if module_node else ''
                names = []
                for child in node.children:
                    if child.type == 'import_list':
                        for name_node in child.children:
                            if name_node.type == 'identifier':
                                names.append({
                                    'name': self._get_node_text(name_node, ''),
                                    'alias': None
                                })
                imports.append({'type': 'from', 'module': module, 'names': names})
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return imports
    
    def _extract_function(self, node, content: str, file_path: str) -> Optional[Entity]:
        """Extract a function entity."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, content)
        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1
        raw_code = self._get_node_text(node, content)
        signature = self._extract_signature(node, content)
        docstring = self._extract_docstring(node, content)
        
        # Extract function calls
        dependencies = self._extract_calls(node, content)
        
        return Entity(
            name=name,
            type='function',
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            signature=signature,
            raw_code=raw_code,
            docstring=docstring,
            dependencies=dependencies
        )
    
    def _extract_class(self, node, content: str, file_path: str) -> Tuple[Optional[Entity], List[Entity]]:
        """Extract a class and its methods."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None, []
        
        name = self._get_node_text(name_node, content)
        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1
        raw_code = self._get_node_text(node, content)
        signature = self._extract_class_signature(node, content)
        docstring = self._extract_docstring(node, content)
        
        class_entity = Entity(
            name=name,
            type='class',
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            signature=signature,
            raw_code=raw_code,
            docstring=docstring
        )
        
        # Extract methods
        methods = []
        body = node.child_by_field_name('body')
        if body:
            for child in body.children:
                if child.type == 'function_definition':
                    method = self._extract_method(child, content, file_path, name)
                    if method:
                        methods.append(method)
        
        return class_entity, methods
    
    def _extract_method(self, node, content: str, file_path: str, class_name: str) -> Optional[Entity]:
        """Extract a method entity."""
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, content)
        line_start = node.start_point[0] + 1
        line_end = node.end_point[0] + 1
        raw_code = self._get_node_text(node, content)
        signature = self._extract_signature(node, content)
        docstring = self._extract_docstring(node, content)
        dependencies = self._extract_calls(node, content)
        
        return Entity(
            name=name,
            type='method',
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            signature=signature,
            raw_code=raw_code,
            docstring=docstring,
            parent=class_name,
            dependencies=dependencies
        )
    
    def _extract_signature(self, node, content: str) -> str:
        """Extract function signature."""
        name_node = node.child_by_field_name('name')
        params_node = node.child_by_field_name('parameters')
        return_type_node = node.child_by_field_name('return_type')
        
        parts = ['def']
        if name_node:
            parts.append(self._get_node_text(name_node, content))
        if params_node:
            parts.append(self._get_node_text(params_node, content))
        if return_type_node:
            parts.append('->')
            parts.append(self._get_node_text(return_type_node, content))
        
        return ' '.join(parts)
    
    def _extract_class_signature(self, node, content: str) -> str:
        """Extract class signature."""
        name_node = node.child_by_field_name('name')
        superclass_node = node.child_by_field_name('superclasses')
        
        parts = ['class']
        if name_node:
            parts.append(self._get_node_text(name_node, content))
        if superclass_node:
            parts.append('(' + self._get_node_text(superclass_node, content) + ')')
        
        return ' '.join(parts)
    
    def _extract_docstring(self, node, content: str) -> Optional[str]:
        """Extract docstring from function/class body."""
        body = node.child_by_field_name('body')
        if not body or not body.children:
            return None
        
        first_stmt = body.children[0]
        if first_stmt.type == 'expression_statement':
            expr = first_stmt.child_by_field_name('expression')
            if expr and expr.type == 'string':
                return self._get_node_text(expr, content)
        return None
    
    def _extract_calls(self, node, content: str) -> List[str]:
        """Extract function calls from a node."""
        calls = []
        
        def traverse(n):
            if n.type == 'call':
                func_node = n.child_by_field_name('function')
                if func_node:
                    call_name = self._get_call_name(func_node, content)
                    if call_name:
                        calls.append(call_name)
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return list(set(calls))  # Deduplicate
    
    def _get_call_name(self, node, content: str) -> str:
        """Get the name of a function call."""
        if node.type == 'identifier':
            return self._get_node_text(node, content)
        elif node.type == 'attribute':
            obj = node.child_by_field_name('object')
            attr = node.child_by_field_name('attribute')
            if obj and attr:
                obj_text = self._get_node_text(obj, content)
                attr_text = self._get_node_text(attr, content)
                return f"{obj_text}.{attr_text}"
        elif node.type == ' dotted_name':
            return self._get_node_text(node, content)
        return ""
    
    def _get_dotted_name(self, node) -> str:
        """Get dotted name from node."""
        if node is None:
            return ''
        parts = []
        for child in node.children:
            if child.type == 'identifier':
                parts.append(child.text.decode('utf-8') if child.text else '')
            elif child.type == '.':
                continue
            elif child.is_named:
                parts.append(self._get_dotted_name(child))
        return '.'.join(parts)
    
    def _get_node_text(self, node, content: str) -> str:
        """Get text content of a node."""
        if node is None:
            return ''
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        return content[node.start_byte:node.end_byte]


class NativeASTExtractor:
    """Extract entities using Python's native ast module (fallback)."""
    
    def parse_file(self, file_path: Path, content: str) -> FileParseResult:
        """Parse a file and extract entities."""
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return FileParseResult(
                file_path=str(file_path),
                entities=[],
                imports=[]
            )
        
        entities = []
        imports = []
        module_docstring = ast.get_docstring(tree)
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'name': alias.name,
                        'alias': alias.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'type': 'from',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname
                    })
        
        # Extract classes and functions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                entity = self._extract_function(node, content, str(file_path), None)
                entities.append(entity)
            elif isinstance(node, ast.ClassDef):
                class_entity, methods = self._extract_class(node, content, str(file_path))
                entities.append(class_entity)
                entities.extend(methods)
        
        return FileParseResult(
            file_path=str(file_path),
            entities=entities,
            imports=imports,
            module_docstring=module_docstring
        )
    
    def _extract_function(self, node: ast.FunctionDef, content: str, 
                          file_path: str, class_name: Optional[str]) -> Entity:
        """Extract a function entity."""
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno
        
        # Get raw code
        lines = content.split('\n')
        raw_lines = lines[line_start - 1:line_end]
        raw_code = '\n'.join(raw_lines)
        
        # Get signature
        signature = self._get_signature(node)
        docstring = ast.get_docstring(node)
        
        # Extract dependencies (function calls)
        dependencies = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._get_call_name(child.func)
                if call_name:
                    dependencies.append(call_name)
        
        return Entity(
            name=node.name,
            type='method' if class_name else 'function',
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            signature=signature,
            raw_code=raw_code,
            docstring=docstring,
            parent=class_name,
            dependencies=list(set(dependencies))
        )
    
    def _extract_class(self, node: ast.ClassDef, content: str, 
                       file_path: str) -> Tuple[Entity, List[Entity]]:
        """Extract a class and its methods."""
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno
        
        lines = content.split('\n')
        raw_lines = lines[line_start - 1:line_end]
        raw_code = '\n'.join(raw_lines)
        
        # Get signature
        bases = [self._get_name(base) for base in node.bases]
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"
        
        docstring = ast.get_docstring(node)
        
        class_entity = Entity(
            name=node.name,
            type='class',
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            signature=signature,
            raw_code=raw_code,
            docstring=docstring
        )
        
        # Extract methods
        methods = []
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                method = self._extract_function(child, content, file_path, node.name)
                methods.append(method)
        
        return class_entity, methods
    
    def _get_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        """Get function signature."""
        args = []
        
        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # Defaults
        defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults)
        
        # Varargs
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        # Kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        sig = f"def {node.name}({', '.join(args)})"
        if node.returns:
            sig += f" -> {ast.unparse(node.returns)}"
        
        return sig
    
    def _get_call_name(self, node: ast.AST) -> str:
        """Get name from a call node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_call_name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        return ""
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return ""


class ASTParser:
    """Main AST parser with automatic fallback."""
    
    def __init__(self, use_tree_sitter: bool = True):
        self.extractor = None
        
        if use_tree_sitter and TREE_SITTER_AVAILABLE:
            try:
                self.extractor = TreeSitterExtractor()
                logger.info("Using Tree-sitter parser")
            except Exception as e:
                logger.warning(f"Failed to initialize Tree-sitter: {e}")
        
        if self.extractor is None:
            self.extractor = NativeASTExtractor()
            logger.info("Using native Python AST parser")
    
    def parse_file(
        self, 
        file_path: Union[str, Path],
        base_path: Optional[Union[str, Path]] = None
    ) -> FileParseResult:
        """Parse a single file.
        
        Args:
            file_path: Path to the file to parse
            base_path: If provided, store relative paths in entities (for cache portability)
        """
        file_path = Path(file_path)
        base_path = Path(base_path) if base_path else None
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content = file_path.read_text(encoding='utf-8', errors='replace')
        
        # Parse with absolute path first
        result = self.extractor.parse_file(file_path, content)
        
        # Convert to relative path if base_path provided
        if base_path:
            try:
                rel_path = str(file_path.relative_to(base_path))
                # Update the result with relative path
                result.file_path = rel_path
                # Update all entities
                for entity in result.entities:
                    entity.file_path = rel_path
            except ValueError:
                # file_path is not under base_path, keep absolute
                pass
        
        return result
    
    def parse_directory(
        self, 
        dir_path: Union[str, Path],
        exclude_patterns: Optional[Set[str]] = None
    ) -> List[FileParseResult]:
        """Parse all Python files in a directory.
        
        Stores relative paths in entities for cache portability across different
        extraction directories.
        """
        dir_path = Path(dir_path).resolve()
        exclude_patterns = exclude_patterns or {
            '__pycache__', '.git', 'venv', '.venv', 'node_modules',
            '.tox', '.pytest_cache', '.mypy_cache', '*.egg-info'
        }
        
        results = []
        py_files = list(dir_path.rglob('*.py'))
        
        logger.info(f"Found {len(py_files)} Python files in {dir_path}")
        
        for i, file_path in enumerate(py_files):
            # Skip excluded patterns
            if any(p in file_path.parts for p in exclude_patterns):
                continue
            
            try:
                # Parse with relative paths (base_path=dir_path)
                result = self.parse_file(file_path, base_path=dir_path)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")
            
            if (i + 1) % 100 == 0:
                logger.info(f"Parsed {i + 1}/{len(py_files)} files...")
        
        logger.info(f"Successfully parsed {len(results)} files")
        return results


class ParseCache:
    """Cache for parsed AST results with MD5 invalidation."""
    
    def __init__(self, cache_dir: Union[str, Path] = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of file content."""
        content = file_path.read_bytes()
        return hashlib.md5(content).hexdigest()
    
    def _get_cache_path(self, file_path: Path) -> Path:
        """Get cache file path for a source file."""
        file_hash = self._get_file_hash(file_path)
        cache_name = f"{file_path.stem}_{file_hash[:16]}.pkl"
        return self.cache_dir / cache_name
    
    def get(self, file_path: Union[str, Path]) -> Optional[FileParseResult]:
        """Get cached parse result if available and valid."""
        file_path = Path(file_path)
        cache_path = self._get_cache_path(file_path)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def set(self, file_path: Union[str, Path], result: FileParseResult) -> None:
        """Cache a parse result."""
        file_path = Path(file_path)
        cache_path = self._get_cache_path(file_path)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def parse_with_cache(
        self, 
        file_path: Union[str, Path],
        parser: ASTParser
    ) -> FileParseResult:
        """Parse file with caching."""
        file_path = Path(file_path)
        
        # Try cache first
        cached = self.get(file_path)
        if cached:
            return cached
        
        # Parse and cache
        result = parser.parse_file(file_path)
        self.set(file_path, result)
        return result
