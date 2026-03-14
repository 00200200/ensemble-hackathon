"""
Tests for Phase 1: AST Parsing & Graph Store
"""

import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase1_ast_parser import (
    ASTParser, 
    NativeASTExtractor, 
    ParseCache,
    Entity,
    FileParseResult
)
from src.graph_store import CodeGraph


class TestNativeASTExtractor:
    """Test the native Python AST extractor."""
    
    def test_simple_function(self):
        """Test extracting a simple function."""
        extractor = NativeASTExtractor()
        code = '''
def hello_world():
    """Say hello."""
    print("Hello, World!")
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extractor.parse_file(Path(f.name), code)
        
        assert len(result.entities) == 1
        entity = result.entities[0]
        assert entity.name == 'hello_world'
        assert entity.type == 'function'
        assert 'hello' in (entity.docstring or '').lower()
    
    def test_class_with_methods(self):
        """Test extracting a class with methods."""
        extractor = NativeASTExtractor()
        code = '''
class Calculator:
    """A simple calculator."""
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    def subtract(self, a: int, b: int) -> int:
        return a - b
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extractor.parse_file(Path(f.name), code)
        
        # Should have 1 class + 2 methods = 3 entities
        assert len(result.entities) == 3
        
        class_entity = [e for e in result.entities if e.type == 'class'][0]
        assert class_entity.name == 'Calculator'
        
        methods = [e for e in result.entities if e.type == 'method']
        assert len(methods) == 2
        assert any(m.name == 'add' for m in methods)
        assert any(m.name == 'subtract' for m in methods)
    
    def test_import_extraction(self):
        """Test extracting import statements."""
        extractor = NativeASTExtractor()
        code = '''
import os
import sys as system
from pathlib import Path
from typing import List, Dict
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extractor.parse_file(Path(f.name), code)
        
        # Should have 5 imports: os, sys, Path, List, Dict
        assert len(result.imports) == 5
        
        # Check regular imports
        regular_imports = [i for i in result.imports if i['type'] == 'import']
        assert len(regular_imports) == 2
        
        # Check from imports
        from_imports = [i for i in result.imports if i['type'] == 'from']
        assert len(from_imports) == 3
    
    def test_dependency_extraction(self):
        """Test extracting function call dependencies."""
        extractor = NativeASTExtractor()
        code = '''
def outer():
    result = inner()
    print(result)
    return helper(result)

def inner():
    return 42

def helper(x):
    return x * 2
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = extractor.parse_file(Path(f.name), code)
        
        outer = [e for e in result.entities if e.name == 'outer'][0]
        assert 'inner' in outer.dependencies
        assert 'helper' in outer.dependencies


class TestCodeGraph:
    """Test the code graph store."""
    
    def test_add_entity(self):
        """Test adding entities to the graph."""
        graph = CodeGraph()
        
        entity = Entity(
            name='test_func',
            type='function',
            file_path='/test.py',
            line_start=1,
            line_end=5,
            signature='def test_func()',
            raw_code='def test_func():\n    pass'
        )
        
        entity_id = graph.add_entity(entity)
        assert graph.graph.has_node(entity_id)
        assert len(graph) == 1
    
    def test_add_relationship(self):
        """Test adding relationships between entities."""
        graph = CodeGraph()
        
        entity1 = Entity(
            name='caller',
            type='function',
            file_path='/test.py',
            line_start=1,
            line_end=5,
            signature='def caller()',
            raw_code='def caller():\n    callee()'
        )
        
        entity2 = Entity(
            name='callee',
            type='function',
            file_path='/test.py',
            line_start=7,
            line_end=10,
            signature='def callee()',
            raw_code='def callee():\n    pass'
        )
        
        id1 = graph.add_entity(entity1)
        id2 = graph.add_entity(entity2)
        
        graph.add_relationship(id1, id2, 'CALLS')
        
        assert graph.graph.has_edge(id1, id2)
        
        callees = graph.get_callees(id1)
        assert len(callees) == 1
        assert callees[0][1].name == 'callee'
    
    def test_high_degree_detection(self):
        """Test detecting high-degree nodes."""
        graph = CodeGraph()
        
        # Create a utility function called by many others
        utility = Entity(
            name='utility',
            type='function',
            file_path='/test.py',
            line_start=1,
            line_end=5,
            signature='def utility()',
            raw_code='def utility():\n    pass'
        )
        utility_id = graph.add_entity(utility)
        
        # Add 25 callers
        for i in range(25):
            caller = Entity(
                name=f'caller_{i}',
                type='function',
                file_path='/test.py',
                line_start=10+i,
                line_end=15+i,
                signature=f'def caller_{i}()',
                raw_code=f'def caller_{i}():\n    utility()'
            )
            caller_id = graph.add_entity(caller)
            graph.add_relationship(caller_id, utility_id, 'CALLS')
        
        assert graph.is_high_degree(utility_id, threshold=20)
        assert graph.get_degree(utility_id) == 25
    
    def test_get_neighbors(self):
        """Test getting neighbors at different depths."""
        graph = CodeGraph()
        
        # Create chain: A -> B -> C -> D
        entities = []
        for name in ['A', 'B', 'C', 'D']:
            entity = Entity(
                name=name,
                type='function',
                file_path='/test.py',
                line_start=len(entities)*10,
                line_end=len(entities)*10+5,
                signature=f'def {name}()',
                raw_code=f'def {name}():\n    pass'
            )
            entity_id = graph.add_entity(entity)
            entities.append(entity_id)
        
        # Create chain
        for i in range(len(entities) - 1):
            graph.add_relationship(entities[i], entities[i+1], 'CALLS')
        
        # Test depth 1
        neighbors = graph.get_neighbors(entities[0], depth=1)
        assert len(neighbors) == 1
        assert neighbors[0][1].name == 'B'
        
        # Test depth 2
        neighbors = graph.get_neighbors(entities[0], depth=2)
        assert len(neighbors) == 2
        names = [n[1].name for n in neighbors]
        assert 'B' in names and 'C' in names
    
    def test_save_load(self):
        """Test saving and loading the graph."""
        graph = CodeGraph()
        
        entity = Entity(
            name='test',
            type='function',
            file_path='/test.py',
            line_start=1,
            line_end=5,
            signature='def test()',
            raw_code='def test():\n    pass'
        )
        graph.add_entity(entity)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        
        graph.save(path)
        
        # Load into new graph
        new_graph = CodeGraph()
        new_graph.load(path)
        
        assert len(new_graph) == len(graph)


class TestParseCache:
    """Test the parse cache."""
    
    def test_cache_hit_miss(self):
        """Test cache hit and miss behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParseCache(cache_dir=tmpdir)
            parser = ASTParser(use_tree_sitter=False)
            
            code = 'def test():\n    pass'
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                path = Path(f.name)
            
            # First parse - should miss
            result1 = cache.parse_with_cache(path, parser)
            assert cache.get(path) is not None
            
            # Second parse - should hit
            result2 = cache.get(path)
            assert result2 is not None
            assert len(result2.entities) == len(result1.entities)


if __name__ == '__main__':
    # Run tests
    import sys
    
    print("Running Phase 1 tests...")
    
    test_classes = [
        TestNativeASTExtractor(),
        TestCodeGraph(),
        TestParseCache()
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}:")
        
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                print(f"  {method_name}...", end=' ')
                try:
                    getattr(test_class, method_name)()
                    print("PASSED")
                    passed += 1
                except Exception as e:
                    print(f"FAILED: {e}")
                    failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
