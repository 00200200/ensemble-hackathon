"""
Tests for Phase 3 & 4: Context Assembly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase3_context_assembly import (
    QueryBuilder,
    ContextAssembler,
    ContextItem
)
from src.phase1_ast_parser import Entity
from src.graph_store import CodeGraph


class TestQueryBuilder:
    """Test query building."""
    
    def test_extract_identifiers_regex(self):
        """Test identifier extraction with regex fallback."""
        builder = QueryBuilder()
        
        code = '''
def get_user_by_id(user_id):
    """Get a user."""
    return db.query(User).filter(User.id == user_id).first()
'''
        identifiers = builder._extract_identifiers_regex(code)
        
        assert 'get_user_by_id' in identifiers
        assert 'user_id' in identifiers
        assert 'db' in identifiers
        assert 'query' in identifiers
        assert 'User' in identifiers
        assert 'filter' in identifiers
        assert 'id' in identifiers
        assert 'first' in identifiers
    
    def test_filter_keywords(self):
        """Test that keywords are filtered out."""
        builder = QueryBuilder()
        
        code = 'if foo and bar or not baz: return True'
        identifiers = builder._extract_identifiers_regex(code)
        
        assert 'if' not in identifiers
        assert 'and' not in identifiers
        assert 'or' not in identifiers
        assert 'not' not in identifiers
        assert 'return' not in identifiers
        assert 'True' not in identifiers
        assert 'foo' in identifiers
        assert 'bar' in identifiers
        assert 'baz' in identifiers
    
    def test_build_query(self):
        """Test query building from prefix/suffix."""
        builder = QueryBuilder()
        
        prefix = 'def calculate_sum(a, b):'
        suffix = '    return result'
        
        query = builder.build_query(prefix, suffix)
        
        # Should contain identifiers from both
        assert 'calculate_sum' in query or 'calculate' in query
        assert 'result' in query


class TestContextAssembler:
    """Test context assembly."""
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        assembler = ContextAssembler(graph=None, retriever=None)
        
        # Roughly 40 chars = 10 tokens
        text = "def hello():\n    print('world')"  # ~28 chars
        tokens = assembler.estimate_tokens(text)
        
        assert tokens > 0
        assert tokens == len(text) // 4
    
    def test_create_abstract(self):
        """Test abstract creation."""
        assembler = ContextAssembler(graph=None, retriever=None)
        
        entity = Entity(
            name='get_user',
            type='function',
            file_path='/test.py',
            line_start=1,
            line_end=10,
            signature='def get_user(id: int) -> User',
            raw_code='def get_user(id: int):\n    """Get user."""\n    return User.query.get(id)',
            docstring='Get user by ID.'
        )
        
        abstract = assembler._create_abstract(entity)
        
        assert 'def get_user(id: int) -> User' in abstract
        assert 'Get user by ID.' in abstract
    
    def test_apply_token_budget(self):
        """Test token budget application."""
        assembler = ContextAssembler(
            graph=None,
            retriever=None,
            max_tokens=1000
        )
        
        # Create items that exceed budget
        items = [
            ContextItem(
                entity=None, entity_id='1', source='current_file',
                relevance_score=1.0, code_to_show='x' * 2000,  # ~500 tokens
                token_estimate=500
            ),
            ContextItem(
                entity=None, entity_id='2', source='current_file',
                relevance_score=0.8, code_to_show='x' * 2000,
                token_estimate=500
            ),
            ContextItem(
                entity=None, entity_id='3', source='retrieval_seed',
                relevance_score=0.6, code_to_show='x' * 800,
                token_estimate=200
            ),
        ]
        
        result = assembler._apply_token_budget(items)
        
        # Should include items within budget
        total_tokens = sum(item.token_estimate for item in result)
        assert total_tokens <= assembler.max_tokens * 0.8  # Allow some margin
    
    def test_ascending_relevance_order(self):
        """Test that context is sorted by ascending relevance in assemble()."""
        # Create mock retriever and graph
        class MockRetriever:
            def search(self, query, k=10):
                return [
                    (Entity(name='high', type='function', file_path='/a.py',
                            line_start=1, line_end=2, signature='def high()', raw_code=''), 0.9),
                    (Entity(name='low', type='function', file_path='/b.py',
                            line_start=1, line_end=2, signature='def low()', raw_code=''), 0.1),
                    (Entity(name='medium', type='function', file_path='/c.py',
                            line_start=1, line_end=2, signature='def medium()', raw_code=''), 0.5),
                ]
        
        class MockGraph:
            def get_degree(self, entity_id):
                return 0
            def get_neighbors(self, entity_id, relation_types=None, depth=1):
                return []
        
        assembler = ContextAssembler(graph=MockGraph(), retriever=MockRetriever())
        
        items = assembler.assemble(
            query='test',
            current_file='/test.py',
            current_file_entities=None
        )
        
        # Should be sorted by ascending relevance
        for i in range(len(items) - 1):
            assert items[i].relevance_score <= items[i+1].relevance_score


class TestContextFormatting:
    """Test context formatting."""
    
    def test_format_context(self):
        """Test context formatting."""
        assembler = ContextAssembler(graph=None, retriever=None)
        
        entity1 = Entity(
            name='func1',
            type='function',
            file_path='/project/utils.py',
            line_start=1,
            line_end=5,
            signature='def func1():',
            raw_code='def func1():\n    pass'
        )
        
        entity2 = Entity(
            name='func2',
            type='function',
            file_path='/project/main.py',
            line_start=10,
            line_end=15,
            signature='def func2():',
            raw_code='def func2():\n    func1()'
        )
        
        items = [
            ContextItem(
                entity=entity1, entity_id='1', source='retrieval_seed',
                relevance_score=0.5, code_to_show='def func1():\n    pass',
                token_estimate=10
            ),
            ContextItem(
                entity=entity2, entity_id='2', source='retrieval_seed',
                relevance_score=0.8, code_to_show='def func2():\n    func1()',
                token_estimate=10
            ),
        ]
        
        formatted = assembler.format_context(items)
        
        # Check format
        assert '<|file_sep|>' in formatted
        assert '# File: utils.py' in formatted or '# File: main.py' in formatted
        assert '# function: func1' in formatted or '# function: func2' in formatted
        assert 'def func1():' in formatted
        assert 'def func2():' in formatted


if __name__ == '__main__':
    print("Running Phase 3 & 4 tests...")
    
    test_classes = [
        TestQueryBuilder(),
        TestContextAssembler(),
        TestContextFormatting(),
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
                    import traceback
                    traceback.print_exc()
                    failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
