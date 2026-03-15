"""
Tests for Phase 2: Vector Store & Hybrid Retrieval
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.vector_store import (
    tokenize_code,
    SimpleEmbedder,
    CodeEmbedder,
    SimpleBM25,
    DenseVectorStore,
    SparseVectorStore,
    HybridRetriever
)
from src.phase1_ast_parser import Entity


class TestTokenization:
    """Test code tokenization."""
    
    def test_camel_case_split(self):
        """Test splitting camelCase identifiers."""
        tokens = tokenize_code("getUserById")
        assert 'get' in tokens
        assert 'user' in tokens
        assert 'by' in tokens
        assert 'id' in tokens
    
    def test_pascal_case_split(self):
        """Test splitting PascalCase identifiers."""
        tokens = tokenize_code("UserAuthentication")
        assert 'user' in tokens
        assert 'authentication' in tokens
    
    def test_snake_case_split(self):
        """Test splitting snake_case identifiers."""
        tokens = tokenize_code("get_user_by_id")
        assert 'get' in tokens
        assert 'user' in tokens
        assert 'by' in tokens
        assert 'id' in tokens
    
    def test_mixed_case(self):
        """Test splitting mixed case identifiers."""
        tokens = tokenize_code("get_userProfile_byID")
        assert 'get' in tokens
        assert 'user' in tokens
        assert 'profile' in tokens
        assert 'by' in tokens
        assert 'id' in tokens


class TestSimpleEmbedder:
    """Test simple embedder fallback."""
    
    def test_embedding_shape(self):
        """Test embedding output shape."""
        embedder = SimpleEmbedder(dimension=128)
        texts = ["def hello(): pass", "class World: pass"]
        embeddings = embedder.encode(texts)
        
        assert embeddings.shape == (2, 128)
    
    def test_embedding_normalization(self):
        """Test that embeddings are normalized."""
        embedder = SimpleEmbedder(dimension=128)
        texts = ["def hello(): pass"]
        embeddings = embedder.encode(texts)
        
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 0.01  # Should be normalized


class TestCodeEmbedder:
    """Test code embedder."""
    
    def test_prepare_entity_text(self):
        """Test entity text preparation."""
        embedder = CodeEmbedder()
        
        entity = Entity(
            name='get_user',
            type='function',
            file_path='/test.py',
            line_start=1,
            line_end=5,
            signature='def get_user(id: int) -> User',
            raw_code='def get_user(id: int):\n    return User(id)',
            docstring='Get a user by ID.'
        )
        
        text = embedder.prepare_entity_text(entity)
        assert 'get_user' in text
        assert 'function' in text
        assert 'def get_user(id: int) -> User' in text
        assert 'Get a user by ID.' in text


class TestSimpleBM25:
    """Test simple BM25 implementation."""
    
    def test_basic_scoring(self):
        """Test basic BM25 scoring."""
        corpus = [
            ["hello", "world"],
            ["hello", "python", "code"],
            ["python", "programming", "tutorial"],
        ]
        
        bm25 = SimpleBM25(corpus)
        scores = bm25.get_scores(["python"])
        
        assert len(scores) == 3
        # Document 1 and 2 contain "python"
        assert scores[1] > 0
        assert scores[2] > 0
        # Document 0 doesn't contain "python"
        assert scores[0] == 0
    
    def test_term_frequency_effect(self):
        """Test that term frequency affects score."""
        corpus = [
            ["python"],
            ["python", "python", "python"],
        ]
        
        bm25 = SimpleBM25(corpus)
        scores = bm25.get_scores(["python"])
        
        # Document with more occurrences should have higher score
        assert scores[1] > scores[0]


class TestDenseVectorStore:
    """Test dense vector store."""
    
    def test_add_and_search(self):
        """Test adding entities and searching."""
        store = DenseVectorStore(dimension=128)
        
        entities = [
            Entity(name=f'func_{i}', type='function', file_path='/test.py',
                   line_start=i, line_end=i+1, signature=f'def func_{i}()',
                   raw_code=f'def func_{i}(): pass')
            for i in range(10)
        ]
        
        # Random embeddings
        embeddings = np.random.randn(10, 128).astype(np.float32)
        
        store.add(entities, embeddings)
        
        # Search
        query = embeddings[0]  # Query with first entity
        results = store.search(query, k=3)
        
        assert len(results) == 3
        # First result should be the query entity itself
        assert results[0][0] == 0
        assert results[0][2].name == 'func_0'


class TestSparseVectorStore:
    """Test sparse vector store."""
    
    def test_add_and_search(self):
        """Test adding entities and searching."""
        store = SparseVectorStore()
        
        entities = [
            Entity(name='get_user', type='function', file_path='/test.py',
                   line_start=1, line_end=5, signature='def get_user(id)',
                   raw_code='def get_user(id): return User(id)'),
            Entity(name='create_post', type='function', file_path='/test.py',
                   line_start=10, line_end=15, signature='def create_post(title)',
                   raw_code='def create_post(title): return Post(title)'),
            Entity(name='delete_comment', type='function', file_path='/test.py',
                   line_start=20, line_end=25, signature='def delete_comment(id)',
                   raw_code='def delete_comment(id): Comment.delete(id)'),
        ]
        
        store.add(entities)
        
        # Search for user-related function
        results = store.search("get user", k=2)
        
        assert len(results) > 0
        # First result should be get_user
        assert results[0][2].name == 'get_user'


class TestHybridRetriever:
    """Test hybrid retriever."""
    
    def test_index_and_search(self):
        """Test indexing and searching."""
        retriever = HybridRetriever(alpha=0.5)
        
        entities = [
            Entity(name='get_user', type='function', file_path='/test.py',
                   line_start=1, line_end=5, signature='def get_user(id: int) -> User',
                   raw_code='def get_user(id): return User(id)',
                   docstring='Retrieve a user by their ID.'),
            Entity(name='create_user', type='function', file_path='/test.py',
                   line_start=10, line_end=15, signature='def create_user(name: str) -> User',
                   raw_code='def create_user(name): return User(name)',
                   docstring='Create a new user.'),
            Entity(name='delete_post', type='function', file_path='/test.py',
                   line_start=20, line_end=25, signature='def delete_post(id: int)',
                   raw_code='def delete_post(id): Post.delete(id)',
                   docstring='Delete a blog post.'),
        ]
        
        retriever.index_entities(entities)
        
        # Search for user-related
        results = retriever.search("get user by id", k=2)
        
        assert len(results) > 0
        # Should find get_user function
        found_names = [r[0].name for r in results]
        assert 'get_user' in found_names
    
    def test_alpha_tuning_identifiers(self):
        """Test alpha tuning for identifier-heavy queries."""
        retriever = HybridRetriever()
        
        # Query with many identifiers
        alpha = retriever.auto_tune_alpha("getUserByIdAndName")
        assert alpha < 0.3  # Should prefer BM25
        
        # Conceptual query
        alpha = retriever.auto_tune_alpha("how to handle errors")
        assert alpha > 0.5  # Should prefer dense


if __name__ == '__main__':
    print("Running Phase 2 tests...")
    
    test_classes = [
        TestTokenization(),
        TestSimpleEmbedder(),
        TestCodeEmbedder(),
        TestSimpleBM25(),
        TestDenseVectorStore(),
        TestSparseVectorStore(),
        TestHybridRetriever(),
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
