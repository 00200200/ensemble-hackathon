"""
Integration tests for the full pipeline.
"""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RepositoryProcessor, CompletionPipeline
from src.phase1_ast_parser import Entity
from src.graph_store import CodeGraph
from src.vector_store import HybridRetriever


class TestRepositoryProcessor:
    """Test repository processing."""
    
    def test_process_repository(self):
        """Test processing a simple repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            repo_dir = Path(tmpdir) / "test_repo"
            repo_dir.mkdir()
            
            # Create a Python file
            (repo_dir / "main.py").write_text('''
def hello():
    """Say hello."""
    print("Hello")

class Greeter:
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"
''')
            
            # Create another file
            (repo_dir / "utils.py").write_text('''
def helper():
    pass
''')
            
            processor = RepositoryProcessor(cache_dir=tmpdir)
            graph, retriever = processor.process_repository(
                repo_dir,
                "test/repo",
                "abc123"
            )
            
            assert len(graph) >= 3  # At least hello, Greeter, greet, helper
            assert isinstance(retriever, HybridRetriever)


class TestEndToEnd:
    """Test end-to-end pipeline."""
    
    def test_context_assembly(self):
        """Test full context assembly."""
        # Create a simple graph
        graph = CodeGraph()
        
        entities = [
            Entity(
                name='get_user',
                type='function',
                file_path='/repo/utils.py',
                line_start=1,
                line_end=5,
                signature='def get_user(id: int) -> User',
                raw_code='def get_user(id: int) -> User:\n    """Get user by ID."""\n    return User.query.get(id)',
                docstring='Get user by ID.'
            ),
            Entity(
                name='create_user',
                type='function',
                file_path='/repo/main.py',
                line_start=1,
                line_end=5,
                signature='def create_user(name: str) -> User',
                raw_code='def create_user(name: str) -> User:\n    """Create a new user."""\n    return User(name=name)',
                docstring='Create a new user.'
            ),
        ]
        
        for entity in entities:
            graph.add_entity(entity)
        
        # Create retriever and index
        retriever = HybridRetriever()
        retriever.index_entities(entities)
        
        # Create task
        task = {
            'id': 'test_001',
            'repo': 'test/repo',
            'revision': 'abc123',
            'path': '/repo/main.py',
            'prefix': 'def process_user(user_id):\n    user = ',
            'suffix': '\n    return user',
        }
        
        # Test context assembly
        from src.phase3_context_assembly import ContextAssembler
        
        assembler = ContextAssembler(
            graph=graph,
            retriever=retriever,
            max_tokens=2000
        )
        
        context = assembler.assemble_for_task(task, graph, retriever)
        
        # Context should contain relevant functions
        assert 'get_user' in context or 'create_user' in context
        assert '<|file_sep|>' in context


def run_quick_test():
    """Run a quick integration test."""
    print("Testing repository processor...")
    test = TestRepositoryProcessor()
    try:
        test.test_process_repository()
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("Testing end-to-end...")
    test = TestEndToEnd()
    try:
        test.test_context_assembly()
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    print("Running pipeline integration tests...\n")
    
    success = run_quick_test()
    
    if success:
        print("\n" + "="*50)
        print("All integration tests passed!")
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("Some tests failed!")
        sys.exit(1)
