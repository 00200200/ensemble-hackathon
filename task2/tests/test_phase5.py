"""
Tests for Phase 5: Style Engine (KL-Gated)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase5_style_engine import (
    StyleProfile,
    StyleAnalyzer,
    KLDivergenceCalculator,
    StyleGatedPrompt
)
from src.phase1_ast_parser import Entity


class TestStyleAnalyzer:
    """Test style analysis."""
    
    def test_naming_detection_snake(self):
        """Test snake_case detection."""
        analyzer = StyleAnalyzer()
        
        assert analyzer._detect_naming_convention('get_user_by_id') == 'snake'
        assert analyzer._detect_naming_convention('process_data') == 'snake'
    
    def test_naming_detection_camel(self):
        """Test camelCase detection."""
        analyzer = StyleAnalyzer()
        
        assert analyzer._detect_naming_convention('getUserById') == 'camel'
        assert analyzer._detect_naming_convention('processData') == 'camel'
    
    def test_naming_detection_pascal(self):
        """Test PascalCase detection."""
        analyzer = StyleAnalyzer()
        
        assert analyzer._detect_naming_convention('UserManager') == 'pascal'
        assert analyzer._detect_naming_convention('DataProcessor') == 'pascal'
    
    def test_type_hint_detection(self):
        """Test type hint detection."""
        analyzer = StyleAnalyzer()
        
        assert analyzer._has_type_hints('def foo(x: int) -> str:') == True
        assert analyzer._has_type_hints('def foo(x):') == False
        assert analyzer._has_type_hints('') == False
    
    def test_docstring_style_google(self):
        """Test Google style docstring detection."""
        analyzer = StyleAnalyzer()
        
        doc = '''Does something.
        
        Args:
            x: The input
        
        Returns:
            The output
        '''
        assert analyzer._detect_docstring_style(doc) == 'google'
    
    def test_docstring_style_rest(self):
        """Test reST style docstring detection."""
        analyzer = StyleAnalyzer()
        
        doc = '''Does something.
        
        :param x: The input
        :return: The output
        '''
        assert analyzer._detect_docstring_style(doc) == 'rest'
    
    def test_analyze_entities(self):
        """Test analyzing multiple entities."""
        analyzer = StyleAnalyzer()
        
        entities = [
            Entity(
                name='get_user',
                type='function',
                file_path='/test.py',
                line_start=1,
                line_end=5,
                signature='def get_user(id: int) -> User',
                raw_code='def get_user(id: int) -> User:\n    """Get user."""\n    pass',
                docstring='Get user.'
            ),
            Entity(
                name='create_post',
                type='function',
                file_path='/test.py',
                line_start=10,
                line_end=15,
                signature='def create_post(title: str) -> Post',
                raw_code='def create_post(title: str) -> Post:\n    """Create post."""\n    pass',
                docstring='Create post.'
            ),
        ]
        
        profile = analyzer.analyze_entities(entities)
        
        assert profile.sample_size == 2
        assert profile.naming_snake_case == 1.0  # All snake_case
        assert profile.type_hints_enabled == 1.0  # All have type hints
        assert profile.functional_ratio == 1.0  # All functions, no methods


class TestKLDivergence:
    """Test KL divergence calculation."""
    
    def test_same_distribution(self):
        """Test KL divergence for identical distributions."""
        calc = KLDivergenceCalculator()
        
        p = StyleProfile(
            naming_snake_case=1.0,
            naming_camel_case=0.0,
            naming_pascal_case=0.0,
            type_hints_enabled=0.8,
            docstring_google=1.0,
            docstring_numpy=0.0,
            docstring_rest=0.0,
            docstring_none=0.0,
            quote_single=0.5,
            quote_double=0.5,
            functional_ratio=0.7
        )
        
        q = StyleProfile(
            naming_snake_case=1.0,
            naming_camel_case=0.0,
            naming_pascal_case=0.0,
            type_hints_enabled=0.8,
            docstring_google=1.0,
            docstring_numpy=0.0,
            docstring_rest=0.0,
            docstring_none=0.0,
            quote_single=0.5,
            quote_double=0.5,
            functional_ratio=0.7
        )
        
        kl = calc.calculate(p, q)
        assert abs(kl) < 0.01  # Should be near zero
    
    def test_different_distributions(self):
        """Test KL divergence for different distributions."""
        calc = KLDivergenceCalculator()
        
        p = StyleProfile(
            naming_snake_case=1.0,
            naming_camel_case=0.0,
            naming_pascal_case=0.0,
        )
        
        q = StyleProfile(
            naming_snake_case=0.0,
            naming_camel_case=1.0,
            naming_pascal_case=0.0,
        )
        
        kl = calc.calculate(p, q)
        assert kl > 1.0  # Should be large
    
    def test_epsilon_smoothing(self):
        """Test that epsilon smoothing prevents division by zero."""
        calc = KLDivergenceCalculator(epsilon=1e-10)
        
        p = StyleProfile(
            naming_snake_case=1.0,
            naming_camel_case=0.0,
            naming_pascal_case=0.0,
        )
        
        q = StyleProfile(
            naming_snake_case=0.0,
            naming_camel_case=0.0,
            naming_pascal_case=0.0,
        )
        
        # Should not raise exception due to smoothing
        kl = calc.calculate(p, q)
        assert kl > 0


class TestStyleGatedPrompt:
    """Test style-gated prompt building."""
    
    def test_should_update_high_divergence(self):
        """Test update triggered for high divergence."""
        gater = StyleGatedPrompt(divergence_threshold=0.15)
        
        local_entities = [
            Entity(
                name='get_user',
                type='function',
                file_path='/local.py',
                line_start=1,
                line_end=5,
                signature='def get_user(id: int) -> User',
                raw_code='def get_user(id: int) -> User:\n    pass'
            ),
        ]
        
        global_entities = [
            Entity(
                name='getUser',
                type='method',
                file_path='/global.py',
                line_start=1,
                line_end=5,
                signature='def getUser(self, id)',
                raw_code='def getUser(self, id):\n    pass'
            ),
        ]
        
        should_update, divergence = gater.should_update_style(local_entities, global_entities)
        
        # Different naming conventions should trigger update
        assert should_update == True or divergence > 0.15
    
    def test_should_not_update_low_divergence(self):
        """Test no update for low divergence."""
        gater = StyleGatedPrompt(divergence_threshold=0.5)  # High threshold
        
        entities = [
            Entity(
                name='get_user',
                type='function',
                file_path='/test.py',
                line_start=1,
                line_end=5,
                signature='def get_user(id: int) -> User',
                raw_code='def get_user(id: int) -> User:\n    pass'
            ),
        ]
        
        should_update, divergence = gater.should_update_style(entities, entities)
        
        # Same entities should not trigger update
        assert should_update == False
    
    def test_build_style_instruction(self):
        """Test style instruction building."""
        gater = StyleGatedPrompt()
        
        profile = StyleProfile(
            naming_snake_case=1.0,
            type_hints_enabled=0.8,
            docstring_google=1.0,
            quote_single=0.7,
            functional_ratio=0.8
        )
        
        instruction = gater.build_style_instruction(profile)
        
        assert 'snake_case' in instruction
        assert 'type hints' in instruction
        assert 'Google' in instruction


if __name__ == '__main__':
    print("Running Phase 5 tests...")
    
    test_classes = [
        TestStyleAnalyzer(),
        TestKLDivergence(),
        TestStyleGatedPrompt(),
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
