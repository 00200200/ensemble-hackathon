"""
Phase 5: Dynamic Style Engine (KL-Gated)

Keeps the LLM's coding style consistent with the specific module being edited.
Uses KL divergence to measure style difference and threshold gating to prevent jitter.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StyleProfile:
    """Profile of coding style."""
    
    # Naming conventions (probabilities)
    naming_snake_case: float = 0.0
    naming_camel_case: float = 0.0
    naming_pascal_case: float = 0.0
    
    # Type hints
    type_hints_enabled: float = 0.0  # Percentage of functions with type hints
    
    # Docstring style (probabilities)
    docstring_google: float = 0.0
    docstring_numpy: float = 0.0
    docstring_rest: float = 0.0
    docstring_none: float = 0.0
    
    # Quote preference
    quote_single: float = 0.0
    quote_double: float = 0.0
    
    # Line length
    avg_line_length: float = 0.0
    
    # Functional vs OOP
    functional_ratio: float = 0.0  # Functions / (functions + methods)
    
    # Sample size
    sample_size: int = 0


class StyleAnalyzer:
    """Analyze coding style from code samples."""
    
    def __init__(self):
        pass
    
    def analyze_entities(self, entities: List[Any]) -> StyleProfile:
        """
        Analyze style from a list of entities.
        
        Args:
            entities: List of code entities (functions, methods, classes)
        
        Returns:
            StyleProfile
        """
        if not entities:
            return StyleProfile()
        
        profile = StyleProfile()
        profile.sample_size = len(entities)
        
        # Naming analysis
        naming_counts = {'snake': 0, 'camel': 0, 'pascal': 0, 'other': 0}
        
        # Type hints
        type_hint_count = 0
        
        # Docstrings
        docstring_counts = {'google': 0, 'numpy': 0, 'rest': 0, 'none': 0}
        
        # Quotes
        quote_counts = {'single': 0, 'double': 0}
        
        # Line length
        total_lines = 0
        total_length = 0
        
        # Functional vs OOP
        function_count = 0
        method_count = 0
        
        for entity in entities:
            name = getattr(entity, 'name', '')
            code = getattr(entity, 'raw_code', '')
            signature = getattr(entity, 'signature', '')
            docstring = getattr(entity, 'docstring', None)
            entity_type = getattr(entity, 'type', '')
            
            # Naming
            naming = self._detect_naming_convention(name)
            naming_counts[naming] = naming_counts.get(naming, 0) + 1
            
            # Type hints
            if self._has_type_hints(signature):
                type_hint_count += 1
            
            # Docstring
            docstyle = self._detect_docstring_style(docstring)
            docstring_counts[docstyle] = docstring_counts.get(docstyle, 0) + 1
            
            # Quotes
            quotes = self._detect_quote_preference(code)
            for q in quotes:
                quote_counts[q] = quote_counts.get(q, 0) + 1
            
            # Line length
            lines = code.split('\n')
            for line in lines:
                total_lines += 1
                total_length += len(line)
            
            # Functional vs OOP
            if entity_type == 'function':
                function_count += 1
            elif entity_type == 'method':
                method_count += 1
        
        # Convert to probabilities
        total = len(entities)
        
        profile.naming_snake_case = naming_counts.get('snake', 0) / total
        profile.naming_camel_case = naming_counts.get('camel', 0) / total
        profile.naming_pascal_case = naming_counts.get('pascal', 0) / total
        
        profile.type_hints_enabled = type_hint_count / total
        
        profile.docstring_google = docstring_counts.get('google', 0) / total
        profile.docstring_numpy = docstring_counts.get('numpy', 0) / total
        profile.docstring_rest = docstring_counts.get('rest', 0) / total
        profile.docstring_none = docstring_counts.get('none', 0) / total
        
        total_quotes = quote_counts.get('single', 0) + quote_counts.get('double', 0)
        if total_quotes > 0:
            profile.quote_single = quote_counts.get('single', 0) / total_quotes
            profile.quote_double = quote_counts.get('double', 0) / total_quotes
        
        if total_lines > 0:
            profile.avg_line_length = total_length / total_lines
        
        total_funcs = function_count + method_count
        if total_funcs > 0:
            profile.functional_ratio = function_count / total_funcs
        
        return profile
    
    def _detect_naming_convention(self, name: str) -> str:
        """Detect naming convention of a name."""
        if not name:
            return 'other'
        
        # Check for snake_case
        if '_' in name and not any(c.isupper() for c in name):
            return 'snake'
        
        # Check for camelCase (starts lowercase, has uppercase)
        if name[0].islower() and any(c.isupper() for c in name):
            return 'camel'
        
        # Check for PascalCase (starts uppercase)
        if name[0].isupper():
            return 'pascal'
        
        return 'other'
    
    def _has_type_hints(self, signature: str) -> bool:
        """Check if signature has type hints."""
        if not signature:
            return False
        
        # Look for -> for return type
        if '->' in signature:
            return True
        
        # Look for : after parameter name (but not at the very end which is just syntax)
        # Pattern: parameter: type
        match = re.search(r'\w+\s*:\s*\w+', signature)
        return match is not None
    
    def _detect_docstring_style(self, docstring: Optional[str]) -> str:
        """Detect docstring style."""
        if not docstring:
            return 'none'
        
        # Check for Google style (Args:, Returns:)
        if re.search(r'\n\s*Args:', docstring) or re.search(r'^\s*Args:', docstring):
            return 'google'
        
        # Check for NumPy style (Parameters, Returns)
        if re.search(r'\n\s*Parameters\s*\n\s*-+', docstring):
            return 'numpy'
        
        # Check for reST style (:param:, :return:)
        if ':param' in docstring or ':return:' in docstring:
            return 'rest'
        
        # Default to Google if has docstring but no specific style
        return 'google'
    
    def _detect_quote_preference(self, code: str) -> List[str]:
        """Detect quote preference in code."""
        if not code:
            return []
        
        quotes = []
        
        # Count single quotes (not apostrophes)
        single_matches = re.findall(r"'[^']*'", code)
        if single_matches:
            quotes.append('single')
        
        # Count double quotes
        double_matches = re.findall(r'"[^"]*"', code)
        if double_matches:
            quotes.append('double')
        
        return quotes


class KLDivergenceCalculator:
    """Calculate KL divergence between style profiles."""
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def calculate(self, p: StyleProfile, q: StyleProfile) -> float:
        """
        Calculate KL divergence D_KL(P || Q).
        
        Args:
            p: Local style profile (P)
            q: Global/repo style profile (Q)
        
        Returns:
            KL divergence value
        """
        divergences = []
        
        # Naming conventions (categorical)
        p_naming = np.array([
            max(p.naming_snake_case, self.epsilon),
            max(p.naming_camel_case, self.epsilon),
            max(p.naming_pascal_case, self.epsilon)
        ])
        q_naming = np.array([
            max(q.naming_snake_case, self.epsilon),
            max(q.naming_camel_case, self.epsilon),
            max(q.naming_pascal_case, self.epsilon)
        ])
        
        # Normalize
        p_naming = p_naming / p_naming.sum()
        q_naming = q_naming / q_naming.sum()
        
        kl_naming = np.sum(p_naming * np.log(p_naming / q_naming))
        divergences.append(kl_naming)
        
        # Docstring style (categorical)
        p_doc = np.array([
            max(p.docstring_google, self.epsilon),
            max(p.docstring_numpy, self.epsilon),
            max(p.docstring_rest, self.epsilon),
            max(p.docstring_none, self.epsilon)
        ])
        q_doc = np.array([
            max(q.docstring_google, self.epsilon),
            max(q.docstring_numpy, self.epsilon),
            max(q.docstring_rest, self.epsilon),
            max(q.docstring_none, self.epsilon)
        ])
        
        p_doc = p_doc / p_doc.sum()
        q_doc = q_doc / q_doc.sum()
        
        kl_doc = np.sum(p_doc * np.log(p_doc / q_doc))
        divergences.append(kl_doc)
        
        # Quote preference (categorical)
        p_quote = np.array([
            max(p.quote_single, self.epsilon),
            max(p.quote_double, self.epsilon)
        ])
        q_quote = np.array([
            max(q.quote_single, self.epsilon),
            max(q.quote_double, self.epsilon)
        ])
        
        p_quote = p_quote / p_quote.sum()
        q_quote = q_quote / q_quote.sum()
        
        kl_quote = np.sum(p_quote * np.log(p_quote / q_quote))
        divergences.append(kl_quote)
        
        # Type hints (continuous - use difference)
        type_diff = abs(p.type_hints_enabled - q.type_hints_enabled)
        divergences.append(type_diff)
        
        # Functional ratio (continuous)
        func_diff = abs(p.functional_ratio - q.functional_ratio)
        divergences.append(func_diff)
        
        # Average divergence
        return float(np.mean(divergences))


class StyleGatedPrompt:
    """Style-gated prompt builder."""
    
    def __init__(
        self,
        divergence_threshold: float = 0.15,
        epsilon: float = 1e-10
    ):
        self.divergence_threshold = divergence_threshold
        self.epsilon = epsilon
        self.analyzer = StyleAnalyzer()
        self.kl_calculator = KLDivergenceCalculator(epsilon)
        self.current_style: Optional[StyleProfile] = None
    
    def should_update_style(
        self,
        local_entities: List[Any],
        global_entities: List[Any]
    ) -> Tuple[bool, float]:
        """
        Determine if style should be updated.
        
        Args:
            local_entities: Entities from current file
            global_entities: Entities from repository
        
        Returns:
            (should_update, divergence)
        """
        local_profile = self.analyzer.analyze_entities(local_entities)
        global_profile = self.analyzer.analyze_entities(global_entities)
        
        divergence = self.kl_calculator.calculate(local_profile, global_profile)
        
        should_update = divergence > self.divergence_threshold
        
        if should_update:
            self.current_style = local_profile
            logger.info(f"Style divergence {divergence:.4f} > {self.divergence_threshold}, updating")
        else:
            logger.info(f"Style divergence {divergence:.4f} <= {self.divergence_threshold}, keeping current")
        
        return should_update, divergence
    
    def build_style_instruction(self, profile: StyleProfile) -> str:
        """Build style instruction from profile."""
        parts = []
        
        # Naming
        naming = []
        if profile.naming_snake_case > 0.5:
            naming.append("snake_case")
        elif profile.naming_camel_case > 0.5:
            naming.append("camelCase")
        elif profile.naming_pascal_case > 0.5:
            naming.append("PascalCase")
        
        if naming:
            parts.append(f"Use {naming[0]} naming convention")
        
        # Type hints
        if profile.type_hints_enabled > 0.5:
            parts.append("include type hints")
        else:
            parts.append("avoid type hints")
        
        # Docstrings
        doc_styles = []
        if profile.docstring_google > 0.3:
            doc_styles.append("Google-style")
        if profile.docstring_numpy > 0.3:
            doc_styles.append("NumPy-style")
        if profile.docstring_rest > 0.3:
            doc_styles.append("reST-style")
        
        if doc_styles:
            parts.append(f"use {doc_styles[0]} docstrings")
        
        # Quotes
        if profile.quote_single > profile.quote_double:
            parts.append("prefer single quotes")
        else:
            parts.append("prefer double quotes")
        
        # Functional vs OOP
        if profile.functional_ratio > 0.7:
            parts.append("favor functional programming")
        elif profile.functional_ratio < 0.3:
            parts.append("favor object-oriented programming")
        
        if parts:
            return "Code style: " + ", ".join(parts) + "."
        return ""
    
    def get_system_prompt_addition(
        self,
        local_entities: List[Any],
        global_entities: List[Any]
    ) -> Optional[str]:
        """
        Get system prompt addition based on style analysis.
        
        Returns:
            Style instruction if divergence is high enough, None otherwise
        """
        should_update, _ = self.should_update_style(local_entities, global_entities)
        
        if should_update and self.current_style:
            return self.build_style_instruction(self.current_style)
        
        return None


def analyze_file_style(
    file_path: str,
    graph: Any
) -> Tuple[StyleProfile, StyleProfile]:
    """
    Analyze style for a file vs repository.
    
    Args:
        file_path: Path to current file
        graph: Code graph
    
    Returns:
        (local_profile, global_profile)
    """
    analyzer = StyleAnalyzer()
    
    # Get local entities
    local_entities = graph.get_entities_by_file(file_path) if hasattr(graph, 'get_entities_by_file') else []
    
    # Sample local entities (max 5)
    if len(local_entities) > 5:
        import random
        local_entities = random.sample(local_entities, 5)
    
    # Get global entities (sample 20)
    all_entities = [e for _, e in graph.get_all_entities()] if hasattr(graph, 'get_all_entities') else []
    global_entities = all_entities[:20]  # Already shuffled from dict, take first 20
    
    # Analyze
    local_profile = analyzer.analyze_entities(local_entities)
    global_profile = analyzer.analyze_entities(global_entities)
    
    return local_profile, global_profile
