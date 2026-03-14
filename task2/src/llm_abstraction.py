"""
LLM-Driven Functional Abstraction

Generates concise summaries of code entities using LLMs.
Caches results to avoid regenerating abstracts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLMAbstractor:
    """Generate functional abstracts using LLMs (OpenAI, DeepSeek, etc.)"""
    
    def __init__(
        self,
        cache_dir: str = ".cache",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize LLM abstractor.
        
        Args:
            cache_dir: Directory for caching abstracts
            model: Model name (e.g., 'gpt-4o-mini', 'deepseek-chat', 'deepseek-coder')
            api_key: API key (or set OPENAI_API_KEY / DEEPSEEK_API_KEY env var)
            base_url: API base URL (e.g., 'https://api.deepseek.com' for DeepSeek)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = None
        
        # Try to initialize OpenAI client
        try:
            import openai
            
            # Build client kwargs
            client_kwargs = {}
            if api_key:
                client_kwargs['api_key'] = api_key
            if base_url:
                client_kwargs['base_url'] = base_url
            
            self._client = openai.OpenAI(**client_kwargs)
            
            provider = "DeepSeek" if base_url and "deepseek" in base_url else "OpenAI"
            logger.info(f"LLM Abstractor initialized: {provider} / {model}")
            
        except ImportError:
            logger.debug("OpenAI not installed. LLM abstraction will use fallback.")
        except Exception as e:
            logger.debug(f"LLM abstraction not available: {e}")
    
    def _get_cache_key(self, code: str) -> str:
        """Get cache key for code."""
        code_hash = hashlib.md5(code.encode()).hexdigest()[:16]
        return f"abstract_{code_hash}.json"
    
    def _get_cached(self, code: str) -> Optional[str]:
        """Get cached abstract if available."""
        cache_file = self.cache_dir / self._get_cache_key(code)
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return data.get('abstract')
            except Exception as e:
                logger.warning(f"Failed to load cached abstract: {e}")
        return None
    
    def _cache_abstract(self, code: str, abstract: str) -> None:
        """Cache generated abstract."""
        cache_file = self.cache_dir / self._get_cache_key(code)
        try:
            with open(cache_file, 'w') as f:
                json.dump({'abstract': abstract, 'code_hash': hashlib.md5(code.encode()).hexdigest()}, f)
        except Exception as e:
            logger.warning(f"Failed to cache abstract: {e}")
    
    def generate_abstract(self, entity: Any) -> Optional[str]:
        """
        Generate LLM abstract for an entity.
        
        Args:
            entity: Code entity with raw_code, name, etc.
        
        Returns:
            Generated abstract string or None if failed
        """
        if not self._client:
            return None
        
        code = getattr(entity, 'raw_code', '')
        if not code:
            return None
        
        # Check cache first
        cached = self._get_cached(code)
        if cached:
            entity_name = getattr(entity, 'name', 'unknown')
            logger.debug(f"[CACHE HIT] {entity_name}")
            return cached
        
        entity_name = getattr(entity, 'name', 'unknown')
        logger.info(f"[API CALL] Generating abstract for: {entity_name}")
        
        # Build prompt
        prompt = self._build_prompt(entity)
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a code summarization assistant. Generate concise, informative summaries of code functions and classes."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            abstract = response.choices[0].message.content.strip()
            
            # Cache the result
            self._cache_abstract(code, abstract)
            
            logger.debug(f"Generated abstract for {getattr(entity, 'name', 'unknown')}")
            return abstract
            
        except Exception as e:
            logger.warning(f"LLM abstract generation failed: {e}")
            return None
    
    def _build_prompt(self, entity: Any) -> str:
        """Build prompt for LLM abstraction."""
        name = getattr(entity, 'name', 'unknown')
        entity_type = getattr(entity, 'type', 'function')
        code = getattr(entity, 'raw_code', '')
        signature = getattr(entity, 'signature', '')
        
        prompt = f"""Analyze this {entity_type} and provide a concise summary.

Name: {name}
Signature: {signature}

Code:
```python
{code}
```

Provide a 1-2 sentence summary of what this {entity_type} does, and list the key dependencies (functions/classes it calls or uses).

Format:
Summary: [what it does, input, output]
Dependencies: [comma-separated list of key dependencies, or "none"]"""
        
        return prompt
    
    def get_abstract(self, entity: Any, use_llm: bool = True) -> str:
        """
        Get abstract for entity, using LLM if available and needed.
        
        Args:
            entity: Code entity
            use_llm: Whether to try LLM generation
        
        Returns:
            Abstract string (LLM-generated or fallback)
        """
        # If entity has good docstring, use it
        docstring = getattr(entity, 'docstring', None)
        if docstring and len(docstring) > 10:
            return f'"""{docstring}"""'
        
        # Try LLM if enabled
        if use_llm and self._client:
            llm_abstract = self.generate_abstract(entity)
            if llm_abstract:
                return llm_abstract
        
        # Fallback: create simple abstract
        return self._create_simple_abstract(entity)
    
    def _create_simple_abstract(self, entity: Any) -> str:
        """Create simple abstract without LLM."""
        parts = []
        
        # Signature
        signature = getattr(entity, 'signature', '')
        if signature:
            parts.append(signature)
        
        # Docstring if exists
        docstring = getattr(entity, 'docstring', None)
        if docstring:
            parts.append(f'"""{docstring}"""')
        
        # Dependencies
        deps = getattr(entity, 'dependencies', [])
        if deps:
            dep_str = ", ".join(deps[:5])
            parts.append(f"# Uses: {dep_str}")
        
        return "\n".join(parts) if parts else getattr(entity, 'name', 'unknown')


class CachedLLMAbstractor:
    """LLM Abstractor with batch processing and caching."""
    
    def __init__(
        self,
        cache_dir: str = ".cache",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 10
    ):
        self.abstractor = LLMAbstractor(cache_dir, model, api_key, base_url)
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self._entity_cache: Dict[str, str] = {}
        
        # Load existing cache
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load all cached abstracts into memory."""
        if not self.cache_dir.exists():
            return
        
        for cache_file in self.cache_dir.glob("abstract_*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    code_hash = cache_file.stem.replace('abstract_', '')
                    self._entity_cache[code_hash] = data.get('abstract', '')
            except Exception:
                pass
        
        logger.info(f"Loaded {len(self._entity_cache)} cached abstracts")
    
    def get_abstract(self, entity: Any) -> str:
        """Get abstract for entity (cached or generated)."""
        code = getattr(entity, 'raw_code', '')
        if not code:
            return getattr(entity, 'name', 'unknown')
        
        code_hash = hashlib.md5(code.encode()).hexdigest()[:16]
        
        # Check memory cache
        if code_hash in self._entity_cache:
            return self._entity_cache[code_hash]
        
        # Generate and cache
        abstract = self.abstractor.get_abstract(entity, use_llm=True)
        self._entity_cache[code_hash] = abstract
        
        return abstract
    
    def batch_generate(self, entities: list) -> Dict[str, str]:
        """
        Generate abstracts for multiple entities.
        
        Args:
            entities: List of entities to abstract
        
        Returns:
            Dict mapping entity IDs to abstracts
        """
        results = {}
        to_generate = []
        
        # Check cache first
        for entity in entities:
            code = getattr(entity, 'raw_code', '')
            code_hash = hashlib.md5(code.encode()).hexdigest()[:16]
            
            if code_hash in self._entity_cache:
                results[getattr(entity, 'id', str(entity))] = self._entity_cache[code_hash]
            else:
                to_generate.append(entity)
        
        cached_count = len(results)
        to_generate_count = len(to_generate)
        
        # Estimate cost (rough: ~500 tokens per function, ~$0.0001 per 1K tokens for DeepSeek)
        estimated_cost = to_generate_count * 0.0001  # Very rough estimate
        
        logger.info(f"[CACHE] {cached_count} entities already cached")
        logger.info(f"[API] {to_generate_count} entities need LLM generation")
        logger.info(f"[COST] Estimated cost: ${estimated_cost:.4f} (rough estimate)")
        
        if to_generate_count == 0:
            logger.info("All abstracts cached! No API calls needed.")
            return results
        
        # Generate in batches
        api_calls = 0
        for i in range(0, len(to_generate), self.batch_size):
            batch = to_generate[i:i + self.batch_size]
            
            for entity in batch:
                abstract = self.abstractor.get_abstract(entity, use_llm=True)
                entity_id = getattr(entity, 'id', str(entity))
                results[entity_id] = abstract
                api_calls += 1
            
            logger.info(f"Progress: {api_calls}/{to_generate_count} API calls made...")
        
        logger.info(f"[DONE] Total API calls: {api_calls}")
        return results
