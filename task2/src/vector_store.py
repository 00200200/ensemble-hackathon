"""
Phase 2: Vector Store & Hybrid Retrieval

Implements dense embeddings (sentence-transformers) + sparse retrieval (BM25)
with hybrid scoring: S(q,d) = α·Dense(q,d) + (1-α)·BM25(q,d)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False

import numpy as np

logger = logging.getLogger(__name__)


def tokenize_code(text: str) -> List[str]:
    """Tokenize code for BM25."""
    # Split camelCase and PascalCase FIRST (before lowercasing)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', text)
    
    # Split snake_case
    text = text.replace('_', ' ')
    
    # Split on non-alphanumeric
    tokens = re.findall(r'[a-zA-Z0-9]+', text)
    
    # Convert to lowercase
    tokens = [t.lower() for t in tokens]
    
    return tokens


class SimpleEmbedder:
    """Simple keyword-based embedder as fallback."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vocab: Dict[str, int] = {}
    
    def _get_vocab_id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Create bag-of-words embeddings."""
        embeddings = []
        for text in texts:
            tokens = tokenize_code(text)
            vec = np.zeros(self.dimension)
            for token in tokens:
                vocab_id = self._get_vocab_id(token) % self.dimension
                vec[vocab_id] += 1
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings)


class CodeEmbedder:
    """Embed code entities using sentence-transformers or fallback."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = None
        self.fallback = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded sentence-transformers model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence-transformers: {e}")
        
        if self.model is None:
            self.fallback = SimpleEmbedder()
            logger.info("Using simple embedder fallback")
    
    def prepare_entity_text(self, entity: Any) -> str:
        """Prepare entity text for embedding."""
        parts = []
        
        # Name and type
        if hasattr(entity, 'name'):
            parts.append(f"Name: {entity.name}")
        if hasattr(entity, 'type'):
            parts.append(f"Type: {entity.type}")
        
        # Signature
        if hasattr(entity, 'signature') and entity.signature:
            parts.append(f"Signature: {entity.signature}")
        
        # Docstring
        if hasattr(entity, 'docstring') and entity.docstring:
            parts.append(f"Documentation: {entity.docstring}")
        
        # Code preview (first 10 lines)
        if hasattr(entity, 'raw_code') and entity.raw_code:
            lines = entity.raw_code.split('\n')[:10]
            parts.append(f"Code:\n" + '\n'.join(lines))
        
        return '\n'.join(parts)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.model:
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        else:
            return self.fallback.encode(texts)
    
    def encode_entities(self, entities: List[Any]) -> np.ndarray:
        """Encode a list of entities."""
        texts = [self.prepare_entity_text(e) for e in entities]
        return self.encode(texts)


class SimpleBM25:
    """Simple BM25 implementation as fallback."""
    
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc) for doc in corpus]
        self.avg_dl = sum(self.doc_len) / len(corpus) if corpus else 0
        
        # Build term frequency
        self.df: Dict[str, int] = {}
        for doc in corpus:
            seen = set()
            for term in doc:
                if term not in seen:
                    self.df[term] = self.df.get(term, 0) + 1
                    seen.add(term)
        
        self.N = len(corpus)
    
    def get_scores(self, query: List[str]) -> np.ndarray:
        """Get BM25 scores for a query."""
        scores = np.zeros(self.N)
        
        for idx, doc in enumerate(self.corpus):
            score = 0
            doc_len = self.doc_len[idx]
            
            for term in query:
                if term not in self.df:
                    continue
                
                # Term frequency in document
                f = doc.count(term)
                if f == 0:
                    continue
                
                # IDF
                idf = np.log((self.N - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)
                
                # BM25 term score
                score += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl))
            
            scores[idx] = score
        
        return scores


class DenseVectorStore:
    """Dense vector store using FAISS or numpy fallback."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.vectors: Optional[np.ndarray] = None
        self.entities: List[Any] = []
        self.use_faiss = FAISS_AVAILABLE
    
    def add(self, entities: List[Any], embeddings: np.ndarray) -> None:
        """Add entities with their embeddings."""
        self.entities = entities
        
        # Ensure float32 for FAISS
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        self.vectors = embeddings
        
        if self.use_faiss:
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine similarity for normalized vectors
            self.index.add(embeddings)
            logger.info(f"Added {len(entities)} entities to FAISS index")
        else:
            logger.info(f"Added {len(entities)} entities to numpy index")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float, Any]]:
        """
        Search for similar entities.
        
        Returns:
            List of (index, score, entity) tuples
        """
        if len(self.entities) == 0:
            return []
        
        if self.use_faiss and self.index is not None:
            # Normalize query
            query = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query)
            
            # Search
            scores, indices = self.index.search(query, min(k, len(self.entities)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.entities):
                    results.append((idx, float(score), self.entities[idx]))
            return results
        else:
            # NumPy fallback
            query = query_embedding.reshape(1, -1)
            
            # Normalize vectors
            vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            
            # Compute cosine similarity
            similarities = np.dot(vectors_norm, query_norm.T).flatten()
            
            # Get top k
            top_k = np.argsort(similarities)[::-1][:k]
            
            return [(idx, float(similarities[idx]), self.entities[idx]) for idx in top_k]


class SparseVectorStore:
    """Sparse vector store using BM25."""
    
    def __init__(self):
        self.bm25 = None
        self.entities: List[Any] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def prepare_entity_text(self, entity: Any) -> str:
        """Prepare entity text for BM25."""
        parts = []
        
        if hasattr(entity, 'name'):
            parts.append(entity.name)
        if hasattr(entity, 'signature') and entity.signature:
            parts.append(entity.signature)
        if hasattr(entity, 'docstring') and entity.docstring:
            parts.append(entity.docstring)
        if hasattr(entity, 'raw_code') and entity.raw_code:
            parts.append(entity.raw_code)
        
        return ' '.join(parts)
    
    def add(self, entities: List[Any]) -> None:
        """Add entities to the BM25 index."""
        self.entities = entities
        
        # Tokenize corpus
        self.tokenized_corpus = []
        for entity in entities:
            text = self.prepare_entity_text(entity)
            tokens = tokenize_code(text)
            self.tokenized_corpus.append(tokens)
        
        # Build BM25 index
        if RANK_BM25_AVAILABLE:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = SimpleBM25(self.tokenized_corpus)
        
        logger.info(f"Added {len(entities)} entities to BM25 index")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float, Any]]:
        """
        Search using BM25.
        
        Returns:
            List of (index, score, entity) tuples
        """
        if len(self.entities) == 0 or self.bm25 is None:
            return []
        
        # Tokenize query
        query_tokens = tokenize_code(query)
        
        # Get scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k
        top_k = np.argsort(scores)[::-1][:k]
        
        return [(idx, float(scores[idx]), self.entities[idx]) for idx in top_k if scores[idx] > 0]


class HybridRetriever:
    """Hybrid retriever combining dense and sparse retrieval."""
    
    def __init__(self, alpha: float = 0.5, dimension: int = 384):
        """
        Initialize hybrid retriever.
        
        Args:
            alpha: Weight for dense retrieval (0-1). 0 = only BM25, 1 = only dense
            dimension: Embedding dimension
        """
        self.alpha = alpha
        self.embedder = CodeEmbedder()
        self.dense_store = DenseVectorStore(dimension=dimension)
        self.sparse_store = SparseVectorStore()
        self.entities: List[Any] = []
    
    def index_entities(self, entities: List[Any]) -> None:
        """Index entities for retrieval."""
        self.entities = entities
        
        # Dense indexing
        logger.info(f"Computing embeddings for {len(entities)} entities...")
        embeddings = self.embedder.encode_entities(entities)
        self.dense_store.add(entities, embeddings)
        
        # Sparse indexing
        self.sparse_store.add(entities)
    
    def search(
        self, 
        query: str, 
        k: int = 10, 
        alpha: Optional[float] = None
    ) -> List[Tuple[Any, float]]:
        """
        Search with hybrid scoring.
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Override alpha for this query
        
        Returns:
            List of (entity, score) tuples sorted by score
        """
        if len(self.entities) == 0:
            return []
        
        alpha = alpha if alpha is not None else self.alpha
        
        # Dense search
        if alpha > 0:
            query_embedding = self.embedder.encode([query])[0]
            dense_results = self.dense_store.search(query_embedding, k=k*2)
        else:
            dense_results = []
        
        # Sparse search
        if alpha < 1:
            sparse_results = self.sparse_store.search(query, k=k*2)
        else:
            sparse_results = []
        
        # Combine scores using Reciprocal Rank Fusion
        scores: Dict[int, float] = {}
        
        # RRF constant
        k_rrf = 60
        
        # Add dense scores (weighted)
        for rank, (idx, score, entity) in enumerate(dense_results):
            if alpha > 0:
                rrf_score = alpha * (1.0 / (k_rrf + rank + 1))
                scores[idx] = scores.get(idx, 0) + rrf_score
        
        # Add sparse scores (weighted)
        for rank, (idx, score, entity) in enumerate(sparse_results):
            if alpha < 1:
                rrf_score = (1 - alpha) * (1.0 / (k_rrf + rank + 1))
                scores[idx] = scores.get(idx, 0) + rrf_score
        
        # Sort by score
        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        
        # Return top k
        results = []
        for idx in sorted_indices[:k]:
            results.append((self.entities[idx], scores[idx]))
        
        return results
    
    def auto_tune_alpha(self, query: str) -> float:
        """
        Auto-tune alpha based on query characteristics.
        
        Returns:
            Tuned alpha value
        """
        # Identifier-heavy query -> lower alpha (more BM25)
        # Conceptual query -> higher alpha (more dense)
        
        # Check for camelCase/PascalCase (likely identifiers)
        identifier_pattern = re.compile(r'[a-z][A-Z]|[A-Z][A-Z][a-z]')
        identifiers = identifier_pattern.findall(query)
        
        # If many identifiers, prefer BM25
        if len(identifiers) >= 3:
            return 0.2
        elif len(identifiers) >= 1:
            return 0.4
        else:
            return 0.7
