import os
from typing import Any, Dict, List, Optional, Tuple

import lancedb
import pyarrow as pa
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


class Retriever:
    """
    Handles creating a LanceDB index for a repository's chunks and performing
    dense vector similarity search combined with sparse BM25 search.

    Optionally, a cross-encoder can be used to re-rank top candidates.
    """

    def __init__(
        self,
        db_path: str = "/tmp/lancedb",
        model_name: Optional[str] = None,
        cross_encoder_model: Optional[str] = None,
    ):
        self.db = lancedb.connect(db_path)
        # Allow overriding the embedding model via env, but keep the original
        # baseline as default to preserve performance unless explicitly changed.
        effective_model = (
            model_name
            or os.getenv("TASK2_EMBEDDING_MODEL")
            or "all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(effective_model)

        # Store bm25 index inside the retriever
        self.bm25_indices: Dict[str, BM25Okapi] = {}
        # Keep chunks in memory for bm25 scoring
        self.repo_chunks: Dict[str, List[Dict[str, Any]]] = {}

        # Optional cross-encoder for higher-precision re-ranking.
        ce_name = cross_encoder_model or os.getenv("TASK2_CROSS_ENCODER_MODEL")
        self.cross_encoder: Optional[CrossEncoder] = None
        if ce_name:
            try:
                self.cross_encoder = CrossEncoder(ce_name)
            except Exception:
                # If the model can't be loaded, silently fall back to hybrid-only.
                self.cross_encoder = None
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        # Using simple sentence embeddings
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def build_index(self, table_name: str, chunks: List[Dict[str, Any]]):
        """
        Creates a new table and inserts embeddings OR loads existing table to save GPU/CPU time.
        Always builds a BM25 index in memory for the given table/repo.
        """
        if not chunks:
            return

        # 1. Check if embeddings already exist on disk
        table_exists = False
        try:
            self.db.open_table(table_name)
            table_exists = True
        except Exception:
            pass

        tokenized_corpus: List[List[str]] = []
        texts_to_embed: List[str] = []

        # Always prepare the corpus for BM25
        for c in chunks:
            # Prefer embedding a compact abstract when available, falling back
            # to raw code. Always include filepath for extra lexical signal.
            abstract = c.get("abstract") or ""
            base_text = abstract or c.get("text", "")
            header = f"File: {c.get('rel_filepath', 'unknown')}"
            doc = f"{header}\n\n{base_text}"

            tokenized_corpus.append(doc.lower().split())
            if not table_exists:
                texts_to_embed.append(doc)

        if not table_exists:
            # 2a. Expensive operation: Embed and create LanceDB table
            embeddings = self.embed(texts_to_embed)

            data = []
            for i, chunk in enumerate(chunks):
                data.append({
                    "id": str(i),
                    "filepath": chunk.get("rel_filepath", ""),
                    "chunk_type": chunk.get("type", "unknown"),
                    "name": chunk.get("name", ""),
                    "text": chunk.get("text", ""),
                    "vector": embeddings[i]
                })

            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("filepath", pa.string()),
                pa.field("chunk_type", pa.string()),
                pa.field("name", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.model.get_sentence_embedding_dimension()))
            ])

            # Force overwrite just in case there are residual broken files
            self.db.create_table(table_name, schema=schema, data=data, mode="overwrite")

        # 2b. Build BM25 sparse index (Fast, in-memory)
        self.bm25_indices[table_name] = BM25Okapi(tokenized_corpus)
        self.repo_chunks[table_name] = chunks
        
    def search(
        self,
        table_name: str,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        re_rank: bool = True,
        re_rank_top_k: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Searches the table for the top_k most similar chunks to the query.

        Primary ranking:
            Hybrid Search (Dense + BM25) with alpha weighting:
                Score = alpha * DenseScore + (1 - alpha) * SparseScore
            Both scores are normalized before combination (min-max).

        Optional re-ranking:
            If a cross-encoder is available and `re_rank` is True, the top
            `re_rank_top_k` hybrid candidates are re-scored using the
            cross-encoder and then truncated to `top_k`.
        """
        if table_name not in self.db.table_names() or table_name not in self.bm25_indices:
            return []
            
        # ----------------
        # A. Dense Search
        # ----------------
        table = self.db.open_table(table_name)
        query_vector = self.embed([query])[0]
        
        # We fetch all chunks to properly merge and normalize scores
        # Can be optimized in a real DB but since repo context is relatively small, this is fine
        num_chunks = len(self.repo_chunks[table_name])
        limit = max(top_k * 5, num_chunks) 
        
        # LanceDB returns _distance (L2 distance), we want to convert it conceptually or just sort
        dense_results = table.search(query_vector).limit(limit).to_list()
        
        # Normalize dense distances (lower is better in L2). We want higher is better.
        # Let's convert L2 distance to a raw similarity score by simply doing -distance
        # or normalized: (max_dist - dist) / (max_dist - min_dist)
        if not dense_results:
            return []
            
        dense_max = max(r['_distance'] for r in dense_results)
        dense_min = min(r['_distance'] for r in dense_results)
        
        dense_scores = {}
        for r in dense_results:
            doc_id = int(r['id'])
            dist = r['_distance']
            if dense_max > dense_min:
                norm_score = (dense_max - dist) / (dense_max - dense_min)
            else:
                norm_score = 1.0
            dense_scores[doc_id] = norm_score

        # ----------------
        # B. Sparse Search (BM25)
        # ----------------
        tokenized_query = query.lower().split()
        bm25_scores_raw = self.bm25_indices[table_name].get_scores(tokenized_query)
        
        sparse_max = max(bm25_scores_raw) if len(bm25_scores_raw) > 0 else 0
        sparse_min = min(bm25_scores_raw) if len(bm25_scores_raw) > 0 else 0
        
        # ----------------
        # C. Fusion
        # ----------------
        hybrid_scores: List[Tuple[float, Dict[str, Any]]] = []
        chunks = self.repo_chunks[table_name]
        
        for i, chunk in enumerate(chunks):
            # Dense score
            d_score = dense_scores.get(i, 0.0)
            
            # Sparse score normalized
            raw_s = bm25_scores_raw[i]
            if sparse_max > sparse_min:
                s_score = (raw_s - sparse_min) / (sparse_max - sparse_min)
            else:
                s_score = 0.0
                
            final_score = (alpha * d_score) + ((1.0 - alpha) * s_score)
            
            hybrid_scores.append((final_score, chunk))
            
        # Sort by final score descending
        hybrid_scores.sort(key=lambda x: x[0], reverse=True)

        # Optional cross-encoder re-ranking of the strongest hybrid candidates.
        if self.cross_encoder is not None and re_rank and hybrid_scores:
            # Take a manageable candidate pool for re-ranking.
            pool = hybrid_scores[: min(re_rank_top_k, len(hybrid_scores))]
            texts = [
                c.get("text", "")
                for _, c in pool
            ]
            # Each item is scored as (query, candidate_chunk).
            try:
                pairs = [(query, t) for t in texts]
                ce_scores = self.cross_encoder.predict(pairs)
                scored = list(zip(ce_scores, (c for _, c in pool)))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [c for _, c in scored[:top_k]]
            except Exception:
                # If anything goes wrong, fall back to the original hybrid ranking.
                pass

        # Default: return top_k chunks by hybrid score.
        return [item[1] for item in hybrid_scores[:top_k]]


if __name__ == "__main__":
    # Quick test
    retriever = Retriever()
    sample_chunks = [
        {"rel_filepath": "math.py", "type": "function", "name": "add", "text": "def add(a, b): return a + b"},
        {"rel_filepath": "math.py", "type": "function", "name": "sub", "text": "def sub(a, b): return a - b"},
        {"rel_filepath": "string.py", "type": "function", "name": "concat", "text": "def concat(a, b): return str(a) + str(b)"}
    ]
    retriever.build_index("test_repo", sample_chunks)
    res = retriever.search("test_repo", "looking for an addition function", top_k=1)
    if res:
        print("Search result:", res[0]["text"])
    else:
        print("No results")

