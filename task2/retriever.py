import lancedb
from sentence_transformers import SentenceTransformer
import pyarrow as pa
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class Retriever:
    """
    Handles creating a LanceDB index for a repository's chunks and performing
    dense vector similarity search combined with sparse BM25 search.
    """
    def __init__(self, db_path: str = "/tmp/lancedb", model_name: str = "all-MiniLM-L6-v2"):
        self.db = lancedb.connect(db_path)
        self.model = SentenceTransformer(model_name)
        # Store bm25 index inside the retriever
        self.bm25_indices: Dict[str, BM25Okapi] = {}
        # Keep chunks in memory for bm25 scoring
        self.repo_chunks: Dict[str, List[Dict[str, Any]]] = {}
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        # Using simple sentence embeddings
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def build_index(self, table_name: str, chunks: List[Dict[str, Any]]):
        """
        Creates a new table and inserts embeddings for all text chunks.
        Also builds a BM25 index in memory for the given table/repo.
        """
        if not chunks:
            return
            
        # We drop the table if it already exists for a new repo
        if table_name in self.db.table_names():
            self.db.drop_table(table_name)

        # -----------------------------
        # 1. Prepare Data & dense index
        # -----------------------------
        texts_to_embed = []
        tokenized_corpus = []

        for c in chunks:
            # Combine filepath and text for richer embedding context
            doc = f"File: {c.get('rel_filepath', 'unknown')}\n\n{c.get('text', '')}"
            texts_to_embed.append(doc)
            # Tokenize for BM25
            tokenized_corpus.append(doc.lower().split())

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

        # Define schema explicitly to avoid PyArrow inference issues on empty fields
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("filepath", pa.string()),
            pa.field("chunk_type", pa.string()),
            pa.field("name", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.model.get_sentence_embedding_dimension()))
        ])

        self.db.create_table(table_name, schema=schema, data=data)

        # -----------------------------
        # 2. Build BM25 sparse index
        # -----------------------------
        self.bm25_indices[table_name] = BM25Okapi(tokenized_corpus)
        self.repo_chunks[table_name] = chunks
        
    def search(self, table_name: str, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Searches the table for the top_k most similar chunks to the query.
        Uses Hybrid Search (Dense + BM25) with alpha weighting:
        Score = alpha * DenseScore + (1 - alpha) * SparseScore
        Both scores are normalized before combination (min-max).
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
        hybrid_scores = []
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
        
        # Return top_k chunks
        top_chunks = [item[1] for item in hybrid_scores[:top_k]]
        return top_chunks


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

