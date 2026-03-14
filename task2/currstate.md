Current State

We have a baseline implementation that meets the basic submission requirements:

    Chunking (chunker.py): Uses Tree‑sitter to split Python files into functions, classes, or whole‑file blocks. Metadata (file path, name, type) is preserved.

    Retrieval (retriever.py): Implements a hybrid dense‑sparse search (SentenceTransformers + BM25) with tunable alpha. LanceDB stores vectors; an in‑memory BM25 index enables fusion.

    Processing (main.py): For each query in the input JSONL, extracts the corresponding repository zip, builds an index, constructs a query from the last 1000 chars of prefix and first 1000 chars of suffix, retrieves the top‑5 chunks, and formats the context using the required <|file_sep|> token.

    Local Evaluation (evaluate_local.py): Computes a chrF score between the generated contexts and the ground‑truth middle code – a reasonable proxy for the final metric.

The pipeline runs end‑to‑end and produces a correctly formatted JSONL file. However, several aspects need improvement to align with the official evaluation process and to boost the final average chrF score across the three target models (Mellum, Codestral, Qwen2.5‑Coder).
What Must Be Improved
1. Context Ordering for Left Truncation

Issue
The evaluation team trims the submitted context from the left to fit each model’s context window (8 K tokens for Mellum, 16 K for the others).
Our current code places the most relevant chunk first (because results are sorted descending by score). When trimmed from the left, this most relevant chunk may be discarded.

Fix
Reverse the order of chunks when building the context – put the least relevant first, most relevant last.
python

# In main.py, replace:
for res in results:
    context_parts.append(...)
# with:
for res in reversed(results):
    context_parts.append(...)

This ensures that if the context is too long, the right‑most (most relevant) parts survive the left truncation.
2. Token‑Aware Context Sizing

Issue
We currently retrieve a fixed number of chunks (5) without considering token limits. If these chunks together exceed 16 K tokens, even the right‑most parts could be trimmed for all models. Worse, for Mellum’s 8 K window, a significant portion of the right side might also be lost if the context is very long.

Improvement

    Add token counting (e.g., using tiktoken or the Qwen2.5 tokenizer) to estimate the length of each chunk.

    Either truncate the context to ≤ 8 K tokens (so no trimming occurs for any model) or use a greedy selection that adds chunks from least to most relevant until the token budget is reached, thereby preserving the most relevant chunks at the end even when trimmed.

    This also allows us to retrieve more than 5 chunks initially and then select the best‑fitting subset.

3. Query Formulation

Issue
The current query is a naive concatenation of the last 1000 characters of prefix and the first 1000 characters of suffix. This may lose important information (e.g., function names, variable declarations) that could guide retrieval.

Improvements

    Use the full prefix and suffix (or a longer context) if the embedding model supports it (all‑MiniLM‑L6‑v2 has a 512‑token limit; consider switching to a model with larger context, such as all‑mpnet‑base‑v2 or a code‑specific model).

    Extract key terms (function names, class names, imports) from the prefix/suffix using a simple heuristic or a lightweight LLM.

    Experiment with using only the prefix (or only the suffix) as the query – sometimes the suffix contains the expected completion and may be less informative.

4. Retrieval Tuning

Issue
The hybrid fusion parameter alpha = 0.5 and the number of retrieved chunks top_k = 5 are arbitrary and likely not optimal.

Improvements

    Tune alpha and top_k using a validation set (if available) or via cross‑validation on the training data. The goal is to maximise the chrF score after generation, but a proxy can be the similarity between retrieved chunks and the ground‑truth middle.

    Consider using a different embedding model – sentence‑transformers trained on code (e.g., codebert, graphcodebert) may yield better semantic matches.

    Experiment with reciprocal rank fusion (RRF) instead of weighted sum for hybrid scores.

5. Advanced Retrieval: Dependency Graph

Issue
Retrieving isolated chunks may miss important context about how a function interacts with others. The arch.md document outlines a graph‑based approach that includes the dependencies (called functions) of the retrieved chunks.

Improvement

    Build a call graph of the repository (using static analysis or Tree‑sitter queries).

    After retrieving the top‑k relevant chunks, expand the context by adding the abstracts (or bodies) of functions they call (1‑hop dependencies). This provides broader architectural context and can significantly improve completion quality.

6. Efficiency and Caching

Issue
For every query, main.py extracts the same repository zip and rebuilds its index from scratch. This is wasteful, especially as the dataset grows.

Improvement

    Cache processed repositories: after extracting and chunking a repo, persist the chunks and embeddings (e.g., in a separate folder or a persistent LanceDB table). Reuse them for subsequent queries on the same repo.

    This reduces runtime and allows for more complex retrieval strategies without excessive overhead.

7. Local Evaluation Alignment

Issue
Our local evaluation (evaluate_local.py) computes chrF between the retrieved context and the ground‑truth middle. While this is a quick proxy, the final score depends on generated code from three different models.

Improvement

    If possible, set up a local pipeline that actually runs the three target models (or smaller stand‑ins) with the correct FIM formatting. This would give a more accurate signal for tuning.

    At minimum, ensure that our context formatting exactly matches what the evaluators expect (they replace <|file_sep|> with the model‑specific token, so we are good).

Summary

We have a solid baseline that meets the submission format. The most critical immediate fix is to reverse the order of chunks to survive left truncation. Beyond that, focusing on token‑aware selection, better query formulation, and dependency‑aware retrieval will bring us closer to state‑of‑the‑art performance. Implementing these improvements step by step, while validating against the provided data, should lift the average chrF score significantly.