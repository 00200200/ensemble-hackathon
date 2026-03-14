# Architectural Proposal: RAG System for Repository-Level Reasoning

## 1. Repository Database
We initialize a database containing the repository's source files, categorized by functions and overall functionalities.

## 2. AST-Based Parsing
We parse these files using an Abstract Syntax Tree (AST) parser (such as Tree-sitter) to cleanly partition the source code at the discrete function or class level.

## 3. LLM-Driven Abstraction
Each extracted function is processed by an LLM to generate a functional abstract. This includes the core logic and a dependency list (the functions invoked by the analyzed function). We persist the abstract, the raw source code, and generate a vector embedding. (Note: It is yet to be determined whether embedding the extracted keywords or the entire abstract yields a more robust latent representation).

## 4. Vector Retrieval
During an editing session, we query the vector space using the embeddings of the task's keywords to retrieve semantically nearest neighbors (similar functions and functionalities).

## 5. Context Assembly (Graph Traversal)
From the retrieved nearest neighbors, we extract the function's abstract, its complete source code body, and the abstracts of all downstream functions invoked by this nearest neighbor.

## 6. Dynamic Style Contextualization
The LLM's context window is populated with the retrieved abstracts and function bodies. The system prompt is augmented with a summary of the repository's coding conventions. This standard is iteratively updated every N∈[15,25] edits. The stylistic summary is derived by sampling and analyzing 5 function bodies from the active file and 20 from external files within the repository.

## 7. Inference
The edit generation is executed.


---

# Architectural Analysis & Suggestions

This pipeline outlines a robust Retrieval-Augmented Generation (RAG) system tailored for repository-level reasoning. To elevate this architecture to a state-of-the-art research standard, the retrieval mechanisms and the context distribution must be formalized.

## 1. The Embedding Space Dilemma (Keywords vs. Full Abstract)
In Step 3, you question what to embed. Embedding keywords maps the input to a sparse lexical space, while embedding the full abstract maps it to a dense semantic space. Relying solely on one often leads to an information bottleneck.

**Suggestion**: Implement a Dense-Sparse Hybrid Retrieval mechanism. Let $q$ be the query. The final similarity score $S$ can be a convex combination of dense cosine similarity and a sparse lexical metric like BM25:

$S(q,d) = \alpha \cdot \cos(E_{dense}(q), E_{dense}(d)) + (1-\alpha) \cdot \text{BM25}(q,d)$

where $\alpha \in [0,1]$ is a tunable hyperparameter. This ensures convergence on both exact variable name matches (sparse) and high-level architectural intent (dense).

## 2. Formalizing the Dependency Retrieval (Graph Theory)
In Step 5, retrieving the abstracts of invoked functions is essentially a 1-hop traversal of the repository's Call Graph.

**Suggestion**: Model the repository as a directed graph $G=(V,E)$, where $V$ represents functions and $E$ represents invocation dependencies. Let $A \in \{0,1\}^{|V| \times |V|}$ be the adjacency matrix. If $x \in \{0,1\}^{|V|}$ is a one-hot vector representing your retrieved nearest neighbor, the 1-hop dependencies can be retrieved via a simple matrix-vector multiplication:

$y = Ax$

The non-zero indices of $y$ directly yield the set of abstracts to append. To prevent context window overflow if a function calls dozens of utilities, you should apply a degree penalty or use PageRank-based node centrality to keep only the top $k$ most structurally significant dependencies. (Refer to the RepoCoder paper by Zhang et al., 2023, for formal proofs on iterative repository graph retrieval).

## 3. Modeling Latent Style Drift
In Step 6, updating the style prompt every $N$ edits assumes the codebase style is a stationary distribution. However, large repositories have varying local styles (e.g., frontend React components vs. backend database connectors).

**Suggestion**: Instead of a hard update every $N$ steps, model the coding style as a continuous probability distribution $P_{style}$. You can measure the Kullback-Leibler divergence $D_{KL}(P_{local} \parallel P_{global})$ between the local file's style distribution and the global repository distribution. Update the system prompt dynamically only when $D_{KL}$ exceeds a certain threshold $\epsilon$, optimizing the LLM's inference compute by avoiding redundant style-generation passes.
