Read @paper.md
From what we know there we have beter idea then presented in featured papers, therefore we have to implement it.

But I There I have huge concern, because almost always where I am letting ai agent code me something it breaks and make sloppy code. 
Therefore I have planned how we should approch it that we will succeed and dont break a thing while coding. Write importatnt information into AGENTS.md and Proceed with code phase by phase. Write important notes, thought into .md files for later review, recall.
I have provided @JetBrains.md for what they expect as an output, the example submission that they have provided and data folder where the data is:
It containes answers (middle part of the edit), suffix-preffix of the edit and big directory which containes multiple independent python repos on which you can test usability and performance
Here is the plan:


Phase 1: Structural Ingestion & AST Partitioning

Goal: Transform the raw repository into a structured, queryable graph and vector database.

    Environment Setup:

        Initialize a Neo4j (or Memgraph) instance for the graph layer and Qdrant or Pinecone for the vector layer.

        Integrate the tree-sitter-language-pack to support over 160 languages without manual compiler setup ``.

    File Parsing:

        Implement a recursive walker that identifies code files and excludes build artifacts (node_modules, venv, etc.).

        Use Tree-sitter to parse each file into an Abstract Syntax Tree (AST) ``.

    Entity Extraction:

        Partition the code into discrete "Opaque Blobs": Functions, Classes, and Constants ``.

        For each entity, record its metadata: file_path, line_range, signature, and raw_code.

    Graph Construction:

        Create nodes in Neo4j for each entity.

        Establish edges based on static analysis: DEFINES, CONTAINS, IMPORTS, and CALLS (intra-file and cross-file) ``.

Phase 2: LLM-Driven Functional Abstraction

Goal: Compress raw code into "Functional Abstracts" to maximize context window efficiency.

    Abstraction Prompting:

        Pipeline each extracted function/class through a fast, instruction-tuned model (e.g., GPT-4o-mini).

        Prompt Requirement: "Summarize the core logic of this function in 2 sentences and list all external functions/classes it invokes (dependencies)" ``.
        
    Data Persistence:

        Store the resulting Functional Abstract in the Neo4j node as a property.

        Generate a dense vector embedding of the Abstract and a sparse representation (for keyword search) of the Signature ``.

    Indexing:

        Upsert the combined data (embeddings + IDs) into the vector database.

Phase 3: Hybrid Retrieval Engine

Goal: Implement a retrieval mechanism that matches both semantic intent and exact identifier names.

    The Formula:

        Implement a convex combination scorer: S(q,d)=α⋅Dense(q,d)+(1−α)⋅BM25(q,d).

        Set the initial α=0.5 to balance meaning and exact match ``.

    Parallel Execution:

        For every edit task, run a vector search (semantic) and a BM25/keyword search (identifiers) in parallel ``.

    Alpha Auto-Tuning (Optional):

        Route "Identifier-heavy" queries (e.g., "where is the UserAuth logic?") to α<0.3 and "Conceptual" queries (e.g., "how do we handle errors?") to α>0.7 ``.

Phase 4: Topological Context Assembly

Goal: Use the repository's "Call Graph" to provide the LLM with the necessary downstream logic.

    Seed Selection:

        Retrieve the Top-K (e.g., K=5) nearest neighbors from Phase 3.

    Neighborhood Expansion:

        For each Top-K node, perform a 1-hop traversal in Neo4j:

            Retrieve the full code body of the node itself.

            Retrieve the Functional Abstracts of its direct callees (downstream dependencies) ``.

    Centrality Filtering:

        If a function has a very high degree (e.g., a utility function called by 100 others), apply a degree penalty to prevent context window overflow ``.

Phase 5: Dynamic Style Engine (KL-Gated)

Goal: Keep the LLM's coding style consistent with the specific module being edited.

    Style Profiling:

        Sample 5 functions from the active file and 20 from the repository.

        Generate a "Style Profile" (e.g., "uses camelCase, favors functional recursion, no type hints").

    Divergence Measurement:

        Calculate the Kullback-Leibler (KL) Divergence between the local file's style distribution Plocal​ and the current system prompt's style Pglobal​ ``.

    Threshold Gating:

        Rule: Update the style-augmentation in the system prompt only if DKL​>ϵ (where ϵ is a small tuned threshold like 0.15) ``. This prevents "jittery" prompt changes and saves compute.

Phase 6: Inference & Feedback Loop

Goal: Execute the edit and refine the system based on performance.

    Prompt Assembly:

        Combine: + + +.

    Generation:

        Execute the inference using a high-reasoning model (e.g., GPT-4o or Claude 3.5 Sonnet) [1].

    Verification:

        Benchmark the system using the Long Code Arena (LCA) task suite to ensure it beats the SOTA baseline of ~0.7 chRF [1, 2].

Implementation Checklist for Developers

    [ ] Language Pack: Use tree-sitter-language-pack to avoid C-compiler errors during install ``.

    [ ] Zero Handling: When calculating KL Divergence, use smoothing (add a small ϵ to Q) to avoid division-by-zero errors ``.

    [ ] Graph Speed: For 10k+ files, ensure graph lookups are under 30 seconds by indexing the node_id in Neo4j ``.

    [ ] Reranking: After retrieving nodes from the graph, use a cross-encoder to perform a final re-ordering of the context before sending it to the LLM ``.