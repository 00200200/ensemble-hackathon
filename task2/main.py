import argparse
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm

from chunker import extract_repo, process_repository
from retriever import Retriever

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AutoTokenizer = None  # type: ignore


FILE_SEP_TOKEN = "<|file_sep|>"

# Conservative shared budget so that no model needs to further trim.
DEFAULT_TOKEN_BUDGET = 8000

# Prefer using the Qwen2.5-Coder tokenizer for budgeting, but allow overrides.
DEFAULT_TOKENIZER_MODEL = os.getenv(
    "TASK2_TOKENIZER_MODEL",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
)

if AutoTokenizer is not None:
    try:
        TOKENIZER = AutoTokenizer.from_pretrained(
            DEFAULT_TOKENIZER_MODEL,
            trust_remote_code=True,
        )
    except Exception:
        TOKENIZER = None
else:
    TOKENIZER = None


@lru_cache(maxsize=10000)
def count_tokens(text: str) -> int:
    """
    Token counter aligned with the target model tokenizer when available.

    Falls back to a simple whitespace-based heuristic if the tokenizer cannot
    be loaded. Cached for efficiency across repeated chunk texts.
    """
    if not text:
        return 0

    if TOKENIZER is None:
        return len(text.split())

    try:
        # Avoid adding special tokens, we just care about raw length.
        return len(TOKENIZER.encode(text, add_special_tokens=False))
    except Exception:
        return len(text.split())


def extract_keywords(prefix: str, suffix: str) -> str:
    """
    Heuristically extract informative identifiers from prefix/suffix:
    function names, class names, and imported modules.
    """
    import re

    text = f"{prefix}\n{suffix}"

    func_names = re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    class_names = re.findall(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:\(]", text)
    imports = re.findall(r"^\s*import\s+([A-Za-z0-9_\.]+)", text, re.MULTILINE)
    from_imports = re.findall(
        r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+([A-Za-z0-9_\*,\s]+)",
        text,
        re.MULTILINE,
    )

    imported_symbols: List[str] = []
    for module, names in from_imports:
        imported_symbols.append(module)
        imported_symbols.extend(
            n.strip()
            for n in names.split(",")
            if n.strip() and n.strip() != "*"
        )

    keywords = sorted(
        {*(func_names or []), *(class_names or []), *imports, *imported_symbols}
    )
    return " ".join(keywords)


def build_query(prefix: str, suffix: str) -> str:
    """
    Build a richer retrieval query from prefix/suffix plus extracted keywords.
    """
    keywords = extract_keywords(prefix, suffix)

    # Use full prefix/suffix – the embedding model will internally truncate
    # if needed, but this gives it maximal signal.
    body = f"{prefix}\n\n<gap>\n\n{suffix}"

    if keywords:
        return f"{keywords}\n\n{body}"
    return body


def budgeted_selection(
    results: List[Dict[str, Any]],
    token_budget: int,
) -> List[Dict[str, Any]]:
    """
    Select a subset of retrieved chunks under a shared token budget.

    Strategy:
    - Iterate from most to least relevant so we always try to include the best
      chunks first.
    - Keep chunks while there is budget left.
    - Later, when assembling the context string, we will reverse the order so
      that the least relevant chunks appear first and the most relevant chunks
      appear last, surviving left‑truncation by the evaluator.
    """
    selected: List[Dict[str, Any]] = []
    used = 0
    for chunk in results:
        text = chunk.get("text", "")
        tokens = count_tokens(text)
        if tokens <= 0:
            continue
        if used + tokens > token_budget:
            continue
        selected.append(chunk)
        used += tokens
    return selected


def expand_with_dependencies(
    base_chunks: List[Dict[str, Any]],
    all_chunks: List[Dict[str, Any]],
    token_budget: int,
    already_used_tokens: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    1‑hop dependency expansion within the remaining token budget.

    - Callees: for each chunk, add functions/classes it calls (via `calls`).
    - Callers: also add functions/classes that call the selected chunk.
    - Class hierarchy: for class chunks, add base classes (via `bases`).

    When the number of candidate dependencies exceeds the remaining token
    budget, we rank candidates by a simple degree-centrality metric:
    how many base chunks depend on them. Higher centrality is expanded first.
    """
    name_to_chunk: Dict[str, Dict[str, Any]] = {}
    for c in all_chunks:
        name = c.get("name")
        if not name or name in name_to_chunk:
            continue
        name_to_chunk[name] = c

    # Build a reverse index for callers: callee_name -> [caller_chunks]
    callers: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_chunks:
        for callee in c.get("calls") or []:
            if not callee:
                continue
            if callee not in name_to_chunk:
                continue
            callers.setdefault(callee, []).append(c)

    selected_ids = {id(c) for c in base_chunks}
    used = already_used_tokens

    # Collect all 1-hop dependency candidates keyed by the chunk object.
    centrality: Dict[int, int] = {}
    candidate_chunks: Dict[int, Dict[str, Any]] = {}

    def register_candidate(dep_chunk: Dict[str, Any], from_base: Dict[str, Any]) -> None:
        dep_id = id(dep_chunk)
        if dep_id in selected_ids:
            return
        candidate_chunks[dep_id] = dep_chunk
        centrality[dep_id] = centrality.get(dep_id, 0) + 1

    for base in base_chunks:
        # 1) Outgoing calls (callees).
        for name in base.get("calls") or []:
            target = name_to_chunk.get(name)
            if target:
                register_candidate(target, base)

        # 2) Incoming edges (callers).
        base_name = base.get("name")
        if base_name and base_name in callers:
            for caller in callers[base_name]:
                register_candidate(caller, base)

        # 3) Class inheritance.
        if base.get("type") == "class":
            for base_name in base.get("bases") or []:
                base_chunk = name_to_chunk.get(base_name)
                if base_chunk:
                    register_candidate(base_chunk, base)

    # Rank candidates by centrality (desc), then by shorter length (asc).
    ranked_ids = sorted(
        candidate_chunks.keys(),
        key=lambda cid: (
            -centrality.get(cid, 0),
            count_tokens(candidate_chunks[cid].get("text", "")),
        ),
    )

    extras: List[Dict[str, Any]] = []
    for cid in ranked_ids:
        chunk = candidate_chunks[cid]
        tokens = count_tokens(chunk.get("text", ""))
        if tokens <= 0:
            continue
        if used + tokens > token_budget:
            continue
        extras.append(chunk)
        selected_ids.add(cid)
        used += tokens

    return extras, used


def format_context(
    chunks_in_order: Iterable[Dict[str, Any]],
    file_sep_token: str = FILE_SEP_TOKEN,
) -> str:
    """
    Turn a list of chunks into a single context string separated by
    `<|file_sep|>` tokens, including minimal file path metadata.
    """
    parts: List[str] = []
    for chunk in chunks_in_order:
        rel_path = chunk.get("rel_filepath") or chunk.get("filepath") or "unknown"
        text = chunk.get("text", "")
        parts.append(f"{file_sep_token}{rel_path}\n{text}")
    return "\n".join(parts)


def resolve_archive_path(archives_root: Path, archive_name: str) -> Path:
    """
    Given the 'archive' field from the dataset and a root directory, resolve
    the expected path to the zip file.
    """
    return archives_root / archive_name


def build_dependency_order(
    all_selected: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Given the final set of selected chunks S (initial retrieval + expansion),
    construct a dependency DAG and return a topologically sorted list of
    chunks. If u depends on v (via calls or bases), v will appear before u
    in the returned order.

    Cycles are broken conservatively using a DFS stack check: edges that
    would create a cycle are ignored.
    """
    if not all_selected:
        return []

    idx_by_id: Dict[int, int] = {id(c): i for i, c in enumerate(all_selected)}

    # Map names to indices within S for quick dependency lookups.
    name_to_indices: Dict[str, List[int]] = {}
    for i, c in enumerate(all_selected):
        name = c.get("name")
        if not name:
            continue
        name_to_indices.setdefault(name, []).append(i)

    # Build adjacency list: u -> {v indices} meaning "u depends on v".
    graph: List[Set[int]] = [set() for _ in all_selected]
    for u_idx, chunk in enumerate(all_selected):
        deps: List[str] = []
        deps.extend(chunk.get("calls") or [])
        deps.extend(chunk.get("bases") or [])

        for dep_name in deps:
            for v_idx in name_to_indices.get(dep_name, []):
                if v_idx == u_idx:
                    continue
                graph[u_idx].add(v_idx)

    order: List[int] = []
    state: List[int] = [0] * len(all_selected)  # 0=unvisited, 1=visiting, 2=done

    def dfs(u: int) -> None:
        if state[u] == 2:
            return
        if state[u] == 1:
            # Detected a cycle; we break it by ignoring this back-edge.
            return
        state[u] = 1
        for v in graph[u]:
            dfs(v)
        state[u] = 2
        order.append(u)

    # Start DFS from all nodes; those with more outgoing edges (more deps)
    # will naturally appear later, but all dependencies are visited first.
    for i in range(len(all_selected)):
        if state[i] == 0:
            dfs(i)

    # `order` is in a valid topological order: dependencies before dependents.
    return [all_selected[i] for i in order]


def run_pipeline(
    input_path: Path,
    archives_root: Path,
    output_path: Path,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    top_k: int = 10,
    alpha: float = 0.5,
) -> None:
    retriever = Retriever()

    # In‑memory cache within a single run:
    # archive_name -> table_name
    repo_cache: Dict[str, str] = {}

    # Persistent on-disk cache of chunked repositories to speed up
    # experimentation across runs.
    cache_root = Path(
        os.getenv(
            "TASK2_CACHE_DIR",
            Path.home() / ".cache" / "task2",
        ),
    )
    chunks_cache_dir = cache_root / "chunks"
    chunks_cache_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w",
        encoding="utf-8",
    ) as fout:
        for line in tqdm(fin, desc="Building contexts"):
            example = json.loads(line)
            qid = example["id"]
            archive_name = example.get("archive")
            prefix = example.get("prefix", "")
            suffix = example.get("suffix", "")

            if not archive_name:
                # Fallback: no repo, just empty context.
                json.dump({"id": qid, "context": ""}, fout)
                fout.write("\n")
                continue

            table_name = repo_cache.get(archive_name)
            if table_name is None:
                # Extract and chunk repository, then build indexes.
                archive_path = resolve_archive_path(archives_root, archive_name)
                if not archive_path.exists():
                    # If we can't find the archive, emit an empty context but
                    # keep the pipeline running.
                    json.dump({"id": qid, "context": ""}, fout)
                    fout.write("\n")
                    continue

                cache_file = chunks_cache_dir / f"{archive_name}.jsonl"
                if cache_file.exists():
                    # Load pre-chunked repository from cache.
                    chunks: List[Dict[str, Any]] = []
                    with cache_file.open("r", encoding="utf-8") as cf:
                        for line_c in cf:
                            chunks.append(json.loads(line_c))
                else:
                    # First time seeing this repo: extract, chunk, and cache.
                    with tempfile.TemporaryDirectory() as tmpdir:
                        extract_repo(str(archive_path), tmpdir)
                        chunks = process_repository(tmpdir)
                    with cache_file.open("w", encoding="utf-8") as cf:
                        for ch in chunks:
                            cf.write(json.dumps(ch) + "\n")

                table_name = archive_name  # unique enough per repo+revision
                retriever.build_index(table_name, chunks)
                repo_cache[archive_name] = table_name

            # Build improved query.
            query = build_query(prefix, suffix)

            # Retrieve more than we will ultimately keep, then perform
            # token‑aware selection with optional cross-encoder re-ranking.
            results = retriever.search(
                table_name,
                query,
                top_k=top_k,
                alpha=alpha,
                re_rank=True,
                re_rank_top_k=max(top_k * 3, 30),
            )

            if not results:
                json.dump({"id": qid, "context": ""}, fout)
                fout.write("\n")
                continue

            selected = budgeted_selection(results, token_budget)
            used_tokens = sum(count_tokens(c.get("text", "")) for c in selected)

            # Dependency‑aware expansion (1‑hop) within remaining budget.
            all_chunks = retriever.repo_chunks.get(table_name, [])
            extras, used_tokens = expand_with_dependencies(
                selected,
                all_chunks,
                token_budget,
                already_used_tokens=used_tokens,
            )

            all_selected = selected + extras

            # Topologically sort S so that dependencies (callees/base classes)
            # always appear before the chunks that use them. Since the
            # evaluation truncates from the left, important retrieved targets
            # still tend to end up near the right, but their prerequisites are
            # never placed after them.
            chunks_for_context = build_dependency_order(all_selected)
            context = format_context(chunks_for_context, FILE_SEP_TOKEN)

            json.dump({"id": qid, "context": context}, fout)
            fout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build token‑aware, dependency‑augmented contexts for task2.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Dataset for Participants/python-public.jsonl"),
        help="Path to input dataset JSONL.",
    )
    parser.add_argument(
        "--archives-root",
        type=Path,
        required=True,
        help="Root directory containing repository zip archives referenced in the dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("context_file.jsonl"),
        help="Where to write the composed context JSONL.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help="Approximate shared token budget per example.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve before token‑aware filtering.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hybrid retrieval weight between dense and sparse scores.",
    )

    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        archives_root=args.archives_root,
        output_path=args.output,
        token_budget=args.token_budget,
        top_k=args.top_k,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
