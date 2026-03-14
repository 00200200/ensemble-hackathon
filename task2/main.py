import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm

from chunker import extract_repo, process_repository
from retriever import Retriever


FILE_SEP_TOKEN = "<|file_sep|>"

# Conservative shared budget so that no model needs to further trim.
DEFAULT_TOKEN_BUDGET = 8000


def simple_token_count(text: str) -> int:
    """
    Lightweight token estimator.

    We avoid heavyweight tokenizer dependencies and approximate tokens by
    whitespace‑separated words. For relative budgeting and ordering this is
    sufficient, and keeps the pipeline fast and dependency‑light.
    """
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
        tokens = simple_token_count(text)
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
    max_extra: int = 8,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Very simple 1‑hop dependency expansion:

    - Each chunk from the chunker may contain a `calls` field listing local
      functions/classes it invokes.
    - For each base chunk, we add chunks whose `name` matches any of those
      calls, subject to the remaining token budget.
    """
    name_to_chunk: Dict[str, Dict[str, Any]] = {}
    for c in all_chunks:
        name = c.get("name")
        if not name or name in name_to_chunk:
            continue
        name_to_chunk[name] = c

    selected_ids = {id(c) for c in base_chunks}
    extras: List[Dict[str, Any]] = []
    used = already_used_tokens

    for chunk in base_chunks:
        calls = chunk.get("calls") or []
        for name in calls:
            target = name_to_chunk.get(name)
            if not target:
                continue
            if id(target) in selected_ids:
                continue

            text = target.get("text", "")
            tokens = simple_token_count(text)
            if tokens <= 0:
                continue
            if used + tokens > token_budget:
                continue

            extras.append(target)
            selected_ids.add(id(target))
            used += tokens

            if len(extras) >= max_extra:
                break

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
    # archive_name -> (table_name, total_tokens_for_repo)
    repo_cache: Dict[str, str] = {}

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

                with tempfile.TemporaryDirectory() as tmpdir:
                    extract_repo(str(archive_path), tmpdir)
                    chunks = process_repository(tmpdir)

                table_name = archive_name  # unique enough per repo+revision
                retriever.build_index(table_name, chunks)
                repo_cache[archive_name] = table_name

            # Build improved query.
            query = build_query(prefix, suffix)

            # Retrieve more than we will ultimately keep, then perform
            # token‑aware selection.
            results = retriever.search(
                table_name,
                query,
                top_k=top_k,
                alpha=alpha,
            )

            if not results:
                json.dump({"id": qid, "context": ""}, fout)
                fout.write("\n")
                continue

            selected = budgeted_selection(results, token_budget)
            used_tokens = sum(simple_token_count(c.get("text", "")) for c in selected)

            # Dependency‑aware expansion (1‑hop) within remaining budget.
            all_chunks = retriever.repo_chunks.get(table_name, [])
            extras, used_tokens = expand_with_dependencies(
                selected,
                all_chunks,
                token_budget,
                already_used_tokens=used_tokens,
            )

            all_selected = selected + extras

            # Reverse order so least relevant appear first, most relevant last,
            # ensuring the best material survives left‑truncation.
            chunks_for_context = list(reversed(all_selected))
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
