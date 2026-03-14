import argparse
import itertools
import json
import tempfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from nltk.translate.chrf_score import corpus_chrf

from main import run_pipeline


def load_subset(
    source_path: Path,
    answers_path: Path,
    max_samples: int,
    work_dir: Path,
) -> Tuple[Path, Path]:
    """
    Create subset JSONL files for source and answers, keeping the first
    `max_samples` aligned examples.
    """
    sub_source = work_dir / "source_subset.jsonl"
    sub_answers = work_dir / "answers_subset.jsonl"

    with source_path.open("r", encoding="utf-8") as fs, answers_path.open(
        "r",
        encoding="utf-8",
    ) as fa, sub_source.open(
        "w",
        encoding="utf-8",
    ) as ws, sub_answers.open(
        "w",
        encoding="utf-8",
    ) as wa:
        for i, (ls, la) in enumerate(zip(fs, fa)):
            if i >= max_samples:
                break
            ws.write(ls)
            wa.write(la)

    return sub_source, sub_answers


def evaluate_chrf_local(
    predictions_file: Path,
    ground_truth_file: Path,
    source_file: Path,
) -> float:
    """
    Local copy of the evaluate_local.evaluate_chrf logic, returning a numeric score.
    """
    # Load ground truths with IDs from source
    truths = {}
    with source_file.open("r", encoding="utf-8") as fs, ground_truth_file.open(
        "r",
        encoding="utf-8",
    ) as fg:
        for line_s, line_g in zip(fs, fg):
            data_s = json.loads(line_s)
            data_g = json.loads(line_g)
            truths[data_s["id"]] = data_g["middle"]

    # Load predictions
    preds = {}
    with predictions_file.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            preds[data["id"]] = data["context"]

    refs: List[List[str]] = []
    hyps: List[str] = []

    for qid, middle_text in truths.items():
        if qid in preds:
            refs.append([middle_text])
            hyps.append(preds[qid])

    if not hyps:
        return 0.0

    score = corpus_chrf(refs, hyps)
    return float(score)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search hyperparameters for the FIM context pipeline.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("Task 2 dataset EnsembleAI 2026/python-public.jsonl"),
        help="Path to the source dataset JSONL.",
    )
    parser.add_argument(
        "--answers",
        type=Path,
        default=Path("Task 2 dataset EnsembleAI 2026/answers-python-public.jsonl"),
        help="Path to the ground-truth answers JSONL.",
    )
    parser.add_argument(
        "--archives-root",
        type=Path,
        required=True,
        help="Root directory containing repository archives.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Number of validation samples to use from the dataset.",
    )

    args = parser.parse_args()

    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    top_ks = [5, 15, 30]
    token_budgets = [6000, 8000]

    results = []

    with tempfile.TemporaryDirectory() as tmpdir_str:
        work_dir = Path(tmpdir_str)

        # Prepare a small validation subset of source and answers once.
        sub_source, sub_answers = load_subset(
            args.dataset,
            args.answers,
            args.max_samples,
            work_dir,
        )

        for alpha, top_k, token_budget in itertools.product(
            alphas,
            top_ks,
            token_budgets,
        ):
            out_path = work_dir / f"context_a{alpha}_k{top_k}_b{token_budget}.jsonl"

            run_pipeline(
                input_path=sub_source,
                archives_root=args.archives_root,
                output_path=out_path,
                token_budget=token_budget,
                top_k=top_k,
                alpha=alpha,
            )

            score = evaluate_chrf_local(
                predictions_file=out_path,
                ground_truth_file=sub_answers,
                source_file=sub_source,
            )

            results.append(
                {
                    "alpha": alpha,
                    "top_k": top_k,
                    "token_budget": token_budget,
                    "chrf": score,
                },
            )

    df = pd.DataFrame(results)
    df = df.sort_values(by="chrf", ascending=False).reset_index(drop=True)
    print("\nGrid Search Results (sorted by chrF):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if not df.empty:
        best = df.iloc[0]
        print("\nBest Configuration:")
        print(
            f"alpha={best['alpha']}, top_k={int(best['top_k'])}, "
            f"token_budget={int(best['token_budget'])}, chrF={best['chrf']:.4f}",
        )


if __name__ == "__main__":
    main()

