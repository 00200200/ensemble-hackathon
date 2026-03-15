import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.chebi.features import extract_features
from src.chebi.hierarchy import (
    build_children_map,
    enforce_hierarchy,
    parse_obo,
    topological_order,
)
from src.chebi.model import optimize_thresholds, train
from src.chebi.submit import save_submission, upload

warnings.filterwarnings("ignore")

DATA_DIR = "data"
TRAIN_FILE = f"{DATA_DIR}/chebi_dataset_train.parquet"
TEST_FILE = f"{DATA_DIR}/chebi_dataset_test_empty.parquet"
OBO_FILE = f"{DATA_DIR}/chebi_classes.obo"
SUBMISSION_FILE = f"{DATA_DIR}/submission.parquet"

N_CLASSES = 500
CLASS_COLS = [f"class_{i}" for i in range(N_CLASSES)]


def main() -> None:
    print("=== ChEBI Classification Pipeline ===\n")

    print("Loading data...")
    train_df = pd.read_parquet(TRAIN_FILE)
    test_df = pd.read_parquet(TEST_FILE)
    y_train = train_df[CLASS_COLS].values.astype(np.float32)
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    print("\nParsing OBO hierarchy...")
    parents = parse_obo(OBO_FILE)
    children_map = build_children_map(parents)
    topo = topological_order(children_map)
    print(f"  DAG edges: {sum(len(v) for v in children_map.values())}")

    print("\nExtracting molecular features...")
    X_train = extract_features(train_df["SMILES"].tolist())
    X_test = extract_features(test_df["SMILES"].tolist())

    print("\nTraining LightGBM (5-fold CV)...")
    oof_probs, test_probs = train(X_train, y_train, X_test)
    f1_default = f1_score(y_train, (oof_probs >= 0.5).astype(int), average="macro", zero_division=0)
    print(f"\n  OOF macro F1 (threshold=0.5): {f1_default:.4f}")

    print("\nOptimizing per-class thresholds...")
    thresholds = optimize_thresholds(y_train, oof_probs)
    f1_opt = f1_score(
        y_train, (oof_probs >= thresholds).astype(int), average="macro", zero_division=0
    )
    print(f"  OOF macro F1 (opt. thresholds): {f1_opt:.4f}")

    print("\nEnforcing hierarchy...")
    test_probs = enforce_hierarchy(test_probs, children_map, topo)
    test_preds = enforce_hierarchy((test_probs >= thresholds).astype(int), children_map, topo)
    incons = sum(
        int(np.sum((test_preds[:, c] == 1) & (test_preds[:, p] == 0)))
        for p, clist in children_map.items()
        for c in clist
    )
    print(f"  Hierarchy inconsistencies: {incons}")

    print("\nSaving and uploading submission...")
    sub = save_submission(test_df, test_preds, CLASS_COLS, SUBMISSION_FILE)
    upload(sub)


if __name__ == "__main__":
    main()
