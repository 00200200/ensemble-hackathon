import lightgbm as lgb
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

N_CLASSES = 500
N_FOLDS = 5
N_PARALLEL_CLASSES = 32


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_train, n_test = len(X_train), len(X_test)
    oof_probs = np.zeros((n_train, N_CLASSES), dtype=np.float32)
    test_probs = np.zeros((n_test, N_CLASSES), dtype=np.float32)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n  === Fold {fold + 1}/{N_FOLDS} ===")
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        results = Parallel(n_jobs=N_PARALLEL_CLASSES, prefer="threads")(
            delayed(_train_one_class)(cls, X_tr, y_tr, X_val, y_val, X_test)
            for cls in range(N_CLASSES)
        )

        fold_test = np.zeros((n_test, N_CLASSES), dtype=np.float32)
        for cls, val_p, test_p in results:
            oof_probs[val_idx, cls] = val_p
            fold_test[:, cls] = test_p

        test_probs += fold_test / N_FOLDS
        print(f"  Fold {fold + 1} done.")

    return oof_probs, test_probs


def optimize_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    print("  Finding best thresholds...")
    thresholds = np.full(N_CLASSES, 0.5)
    for cls in range(N_CLASSES):
        yt, yp = y_true[:, cls], y_prob[:, cls]
        if yt.sum() == 0:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in np.arange(0.05, 0.95, 0.02):
            f = f1_score(yt, (yp >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds[cls] = best_t
    return thresholds


def _train_one_class(
    cls: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    yt, yv = y_tr[:, cls], y_val[:, cls]
    pos, neg = yt.sum(), len(yt) - yt.sum()
    scale_pos_weight = neg / pos if (pos > 0 and neg > 0) else 1.0

    model = lgb.LGBMClassifier(
        verbosity=-1,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=5,
        scale_pos_weight=scale_pos_weight,
        n_jobs=2,
        random_state=42,
    )
    model.fit(
        X_tr,
        yt,
        eval_set=[(X_val, yv)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    return cls, model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]
