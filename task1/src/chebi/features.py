import os

import numpy as np
from skfp.fingerprints import (
    ECFPFingerprint,
    MACCSFingerprint,
    RDKit2DDescriptorsFingerprint,
)
from skfp.preprocessing import MolFromSmilesTransformer
from sklearn.pipeline import make_union

N_JOBS = -1


def extract_features(smiles_list: list[str]) -> np.ndarray:
    print(f"  Converting {len(smiles_list)} SMILES to molecules...")
    mols = _smiles_to_mols(smiles_list)

    print("  Computing fingerprints...")
    X = make_union(
        ECFPFingerprint(fp_size=2048, radius=2, count=True, n_jobs=N_JOBS),
        MACCSFingerprint(n_jobs=N_JOBS, count=True),
        RDKit2DDescriptorsFingerprint(n_jobs=N_JOBS),
    ).transform(mols)

    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.nan_to_num(np.array(X, dtype=np.float32))
    print(f"  Feature matrix: {X.shape}")
    return X


def _smiles_to_mols(smiles_list: list[str]):
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        return MolFromSmilesTransformer(n_jobs=N_JOBS).transform(smiles_list)
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
