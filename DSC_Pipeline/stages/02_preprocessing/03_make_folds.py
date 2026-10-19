"""Step 3 - Group-aware cross-validation folds.

Every source_group is a single species, so a random split lets the model
score well by recognising recordings instead of birds. We hold out whole
groups (StratifiedGroupKFold), trying many seeds and keeping the split whose
validation folds have the most even class mix.

Outputs  outputs/folds.csv  (file_id, label, source_group, fold)
"""
import _common  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C


def main():
    meta = pd.read_csv(C.TRAIN_META)
    y, g = meta.label.values, meta.source_group.values
    overall = meta.label.value_counts(normalize=True)
    best = None
    for seed in range(C.FOLD_SEED_SEARCH):
        cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=seed)
        fold = np.empty(len(meta), int)
        for k, (_, va) in enumerate(cv.split(meta, y, g)):
            fold[va] = k
        tab = pd.crosstab(fold, y)
        if (tab == 0).any().any():
            continue                         # every fold must validate every class
        prop = tab.div(tab.sum(1), axis=0)
        score = float(((prop - overall) ** 2).values.sum()) + 1e-4 * tab.sum(1).std()
        if best is None or score < best[0]:
            best = (score, seed, fold.copy())
    if best is None:
        raise RuntimeError("no seed produced folds that contain every class")
    _, seed, fold = best
    meta["fold"] = fold
    meta[["file_id", "label", "source_group", "fold"]].to_csv(C.OUT_DIR / "folds.csv", index=False)

    print(f"chosen seed {seed}; validation clips per fold x class:")
    print(pd.crosstab(meta.fold, meta.label, margins=True).to_string())
    print("\nvalidation groups per fold x class:")
    print(meta.groupby(["fold", "label"]).source_group.nunique().unstack(fill_value=0).to_string())
    assert meta.groupby("source_group").fold.nunique().max() == 1, "a group spans two folds!"
    print("\ncheck passed: no source_group appears in more than one fold")
    print("note: metrics should be computed on pooled out-of-fold predictions;"
          " single folds have very few Cendrawasih clips.")


if __name__ == "__main__":
    main()
