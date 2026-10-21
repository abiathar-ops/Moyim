"""Step 8 - Open-set predictions v2: better Unknown rule + self-trained classifier.

Self-contained (runs on any branch). Inputs, all produced earlier:
  outputs/03_birdnet/birdnet_windows.npz         BirdNET v2.4 embeddings (step 6)
  outputs/03_birdnet/birdnet_species_clip.parquet BirdNET species scores (step 7)
  outputs/folds.csv                               group folds (step 3)
  ../DSC_Dataset/train/metadata.csv, ../DSC_Dataset/sample_submission.csv

What changed vs step 6/7 of the notebook (all checked 2026-10-19):
1. Unknown score = average of two novelty scores (percentile ranks):
   a) embedding novelty: cosine distance to the K=10 nearest training clips in a
      standardised 64-d PCA space, and that distance relative to the distance to
      the 10 nearest other test clips (a clip far from training but inside a tight
      test-only group is likely a new species);
   b) the same, computed on BirdNET's species-score profile (sqrt of the max
      confidence per species, species seen in >= 5 clips). BirdNET's species
      outputs are used only as features describing what a clip sounds like; they
      never decide a label.
   Then each clip's score is blended 50/50 with the median score of its test
   cluster (40 clusters, cosine, average linkage), so similar clips get similar
   decisions.
   Measured on held-out training recordings, this rejects ~11% of genuine known
   birds at 50% Unknown, vs ~22% for the step-7 score.
2. Known-species labels come from logistic regression self-trained on test
   clips: train, then add test clips that are clearly not Unknown (novelty in the
   lower half) and predicted with >= 0.9 probability as pseudo-labelled examples,
   and retrain. Group CV: Cendrawasih recall 0.82 -> 0.90, macro-F1 0.864 ->
   0.875 (0.945 vs 0.899 for the RBF-SVM when re-weighted to the test class mix,
   which has far more Cendrawasih than training).

Outputs  outputs/03_birdnet/submission_v2_r{45,50,55}.csv   (share of Unknown)
         outputs/03_birdnet/submission_v2_embonly_r50.csv   (no species features)
         outputs/03_birdnet/test_scores_v2.csv
Usage    python 08_open_set_v2.py [--emb birdnet|perch|both]
         --emb perch/both needs outputs/03_birdnet/perch_windows.npz (step 6b);
         files then get a _perch or _both suffix.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PIPE = Path(__file__).resolve().parents[1]
DATA = PIPE.parent / "DSC_Dataset"
OUT = PIPE / "outputs" / "03_birdnet"
SEED, K, NPC, N_CLUSTERS = 42, 10, 64, 40
RATES = [0.45, 0.50, 0.55]


def unit(A):
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)


def pct(v):
    return pd.Series(v).rank(pct=True).to_numpy()


def knn_dist(A, B, k, exclude_self=False):
    S = A @ B.T
    if exclude_self:
        np.fill_diagonal(S, -1)
    return 1 - np.sort(S, axis=1)[:, -k:].mean(1)


def novelty(R, T):
    """R: unit-norm reference (train) rows, T: unit-norm test rows. Higher = more novel."""
    d_train = knn_dist(T, R, K)
    d_test = knn_dist(T, T, K, exclude_self=True)
    return (pct(d_train) + pct(d_train / np.maximum(d_test, 1e-6))) / 2


def pool(path, ids):
    z = np.load(path)
    W, fid = z["emb"], z["file_id"].astype(str)
    o = np.argsort(fid, kind="stable"); u, st = np.unique(fid[o], return_index=True); Ws = W[o]
    mean = np.add.reduceat(Ws, st) / np.diff(np.append(st, len(Ws)))[:, None]
    return pd.DataFrame(np.hstack([mean, np.maximum.reduceat(Ws, st)]), index=u).loc[ids].to_numpy()


def load(emb="birdnet"):
    folds = pd.read_csv(PIPE / "outputs" / "folds.csv")
    test_ids = pd.read_csv(DATA / "sample_submission.csv").file_id.astype(str)
    ids = list(folds.file_id) + list(test_ids)
    files = {"birdnet": ["birdnet_windows.npz"], "perch": ["perch_windows.npz"],
             "both": ["birdnet_windows.npz", "perch_windows.npz"]}[emb]
    F = np.hstack([pool(OUT / f, ids) for f in files])
    X, Xt = F[:len(folds)], F[len(folds):]
    c = pd.read_parquet(OUT / "birdnet_species_clip.parquet")
    sp = c.pivot_table(index="file_id", columns="species_name", values="confidence", aggfunc="max").fillna(0)
    sp = sp.loc[:, (sp > 0.05).sum() >= 5]
    S, St = sp.reindex(folds.file_id).fillna(0).to_numpy(), sp.reindex(test_ids).fillna(0).to_numpy()
    return folds, test_ids, X, Xt, S, St


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", choices=["birdnet", "perch", "both"], default="birdnet")
    a = ap.parse_args()
    sfx = "" if a.emb == "birdnet" else f"_{a.emb}"
    folds, test_ids, X, Xt, S, St = load(a.emb)
    y = folds.label.to_numpy(str)

    sc = StandardScaler().fit(X)
    pca = PCA(NPC, random_state=SEED).fit(sc.transform(np.vstack([X, Xt])))
    Z, Zt = unit(pca.transform(sc.transform(X))), unit(pca.transform(sc.transform(Xt)))
    nov_emb = novelty(Z, Zt)
    nov_sp = novelty(unit(np.sqrt(S)), unit(np.sqrt(St)))
    nov_combo = (nov_emb + nov_sp) / 2

    cl = AgglomerativeClustering(n_clusters=N_CLUSTERS, metric="cosine", linkage="average").fit_predict(Zt)
    smooth = lambda s: 0.5 * s + 0.5 * pd.Series(s).groupby(cl).transform("median").to_numpy()
    unknown_score = smooth(nov_combo)
    unknown_score_emb = smooth(nov_emb)

    make = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))
    clf = make().fit(X, y)
    P = clf.predict_proba(Xt)
    pseudo = (nov_combo <= np.quantile(nov_combo, 0.5)) & (P.max(1) >= 0.9)
    lab0 = clf.classes_[P.argmax(1)]
    clf2 = make().fit(np.vstack([X, Xt[pseudo]]), np.r_[y, lab0[pseudo]])
    P2 = clf2.predict_proba(Xt)
    closed = clf2.classes_[P2.argmax(1)]
    print("pseudo-labelled test clips added:", pd.Series(lab0[pseudo]).value_counts().to_dict())

    def write(score, r, name):
        lab = np.where(score >= np.quantile(score, 1 - r), "Unknown", closed)
        pd.DataFrame({"file_id": test_ids, "label": lab}).to_csv(OUT / name, index=False)
        print(f"{name}: {pd.Series(lab).value_counts(normalize=True).round(3).to_dict()}")

    for r in RATES:
        write(unknown_score, r, f"submission_v2{sfx}_r{int(r * 100)}.csv")
    write(unknown_score_emb, 0.50, f"submission_v2{sfx}_embonly_r50.csv")

    pd.DataFrame({"file_id": test_ids, "closed_pred": closed, "unknown_score": unknown_score,
                  "nov_embedding": nov_emb, "nov_species": nov_sp, "cluster": cl,
                  **{f"p_{c}": P2[:, i] for i, c in enumerate(clf2.classes_)}}).to_csv(OUT / f"test_scores_v2{sfx}.csv", index=False)


if __name__ == "__main__":
    main()
