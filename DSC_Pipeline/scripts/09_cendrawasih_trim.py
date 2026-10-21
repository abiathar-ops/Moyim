"""Step 9 - Trim over-predicted Cendrawasih (hypothesis test, 2026-10-20).

Evidence: in training, Perch's top species for Cendrawasih clips include
birds-of-paradise (20% in top-5) and almost never parrots/cockatoos/shamas
(2% in top-3). Test clips that v2 labels Cendrawasih (133, vs 4.5% of
training) are dominated by White-rumped Shama, cockatoos, macaws and similar,
and Cendrawasih "fingerprint" outputs fire on them at about a third of the
training rate. So most of them are likely Unknown birds that sound loud and
low like Cendrawasih.

Rule, starting from submission_v2_both_r45.csv (step 8 --emb both):
1. Rank clips labelled Cendrawasih by Cendrawasih-likeness = rank of the
   logistic-regression probability (BirdNET+Perch embeddings, trained on all
   training clips) + rank of closeness to training Cendrawasih (mean cosine
   distance to the 5 nearest, standardised 64-d PCA space). Keep the top 45;
   the rest become Unknown.
2. Veto: a kept clip becomes Unknown if Perch ranks a parrot/cockatoo genus or
   White-rumped Shama in its top 3 species. On training clips this fires on
   2.4% of Cendrawasih and 0.2% of the other two species.
Perch species scores are used as a referee feature here; say so in the write-up.

Inputs   outputs/03_birdnet/{birdnet_windows,perch_windows,perch_species_clip}.npz,
         submission_v2_both_r45.csv, outputs/folds.csv
Output   outputs/03_birdnet/submission_v3_cend45_veto.csv
Usage    python 09_cendrawasih_trim.py [--keep 45]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PIPE = Path(__file__).resolve().parents[1]
DATA = PIPE.parent / "DSC_Dataset"
OUT = PIPE / "outputs" / "03_birdnet"
PARROTS = ("Cacatua", "Ara", "Psittacula", "Eclectus", "Probosciger", "Tanygnathus", "Lorius",
           "Trichoglossus", "Loriculus", "Psittacus", "Amazona")


def pool(path, ids):
    z = np.load(path); W, fid = z["emb"], z["file_id"].astype(str)
    o = np.argsort(fid, kind="stable"); u, st = np.unique(fid[o], return_index=True); Ws = W[o]
    mean = np.add.reduceat(Ws, st) / np.diff(np.append(st, len(Ws)))[:, None]
    return pd.DataFrame(np.hstack([mean, np.maximum.reduceat(Ws, st)]), index=u).loc[ids].to_numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--keep", type=int, default=45); a = ap.parse_args()
    folds = pd.read_csv(PIPE / "outputs" / "folds.csv")
    base = pd.read_csv(OUT / "submission_v2_both_r45.csv")
    ids = list(folds.file_id) + list(base.file_id)
    F = np.hstack([pool(OUT / "birdnet_windows.npz", ids), pool(OUT / "perch_windows.npz", ids)])
    X, Xt, y = F[:len(folds)], F[len(folds):], folds.label.to_numpy(str)

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")).fit(X, y)
    p_c = clf.predict_proba(Xt)[:, list(clf.classes_).index("Cendrawasih")]
    sc = StandardScaler().fit(X); pca = PCA(64, random_state=42).fit(sc.transform(np.vstack([X, Xt])))
    unit = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)
    Zr, Zt = unit(pca.transform(sc.transform(X))), unit(pca.transform(sc.transform(Xt)))
    d_c = 1 - np.sort(Zt @ Zr[y == "Cendrawasih"].T, axis=1)[:, -5:].mean(1)
    likeness = pd.Series(p_c).rank(pct=True).to_numpy() + pd.Series(-d_c).rank(pct=True).to_numpy()

    lab = base.label.to_numpy().copy()
    idx = np.flatnonzero(lab == "Cendrawasih")
    lab[idx[np.argsort(-likeness[idx])][a.keep:]] = "Unknown"

    z = np.load(OUT / "perch_species_clip.npz")
    L = pd.DataFrame(z["logit"].astype(np.float32), index=z["file_id"], columns=z["species"]).loc[base.file_id]
    cols = [c for c in L.columns if c.split()[0] in PARROTS] + ["Copsychus malabaricus"]
    veto = (L.rank(axis=1, ascending=False)[cols].min(axis=1) <= 3).to_numpy()
    lab[(lab == "Cendrawasih") & veto] = "Unknown"

    out = OUT / f"submission_v3_cend{a.keep}_veto.csv"
    pd.DataFrame({"file_id": base.file_id, "label": lab}).to_csv(out, index=False)
    print(out.name, pd.Series(lab).value_counts().to_dict())


if __name__ == "__main__":
    main()
