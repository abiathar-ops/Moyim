"""Regenerates 03_birdnet_baseline.ipynb from the cells below."""
from pathlib import Path
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# NPC DSC 2026: BirdNET embeddings + simple classifiers (baseline)

Run this **after `02_preprocessing.ipynb`**. It reuses the group folds from step 3 (`outputs/folds.csv`), so its scores compare directly with the front-end table from step 5.

**Idea.** BirdNET v2.4 is a network already trained on thousands of bird species. We use it only as a *feature extractor*: each 3 s window of a clip becomes a 1024-number embedding. Everything after that is ours and explainable: pooling, a simple classifier, and an `Unknown` rule. BirdNET's own species predictions are not used.

| Step | What | Why |
|---|---|---|
| 1 | `06_birdnet_embeddings.py`: 1024-d embedding per 3 s window, cached | Pretrained bird features need little labelled data |
| 2 | Pool windows into one vector per clip (`mean`, `mean+max`) | Clip-level labels |
| 3 | Logistic regression, RBF-SVM, cosine kNN, shrinkage LDA on group CV (plus random CV) | Compare against `pcen_norm` (0.835 macro-F1) |
| 4 | Within-species group probe | Do the features still recognise the *recording*? |
| 5 | Leave-one-class-out (LOCO): hide one species and treat it as `Unknown` | Only honest way to tune rejection without Unknown labels |
| 6 | Threshold, test rejection share, submission | Sanity check against the ~17% "far" share of test from EDA |
""")

code(r"""
import subprocess, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

PIPELINE = Path.cwd()
assert (PIPELINE / "scripts").exists(), "open this notebook from the DSC_Pipeline/ folder"
sys.path.insert(0, str(PIPELINE))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import config as C

CLASS_COLOR = {"Bondol Jawa": "#2a78d6", "Cendrawasih": "#eb6834", "Kutilang": "#1baf7a"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold"})
pd.set_option("display.width", 160)

def run(script, *args):
    # Run one pipeline script with this notebook's Python, streaming its output
    cmd = [sys.executable, script, *map(str, args)]
    print("$", " ".join(cmd[1:]), flush=True)
    p = subprocess.Popen(cmd, cwd=PIPELINE / "scripts", stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    if p.wait() != 0:
        raise RuntimeError(f"{script} failed (exit code {p.returncode})")

OUT = C.OUT_DIR / "03_birdnet"
CLASSES = C.CLASSES
print("data:", C.DATA_DIR)
""")

md(r"""
## Step 1: Extract BirdNET embeddings

Runs as a separate script because BirdNET starts worker processes, which are unreliable when launched from inside a Jupyter kernel. The first run downloads the model (~77 MB from Zenodo) and takes several minutes on CPU. After that the cache is reused (pass `"--force"` to re-extract).

- 3 s windows with 1.5 s overlap, so a 10 s clip gives about 6 windows.
- `bandpass_fmin=0`: no high-pass, same reason as `FMIN` in `config.py` (protect low-pitched Cendrawasih and Unknown calls).
""")
code(r"""run("06_birdnet_embeddings.py")""")
code(r"""
z = np.load(OUT / "birdnet_windows.npz")
W = z["emb"]
win_df = pd.DataFrame({"file_id": z["file_id"].astype(str), "win": z["win"], "split": z["split"].astype(str)})

folds = pd.read_csv(C.OUT_DIR / "folds.csv")          # file_id, label, source_group, fold (step 3)
test_ids = pd.read_csv(C.SAMPLE_SUB).file_id.astype(str)

print("windows:", W.shape)
print(win_df.groupby("split").file_id.nunique().to_string())
print("windows per clip:", win_df.groupby("file_id").size().describe()[["min", "50%", "max"]].to_dict())
assert set(folds.file_id) <= set(win_df.file_id) and set(test_ids) <= set(win_df.file_id)
""")

md(r"""
## Step 2: One vector per clip

- `mean`: average of the window embeddings. Robust, smooths out windows that contain only noise.
- `mean+max`: mean concatenated with the element-wise max (2048-d). Keeps the strongest call evidence, which helps when the bird calls only briefly.
""")
code(r"""
def pool(how):
    order = np.argsort(win_df.file_id.to_numpy(), kind="stable")
    ids, start = np.unique(win_df.file_id.to_numpy()[order], return_index=True)
    Ws = W[order]
    mean = np.add.reduceat(Ws, start) / np.diff(np.append(start, len(Ws)))[:, None]
    feats = mean if how == "mean" else np.hstack([mean, np.maximum.reduceat(Ws, start)])
    return pd.DataFrame(feats, index=ids)

POOLED = {h: pool(h) for h in ["mean", "mean+max"]}
X_tr = {h: P.loc[folds.file_id].to_numpy() for h, P in POOLED.items()}
X_te = {h: P.loc[test_ids].to_numpy() for h, P in POOLED.items()}
y = folds.label.to_numpy(str)
groups = folds.source_group.to_numpy(str)
fold = folds.fold.to_numpy()
print({h: x.shape for h, x in X_tr.items()})
print(pd.crosstab(folds.label, folds.fold).to_string())
""")

md(r"""
## Step 3: Closed-set classifiers (3 known species)

Out-of-fold predictions on the step-3 group folds. **Random CV** is shown only to measure how much a leaky split would flatter the model; decisions use group CV.
""")
code(r"""
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, recall_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

rfold = np.empty(len(y), int)
for k, (_, va) in enumerate(StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=C.SEED).split(y, y)):
    rfold[va] = k

MODELS = {
    "logreg":  lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced")),
    "svm_rbf": lambda: make_pipeline(StandardScaler(), SVC(C=1.0, probability=True, class_weight="balanced", random_state=C.SEED)),
    "knn_cos": lambda: make_pipeline(Normalizer(), KNeighborsClassifier(n_neighbors=10, weights="distance")),
    "lda":     lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
}

def oof_proba(make, X, fold_ids):
    P = np.zeros((len(y), len(CLASSES)))
    for k in np.unique(fold_ids):
        tr, va = fold_ids != k, fold_ids == k
        m = make().fit(X[tr], y[tr])
        P[np.ix_(va, [CLASSES.index(c) for c in m.classes_])] = m.predict_proba(X[va])
    return P

rows, OOF = [], {}
for pool_name, X in X_tr.items():
    for name, make in MODELS.items():
        P = oof_proba(make, X, fold); pred = np.array(CLASSES)[P.argmax(1)]
        pred_r = np.array(CLASSES)[oof_proba(make, X, rfold).argmax(1)]
        OOF[(pool_name, name)] = P
        rec = recall_score(y, pred, labels=CLASSES, average=None)
        rows.append(dict(pool=pool_name, model=name,
                         group_cv_f1=f1_score(y, pred, average="macro"),
                         random_cv_f1=f1_score(y, pred_r, average="macro"),
                         rec_bondol=rec[0], rec_cendrawasih=rec[1], rec_kutilang=rec[2]))
res = pd.DataFrame(rows).sort_values("group_cv_f1", ascending=False).reset_index(drop=True).round(3)
res.to_csv(OUT / "closed_set_results.csv", index=False)
res
""")
code(r"""
# Reference: best hand-made front-end from step 5 (same folds, logistic-regression probe)
ref = pd.read_csv(C.OUT_DIR / "frontend_comparison.csv")
ref
""")
code(r"""
BEST = tuple(res.loc[0, ["pool", "model"]])
print("best:", BEST)
pred = np.array(CLASSES)[OOF[BEST].argmax(1)]
display(pd.DataFrame(confusion_matrix(y, pred, labels=CLASSES), index=CLASSES, columns=CLASSES)
        .rename_axis(index="true", columns="predicted"))

# Accuracy per recording group: which sources does the model fail on?
(folds.assign(correct=pred == y).groupby(["label", "source_group"])
      .agg(clips=("correct", "size"), acc=("correct", "mean"))
      .round(2).sort_values("acc").head(10))
""")

md(r"""
## Step 4: Shortcut check

Within each species, can the embeddings tell the recording groups apart? Same probe as step 5 of `02_preprocessing`, where every spectrogram front-end scored about **0.66** balanced accuracy (chance about 0.24). Lower is better here: it means the features describe the bird more than the recording.
""")
code(r"""
def group_probe(X):
    out = []
    for c in CLASSES:
        m = y == c
        g = groups[m]
        keep = np.isin(g, pd.Series(g).value_counts().loc[lambda s: s >= 5].index)
        Xc, gc = X[m][keep], g[keep]
        if len(np.unique(gc)) < 2:
            continue
        p = np.empty(len(gc), object)
        for tr, va in StratifiedKFold(4, shuffle=True, random_state=C.SEED).split(Xc, gc):
            p[va] = MODELS["logreg"]().fit(Xc[tr], gc[tr]).predict(Xc[va])
        out.append((c, len(np.unique(gc)), 1 / len(np.unique(gc)), balanced_accuracy_score(gc, p)))
    return pd.DataFrame(out, columns=["species", "groups", "chance", "group_bal_acc"]).round(3)

group_probe(X_tr[BEST[0]])
""")

md(r"""
## Step 5: `Unknown` rejection, leave-one-class-out

For each fold `k` and each species `c`: train on the training folds **without** `c`, then score the validation fold. Clips of the other two species are "known"; clips of `c` play the role of `Unknown`.

Rejection scores (higher means more likely `Unknown`):
- `msp`: 1 − the classifier's highest probability (low confidence).
- `knn`: mean cosine distance to the 5 nearest training clips.
- `maha`: smallest Mahalanobis distance to a class mean (standardised features, shrunk shared covariance).

AUROC is threshold-free: 0.5 is chance, 1.0 is perfect separation. Treat these as a ranking of methods, not a leaderboard estimate: the held-out species is a close relative of the others, while real Unknowns may also come with new recording conditions.
""")
code(r"""
from sklearn.covariance import LedoitWolf

class Rejector:
    # Fit on known-class embeddings; score(X) returns Unknown scores, probabilities, class order
    def __init__(self, make_clf, k=5):
        self.make_clf, self.k = make_clf, k
    def fit(self, X, yy):
        self.clf = self.make_clf().fit(X, yy)
        self.ref = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.sc = StandardScaler().fit(X); Z = self.sc.transform(X)
        cls = np.unique(yy)
        self.mu = np.stack([Z[yy == c].mean(0) for c in cls])
        resid = np.vstack([Z[yy == c] - self.mu[i] for i, c in enumerate(cls)])
        self.prec = LedoitWolf().fit(resid).precision_
        return self
    def score(self, X):
        P = self.clf.predict_proba(X)
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        knn = 1 - np.sort(Xn @ self.ref.T, axis=1)[:, -self.k:].mean(1)
        Z = self.sc.transform(X)
        d = np.stack([np.einsum("ij,jk,ik->i", Z - m, self.prec, Z - m) for m in self.mu], 1)
        return {"msp": 1 - P.max(1), "knn": knn, "maha": np.sqrt(d.min(1))}, P, self.clf.classes_

METHODS = ["msp", "knn", "maha"]
X, X_test, make = X_tr[BEST[0]], X_te[BEST[0]], MODELS[BEST[1]]
parts = []
for c in CLASSES:
    for k in np.unique(fold):
        tr, va = (fold != k) & (y != c), fold == k
        S, P, cls = Rejector(make).fit(X[tr], y[tr]).score(X[va])
        d = pd.DataFrame(S)
        d["held_out"], d["true"], d["is_unknown"], d["closed_pred"] = c, y[va], y[va] == c, cls[P.argmax(1)]
        parts.append(d)
loco = pd.concat(parts, ignore_index=True)

auc = pd.DataFrame({c: {s: roc_auc_score(d.is_unknown, d[s]) for s in METHODS}
                    for c, d in loco.groupby("held_out")}).T
auc.loc["mean"] = auc.mean()
auc.round(3)
""")

md(r"""
### Choosing the threshold

A raw score threshold doesn't carry over from the 2-class LOCO models to the final 3-class model, so the threshold is set **relative to known clips**: reject a clip if its score is above the `q`-th percentile of out-of-fold scores of known training clips. `q = 95` means about 5% of genuine known clips will wrongly be called `Unknown`.

For each `q` we compute macro-F1 inside the LOCO simulation (two known species + `Unknown`).
""")
code(r"""
def loco_macro_f1(score, q):
    f1s = []
    for c, d in loco.groupby("held_out"):
        thr = np.percentile(d.loc[~d.is_unknown, score], q)
        pred = np.where(d[score] > thr, C.UNKNOWN, d.closed_pred)
        true = np.where(d.is_unknown, C.UNKNOWN, d.true)
        f1s.append(f1_score(true, pred, average="macro"))
    return np.mean(f1s)

QS = [80, 85, 90, 93, 95, 97, 99]
sweep = pd.DataFrame({s: [loco_macro_f1(s, q) for q in QS] for s in METHODS}, index=pd.Index(QS, name="q"))
sweep.round(3)
""")

md(r"""
## Step 6: Final model, test check, submission

- The threshold comes from **out-of-fold** scores on the 3 known species, so it reflects clips the model hasn't seen.
- The final model is then fitted on all training clips.
- **Sanity check:** EDA estimated about 17% of test clips are far from the training data. A rejection share somewhere around 10–25% is plausible. Something like 2% or 50% means the threshold or the method is off.
""")
code(r"""
METHOD = auc.loc["mean"].idxmax()
Q_OVERRIDE = None   # e.g. 95: set by hand if the test rejection share below is implausible
Q = Q_OVERRIDE or int(sweep[METHOD].idxmax())
print(f"rejection method: {METHOD} | percentile q: {Q}")

oof_scores = np.zeros(len(y))
for k in np.unique(fold):
    tr, va = fold != k, fold == k
    oof_scores[va] = Rejector(make).fit(X[tr], y[tr]).score(X[va])[0][METHOD]
thr = np.percentile(oof_scores, Q)

S_te, P_te, cls = Rejector(make).fit(X, y).score(X_test)
closed = cls[P_te.argmax(1)]
label = np.where(S_te[METHOD] > thr, C.UNKNOWN, closed)

print(pd.Series({q: (S_te[METHOD] > np.percentile(oof_scores, q)).mean() for q in QS},
                name="test share rejected").round(3).to_string())
pd.Series(label).value_counts().to_frame("test clips").assign(share=lambda d: (d["test clips"] / len(label)).round(3))
""")
code(r"""
fig, ax = plt.subplots(figsize=(8, 3.6))
bins = np.histogram_bin_edges(np.r_[oof_scores, S_te[METHOD]], 50)
ax.hist(oof_scores, bins, density=True, alpha=0.6, color="#2a78d6", label="train, known (out-of-fold)")
ax.hist(S_te[METHOD], bins, density=True, alpha=0.6, color="#eb6834", label="test")
ax.axvline(thr, color="black", lw=1.2, ls="--", label=f"threshold (q={Q})")
ax.set(xlabel=f"Unknown score ({METHOD})", ylabel="density",
       title="Test clips to the right of the threshold become Unknown")
ax.legend(frameon=False)
plt.tight_layout(); plt.show()
""")
code(r"""
label_col = [c for c in pd.read_csv(C.SAMPLE_SUB).columns if c != "file_id"][0]
submission = pd.DataFrame({"file_id": test_ids, label_col: label})
assert submission[label_col].isin(CLASSES + [C.UNKNOWN]).all() and len(submission) == len(test_ids)
submission.to_csv(OUT / "submission_birdnet_baseline.csv", index=False)

pd.DataFrame({"file_id": test_ids, "closed_pred": closed, "unknown_score": S_te[METHOD],
              **{f"p_{c}": P_te[:, i] for i, c in enumerate(cls)}}).to_csv(OUT / "test_scores.csv", index=False)
print("wrote", (OUT / "submission_birdnet_baseline.csv").relative_to(PIPELINE))
submission.head()
""")

md(r"""
## Step 7: Distance-based `Unknown` detection (uses the unlabelled test audio)

**Why step 6 is not enough.** Leaderboard results (q=85 → 51% rejected → 0.577; q=97 → 16% → 0.531) and the diagnostics below show:
- Test clips that look like known birds are hardly distinguishable from training clips, so recording conditions are not the main problem.
- About half of test sits in clusters **far from every training clip but close to each other**: unseen species.
- The classifier is often *very confident* on those clusters (one cluster of ~56 clips is labelled Bondol Jawa with near-zero `msp`). Linear models and SVMs grow more confident far from the training data, so confidence cannot reject them.

**Novelty score** (higher = more likely `Unknown`), computed in a standardised 64-component PCA space with cosine distance:
- `d_train`: mean distance to the 5 nearest **training** clips.
- `ratio` = `d_train` / mean distance to the 5 nearest **other test** clips. A clip far from training but inside a tight test-only group gets a high ratio.
- `novelty` = average of the two percentile ranks.

This is transductive: it looks at the test audio's structure, but never at test labels. PCA and the scaler are fitted on unlabelled train + test features.
""")
code(r"""
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

unit = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)
sc7 = StandardScaler().fit(X)
pca7 = PCA(64, random_state=C.SEED).fit(sc7.transform(np.vstack([X, X_test])))
Z_tr, Z_te = unit(pca7.transform(sc7.transform(X))), unit(pca7.transform(sc7.transform(X_test)))

K = 5
d_train = 1 - np.sort(Z_te @ Z_tr.T, axis=1)[:, -K:].mean(1)
S = Z_te @ Z_te.T; np.fill_diagonal(S, -1)
d_test = 1 - np.sort(S, axis=1)[:, -K:].mean(1)
ratio = d_train / np.maximum(d_test, 1e-6)
pct = lambda v: pd.Series(v).rank(pct=True).to_numpy()
novelty = (pct(d_train) + pct(ratio)) / 2

# Reference: how far is a training clip from training clips of *other* recording groups?
St = Z_tr @ Z_tr.T; St[groups[:, None] == groups[None, :]] = -1
d_ref = 1 - np.sort(St, axis=1)[:, -K:].mean(1)
print(f"test clips farther from train than 95% of leave-group-out train clips: {(d_train > np.percentile(d_ref, 95)).mean():.1%}")

# Test clusters: where does the classifier put each one, and how novel is it?
cl = AgglomerativeClustering(n_clusters=12, metric="cosine", linkage="average").fit_predict(Z_te)
clusters = (pd.DataFrame({"cluster": cl, "closed_pred": closed, "msp": S_te["msp"], "d_train": d_train,
                          "ratio": ratio, "novelty": novelty})
            .groupby("cluster")
            .agg(clips=("closed_pred", "size"),
                 main_pred=("closed_pred", lambda s: s.value_counts().index[0]),
                 purity=("closed_pred", lambda s: s.value_counts(normalize=True).iloc[0]),
                 msp=("msp", "median"), d_train=("d_train", "median"),
                 ratio=("ratio", "median"), novelty=("novelty", "median"))
            .sort_values("novelty").round(2))
clusters
""")
code(r"""
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for c, col in CLASS_COLOR.items():
    m = closed == c
    ax.scatter(d_train[m], S_te["msp"][m], s=10, alpha=0.6, color=col, label=f"predicted {c}")
ax.axvline(np.percentile(d_ref, 95), color="black", lw=1, ls="--", label="95% of leave-group-out train")
ax.set(xlabel="distance to nearest training clips (d_train)", ylabel="msp (1 - confidence)",
       title="Far-from-train clips are often predicted with high confidence")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()
""")
md(r"""
### Submissions at several rejection rates

The leaderboard suggests roughly 40–50% of test is `Unknown`. One file per rejection rate, so the leaderboard can tell us which side of the optimum we are on. The known-species labels are the same as in step 6; only the `Unknown` rule changes.
""")
code(r"""
for r in [0.35, 0.45, 0.55]:
    lab = np.where(novelty >= np.quantile(novelty, 1 - r), C.UNKNOWN, closed)
    sub = pd.DataFrame({"file_id": test_ids, label_col: lab})
    path = OUT / f"submission_novelty_r{int(r * 100)}.csv"
    sub.to_csv(path, index=False)
    mix = pd.Series(lab).value_counts(normalize=True).round(3).to_dict()
    print(path.name, mix)

pd.DataFrame({"file_id": test_ids, "closed_pred": closed, "msp": S_te["msp"], "d_train": d_train,
              "ratio": ratio, "novelty": novelty, "cluster": cl}).to_csv(OUT / "test_novelty.csv", index=False)
""")

md(r"""
## Next experiments (one at a time, same folds)
- **Tune the novelty score:** number of neighbours `K`, PCA size, and whether adding `msp` to the rank average helps.
- **Activity-weighted pooling:** weight windows by the step-4 activity curve so noise-only windows count less.
- **Train on windows** instead of clips, then average window probabilities per clip.
- **Perch v2** embeddings (`birdnet.load_perch_v2()`) on the same folds, and a BirdNET + Perch concatenation.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
out = Path(__file__).resolve().parents[1] / "03_birdnet_baseline.ipynb"
nbf.write(nb, out)
print("wrote", out)
