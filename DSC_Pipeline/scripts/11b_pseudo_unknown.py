"""Step 11b: pick the 200 pseudo-Unknown test clips that teach the CNN its 4th class (-> outputs/pseudo_unknown.csv).

Candidates are the clips v3 already calls Unknown. They are ranked by a BirdNET distance ratio
(how far a clip is from its nearest training clips vs. from its nearest test clips, 64-d PCA, cosine),
and the 200 most novel are kept. Validator anchor clips are excluded so the validator stays a fair check.

Inputs : outputs/03_birdnet/birdnet_windows.npz, outputs/03_birdnet/submission_v3_cend45_veto.csv,
         outputs/folds.csv, ../validator/data/anchors.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PIPE = Path(__file__).resolve().parents[1]
OUTS = PIPE / "outputs"

folds = pd.read_csv(OUTS / "folds.csv")
v3 = pd.read_csv(OUTS / "03_birdnet" / "submission_v3_cend45_veto.csv").set_index("file_id")
anchors = pd.read_csv(PIPE.parent / "validator" / "data" / "anchors.csv")
b = np.load(OUTS / "03_birdnet" / "birdnet_windows.npz", allow_pickle=True)
w = pd.DataFrame(b["emb"]); w["fid"] = b["file_id"]
g = w.groupby("fid"); X = pd.concat([g.mean(), g.max()], axis=1)          # mean+max pooled clip embedding

Etr, Ete = X.loc[folds.file_id].to_numpy(), X.loc[v3.index].to_numpy()
mu = Etr.mean(0)
def unit(E):
    E = E - mu
    return E / np.linalg.norm(E, axis=1, keepdims=True)
pca = PCA(64, random_state=0).fit(unit(Etr))
A, B = pca.transform(unit(Etr)), pca.transform(unit(Ete))
A /= np.linalg.norm(A, axis=1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True)
d_train = 1 - np.sort(B @ A.T, axis=1)[:, -5:].mean(1)                  # mean distance to 5 nearest train clips
d_test = 1 - np.sort(B @ B.T, axis=1)[:, -6:-1].mean(1)                 # ... to 5 nearest other test clips
ratio = d_train / (d_test + 1e-3)

s = pd.DataFrame({"file_id": v3.index, "ratio": ratio, "label": v3.label.to_numpy()})
cand = s[(s.label == "Unknown") & ~s.file_id.isin(anchors.file_id)].sort_values("ratio", ascending=False)
cand.head(200)[["file_id"]].to_csv(OUTS / "pseudo_unknown.csv", index=False)
print("wrote", OUTS / "pseudo_unknown.csv", f"(ratio {cand.ratio.iloc[199]:.2f}-{cand.ratio.iloc[0]:.2f})")
