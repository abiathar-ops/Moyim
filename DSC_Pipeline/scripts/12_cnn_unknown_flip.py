"""Step 12: v3 + CNN Unknown flips  ->  submission_v5_cnnflip20.csv (public LB 0.69491).

Takes the v3 submission (step 9) and moves to Unknown the K clips that v3 calls a known bird
but that the pseudo-Unknown CNN (step 11, run b0_pu) is most sure are Unknown.

Inputs
  outputs/03_birdnet/submission_v3_cend45_veto.csv         (step 9)
  outputs/04_cnn/b0_pu/fold{0..3}.npz                       (step 11: --run b0_pu --pseudo-unknown outputs/pseudo_unknown.csv)
Output
  outputs/submission_v5_cnnflip{K}.csv

Usage
  python scripts/12_cnn_unknown_flip.py            # K = 20
  python scripts/12_cnn_unknown_flip.py --k 30
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[1]
OUTS = PIPE / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20, help="number of known-bird clips to move to Unknown")
    ap.add_argument("--base", default=str(OUTS / "03_birdnet" / "submission_v3_cend45_veto.csv"))
    ap.add_argument("--run", default="b0_pu", help="step-11 run with an Unknown class")
    a = ap.parse_args()

    sub = pd.read_csv(a.base)
    # average the 4 fold models' test probabilities
    probs = []
    for k in range(4):
        z = np.load(OUTS / "04_cnn" / a.run / f"fold{k}.npz")
        test = z["split"] == "test"
        probs.append(pd.DataFrame(z["probs"][test], index=z["file_id"][test], columns=list(z["classes"])))
    p_unknown = (sum(probs) / len(probs)).loc[sub.file_id, "Unknown"].to_numpy()

    known = (sub.label != "Unknown").to_numpy()
    order = np.argsort(-np.where(known, p_unknown, -1.0))[: a.k]   # most Unknown-looking known predictions
    print(f"moving {a.k} clips to Unknown (CNN P(Unknown) >= {p_unknown[order].min():.3f}):",
          sub.label.iloc[order].value_counts().to_dict())
    sub.loc[order, "label"] = "Unknown"
    out = OUTS / f"submission_v5_cnnflip{a.k}.csv"
    sub.to_csv(out, index=False)
    print("wrote", out, "| mix:", sub.label.value_counts().to_dict())


if __name__ == "__main__":
    main()
