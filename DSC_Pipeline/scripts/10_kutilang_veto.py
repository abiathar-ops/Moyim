"""Step 10 - Veto Kutilang impostors (2026-10-20).

Evidence: Kutilang "fingerprint" outputs (Perch/BirdNET species that, on
training data, fire only on Kutilang) fire on 54% of real Kutilang clips,
stable across all 4 recording groups (43-62%), but on only 32% of the test
clips we label Kutilang. So ~40% of those are likely Unknown birds,
including other bulbuls.

Rule, starting from submission_v3_cend45_veto.csv (step 9):
A clip labelled Kutilang becomes Unknown if Perch's top-3 species include a
"veto species": a species that is in the top-3 of >= 2 test clips labelled
Kutilang but in the top-3 of NO training Kutilang clip. (12 species, incl.
Yellow-vented Bulbul, White-browed Bulbul, Collared Finchbill.)
- Honest cost: choosing veto species without one Kutilang recording group,
  the veto fires on 1.6% of that group's real Kutilang (per group 0.6-4.2%).
- Effect: vetoes 34 of 151; their fingerprint rate is 12% (vs 54% for real
  Kutilang; vetoed real Kutilang keep a normal 50% rate, so no bias), i.e.
  ~27 impostors and ~7 real Kutilang. Implied Kutilang precision 60% -> 71%.
Perch species scores are used as a decision feature here; say so in the
write-up.

Inputs   outputs/03_birdnet/perch_species_clip.npz, submission_v3_cend45_veto.csv,
         ../DSC_Dataset/train/metadata.csv
Output   outputs/03_birdnet/submission_v4_kutveto.csv
Usage    python 10_kutilang_veto.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[1]
DATA = PIPE.parent / "DSC_Dataset"
OUT = PIPE / "outputs" / "03_birdnet"
TOP_K, MIN_TEST = 3, 2


def main():
    z = np.load(OUT / "perch_species_clip.npz")
    L = pd.DataFrame(z["logit"].astype(np.float32), index=z["file_id"], columns=z["species"])
    rank = L.rank(axis=1, ascending=False, method="first")
    top = {f: set(rank.columns[np.flatnonzero(rank.loc[f].to_numpy() <= TOP_K)]) for f in rank.index}

    meta = pd.read_csv(DATA / "train" / "metadata.csv")
    base = pd.read_csv(OUT / "submission_v3_cend45_veto.csv")
    train_k = meta.file_id[meta.label == "Kutilang"]
    test_k = base.file_id[base.label == "Kutilang"]
    seen = set(s for f in train_k for s in top[f])
    counts = pd.Series([s for f in test_k for s in top[f]]).value_counts()
    veto = sorted(s for s, n in counts.items() if n >= MIN_TEST and s not in seen)
    print("veto species:", veto)

    hit = base.label.eq("Kutilang") & base.file_id.map(lambda f: bool(top[f] & set(veto)))
    out = base.copy(); out.loc[hit, "label"] = "Unknown"
    path = OUT / "submission_v4_kutveto.csv"
    out.to_csv(path, index=False)
    print(f"{path.name}: {int(hit.sum())} Kutilang -> Unknown |", out.label.value_counts().to_dict())


if __name__ == "__main__":
    main()
