"""Step 7 - Offline check of submission files against BirdNET "anchor" labels.

Used when no leaderboard submissions are left. BirdNET's own species
predictions are NOT used to make predictions; they are only a referee for
the subset of test clips where they are reliable, judged on the training set:

  Kutilang / Bondol Jawa  "fingerprint" species: BirdNET outputs that, when
                          confident (>0.5), belong to one training class
                          >= 95% of the time, over >= 8 clips and >= 2
                          recording groups.
  Cendrawasih             Lesser Bird-of-Paradise > 0.3 (never fires on the
                          other training classes).
  Unknown                 species confident (>0.5) in >= 3 test clips but
                          in no training clip (not even weakly, > 0.1, in
                          more than one), excluding birds-of-paradise.

Clips matching more than one anchor are dropped. Anchors are a biased,
"easy" subset, so the anchor macro-F1 ranks submissions; it does not
predict the leaderboard value. Sanity check from 2026-10-19: it reproduced
the leaderboard order of the two real submissions (0.577 > 0.531).

Inputs   outputs/03_birdnet/submission_*.csv
Cache    outputs/03_birdnet/birdnet_species_clip.parquet
         (file_id, split, species_name, confidence = max over windows)
Usage    python 07_anchor_check.py [--workers N]
"""
import _common  # noqa: F401
import argparse, os, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src import config as C

OUT = C.OUT_DIR / "03_birdnet"
CACHE = OUT / "birdnet_species_clip.parquet"
LABELS = C.CLASSES + [C.UNKNOWN]
BOP = ("Bird-of-Paradise", "Manucode", "Riflebird", "Lophorina", "Paradise-crow")


def predict_species(workers):
    import importlib.util
    spec = importlib.util.spec_from_file_location("bn", Path(__file__).with_name("06_birdnet_embeddings.py"))
    bn = importlib.util.module_from_spec(spec); spec.loader.exec_module(bn)
    meta = pd.read_csv(C.TRAIN_META); test_ids = pd.read_csv(C.SAMPLE_SUB).file_id
    paths = [str(C.DATA_DIR / p) for p in meta.audio_path] + [str(C.TEST_AUDIO / f"{f}.flac") for f in test_ids]
    model = bn.load_birdnet()
    res = model.predict(paths, top_k=10, default_confidence_threshold=0.01, overlap_duration_s=bn.OVERLAP_SEC,
                        bandpass_fmin=0, n_workers=workers, batch_size=16)
    d = res.to_dataframe()
    d["file_id"] = d["input"].map(lambda p: Path(str(p)).stem)
    d["split"] = np.where(d["input"].astype(str).str.contains("/train/"), "train", "test")
    clip = d.groupby(["file_id", "split", "species_name"]).confidence.max().reset_index()
    clip.to_parquet(CACHE)
    return clip


def build_anchors(clip):
    clip = clip.assign(sp=clip.species_name.str.split("_").str[1])
    meta = pd.read_csv(C.TRAIN_META).set_index("file_id")
    test_ids = pd.read_csv(C.SAMPLE_SUB).file_id
    piv = clip.pivot_table(index="file_id", columns="sp", values="confidence", aggfunc="max").fillna(0)
    TR, TE = piv.reindex(meta.index).fillna(0), piv.reindex(test_ids).fillna(0)

    fingerprints = {c: [] for c in C.CLASSES}
    for s in piv.columns:
        hit = TR[s] > 0.5
        if hit.sum() < 8:
            continue
        lab = meta.label[hit]
        if lab.value_counts(normalize=True).iloc[0] >= 0.95 and meta.source_group[hit].nunique() >= 2:
            fingerprints[lab.value_counts().index[0]].append(s)

    test_only = [s for s in piv.columns
                 if (TE[s] > 0.5).sum() >= 3 and (TR[s] > 0.5).sum() == 0 and (TR[s] > 0.1).sum() <= 1
                 and not any(k in s for k in BOP)]

    hit = lambda sps, t=0.5: (TE[sps] > t).any(axis=1) if sps else pd.Series(False, index=TE.index)
    A = pd.DataFrame({"Bondol Jawa": hit(fingerprints["Bondol Jawa"]),
                      "Cendrawasih": hit([s for s in ["Lesser Bird-of-Paradise"] if s in TE], 0.3),
                      "Kutilang": hit(fingerprints["Kutilang"]),
                      C.UNKNOWN: hit(test_only)})
    A[A.sum(axis=1) > 1] = False
    anchors = A.idxmax(axis=1).where(A.any(axis=1))
    return anchors, fingerprints, test_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    a = ap.parse_args()
    clip = pd.read_parquet(CACHE) if CACHE.exists() else predict_species(a.workers)
    anchors, fingerprints, test_only = build_anchors(clip)
    print("fingerprint species:", {k: len(v) for k, v in fingerprints.items()}, "| test-only species:", test_only)
    print("anchor clips:", anchors.value_counts().to_dict(), "of", len(anchors), "test clips\n")
    anchors.rename("anchor").to_csv(OUT / "anchors.csv")

    m = anchors.notna().to_numpy(); t = anchors.to_numpy()[m]
    rows = []
    for f in sorted(OUT.glob("submission_*.csv")):
        p = pd.read_csv(f).set_index("file_id").label.reindex(anchors.index).to_numpy()[m]
        r = {"file": f.name, "anchor_macroF1": f1_score(t, p, average="macro", labels=LABELS)}
        for k in LABELS:
            r[k] = f"{(p[t == k] == k).sum()}/{(t == k).sum()}"
        r["known anchors → Unknown"] = int(((t != C.UNKNOWN) & (p == C.UNKNOWN)).sum())
        rows.append(r)
    res = pd.DataFrame(rows).sort_values("anchor_macroF1", ascending=False).round(3)
    res.to_csv(OUT / "anchor_check.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
