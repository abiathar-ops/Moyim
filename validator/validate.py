#!/usr/bin/env python3
"""Offline validator for NPC DSC 2026 submissions (team-only).

Checks a submission against the competition's format rules, then scores it
on 121 "core anchor" test clips whose label is known with high confidence
(see README). Use it to RANK candidate submissions before spending a
leaderboard submission; it does not predict the leaderboard value.

The 50 test clips BirdNET calls White-rumped Munia are reported separately:
whether they are Unknown or Bondol Jawa is unresolved, and they barely move
the public leaderboard but may matter for the private one.

Usage
  python validate.py my_submission.csv
  python validate.py a.csv b.csv c.csv          # rank several, pairwise comparison
  python validate.py a.csv --details mistakes.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LABELS = ["Bondol Jawa", "Cendrawasih", "Kutilang", "Unknown"]
N_BOOT = 2000
# Public-LB estimate = LB_RATIO * core anchor F1 (median of LB/anchor over the
# 8 calibration submissions below, refitted 2026-10-20). Leave-one-out error
# <= 0.008 for 6 of 8; -0.03 for the 65%-Unknown file and -0.075 for the one
# with only 16% Unknown. The public LB itself (~244 clips) moves about +-0.03.
LB_RATIO = 0.769
# Calibration: anchor scores of submissions whose public leaderboard score is known
# (core-anchor macro-F1 reproduced the public-LB order of all 8)
REFERENCE = [("v3_cend45_veto (Danta)", 0.682, 0.885),
             ("v2_r50 (Danta)", 0.632, 0.826),
             ("friend: submission.csv", 0.621, 0.769),
             ("friend: submission_v3_pi40.csv", 0.589, 0.758),
             ("SVM + msp, 51% Unknown", 0.577, 0.752),
             ("friend: submission_v2_pi25.csv", 0.566, 0.743),
             ("friend: submission_v2.csv", 0.552, 0.727),
             ("SVM + msp, 16% Unknown", 0.531, 0.594)]


def macro_f1(t, p):
    f1s = []
    for c in LABELS:
        tp = np.sum((p == c) & (t == c)); fp = np.sum((p == c) & (t != c)); fn = np.sum((p != c) & (t == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def check_format(path, test_ids):
    """Return (labels Series indexed by file_id, list of problems)."""
    problems = []
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        return None, [f"cannot read CSV: {e}"]
    if list(df.columns) != ["file_id", "label"]:
        problems.append(f"columns must be exactly ['file_id', 'label'], got {list(df.columns)}")
        if not {"file_id", "label"} <= set(df.columns):
            return None, problems
    dup = df.file_id[df.file_id.duplicated()]
    if len(dup):
        problems.append(f"{dup.nunique()} file_id(s) appear more than once, e.g. {dup.iloc[0][:16]}...")
    ids = set(df.file_id)
    missing, extra = set(test_ids) - ids, ids - set(test_ids)
    if missing:
        problems.append(f"{len(missing)} test file_id(s) missing")
    if extra:
        problems.append(f"{len(extra)} file_id(s) not in the test set")
    bad = df.label[~df.label.isin(LABELS)]
    if len(bad):
        hints = []
        for v in bad.unique()[:5]:
            fix = next((l for l in LABELS if l.lower() == v.strip().lower()), None)
            hints.append(f"'{v}'" + (f" (did you mean '{fix}'?)" if fix else ""))
        problems.append(f"{len(bad)} invalid label(s): {', '.join(hints)}. Allowed: {LABELS}")
    labels = df.drop_duplicates("file_id").set_index("file_id").label
    return labels, problems


def anchor_report(labels, anchors, munia):
    t = anchors.anchor.to_numpy()
    p = labels.reindex(anchors.file_id).fillna("<missing>").to_numpy()
    pm = labels.reindex(munia.file_id).fillna("<missing>")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(t), (N_BOOT, len(t)))
    boot = np.array([macro_f1(t[i], p[i]) for i in idx])
    per_class = {c: (int(np.sum((t == c) & (p == c))), int(np.sum(t == c))) for c in LABELS}
    return {"f1": macro_f1(t, p), "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5),
            "per_class": per_class, "known_to_unknown": int(np.sum((t != "Unknown") & (p == "Unknown"))),
            "pred": p, "boot_idx": idx, "munia": pm.value_counts().to_dict()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submissions", nargs="+", help="submission CSV file(s)")
    ap.add_argument("--details", help="write per-anchor predictions of the FIRST submission to this CSV")
    a = ap.parse_args()

    test_ids = pd.read_csv(HERE / "data" / "test_ids.csv").file_id.tolist()
    all_anchors = pd.read_csv(HERE / "data" / "anchors.csv")
    anchors = all_anchors[all_anchors.group == "core"].reset_index(drop=True)
    munia = all_anchors[all_anchors.group == "white_rumped_munia"]
    t = anchors.anchor.to_numpy()

    results, any_invalid = [], False
    for path in a.submissions:
        name = Path(path).name
        print(f"\n=== {name}")
        labels, problems = check_format(path, test_ids)
        if problems:
            any_invalid = True
            print("  FORMAT: INVALID (Kaggle would reject or mis-score this file)")
            for pr in problems:
                print("   -", pr)
            if labels is None:
                continue
        else:
            print("  FORMAT: OK (columns, all 814 file_ids exactly once, valid labels)")

        mix = labels.reindex(test_ids).value_counts(normalize=True)
        print("  Prediction mix on all 814 test clips: " +
              ", ".join(f"{c} {mix.get(c, 0):.0%}" for c in LABELS))
        unk = mix.get("Unknown", 0)
        if unk < 0.30 or unk > 0.60:
            print(f"  NOTE: {unk:.0%} Unknown. Leaderboard evidence so far points to roughly 40-50% Unknown in test.")

        r = anchor_report(labels, anchors, munia)
        r["name"], r["unknown_share"] = name, unk
        results.append(r)
        print(f"  Estimated public LB: ~{LB_RATIO * r['f1']:.3f}  (typically within +-0.01-0.03)")
        print(f"  Core anchor macro-F1: {r['f1']:.3f}  (95% bootstrap interval {r['lo']:.3f}-{r['hi']:.3f})")
        print("  Core anchors correct: " + ", ".join(f"{c} {k}/{n}" for c, (k, n) in r["per_class"].items()))
        print(f"  Known-bird anchors wrongly called Unknown: {r['known_to_unknown']}")
        print(f"  White-rumped Munia clips (unresolved, {len(munia)}): " +
              ", ".join(f"{k} {v}" for k, v in sorted(r["munia"].items(), key=lambda kv: -kv[1])))

    if a.details and results:
        r = results[0]
        out = anchors.assign(prediction=r["pred"], correct=r["pred"] == t)
        out.to_csv(a.details, index=False)
        print(f"\nPer-anchor predictions of {r['name']} written to {a.details}")

    if len(results) > 1:
        results.sort(key=lambda r: r["f1"], reverse=True)
        print("\n=== Ranking (estimated public LB | core anchor macro-F1 [95% interval])")
        for i, r in enumerate(results, 1):
            mu = r["munia"].get("Unknown", 0)
            print(f"  {i}. ~{LB_RATIO * r['f1']:.3f} | {r['f1']:.3f} [{r['lo']:.3f}-{r['hi']:.3f}]  {r['unknown_share']:.0%} Unknown, "
                  f"munia->Unknown {mu}/{len(munia)}  {r['name']}")
        best = results[0]
        print(f"\n=== Is {best['name']} really better? (paired bootstrap on the same anchor clips)")
        for r in results[1:]:
            wins = np.mean([macro_f1(t[i], best["pred"][i]) > macro_f1(t[i], r["pred"][i]) for i in best["boot_idx"]])
            diff = int(np.sum(best["pred"] != r["pred"]))
            verdict = "clearly better" if wins >= 0.95 else "probably better" if wins >= 0.8 else "too close to call"
            print(f"  vs {r['name']}: better in {wins:.0%} of resamples ({verdict}); "
                  f"they disagree on {diff} of {len(t)} anchor clips")

    print("\nCalibration (public LB vs estimate; same order for all 8):")
    for n, lb, an in REFERENCE:
        print(f"  {n}: LB {lb:.3f}, estimate {LB_RATIO * an:.3f}")
    print("Trust the ranking more than the number; differences under ~0.02 are noise. The 50 munia clips are not in the estimate.")
    sys.exit(1 if any_invalid else 0)


if __name__ == "__main__":
    main()
