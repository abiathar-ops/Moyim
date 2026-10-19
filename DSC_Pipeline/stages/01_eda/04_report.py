"""Step 4: assemble outputs/EDA_REPORT.md from summary.json and the saved tables."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config as C


def md_table(df, floatfmt="{:.3f}"):
    if df is None or len(df) == 0:
        return "_(no data)_\n"
    df = df.copy()
    head = "| " + " | ".join(map(str, df.columns)) + " |\n"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|\n"
    rows = ""
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        rows += "| " + " | ".join(cells) + " |\n"
    return head + sep + rows


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def main():
    T = C.TABLES
    S = json.load(open(T / "summary.json"))
    A, B, Cc, D, E, F = (S.get(k, {}) for k in "ABCDEF")
    flags = []  # (severity, text)

    def flag(sev, txt):
        flags.append((sev, txt))

    counts = A.get("class_counts", {})
    imb = A.get("imbalance_ratio")
    if imb and imb > 3:
        small = min(counts, key=counts.get)
        flag("WARN", f"Strong class imbalance ({imb:.1f}:1). Smallest class is **{small}** with {counts[small]} files. "
                     "Macro F1 weights every class equally, so use class weights / balanced sampling and report per-class F1.")
    gpc = A.get("groups_per_class", {})
    for c, n in gpc.items():
        if n < 3:
            flag("ALERT", f"**{c}** comes from only {n} source group(s). Group-aware CV cannot hold out whole sources "
                          "for this class reliably; expect noisy validation for it.")
    if A.get("groups_with_mixed_labels"):
        flag("ALERT", f"{A['groups_with_mixed_labels']} source group(s) contain more than one label. Check metadata.")
    if A.get("n_groups") and A["n_groups"] <= 20:
        flag("WARN", f"Only {A['n_groups']} source groups in total: group K-fold has very few units, so CV scores will have high variance.")
    for c, s in A.get("largest_group_share_per_class", {}).items():
        if s > 0.5:
            flag("WARN", f"One source group holds {s:.0%} of **{c}** files, so that class's features may be dominated by a single recording setup.")

    inv = A.get("inventory", {})
    if inv.get("n_train_missing_files") or inv.get("n_test_missing_files"):
        flag("ALERT", f"Missing audio files: train {inv.get('n_train_missing_files')}, test {inv.get('n_test_missing_files')}.")
    if inv.get("n_train_extra_files") or inv.get("n_test_extra_files"):
        flag("INFO", f"Audio files on disk not listed in metadata/submission: train {inv.get('n_train_extra_files')}, test {inv.get('n_test_extra_files')}.")
    if B.get("n_unreadable"):
        flag("ALERT", f"{B['n_unreadable']} unreadable/corrupt files, e.g. {B.get('unreadable_examples')}")
    if B.get("duplicate_sets_with_label_conflict"):
        flag("ALERT", f"{B['duplicate_sets_with_label_conflict']} exact-duplicate audio sets carry conflicting labels.")
    if B.get("duplicate_sets_spanning_train_and_test"):
        flag("WARN", f"{B['duplicate_sets_spanning_train_and_test']} exact-duplicate sets appear in both train and test.")
    elif B.get("n_duplicate_sets"):
        flag("INFO", f"{B['n_duplicate_sets']} exact-duplicate audio sets inside the data ({B['n_files_in_exact_duplicate_sets']} files).")
    sr = B.get("sr_by_split", {})
    if len({k for v in sr.values() for k in v}) > 1:
        flag("WARN", f"Mixed sample rates: {sr}. Resample everything to one rate.")
    ch = B.get("channels_by_split", {})
    if len({k for v in ch.values() for k in v}) > 1:
        flag("WARN", f"Mixed channel counts: {ch}. Down-mix to mono consistently.")
    if B.get("clipped_files"):
        flag("WARN", f"{B['clipped_files']} files have >0.1% clipped samples.")
    if B.get("near_silent_files"):
        flag("WARN", f"{B['near_silent_files']} files are >90% silence by the frame-energy rule; silence trimming may leave almost nothing.")
    if B.get("very_quiet_files"):
        flag("INFO", f"{B['very_quiet_files']} files have RMS below -60 dBFS; normalise loudness.")
    dtr, dte = g(B, "duration_by_split", "train", default={}), g(B, "duration_by_split", "test", default={})
    if dtr and dte and abs(dtr.get("50%", 0) - dte.get("50%", 0)) > 1.0:
        flag("WARN", f"Median duration differs: train {dtr.get('50%')} s vs test {dte.get('50%')} s.")

    pa = B.get("periodic_artifact")
    if pa and pa.get("share_train") is not None:
        rng = pa.get("group_share_range") or 0
        sev = "WARN" if rng > 0.3 else "INFO"
        cls_rng = max(pa["share_by_class"].values()) - min(pa["share_by_class"].values()) if pa.get("share_by_class") else 0
        if rng > 0.3 or cls_rng > 0.3:
            tail = ("Its share differs strongly across source groups/classes, so it is a recording-setup signature a model "
                    "could learn instead of the species. Consider augmentation or masking.")
        else:
            tail = (f"Its share is similar across classes (spread {cls_rng:.0%}) and source groups (spread {rng:.0%}), so it is "
                    "unlikely to leak the label, but it is a nuisance pattern worth masking or augmenting around.")
        flag(sev, f"Periodic stripe/gating artifact (strongly periodic 8-14 kHz envelope, threshold {pa['threshold']}): "
                  f"{pa['share_train']:.0%} of train and {pa['share_test']:.0%} of test files; by class {pa['share_by_class']}. {tail}")
    if Cc and Cc.get("median_adj_r2_group_within_class", 0) > 0.3:
        flag("WARN", f"Source-group effects are strong even after removing class means (median adj R^2 = "
                     f"{Cc['median_adj_r2_group_within_class']:.2f}). Models can learn the recording setup instead of the species.")
    res = D.get("results", {})
    for m in ["logreg", "random_forest"]:
        gap = res.get(f"{m}_leakage_gap")
        if gap is not None and gap > 0.1:
            flag("ALERT", f"Random CV overestimates {m} macro F1 by {gap:.2f} versus group-aware CV. Always validate on `source_group`.")
    if E.get("adversarial_auc", 0) > 0.7:
        flag("WARN", f"A classifier separates train from test with AUC {E['adversarial_auc']:.2f}: test comes from a different "
                     "distribution (new sources, conditions, or unseen species).")
    f95 = E.get("test_frac_above_train_lgo_p95")
    if f95 is not None and f95 > 0.15:
        flag("INFO", f"{f95:.0%} of test files are farther from train than 95% of train files are from OTHER train sources "
                     "(5% expected if equally novel). Part of this is likely the Unknown class and part is domain shift.")
    if S.get("section_errors"):
        flag("ALERT", f"Some analysis sections failed: {S['section_errors']}")

    L = []
    L.append("# Audio EDA report: Schematics NPC DSC 2026 bird classification\n")
    L.append("_Generated by `DSC_Pipeline/eda/run_eda.py`. All numbers are reproducible with SEED=%d._\n" % C.SEED)

    L.append("## 1. Snapshot\n")
    L.append(f"- Train files scanned: **{A.get('n_train_scanned')}** (metadata rows: {A.get('n_train_rows_meta')}), "
             f"test files scanned: **{A.get('n_test_scanned')}**\n")
    L.append(f"- Classes: {counts}\n- Source groups (train): **{A.get('n_groups')}**; per class: {gpc}\n")
    L.append(f"- Sample rate by split: {B.get('sr_by_split')}; channels: {B.get('channels_by_split')}; subtype: {B.get('subtype_by_split')}\n")
    L.append("- Duration (s): train " + str({k: dtr.get(k) for k in ['min', '25%', '50%', '75%', 'max']}) +
             ", test " + str({k: dte.get(k) for k in ['min', '25%', '50%', '75%', 'max']}) + "\n")

    L.append("\n## 2. Findings and flags\n")
    order = {"ALERT": 0, "WARN": 1, "INFO": 2}
    for sev, txt in sorted(flags, key=lambda x: order[x[0]]):
        L.append(f"- **{sev}** {txt}\n")
    if not flags:
        L.append("- No automatic flags raised.\n")

    L.append("\n## 3. Class and source-group structure\n")
    L.append("![class counts](figures/01_class_counts.png)\n\n![group x class](figures/02_group_by_class.png)\n\n")
    gs = T / "group_summary.csv"
    if gs.exists():
        L.append(md_table(pd.read_csv(gs)[["gid", "label", "n", "n_labels", "total_minutes"]]))
    L.append("\n![duration](figures/03_duration.png)\n")

    L.append("\n## 4. Recording quality and spectral character\n")
    L.append("Rain and wind raise energy below ~500 Hz and flatten the spectrum; insects add narrow high-frequency bands; "
             "a high noise floor with low dynamic range indicates a noisy recording.\n\n")
    L.append("![quality](figures/04_quality_by_class.png)\n\n![mean spectrum](figures/05_mean_spectrum.png)\n\n"
             "![bands](figures/06_band_fractions.png)\n\n![examples](figures/13_example_spectrograms.png)\n")

    L.append("\n## 5. Class signal versus recording-source signal\n")
    L.append("`eta^2` is the share of a feature's variation explained by species; `adj R^2 (group | class)` is what the source "
             "group still explains after the class mean is removed. Features high in the second column describe the "
             "recording setup, not the bird.\n\n")
    fe = T / "feature_class_effects.csv"
    if fe.exists():
        t = pd.read_csv(fe).head(12)[["feature", "eta2_class", "adj_r2_group_within_class"]]
        L.append(md_table(t))
    L.append("\n![class vs group](figures/08_class_vs_group_effect.png)\n\n![pca](figures/07_pca.png)\n")

    L.append("\n### Baseline probes (hand-crafted features + mel profile)\n")
    if res:
        rows = [{"setting": k, "macro_f1_3_classes": v["macro_f1"],
                 **{k2.replace("f1_", "F1 "): v2 for k2, v2 in v.items() if k2.startswith("f1_")}}
                for k, v in res.items() if isinstance(v, dict)]
        L.append(md_table(pd.DataFrame(rows)))
        L.append(f"\nGroup CV used {D.get('group_cv_n_splits')} folds. Pooled leave-one-group-out macro F1 (logreg): "
                 f"{round(D['logo_macro_f1_pooled'], 3) if D.get('logo_macro_f1_pooled') is not None else None}\n")
    L.append("\n![cv gap](figures/09_cv_leakage_gap.png)\n")
    lg = T / "leave_one_group_out.csv"
    if lg.exists():
        L.append("\nAccuracy when each source group is held out entirely:\n\n")
        L.append(md_table(pd.read_csv(lg)))

    L.append("\n## 6. Train versus test shift and the Unknown class\n")
    rd = lambda v, n=3: round(v, n) if isinstance(v, (int, float)) else v
    L.append(f"- Adversarial validation AUC (train vs test): **{rd(E.get('adversarial_auc'))}** "
             "(0.5 = indistinguishable)\n")
    L.append(f"- Features with BH-adjusted KS p<0.01: {E.get('n_features_shifted_bh_0.01')} of {E.get('n_features_tested')}\n")
    L.append(f"- Nearest-neighbour distance quantiles, train (other sources): {E.get('train_leave_group_out_nn_dist_quantiles')}\n"
             f"- Test: {E.get('test_nn_dist_quantiles')}\n"
             f"- Share of test above the train p95 / p99 distance: {rd(E.get('test_frac_above_train_lgo_p95'))} / "
             f"{rd(E.get('test_frac_above_train_lgo_p99'))} (5% / 1% if test were as familiar as unseen train sources)\n\n")
    L.append("![shift](figures/10_train_test_shift.png)\n\n![novelty](figures/11_novelty_preview.png)\n")
    L.append("\n### Open-set feasibility (leave-one-class-out)\n")
    L.append("Each known class is treated as the unseen one; AUROC measures how well a detector built on the other classes "
             "scores it above held-out known files. Values near 0.5 mean the feature space cannot reject novel species.\n\n")
    if F:
        L.append(md_table(pd.read_csv(T / "loco_novelty_auroc.csv")))
        L.append(f"\nMean AUROC by method: {F.get('mean_auroc_by_method')}\n\n![loco](figures/12_loco_novelty.png)\n")
    L.append("\n`tables/test_novelty_scores.csv` holds an unsupervised distance score per test file. It is descriptive only: "
             "it is not a label and must not be written into a submission without a trained, validated model.\n")

    L.append("\n## 7. Modelling implications\n")
    ng = D.get("group_cv_n_splits")
    rec = []
    rec.append(f"Validation: `StratifiedGroupKFold(n_splits={ng or 'k'}, shuffle=True, random_state={C.SEED})` on `source_group`; "
               "never split segments of one source across train and validation.")
    if imb:
        rec.append(f"Imbalance {imb:.1f}:1: use class-weighted loss or balanced sampling; tune per-class thresholds on out-of-fold predictions.")
    sr_set = sorted({k for v in sr.values() for k in v})
    if sr_set:
        sc = pd.read_csv(T / "features.csv")["stereo_corr"].median()
        rec.append(f"Audio is {sr_set} Hz with median L/R correlation {sc:.2f}; down-mix to mono (or keep both channels as "
                   "augmentation) and resample once.")
    if dtr:
        rec.append(f"Median clip is {dtr.get('50%')} s (max {dtr.get('max')} s): crop or tile to fixed windows and average window predictions per file.")
    if F:
        best = max(F.get("mean_auroc_by_method", {"-": 0}).items(), key=lambda kv: kv[1])
        rec.append(f"For Unknown, the best simple detector in feature space was `{best[0]}` (mean AUROC {best[1]}). "
                   "Combine it with a closed-set confidence score and choose the threshold by macro F1 on simulated unknowns "
                   "(hold out one class at a time) using group-aware folds.")
    if Cc.get("median_adj_r2_group_within_class", 0) > 0.2:
        rec.append("Use augmentation that varies the channel and background (gain, EQ tilt, synthetic noise, crops) so the model cannot key on the recording setup.")
    for r in rec:
        L.append(f"- {r}\n")

    L.append("\n## 8. Files produced\n")
    L.append("- `tables/features.csv`, `tables/mel_stats.npz`, `tables/summary.json`, `tables/*.csv`\n- `figures/*.png`\n")
    L.append("\n## 9. Rule compliance notes\n")
    L.append("- Only competition-provided data is used; no external audio or pretrained models.\n"
             "- Test audio is used as unlabeled data for distribution comparison only; no hand labeling or manual inspection of test labels.\n"
             "- All random operations use SEED=%d. Plots of test spectrograms are random draws, not curated.\n" % C.SEED)

    out = C.OUT / "EDA_REPORT.md"
    out.write_text("".join(L), encoding="utf-8")
    print(f"[04] wrote {out}")


if __name__ == "__main__":
    main()
