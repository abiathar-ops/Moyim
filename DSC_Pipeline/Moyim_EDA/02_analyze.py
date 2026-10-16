"""Step 2: statistical analysis of the extracted features.

Sections (each isolated, a failure in one does not stop the others):
  A. metadata / label / source_group audit
  B. audio integrity & format audit (corrupt, duplicates, sample rate, clipping, silence)
  C. class separability and source-group confounding (how much is "recording", not "species")
  D. baseline probes: random CV vs group-aware CV (quantifies leakage) + leave-one-group-out
  E. train vs test shift (adversarial validation, KS tests, nearest-neighbour novelty)
  F. open-set feasibility: leave-one-class-out novelty detection AUROC

Everything uses SEED from config.py. Test audio is only used as UNLABELED data for distribution
comparison; nothing here produces labels for the test set.
"""
import json
import sys
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import (GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_predict)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C

S = {"section_errors": {}}


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def section(name):
    def deco(fn):
        def wrapped(*a, **k):
            try:
                print(f"[02] {name}", flush=True)
                return fn(*a, **k)
            except Exception as e:  # keep going, record the failure
                S["section_errors"][name] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
        return wrapped
    return deco


# --------------------------------------------------------------------------- helpers
def feature_columns(df):
    drop = {"sr", "channels", "frames", "file_bytes", "n_frames", "lr_identical"}
    cols = []
    for c in df.columns:
        if c in drop or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].notna().mean() < 0.5 or not np.nanstd(df[c].astype(float)) > 1e-12:
            continue
        cols.append(c)
    return cols


def macro_f1(y, p):
    return float(np.mean(f1_score(y, p, labels=C.KNOWN_CLASSES, average=None, zero_division=0)))


def per_class_f1(y, p):
    return dict(zip(C.KNOWN_CLASSES, f1_score(y, p, labels=C.KNOWN_CLASSES, average=None, zero_division=0)))


def kruskal_eta(df, col, by="label"):
    gs = [g[col].dropna().values for _, g in df.groupby(by)]
    gs = [g for g in gs if len(g) > 1]
    if len(gs) < 2:
        return np.nan, np.nan, np.nan
    try:
        h, p = stats.kruskal(*gs)
    except ValueError:
        return np.nan, np.nan, np.nan
    n, k = sum(len(g) for g in gs), len(gs)
    return float(h), float(p), float(max((h - k + 1) / (n - k), 0.0)) if n > k else np.nan


def r2_by(x, key):
    x = pd.Series(np.asarray(x, dtype=float))
    key = pd.Series(np.asarray(key))
    m = x.notna()
    x, key = x[m], key[m]
    sst = ((x - x.mean()) ** 2).sum()
    if sst <= 0:
        return np.nan
    fitted = x.groupby(key.values).transform("mean")
    return float(((fitted - x.mean()) ** 2).sum() / sst)


def r2_within(x, cls, grp):
    x = pd.Series(np.asarray(x, dtype=float))
    resid = x - x.groupby(np.asarray(cls)).transform("mean")
    return r2_by(resid.values, grp)


def adj_r2(r2, n, k):
    if r2 is None or np.isnan(r2) or n <= k:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - k)


def knn_mean_dist(fit, query, k=3):
    nn = NearestNeighbors(n_neighbors=min(k, len(fit))).fit(fit)
    d, _ = nn.kneighbors(query)
    return d.mean(1)


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def make_rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                  min_samples_leaf=2, random_state=C.SEED, n_jobs=2)


def oof_predict(model_fn, X, y, groups, splitter):
    pred = np.empty(len(y), dtype=object)
    for tri, vai in splitter.split(X, y, groups):
        m = model_fn()
        m.fit(X[tri], y[tri])
        pred[vai] = m.predict(X[vai])
    return pred


# --------------------------------------------------------------------------- main
def main():
    C.seed_everything()
    C.ensure_dirs()
    T = C.TABLES

    feats = pd.read_csv(T / "features.csv")
    feats["error"] = feats["error"].fillna("")
    inv = json.load(open(T / "file_inventory.json"))
    npz = np.load(T / "mel_stats.npz", allow_pickle=True)
    idx = {f: i for i, f in enumerate(npz["file_ids"].astype(str))}
    MEL = npz["mel_act"]

    ok = feats[feats["error"] == ""].copy()
    tr = ok[ok["split"] == "train"].reset_index(drop=True)
    te = ok[ok["split"] == "test"].reset_index(drop=True)
    cols = feature_columns(ok)
    med = tr[cols].astype(float).median()
    S["feature_columns"] = cols

    def design(df):
        A = df[cols].astype(float).fillna(med).replace([np.inf, -np.inf], 0.0)
        M = np.nan_to_num(MEL[[idx[f] for f in df["file_id"]]])
        return np.hstack([A.values, M])

    names_all = cols + [f"mel_{i:02d}" for i in range(MEL.shape[1])]
    Xtr, Xte = design(tr), design(te)
    ytr = tr["label"].values.astype(object)
    gtr = tr["source_group"].values.astype(object)
    gid = C.short_group_ids(tr) if len(tr) else {}
    S["group_ids"] = gid

    # ---------------------------------------------------------------- A. metadata audit
    @section("A. metadata / label / group audit")
    def sec_a():
        a = {"n_train_rows_meta": inv["train_rows"], "n_train_scanned": int((feats["split"] == "train").sum()),
             "n_test_scanned": int((feats["split"] == "test").sum()), "inventory": inv}
        counts = tr["label"].value_counts()
        a["class_counts"] = counts.to_dict()
        a["class_share"] = (counts / counts.sum()).round(4).to_dict()
        a["imbalance_ratio"] = float(counts.max() / counts.min()) if len(counts) else np.nan
        g = (tr.groupby("source_group")
             .agg(n=("file_id", "size"), n_labels=("label", "nunique"), label=("label", lambda s: s.mode().iloc[0]),
                  total_minutes=("duration", lambda s: s.sum() / 60)).reset_index())
        g["gid"] = g["source_group"].map(gid)
        g = g.sort_values("gid")
        g.to_csv(T / "group_summary.csv", index=False)
        a["n_groups"] = int(len(g))
        a["groups_with_mixed_labels"] = int((g["n_labels"] > 1).sum())
        a["group_size"] = g["n"].describe().round(2).to_dict()
        a["groups_per_class"] = tr.groupby("label")["source_group"].nunique().to_dict()
        a["files_per_group_per_class"] = {k: v["n"].describe().round(1).to_dict()
                                          for k, v in g.groupby("label")}
        ct = pd.crosstab(tr["source_group"].map(gid), tr["label"])
        ct.to_csv(T / "group_by_class_counts.csv")
        # share of each class held by its largest group (limits what group CV can do)
        a["largest_group_share_per_class"] = {
            c: float(v.groupby("source_group").size().max() / len(v)) for c, v in tr.groupby("label")}
        S["A"] = a

    sec_a()

    # ---------------------------------------------------------------- B. audio audit
    @section("B. audio integrity & format audit")
    def sec_b():
        b = {"n_unreadable": int((feats["error"] != "").sum()),
             "unreadable_examples": feats.loc[feats["error"] != "", ["file_id", "error"]].head(5).to_dict("records")}
        for col in ["sr", "channels", "subtype", "format"]:
            b[f"{col}_by_split"] = {s: d[col].value_counts().to_dict() for s, d in ok.groupby("split")}
        b["lr_identical_files"] = int(ok["lr_identical"].sum())
        b["duration_by_split"] = {s: d["duration"].describe().round(3).to_dict() for s, d in ok.groupby("split")}
        b["duration_by_class"] = {s: d["duration"].describe().round(3).to_dict() for s, d in tr.groupby("label")}
        b["clipped_files"] = int((ok["clip_frac"] > 1e-3).sum())
        b["near_silent_files"] = int((ok["silence_ratio"] > 0.9).sum())
        b["very_quiet_files"] = int((ok["rms_dbfs"] < -60).sum())
        b["short_files_lt_1s"] = int((ok["duration"] < 1.0).sum())
        # exact duplicates
        dup = ok[ok.duplicated("pcm_md5", keep=False)].sort_values("pcm_md5")
        b["n_files_in_exact_duplicate_sets"] = int(len(dup))
        b["n_duplicate_sets"] = int(dup["pcm_md5"].nunique())
        if len(dup):
            conflict = dup[dup["split"] == "train"].groupby("pcm_md5")["label"].nunique()
            b["duplicate_sets_with_label_conflict"] = int((conflict > 1).sum())
            b["duplicate_sets_spanning_train_and_test"] = int((dup.groupby("pcm_md5")["split"].nunique() > 1).sum())
            dup[["pcm_md5", "file_id", "split", "label", "source_group"]].to_csv(T / "exact_duplicates.csv", index=False)
        else:
            b["duplicate_sets_with_label_conflict"] = 0
            b["duplicate_sets_spanning_train_and_test"] = 0
        # periodic gating / stripe artifact: share of files with strongly periodic 8-14 kHz envelope
        if "hf_periodicity" in ok:
            thr = 0.5
            flagged = ok["hf_periodicity"] > thr
            art = {"threshold": thr,
                   "share_train": float(flagged[ok.split == "train"].mean()),
                   "share_test": float(flagged[ok.split == "test"].mean()),
                   "share_by_class": flagged[ok.split == "train"].groupby(ok["label"]).mean().round(3).to_dict()}
            by_g = flagged[ok.split == "train"].groupby(ok["source_group"].map(gid)).mean().round(3)
            art["share_by_group"] = by_g.to_dict()
            art["group_share_range"] = float(by_g.max() - by_g.min()) if len(by_g) else np.nan
            b["periodic_artifact"] = art
        S["B"] = b

    sec_b()

    # ---------------------------------------------------------------- profiles
    @section("feature profiles")
    def sec_profile():
        ok2 = ok.copy()
        ok2["profile_group"] = np.where(ok2["split"] == "test", "TEST", ok2["label"])
        prof = ok2.groupby("profile_group")[cols].describe(percentiles=[0.05, 0.5, 0.95]).T
        prof.to_csv(T / "feature_profile_by_class.csv")
        gprof = tr.assign(gid=tr["source_group"].map(gid)).groupby("gid")[cols].median()
        gprof.to_csv(T / "feature_median_by_group.csv")

    sec_profile()

    # ---------------------------------------------------------------- C. separability & confounding
    @section("C. class separability & source confounding")
    def sec_c():
        rows = []
        n, k = len(tr), tr["source_group"].nunique()
        for c in cols:
            h, p, eta = kruskal_eta(tr, c)
            r2c = r2_by(tr[c], tr["label"])
            r2g_within = r2_within(tr[c], tr["label"], tr["source_group"])
            rows.append({"feature": c, "kruskal_H": h, "kruskal_p": p, "eta2_class": eta,
                         "r2_class": r2c,
                         "r2_group_within_class": r2g_within,
                         "adj_r2_group_within_class": adj_r2(r2g_within, n, k)})
        t = pd.DataFrame(rows).sort_values("eta2_class", ascending=False)
        t.to_csv(T / "feature_class_effects.csv", index=False)
        S["C"] = {
            "top_class_features": t.head(10)[["feature", "eta2_class"]].to_dict("records"),
            "top_source_confounded_features": t.sort_values("adj_r2_group_within_class", ascending=False)
            .head(10)[["feature", "adj_r2_group_within_class"]].to_dict("records"),
            "median_adj_r2_group_within_class": float(t["adj_r2_group_within_class"].median()),
            "n_features_with_group_effect_gt_0.3": int((t["adj_r2_group_within_class"] > 0.3).sum()),
        }
        # pairwise class tests on a few headline features
        pair = []
        for c in ["rms_dbfs", "snr_est_db", "spec_centroid_hz", "peak_freq_hz", "flatness_active",
                  "lf_ratio", "hf_ratio", "event_ratio", "n_events", "duration"]:
            if c not in tr:
                continue
            for i, a in enumerate(C.KNOWN_CLASSES):
                for b2 in C.KNOWN_CLASSES[i + 1:]:
                    xa, xb = tr.loc[tr.label == a, c].dropna(), tr.loc[tr.label == b2, c].dropna()
                    if len(xa) > 2 and len(xb) > 2:
                        u, p = stats.mannwhitneyu(xa, xb)
                        pair.append({"feature": c, "a": a, "b": b2, "auc_a_gt_b": u / (len(xa) * len(xb)), "p": p})
        pd.DataFrame(pair).to_csv(T / "pairwise_class_tests.csv", index=False)
        # feature correlation (train)
        tr[cols].corr(method="spearman").to_csv(T / "feature_correlation_spearman.csv")

    sec_c()

    # ---------------------------------------------------------------- D. CV probes
    @section("D. baseline probes: random vs group-aware CV")
    def sec_d():
        d = {}
        n_groups_per_class = tr.groupby("label")["source_group"].nunique()
        n_splits = int(min(5, n_groups_per_class.min())) if len(n_groups_per_class) else 0
        d["group_cv_n_splits"] = n_splits
        rand_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
        models = {"logreg": make_lr, "random_forest": make_rf}
        res = {}
        for mname, fn in models.items():
            p = oof_predict(fn, Xtr, ytr, gtr, rand_cv)
            res[f"{mname}_random_cv"] = {"macro_f1": macro_f1(ytr, p), **{f"f1_{k}": v for k, v in per_class_f1(ytr, p).items()}}
            if n_splits >= 2:
                gcv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED)
                p = oof_predict(fn, Xtr, ytr, gtr, gcv)
                res[f"{mname}_group_cv"] = {"macro_f1": macro_f1(ytr, p), **{f"f1_{k}": v for k, v in per_class_f1(ytr, p).items()}}
                res[f"{mname}_leakage_gap"] = res[f"{mname}_random_cv"]["macro_f1"] - res[f"{mname}_group_cv"]["macro_f1"]
        d["results"] = res
        # leave-one-group-out accuracy (which sources are hard / unique)
        rows = []
        if len(np.unique(gtr)) >= 2:
            pred = oof_predict(make_lr, Xtr, ytr, gtr, LeaveOneGroupOut())
            for g in np.unique(gtr):
                m = gtr == g
                rows.append({"gid": gid[g], "label": pd.Series(ytr[m]).mode().iloc[0], "n": int(m.sum()),
                             "accuracy_when_held_out": float((pred[m] == ytr[m]).mean()),
                             "class_present_in_other_groups": int(len(set(gtr[ytr == pd.Series(ytr[m]).mode().iloc[0]]) - {g}))})
            pd.DataFrame(rows).sort_values("gid").to_csv(T / "leave_one_group_out.csv", index=False)
            d["logo_macro_f1_pooled"] = macro_f1(ytr, pred)
        S["D"] = d

    sec_d()

    # ---------------------------------------------------------------- E. shift & novelty
    @section("E. train vs test shift")
    def sec_e():
        e = {}
        Xs = np.vstack([Xtr, Xte])
        ys = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                     random_state=C.SEED, n_jobs=2)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
        proba = cross_val_predict(clf, Xs, ys, cv=skf, method="predict_proba")[:, 1]
        e["adversarial_auc"] = float(roc_auc_score(ys, proba))
        clf.fit(Xs, ys)
        imp = pd.DataFrame({"feature": names_all, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
        imp.to_csv(T / "adversarial_importance.csv", index=False)
        e["adversarial_top_features"] = imp.head(8).to_dict("records")

        rows = []
        for c in cols:
            a, b = tr[c].dropna(), te[c].dropna()
            if len(a) > 2 and len(b) > 2:
                ks = stats.ks_2samp(a, b)
                rows.append({"feature": c, "ks": ks.statistic, "p": ks.pvalue,
                             "train_median": a.median(), "test_median": b.median()})
        ksd = pd.DataFrame(rows).sort_values("p")
        m = len(ksd)
        ksd["p_bh"] = np.minimum.accumulate((ksd["p"].values * m / np.arange(1, m + 1))[::-1])[::-1].clip(max=1)
        ksd.sort_values("ks", ascending=False).to_csv(T / "shift_ks.csv", index=False)
        e["n_features_shifted_bh_0.01"] = int((ksd["p_bh"] < 0.01).sum())
        e["n_features_tested"] = int(m)
        e["top_ks_features"] = ksd.sort_values("ks", ascending=False).head(8)[["feature", "ks", "train_median", "test_median"]].to_dict("records")

        # nearest-neighbour novelty in a PCA space fitted on train only
        sc = StandardScaler().fit(Xtr)
        pca = PCA(n_components=min(20, Xtr.shape[1], len(Xtr) - 1), random_state=C.SEED).fit(sc.transform(Xtr))
        Ztr, Zte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
        e["pca_explained_variance_20"] = float(pca.explained_variance_ratio_.sum())
        d_lgo = np.full(len(Ztr), np.nan)
        for g in np.unique(gtr):
            msk = gtr == g
            if (~msk).sum() >= 3:
                d_lgo[msk] = knn_mean_dist(Ztr[~msk], Ztr[msk])
        d_te = knn_mean_dist(Ztr, Zte)
        ref = d_lgo[~np.isnan(d_lgo)]
        if len(ref):
            q = np.percentile(ref, [50, 90, 95, 99])
            e["train_leave_group_out_nn_dist_quantiles"] = dict(zip(["p50", "p90", "p95", "p99"], q.round(3)))
            e["test_nn_dist_quantiles"] = dict(zip(["p50", "p90", "p95", "p99"], np.percentile(d_te, [50, 90, 95, 99]).round(3)))
            e["test_frac_above_train_lgo_p95"] = float((d_te > q[2]).mean())
            e["test_frac_above_train_lgo_p99"] = float((d_te > q[3]).mean())
            mad = 1.4826 * np.median(np.abs(ref - np.median(ref))) + 1e-9
            nov = pd.DataFrame({"file_id": te["file_id"], "knn_dist": d_te,
                                "novelty_z": (d_te - np.median(ref)) / mad,
                                "pct_vs_train_lgo": [(ref < v).mean() for v in d_te]})
            nov.to_csv(T / "test_novelty_scores.csv", index=False)
        S["E"] = e
        np.save(T / "nn_dist_train_lgo.npy", d_lgo)
        np.save(T / "nn_dist_test.npy", d_te)

    sec_e()

    # ---------------------------------------------------------------- F. open-set feasibility
    @section("F. leave-one-class-out novelty detection")
    def sec_f():
        rows = []
        for held in C.KNOWN_CLASSES:
            is_held = ytr == held
            known_idx, held_idx = np.where(~is_held)[0], np.where(is_held)[0]
            kg = gtr[known_idx]
            ng = len(np.unique(kg))
            if ng < 2 or len(held_idx) < 3:
                continue
            aucs = {"knn": [], "isolation_forest": [], "max_softmax_prob": []}
            for fit_i, neg_i in GroupKFold(n_splits=min(3, ng)).split(known_idx, groups=kg):
                fit, neg, pos = Xtr[known_idx[fit_i]], Xtr[known_idx[neg_i]], Xtr[held_idx]
                yfit = ytr[known_idx[fit_i]]
                sc = StandardScaler().fit(fit)
                pca = PCA(n_components=min(20, fit.shape[1], len(fit) - 1), random_state=C.SEED).fit(sc.transform(fit))
                tf = lambda A: pca.transform(sc.transform(A))
                Ff, Fn, Fp = tf(fit), tf(neg), tf(pos)
                yv = np.r_[np.zeros(len(Fn)), np.ones(len(Fp))]
                aucs["knn"].append(roc_auc_score(yv, np.r_[knn_mean_dist(Ff, Fn), knn_mean_dist(Ff, Fp)]))
                iso = IsolationForest(n_estimators=200, random_state=C.SEED).fit(Ff)
                aucs["isolation_forest"].append(roc_auc_score(yv, -np.r_[iso.score_samples(Fn), iso.score_samples(Fp)]))
                if len(np.unique(yfit)) >= 2:
                    lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Ff, yfit)
                    msp = lambda F: 1 - lr.predict_proba(F).max(1)
                    aucs["max_softmax_prob"].append(roc_auc_score(yv, np.r_[msp(Fn), msp(Fp)]))
            for meth, v in aucs.items():
                if v:
                    rows.append({"held_out_class": held, "method": meth, "auroc": float(np.mean(v)), "folds": len(v)})
        t = pd.DataFrame(rows)
        t.to_csv(T / "loco_novelty_auroc.csv", index=False)
        if len(t):
            S["F"] = {"mean_auroc_by_method": t.groupby("method")["auroc"].mean().round(3).to_dict(),
                      "table": t.to_dict("records")}

    sec_f()

    with open(T / "summary.json", "w") as f:
        json.dump(S, f, indent=2, default=jsonable)
    print(f"[02] wrote {T / 'summary.json'}  (section errors: {len(S['section_errors'])})")


if __name__ == "__main__":
    main()
