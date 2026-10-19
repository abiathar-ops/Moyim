"""Step 5 - Compare preprocessing choices on honest (group-held-out) folds.

This is a cheap, transparent probe - not the final model. For each variant
we summarise every clip as the per-band mean and std of its spectrogram
(256 numbers), train a class-balanced logistic regression on 3 folds and
predict the held-out 4th. Metrics are on pooled out-of-fold predictions.

Three diagnostics per variant:
  group_cv_macro_f1   the honest score (whole recording groups held out)
  random_cv_macro_f1  the same model on a random split - the gap shows how much
                      a careless split would flatter us
  group_id_probe      how well the features identify the RECORDING GROUP within
                      a species (balanced accuracy). Lower = less recording /
                      microphone information in the features, which is what we want.

Augmentation ablation (on the best-expected front-end, pcen_norm):
  crop        training copies are activity-weighted 5 s crops only
  crop+noise  + cross-group background mixing
  full        + random EQ and gain
Augmented copies come from activity-weighted crops with cross-group noise
mixing, EQ and gain (noise bank restricted to training-fold groups, so no
validation recording ever leaks into training). Validation stays clean.

Outputs  outputs/frontend_comparison.csv, outputs/oof_<variant>.csv
"""
import _common  # noqa: F401
import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src import features as F
from src import augment as A
from src.dataset import eval_windows, load_activity, load_audio, load_train_table


def pool(S):
    return np.concatenate([S.mean(axis=1), S.std(axis=1)])


def clean_features(file_id, fe_name):
    fe = F.frontend(fe_name)
    y = load_audio("train", file_id)
    return np.mean([pool(fe(w)) for w in eval_windows(y)], axis=0)


AUG_MODES = {
    # name: (use activity crop, augmentation config or None)
    "crop":       A.AugConfig(p_noise=0, p_eq=0, p_gain=0),
    "crop+noise": A.AugConfig(p_eq=0, p_gain=0),
    "full":       A.AugConfig(),
}
_BANKS = {}


def aug_features(file_id, group, fe_name, allowed_groups, n_copies, seed, mode):
    rng = np.random.default_rng(seed)
    fe = F.frontend(fe_name)
    key = (tuple(sorted(allowed_groups)))
    if key not in _BANKS:                      # one noise bank per fold per worker
        _BANKS.clear(); _BANKS[key] = A.NoiseBank.load(allowed_groups=allowed_groups)
    bank = _BANKS[key]
    cfg = AUG_MODES[mode]
    y, curve = load_audio("train", file_id), load_activity(file_id)
    out = []
    for _ in range(n_copies):
        w = A.choose_crop(y, curve, C.WIN, rng, p_random=cfg.p_crop_random)
        w = A.augment_wave(w, group, rng, bank, cfg)
        out.append(pool(fe(w)))
    return np.stack(out)


def model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))


def scores(y, p):
    rec = recall_score(y, p, average=None, labels=range(len(C.CLASSES)))
    d = dict(accuracy=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro"))
    d.update({f"recall_{c.split()[0]}": r for c, r in zip(C.CLASSES, rec)})
    return d


def group_probe(X, t):
    """Within each species: can the features tell recording groups apart?"""
    accs, chance = [], []
    for lab in C.CLASSES:
        m = (t.label == lab).values
        g = t.source_group.values[m]
        keep = pd.Series(g).map(pd.Series(g).value_counts()).values >= 5
        Xm, gm = X[m][keep], g[keep]
        if len(np.unique(gm)) < 2:
            continue
        pred = np.empty(len(gm), dtype=object)
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=C.SEED).split(Xm, gm):
            pred[va] = model().fit(Xm[tr], gm[tr]).predict(Xm[va])
        accs.append(balanced_accuracy_score(gm, pred)); chance.append(1 / len(np.unique(gm)))
    return float(np.mean(accs)), float(np.mean(chance))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontends", default="logmel,logmel_norm,pcen,pcen_norm")
    ap.add_argument("--aug-frontend", default="pcen_norm",
                    help="frontend used for the augmentation ablation ('' to skip)")
    ap.add_argument("--aug-modes", default="crop,crop+noise,full")
    ap.add_argument("--aug-copies", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    t = load_train_table()
    y = t.y.values
    results = []
    clean = {}
    for fe_name in [f for f in args.frontends.split(",") if f]:
        t0 = time.time()
        X = np.stack(Parallel(n_jobs=args.jobs, batch_size=16)(
            delayed(clean_features)(f, fe_name) for f in t.file_id))
        clean[fe_name] = X
        oof = np.empty(len(t), int)
        for k in range(C.N_FOLDS):
            tr, va = (t.fold != k).values, (t.fold == k).values
            oof[va] = model().fit(X[tr], y[tr]).predict(X[va])
        rnd = np.empty(len(t), int)
        for tr, va in StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=C.SEED).split(X, y):
            rnd[va] = model().fit(X[tr], y[tr]).predict(X[va])
        probe, chance = group_probe(X, t)
        r = dict(variant=fe_name, augmented=False, **scores(y, oof),
                 random_cv_macro_f1=f1_score(y, rnd, average="macro"),
                 group_id_probe=probe, group_id_chance=chance)
        results.append(r)
        pd.DataFrame(dict(file_id=t.file_id, label=t.label, fold=t.fold,
                          pred=[C.CLASSES[i] for i in oof])).to_csv(C.OUT_DIR / f"oof_{fe_name}.csv", index=False)
        print(f"{fe_name:12s} group-CV macroF1={r['macro_f1']:.3f} acc={r['accuracy']:.3f} | "
              f"random-CV macroF1={r['random_cv_macro_f1']:.3f} | group-ID probe={probe:.2f} "
              f"(chance {chance:.2f})  [{time.time() - t0:.0f}s]", flush=True)

    fe_name = args.aug_frontend
    for mode in ([m for m in args.aug_modes.split(",") if m] if fe_name else []):
        t0 = time.time()
        Xc = clean.get(fe_name)
        if Xc is None:
            Xc = np.stack(Parallel(n_jobs=args.jobs)(delayed(clean_features)(f, fe_name) for f in t.file_id))
        oof = np.empty(len(t), int)
        for k in range(C.N_FOLDS):
            trm = (t.fold != k).values
            tt = t[trm]
            allowed = tt.source_group.unique()
            Xa = Parallel(n_jobs=args.jobs, batch_size=8)(
                delayed(aug_features)(r.file_id, r.source_group, fe_name, allowed, args.aug_copies,
                                      C.SEED * 1000 + k * 10_000 + i, mode)
                for i, r in enumerate(tt.itertuples()))
            Xtr = np.concatenate([Xc[trm]] + [np.stack([a[j] for a in Xa]) for j in range(args.aug_copies)])
            ytr = np.concatenate([y[trm]] * (1 + args.aug_copies))
            va = (t.fold == k).values
            oof[va] = model().fit(Xtr, ytr).predict(Xc[va])
        name = f"{fe_name}+{mode}"
        r = dict(variant=name, augmented=True, **scores(y, oof))
        results.append(r)
        pd.DataFrame(dict(file_id=t.file_id, label=t.label, fold=t.fold,
                          pred=[C.CLASSES[i] for i in oof])).to_csv(C.OUT_DIR / f"oof_{name}.csv", index=False)
        print(f"{name:12s} group-CV macroF1={r['macro_f1']:.3f} acc={r['accuracy']:.3f}"
              f"  [{time.time() - t0:.0f}s]", flush=True)

    res = pd.DataFrame(results)
    path = C.OUT_DIR / "frontend_comparison.csv"
    if path.exists():                       # keep rows from earlier partial runs
        old = pd.read_csv(path)
        res = pd.concat([old[~old.variant.isin(res.variant)], res], ignore_index=True)
    order = {v: i for i, v in enumerate(["logmel", "logmel_norm", "pcen", "pcen_norm"])}
    res = res.sort_values(["augmented", "variant"], key=lambda c: c.map(order).fillna(9) if c.name == "variant" else c)
    res.to_csv(path, index=False)
    print("\n" + res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
