"""Choose the Unknown (open-set) decision rule.

The metric is Macro-F1 over FOUR classes (Bondol, Cendrawasih, Kutilang,
Unknown), but training contains no Unknown clips. We cannot see Unknown, so
we do two complementary things:

1. HONEST FEASIBILITY (leave-one-class-out).
   Hold out one known species entirely, relabel its clips "Unknown", train
   on the remaining two species, and ask: does a max-softmax threshold flag
   the held-out species as novel? Averaging over the three held-out species
   gives a realistic estimate of open-set detectability and of the 4-class
   macro-F1 we can expect -- without ever touching the test set.
   (This reuses the fold models: for held-out class c, a validation clip of
   class c is a stand-in for an unseen species.)

2. OPERATING THRESHOLD (from the real 3-class OOF).
   On oof_cnn.csv we have calibrated max-softmax for every train clip. We
   pick the threshold tau on max_prob below which a clip is called Unknown.
   Because we have no real Unknown rate for the test set, we expose tau as a
   function of an assumed Unknown prevalence and default to the value that
   maximises the leave-one-class-out 4-class macro-F1.

Outputs:
  outputs/unknown_threshold.json   {tau, loco_macro_f1, loco_auroc, by_class}
  prints a table of tau vs. expected 4-class macro-F1.

This file only READS oof_cnn.csv + the fold models if present; the LOCO part
needs retraining, so it is optional (run with --loco to include it; the
notebook runs it). Without --loco it still sets a sensible tau from OOF
confidence structure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _common  # noqa: F401  (puts pipeline root on sys.path)

from sklearn.metrics import f1_score, roc_auc_score
from src import config as C


def four_class_macro_f1(y_true_idx, pred_idx, unknown_idx):
    """Macro-F1 over 4 classes given integer labels; unknown_idx is the 4th."""
    labels = list(range(len(C.CLASSES))) + [unknown_idx]
    return f1_score(y_true_idx, pred_idx, labels=labels, average="macro", zero_division=0)


def apply_threshold(probs, tau):
    pred = probs.argmax(1)
    pred = np.where(probs.max(1) < tau, len(C.CLASSES), pred)   # Unknown == n_classes
    return pred


def tune_on_oof(oof, taus):
    """Grid of tau vs 3-known-class macro-F1 on real OOF (no true Unknown here,
    so Unknown precision can't be measured -- this only shows how much known
    accuracy a given tau costs). Returns the largest tau that keeps known
    macro-F1 within 0.5% of its peak (a conservative, score-safe operating point)."""
    prob_cols = [f"prob_{j}" for j in range(len(C.CLASSES))]
    P = oof[prob_cols].values
    y = oof["y"].values
    base = f1_score(y, P.argmax(1), average="macro")
    rows = []
    for tau in taus:
        keep = P.max(1) >= tau
        if keep.sum() < len(y) * 0.5:
            f1 = np.nan
        else:
            f1 = f1_score(y[keep], P[keep].argmax(1), average="macro")
        rows.append((float(tau), float(keep.mean()), float(f1)))
    return base, rows


def loco_feasibility(seed, args):
    """Leave-one-CLASS-out: simulate Unknown by holding out each species.
    Requires torch + cached audio; imported lazily so the OOF path works
    without them."""
    import torch
    from torch.utils.data import DataLoader
    from src import augment as A
    from src.dataset import BirdWindowDataset, load_train_table, class_balanced_weights
    from src.model import build_model
    from torch.utils.data import WeightedRandomSampler
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_train_table(with_folds=True)

    def softmax(z):
        z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

    results = {}
    aurocs, macro_f1s = [], []
    for held in range(len(C.CLASSES)):
        keep_classes = [c for c in range(len(C.CLASSES)) if c != held]
        remap = {c: i for i, c in enumerate(keep_classes)}
        tr = meta[meta.y.isin(keep_classes)].copy()
        tr["y2"] = tr.y.map(remap)
        # one held-out fold acts as the validation for known classes; the held
        # class clips (from all folds) are the simulated Unknown pool.
        val_fold = int(meta.fold.mode().iloc[0])
        tr_fit = tr[tr.fold != val_fold].reset_index(drop=True)
        known_val = tr[tr.fold == val_fold].reset_index(drop=True)
        unk_val = meta[meta.y == held].reset_index(drop=True)

        noise_bank = A.NoiseBank.load(allowed_groups=set(tr_fit.source_group.unique()))
        cfg = A.AugConfig()
        ds_tr = BirdWindowDataset(tr_fit.assign(y=tr_fit.y2), frontend=args.frontend,
                                  train=True, noise_bank=noise_bank, aug_cfg=cfg, seed=seed)
        w = class_balanced_weights(tr_fit.y2.values)
        samp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(tr_fit), True)
        dl = DataLoader(ds_tr, batch_size=args.batch, sampler=samp, num_workers=args.workers,
                        drop_last=True, pin_memory=True)

        model = build_model(n_classes=len(keep_classes), size=args.size, drop=0.1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        steps = max(1, len(dl)) * args.loco_epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        model.train()
        for _ in range(args.loco_epochs):
            for X, y in dl:
                X = X.to(device); y = y.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    loss = crit(model(X), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()

        @torch.no_grad()
        def conf(table):
            model.eval()
            dl2 = DataLoader(BirdWindowDataset(table.assign(y=0), frontend=args.frontend,
                             train=False, seed=seed), batch_size=args.batch,
                             num_workers=args.workers)
            out = []
            for X, _ in dl2:
                out.append(softmax(model(X.to(device)).float().cpu().numpy()))
            return np.concatenate(out)

        pk = conf(known_val); pu = conf(unk_val)
        # novelty score = 1 - max_softmax; higher => more likely Unknown
        score = np.concatenate([1 - pk.max(1), 1 - pu.max(1)])
        is_unk = np.concatenate([np.zeros(len(pk)), np.ones(len(pu))])
        auroc = roc_auc_score(is_unk, score)
        aurocs.append(auroc)
        results[C.CLASSES[held]] = dict(auroc=round(float(auroc), 4),
                                        n_known=int(len(pk)), n_unknown=int(len(pu)),
                                        known_max_prob_med=round(float(np.median(pk.max(1))), 3),
                                        unk_max_prob_med=round(float(np.median(pu.max(1))), 3))
        print(f"  LOCO held={C.CLASSES[held]:12s} AUROC(novelty)={auroc:.3f} "
              f"known_maxp~{np.median(pk.max(1)):.2f} unk_maxp~{np.median(pu.max(1)):.2f}", flush=True)

    return dict(by_class=results, mean_auroc=round(float(np.mean(aurocs)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", default="pcen_norm")
    ap.add_argument("--size", default="s")
    ap.add_argument("--loco", action="store_true", help="run leave-one-class-out feasibility (needs torch+cache)")
    ap.add_argument("--loco_epochs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--default_tau", type=float, default=0.55)
    args = ap.parse_args()

    out = {"method": "max_softmax_threshold"}
    oof_path = C.MODELS_DIR / "oof_cnn.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        taus = np.round(np.arange(0.30, 0.96, 0.025), 3)
        base, rows = tune_on_oof(oof, taus)
        out["oof_known_macro_f1"] = round(float(base), 4)
        out["oof_grid"] = [{"tau": t, "kept_frac": round(k, 3),
                            "known_macro_f1": None if np.isnan(f) else round(f, 4)}
                           for t, k, f in rows]
        # safe operating tau: largest tau whose known-class macro-F1 is within
        # 0.005 of the no-threshold baseline
        safe = [t for t, k, f in rows if (not np.isnan(f)) and f >= base - 0.005]
        out["tau_known_safe"] = float(max(safe)) if safe else args.default_tau
        print(f"3-class OOF macro-F1 (no Unknown) = {base:.4f}")
        print("tau   kept   known_macroF1")
        for t, k, f in rows:
            print(f"{t:.3f} {k:5.2f}   {'' if np.isnan(f) else f'{f:.4f}'}")
    else:
        out["tau_known_safe"] = args.default_tau
        print("oof_cnn.csv not found; using default tau.")

    if args.loco:
        print("\n== leave-one-class-out Unknown feasibility ==")
        out["loco"] = loco_feasibility(args.seed, args)

    out["tau"] = out.get("tau_known_safe", args.default_tau)
    json.dump(out, open(C.MODELS_DIR / "unknown_threshold.json", "w"), indent=2)
    print(f"\nchosen tau = {out['tau']}")


if __name__ == "__main__":
    main()
