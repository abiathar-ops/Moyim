"""Train the from-scratch CNN with group cross-validation.

For each of the N_FOLDS group folds:
  * train on the other folds, with activity-weighted crops + cross-group
    noise mixing + SpecAugment (all defined in src/augment.py);
  * the noise bank is restricted to TRAINING groups of that fold, so a
    validation clip is never mixed with noise from its own held-out source;
  * validate on the held-out groups using the single most active window;
  * save out-of-fold logits AND penultimate embeddings for every train clip
    (used later to tune the Unknown threshold and for distance novelty).

Outputs (outputs/):
  model_fold{k}.pt         fold weights
  oof_cnn.csv              file_id, label, fold, logit_*, prob_*, pred, max_prob
  oof_embed.npy            (n_train, embed_dim) penultimate features
  oof_embed_ids.json       row order for oof_embed.npy
  train_log.json           per-fold/epoch metrics + config echo
Macro-F1 (3-class) over the OOF predictions is printed at the end.

Reproducible: a single --seed controls numpy, torch, CUDA and the sampler.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _common  # noqa: F401  (puts pipeline root on sys.path)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, recall_score

from src import config as C
from src import augment as A
from src.dataset import BirdWindowDataset, load_train_table, class_balanced_weights
from src.model import build_model


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(table, frontend, noise_bank, aug_cfg, split, train, batch, workers, seed):
    ds = BirdWindowDataset(table, frontend=frontend, train=train, noise_bank=noise_bank,
                           aug_cfg=aug_cfg, split=split, seed=seed)
    if train:
        w = class_balanced_weights(table["y"].values)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(table), replacement=True)
        return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers,
                          pin_memory=torch.cuda.is_available(), drop_last=True)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    logits_all, emb_all, y_all = [], [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        logits, emb = model(X, return_embed=True)
        logits_all.append(logits.float().cpu().numpy())
        emb_all.append(emb.float().cpu().numpy())
        y_all.append(y.numpy())
    return (np.concatenate(logits_all), np.concatenate(emb_all), np.concatenate(y_all))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train_fold(fold, meta, args, device):
    tr = meta[meta.fold != fold].reset_index(drop=True)
    va = meta[meta.fold == fold].reset_index(drop=True)
    # noise bank restricted to TRAIN groups only -> no leakage into val
    noise_bank = A.NoiseBank.load(allowed_groups=set(tr.source_group.unique()))
    aug_cfg = A.AugConfig(p_noise=args.p_noise, p_specaug=args.p_specaug,
                          p_gain=args.p_gain, p_eq=args.p_eq,
                          p_crop_random=args.p_crop_random)

    tl = make_loader(tr, args.frontend, noise_bank, aug_cfg, "train", True,
                     args.batch, args.workers, args.seed + fold)
    vl = make_loader(va, args.frontend, None, aug_cfg, "train", False,
                     args.batch, args.workers, args.seed + fold)

    model = build_model(n_classes=len(C.CLASSES), size=args.size, drop=args.dropout).to(device)
    # class-weighted loss on top of the balanced sampler: extra emphasis on the
    # rare Cendrawasih class, with label smoothing to curb overconfidence (which
    # also makes the max-softmax Unknown signal better calibrated).
    counts = np.bincount(tr.y.values, minlength=len(C.CLASSES)).astype(float)
    cls_w = torch.as_tensor((counts.sum() / (len(counts) * counts)) ** 0.5,
                            dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=args.label_smooth)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, len(tl)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    hist, best_f1, best_state = [], -1.0, None
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for X, y in tl:
            X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = crit(model(X), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            run += float(loss) * len(X)
        logits, _, yv = evaluate(model, vl, device, len(C.CLASSES))
        pred = logits.argmax(1)
        f1 = f1_score(yv, pred, average="macro")
        rec = recall_score(yv, pred, average=None, labels=list(range(len(C.CLASSES))),
                           zero_division=0)
        hist.append(dict(epoch=ep, train_loss=run / max(1, len(tr)), val_macro_f1=float(f1),
                         val_recall=[round(float(r), 3) for r in rec], sec=round(time.time() - t0, 1)))
        print(f"  fold{fold} ep{ep} loss={hist[-1]['train_loss']:.3f} "
              f"valF1={f1:.4f} rec={hist[-1]['val_recall']} ({hist[-1]['sec']}s)", flush=True)
        if f1 >= best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    # store only JSON-safe scalars in the checkpoint meta so torch.load works
    # under weights_only as well; weights live in state_dict (plain tensors).
    cfg = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
           for k, v in vars(args).items()}
    torch.save({"state_dict": best_state, "config": cfg, "fold": int(fold),
                "classes": list(C.CLASSES)}, C.MODELS_DIR / f"model_fold{fold}.pt")
    logits, emb, yv = evaluate(model, vl, device, len(C.CLASSES))
    return va, logits, emb, hist, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", default="pcen_norm")
    ap.add_argument("--size", default="s", choices=["xs", "s", "m"])
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--p_noise", type=float, default=0.8)
    ap.add_argument("--p_specaug", type=float, default=0.5)
    ap.add_argument("--p_gain", type=float, default=0.5)
    ap.add_argument("--p_eq", type=float, default=0.5)
    ap.add_argument("--p_crop_random", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    C.ensure_output_dirs()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} frontend={args.frontend} size={args.size} "
          f"epochs={args.epochs} seed={args.seed}", flush=True)

    meta = load_train_table(with_folds=True)
    folds = sorted(meta.fold.unique()) if args.folds == "all" else [int(x) for x in args.folds.split(",")]

    oof_rows, emb_rows, emb_ids, logs = [], [], [], {}
    for k in folds:
        va, logits, emb, hist, best = train_fold(k, meta, args, device)
        probs = softmax(logits)
        for i, (_, row) in enumerate(va.iterrows()):
            oof_rows.append(dict(file_id=row.file_id, label=row.label, fold=int(k), y=int(row.y),
                                 **{f"logit_{j}": float(logits[i, j]) for j in range(len(C.CLASSES))},
                                 **{f"prob_{j}": float(probs[i, j]) for j in range(len(C.CLASSES))},
                                 pred=int(probs[i].argmax()), max_prob=float(probs[i].max())))
            emb_rows.append(emb[i]); emb_ids.append(row.file_id)
        logs[f"fold{k}"] = dict(best_val_macro_f1=float(best), history=hist)

    oof = pd.DataFrame(oof_rows)
    oof.to_csv(C.MODELS_DIR / "oof_cnn.csv", index=False)
    np.save(C.MODELS_DIR / "oof_embed.npy", np.asarray(emb_rows, dtype=np.float32))
    json.dump(emb_ids, open(C.MODELS_DIR / "oof_embed_ids.json", "w"))

    f1 = f1_score(oof.y, oof.pred, average="macro")
    rec = recall_score(oof.y, oof.pred, average=None, labels=list(range(len(C.CLASSES))),
                       zero_division=0)
    logs["oof_macro_f1_3class"] = float(f1)
    logs["oof_recall"] = {C.CLASSES[i]: round(float(r), 4) for i, r in enumerate(rec)}
    logs["config"] = vars(args)
    json.dump(logs, open(C.MODELS_DIR / "train_log.json", "w"), indent=2)

    print(f"\nOOF 3-class macro-F1 = {f1:.4f}")
    for i, c in enumerate(C.CLASSES):
        print(f"  recall[{c}] = {rec[i]:.4f}")


if __name__ == "__main__":
    main()
