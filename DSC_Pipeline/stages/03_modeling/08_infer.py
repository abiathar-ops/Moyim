"""Predict the test set and write a Kaggle submission.

For each test clip:
  * slide a 5 s window with 50% overlap (src.dataset.eval_windows);
  * run every fold model on every window;
  * average the softmax probabilities over windows AND folds (a simple,
    robust ensemble -- no per-fold weighting, nothing to overfit);
  * if the top averaged probability is below tau, label the clip "Unknown",
    otherwise take the arg-max class.

Writes outputs/submission.csv with columns [file_id, label] matching
sample_submission.csv exactly (same rows, same order).
Also writes outputs/test_probs.csv (file_id, prob_*, max_prob, pred) so the
decisions are auditable and tau can be changed without re-running the net.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _common  # noqa: F401  (puts pipeline root on sys.path)

import torch

from src import config as C
from src import features as F
from src.dataset import eval_windows, load_audio, write_submission, submission_name
from src.model import build_model


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def load_fold_models(size, device):
    models = []
    for p in sorted(C.MODELS_DIR.glob("model_fold*.pt")):
        # our own trusted checkpoint; weights_only=False is needed because the
        # meta dict carries non-tensor fields (config/classes).
        ckpt = torch.load(p, map_location=device, weights_only=False)
        m = build_model(n_classes=len(C.CLASSES), size=ckpt["config"].get("size", size)).to(device)
        m.load_state_dict(ckpt["state_dict"]); m.eval()
        models.append(m)
    if not models:
        raise FileNotFoundError("no model_fold*.pt in outputs/models -- train first")
    return models


@torch.no_grad()
def predict_clip(y, fe, models, device, max_windows=12):
    wins = eval_windows(y)
    if len(wins) > max_windows:                       # cap cost on long clips
        idx = np.linspace(0, len(wins) - 1, max_windows).round().astype(int)
        wins = [wins[i] for i in idx]
    X = np.stack([fe(w)[None] for w in wins]).astype(np.float32)   # (W,1,mel,frames)
    Xt = torch.as_tensor(X, device=device)
    probs = np.zeros((len(wins), len(C.CLASSES)), dtype=np.float64)
    for m in models:
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            probs += softmax(m(Xt).float().cpu().numpy())
    probs /= len(models)
    return probs.mean(0)                              # average over windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", default="pcen_norm")
    ap.add_argument("--size", default="s")
    ap.add_argument("--tau", type=float, default=None, help="override Unknown threshold")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    C.ensure_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = F.frontend(args.frontend)
    models = load_fold_models(args.size, device)
    print(f"device={device} models={len(models)} frontend={args.frontend}", flush=True)

    tau = args.tau
    if tau is None:
        tp = C.MODELS_DIR / "unknown_threshold.json"
        tau = json.load(open(tp))["tau"] if tp.exists() else 0.55
    print(f"Unknown threshold tau = {tau}", flush=True)

    sub = pd.read_csv(C.SAMPLE_SUB)
    rows = []
    for n, fid in enumerate(sub.file_id):
        y = load_audio(args.split, fid)
        p = predict_clip(y, fe, models, device)
        k = int(p.argmax()); mp = float(p.max())
        label = C.UNKNOWN if mp < tau else C.CLASSES[k]
        rows.append(dict(file_id=fid, label=label, max_prob=mp,
                         **{f"prob_{j}": float(p[j]) for j in range(len(C.CLASSES))}))
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(sub)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(C.SUBMISSION_DIR / "test_probs.csv", index=False)
    # restore sample-submission row order, then write in the organiser template
    out = sub[["file_id"]].merge(df[["file_id", "label"]], on="file_id", how="left")
    path = write_submission(out, sub)          # {AUTHOR}_{NN}_Submission.csv + identical submission.csv

    print("\nlabel distribution on test:")
    print(out.label.value_counts().to_string())
    print(f"\nwrote {path} (+ identical submission.csv) ({len(out)} rows)")


if __name__ == "__main__":
    main()
