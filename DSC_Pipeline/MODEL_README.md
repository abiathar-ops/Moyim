# Prediction model — Schematics NPC DSC 2026 (bird-sound)

A CNN **trained from scratch** on the competition audio only, with an open-set
**Unknown** class decided by a tunable confidence threshold. Built on top of the
existing preprocessing pipeline (`src/`, `scripts/01–05`, `outputs/folds.csv`,
`outputs/noise_bank.csv`).

**Target metric:** Macro-F1 over 4 classes (Bondol Jawa, Cendrawasih, Kutilang,
Unknown).

---

## Two ways to run

### A. The Kaggle notebook (what you submit) — `DSC2026_bird_submission.ipynb`
Fully self-contained: every step is inlined, so you only attach the competition
dataset. No external weights, no uploaded helper package, no AutoML — compliant
with "hanya data panitia" and the explainability rule. Steps: config → cache
audio → group folds (300-seed balanced search) → noise bank → train the K-fold
ensemble → tune Unknown τ by leave-one-class-out → infer → `submission.csv`.

Run: set `FAST = False`, Run All on a GPU kernel. Output: `/kaggle/working/submission.csv`.
`FAST = True` does a 1-fold smoke run in a couple of minutes.

### B. The scripts (for local experiments / the repo) — `scripts/06–08`
These reuse `src/` and the artifacts from `scripts/01–05`. They need the audio
cache built first (`scripts/02_cache_audio.py`) — note the cache is **not** in
git and was lost when the dataset folder was moved, so rebuild it before training.

```bash
# 0. (once) rebuild caches the training needs
python scripts/02_cache_audio.py          # cache/audio/{train,test}
python scripts/04_activity_and_noise.py   # cache/activity, cache/noise, noise_bank.csv

# 1. train the group-fold ensemble (from scratch)
python scripts/06_train.py --frontend pcen_norm --size s --epochs 22

# 2. choose the Unknown threshold (leave-one-class-out feasibility)
python scripts/07_tune_unknown.py --loco

# 3. predict the test set -> outputs/submission.csv
python scripts/08_infer.py
```

Outputs land in `outputs/`: `model_fold{k}.pt`, `oof_cnn.csv`, `oof_embed.npy`,
`train_log.json`, `unknown_threshold.json`, `test_probs.csv`, `submission.csv`.

---

## Design choices (and why)

| Choice | Why |
|---|---|
| **CNN from scratch** (`src/model.py`, EfficientNet-style MBConv+SE) | Rules forbid external data; random init is unambiguously compliant and still strong on 1.8k clips with heavy augmentation. |
| **PCEN + per-clip norm front-end** | Best front-end in your own `frontend_comparison.csv` (macro-F1 0.835 vs 0.803 log-mel). PCEN suppresses stationary background; per-clip norm stops recording level leaking the source. |
| **StratifiedGroupKFold on `source_group`** | Each group = one species, so whole groups must be held out or the model learns the recording, not the bird. |
| **300-seed balanced fold search** | macro-F1 swings 0.64–0.83 across group splits; only a few put all 3 species in every fold. Pick a good one deterministically. |
| **Class-balanced sampler + √-inverse-freq loss + label smoothing** | 13.5:1 imbalance (Cendrawasih = 83 clips). Smoothing also calibrates the max-softmax used for Unknown. |
| **Cross-group background mixing** | The single biggest known gain in bird-sound tasks (Lasseck 2018); breaks the background→species shortcut. Noise is drawn only from *training* groups of each fold → no leakage. |
| **Activity-weighted crops, SpecAugment, mild EQ/gain** | Standard, label-preserving. No pitch-shift / time-stretch (they move species cues). Mixup off (hurt the BirdCLEF-2024 winner). |
| **Unknown = max-softmax < τ, τ from leave-one-class-out** | No Unknown clips exist in training, so each species is held out in turn as a stand-in for an unseen one; τ and the achievable 4-class F1 are estimated **without the test set**. |
| **Window+fold ensemble by plain averaging** | Robust, nothing to overfit. |

## Expected performance
On held-out groups, 3-class macro-F1 ≈ **0.80–0.86** depending on the fold split
(Cendrawasih is the fragile class — only 4 groups, one fold may hold very few
clips). The Unknown AUROC from leave-one-class-out is the honest read on how much
the 4-class score will sit below the 3-class score; tune τ there, never on the
leaderboard.

## Reproducibility & rules
- One `SEED` seeds Python, numpy, torch, CUDA and the sampler; the fold split is
  saved to `folds.csv`.
- No pretrained weights, no external data, no full-AutoML; every step is plain,
  explainable code.
- Daily submission cap is 5 — validate changes on CV first.

## Files
```
src/model.py                 CNN (from scratch)
scripts/06_train.py          group-fold training -> OOF + embeddings + fold models
scripts/07_tune_unknown.py   Unknown threshold (OOF grid + optional --loco)
scripts/08_infer.py          sliding-window ensemble inference -> submission.csv
DSC2026_bird_submission.ipynb  self-contained Kaggle GPU notebook
requirements-model.txt       extra deps for training (torch)
```
