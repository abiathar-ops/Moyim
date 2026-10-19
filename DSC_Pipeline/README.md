# DSC_Pipeline — Schematics NPC DSC 2026 (bird-sound classification)

Classify each recording as **Bondol Jawa**, **Cendrawasih**, **Kutilang**, or
**Unknown** (open set). Metric: **Macro-F1 over the 4 classes**.

The pipeline runs in three numbered stages that share one source package and
one outputs tree. Everything is from the competition data only (no external
data, no pretrained weights) and is reproducible from a single `SEED`.

```
DSC_Pipeline/
├── src/                      # shared library (imported everywhere)
│   ├── config.py             # every path & hyper-parameter, one place
│   ├── features.py           # front-ends (log-mel / PCEN / norm) + activity curve
│   ├── augment.py            # crops, cross-group noise bank, SpecAugment, EQ/gain
│   ├── dataset.py            # cached-audio loading, BirdWindowDataset, eval windows
│   ├── model.py              # from-scratch EfficientNet-style CNN
│   └── _common.py            # puts the pipeline root on sys.path from any depth
│
├── stages/
│   ├── 01_eda/               # evidence before any modelling choice
│   │   ├── 00_eda_features.py       # raw-48kHz clip stats  (feeds notebooks/01_EDA)
│   │   ├── 01_extract_features.py…04_report.py   # standalone EDA sub-pipeline
│   │   ├── run_eda.py               # runs 01→04, writes outputs/eda/EDA_REPORT.md
│   │   ├── config.py, audio_features.py          # self-contained EDA config
│   │   └── README.md
│   ├── 02_preprocessing/
│   │   ├── 01_inventory.py          # headers, groups, stereo, duplicates
│   │   ├── 02_cache_audio.py        # decode → 32 kHz mono .npy cache  (RUN FIRST)
│   │   ├── 03_make_folds.py         # StratifiedGroupKFold, seed-searched → folds.csv
│   │   ├── 04_activity_and_noise.py # activity curves + cross-group noise bank
│   │   └── 05_compare_frontends.py  # honest group-CV front-end comparison
│   └── 03_modeling/
│       ├── 06_train.py              # group-fold CNN ensemble → OOF + embeddings + models
│       ├── 07_tune_unknown.py       # Unknown threshold via leave-one-class-out
│       └── 08_infer.py              # sliding-window ensemble → submission.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb                 # narrated EDA (runs stage 01)
│   ├── 02_preprocessing.ipynb       # narrated preprocessing (runs stage 02)
│   └── DSC2026_bird_submission.ipynb# self-contained Kaggle GPU notebook (full run)
│
├── outputs/
│   ├── eda/                  # EDA tables, figures, EDA_REPORT.md
│   ├── preprocessing/        # inventory, activity, frontend comparison, OOF probes
│   ├── folds.csv             # the fold assignment (shared contract)
│   ├── noise_bank.csv        # noise-segment index (shared contract)
│   ├── models/               # model_fold*.pt, oof_cnn.csv, train_log.json  (git-ignored weights)
│   └── submission/           # submission.csv, test_probs.csv
│
├── tools/                    # notebook builders
├── cache/                    # decoded audio + activity + noise (git-ignored, regenerable)
├── requirements.txt          # base (EDA + preprocessing)
└── requirements-model.txt    # adds torch for stage 03
```

## Run order

```bash
# from the DSC_Pipeline/ root. Scripts find src/ automatically (src/_common.py),
# so they also run from inside their stage folder.

# Stage 1 — EDA (optional, evidence for the choices below)
python stages/01_eda/run_eda.py                 # → outputs/eda/EDA_REPORT.md

# Stage 2 — preprocessing (REQUIRED before training; builds the cache)
python stages/02_preprocessing/01_inventory.py
python stages/02_preprocessing/02_cache_audio.py        # decode → cache/ (do first)
python stages/02_preprocessing/03_make_folds.py         # → outputs/folds.csv
python stages/02_preprocessing/04_activity_and_noise.py # → outputs/noise_bank.csv
python stages/02_preprocessing/05_compare_frontends.py  # → outputs/preprocessing/

# Stage 3 — modelling  (needs torch: pip install -r requirements-model.txt)
python stages/03_modeling/06_train.py --frontend pcen_norm --size s --epochs 22
python stages/03_modeling/07_tune_unknown.py --loco
python stages/03_modeling/08_infer.py                   # → outputs/submission/submission.csv
```

Or open the notebooks in `notebooks/` from anywhere in the tree (they locate the
root themselves). **For the actual Kaggle submission, use
`notebooks/DSC2026_bird_submission.ipynb`** — it is fully self-contained (inlines
every stage), so on Kaggle you only attach the competition dataset, set
`FAST=False`, and Run All on a GPU kernel. See `MODEL_README.md` for the modelling
design, choices, and rules-compliance notes.

## Notes
- **Cache is regenerable and git-ignored.** It was lost when the dataset folder
  was moved; rebuild with `02_cache_audio.py` + `04_activity_and_noise.py` before
  training locally. The Kaggle notebook builds its own cache.
- **Two shared contracts** live at `outputs/` root: `folds.csv` and
  `noise_bank.csv`. Everything downstream reads them from there (`C.OUT_DIR`).
- **Reproducible & explainable:** one `SEED`; the fold split is saved; no AutoML;
  no external data or pretrained weights.
```
