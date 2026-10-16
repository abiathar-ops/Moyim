# Audio EDA pipeline (NPC DSC 2026 bird classification)

Exploratory analysis of the competition audio. Pure numpy/scipy/scikit-learn/soundfile, seeded (SEED=42), uses only competition data.

## Run

```bash
pip install -r requirements.txt
python run_eda.py --limit 60      # smoke test, ~1 minute
python run_eda.py --force         # full run (2,640 files)
python run_eda.py --skip-extract  # re-run analysis/plots/report from cached features
```

The dataset root defaults to two folders above this one (`DSC_Moyim/`, containing `train/`, `test/`, `sample_submission.csv`). Override with `DSC_ROOT`. Outputs go to `outputs/` (override with `DSC_OUT`).

## Steps

| Script | Purpose | Main outputs |
|---|---|---|
| `01_extract_features.py` | One pass over every file: format, loudness, SNR/noise floor, silence, spectral and band energy, rain/wind (<300 Hz) and insect (>10 kHz) proxies, log-mel summaries, audio hash | `tables/features.csv`, `mel_stats.npz`, `file_inventory.json` |
| `02_analyze.py` | Metadata/group audit, integrity and duplicate checks, class vs source-group effects, random vs group-aware CV probes, train/test shift, novelty preview, leave-one-class-out open-set test | `tables/summary.json` + CSVs |
| `03_plots.py` | 13 figures | `figures/*.png` |
| `04_report.py` | Readable report with automatic flags and modelling implications | `EDA_REPORT.md` |

## Why these checks

- **`source_group` leakage:** segments from one source are near-duplicates. The pipeline measures how much random CV overstates performance versus group-aware CV.
- **Source vs species:** with few recording sources, features can describe the setup instead of the bird. The report separates class effect from group effect.
- **Unknown class:** there is no labeled Unknown data. The leave-one-class-out test shows whether novelty detection is feasible in feature space, and the train-vs-test distance analysis previews how far test audio sits from the training set.

## Rule compliance

- No external data or pretrained models.
- Test audio is only used unlabeled, to compare distributions. `test_novelty_scores.csv` is a descriptive distance score, not a label, and should not be copied into a submission.
- The EDA does not generate any submission.
