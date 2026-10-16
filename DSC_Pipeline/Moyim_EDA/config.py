"""Shared configuration for the audio EDA pipeline (NPC DSC 2026 bird classification).

Paths can be overridden with environment variables:
    DSC_ROOT  dataset root (folder that contains train/, test/, sample_submission.csv)
    DSC_OUT   output folder (default: <this folder>/outputs)
"""
import os
import random
from pathlib import Path

import numpy as np

SEED = 42
KNOWN_CLASSES = ["Bondol Jawa", "Cendrawasih", "Kutilang"]
ALL_LABELS = KNOWN_CLASSES + ["Unknown"]

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DSC_ROOT", HERE.parents[1]))
OUT = Path(os.environ.get("DSC_OUT", HERE / "outputs"))
TABLES = OUT / "tables"
FIGS = OUT / "figures"

# Analysis resolution used for frame-level features and the log-mel spectrogram.
TARGET_SR = 32000
N_FFT = 1024
HOP = 320          # 10 ms at 32 kHz
N_MELS = 64
FMIN = 50.0
FMAX = 15000.0


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)


def short_group_ids(train_df):
    """Map long source_group hashes to short ids (G01, G02, ...) sorted by label then size."""
    g = (train_df.groupby("source_group")
         .agg(label=("label", "first"), n=("file_id", "size"))
         .reset_index())
    g = g.sort_values(["label", "n", "source_group"], ascending=[True, False, True]).reset_index(drop=True)
    g["gid"] = [f"G{i + 1:02d}" for i in range(len(g))]
    return dict(zip(g["source_group"], g["gid"]))
