"""Central configuration for the NPC DSC 2026 bird-sound pipeline.

Every number used anywhere in the pipeline lives here, so the choices are
easy to explain (and to change) in one place.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
PIPELINE_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA = next((p for p in (PIPELINE_DIR.parent / "DSC_Dataset", PIPELINE_DIR.parent / "kaggle")
                      if p.exists()), PIPELINE_DIR.parent / "DSC_Dataset")
DATA_DIR = Path(os.environ.get("NPC_DATA", _DEFAULT_DATA))
CACHE_DIR = Path(os.environ.get("NPC_CACHE", PIPELINE_DIR / "cache"))
OUT_DIR = PIPELINE_DIR / "outputs"

TRAIN_META = DATA_DIR / "train" / "metadata.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"
TRAIN_AUDIO = DATA_DIR / "train" / "audio"
TEST_AUDIO = DATA_DIR / "test" / "audio"

AUDIO_CACHE = CACHE_DIR / "audio"          # cache/audio/{train,test}/<file_id>.npy
ACTIVITY_CACHE = CACHE_DIR / "activity"    # cache/activity/<file_id>.npy
NOISE_CACHE = CACHE_DIR / "noise"          # cache/noise/<file_id>_<k>.npy

# ---------------------------------------------------------------- labels
CLASSES = ["Bondol Jawa", "Cendrawasih", "Kutilang"]
UNKNOWN = "Unknown"
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------- audio
ORIG_SR = 48_000
SR = 32_000                 # BirdCLEF / Perch standard; 48k -> 32k is an exact 2/3 ratio
RESAMPLE_UP, RESAMPLE_DOWN = 2, 3
WIN_SEC = 5.0               # model input window (BirdCLEF standard)
WIN = int(SR * WIN_SEC)
EVAL_HOP_SEC = 2.5          # sliding-window stride at inference (50% overlap)

# ---------------------------------------------------------------- spectrogram
N_FFT = 1024
HOP = 320                   # 10 ms -> 501 frames per 5 s window
N_MELS = 128
FMIN = 40                   # BirdCLEF 2024 1st place used 40 Hz
FMAX = 15_000

# PCEN parameters tuned for outdoor bioacoustics (Lostanlen et al., 2019)
PCEN = dict(gain=0.8, bias=10.0, power=0.25, time_constant=0.06, eps=1e-6)

# ---------------------------------------------------------------- activity / noise bank
ACT_FMIN, ACT_FMAX = 500, 12_000   # band used to detect bird activity
ACT_SMOOTH_SEC = 0.25
NOISE_MIN_SEC = 1.0                # Lasseck: keep only noise stretches >= 1 s
NOISE_MAX_SEC = 3.0
NOISE_PER_CLIP = 2
NOISE_QUANTILE = 0.30              # frames below this activity quantile count as "quiet"

# ---------------------------------------------------------------- validation
N_FOLDS = 4                 # Cendrawasih has only 4 source groups
FOLD_SEED_SEARCH = 300      # seeds tried to find the most class-balanced group split
SEED = 42
